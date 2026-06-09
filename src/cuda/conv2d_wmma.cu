// Tensor-core implicit-GEMM conv2d forward (FP16 storage, FP32 accumulator).
//
// Convolution Y[n, oc, oh, ow] = sum_{ic, kh, kw} X[n, ic, in_h, in_w] *
//                                W[oc, ic, kh, kw] (+ bias[oc]),
// reorganised as a GEMM:
//   C(M, N_gemm) = A(M, K) @ B(N_gemm, K)^T
// where
//   M       = N * H_out * W_out   (rows; each row is one output spatial pixel)
//   N_gemm  = C_out               (cols; each col is one output channel)
//   K       = C_in * kH * kW      (reduction; im2col virtual dim)
//   A[m, k] = X[n, ic, in_h, in_w]  (or 0 if OOB padded),
//             with m -> (n, oh, ow) and k -> (ic, kh, kw)
//   B[oc, k]= W[oc, ic, kh, kw]   (already laid out OIHW = (C_out, K))
//
// The im2col matrix A is NEVER materialised — A-tile rows are gathered
// directly from X in shared memory each tile-iteration. WMMA 16x16x16 fragments
// (FP16 input, FP32 accumulator), 4-warp CTA, 64x64 output tile, BK=32.
// Bias is folded into the C-store epilogue. Output is written in NCHW layout,
// so the store re-scatters the (m=(n,oh,ow), oc) tile cell to
// Y[n*C_out*HW + oc*HW + oh*W_out + ow].
//
// Dispatch is restricted to the SD1.5-relevant cases:
//   * 3x3, stride 1, pad 1, dilation 1   (almost every UNet/VAE conv)
//   * 1x1, stride 1, pad 0, dilation 1   (point-wise convs in attention blocks)
// Other shapes (stride 2 downsamplers, dilated, asymmetric) fall through to
// the naive direct-conv kernel in conv2d.cu via the dispatch helper in
// conv2d.cu — this file only provides the WMMA entry point.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor {
namespace conv2d_wmma_internal {

using namespace nvcuda;

// Element-type traits so the WMMA kernel can be a single template over both
// __half and __nv_bfloat16. RTX 4090 (sm_89) supports BF16 WMMA fragments;
// load_matrix_sync / mma_sync / store_matrix_sync are identical — only the
// fragment element type and the host-side conversions differ.
template <typename T> struct wmma_traits;
template <> struct wmma_traits<__half> {
    __device__ static __half from_f32(float v)  { return __float2half(v); }
    __device__ static float   to_f32(__half v)  { return __half2float(v); }
};
template <> struct wmma_traits<__nv_bfloat16> {
    __device__ static __nv_bfloat16 from_f32(float v)         { return __float2bfloat16(v); }
    __device__ static float         to_f32(__nv_bfloat16 v)   { return __bfloat162float(v); }
};

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BM = 64;       // rows of A-tile / output pixels per CTA
static constexpr int BN = 64;       // cols of B-tile / output channels per CTA
static constexpr int BK = 32;       // inner-product chunk size
static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 2;
static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;   // 4
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32; // 128
static constexpr int WM = BM / WARPS_M;  // 32
static constexpr int WN = BN / WARPS_N;  // 32
static constexpr int FRAGS_M = WM / WMMA_M;  // 2
static constexpr int FRAGS_N = WN / WMMA_N;  // 2

static constexpr int LDA_SMEM = BK + 8;  // padding to reduce bank conflicts
static constexpr int LDB_SMEM = BK + 8;

// kH * kW must be a compile-time constant for index unrolling, so the kernel
// is templated on (KH, KW, PAD, STRIDE).  For SD1.5 we instantiate KH=KW=3,
// PAD=1, STRIDE=1 and KH=KW=1, PAD=0, STRIDE=1.
template <typename T,
          int KH, int KW, int PAD_H, int PAD_W, int STRIDE_H, int STRIDE_W>
__launch_bounds__(THREADS_PER_CTA)
__global__ void conv2d_implicit_gemm_wmma_kernel(
        const T* __restrict__ X,
        const T* __restrict__ Wt,
        const T* __restrict__ bias,   // may be null
        T* __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int H_out, int W_out) {
    constexpr int KHW = KH * KW;
    using TR = wmma_traits<T>;

    __shared__ T As[BM][LDA_SMEM];
    __shared__ T Bs[BN][LDB_SMEM];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;   // row of A-tile (output-pixel index)
    const int block_n = blockIdx.x * BN;   // col of B-tile (output channel)

    const int HW_out  = H_out * W_out;
    const int K_total = C_in * KHW;

    // Accumulator fragments (FP32).
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Loop over K dimension in chunks of BK.
    for (int k0 = 0; k0 < K_total; k0 += BK) {
        // ---- Load A tile (BM rows by BK cols), gathering from X ----
        // Each thread loads (BM*BK)/THREADS_PER_CTA = 16 halves total.
        // Layout: 128 threads cover 64 rows x 32 cols. Two threads per row
        // (cols [0..15] and [16..31]) — but per-element load (not vectorised)
        // because each col can come from a different (n, ic, kh, kw) and the
        // address pattern is gather-shaped.
        {
            constexpr int kElemsPerRow = BK;            // 32
            constexpr int kElemsPerCol = BM;            // 64
            constexpr int kElemsTotal  = BM * BK;       // 2048
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;  // 16

            #pragma unroll
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / kElemsPerRow;       // 0..BM-1
                const int col = lin - row * kElemsPerRow; // 0..BK-1
                const int gk  = k0 + col;                 // global K index
                const int m_g = block_m + row;            // global output-pixel index

                T v = TR::from_f32(0.0f);
                if (m_g < N * HW_out && gk < K_total) {
                    // Decompose m_g -> (n, oh, ow).
                    const int n     = m_g / HW_out;
                    const int sp    = m_g - n * HW_out;
                    const int oh    = sp / W_out;
                    const int ow    = sp - oh * W_out;

                    // Decompose gk -> (ic, kh, kw).
                    const int ic    = gk / KHW;
                    const int khw   = gk - ic * KHW;
                    const int kh    = khw / KW;
                    const int kw    = khw - kh * KW;

                    const int in_h  = oh * STRIDE_H - PAD_H + kh;  // dil=1
                    const int in_w  = ow * STRIDE_W - PAD_W + kw;
                    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        v = X[((n * C_in + ic) * H + in_h) * W + in_w];
                    }
                }
                As[row][col] = v;
            }
        }

        // ---- Load B tile (BN rows by BK cols) from W ----
        // W is laid out (C_out, C_in*KHW) row-major == (N_gemm, K) row-major.
        // The int4 fast path requires both K_total and the per-thread gk to be
        // 8-aligned; for conv2d K = C_in*KH*KW which is often NOT a multiple of
        // 8 (e.g. C_in=4, K=4 or C_in=3, KHW=9 → K=27). Pick per-element fallback
        // unless K_total is 8-aligned. int4 carries 8 16-bit elements for both
        // __half and __nv_bfloat16.
        {
            constexpr int kHalvesPerLoad = 8;  // int4 = 8 16-bit elements
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;  // 256
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;  // 2

            const bool k_aligned8 = ((K_total & 7) == 0);

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);     // 0..BN-1
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_n + row;                  // oc
                const int gk   = k0 + gcol;

                T tmp[kHalvesPerLoad];
                if (k_aligned8 && grow < C_out && gk + kHalvesPerLoad <= K_total) {
                    const int4* src = reinterpret_cast<const int4*>(&Wt[grow * K_total + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        if (grow < C_out && gk_q < K_total) {
                            tmp[q] = Wt[grow * K_total + gk_q];
                        } else {
                            tmp[q] = TR::from_f32(0.0f);
                        }
                    }
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        __syncthreads();

        // ---- WMMA compute on shared-mem tiles ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> b_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const T* a_ptr = &As[warp_m * WM + i * WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const T* b_ptr = &Bs[warp_n * WN + j * WMMA_N][kk];
                wmma::load_matrix_sync(b_frag[j], b_ptr, LDB_SMEM);
            }
            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store C tile through shared mem, with bias epilogue ----
    // The accumulator is FP32; store it as FP32 in shared memory (WMMA has no
    // BF16 accumulator fragment, and an FP32 staging tile is numerically exact
    // for both the FP16 and BF16 storage paths — the final narrowing happens in
    // the scatter epilogue below).
    __shared__ float Cs[BM][BN + 8];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            float* c_ptr = &Cs[warp_m * WM + i * WMMA_M][warp_n * WN + j * WMMA_N];
            wmma::store_matrix_sync(c_ptr, c_frag[i][j], BN + 8, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // ---- Scatter Cs -> Y in NCHW layout, adding bias ----
    // BM*BN = 4096 cells, 128 threads, 32 per thread.  We don't vectorise
    // because per output channel we write to different (oh, ow) destinations
    // and the dest address is not contiguous in (oc, ow); we still benefit
    // from the WMMA tile reuse.
    {
        constexpr int kElemsPerCol = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;  // 32

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / kElemsPerCol;        // 0..BM-1, output-pixel index within tile
            const int col = lin - row * kElemsPerCol;  // 0..BN-1, oc within tile

            const int m_g = block_m + row;
            const int oc  = block_n + col;
            if (oc >= C_out) continue;
            if (m_g >= N * HW_out) continue;

            const int n   = m_g / HW_out;
            const int sp  = m_g - n * HW_out;
            // sp = oh * W_out + ow, so Y index = ((n * C_out + oc) * HW_out + sp).
            float v = Cs[row][col];
            if (bias) v += TR::to_f32(bias[oc]);
            Y[(n * C_out + oc) * HW_out + sp] = TR::from_f32(v);
        }
    }
}

// Element-type-generic dispatcher shared by the FP16 and BF16 entry points.
template <typename T>
static bool launch_conv2d_implicit_gemm_wmma_impl(
        const T* X, const T* Wt, const T* bias, T* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    if (dil_h != 1 || dil_w != 1) return false;

    // Only meaningful when the GEMM is non-trivial.
    const int M = N * H_out * W_out;
    if (M <= 0 || C_out <= 0 || C_in <= 0) return false;
    // Tiny problems: not worth the launch overhead.
    if (size_t(M) * size_t(C_out) < 1024) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((C_out + BN - 1) / BN, (M + BM - 1) / BM);

    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1) {
        conv2d_implicit_gemm_wmma_kernel<T, 3, 3, 1, 1, 1, 1>
            <<<grid, block, 0, cur_stream()>>>(X, Wt, bias, Y, N, C_in, H, W, C_out, H_out, W_out);
        return true;
    }
    if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) {
        conv2d_implicit_gemm_wmma_kernel<T, 1, 1, 0, 0, 1, 1>
            <<<grid, block, 0, cur_stream()>>>(X, Wt, bias, Y, N, C_in, H, W, C_out, H_out, W_out);
        return true;
    }
    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 2 && stride_w == 2) {
        conv2d_implicit_gemm_wmma_kernel<T, 3, 3, 1, 1, 2, 2>
            <<<grid, block, 0, cur_stream()>>>(X, Wt, bias, Y, N, C_in, H, W, C_out, H_out, W_out);
        return true;
    }
    return false;
}

// Entry point invoked from conv2d.cu's dispatch helper. Returns true if the
// WMMA path was used (caller skips the naive fallback in that case).
bool launch_conv2d_implicit_gemm_wmma(
        const __half* X, const __half* Wt, const __half* bias, __half* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    return launch_conv2d_implicit_gemm_wmma_impl<__half>(
        X, Wt, bias, Y, N, C_in, H, W, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out);
}

// BF16 twin. RTX 4090 (sm_89) supports BF16 WMMA fragments; the kernel is the
// same template instantiated with __nv_bfloat16.
bool launch_conv2d_implicit_gemm_wmma_bf16(
        const __nv_bfloat16* X, const __nv_bfloat16* Wt,
        const __nv_bfloat16* bias, __nv_bfloat16* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    return launch_conv2d_implicit_gemm_wmma_impl<__nv_bfloat16>(
        X, Wt, bias, Y, N, C_in, H, W, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out);
}

} // namespace conv2d_wmma_internal
} // namespace brotensor
