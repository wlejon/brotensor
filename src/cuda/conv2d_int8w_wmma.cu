// Tensor-core implicit-GEMM conv2d forward, W8A16 variant.
//
// Identical to conv2d_wmma.cu (FP16 WMMA implicit-GEMM) except the B-tile
// loader dequantises INT8 weights to FP16 on the way into shared memory using
// the per-output-row FP32 scale. A-tile gather, WMMA mma_sync, and the FP16
// epilogue (including bias add) are unchanged. The per-row scale vector for
// the 64 BN rows of this CTA is loaded into shared scratch once before the
// k-loop (it is invariant in k).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace brotensor {
namespace conv2d_int8w_wmma_internal {

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 2;
static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
static constexpr int WM = BM / WARPS_M;
static constexpr int WN = BN / WARPS_N;
static constexpr int FRAGS_M = WM / WMMA_M;
static constexpr int FRAGS_N = WN / WMMA_N;

static constexpr int LDA_SMEM = BK + 8;
static constexpr int LDB_SMEM = BK + 8;

template <int KH, int KW, int PAD_H, int PAD_W, int STRIDE_H, int STRIDE_W>
__launch_bounds__(THREADS_PER_CTA)
__global__ void conv2d_int8w_implicit_gemm_wmma_kernel(
        const __half*  __restrict__ X,
        const int8_t*  __restrict__ W_int8,
        const float*   __restrict__ scales,
        const __half*  __restrict__ bias,
        __half*        __restrict__ Y,
        int N, int C_in, int H, int W,
        int C_out, int H_out, int W_out) {
    constexpr int KHW = KH * KW;

    __shared__ __half As[BM][LDA_SMEM];
    __shared__ __half Bs[BN][LDB_SMEM];
    __shared__ float  Bs_scale[BN];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    const int HW_out  = H_out * W_out;
    const int K_total = C_in * KHW;

    // Per-output-row scale is invariant across k; load once before the loop.
    if (tid < BN) {
        const int oc = block_n + tid;
        Bs_scale[tid] = (oc < C_out) ? scales[oc] : 0.0f;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K_total; k0 += BK) {
        // ---- Load A tile (BM rows by BK cols), gathering from X ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BM * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;
                const int m_g = block_m + row;

                __half v = __float2half(0.0f);
                if (m_g < N * HW_out && gk < K_total) {
                    const int n     = m_g / HW_out;
                    const int sp    = m_g - n * HW_out;
                    const int oh    = sp / W_out;
                    const int ow    = sp - oh * W_out;

                    const int ic    = gk / KHW;
                    const int khw   = gk - ic * KHW;
                    const int kh    = khw / KW;
                    const int kw    = khw - kh * KW;

                    const int in_h  = oh * STRIDE_H - PAD_H + kh;
                    const int in_w  = ow * STRIDE_W - PAD_W + kw;
                    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        v = X[((n * C_in + ic) * H + in_h) * W + in_w];
                    }
                }
                As[row][col] = v;
            }
        }

        // ---- Load B tile (BN rows by BK cols), dequantising INT8 -> FP16 ----
        {
            constexpr int kHalvesPerLoad = 8;  // 8 INT8 bytes -> 8 __half
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            const bool k_aligned8 = ((K_total & 7) == 0);

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_n + row;
                const int gk   = k0 + gcol;

                int8_t tmp8[kHalvesPerLoad];
                if (k_aligned8 && grow < C_out && gk + kHalvesPerLoad <= K_total) {
                    const int2* src = reinterpret_cast<const int2*>(
                        &W_int8[grow * K_total + gk]);
                    *reinterpret_cast<int2*>(tmp8) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        if (grow < C_out && gk_q < K_total) {
                            tmp8[q] = W_int8[grow * K_total + gk_q];
                        } else {
                            tmp8[q] = 0;
                        }
                    }
                }
                const float s = Bs_scale[row];
                __half tmp_h[kHalvesPerLoad];
                #pragma unroll
                for (int q = 0; q < kHalvesPerLoad; ++q) {
                    tmp_h[q] = __float2half(static_cast<float>(tmp8[q]) * s);
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) =
                    *reinterpret_cast<const int4*>(tmp_h);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const __half* a_ptr = &As[warp_m * WM + i * WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const __half* b_ptr = &Bs[warp_n * WN + j * WMMA_N][kk];
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

    __shared__ __half Cs[BM][BN + 8];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_h;
            #pragma unroll
            for (int e = 0; e < c_frag[i][j].num_elements; ++e) {
                c_h.x[e] = __float2half(c_frag[i][j].x[e]);
            }
            __half* c_ptr = &Cs[warp_m * WM + i * WMMA_M][warp_n * WN + j * WMMA_N];
            wmma::store_matrix_sync(c_ptr, c_h, BN + 8, wmma::mem_row_major);
        }
    }

    __syncthreads();

    {
        constexpr int kElemsPerCol = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / kElemsPerCol;
            const int col = lin - row * kElemsPerCol;

            const int m_g = block_m + row;
            const int oc  = block_n + col;
            if (oc >= C_out) continue;
            if (m_g >= N * HW_out) continue;

            const int n   = m_g / HW_out;
            const int sp  = m_g - n * HW_out;
            float v = __half2float(Cs[row][col]);
            if (bias) v += __half2float(bias[oc]);
            Y[(n * C_out + oc) * HW_out + sp] = __float2half(v);
        }
    }
}

bool launch_conv2d_int8w_implicit_gemm_wmma(
        const __half* X, const int8_t* W_int8, const float* scales,
        const __half* bias, __half* Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out) {
    if (dil_h != 1 || dil_w != 1) return false;

    const int M = N * H_out * W_out;
    if (M <= 0 || C_out <= 0 || C_in <= 0) return false;
    if (size_t(M) * size_t(C_out) < 1024) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((C_out + BN - 1) / BN, (M + BM - 1) / BM);

    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1) {
        conv2d_int8w_implicit_gemm_wmma_kernel<3, 3, 1, 1, 1, 1>
            <<<grid, block>>>(X, W_int8, scales, bias, Y, N, C_in, H, W, C_out, H_out, W_out);
        return true;
    }
    if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1) {
        conv2d_int8w_implicit_gemm_wmma_kernel<1, 1, 0, 0, 1, 1>
            <<<grid, block>>>(X, W_int8, scales, bias, Y, N, C_in, H, W, C_out, H_out, W_out);
        return true;
    }
    if (kH == 3 && kW == 3 && pad_h == 1 && pad_w == 1 && stride_h == 2 && stride_w == 2) {
        conv2d_int8w_implicit_gemm_wmma_kernel<3, 3, 1, 1, 2, 2>
            <<<grid, block>>>(X, W_int8, scales, bias, Y, N, C_in, H, W, C_out, H_out, W_out);
        return true;
    }
    return false;
}

} // namespace conv2d_int8w_wmma_internal
} // namespace brotensor
