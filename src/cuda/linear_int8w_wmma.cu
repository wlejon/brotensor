// Tensor-core batched-linear forward, W8A16 variant.
//
// Y(B, M) = X(B, K) @ dequant(W_int8(M, K))^T + bias(M).
// Structural copy of fp16_matmul.cu's matmul_ABT WMMA kernel (16-bit A, 16-bit
// B, FP32 accum, 16-bit store) with two changes: the B-tile loader reads 8
// INT8 bytes as int2 and dequantises to the activation dtype using a
// per-output-row FP32 scale preloaded once into shared, and the C-store
// epilogue folds in an optional bias broadcast along M. The kernel is a single
// template over __half and __nv_bfloat16 (sm_89 supports BF16 WMMA fragments;
// only the fragment element type and the float conversions differ — see
// conv2d_wmma.cu for the same treatment). Same dispatch heuristic as the conv
// WMMA path (K%8 alignment, problem-size floor); returns false to fall back to
// the existing tiled kernel.
//
// The K loop is a two-stage software pipeline: tile kt+1's global reads are
// issued into registers before tile kt's MMAs so their latency hides behind
// tensor-core work, and the smem tiles are double-buffered so the register
// spill into stage kt+1 needs no extra barrier — one __syncthreads() per
// K tile total. This matters here more than in the plain FP16 kernel: at the
// big SwiGLU shapes the weight stream is ~100 MB per call and the serialized
// load->sync->mma form left the tensor cores idle for most of it.

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
namespace detail {
namespace cuda {
namespace linear_int8w_wmma_internal {

using namespace nvcuda;

// Element-type traits so the WMMA kernel can be a single template over both
// __half and __nv_bfloat16 (mirrors conv2d_wmma.cu's wmma_traits).
template <typename T> struct wmma_traits;
template <> struct wmma_traits<__half> {
    __device__ static __half from_f32(float v)  { return __float2half(v); }
    __device__ static float   to_f32(__half v)  { return __half2float(v); }
};
template <> struct wmma_traits<__nv_bfloat16> {
    __device__ static __nv_bfloat16 from_f32(float v)        { return __float2bfloat16(v); }
    __device__ static float         to_f32(__nv_bfloat16 v)  { return __bfloat162float(v); }
};

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BM = 128;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int WARPS_M = 4;
static constexpr int WARPS_N = 2;
static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
static constexpr int WM = BM / WARPS_M;
static constexpr int WN = BN / WARPS_N;
static constexpr int FRAGS_M = WM / WMMA_M;
static constexpr int FRAGS_N = WN / WMMA_N;

static constexpr int LDA_SMEM = BK + 8;
static constexpr int LDB_SMEM = BK + 8;

template <typename T>
__launch_bounds__(THREADS_PER_CTA)
__global__ void linear_int8w_a16_wmma_kernel(
        const T*       __restrict__ X,
        const int8_t*  __restrict__ W_int8,
        const float*   __restrict__ scales,
        const T*       __restrict__ bias,
        T*             __restrict__ Y,
        int B, int M, int K) {
    using TR = wmma_traits<T>;
    // Double-buffered K-loop tiles; the FP32 epilogue staging tile Cs aliases
    // them (both stages are dead once the loop's final __syncthreads()
    // passes) so the 128x64 CTA tile stays under the 48 KB static-shared
    // limit.
    constexpr int kAsBytes = 2 * BM * LDA_SMEM * static_cast<int>(sizeof(T));
    constexpr int kBsBytes = 2 * BN * LDB_SMEM * static_cast<int>(sizeof(T));
    constexpr int kCsBytes = BM * (BN + 8) * static_cast<int>(sizeof(float));
    constexpr int kSmemBytes =
        (kAsBytes + kBsBytes > kCsBytes) ? kAsBytes + kBsBytes : kCsBytes;
    __shared__ __align__(32) char smem_buf[kSmemBytes];
    auto* As = reinterpret_cast<T(*)[BM][LDA_SMEM]>(smem_buf);
    auto* Bs = reinterpret_cast<T(*)[BN][LDB_SMEM]>(smem_buf + kAsBytes);
    auto* Cs = reinterpret_cast<float(*)[BN + 8]>(smem_buf);
    __shared__ float Bs_scale[BN];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    // Panel swizzle: CTAs launch x-major, so with a naive (x=N, y=M) mapping
    // a concurrent wave is ~256 distinct output-column blocks whose combined
    // weight stream (e.g. 100 MB at 16384x6144) thrashes L2. Remap the linear
    // launch index so a wave covers a PANEL_N-column by full-M panel instead:
    // its weight columns and A rows both stay L2-resident.
    constexpr int PANEL_N = 16;
    const int lid = blockIdx.y * gridDim.x + blockIdx.x;
    const int p   = lid / (PANEL_N * gridDim.y);
    const int rem = lid - p * (PANEL_N * gridDim.y);
    const int pw  = min(PANEL_N, static_cast<int>(gridDim.x) - p * PANEL_N);
    const int block_m = (rem / pw) * BM;              // batch axis (B)
    const int block_n = (p * PANEL_N + rem % pw) * BN; // out axis (M)

    // Per-output-row scale is invariant across k; load once before the loop.
    if (tid < BN) {
        const int m_g = block_n + tid;
        Bs_scale[tid] = (m_g < M) ? scales[m_g] : 0.0f;
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

    // Per-thread register staging for one K tile (8-wide vector chunks).
    constexpr int kChunk   = 8;
    constexpr int kALoads  = (BM * BK / kChunk) / THREADS_PER_CTA;
    constexpr int kBLoads  = (BN * BK / kChunk) / THREADS_PER_CTA;
    static_assert(kALoads * THREADS_PER_CTA * kChunk == BM * BK, "A tile split");
    static_assert(kBLoads * THREADS_PER_CTA * kChunk == BN * BK, "B tile split");

    T      a_regs[kALoads][kChunk];
    int8_t b_regs[kBLoads][kChunk];

    // Issue tile k0's global reads into registers.
    auto gload = [&](int k0) {
        #pragma unroll
        for (int li = 0; li < kALoads; ++li) {
            const int lin  = tid + li * THREADS_PER_CTA;
            const int row  = lin / (BK / kChunk);
            const int gcol = (lin % (BK / kChunk)) * kChunk;
            const int grow = block_m + row;
            const int gk   = k0 + gcol;
            if (grow < B && gk + kChunk <= K) {
                *reinterpret_cast<int4*>(a_regs[li]) =
                    *reinterpret_cast<const int4*>(&X[grow * K + gk]);
            } else {
                #pragma unroll
                for (int q = 0; q < kChunk; ++q) {
                    const int gk_q = gk + q;
                    a_regs[li][q] = (grow < B && gk_q < K)
                                    ? X[grow * K + gk_q]
                                    : TR::from_f32(0.0f);
                }
            }
        }
        #pragma unroll
        for (int li = 0; li < kBLoads; ++li) {
            const int lin  = tid + li * THREADS_PER_CTA;
            const int row  = lin / (BK / kChunk);
            const int gcol = (lin % (BK / kChunk)) * kChunk;
            const int grow = block_n + row;
            const int gk   = k0 + gcol;
            // Dispatch guarantees K%8==0, so the vector path only needs the
            // row/tail bound checks.
            if (grow < M && gk + kChunk <= K) {
                *reinterpret_cast<int2*>(b_regs[li]) =
                    *reinterpret_cast<const int2*>(&W_int8[grow * K + gk]);
            } else {
                #pragma unroll
                for (int q = 0; q < kChunk; ++q) {
                    const int gk_q = gk + q;
                    b_regs[li][q] = (grow < M && gk_q < K)
                                    ? W_int8[grow * K + gk_q]
                                    : (int8_t)0;
                }
            }
        }
    };

    // Spill the staged registers into smem stage `st` (B dequantises here).
    auto sstore = [&](int st) {
        #pragma unroll
        for (int li = 0; li < kALoads; ++li) {
            const int lin  = tid + li * THREADS_PER_CTA;
            const int row  = lin / (BK / kChunk);
            const int gcol = (lin % (BK / kChunk)) * kChunk;
            *reinterpret_cast<int4*>(&As[st][row][gcol]) =
                *reinterpret_cast<int4*>(a_regs[li]);
        }
        #pragma unroll
        for (int li = 0; li < kBLoads; ++li) {
            const int lin  = tid + li * THREADS_PER_CTA;
            const int row  = lin / (BK / kChunk);
            const int gcol = (lin % (BK / kChunk)) * kChunk;
            const float s = Bs_scale[row];
            T tmp_h[kChunk];
            #pragma unroll
            for (int q = 0; q < kChunk; ++q) {
                tmp_h[q] = TR::from_f32(static_cast<float>(b_regs[li][q]) * s);
            }
            *reinterpret_cast<int4*>(&Bs[st][row][gcol]) =
                *reinterpret_cast<const int4*>(tmp_h);
        }
    };

    const int ktiles = (K + BK - 1) / BK;
    gload(0);
    sstore(0);
    __syncthreads();

    for (int kt = 0; kt < ktiles; ++kt) {
        const int cur = kt & 1;
        // Issue the next tile's global reads first: their latency hides
        // behind this tile's MMAs, and the spill below only stalls if the
        // loads still haven't landed by then.
        if (kt + 1 < ktiles) gload((kt + 1) * BK);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> b_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const T* a_ptr = &As[cur][warp_m * WM + i * WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const T* b_ptr = &Bs[cur][warp_n * WN + j * WMMA_N][kk];
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

        if (kt + 1 < ktiles) sstore(cur ^ 1);
        // One barrier per tile: it both publishes stage cur^1 for the next
        // iteration's MMAs and fences this iteration's MMA reads of stage
        // cur before iteration kt+1 overwrites it.
        __syncthreads();
    }

    // FP32 staging via Cs (aliasing As/Bs — see the smem_buf carve above;
    // WMMA has no BF16 accumulator fragment, and FP32 is numerically exact
    // for both storage paths — narrowing happens in the scatter epilogue
    // below). The k-loop's trailing __syncthreads() fences the aliased reads.
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            float* c_ptr = &Cs[warp_m * WM + i * WMMA_M][warp_n * WN + j * WMMA_N];
            wmma::store_matrix_sync(c_ptr, c_frag[i][j], BN + 8, wmma::mem_row_major);
        }
    }

    __syncthreads();

    {
        constexpr int kElemsPerRow = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / kElemsPerRow;
            const int col = lin - row * kElemsPerRow;

            const int grow = block_m + row;   // batch
            const int gcol = block_n + col;   // out
            if (grow >= B || gcol >= M) continue;

            float v = Cs[row][col];
            if (bias) v += TR::to_f32(bias[gcol]);
            Y[grow * M + gcol] = TR::from_f32(v);
        }
    }
}

// Element-type-generic dispatcher shared by the FP16 and BF16 entry points.
template <typename T>
static bool launch_linear_int8w_a16_wmma_impl(
        const T* X, const int8_t* W, const float* scales,
        const T* bias, T* Y,
        int B, int M, int K) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if ((K & 7) != 0) return false;
    if (K < 32) return false;
    if (size_t(B) * size_t(M) < 1024) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((M + BN - 1) / BN, (B + BM - 1) / BM);
    linear_int8w_a16_wmma_kernel<T><<<grid, block, 0, cur_stream()>>>(
        X, W, scales, bias, Y, B, M, K);
    return true;
}

bool launch_linear_int8w_fp16_wmma(
        const __half* X, const int8_t* W, const float* scales,
        const __half* bias, __half* Y,
        int B, int M, int K) {
    return launch_linear_int8w_a16_wmma_impl<__half>(
        X, W, scales, bias, Y, B, M, K);
}

// BF16-activation twin of launch_linear_int8w_fp16_wmma — exactly the same
// template instantiated with __nv_bfloat16 (sm_89 supports BF16 fragments).
bool launch_linear_int8w_bf16_wmma(
        const __nv_bfloat16* X, const int8_t* W, const float* scales,
        const __nv_bfloat16* bias, __nv_bfloat16* Y,
        int B, int M, int K) {
    return launch_linear_int8w_a16_wmma_impl<__nv_bfloat16>(
        X, W, scales, bias, Y, B, M, K);
}

} // namespace linear_int8w_wmma_internal
} // namespace cuda
} // namespace detail
} // namespace brotensor
