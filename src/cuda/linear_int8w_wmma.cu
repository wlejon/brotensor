// Tensor-core batched-linear forward, W8A16 variant.
//
// Y(B, M) = X(B, K) @ dequant(W_int8(M, K))^T + bias(M).
// Structural copy of fp16_matmul.cu's matmul_ABT WMMA kernel (FP16 A, FP16 B,
// FP32 accum, FP16 store) with two changes: the B-tile loader reads 8 INT8
// bytes as int2 and dequantises to FP16 using a per-output-row FP32 scale
// preloaded once into shared, and the C-store epilogue folds in an optional
// FP16 bias broadcast along M. Same dispatch heuristic as the conv WMMA path
// (K%8 alignment, problem-size floor); returns false to fall back to the
// existing tiled kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace brotensor {
namespace linear_int8w_wmma_internal {

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

__launch_bounds__(THREADS_PER_CTA)
__global__ void linear_int8w_fp16_wmma_kernel(
        const __half*  __restrict__ X,
        const int8_t*  __restrict__ W_int8,
        const float*   __restrict__ scales,
        const __half*  __restrict__ bias,
        __half*        __restrict__ Y,
        int B, int M, int K) {
    __shared__ __half As[BM][LDA_SMEM];
    __shared__ __half Bs[BN][LDB_SMEM];
    __shared__ float  Bs_scale[BN];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;   // batch axis (B)
    const int block_n = blockIdx.x * BN;   // out axis (M)

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

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile (BM x BK) from X(B, K) ----
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BM * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_m + row;
                const int gk   = k0 + gcol;

                __half tmp[kHalvesPerLoad];
                if (grow < B && gk + kHalvesPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&X[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        tmp[q] = (grow < B && gk_q < K)
                                 ? X[grow * K + gk_q]
                                 : __float2half(0.0f);
                    }
                }
                *reinterpret_cast<int4*>(&As[row][gcol]) =
                    *reinterpret_cast<int4*>(tmp);
            }
        }

        // ---- Load B tile (BN x BK), dequantising INT8 -> FP16 ----
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            const bool k_aligned8 = ((K & 7) == 0);

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_n + row;
                const int gk   = k0 + gcol;

                int8_t tmp8[kHalvesPerLoad];
                if (k_aligned8 && grow < M && gk + kHalvesPerLoad <= K) {
                    const int2* src = reinterpret_cast<const int2*>(
                        &W_int8[grow * K + gk]);
                    *reinterpret_cast<int2*>(tmp8) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        tmp8[q] = (grow < M && gk_q < K)
                                  ? W_int8[grow * K + gk_q]
                                  : (int8_t)0;
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

            float v = __half2float(Cs[row][col]);
            if (bias) v += __half2float(bias[gcol]);
            Y[grow * M + gcol] = __float2half(v);
        }
    }
}

bool launch_linear_int8w_fp16_wmma(
        const __half* X, const int8_t* W, const float* scales,
        const __half* bias, __half* Y,
        int B, int M, int K) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if ((K & 7) != 0) return false;
    if (K < 32) return false;
    if (size_t(B) * size_t(M) < 1024) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((M + BN - 1) / BN, (B + BM - 1) / BM);
    linear_int8w_fp16_wmma_kernel<<<grid, block>>>(
        X, W, scales, bias, Y, B, M, K);
    return true;
}

} // namespace linear_int8w_wmma_internal
} // namespace brotensor
