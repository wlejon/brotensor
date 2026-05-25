// Fused WMMA tensor-core GEMM for Q8_0 weights.
//
// Y(B, M) = X(B, K) @ dequant(W_q8_0(M, K))^T + bias(M).
//
// Structural copy of q4k_wmma.cu's tile geometry (BM=64, BN=64, BK=32,
// 2x2 warps, FP32 accum, FP16 store, optional FP16 bias along M). BK == 32
// equals one Q8_0 block, so every K-tile is a fresh block (no super-block
// reuse to amortise). Per K-tile each output row needs its 2-byte d + 32
// int8 qs[]; these are preloaded into shared, then each B-tile thread reads
// 8 contiguous int8 bytes, multiplies by d_f32, and stores 8 __halves into
// Bs via an int4 FP16 store.

#include "q8_0_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace brotensor::detail::cuda::q8_0_wmma_internal {

using namespace nvcuda;
namespace q8_ns = ::brotensor::detail::cuda::q8_0;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;   // = q8_0::kBlockElems
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

__device__ unsigned long long g_q8_0_wmma_call_counter = 0;

__launch_bounds__(THREADS_PER_CTA)
__global__ void linear_q8_0_fp16_wmma_kernel(
        const __half*  __restrict__ X,
        const uint8_t* __restrict__ W_q8,
        const __half*  __restrict__ bias,
        __half*        __restrict__ Y,
        int B, int M, int K) {
    __shared__ __half  As[BM][LDA_SMEM];
    __shared__ __half  Bs[BN][LDB_SMEM];

    // Per-K-tile (one Q8_0 block) header cache.
    __shared__ float   rowD[BN];
    __shared__ int8_t  rowQs[BN][BK];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    if (tid == 0) {
        atomicAdd(&g_q8_0_wmma_call_counter, 1ull);
    }

    const int blocks_per_row = K / q8_ns::kBlockElems;
    const size_t row_stride_bytes =
        static_cast<size_t>(blocks_per_row) * q8_ns::kBlockBytes;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[FRAGS_M][FRAGS_N];
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

        const int block_idx = k0 / q8_ns::kBlockElems;

        // ---- Refresh per-row block header (d + qs[32]) for this K-tile ----
        __syncthreads();
        // 64 rows; 256 threads => 4 rows per thread, one thread loads d.
        // Load d (one half per row): tid < BN handles its row's d.
        if (tid < BN) {
            const int g_row = block_n + tid;
            if (g_row < M) {
                const uint8_t* blk = W_q8
                    + static_cast<size_t>(g_row) * row_stride_bytes
                    + static_cast<size_t>(block_idx) * q8_ns::kBlockBytes;
                const __half d_h = *reinterpret_cast<const __half*>(blk + q8_ns::kDOffset);
                rowD[tid] = __half2float(d_h);
            } else {
                rowD[tid] = 0.0f;
            }
        }
        // Load qs[32]: 64 rows * 32 bytes = 2048 bytes; 256 threads => 8 bytes each.
        {
            constexpr int kBytesPerRow = BK;            // 32
            constexpr int kBytesTotal  = BN * kBytesPerRow;  // 2048
            constexpr int kBytesPerThr = kBytesTotal / THREADS_PER_CTA;  // 8

            #pragma unroll
            for (int li = 0; li < kBytesPerThr / 8; ++li) {
                // Each thread does one 8-byte int2 load.
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (kBytesPerRow / 8);
                const int col_grp = lin % (kBytesPerRow / 8);
                const int bcol = col_grp * 8;
                const int g_row = block_n + row;

                int8_t tmp[8];
                if (g_row < M) {
                    // qs starts at +2 within a 34-byte block, so blocks are
                    // not 8-byte aligned overall. Read bytes individually.
                    const uint8_t* blk = W_q8
                        + static_cast<size_t>(g_row) * row_stride_bytes
                        + static_cast<size_t>(block_idx) * q8_ns::kBlockBytes;
                    const uint8_t* src = blk + q8_ns::kQsOffset + bcol;
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        tmp[q] = static_cast<int8_t>(src[q]);
                    }
                } else {
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) tmp[q] = 0;
                }
                *reinterpret_cast<int2*>(&rowQs[row][bcol]) =
                    *reinterpret_cast<int2*>(tmp);
            }
        }
        __syncthreads();

        // ---- Dequantise per row, store into Bs ----
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int g_row = block_n + row;

                __half tmp_h[kHalvesPerLoad];
                if (g_row < M) {
                    int8_t qbytes[kHalvesPerLoad];
                    *reinterpret_cast<int2*>(qbytes) =
                        *reinterpret_cast<const int2*>(&rowQs[row][gcol]);
                    const float d_f = rowD[row];
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        tmp_h[q] = __float2half_rn(d_f * static_cast<float>(qbytes[q]));
                    }
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        tmp_h[q] = __float2half(0.0f);
                    }
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) =
                    *reinterpret_cast<int4*>(tmp_h);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
                a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major>
                b_frag[FRAGS_N];

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

            const int grow = block_m + row;
            const int gcol = block_n + col;
            if (grow >= B || gcol >= M) continue;

            float v = __half2float(Cs[row][col]);
            if (bias) v += __half2float(bias[gcol]);
            Y[grow * M + gcol] = __float2half(v);
        }
    }
}

__global__ void q8_0_wmma_read_counter_kernel(unsigned long long* out, int reset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = g_q8_0_wmma_call_counter;
        if (reset) g_q8_0_wmma_call_counter = 0;
    }
}

bool launch_linear_q8_0_fp16_wmma(const __half* X, const uint8_t* W_q8,
                                  const __half* bias, __half* Y,
                                  int B, int M, int K, cudaStream_t stream) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if (K % q8_ns::kBlockElems != 0) return false;
    if (B < 4) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((M + BN - 1) / BN, (B + BM - 1) / BM);
    linear_q8_0_fp16_wmma_kernel<<<grid, block, 0, stream>>>(
        X, W_q8, bias, Y, B, M, K);
    return true;
}

}  // namespace brotensor::detail::cuda::q8_0_wmma_internal

extern "C" unsigned long long brotensor_q8_0_wmma_calls_consume() {
    using namespace brotensor::detail::cuda::q8_0_wmma_internal;
    unsigned long long h_val = 0;
    unsigned long long* d_buf = nullptr;
    if (cudaMalloc(&d_buf, sizeof(unsigned long long)) != cudaSuccess) return 0;
    q8_0_wmma_read_counter_kernel<<<1, 1>>>(d_buf, /*reset=*/1);
    cudaMemcpy(&h_val, d_buf, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    return h_val;
}
