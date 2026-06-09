// Fused WMMA tensor-core GEMM for Q4_K weights.
//
// Y(B, M) = X(B, K) @ dequant(W_q4k(M, K))^T + bias(M).
//
// Structural copy of linear_int8w_wmma.cu's tile geometry (BM=64, BN=64,
// BK=32, 2x2 warps, FP32 accum, FP16 store, optional FP16 bias along M).
// BK == 32 deliberately matches one Q4_K sub-block: each K-tile covers a
// single sub-block, so the B-tile loader runs the Q4_K decode once per
// sub-block per output row. Per-super-block header bytes (d, dmin, scales[12])
// are preloaded into shared once every 8 K-tiles; (sc, m) for the current
// sub-block are likewise resolved once per K-tile per row. Inside the K-tile
// each thread decodes 8 nibbles via one int2 load of qs[] and writes them
// to Bs via an int4 FP16 store.

#include "q4k_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

#include "detail/cuda_check.h"

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda::q4k_wmma_internal {

using namespace nvcuda;
namespace q4k_ns = ::brotensor::detail::cuda::q4k;

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

// Counter for tests: incremented once per kernel launch.
__device__ unsigned long long g_q4k_wmma_call_counter = 0;

__launch_bounds__(THREADS_PER_CTA)
__global__ void linear_q4k_fp16_wmma_kernel(
        const __half*  __restrict__ X,
        const uint8_t* __restrict__ W_q4k,
        const __half*  __restrict__ bias,
        __half*        __restrict__ Y,
        int B, int M, int K) {
    __shared__ __half  As[BM][LDA_SMEM];
    __shared__ __half  Bs[BN][LDB_SMEM];

    // Super-block header cache (preloaded once per 256-element super-block).
    __shared__ __half  rowD   [BN];
    __shared__ __half  rowDmin[BN];
    __shared__ uint8_t rowScales[BN][12];

    // Per-K-tile (sub-block) decoded sub-scale + sub-min.
    __shared__ float   rowDS[BN];
    __shared__ float   rowDM[BN];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;   // batch axis (B)
    const int block_n = blockIdx.x * BN;   // out axis (M)

    if (tid == 0) {
        atomicAdd(&g_q4k_wmma_call_counter, 1ull);
    }

    const int blocks_per_row = K / q4k_ns::kBlockElems;
    const size_t row_stride_bytes =
        static_cast<size_t>(blocks_per_row) * q4k_ns::kBlockBytes;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int last_super_idx = -1;

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

        // ---- Load B tile (BN x BK), dequantising Q4_K -> FP16 ----
        const int super_idx = k0 / q4k_ns::kBlockElems;
        const int sub_idx   = (k0 / q4k_ns::kSubBlockElems) % q4k_ns::kSubBlocks;

        // (a) On entering a new super-block, refresh the row header cache.
        if (super_idx != last_super_idx) {
            __syncthreads();
            // 64 rows x 16 bytes per row = 1024 bytes; 128 threads => 8 bytes each.
            // Simpler: each thread handles one (row, byte) pair across two passes.
            constexpr int kHdrBytesPerRow = 16;  // d(2) + dmin(2) + scales(12)
            constexpr int kHdrBytesTotal  = BN * kHdrBytesPerRow;
            constexpr int kHdrPerThr      = kHdrBytesTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int hi = 0; hi < kHdrPerThr; ++hi) {
                const int lin = tid + hi * THREADS_PER_CTA;
                const int row = lin / kHdrBytesPerRow;
                const int off = lin - row * kHdrBytesPerRow;
                const int g_row = block_n + row;

                uint8_t v = 0;
                if (g_row < M) {
                    const uint8_t* blk = W_q4k
                        + static_cast<size_t>(g_row) * row_stride_bytes
                        + static_cast<size_t>(super_idx) * q4k_ns::kBlockBytes;
                    v = blk[off];
                }
                if (off == 0) {
                    reinterpret_cast<uint8_t*>(&rowD[row])[0] = v;
                } else if (off == 1) {
                    reinterpret_cast<uint8_t*>(&rowD[row])[1] = v;
                } else if (off == 2) {
                    reinterpret_cast<uint8_t*>(&rowDmin[row])[0] = v;
                } else if (off == 3) {
                    reinterpret_cast<uint8_t*>(&rowDmin[row])[1] = v;
                } else {
                    rowScales[row][off - 4] = v;
                }
            }
            __syncthreads();
            last_super_idx = super_idx;
        }

        // (b) Decode (sc, m) for the current sub_idx, per output row.
        __syncthreads();
        if (tid < BN) {
            uint8_t sc, m;
            q4k_ns::unpack_sc_m(sub_idx, rowScales[tid], sc, m);
            const float d_f    = __half2float(rowD[tid]);
            const float dmin_f = __half2float(rowDmin[tid]);
            rowDS[tid] = d_f    * static_cast<float>(sc);
            rowDM[tid] = dmin_f * static_cast<float>(m);
        }
        __syncthreads();

        // (c) Load and dequantise this sub-block's 32 nibbles per row.
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            const int pair = sub_idx >> 1;
            const bool hi_nib = (sub_idx & 1) != 0;
            const int qs_pair_off = q4k_ns::kQsOffset + pair * 32;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int g_row = block_n + row;

                __half tmp_h[kHalvesPerLoad];
                if (g_row < M) {
                    const uint8_t* blk = W_q4k
                        + static_cast<size_t>(g_row) * row_stride_bytes
                        + static_cast<size_t>(super_idx) * q4k_ns::kBlockBytes;
                    const uint8_t* qs8 = blk + qs_pair_off + gcol;
                    // Read 8 nibble-bytes via int2.
                    uint8_t qbytes[kHalvesPerLoad];
                    *reinterpret_cast<int2*>(qbytes) =
                        *reinterpret_cast<const int2*>(qs8);
                    const float ds = rowDS[row];
                    const float dm = rowDM[row];
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int nib = hi_nib ? (qbytes[q] >> 4)
                                               : (qbytes[q] & 0x0F);
                        tmp_h[q] = __float2half_rn(ds * static_cast<float>(nib) - dm);
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

            const int grow = block_m + row;   // batch
            const int gcol = block_n + col;   // out
            if (grow >= B || gcol >= M) continue;

            float v = __half2float(Cs[row][col]);
            if (bias) v += __half2float(bias[gcol]);
            Y[grow * M + gcol] = __float2half(v);
        }
    }
}

// Host helper: read + reset the WMMA call counter. Used by parity tests.
__global__ void q4k_wmma_read_counter_kernel(unsigned long long* out, int reset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = g_q4k_wmma_call_counter;
        if (reset) g_q4k_wmma_call_counter = 0;
    }
}

bool launch_linear_q4k_fp16_wmma(const __half* X, const uint8_t* W_q4k,
                                 const __half* bias, __half* Y,
                                 int B, int M, int K, cudaStream_t stream) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if (K % q4k_ns::kBlockElems != 0) return false;
    if (B < 4) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((M + BN - 1) / BN, (B + BM - 1) / BM);
    linear_q4k_fp16_wmma_kernel<<<grid, block, 0, stream>>>(
        X, W_q4k, bias, Y, B, M, K);
    return true;
}

}  // namespace brotensor::detail::cuda::q4k_wmma_internal

// C-linkage accessor for tests: returns the current call counter and resets it.
extern "C" unsigned long long brotensor_q4k_wmma_calls_consume() {
    using namespace brotensor::detail::cuda::q4k_wmma_internal;
    unsigned long long h_val = 0;
    unsigned long long* d_buf = nullptr;
    if (cudaMalloc(&d_buf, sizeof(unsigned long long)) != cudaSuccess) return 0;
    q4k_wmma_read_counter_kernel<<<1, 1, 0, cur_stream()>>>(d_buf, /*reset=*/1);
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(&h_val, d_buf, sizeof(unsigned long long),
                                         cudaMemcpyDeviceToHost, cur_stream()));
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(cur_stream()));
    cudaFree(d_buf);
    return h_val;
}
