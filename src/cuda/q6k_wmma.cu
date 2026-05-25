// Fused WMMA tensor-core GEMM for Q6_K weights.
//
// Y(B, M) = X(B, K) @ dequant(W_q6k(M, K))^T + bias(M).
//
// Structural copy of q4k_wmma.cu's tile geometry (BM=64, BN=64, BK=32,
// 2x2 warps, FP32 accum, FP16 store, optional FP16 bias along M).
//
// Q6_K super-block = 256 elements = 8 K-tiles of 32. tile_in_sb = 0..7
// maps to (group, quad) = (tile_in_sb/4, tile_in_sb%4). Per K-tile and per
// output row we need:
//   - 32 ql bytes from ql[group*64 + (quad%2)*32 + 0..31]
//   - 32 qh bytes from qh[group*32 + 0..31]
//   - two int8 scales: scales[group*8 + quad*2 + {0,1}] (covers l<16 vs l>=16)
//   - the super-block fp16 scale `d` (cached per super-block)
//
// On entering a new super-block we preload `d` and all 16 scales per row.
// At each K-tile we preload the 64 ql+qh bytes per row, then in the B-tile
// loader each thread reads 8 ql + 8 qh bytes, reconstructs the 8 signed
// 6-bit values (val6 in [-32, 31]), and writes them as 8 FP16 values into
// Bs via an int4 store. The scale index sb depends on (group, quad, l/16).

#include "q6k_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace brotensor::detail::cuda::q6k_wmma_internal {

using namespace nvcuda;
namespace q6k_ns = ::brotensor::detail::cuda::q6k;

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

__device__ unsigned long long g_q6k_wmma_call_counter = 0;

__launch_bounds__(THREADS_PER_CTA)
__global__ void linear_q6k_fp16_wmma_kernel(
        const __half*  __restrict__ X,
        const uint8_t* __restrict__ W_q6k,
        const __half*  __restrict__ bias,
        __half*        __restrict__ Y,
        int B, int M, int K) {
    __shared__ __half  As[BM][LDA_SMEM];
    __shared__ __half  Bs[BN][LDB_SMEM];

    // Super-block header cache (refreshed once per 256-element super-block).
    __shared__ float   rowD[BN];
    __shared__ int8_t  rowScales[BN][16];

    // Per-K-tile (32-elem sub-tile) byte cache: 32 ql bytes + 32 qh bytes per row.
    __shared__ uint8_t rowQl[BN][BK];
    __shared__ uint8_t rowQh[BN][BK];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    if (tid == 0) {
        atomicAdd(&g_q6k_wmma_call_counter, 1ull);
    }

    const int blocks_per_row = K / q6k_ns::kBlockElems;
    const size_t row_stride_bytes =
        static_cast<size_t>(blocks_per_row) * q6k_ns::kBlockBytes;

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

        const int super_idx   = k0 / q6k_ns::kBlockElems;
        const int tile_in_sb  = (k0 % q6k_ns::kBlockElems) / BK;  // 0..7
        const int group       = tile_in_sb >> 2;                  // 0..1
        const int quad        = tile_in_sb & 3;                   // 0..3

        // (a) On entering a new super-block, refresh d + 16 scales per row.
        if (super_idx != last_super_idx) {
            __syncthreads();
            // 64 rows -- one thread per row loads d.
            if (tid < BN) {
                const int g_row = block_n + tid;
                if (g_row < M) {
                    const uint8_t* blk = W_q6k
                        + static_cast<size_t>(g_row) * row_stride_bytes
                        + static_cast<size_t>(super_idx) * q6k_ns::kBlockBytes;
                    const __half d_h = *reinterpret_cast<const __half*>(blk + q6k_ns::kDOffset);
                    rowD[tid] = __half2float(d_h);
                } else {
                    rowD[tid] = 0.0f;
                }
            }
            // 64 rows * 16 scale bytes = 1024 bytes; 256 threads => 4 bytes each.
            {
                constexpr int kBytesPerRow = 16;
                constexpr int kBytesTotal  = BN * kBytesPerRow;       // 1024
                constexpr int kBytesPerThr = kBytesTotal / THREADS_PER_CTA;  // 4

                #pragma unroll
                for (int li = 0; li < kBytesPerThr; ++li) {
                    const int lin = tid + li * THREADS_PER_CTA;
                    const int row = lin / kBytesPerRow;
                    const int off = lin - row * kBytesPerRow;
                    const int g_row = block_n + row;

                    int8_t v = 0;
                    if (g_row < M) {
                        const uint8_t* blk = W_q6k
                            + static_cast<size_t>(g_row) * row_stride_bytes
                            + static_cast<size_t>(super_idx) * q6k_ns::kBlockBytes;
                        v = static_cast<int8_t>(blk[q6k_ns::kScalesOffset + off]);
                    }
                    rowScales[row][off] = v;
                }
            }
            __syncthreads();
            last_super_idx = super_idx;
        }

        // (b) Load per-K-tile ql[32] and qh[32] for each row.
        __syncthreads();
        {
            // 64 rows * 32 ql bytes = 2048 bytes; 256 threads => 8 bytes each.
            constexpr int kBytesPerRow = BK;   // 32
            constexpr int kBytesTotal  = BN * kBytesPerRow;
            constexpr int kBytesPerThr = kBytesTotal / THREADS_PER_CTA;  // 8
            constexpr int kInt2PerThr  = kBytesPerThr / 8;               // 1

            const int ql_tile_off = q6k_ns::kQlOffset + group * 64 + (quad & 1) * 32;
            const int qh_tile_off = q6k_ns::kQhOffset + group * 32;

            #pragma unroll
            for (int li = 0; li < kInt2PerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (kBytesPerRow / 8);
                const int col_grp = lin % (kBytesPerRow / 8);
                const int bcol = col_grp * 8;
                const int g_row = block_n + row;

                int8_t ql_tmp[8], qh_tmp[8];
                if (g_row < M) {
                    // Blocks are 210 bytes (210 % 8 = 2), so block bases are
                    // not 8-aligned; read bytes individually.
                    const uint8_t* blk = W_q6k
                        + static_cast<size_t>(g_row) * row_stride_bytes
                        + static_cast<size_t>(super_idx) * q6k_ns::kBlockBytes;
                    const uint8_t* ql_src = blk + ql_tile_off + bcol;
                    const uint8_t* qh_src = blk + qh_tile_off + bcol;
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        ql_tmp[q] = static_cast<int8_t>(ql_src[q]);
                        qh_tmp[q] = static_cast<int8_t>(qh_src[q]);
                    }
                } else {
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) { ql_tmp[q] = 0; qh_tmp[q] = 0; }
                }
                *reinterpret_cast<int2*>(&rowQl[row][bcol]) =
                    *reinterpret_cast<int2*>(ql_tmp);
                *reinterpret_cast<int2*>(&rowQh[row][bcol]) =
                    *reinterpret_cast<int2*>(qh_tmp);
            }
        }
        __syncthreads();

        // (c) Decode + dequantise this K-tile per row, store into Bs.
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            const int sb_base = group * 8 + quad * 2;   // l<16 -> sb_base; l>=16 -> sb_base+1
            const bool quad_low = (quad < 2);

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int g_row = block_n + row;

                __half tmp_h[kHalvesPerLoad];
                if (g_row < M) {
                    uint8_t ql_b[kHalvesPerLoad], qh_b[kHalvesPerLoad];
                    *reinterpret_cast<int2*>(ql_b) =
                        *reinterpret_cast<const int2*>(&rowQl[row][gcol]);
                    *reinterpret_cast<int2*>(qh_b) =
                        *reinterpret_cast<const int2*>(&rowQh[row][gcol]);

                    const float d_f = rowD[row];
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int l = gcol + q;                     // 0..31
                        const int sb = sb_base + (l >> 4);          // (l<16)?sb_base:sb_base+1
                        const int raw4  = quad_low ? (ql_b[q] & 0x0F)
                                                   : (ql_b[q] >> 4);
                        const int high2 = (qh_b[q] >> (quad * 2)) & 0x03;
                        const int val6  = static_cast<int>(raw4 | (high2 << 4)) - 32;
                        const int8_t scv = rowScales[row][sb];
                        tmp_h[q] = __float2half_rn(
                            d_f * static_cast<float>(scv) * static_cast<float>(val6));
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

__global__ void q6k_wmma_read_counter_kernel(unsigned long long* out, int reset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = g_q6k_wmma_call_counter;
        if (reset) g_q6k_wmma_call_counter = 0;
    }
}

bool launch_linear_q6k_fp16_wmma(const __half* X, const uint8_t* W_q6k,
                                 const __half* bias, __half* Y,
                                 int B, int M, int K, cudaStream_t stream) {
    if (B <= 0 || M <= 0 || K <= 0) return false;
    if (K % q6k_ns::kBlockElems != 0) return false;
    if (B < 4) return false;

    dim3 block(THREADS_PER_CTA);
    dim3 grid((M + BN - 1) / BN, (B + BM - 1) / BM);
    linear_q6k_fp16_wmma_kernel<<<grid, block, 0, stream>>>(
        X, W_q6k, bias, Y, B, M, K);
    return true;
}

}  // namespace brotensor::detail::cuda::q6k_wmma_internal

extern "C" unsigned long long brotensor_q6k_wmma_calls_consume() {
    using namespace brotensor::detail::cuda::q6k_wmma_internal;
    unsigned long long h_val = 0;
    unsigned long long* d_buf = nullptr;
    if (cudaMalloc(&d_buf, sizeof(unsigned long long)) != cudaSuccess) return 0;
    q6k_wmma_read_counter_kernel<<<1, 1>>>(d_buf, /*reset=*/1);
    cudaMemcpy(&h_val, d_buf, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    return h_val;
}
