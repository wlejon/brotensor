// WMMA tensor-core implementation of:
//     C(M, N) = A(M, K) @ B(N, K)^T,  FP16 storage, FP32 accumulator
//
// Memory layout: A is row-major (M, K). B is row-major (N, K). C is row-major (M, N).
// For wmma::load_matrix_sync, A is loaded as row_major (M x K), and B is loaded as
// col_major. With B stored row-major over (N, K), a col_major load of leading dim K
// treats the same buffer as a (K x N) matrix — exactly what wmma needs for A @ B'.
//
// CTA tile: BM x BN output, accumulated in chunks of BK across K.
// Warp tile: WM x WN, built from 16x16x16 wmma fragments.
//   BM=64, BN=64, BK=32, 4 warps per CTA (128 threads), WM=WN=32 (2x2 fragments / warp).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace brotensor {
namespace fp16_internal {

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 2;
static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;  // 4
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;  // 128
static constexpr int WM = BM / WARPS_M;  // 32
static constexpr int WN = BN / WARPS_N;  // 32
static constexpr int FRAGS_M = WM / WMMA_M;  // 2
static constexpr int FRAGS_N = WN / WMMA_N;  // 2

// Pad shared-memory leading dim to reduce bank conflicts.
static constexpr int LDA_SMEM = BK + 8;   // for A tile (BM rows, BK cols)
static constexpr int LDB_SMEM = BK + 8;   // for B tile (BN rows, BK cols)

__launch_bounds__(THREADS_PER_CTA)
__global__ void matmul_ABT_wmma_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       __half* __restrict__ C,
                                       int M, int N, int K) {
    __shared__ __half As[BM][LDA_SMEM];
    __shared__ __half Bs[BN][LDB_SMEM];

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int warp_m    = warp_id / WARPS_N;
    const int warp_n    = warp_id % WARPS_N;

    const int block_m   = blockIdx.y * BM;
    const int block_n   = blockIdx.x * BN;

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
    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile (BM x BK) ----
        // Vectorized: each thread loads 8 halves (one int4 = 16 bytes) per iter.
        // Total bytes: BM*BK*2 = 64*32*2 = 4096. Threads=128. Per thread: 32 B = 2 int4.
        // We'll load with strided pattern.
        {
            constexpr int kHalvesPerLoad = 8;  // int4 = 8 __half
            constexpr int kTotalHalves   = BM * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;  // 256
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;  // 2

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;  // 0..kLoadsTotal-1
                const int row = lin / (BK / kHalvesPerLoad); // BK/8 = 4 loads per row
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_m + row;
                const int gk   = k0 + gcol;

                __half tmp[kHalvesPerLoad];
                if (grow < M && gk + kHalvesPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&A[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        int gk_q = gk + q;
                        if (grow < M && gk_q < K) {
                            tmp[q] = A[grow * K + gk_q];
                        } else {
                            tmp[q] = __float2half(0.0f);
                        }
                    }
                }
                *reinterpret_cast<int4*>(&As[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        // ---- Load B tile (BN x BK) ----
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
                const int grow = block_n + row;
                const int gk   = k0 + gcol;

                __half tmp[kHalvesPerLoad];
                if (grow < N && gk + kHalvesPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&B[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        int gk_q = gk + q;
                        if (grow < N && gk_q < K) {
                            tmp[q] = B[grow * K + gk_q];
                        } else {
                            tmp[q] = __float2half(0.0f);
                        }
                    }
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        __syncthreads();

        // ---- Compute on shared mem tiles ----
        // A frag: row_major from As, leading dim LDA_SMEM, sub-tile starts at (warp_m*WM, kk).
        // B frag: col_major from Bs (which is BN-rows by BK-cols row-major) -> viewed as
        //        (BK x BN) col_major with leading dim LDB_SMEM, sub-tile at (kk, warp_n*WN).
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
                // col_major view of Bs: element (kk_row, n_col) lives at Bs[n_col][kk_row].
                // Sub-tile origin (row=kk in K, col=warp_n*WN+j*WMMA_N in N) =>
                // pointer = &Bs[warp_n*WN + j*WMMA_N][kk]; leading dim = LDB_SMEM.
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

    // ---- Store C tile ----
    // Store FP32 accumulator into shared mem (as __half) then write out, OR convert
    // directly. Simpler: stage in shared. But we have 4096 halves of As we can reuse.
    // Use the As buffer (BM x LDA_SMEM) — sufficient since BM*BN halves = 8192 vs
    // As size 64*40=2560. Need bigger; let's just write fragment to shared as fp16.
    __shared__ __half Cs[BM][BN + 8];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            // Convert FP32 frag -> FP16 via a temp FP16 frag.
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

    // Write Cs -> C global.
    // BM*BN = 4096 halves, 128 threads, 32 per thread (4 int4).
    {
        constexpr int kHalvesPerStore = 8;
        constexpr int kTotal          = BM * BN;
        constexpr int kStoresTotal    = kTotal / kHalvesPerStore;  // 512
        constexpr int kStoresPerThr   = kStoresTotal / THREADS_PER_CTA;  // 4

        #pragma unroll
        for (int si = 0; si < kStoresPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / (BN / kHalvesPerStore);  // BN/8 = 8 per row
            const int col_grp = lin % (BN / kHalvesPerStore);
            const int gcol = col_grp * kHalvesPerStore;
            const int grow = block_m + row;
            const int gn   = block_n + gcol;

            if (grow >= M) continue;
            if (gn + kHalvesPerStore <= N) {
                int4 v = *reinterpret_cast<const int4*>(&Cs[row][gcol]);
                *reinterpret_cast<int4*>(&C[grow * N + gn]) = v;
            } else {
                #pragma unroll
                for (int q = 0; q < kHalvesPerStore; ++q) {
                    int gn_q = gn + q;
                    if (gn_q < N) {
                        C[grow * N + gn_q] = Cs[row][gcol + q];
                    }
                }
            }
        }
    }
}

// Naive fallback for tiny problems.
__global__ void matmul_ABT_naive_kernel(const __half* __restrict__ A,
                                        const __half* __restrict__ B,
                                        __half* __restrict__ C,
                                        int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;
    const int m = idx / N;
    const int n = idx % N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
    }
    C[idx] = __float2half(acc);
}

void launch_matmul_ABT_impl(const __half* A, const __half* B, __half* C,
                            int M, int N, int K) {
    if (M == 0 || N == 0) return;
    if (K == 0) {
        cudaMemsetAsync(C, 0, sizeof(__half) * size_t(M) * size_t(N));
        return;
    }

    // Use naive for very tiny problems.
    if (size_t(M) * size_t(N) < 256) {
        const int total = M * N;
        const int block = 128;
        const int grid  = (total + block - 1) / block;
        matmul_ABT_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    dim3 block(THREADS_PER_CTA);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_ABT_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

} // namespace fp16_internal
} // namespace brotensor
