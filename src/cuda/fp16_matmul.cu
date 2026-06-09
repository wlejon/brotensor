// WMMA tensor-core implementation of:
//     C(M, N) = A(M, K) @ B(N, K)^T,  FP16/BF16 storage, FP32 accumulator
//
// Memory layout: A is row-major (M, K). B is row-major (N, K). C is row-major (M, N).
// For wmma::load_matrix_sync, A is loaded as row_major (M x K), and B is loaded as
// col_major. With B stored row-major over (N, K), a col_major load of leading dim K
// treats the same buffer as a (K x N) matrix — exactly what wmma needs for A @ B'.
//
// CTA tile: BM x BN output, accumulated in chunks of BK across K.
// Warp tile: WM x WN, built from 16x16x16 wmma fragments.
//   BM=64, BN=64, BK=32, 4 warps per CTA (128 threads), WM=WN=32 (2x2 fragments / warp).
//
// One template over both __half and __nv_bfloat16 (RTX 4090 / sm_89 supports BF16
// WMMA fragments; load_matrix_sync / mma_sync / store_matrix_sync are identical —
// only the fragment element type and the host-side conversions differ). The FP16
// path keeps its original __half-accumulator + vectorised int4 store epilogue
// untouched; the BF16 path stages the FP32 accumulator through a float tile (WMMA
// has no BF16 accumulator fragment) and narrows in the scatter.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <type_traits>

#include "detail/activations.cuh"

namespace brotensor {
void* cuda_current_stream();      // shim defined in runtime.cu
namespace fp16_internal {

using namespace nvcuda;

// Element-type traits: host-side conversions differ between the two 16-bit
// storage types; the WMMA fragment ops are identical.
template <typename T> struct abt_traits;
template <> struct abt_traits<__half> {
    __device__ static __half from_f32(float v) { return __float2half(v); }
    __device__ static float   to_f32(__half v) { return __half2float(v); }
};
template <> struct abt_traits<__nv_bfloat16> {
    __device__ static __nv_bfloat16 from_f32(float v)       { return __float2bfloat16(v); }
    __device__ static float         to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
};

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

// Optional epilogue (bias + activation) is fused into the global store stage:
// `bias` (length N, broadcast over rows) is added and `act` applied in-register
// before writing C, so a linear-forward needs neither a separate bias-add nor a
// separate activation kernel. For the FP16 path, bias == nullptr && act == 0
// keeps the fast int4 store path untouched — the matmul / attention callers that
// pass no epilogue are unaffected.
template <typename T>
__launch_bounds__(THREADS_PER_CTA)
__global__ void matmul_ABT_wmma_kernel(const T* __restrict__ A,
                                       const T* __restrict__ B,
                                       T* __restrict__ C,
                                       int M, int N, int K,
                                       const T* __restrict__ bias, int act) {
    using TR = abt_traits<T>;
    __shared__ T As[BM][LDA_SMEM];
    __shared__ T Bs[BN][LDB_SMEM];

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
        // Vectorized: each thread loads 8 elements (one int4 = 16 bytes) per iter.
        // Both __half and __nv_bfloat16 are 16-bit, so 8 pack into one int4.
        {
            constexpr int kElemsPerLoad = 8;  // int4 = 8 x 16-bit
            constexpr int kTotalElems    = BM * BK;
            constexpr int kLoadsTotal    = kTotalElems / kElemsPerLoad;  // 256
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;  // 2

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;  // 0..kLoadsTotal-1
                const int row = lin / (BK / kElemsPerLoad); // BK/8 = 4 loads per row
                const int col_grp = lin % (BK / kElemsPerLoad);
                const int gcol = col_grp * kElemsPerLoad;
                const int grow = block_m + row;
                const int gk   = k0 + gcol;

                T tmp[kElemsPerLoad];
                if (grow < M && gk + kElemsPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&A[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kElemsPerLoad; ++q) {
                        int gk_q = gk + q;
                        if (grow < M && gk_q < K) {
                            tmp[q] = A[grow * K + gk_q];
                        } else {
                            tmp[q] = TR::from_f32(0.0f);
                        }
                    }
                }
                *reinterpret_cast<int4*>(&As[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        // ---- Load B tile (BN x BK) ----
        {
            constexpr int kElemsPerLoad = 8;
            constexpr int kTotalElems    = BN * BK;
            constexpr int kLoadsTotal    = kTotalElems / kElemsPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kElemsPerLoad);
                const int col_grp = lin % (BK / kElemsPerLoad);
                const int gcol = col_grp * kElemsPerLoad;
                const int grow = block_n + row;
                const int gk   = k0 + gcol;

                T tmp[kElemsPerLoad];
                if (grow < N && gk + kElemsPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&B[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kElemsPerLoad; ++q) {
                        int gk_q = gk + q;
                        if (grow < N && gk_q < K) {
                            tmp[q] = B[grow * K + gk_q];
                        } else {
                            tmp[q] = TR::from_f32(0.0f);
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
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> b_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const T* a_ptr = &As[warp_m * WM + i * WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                // col_major view of Bs: element (kk_row, n_col) lives at Bs[n_col][kk_row].
                // Sub-tile origin (row=kk in K, col=warp_n*WN+j*WMMA_N in N) =>
                // pointer = &Bs[warp_n*WN + j*WMMA_N][kk]; leading dim = LDB_SMEM.
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

    // ---- Store C tile ----
    if constexpr (std::is_same<T, __half>::value) {
        // FP16 path: stage the accumulator as __half and write out, reusing the
        // vectorised int4 store for the no-epilogue case (unchanged from the
        // original FP16-only kernel).
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
            if (bias == nullptr && act == 0 && gn + kHalvesPerStore <= N) {
                int4 v = *reinterpret_cast<const int4*>(&Cs[row][gcol]);
                *reinterpret_cast<int4*>(&C[grow * N + gn]) = v;
            } else {
                #pragma unroll
                for (int q = 0; q < kHalvesPerStore; ++q) {
                    int gn_q = gn + q;
                    if (gn_q < N) {
                        float cv = __half2float(Cs[row][gcol + q]);
                        if (bias) cv += __half2float(bias[gn_q]);
                        cv = ::brotensor::detail::cuda::apply_linear_act(act, cv);
                        C[grow * N + gn_q] = __float2half(cv);
                    }
                }
            }
        }
    } else {
        // BF16 path: WMMA has no BF16 accumulator fragment, so stage the FP32
        // accumulator through a float tile (numerically exact) and narrow to
        // BF16 in the scatter, applying the bias + activation epilogue.
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

        // BM*BN = 4096 cells, 128 threads, 32 per thread. The float->BF16 narrowing
        // is per element, so there is no vectorised int4 fast path here.
        constexpr int kTotal       = BM * BN;
        constexpr int kElemsPerThr = kTotal / THREADS_PER_CTA;  // 32

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / BN;
            const int col = lin - row * BN;
            const int grow = block_m + row;
            const int gn   = block_n + col;
            if (grow >= M || gn >= N) continue;
            float cv = Cs[row][col];
            if (bias) cv += TR::to_f32(bias[gn]);
            cv = ::brotensor::detail::cuda::apply_linear_act(act, cv);
            C[grow * N + gn] = TR::from_f32(cv);
        }
    }
}

// Naive fallback for tiny / unaligned problems.
template <typename T>
__global__ void matmul_ABT_naive_kernel(const T* __restrict__ A,
                                        const T* __restrict__ B,
                                        T* __restrict__ C,
                                        int M, int N, int K,
                                        const T* __restrict__ bias, int act) {
    using TR = abt_traits<T>;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;
    const int m = idx / N;
    const int n = idx % N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += TR::to_f32(A[m * K + k]) * TR::to_f32(B[n * K + k]);
    }
    if (bias) acc += TR::to_f32(bias[n]);
    acc = ::brotensor::detail::cuda::apply_linear_act(act, acc);
    C[idx] = TR::from_f32(acc);
}

template <typename T>
static void launch_abt(const T* A, const T* B, T* C,
                       int M, int N, int K,
                       const T* bias, int act) {
    if (M == 0 || N == 0) return;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (K == 0) {
        cudaMemsetAsync(C, 0, sizeof(T) * size_t(M) * size_t(N), stream);
        return;
    }

    // Use naive for very tiny problems, when K is too small for the WMMA
    // kernel's tile shape (BK=32, WMMA_K=16), or when K/N is not a multiple
    // of 8: the WMMA path's vectorised int4 loads of A/B (and the FP16 int4
    // store of C) require an 8-element-aligned row stride along K (for A and B)
    // and N (for C); the kernel does not currently emit a scalar-aligned-store
    // fallback for those cases.
    if (size_t(M) * size_t(N) < 256 || K < 16 ||
        (K & 7) != 0 || (N & 7) != 0) {
        const int total = M * N;
        const int block = 128;
        const int grid  = (total + block - 1) / block;
        matmul_ABT_naive_kernel<T><<<grid, block, 0, stream>>>(A, B, C, M, N, K, bias, act);
        return;
    }

    dim3 block(THREADS_PER_CTA);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_ABT_wmma_kernel<T><<<grid, block, 0, stream>>>(A, B, C, M, N, K, bias, act);
}

void launch_matmul_ABT_impl(const __half* A, const __half* B, __half* C,
                            int M, int N, int K,
                            const __half* bias, int act) {
    launch_abt<__half>(A, B, C, M, N, K, bias, act);
}

void launch_matmul_ABT_impl(const __nv_bfloat16* A, const __nv_bfloat16* B,
                            __nv_bfloat16* C,
                            int M, int N, int K,
                            const __nv_bfloat16* bias, int act) {
    launch_abt<__nv_bfloat16>(A, B, C, M, N, K, bias, act);
}

} // namespace fp16_internal
} // namespace brotensor
