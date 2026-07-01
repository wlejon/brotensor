// Public matmul: row-major C(M,N) = A(M,K) @ B(K,N), no bias.
// Dispatched on A.dtype (FP32 + FP16). FP32 accumulation throughout.
//
// The existing internal kernels (`gemm.cu`'s `linear_forward_kernel` and
// `fp16_matmul.cu`'s `launch_matmul_ABT`) compute Y = X @ W^T, which is the
// wrong layout for a plain row-major matmul: B is stored row-major as
// (K, N) but we want to read its columns as the inner dimension. We therefore
// provide direct (M,K) @ (K,N) kernels here.

#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace brotensor {

// Forward decl: thread-local current stream from runtime.cu.
void* cuda_current_stream();

// Internal WMMA tensor-core A @ B^T launchers (defined in fp16_matmul.cu).
namespace fp16_internal {
void launch_matmul_ABT_batched_impl(const __half* A, const __half* B, __half* C,
                                    int batch, int M, int N, int K,
                                    size_t strideA, size_t strideB, size_t strideC,
                                    const __half* bias, int act);
void launch_matmul_ABT_batched_impl(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                    __nv_bfloat16* C,
                                    int batch, int M, int N, int K,
                                    size_t strideA, size_t strideB, size_t strideC,
                                    const __nv_bfloat16* bias, int act);
} // namespace fp16_internal

namespace detail::cuda {

namespace {

constexpr int MM_TILE = 16;

// 2D tiled GEMM, naive but tiled. One thread per output element; cooperative
// tile loads of A and B into shared memory.
__global__ void matmul_fp32_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * MM_TILE + threadIdx.x;
        const int b_row = t * MM_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

__global__ void matmul_fp16_kernel(const __half* __restrict__ A,
                                   const __half* __restrict__ B,
                                   __half* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * MM_TILE + threadIdx.x;
        const int b_row = t * MM_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __half2float(A[row * K + a_col]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __half2float(B[b_row * N + col]) : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

__global__ void matmul_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   __nv_bfloat16* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * MM_TILE + threadIdx.x;
        const int b_row = t * MM_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __bfloat162float(A[row * K + a_col]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __bfloat162float(B[b_row * N + col]) : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

// ─── WMMA tensor-core path: C(M,N) = A(M,K) @ B(K,N), both row-major ───────
//
// fp16_matmul.cu's `matmul_ABT_wmma_kernel` computes C = A @ B^T where B is
// stored row-major as (N,K); it loads B tiles with a col_major fragment so
// the WMMA math sees B as (K,N) without physically transposing anything.
// Here B is *already* stored row-major as (K,N) — the layout WMMA's
// `matrix_b, row_major` fragment expects directly — so B tiles are staged
// into shared memory with a plain, untransposed (BK x BN) row-major copy and
// consumed with a row_major fragment. A's tiling (row-major (M,K), row_major
// fragment) and the FP32-accumulator / per-type store epilogue mirror
// fp16_matmul.cu's kernel exactly; only the B tile's shape/orientation and
// the (unused here) bias/act epilogue differ.
//
// Same CTA/warp tile shape as fp16_matmul.cu: BM=64, BN=64, BK=32, 4 warps
// (2x2) per 128-thread CTA, WM=WN=32 (2x2 16x16x16 fragments per warp).
template <typename T> struct rm_traits;
template <> struct rm_traits<__half> {
    __device__ static __half from_f32(float v) { return __float2half(v); }
    __device__ static float  to_f32(__half v)  { return __half2float(v); }
};
template <> struct rm_traits<__nv_bfloat16> {
    __device__ static __nv_bfloat16 from_f32(float v)       { return __float2bfloat16(v); }
    __device__ static float         to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
};

constexpr int RM_WMMA_M = 16;
constexpr int RM_WMMA_N = 16;
constexpr int RM_WMMA_K = 16;

constexpr int RM_BM = 64;
constexpr int RM_BN = 64;
constexpr int RM_BK = 32;
constexpr int RM_WARPS_M = 2;
constexpr int RM_WARPS_N = 2;
constexpr int RM_WARPS_PER_CTA = RM_WARPS_M * RM_WARPS_N;   // 4
constexpr int RM_THREADS_PER_CTA = RM_WARPS_PER_CTA * 32;   // 128
constexpr int RM_WM = RM_BM / RM_WARPS_M;                   // 32
constexpr int RM_WN = RM_BN / RM_WARPS_N;                   // 32
constexpr int RM_FRAGS_M = RM_WM / RM_WMMA_M;                // 2
constexpr int RM_FRAGS_N = RM_WN / RM_WMMA_N;                // 2

// Pad shared-memory leading dims to a multiple of 8 (required by
// wmma::load_matrix_sync's `ldm` for row_major/col_major fragments) while
// reducing bank conflicts — same trick as fp16_matmul.cu.
constexpr int RM_LDA_SMEM = RM_BK + 8;   // 40, for A tile (BM rows, BK cols)
constexpr int RM_LDB_SMEM = RM_BN + 8;   // 72, for B tile (BK rows, BN cols)

template <typename T>
__launch_bounds__(RM_THREADS_PER_CTA)
__global__ void matmul_rm_wmma_kernel(const T* __restrict__ A,
                                      const T* __restrict__ B,
                                      T* __restrict__ C,
                                      int M, int N, int K) {
    using namespace nvcuda;
    using TR = rm_traits<T>;
    __shared__ T As[RM_BM][RM_LDA_SMEM];
    __shared__ T Bs[RM_BK][RM_LDB_SMEM];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / RM_WARPS_N;
    const int warp_n  = warp_id % RM_WARPS_N;

    const int block_m = blockIdx.y * RM_BM;
    const int block_n = blockIdx.x * RM_BN;

    wmma::fragment<wmma::accumulator, RM_WMMA_M, RM_WMMA_N, RM_WMMA_K, float> c_frag[RM_FRAGS_M][RM_FRAGS_N];
    #pragma unroll
    for (int i = 0; i < RM_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < RM_FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += RM_BK) {
        // ---- Load A tile (RM_BM x RM_BK): natural row-major (M,K) access ----
        {
            constexpr int kElemsPerLoad = 8;  // int4 = 8 x 16-bit
            constexpr int kTotalElems   = RM_BM * RM_BK;
            constexpr int kLoadsTotal   = kTotalElems / kElemsPerLoad;
            constexpr int kLoadsPerThr  = kLoadsTotal / RM_THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * RM_THREADS_PER_CTA;
                const int row = lin / (RM_BK / kElemsPerLoad);
                const int col_grp = lin % (RM_BK / kElemsPerLoad);
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
                        const int gk_q = gk + q;
                        tmp[q] = (grow < M && gk_q < K) ? A[grow * K + gk_q] : TR::from_f32(0.0f);
                    }
                }
                *reinterpret_cast<int4*>(&As[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        // ---- Load B tile (RM_BK x RM_BN): natural row-major (K,N) access,
        // no transpose — this is the key difference from the ABT kernel. ----
        {
            constexpr int kElemsPerLoad = 8;
            constexpr int kTotalElems   = RM_BK * RM_BN;
            constexpr int kLoadsTotal   = kTotalElems / kElemsPerLoad;
            constexpr int kLoadsPerThr  = kLoadsTotal / RM_THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * RM_THREADS_PER_CTA;
                const int row = lin / (RM_BN / kElemsPerLoad);   // K-tile row
                const int col_grp = lin % (RM_BN / kElemsPerLoad);
                const int gcol = col_grp * kElemsPerLoad;        // N-tile col
                const int gk   = k0 + row;
                const int gn   = block_n + gcol;

                T tmp[kElemsPerLoad];
                if (gk < K && gn + kElemsPerLoad <= N) {
                    const int4* src = reinterpret_cast<const int4*>(&B[size_t(gk) * N + gn]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kElemsPerLoad; ++q) {
                        const int gn_q = gn + q;
                        tmp[q] = (gk < K && gn_q < N) ? B[size_t(gk) * N + gn_q] : TR::from_f32(0.0f);
                    }
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        __syncthreads();

        // ---- Compute on shared mem tiles ----
        // A frag: row_major from As, leading dim RM_LDA_SMEM, sub-tile at
        //         (warp_m*WM+i*WMMA_M, kk).
        // B frag: row_major from Bs (which is BK-rows by BN-cols row-major,
        //         exactly the (K,N) shape wmma::matrix_b/row_major expects),
        //         leading dim RM_LDB_SMEM, sub-tile at (kk, warp_n*WN+j*WMMA_N).
        #pragma unroll
        for (int kk = 0; kk < RM_BK; kk += RM_WMMA_K) {
            wmma::fragment<wmma::matrix_a, RM_WMMA_M, RM_WMMA_N, RM_WMMA_K, T, wmma::row_major> a_frag[RM_FRAGS_M];
            wmma::fragment<wmma::matrix_b, RM_WMMA_M, RM_WMMA_N, RM_WMMA_K, T, wmma::row_major> b_frag[RM_FRAGS_N];

            #pragma unroll
            for (int i = 0; i < RM_FRAGS_M; ++i) {
                const T* a_ptr = &As[warp_m * RM_WM + i * RM_WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, RM_LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < RM_FRAGS_N; ++j) {
                const T* b_ptr = &Bs[kk][warp_n * RM_WN + j * RM_WMMA_N];
                wmma::load_matrix_sync(b_frag[j], b_ptr, RM_LDB_SMEM);
            }

            #pragma unroll
            for (int i = 0; i < RM_FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < RM_FRAGS_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store C tile ----
    if constexpr (std::is_same<T, __half>::value) {
        __shared__ __half Cs[RM_BM][RM_BN + 8];

        #pragma unroll
        for (int i = 0; i < RM_FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < RM_FRAGS_N; ++j) {
                wmma::fragment<wmma::accumulator, RM_WMMA_M, RM_WMMA_N, RM_WMMA_K, __half> c_h;
                #pragma unroll
                for (int e = 0; e < c_frag[i][j].num_elements; ++e) {
                    c_h.x[e] = __float2half(c_frag[i][j].x[e]);
                }
                __half* c_ptr = &Cs[warp_m * RM_WM + i * RM_WMMA_M][warp_n * RM_WN + j * RM_WMMA_N];
                wmma::store_matrix_sync(c_ptr, c_h, RM_BN + 8, wmma::mem_row_major);
            }
        }

        __syncthreads();

        constexpr int kHalvesPerStore = 8;
        constexpr int kTotal          = RM_BM * RM_BN;
        constexpr int kStoresTotal    = kTotal / kHalvesPerStore;
        constexpr int kStoresPerThr   = kStoresTotal / RM_THREADS_PER_CTA;

        #pragma unroll
        for (int si = 0; si < kStoresPerThr; ++si) {
            const int lin = tid + si * RM_THREADS_PER_CTA;
            const int row = lin / (RM_BN / kHalvesPerStore);
            const int col_grp = lin % (RM_BN / kHalvesPerStore);
            const int gcol = col_grp * kHalvesPerStore;
            const int grow = block_m + row;
            const int gn   = block_n + gcol;

            if (grow >= M) continue;
            if (gn + kHalvesPerStore <= N) {
                int4 v = *reinterpret_cast<const int4*>(&Cs[row][gcol]);
                *reinterpret_cast<int4*>(&C[size_t(grow) * N + gn]) = v;
            } else {
                #pragma unroll
                for (int q = 0; q < kHalvesPerStore; ++q) {
                    const int gn_q = gn + q;
                    if (gn_q < N) {
                        C[size_t(grow) * N + gn_q] = Cs[row][gcol + q];
                    }
                }
            }
        }
    } else {
        // BF16: WMMA has no BF16 accumulator fragment, so stage the FP32
        // accumulator through a float tile and narrow to BF16 in the scatter
        // (same approach as fp16_matmul.cu's BF16 path).
        __shared__ float Cs[RM_BM][RM_BN + 8];

        #pragma unroll
        for (int i = 0; i < RM_FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < RM_FRAGS_N; ++j) {
                float* c_ptr = &Cs[warp_m * RM_WM + i * RM_WMMA_M][warp_n * RM_WN + j * RM_WMMA_N];
                wmma::store_matrix_sync(c_ptr, c_frag[i][j], RM_BN + 8, wmma::mem_row_major);
            }
        }

        __syncthreads();

        constexpr int kTotal       = RM_BM * RM_BN;
        constexpr int kElemsPerThr = kTotal / RM_THREADS_PER_CTA;

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * RM_THREADS_PER_CTA;
            const int row = lin / RM_BN;
            const int col = lin - row * RM_BN;
            const int grow = block_m + row;
            const int gn   = block_n + col;
            if (grow >= M || gn >= N) continue;
            C[size_t(grow) * N + gn] = TR::from_f32(Cs[row][col]);
        }
    }
}

// Gating: only take the WMMA path when the vectorised int4 tile loads (A
// along K, B/C along N) are alignment-safe and the problem is large enough
// to be worth tensor cores. Mirrors fp16_matmul.cu's `launch_abt` gating
// (same reasoning: K/N not a multiple of 8, K below one WMMA_K tile, or a
// tiny M*N all fall back to the existing scalar kernel rather than risk an
// unhandled edge case in the tensor-core path).
static bool matmul_rm_wmma_eligible(int M, int N, int K) {
    return size_t(M) * size_t(N) >= 256 && K >= 16 &&
           (K & 7) == 0 && (N & 7) == 0;
}

} // namespace

void matmul(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
            ::brotensor::Tensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("matmul: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) {
        C.resize(M, N, A.dtype);
    }
    if (M == 0 || N == 0) return;
    if (K == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(C.data, 0, C.bytes(),
            reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream())));
        return;
    }

    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());

    if (A.dtype == Dtype::FP16 || A.dtype == Dtype::BF16) {
        if (matmul_rm_wmma_eligible(M, N, K)) {
            dim3 block(RM_THREADS_PER_CTA);
            dim3 grid((N + RM_BN - 1) / RM_BN, (M + RM_BM - 1) / RM_BM);
            if (A.dtype == Dtype::FP16) {
                matmul_rm_wmma_kernel<__half><<<grid, block, 0, stream>>>(
                    static_cast<const __half*>(A.data),
                    static_cast<const __half*>(B.data),
                    static_cast<__half*>(C.data),
                    M, N, K);
            } else {
                matmul_rm_wmma_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(A.data),
                    static_cast<const __nv_bfloat16*>(B.data),
                    static_cast<__nv_bfloat16*>(C.data),
                    M, N, K);
            }
        } else {
            dim3 block(MM_TILE, MM_TILE);
            dim3 grid((N + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
            if (A.dtype == Dtype::FP16) {
                matmul_fp16_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const __half*>(A.data),
                    static_cast<const __half*>(B.data),
                    static_cast<__half*>(C.data),
                    M, N, K);
            } else {
                matmul_bf16_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(A.data),
                    static_cast<const __nv_bfloat16*>(B.data),
                    static_cast<__nv_bfloat16*>(C.data),
                    M, N, K);
            }
        }
    } else {
        dim3 block(MM_TILE, MM_TILE);
        dim3 grid((N + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
        matmul_fp32_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(A.data),
            static_cast<const float*>(B.data),
            static_cast<float*>(C.data),
            M, N, K);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// Public batched A @ B^T exposing the internal WMMA tensor-core kernel.
// C[b](M,N) = A[b](M,K) @ B[b](N,K)^T, FP32 accumulation, optional fused
// per-N bias + activation. A/B/C share dtype ∈ {FP16, BF16}; C is caller-sized.
void matmul_abt(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                ::brotensor::Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const ::brotensor::Tensor* bias, int act) {
    if (A.dtype != B.dtype || A.dtype != C.dtype) {
        throw std::runtime_error("matmul_abt: A, B, C must share dtype");
    }
    if (A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_abt: dtype must be FP16 or BF16");
    }
    if (bias && bias->dtype != A.dtype) {
        throw std::runtime_error("matmul_abt: bias dtype must match operands");
    }
    if (batch <= 0 || M == 0 || N == 0) return;

    if (A.dtype == Dtype::FP16) {
        fp16_internal::launch_matmul_ABT_batched_impl(
            static_cast<const __half*>(A.data),
            static_cast<const __half*>(B.data),
            static_cast<__half*>(C.data),
            batch, M, N, K,
            static_cast<size_t>(strideA), static_cast<size_t>(strideB),
            static_cast<size_t>(strideC),
            bias ? static_cast<const __half*>(bias->data) : nullptr, act);
    } else {
        fp16_internal::launch_matmul_ABT_batched_impl(
            static_cast<const __nv_bfloat16*>(A.data),
            static_cast<const __nv_bfloat16*>(B.data),
            static_cast<__nv_bfloat16*>(C.data),
            batch, M, N, K,
            static_cast<size_t>(strideA), static_cast<size_t>(strideB),
            static_cast<size_t>(strideC),
            bias ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr, act);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor

// ─── Vtable contribution ────────────────────────────────────────────────────
//
// Filled into the CUDA OpsVTable from the per-cluster registration entry in
// src/cuda/register.cu. Includes every op in this cluster (matmul +
// utilities) as well as the slots whose implementations live in sibling files
// (linear_forward_batched, linear_backward_batched) — those share the
// `detail::cuda` namespace.

namespace brotensor::detail::cuda {

// Forward decls — implementations in sibling .cu files within this namespace.
void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB);
void linear_forward_batched(const ::brotensor::Tensor& W,
                            const ::brotensor::Tensor& bias,
                            const ::brotensor::Tensor& X_BD,
                            ::brotensor::Tensor& Y_BD);
void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB);
void linear_forward_batched_fp16_act(const ::brotensor::Tensor& W,
                                     const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD,
                                     int act,
                                     ::brotensor::Tensor& Y_BD);
void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD);
void linear_forward_batched_int8w_fp16(const ::brotensor::Tensor& W_int8,
                                       const ::brotensor::Tensor& scales,
                                       const ::brotensor::Tensor* bias,
                                       const ::brotensor::Tensor& X_BD,
                                       ::brotensor::Tensor& Y_BD);
void matmul_int8w_fp16(const ::brotensor::Tensor& W_int8,
                       const ::brotensor::Tensor& scales,
                       const ::brotensor::Tensor& X,
                       ::brotensor::Tensor& Y);
void matmul_backward(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                     const ::brotensor::Tensor& dC,
                     ::brotensor::Tensor& dA, ::brotensor::Tensor& dB);
void rope_forward(const ::brotensor::Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, ::brotensor::Tensor& Y);
void rope_backward(const ::brotensor::Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, ::brotensor::Tensor& dX);
void rope_apply(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                ::brotensor::Tensor& Y);
void rope_apply_perhead(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                        const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                        ::brotensor::Tensor& Y);
void rope_apply_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         int head_dim, int num_heads, ::brotensor::Tensor& dX);
void softmax_forward(const ::brotensor::Tensor& logits,
                     ::brotensor::Tensor& probs, const float* mask);
void softmax_rows_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                          int rows, int cols);
void softmax_backward(const ::brotensor::Tensor& probs,
                      const ::brotensor::Tensor& dProbs,
                      ::brotensor::Tensor& dLogits);
void sgd_step(::brotensor::Tensor& param, ::brotensor::Tensor& grad,
              ::brotensor::Tensor& velocity, float lr, float momentum);
void adam_step(::brotensor::Tensor& param, const ::brotensor::Tensor& grad,
               ::brotensor::Tensor& m, ::brotensor::Tensor& v,
               float lr, float beta1, float beta2, float eps, int step);
void sum_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void sum_cols(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y);
void argmax_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx);
void rows_count_above(const ::brotensor::Tensor& X, float t_lo, float t_hi,
                      ::brotensor::Tensor& counts);
void ddim_step(const ::brotensor::Tensor& x_t,
               const ::brotensor::Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               ::brotensor::Tensor& x_prev);
void euler_step(const ::brotensor::Tensor& x_t,
                const ::brotensor::Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                ::brotensor::Tensor& x_prev);
void dpmpp_2m_step(const ::brotensor::Tensor& x_t,
                   const ::brotensor::Tensor& eps_pred,
                   const ::brotensor::Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   ::brotensor::Tensor& x_prev, ::brotensor::Tensor& x0_out);
void timestep_embedding(const ::brotensor::Tensor& timesteps,
                        int dim, float max_period, ::brotensor::Tensor& Y);
// Defined in transpose.cu.
void nchw_to_sequence(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);
void sequence_to_nchw(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W, ::brotensor::Tensor& Y);

void fill_cuda_vtable_utils(::brotensor::detail::OpsVTable& v) {
    v.matmul                              = &matmul;
    v.matmul_abt                          = &matmul_abt;
    v.matmul_backward                     = &matmul_backward;
    v.linear_forward                      = &linear_forward;
    v.linear_backward                     = &linear_backward;
    v.linear_forward_batched              = &linear_forward_batched;
    v.linear_backward_batched             = &linear_backward_batched;
    v.linear_forward_batched_fp16         = &linear_forward_batched_fp16;
    v.linear_forward_batched_fp16_act     = &linear_forward_batched_fp16_act;
    v.linear_forward_batched_int8w_fp16   = &linear_forward_batched_int8w_fp16;
    v.matmul_int8w_fp16                   = &matmul_int8w_fp16;
    v.rope_forward                        = &rope_forward;
    v.rope_backward                       = &rope_backward;
    v.rope_apply                          = &rope_apply;
    v.rope_apply_perhead                  = &rope_apply_perhead;
    v.rope_apply_backward                 = &rope_apply_backward;
    v.softmax_forward                     = &softmax_forward;
    v.softmax_rows_forward                = &softmax_rows_forward;
    v.softmax_backward                    = &softmax_backward;
    v.sgd_step                            = &sgd_step;
    v.adam_step                           = &adam_step;
    v.sum_rows                            = &sum_rows;
    v.sum_cols                            = &sum_cols;
    v.argmax_rows                         = &argmax_rows;
    v.rows_count_above                    = &rows_count_above;
    v.ddim_step                           = &ddim_step;
    v.euler_step                          = &euler_step;
    v.dpmpp_2m_step                       = &dpmpp_2m_step;
    v.timestep_embedding                  = &timestep_embedding;
    v.nchw_to_sequence                    = &nchw_to_sequence;
    v.sequence_to_nchw                    = &sequence_to_nchw;
}

} // namespace brotensor::detail::cuda
