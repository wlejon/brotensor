// Backward of matmul. Row-major. No bias.
//   forward: C(M, N) = A(M, K) @ B(K, N)
//   dA(M, K) += dC(M, N) @ B^T(N, K)
//   dB(K, N) += A^T(K, M) @ dC(M, N)
//
// FP32 and FP16 are both supported. FP32 path atomic-adds FP32 partial
// products directly into the caller's dA / dB buffers. FP16 path follows the
// FP32-scratch-fold pattern (rms_norm_backward / layernorm_backward / conv2d
// weight backward): allocate FP32 scratch shaped like dA / dB, atomicAdd FP32
// partial products into the scratch, then a fold kernel rounds each FP32
// accumulator into the corresponding FP16 slot via `acc + cur` → __float2half.
// FP16 atomic-adds are unsafe across blocks at small sub-normals and would
// silently swallow gradient.

#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr int MMB_TILE = 16;

// ───────────── FP32 path: tiled GEMMs with atomicAdd accumulation ──────────

// dA[m, k] += sum_n dC[m, n] * B[k, n]
__global__ void mmb_dA_fp32_kernel(const float* __restrict__ dC,
                                   const float* __restrict__ B,
                                   float* __restrict__ dA,
                                   int M, int N, int K) {
    __shared__ float dCs[MMB_TILE][MMB_TILE];
    __shared__ float Bts[MMB_TILE][MMB_TILE];

    const int row = blockIdx.y * MMB_TILE + threadIdx.y;  // m
    const int col = blockIdx.x * MMB_TILE + threadIdx.x;  // k

    float acc = 0.0f;
    const int n_tiles = (N + MMB_TILE - 1) / MMB_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int dc_col = t * MMB_TILE + threadIdx.x;       // n
        const int bt_row = t * MMB_TILE + threadIdx.y;       // n
        dCs[threadIdx.y][threadIdx.x] =
            (row < M && dc_col < N) ? dC[row * N + dc_col] : 0.0f;
        // B^T[n, k] = B[k, n]; bt_row=n, bt_col(==threadIdx.x)→k slot.
        Bts[threadIdx.y][threadIdx.x] =
            (col < K && bt_row < N) ? B[col * N + bt_row] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < MMB_TILE; ++n) {
            acc += dCs[threadIdx.y][n] * Bts[n][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        atomicAdd(&dA[row * K + col], acc);
    }
}

// dB[k, n] += sum_m A[m, k] * dC[m, n]
__global__ void mmb_dB_fp32_kernel(const float* __restrict__ A,
                                   const float* __restrict__ dC,
                                   float* __restrict__ dB,
                                   int M, int N, int K) {
    __shared__ float Ats[MMB_TILE][MMB_TILE];
    __shared__ float dCs[MMB_TILE][MMB_TILE];

    const int row = blockIdx.y * MMB_TILE + threadIdx.y;  // k
    const int col = blockIdx.x * MMB_TILE + threadIdx.x;  // n

    float acc = 0.0f;
    const int n_tiles = (M + MMB_TILE - 1) / MMB_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_row = t * MMB_TILE + threadIdx.x;        // m (paired with k=row)
        const int dc_row = t * MMB_TILE + threadIdx.y;       // m
        // A^T[k, m] = A[m, k]; reading A[a_row, row=k].
        Ats[threadIdx.y][threadIdx.x] =
            (row < K && a_row < M) ? A[a_row * K + row] : 0.0f;
        dCs[threadIdx.y][threadIdx.x] =
            (dc_row < M && col < N) ? dC[dc_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < MMB_TILE; ++m) {
            acc += Ats[threadIdx.y][m] * dCs[m][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        atomicAdd(&dB[row * N + col], acc);
    }
}

// ───────────── FP16 path: same tiled GEMMs into FP32 scratch ───────────────

__global__ void mmb_dA_fp16_kernel(const __half* __restrict__ dC,
                                   const __half* __restrict__ B,
                                   float* __restrict__ dA_scratch,
                                   int M, int N, int K) {
    __shared__ float dCs[MMB_TILE][MMB_TILE];
    __shared__ float Bts[MMB_TILE][MMB_TILE];

    const int row = blockIdx.y * MMB_TILE + threadIdx.y;  // m
    const int col = blockIdx.x * MMB_TILE + threadIdx.x;  // k

    float acc = 0.0f;
    const int n_tiles = (N + MMB_TILE - 1) / MMB_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int dc_col = t * MMB_TILE + threadIdx.x;
        const int bt_row = t * MMB_TILE + threadIdx.y;
        dCs[threadIdx.y][threadIdx.x] =
            (row < M && dc_col < N) ? __half2float(dC[row * N + dc_col]) : 0.0f;
        Bts[threadIdx.y][threadIdx.x] =
            (col < K && bt_row < N) ? __half2float(B[col * N + bt_row]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < MMB_TILE; ++n) {
            acc += dCs[threadIdx.y][n] * Bts[n][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        atomicAdd(&dA_scratch[row * K + col], acc);
    }
}

__global__ void mmb_dB_fp16_kernel(const __half* __restrict__ A,
                                   const __half* __restrict__ dC,
                                   float* __restrict__ dB_scratch,
                                   int M, int N, int K) {
    __shared__ float Ats[MMB_TILE][MMB_TILE];
    __shared__ float dCs[MMB_TILE][MMB_TILE];

    const int row = blockIdx.y * MMB_TILE + threadIdx.y;  // k
    const int col = blockIdx.x * MMB_TILE + threadIdx.x;  // n

    float acc = 0.0f;
    const int n_tiles = (M + MMB_TILE - 1) / MMB_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_row  = t * MMB_TILE + threadIdx.x;
        const int dc_row = t * MMB_TILE + threadIdx.y;
        Ats[threadIdx.y][threadIdx.x] =
            (row < K && a_row < M) ? __half2float(A[a_row * K + row]) : 0.0f;
        dCs[threadIdx.y][threadIdx.x] =
            (dc_row < M && col < N) ? __half2float(dC[dc_row * N + col]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < MMB_TILE; ++m) {
            acc += Ats[threadIdx.y][m] * dCs[m][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        atomicAdd(&dB_scratch[row * N + col], acc);
    }
}

// Fold FP32 scratch into FP16 destination: dst[i] = __float2half(scratch[i] +
// __half2float(dst[i])). Caller-zeros-and-accumulates contract: dst already
// holds the running sum from prior calls (typically zero on first call).
__global__ void mmb_fold_fp16_kernel(__half* __restrict__ dst,
                                     const float* __restrict__ scratch,
                                     int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float cur = __half2float(dst[i]);
    dst[i] = __float2half(cur + scratch[i]);
}

// ───────────── BF16 path: same tiled GEMMs into FP32 scratch ───────────────

__global__ void mmb_dA_bf16_kernel(const __nv_bfloat16* __restrict__ dC,
                                   const __nv_bfloat16* __restrict__ B,
                                   float* __restrict__ dA_scratch,
                                   int M, int N, int K) {
    __shared__ float dCs[MMB_TILE][MMB_TILE];
    __shared__ float Bts[MMB_TILE][MMB_TILE];

    const int row = blockIdx.y * MMB_TILE + threadIdx.y;  // m
    const int col = blockIdx.x * MMB_TILE + threadIdx.x;  // k

    float acc = 0.0f;
    const int n_tiles = (N + MMB_TILE - 1) / MMB_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int dc_col = t * MMB_TILE + threadIdx.x;
        const int bt_row = t * MMB_TILE + threadIdx.y;
        dCs[threadIdx.y][threadIdx.x] =
            (row < M && dc_col < N) ? __bfloat162float(dC[row * N + dc_col]) : 0.0f;
        Bts[threadIdx.y][threadIdx.x] =
            (col < K && bt_row < N) ? __bfloat162float(B[col * N + bt_row]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < MMB_TILE; ++n) {
            acc += dCs[threadIdx.y][n] * Bts[n][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        atomicAdd(&dA_scratch[row * K + col], acc);
    }
}

__global__ void mmb_dB_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ dC,
                                   float* __restrict__ dB_scratch,
                                   int M, int N, int K) {
    __shared__ float Ats[MMB_TILE][MMB_TILE];
    __shared__ float dCs[MMB_TILE][MMB_TILE];

    const int row = blockIdx.y * MMB_TILE + threadIdx.y;  // k
    const int col = blockIdx.x * MMB_TILE + threadIdx.x;  // n

    float acc = 0.0f;
    const int n_tiles = (M + MMB_TILE - 1) / MMB_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_row  = t * MMB_TILE + threadIdx.x;
        const int dc_row = t * MMB_TILE + threadIdx.y;
        Ats[threadIdx.y][threadIdx.x] =
            (row < K && a_row < M) ? __bfloat162float(A[a_row * K + row]) : 0.0f;
        dCs[threadIdx.y][threadIdx.x] =
            (dc_row < M && col < N) ? __bfloat162float(dC[dc_row * N + col]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < MMB_TILE; ++m) {
            acc += Ats[threadIdx.y][m] * dCs[m][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < K && col < N) {
        atomicAdd(&dB_scratch[row * N + col], acc);
    }
}

__global__ void mmb_fold_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                     const float* __restrict__ scratch,
                                     int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float cur = __bfloat162float(dst[i]);
    dst[i] = __float2bfloat16(cur + scratch[i]);
}

} // namespace

void matmul_backward(const ::brotensor::Tensor& A,
                     const ::brotensor::Tensor& B,
                     const ::brotensor::Tensor& dC,
                     ::brotensor::Tensor& dA,
                     ::brotensor::Tensor& dB) {
    if (A.dtype != B.dtype || A.dtype != dC.dtype ||
        A.dtype != dA.dtype || A.dtype != dB.dtype) {
        throw std::runtime_error("matmul_backward: dtype mismatch");
    }
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16 &&
        A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_backward: only FP32/FP16/BF16 supported");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_backward: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (dC.rows != M || dC.cols != N) {
        throw std::runtime_error("matmul_backward: dC shape mismatch");
    }
    if (dA.rows != M || dA.cols != K) {
        throw std::runtime_error("matmul_backward: dA must be pre-sized to (M, K)");
    }
    if (dB.rows != K || dB.cols != N) {
        throw std::runtime_error("matmul_backward: dB must be pre-sized to (K, N)");
    }
    if (M == 0 || N == 0 || K == 0) return;

    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());

    if (A.dtype == Dtype::FP32) {
        // dA(M, K) += dC(M, N) · B^T(N, K)
        {
            dim3 block(MMB_TILE, MMB_TILE);
            dim3 grid((K + MMB_TILE - 1) / MMB_TILE,
                      (M + MMB_TILE - 1) / MMB_TILE);
            mmb_dA_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(dC.data),
                static_cast<const float*>(B.data),
                static_cast<float*>(dA.data), M, N, K);
        }
        // dB(K, N) += A^T(K, M) · dC(M, N)
        {
            dim3 block(MMB_TILE, MMB_TILE);
            dim3 grid((N + MMB_TILE - 1) / MMB_TILE,
                      (K + MMB_TILE - 1) / MMB_TILE);
            mmb_dB_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(A.data),
                static_cast<const float*>(dC.data),
                static_cast<float*>(dB.data), M, N, K);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (A.dtype == Dtype::BF16) {
        // BF16 path: FP32 scratch + fold (FP16-pattern twin).
        ::brotensor::Tensor dA_scratch =
            ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, M, K, Dtype::FP32);
        ::brotensor::Tensor dB_scratch =
            ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, K, N, Dtype::FP32);

        {
            dim3 block(MMB_TILE, MMB_TILE);
            dim3 grid((K + MMB_TILE - 1) / MMB_TILE,
                      (M + MMB_TILE - 1) / MMB_TILE);
            mmb_dA_bf16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(dC.data),
                static_cast<const __nv_bfloat16*>(B.data),
                static_cast<float*>(dA_scratch.data), M, N, K);
        }
        {
            dim3 block(MMB_TILE, MMB_TILE);
            dim3 grid((N + MMB_TILE - 1) / MMB_TILE,
                      (K + MMB_TILE - 1) / MMB_TILE);
            mmb_dB_bf16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(A.data),
                static_cast<const __nv_bfloat16*>(dC.data),
                static_cast<float*>(dB_scratch.data), M, N, K);
        }
        {
            const int total = M * K;
            const int blocks = (total + 255) / 256;
            mmb_fold_bf16_kernel<<<blocks, 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(dA.data),
                static_cast<const float*>(dA_scratch.data), total);
        }
        {
            const int total = K * N;
            const int blocks = (total + 255) / 256;
            mmb_fold_bf16_kernel<<<blocks, 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(dB.data),
                static_cast<const float*>(dB_scratch.data), total);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // FP16 path: FP32 scratch + fold.
    ::brotensor::Tensor dA_scratch =
        ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, M, K, Dtype::FP32);
    ::brotensor::Tensor dB_scratch =
        ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, K, N, Dtype::FP32);

    {
        dim3 block(MMB_TILE, MMB_TILE);
        dim3 grid((K + MMB_TILE - 1) / MMB_TILE,
                  (M + MMB_TILE - 1) / MMB_TILE);
        mmb_dA_fp16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(dC.data),
            static_cast<const __half*>(B.data),
            static_cast<float*>(dA_scratch.data), M, N, K);
    }
    {
        dim3 block(MMB_TILE, MMB_TILE);
        dim3 grid((N + MMB_TILE - 1) / MMB_TILE,
                  (K + MMB_TILE - 1) / MMB_TILE);
        mmb_dB_fp16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half*>(A.data),
            static_cast<const __half*>(dC.data),
            static_cast<float*>(dB_scratch.data), M, N, K);
    }
    {
        const int total = M * K;
        const int blocks = (total + 255) / 256;
        mmb_fold_fp16_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<__half*>(dA.data),
            static_cast<const float*>(dA_scratch.data), total);
    }
    {
        const int total = K * N;
        const int blocks = (total + 255) / 256;
        mmb_fold_fp16_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<__half*>(dB.data),
            static_cast<const float*>(dB_scratch.data), total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
