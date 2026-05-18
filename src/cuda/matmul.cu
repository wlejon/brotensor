// Public matmul_gpu: row-major C(M,N) = A(M,K) @ B(K,N), no bias.
// Dispatched on A.dtype (FP32 + FP16). FP32 accumulation throughout.
//
// The existing internal kernels (`gemm.cu`'s `linear_forward_kernel` and
// `fp16_matmul.cu`'s `launch_matmul_ABT`) compute Y = X @ W^T, which is the
// wrong layout for a plain row-major matmul: B is stored row-major as
// (K, N) but we want to read its columns as the inner dimension. We therefore
// provide direct (M,K) @ (K,N) kernels here.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

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

} // namespace

void matmul_gpu(const GpuTensor& A, const GpuTensor& B, GpuTensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("matmul_gpu: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_gpu: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) {
        C.resize(M, N, A.dtype);
    }
    if (M == 0 || N == 0) return;
    if (K == 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(C.data, 0, C.bytes()));
        return;
    }

    dim3 block(MM_TILE, MM_TILE);
    dim3 grid((N + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
    if (A.dtype == Dtype::FP16) {
        matmul_fp16_kernel<<<grid, block>>>(
            reinterpret_cast<const __half*>(A.data_fp16()),
            reinterpret_cast<const __half*>(B.data_fp16()),
            reinterpret_cast<__half*>(C.data_fp16()),
            M, N, K);
    } else {
        matmul_fp32_kernel<<<grid, block>>>(A.data, B.data, C.data, M, N, K);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
