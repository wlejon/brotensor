#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Internal shared FP16 matmul kernel + helpers. Not exported in any public
// header. Several FP16 ops (cross_attention, self_attention, flash_attn,
// linear, resblock skip-conv staging) need the same naive matmul:
//
//   C(M, N) = A(M, K) @ B(N, K)^T   i.e. C[m, n] = sum_k A[m, k] * B[n, k].
//
// FP16 storage, FP32 accumulator, one thread per output element.

namespace brotensor {
namespace fp16_internal {

constexpr int MATMUL_BLOCK = 128;

static __global__ void matmul_ABT_fp16_kernel(const __half* __restrict__ A,
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

inline int grid_for_matmul(int total) {
    int b = (total + MATMUL_BLOCK - 1) / MATMUL_BLOCK;
    if (b < 1) b = 1;
    return b;
}

// Launches C(M, N) = A(M, K) @ B(N, K)^T on the default stream.
inline void launch_matmul_ABT(const __half* A, const __half* B, __half* C,
                              int M, int N, int K) {
    const int total = M * N;
    if (total == 0) return;
    matmul_ABT_fp16_kernel<<<grid_for_matmul(total), MATMUL_BLOCK>>>(
        A, B, C, M, N, K);
}

} // namespace fp16_internal
} // namespace brotensor
