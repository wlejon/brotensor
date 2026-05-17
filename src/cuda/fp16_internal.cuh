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

// Defined in fp16_matmul.cu — WMMA tensor-core implementation with naive fallback.
void launch_matmul_ABT_impl(const __half* A, const __half* B, __half* C,
                            int M, int N, int K);

// Launches C(M, N) = A(M, K) @ B(N, K)^T on the default stream.
inline void launch_matmul_ABT(const __half* A, const __half* B, __half* C,
                              int M, int N, int K) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K);
}

} // namespace fp16_internal
} // namespace brotensor
