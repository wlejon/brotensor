#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Internal shared 16-bit matmul kernel + helpers. Not exported in any public
// header. Several FP16 ops (cross_attention, self_attention, flash_attn,
// linear, resblock skip-conv staging) need the same matmul:
//
//   C(M, N) = A(M, K) @ B(N, K)^T   i.e. C[m, n] = sum_k A[m, k] * B[n, k].
//
// FP16 or BF16 storage, FP32 accumulator. WMMA tensor-core path with a naive
// fallback for tiny / unaligned problems. Both 16-bit storage types share one
// templated kernel (sm_80+ supports BF16 WMMA fragments).

namespace brotensor {
namespace fp16_internal {

// Defined in fp16_matmul.cu — WMMA tensor-core implementation with naive
// fallback. Optional epilogue: `bias` (length N, broadcast over rows) is added
// and `act` applied (see detail/activations.cuh) in the output-store stage.
// For FP16, bias == nullptr && act == 0 is the plain matmul, with the fast int4
// store.
void launch_matmul_ABT_impl(const __half* A, const __half* B, __half* C,
                            int M, int N, int K,
                            const __half* bias = nullptr, int act = 0);
void launch_matmul_ABT_impl(const __nv_bfloat16* A, const __nv_bfloat16* B,
                            __nv_bfloat16* C,
                            int M, int N, int K,
                            const __nv_bfloat16* bias = nullptr, int act = 0);

// Launches C(M, N) = A(M, K) @ B(N, K)^T on the current stream.
inline void launch_matmul_ABT(const __half* A, const __half* B, __half* C,
                              int M, int N, int K) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K);
}
inline void launch_matmul_ABT(const __nv_bfloat16* A, const __nv_bfloat16* B,
                              __nv_bfloat16* C, int M, int N, int K) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K);
}

// As launch_matmul_ABT but fuses bias + activation into the store (epilogue).
inline void launch_matmul_ABT_act(const __half* A, const __half* B, __half* C,
                                  int M, int N, int K,
                                  const __half* bias, int act) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K, bias, act);
}
inline void launch_matmul_ABT_act(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                  __nv_bfloat16* C, int M, int N, int K,
                                  const __nv_bfloat16* bias, int act) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K, bias, act);
}

} // namespace fp16_internal
} // namespace brotensor
