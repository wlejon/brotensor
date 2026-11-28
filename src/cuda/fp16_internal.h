#pragma once

#include "cuda_driver.h"

#include <cstddef>

namespace brotensor {
namespace fp16_internal {

using Fp16 = ::brotensor::detail::cuda::Fp16;
using Bf16 = ::brotensor::detail::cuda::Bf16;

void launch_matmul_ABT_impl(const Fp16* A, const Fp16* B, Fp16* C,
                            int M, int N, int K,
                            const Fp16* bias = nullptr, int act = 0);
void launch_matmul_ABT_impl(const Bf16* A, const Bf16* B, Bf16* C,
                            int M, int N, int K,
                            const Bf16* bias = nullptr, int act = 0);

void launch_matmul_ABT_batched_impl(const Fp16* A, const Fp16* B, Fp16* C,
                                    int batch, int M, int N, int K,
                                    std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                    const Fp16* bias = nullptr, int act = 0);
void launch_matmul_ABT_batched_impl(const Bf16* A, const Bf16* B, Bf16* C,
                                    int batch, int M, int N, int K,
                                    std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                    const Bf16* bias = nullptr, int act = 0);

inline void launch_matmul_ABT(const Fp16* A, const Fp16* B, Fp16* C,
                              int M, int N, int K) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K);
}
inline void launch_matmul_ABT(const Bf16* A, const Bf16* B, Bf16* C,
                              int M, int N, int K) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K);
}

inline void launch_matmul_ABT_batched(const Fp16* A, const Fp16* B, Fp16* C,
                                      int batch, int M, int N, int K,
                                      std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                      const Fp16* bias = nullptr) {
    launch_matmul_ABT_batched_impl(A, B, C, batch, M, N, K, strideA, strideB, strideC, bias, 0);
}
inline void launch_matmul_ABT_batched(const Bf16* A, const Bf16* B, Bf16* C,
                                      int batch, int M, int N, int K,
                                      std::size_t strideA, std::size_t strideB, std::size_t strideC,
                                      const Bf16* bias = nullptr) {
    launch_matmul_ABT_batched_impl(A, B, C, batch, M, N, K, strideA, strideB, strideC, bias, 0);
}

inline void launch_matmul_ABT_act(const Fp16* A, const Fp16* B, Fp16* C,
                                  int M, int N, int K,
                                  const Fp16* bias, int act) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K, bias, act);
}
inline void launch_matmul_ABT_act(const Bf16* A, const Bf16* B, Bf16* C,
                                  int M, int N, int K,
                                  const Bf16* bias, int act) {
    if (M == 0 || N == 0) return;
    launch_matmul_ABT_impl(A, B, C, M, N, K, bias, act);
}

}  // namespace fp16_internal
}  // namespace brotensor
