#pragma once

#include "cuda_driver.h"

#include <cstddef>

namespace brotensor::detail::cuda::gemm_skinny {

std::size_t workspace_floats(int M, int N, int K, int epi);

bool launch(const Fp16* X, const Fp16* W, Fp16* Y, int M, int N, int K,
            const Fp16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            CUstream stream);
bool launch(const Bf16* X, const Bf16* W, Bf16* Y, int M, int N, int K,
            const Bf16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            CUstream stream);

}  // namespace brotensor::detail::cuda::gemm_skinny
