#pragma once

#include "cuda_driver.h"

#include <cstddef>

namespace brotensor::detail::cuda::mma_gemm {

enum Epilogue : int { kStore = 0, kAccumulate = 1, kGeglu = 2 };
enum AccMode : int { kAccF32 = 0, kAccHybrid = 1 };

std::size_t workspace_floats(int M, int N, int K);

bool launch(const Fp16* A, const Fp16* W, Fp16* C, int M, int N, int K,
            const Fp16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            CUstream stream, int acc_mode = kAccF32);
bool launch(const Bf16* A, const Bf16* W, Bf16* C, int M, int N, int K,
            const Bf16* bias, int act, int epi, float* ws, std::size_t ws_floats,
            CUstream stream, int acc_mode = kAccF32);

bool launch_f32out(const Fp16* A, const Fp16* W, float* C, int M, int N, int K, CUstream stream);
bool launch_f32out(const Bf16* A, const Bf16* W, float* C, int M, int N, int K, CUstream stream);

}  // namespace brotensor::detail::cuda::mma_gemm
