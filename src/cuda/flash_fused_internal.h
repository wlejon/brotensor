#pragma once

#include "cuda_driver.h"

// Internal fused flash-attention forward (FlashAttention-2 style).

namespace brotensor {
namespace flash_fused {

using Fp16 = ::brotensor::detail::cuda::Fp16;
using Bf16 = ::brotensor::detail::cuda::Bf16;

bool supported(int head_dim);

void launch(const Fp16* Q, const Fp16* K, const Fp16* V,
            const float* mask, Fp16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            CUstream stream, int batch = 1, int num_kv_heads = 0);
void launch(const Bf16* Q, const Bf16* K,
            const Bf16* V,
            const float* mask, Bf16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            CUstream stream, int batch = 1, int num_kv_heads = 0);

}  // namespace flash_fused
}  // namespace brotensor
