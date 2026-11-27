#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Internal fused flash-attention forward (FlashAttention-2 style).

namespace brotensor {
namespace flash_fused {

bool supported(int head_dim);

void launch(const __half* Q, const __half* K, const __half* V,
            const float* mask, __half* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            cudaStream_t stream, int batch = 1, int num_kv_heads = 0);
void launch(const __nv_bfloat16* Q, const __nv_bfloat16* K,
            const __nv_bfloat16* V,
            const float* mask, __nv_bfloat16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            cudaStream_t stream, int batch = 1, int num_kv_heads = 0);

}  // namespace flash_fused
}  // namespace brotensor
