#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Internal fused flash-attention forward (FlashAttention-2 style): tiled
// online-softmax WMMA kernel over the interleaved (L, num_heads*head_dim)
// Q/K/V layout — no per-head extraction and no (Lq, Lk) score materialisation.
// Defined in flash_attention_fused.cu; consumed by flash_attention.cu's
// dispatcher. Not exported in any public header.

namespace brotensor {
namespace flash_fused {

// True if the fused kernel covers this problem (head_dim instantiation).
bool supported(int head_dim);

// O(Lq, D) = softmax(Q K^T / sqrt(head_dim), mask) V, non-causal, per head.
// Q/K/V/O are (L, D) row-major with D = num_heads*head_dim and the head dim
// contiguous within each row. `mask` is an optional Lk-length float vector
// (positions with mask[k] <= 0.5 drop out). Caller checks supported() first.
void launch(const __half* Q, const __half* K, const __half* V,
            const float* mask, __half* O,
            int Lq, int Lk, int D, int num_heads, int head_dim,
            cudaStream_t stream);
void launch(const __nv_bfloat16* Q, const __nv_bfloat16* K,
            const __nv_bfloat16* V,
            const float* mask, __nv_bfloat16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim,
            cudaStream_t stream);

}  // namespace flash_fused
}  // namespace brotensor
