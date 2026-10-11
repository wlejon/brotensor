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

// O(Lq, D) = softmax(Q K^T / sqrt(head_dim), mask) V, per head.
// Q/K/V/O are (L, D) row-major with D = num_heads*head_dim and the head dim
// contiguous within each row. `mask` is an optional Lk-length float vector
// (positions with mask[k] <= 0.5 drop out). `causal` additionally drops every
// key j > query row i, and requires Lq == Lk (the caller enforces this); it
// composes with `mask`. Caller checks supported() first.
//
// `causal` selects a separate kernel instantiation rather than branching at
// run time, so the non-causal path keeps exactly the codegen it had before
// causal existed.
//
// `batch` > 1 runs that many independent problems packed back to back:
// Q/O are (batch*Lq, D), K/V (batch*Lk, D), and a mask holds batch*Lk
// entries — the equal-length case of packed varlen attention.
void launch(const __half* Q, const __half* K, const __half* V,
            const float* mask, __half* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            cudaStream_t stream, int batch = 1);
void launch(const __nv_bfloat16* Q, const __nv_bfloat16* K,
            const __nv_bfloat16* V,
            const float* mask, __nv_bfloat16* O,
            int Lq, int Lk, int D, int num_heads, int head_dim, bool causal,
            cudaStream_t stream, int batch = 1);

}  // namespace flash_fused
}  // namespace brotensor
