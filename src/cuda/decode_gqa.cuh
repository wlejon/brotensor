#pragma once

// Grouped-query single-token decode attention over the first *d_len keys of a
// fixed-capacity cache (flash_attention_decode_len's fast path). Internal to
// the CUDA backend; defined in decode_gqa.cu.

#include <cuda_runtime.h>

#include <cstdint>

namespace brotensor::detail::cuda::decode_gqa {

// True when the kernel covers the shape: FP16/BF16, head_dim 64 / 128 / 256,
// group (num_q_heads / num_kv_heads) 1, 2, 4 or 8.
bool supported(int head_dim, int group);

// Q (1, Dq), K/V (cap, Dkv), O (1, Dq); `bf16` selects the element type.
void launch(const void* Q, const void* K, const void* V, const int32_t* d_len, void* O, int cap, int num_q_heads,
            int num_kv_heads, int head_dim, bool bf16, cudaStream_t stream);

}  // namespace brotensor::detail::cuda::decode_gqa
