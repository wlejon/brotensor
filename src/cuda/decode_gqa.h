#pragma once

// Grouped-query single-token decode attention over the first *d_len keys of a
// fixed-capacity cache (flash_attention_decode_len's fast path).

#include <cuda_runtime.h>

#include <cstdint>

namespace brotensor::detail::cuda::decode_gqa {

bool supported(int head_dim, int group);

void launch(const void* Q, const void* K, const void* V, const std::int32_t* d_len, void* O, int cap, int num_q_heads,
            int num_kv_heads, int head_dim, int rows, bool bf16, cudaStream_t stream);

}  // namespace brotensor::detail::cuda::decode_gqa
