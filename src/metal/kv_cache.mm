// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void kv_cache_append_gpu(const GpuTensor&, const GpuTensor&, int,
                         GpuTensor&, GpuTensor&) {
    throw std::runtime_error("kv_cache_append_gpu: Metal backend not yet implemented");
}

void flash_attention_decode_gpu(const GpuTensor&, const GpuTensor&, const GpuTensor&,
                                int, int, GpuTensor&) {
    throw std::runtime_error("flash_attention_decode_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
