// Metal: TODO — flash-attention-style fused attention not yet implemented.

#include <brotensor/ops.h>

#include <stdexcept>

namespace brotensor {

void flash_attention_forward_gpu(const GpuTensor& /*Q*/,
                                 const GpuTensor& /*K*/,
                                 const GpuTensor& /*V*/,
                                 const float* /*d_mask*/,
                                 int /*num_heads*/,
                                 GpuTensor& /*O*/) {
    throw std::runtime_error("brotensor::flash_attention_forward_gpu: Metal backend not yet implemented");
}

void flash_attention_qkvo_forward_gpu(const GpuTensor& /*X*/,
                                      const GpuTensor* /*Ctx*/,
                                      const GpuTensor& /*Wq*/, const GpuTensor& /*Wk*/,
                                      const GpuTensor& /*Wv*/, const GpuTensor& /*Wo*/,
                                      const float* /*d_mask*/,
                                      int /*num_heads*/,
                                      GpuTensor& /*O*/) {
    throw std::runtime_error("brotensor::flash_attention_qkvo_forward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
