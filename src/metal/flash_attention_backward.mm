// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void flash_attention_backward_gpu(const GpuTensor&, const GpuTensor&,
                                  const GpuTensor&, const GpuTensor&,
                                  const GpuTensor&, const float*,
                                  int, bool,
                                  GpuTensor&, GpuTensor&, GpuTensor&) {
    throw std::runtime_error("flash_attention_backward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
