// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void rms_norm_forward_gpu(const GpuTensor&, const GpuTensor&, float, GpuTensor&) {
    throw std::runtime_error("rms_norm_forward_gpu: Metal backend not yet implemented");
}

void rms_norm_backward_gpu(const GpuTensor&, const GpuTensor&, const GpuTensor&,
                           float, GpuTensor&, GpuTensor&) {
    throw std::runtime_error("rms_norm_backward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
