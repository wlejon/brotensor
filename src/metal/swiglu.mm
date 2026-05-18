// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void swiglu_forward_gpu(const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("swiglu_forward_gpu: Metal backend not yet implemented");
}

void swiglu_backward_gpu(const GpuTensor&, const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("swiglu_backward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
