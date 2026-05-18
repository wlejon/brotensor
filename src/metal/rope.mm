// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void rope_forward_gpu(const GpuTensor&, int, int, int, float, GpuTensor&) {
    throw std::runtime_error("rope_forward_gpu: Metal backend not yet implemented");
}

void rope_backward_gpu(const GpuTensor&, int, int, int, float, GpuTensor&) {
    throw std::runtime_error("rope_backward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
