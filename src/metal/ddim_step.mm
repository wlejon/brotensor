// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void ddim_step_gpu(const GpuTensor&, const GpuTensor&,
                   float, float, float,
                   GpuTensor&) {
    throw std::runtime_error("ddim_step_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
