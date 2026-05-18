// Metal stub. CUDA-only op pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void matmul_gpu(const GpuTensor&, const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("matmul_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
