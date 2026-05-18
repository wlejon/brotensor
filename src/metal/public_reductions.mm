// Metal stub. CUDA-only ops pending Metal implementation.
#include <brotensor/ops.h>
#include <stdexcept>

namespace brotensor {

void sum_rows_gpu(const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("sum_rows_gpu: Metal backend not yet implemented");
}

void sum_cols_gpu(const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("sum_cols_gpu: Metal backend not yet implemented");
}

void argmax_rows_gpu(const GpuTensor&, GpuTensor&) {
    throw std::runtime_error("argmax_rows_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
