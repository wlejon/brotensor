// Metal: TODO — upsample/downsample kernels not yet implemented.

#include <brotensor/ops.h>

#include <stdexcept>

namespace brotensor {

void upsample_nearest_2x_gpu(const GpuTensor& /*X*/, int /*N*/, int /*C*/,
                             int /*H*/, int /*W*/, GpuTensor& /*Y*/) {
    throw std::runtime_error("brotensor::upsample_nearest_2x_gpu: Metal backend not yet implemented");
}
void upsample_bilinear_2x_gpu(const GpuTensor& /*X*/, int /*N*/, int /*C*/,
                              int /*H*/, int /*W*/, GpuTensor& /*Y*/) {
    throw std::runtime_error("brotensor::upsample_bilinear_2x_gpu: Metal backend not yet implemented");
}
void downsample_avg_2x_gpu(const GpuTensor& /*X*/, int /*N*/, int /*C*/,
                           int /*H*/, int /*W*/, GpuTensor& /*Y*/) {
    throw std::runtime_error("brotensor::downsample_avg_2x_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
