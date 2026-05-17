// Metal: TODO — group_norm_forward_gpu not yet implemented.

#include <brotensor/ops.h>

#include <stdexcept>

namespace brotensor {

void group_norm_forward_gpu(const GpuTensor& /*X*/,
                            const GpuTensor& /*gamma*/,
                            const GpuTensor& /*beta*/,
                            int /*N*/, int /*C*/, int /*H*/, int /*W*/,
                            int /*num_groups*/,
                            float /*eps*/,
                            GpuTensor& /*Y*/) {
    throw std::runtime_error("brotensor::group_norm_forward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
