// Metal: TODO — fused resblock not yet implemented.

#include <brotensor/ops.h>

#include <stdexcept>

namespace brotensor {

void resblock_forward_gpu(const GpuTensor& /*X*/,
                          const GpuTensor& /*gamma1*/, const GpuTensor& /*beta1*/,
                          const GpuTensor& /*W1*/, const GpuTensor* /*b1*/,
                          const GpuTensor* /*t_emb_shift*/,
                          const GpuTensor& /*gamma2*/, const GpuTensor& /*beta2*/,
                          const GpuTensor& /*W2*/, const GpuTensor* /*b2*/,
                          const GpuTensor* /*Wskip*/, const GpuTensor* /*bskip*/,
                          int /*N*/, int /*C_in*/, int /*C_out*/, int /*H*/, int /*W*/,
                          int /*num_groups*/, float /*eps*/,
                          GpuTensor& /*Y*/) {
    throw std::runtime_error("brotensor::resblock_forward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
