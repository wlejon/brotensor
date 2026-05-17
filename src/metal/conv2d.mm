// Metal: TODO — conv2d_forward_gpu not yet implemented.
//
// brodiff's inference path drives this through the CUDA backend on Windows.
// When the Metal mirror lands, replace this stub with an MPS-backed
// implementation (MPSCNNConvolution / MPSGraph conv).

#include <brotensor/ops.h>

#include <stdexcept>

namespace brotensor {

void conv2d_forward_gpu(const GpuTensor& /*X*/,
                        const GpuTensor& /*Wt*/,
                        const GpuTensor* /*bias*/,
                        int /*N*/, int /*C_in*/, int /*H*/, int /*W*/,
                        int /*C_out*/, int /*kH*/, int /*kW*/,
                        int /*stride_h*/, int /*stride_w*/,
                        int /*pad_h*/, int /*pad_w*/,
                        int /*dil_h*/, int /*dil_w*/,
                        GpuTensor& /*Y*/) {
    throw std::runtime_error("brotensor::conv2d_forward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
