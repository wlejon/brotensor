// Fused Euler-discrete sampler step (Metal, FP16). ε-prediction.
//   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_euler_step(device const half* x_t      [[buffer(0)]],
                         device const half* eps_pred [[buffer(1)]],
                         device half*       x_prev   [[buffer(2)]],
                         constant float& dsigma      [[buffer(3)]],
                         constant uint&  total       [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float xt  = float(x_t[gid]);
    float eps = float(eps_pred[gid]);
    x_prev[gid] = half(xt + dsigma * eps);
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_euler_step"); });
    return p;
}

} // namespace

void euler_step_gpu(const GpuTensor& x_t, const GpuTensor& eps_pred,
                    float sigma_t, float sigma_prev,
                    GpuTensor& x_prev) {
    if (x_t.dtype != Dtype::FP16 || eps_pred.dtype != Dtype::FP16) {
        throw std::runtime_error("euler_step_gpu: x_t and eps_pred must be FP16");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("euler_step_gpu: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols ||
        x_prev.dtype != Dtype::FP16) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    const NSUInteger total = static_cast<NSUInteger>(x_t.size());
    if (total == 0) return;

    const float dsigma = sigma_prev - sigma_t;

    id<MTLComputePipelineState> p = pso();
    id<MTLBuffer> bX = buffer_for(x_t);
    id<MTLBuffer> bE = buffer_for(eps_pred);
    id<MTLBuffer> bP = buffer_for(x_prev);
    const NSUInteger oX = buffer_offset_for(x_t);
    const NSUInteger oE = buffer_offset_for(eps_pred);
    const NSUInteger oP = buffer_offset_for(x_prev);
    const uint32_t totalu = static_cast<uint32_t>(total);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bE offset:oE atIndex:1];
        [enc setBuffer:bP offset:oP atIndex:2];
        [enc setBytes:&dsigma length:sizeof(float) atIndex:3];
        [enc setBytes:&totalu length:sizeof(uint32_t) atIndex:4];
        NSUInteger tpt = [p maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor
