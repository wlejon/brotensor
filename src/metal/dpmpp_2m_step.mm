// Fused DPM-Solver++ 2M sampler step (Metal, FP16). Multistep, ε-prediction.
// See src/cuda/dpmpp_2m_step.cu for derivation.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_dpmpp_2m_step(device const half* x_t      [[buffer(0)]],
                            device const half* eps_pred [[buffer(1)]],
                            device const half* x0_prev  [[buffer(2)]],
                            device half*       x_prev   [[buffer(3)]],
                            device half*       x0_out   [[buffer(4)]],
                            constant float& sigma_t     [[buffer(5)]],
                            constant float& c_xt        [[buffer(6)]],
                            constant float& c_x0t       [[buffer(7)]],
                            constant float& c_x0prev    [[buffer(8)]],
                            constant uint&  total       [[buffer(9)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float xt  = float(x_t[gid]);
    float eps = float(eps_pred[gid]);
    float x0p = float(x0_prev[gid]);
    float x0t = xt - sigma_t * eps;
    x_prev[gid] = half(c_xt * xt + c_x0t * x0t + c_x0prev * x0p);
    x0_out[gid] = half(x0t);
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_dpmpp_2m_step"); });
    return p;
}

} // namespace

void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred,
                   const Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   Tensor& x_prev, Tensor& x0_out) {
    if (x_t.dtype != Dtype::FP16 || eps_pred.dtype != Dtype::FP16 ||
        x0_prev.dtype != Dtype::FP16) {
        throw std::runtime_error("dpmpp_2m_step_gpu: all inputs must be FP16");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols ||
        x_t.rows != x0_prev.rows  || x_t.cols != x0_prev.cols) {
        throw std::runtime_error("dpmpp_2m_step_gpu: shape mismatch");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols ||
        x_prev.dtype != Dtype::FP16) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    if (x0_out.rows != x_t.rows || x0_out.cols != x_t.cols ||
        x0_out.dtype != Dtype::FP16) {
        x0_out.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    const NSUInteger total = static_cast<NSUInteger>(x_t.size());
    if (total == 0) return;

    id<MTLComputePipelineState> p = pso();
    id<MTLBuffer> bXt = buffer_for(x_t);
    id<MTLBuffer> bE  = buffer_for(eps_pred);
    id<MTLBuffer> bX0p= buffer_for(x0_prev);
    id<MTLBuffer> bXp = buffer_for(x_prev);
    id<MTLBuffer> bX0o= buffer_for(x0_out);
    const NSUInteger oXt = buffer_offset_for(x_t);
    const NSUInteger oE  = buffer_offset_for(eps_pred);
    const NSUInteger oX0p= buffer_offset_for(x0_prev);
    const NSUInteger oXp = buffer_offset_for(x_prev);
    const NSUInteger oX0o= buffer_offset_for(x0_out);
    const uint32_t totalu = static_cast<uint32_t>(total);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:bXt  offset:oXt  atIndex:0];
        [enc setBuffer:bE   offset:oE   atIndex:1];
        [enc setBuffer:bX0p offset:oX0p atIndex:2];
        [enc setBuffer:bXp  offset:oXp  atIndex:3];
        [enc setBuffer:bX0o offset:oX0o atIndex:4];
        [enc setBytes:&sigma_t   length:sizeof(float) atIndex:5];
        [enc setBytes:&c_xt      length:sizeof(float) atIndex:6];
        [enc setBytes:&c_x0t     length:sizeof(float) atIndex:7];
        [enc setBytes:&c_x0prev  length:sizeof(float) atIndex:8];
        [enc setBytes:&totalu    length:sizeof(uint32_t) atIndex:9];
        NSUInteger tpt = [p maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
