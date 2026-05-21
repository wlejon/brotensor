// Fused DDIM update (Metal, FP16). FP32 internal math.

#include <brotensor/runtime.h>

#include <algorithm>
#include <cmath>
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

kernel void k_ddim_step(device const half* x_t      [[buffer(0)]],
                        device const half* eps_pred [[buffer(1)]],
                        device half*       x_prev   [[buffer(2)]],
                        constant float& inv_sqrt_alpha_t      [[buffer(3)]],
                        constant float& sqrt_one_minus_alpha_t [[buffer(4)]],
                        constant float& sqrt_alpha_prev       [[buffer(5)]],
                        constant float& dir_coef              [[buffer(6)]],
                        constant uint&  total                 [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float xt = float(x_t[gid]);
    float eps = float(eps_pred[gid]);
    float x0_pred = (xt - sqrt_one_minus_alpha_t * eps) * inv_sqrt_alpha_t;
    float dir = dir_coef * eps;
    x_prev[gid] = half(sqrt_alpha_prev * x0_pred + dir);
}

kernel void k_ddim_step_bf16(device const bfloat* x_t      [[buffer(0)]],
                             device const bfloat* eps_pred [[buffer(1)]],
                             device bfloat*       x_prev   [[buffer(2)]],
                             constant float& inv_sqrt_alpha_t      [[buffer(3)]],
                             constant float& sqrt_one_minus_alpha_t [[buffer(4)]],
                             constant float& sqrt_alpha_prev       [[buffer(5)]],
                             constant float& dir_coef              [[buffer(6)]],
                             constant uint&  total                 [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float xt = float(x_t[gid]);
    float eps = float(eps_pred[gid]);
    float x0_pred = (xt - sqrt_one_minus_alpha_t * eps) * inv_sqrt_alpha_t;
    float dir = dir_coef * eps;
    x_prev[gid] = bfloat(sqrt_alpha_prev * x0_pred + dir);
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_ddim_step"); });
    return p;
}
id<MTLComputePipelineState> pso_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_ddim_step_bf16"); });
    return p;
}

} // namespace

void ddim_step(const Tensor& x_t, const Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               Tensor& x_prev) {
    if ((x_t.dtype != Dtype::FP16 && x_t.dtype != Dtype::BF16) ||
        eps_pred.dtype != x_t.dtype) {
        throw std::runtime_error("ddim_step_gpu: x_t and eps_pred must be FP16 or BF16 (matching)");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("ddim_step_gpu: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols ||
        x_prev.dtype != x_t.dtype) {
        x_prev.resize(x_t.rows, x_t.cols, x_t.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(x_t.size());
    if (total == 0) return;

    const float sqrt_alpha_t      = std::sqrt(alpha_t);
    const float inv_sqrt_alpha_t  = sqrt_alpha_t > 0.0f ? 1.0f / sqrt_alpha_t : 0.0f;
    const float sqrt_1m_alpha_t   = std::sqrt(std::max(0.0f, 1.0f - alpha_t));
    const float sqrt_alpha_prev   = std::sqrt(std::max(0.0f, alpha_prev));
    const float dir_inner         = 1.0f - alpha_prev - sigma_t * sigma_t;
    const float dir_coef          = std::sqrt(std::max(0.0f, dir_inner));

    id<MTLComputePipelineState> p = (x_t.dtype == Dtype::BF16) ? pso_bf16() : pso();
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
        [enc setBytes:&inv_sqrt_alpha_t length:sizeof(float) atIndex:3];
        [enc setBytes:&sqrt_1m_alpha_t  length:sizeof(float) atIndex:4];
        [enc setBytes:&sqrt_alpha_prev  length:sizeof(float) atIndex:5];
        [enc setBytes:&dir_coef         length:sizeof(float) atIndex:6];
        [enc setBytes:&totalu           length:sizeof(uint32_t) atIndex:7];
        NSUInteger tpt = [p maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
