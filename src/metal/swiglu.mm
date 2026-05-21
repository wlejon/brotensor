// SwiGLU forward + backward (Metal). FP32 + FP16; FP32 internal math.

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

static inline float silu_scalar(float v)      { return v / (1.0f + exp(-v)); }
static inline float silu_grad_scalar(float v) {
    float s = 1.0f / (1.0f + exp(-v));
    return s * (1.0f + v * (1.0f - s));
}

kernel void k_swiglu_fw_fp32(device const float* X [[buffer(0)]],
                             device float*       Y [[buffer(1)]],
                             constant uint& B [[buffer(2)]],
                             constant uint& D [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = B * D;
    if (gid >= total) return;
    uint b = gid / D;
    uint d = gid % D;
    uint two_d = 2u * D;
    float a  = X[b * two_d + d];
    float bh = X[b * two_d + D + d];
    Y[gid] = silu_scalar(a) * bh;
}

kernel void k_swiglu_fw_fp16(device const half* X [[buffer(0)]],
                             device half*       Y [[buffer(1)]],
                             constant uint& B [[buffer(2)]],
                             constant uint& D [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = B * D;
    if (gid >= total) return;
    uint b = gid / D;
    uint d = gid % D;
    uint two_d = 2u * D;
    float a  = float(X[b * two_d + d]);
    float bh = float(X[b * two_d + D + d]);
    Y[gid] = half(silu_scalar(a) * bh);
}

kernel void k_swiglu_bw_fp32(device const float* X  [[buffer(0)]],
                             device const float* dY [[buffer(1)]],
                             device float*       dX [[buffer(2)]],
                             constant uint& B [[buffer(3)]],
                             constant uint& D [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = B * D;
    if (gid >= total) return;
    uint b = gid / D;
    uint d = gid % D;
    uint two_d = 2u * D;
    float a  = X[b * two_d + d];
    float bh = X[b * two_d + D + d];
    float dy = dY[gid];
    float s  = silu_scalar(a);
    float sp = silu_grad_scalar(a);
    dX[b * two_d + d]     = dy * bh * sp;
    dX[b * two_d + D + d] = dy * s;
}

kernel void k_swiglu_bw_fp16(device const half* X  [[buffer(0)]],
                             device const half* dY [[buffer(1)]],
                             device half*       dX [[buffer(2)]],
                             constant uint& B [[buffer(3)]],
                             constant uint& D [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = B * D;
    if (gid >= total) return;
    uint b = gid / D;
    uint d = gid % D;
    uint two_d = 2u * D;
    float a  = float(X[b * two_d + d]);
    float bh = float(X[b * two_d + D + d]);
    float dy = float(dY[gid]);
    float s  = silu_scalar(a);
    float sp = silu_grad_scalar(a);
    dX[b * two_d + d]     = half(dy * bh * sp);
    dX[b * two_d + D + d] = half(dy * s);
}

kernel void k_swiglu_fw_bf16(device const bfloat* X [[buffer(0)]],
                              device bfloat*       Y [[buffer(1)]],
                              constant uint& B [[buffer(2)]],
                              constant uint& D [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total = B * D;
    if (gid >= total) return;
    uint b = gid / D;
    uint d = gid % D;
    uint two_d = 2u * D;
    float a  = float(X[b * two_d + d]);
    float bh = float(X[b * two_d + D + d]);
    Y[gid] = bfloat(silu_scalar(a) * bh);
}

kernel void k_swiglu_bw_bf16(device const bfloat* X  [[buffer(0)]],
                              device const bfloat* dY [[buffer(1)]],
                              device bfloat*       dX [[buffer(2)]],
                              constant uint& B [[buffer(3)]],
                              constant uint& D [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total = B * D;
    if (gid >= total) return;
    uint b = gid / D;
    uint d = gid % D;
    uint two_d = 2u * D;
    float a  = float(X[b * two_d + d]);
    float bh = float(X[b * two_d + D + d]);
    float dy = float(dY[gid]);
    float s  = silu_scalar(a);
    float sp = silu_grad_scalar(a);
    dX[b * two_d + d]     = bfloat(dy * bh * sp);
    dX[b * two_d + D + d] = bfloat(dy * s);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_fw_fp32,  @"k_swiglu_fw_fp32")
DEF_PSO(pso_fw_fp16,  @"k_swiglu_fw_fp16")
DEF_PSO(pso_fw_bf16,  @"k_swiglu_fw_bf16")
DEF_PSO(pso_bw_fp32,  @"k_swiglu_bw_fp32")
DEF_PSO(pso_bw_fp16,  @"k_swiglu_bw_fp16")
DEF_PSO(pso_bw_bf16,  @"k_swiglu_bw_bf16")
#undef DEF_PSO

} // namespace

void swiglu_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(B) * D;
    if (total == 0) return;
    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_fw_fp16()
                                   : (X.dtype == Dtype::BF16) ? pso_fw_bf16()
                                   : pso_fw_fp32();
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oY = buffer_offset_for(Y);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void swiglu_backward(const Tensor& X, const Tensor& dY,
                     Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_backward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(B) * D;
    if (total == 0) return;
    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_bw_fp16()
                                   : (X.dtype == Dtype::BF16) ? pso_bw_bf16()
                                   : pso_bw_fp32();
    id<MTLBuffer> bX  = buffer_for(X);
    id<MTLBuffer> bdY = buffer_for(dY);
    id<MTLBuffer> bdX = buffer_for(dX);
    const NSUInteger oX  = buffer_offset_for(X);
    const NSUInteger odY = buffer_offset_for(dY);
    const NSUInteger odX = buffer_offset_for(dX);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bdY offset:odY atIndex:1];
        [enc setBuffer:bdX offset:odX atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
