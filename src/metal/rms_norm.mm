// RMSNorm forward + backward (Metal). One threadgroup per row.
// FP16 dGamma accumulates into an FP32 scratch (atomic_float), then a fold
// kernel adds into the caller-owned FP16 dGamma.

#include <brotensor/runtime.h>

#include <cstring>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger RMS_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint RMS_BLOCK = 256;

kernel void k_rms_fw_fp32(device const float* X     [[buffer(0)]],
                          device const float* gamma [[buffer(1)]],
                          device float*       Y     [[buffer(2)]],
                          constant uint& B   [[buffer(3)]],
                          constant uint& D   [[buffer(4)]],
                          constant float& eps [[buffer(5)]],
                          uint b   [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sdata[RMS_BLOCK];
    if (b >= B) return;
    device const float* xrow = X + b * D;
    device       float* yrow = Y + b * D;
    float local = 0.0f;
    for (uint j = tid; j < D; j += tgs) {
        float v = xrow[j];
        local += v * v;
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rrms = rsqrt(sdata[0] / float(D) + eps);
    for (uint j = tid; j < D; j += tgs) {
        yrow[j] = xrow[j] * gamma[j] * rrms;
    }
}

kernel void k_rms_fw_fp16(device const half* X     [[buffer(0)]],
                          device const half* gamma [[buffer(1)]],
                          device half*       Y     [[buffer(2)]],
                          constant uint& B   [[buffer(3)]],
                          constant uint& D   [[buffer(4)]],
                          constant float& eps [[buffer(5)]],
                          uint b   [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sdata[RMS_BLOCK];
    if (b >= B) return;
    device const half* xrow = X + b * D;
    device       half* yrow = Y + b * D;
    float local = 0.0f;
    for (uint j = tid; j < D; j += tgs) {
        float v = float(xrow[j]);
        local += v * v;
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rrms = rsqrt(sdata[0] / float(D) + eps);
    for (uint j = tid; j < D; j += tgs) {
        float xv = float(xrow[j]);
        float gv = float(gamma[j]);
        yrow[j] = half(xv * gv * rrms);
    }
}

kernel void k_rms_bw_fp32(device const float* X     [[buffer(0)]],
                          device const float* gamma [[buffer(1)]],
                          device const float* dY    [[buffer(2)]],
                          device float*       dX    [[buffer(3)]],
                          device atomic_float* dGamma [[buffer(4)]],
                          constant uint& B   [[buffer(5)]],
                          constant uint& D   [[buffer(6)]],
                          constant float& eps [[buffer(7)]],
                          uint b   [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sdata[RMS_BLOCK];
    if (b >= B) return;
    device const float* xrow  = X  + b * D;
    device const float* dyrow = dY + b * D;
    device       float* dxrow = dX + b * D;

    float local = 0.0f;
    for (uint j = tid; j < D; j += tgs) {
        float v = xrow[j];
        local += v * v;
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rrms = rsqrt(sdata[0] / float(D) + eps);

    float local2 = 0.0f;
    for (uint j = tid; j < D; j += tgs) {
        local2 += xrow[j] * dyrow[j] * gamma[j];
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_xdy = sdata[0];
    float inv_D = 1.0f / float(D);
    float coeff = inv_D * rrms * rrms * sum_xdy;

    for (uint j = tid; j < D; j += tgs) {
        float g  = gamma[j];
        float dy = dyrow[j];
        float x  = xrow[j];
        dxrow[j] = rrms * (g * dy - x * coeff);
        atomic_fetch_add_explicit(&dGamma[j], dy * x * rrms, memory_order_relaxed);
    }
}

kernel void k_rms_bw_fp16(device const half* X     [[buffer(0)]],
                          device const half* gamma [[buffer(1)]],
                          device const half* dY    [[buffer(2)]],
                          device half*       dX    [[buffer(3)]],
                          device atomic_float* dGamma_scratch [[buffer(4)]],
                          constant uint& B   [[buffer(5)]],
                          constant uint& D   [[buffer(6)]],
                          constant float& eps [[buffer(7)]],
                          uint b   [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sdata[RMS_BLOCK];
    if (b >= B) return;
    device const half* xrow  = X  + b * D;
    device const half* dyrow = dY + b * D;
    device       half* dxrow = dX + b * D;

    float local = 0.0f;
    for (uint j = tid; j < D; j += tgs) {
        float v = float(xrow[j]);
        local += v * v;
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rrms = rsqrt(sdata[0] / float(D) + eps);

    float local2 = 0.0f;
    for (uint j = tid; j < D; j += tgs) {
        local2 += float(xrow[j]) * float(dyrow[j]) * float(gamma[j]);
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_xdy = sdata[0];
    float inv_D = 1.0f / float(D);
    float coeff = inv_D * rrms * rrms * sum_xdy;

    for (uint j = tid; j < D; j += tgs) {
        float g  = float(gamma[j]);
        float dy = float(dyrow[j]);
        float x  = float(xrow[j]);
        dxrow[j] = half(rrms * (g * dy - x * coeff));
        atomic_fetch_add_explicit(&dGamma_scratch[j], dy * x * rrms, memory_order_relaxed);
    }
}

kernel void k_rms_fold_fp16(device half*        dst [[buffer(0)]],
                            device const float* src [[buffer(1)]],
                            constant uint& n        [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = half(float(dst[gid]) + src[gid]);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_fw_fp32, @"k_rms_fw_fp32")
DEF_PSO(pso_fw_fp16, @"k_rms_fw_fp16")
DEF_PSO(pso_bw_fp32, @"k_rms_bw_fp32")
DEF_PSO(pso_bw_fp16, @"k_rms_bw_fp16")
DEF_PSO(pso_fold,    @"k_rms_fold_fp16")
#undef DEF_PSO

} // namespace

void rms_norm_forward(const Tensor& X, const Tensor& gamma,
                      float eps, Tensor& Y) {
    if (gamma.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_forward_gpu: gamma.dtype must match X.dtype");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (gamma.size() != D) {
        throw std::runtime_error("rms_norm_forward_gpu: gamma must have D elements");
    }
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    if (B == 0 || D == 0) return;

    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_fw_fp16() : pso_fw_fp32();
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bG = buffer_for(gamma);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oG = buffer_offset_for(gamma);
    const NSUInteger oY = buffer_offset_for(Y);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bG offset:oG atIndex:1];
        [enc setBuffer:bY offset:oY atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(RMS_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void rms_norm_backward(const Tensor& X, const Tensor& gamma,
                       const Tensor& dY, float eps,
                       Tensor& dX, Tensor& dGamma) {
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype || dGamma.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_backward_gpu: dtypes must match");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (dY.rows != B || dY.cols != D) {
        throw std::runtime_error("rms_norm_backward_gpu: dY shape mismatch");
    }
    if (gamma.size() != D || dGamma.size() != D) {
        throw std::runtime_error("rms_norm_backward_gpu: gamma/dGamma size mismatch");
    }
    if (dX.rows != B || dX.cols != D || dX.dtype != X.dtype) {
        dX.resize(B, D, X.dtype);
    }
    if (B == 0 || D == 0) return;

    id<MTLBuffer> bX  = buffer_for(X);
    id<MTLBuffer> bG  = buffer_for(gamma);
    id<MTLBuffer> bdY = buffer_for(dY);
    id<MTLBuffer> bdX = buffer_for(dX);
    id<MTLBuffer> bdG = buffer_for(dGamma);
    const NSUInteger oX  = buffer_offset_for(X);
    const NSUInteger oG  = buffer_offset_for(gamma);
    const NSUInteger odY = buffer_offset_for(dY);
    const NSUInteger odX = buffer_offset_for(dX);
    const NSUInteger odG = buffer_offset_for(dGamma);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);

    if (X.dtype == Dtype::FP32) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_bw_fp32()];
            [enc setBuffer:bX  offset:oX  atIndex:0];
            [enc setBuffer:bG  offset:oG  atIndex:1];
            [enc setBuffer:bdY offset:odY atIndex:2];
            [enc setBuffer:bdX offset:odX atIndex:3];
            [enc setBuffer:bdG offset:odG atIndex:4];
            [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&eps length:sizeof(float) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(RMS_BLOCK, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    // FP16: scratch + fold.
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:D * sizeof(float)
                        options:MTLResourceStorageModeShared];
        std::memset([scratch contents], 0, D * sizeof(float));

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_bw_fp16()];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bG  offset:oG  atIndex:1];
        [enc setBuffer:bdY offset:odY atIndex:2];
        [enc setBuffer:bdX offset:odX atIndex:3];
        [enc setBuffer:scratch offset:0 atIndex:4];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&eps length:sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(RMS_BLOCK, 1, 1)];

        [enc setComputePipelineState:pso_fold()];
        const uint32_t Dn = static_cast<uint32_t>(D);
        const NSUInteger tpt = 64;
        [enc setBuffer:bdG offset:odG atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&Dn length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((D + tpt - 1) / tpt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
