#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::dispatch1d_sync;
using metal_impl::new_command_buffer;

namespace {

void launch_unary(NSString* name,
                  const GpuTensor& in, GpuTensor& out) {
    if (out.rows != in.rows || out.cols != in.cols) out.resize(in.rows, in.cols);
    const uint32_t n = static_cast<uint32_t>(in.size());
    if (n == 0) return;
    id<MTLBuffer> bin  = buffer_for(in);
    id<MTLBuffer> bout = buffer_for(out);
    const NSUInteger off_in  = buffer_offset_for(in);
    const NSUInteger off_out = buffer_offset_for(out);
    dispatch1d_sync(name, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bin offset:off_in atIndex:0];
        [enc setBuffer:bout offset:off_out atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_binary_back(NSString* name,
                        const GpuTensor& a, const GpuTensor& dY,
                        GpuTensor& dX, int rows, int cols) {
    if (dX.rows != rows || dX.cols != cols) dX.resize(rows, cols);
    const uint32_t n = static_cast<uint32_t>(rows * cols);
    if (n == 0) return;
    id<MTLBuffer> ba  = buffer_for(a);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger off_a  = buffer_offset_for(a);
    const NSUInteger off_dy = buffer_offset_for(dY);
    const NSUInteger off_dx = buffer_offset_for(dX);
    dispatch1d_sync(name, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba  offset:off_a  atIndex:0];
        [enc setBuffer:bdy offset:off_dy atIndex:1];
        [enc setBuffer:bdx offset:off_dx atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

} // namespace

void relu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_unary(@"k_relu_forward", x, y);
}
void relu_backward_gpu(const GpuTensor& x, const GpuTensor& dY, GpuTensor& dX) {
    launch_binary_back(@"k_relu_backward", x, dY, dX, x.rows, x.cols);
}
void tanh_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_unary(@"k_tanh_forward", x, y);
}
void tanh_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    launch_binary_back(@"k_tanh_backward", y, dY, dX, y.rows, y.cols);
}
void sigmoid_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_unary(@"k_sigmoid_forward", x, y);
}
void sigmoid_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    launch_binary_back(@"k_sigmoid_backward", y, dY, dX, y.rows, y.cols);
}

void add_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger off_y = buffer_offset_for(y);
    const NSUInteger off_x = buffer_offset_for(x);
    dispatch1d_sync(@"k_add_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBuffer:bx offset:off_x atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void add_scalar_inplace_gpu(GpuTensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    dispatch1d_sync(@"k_add_scalar_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void scale_inplace_gpu(GpuTensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    dispatch1d_sync(@"k_scale_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void build_slot_mask_gpu(const GpuTensor& x, int offset, int K, int stride,
                         GpuTensor& mask) {
    if (mask.rows != K || mask.cols != 1) mask.resize(K, 1);
    if (K <= 0) return;
    id<MTLBuffer> bx = buffer_for(x);
    id<MTLBuffer> bm = buffer_for(mask);
    const NSUInteger off_x = buffer_offset_for(x);
    const NSUInteger off_m = buffer_offset_for(mask);
    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint32_t Ou = static_cast<uint32_t>(offset);
    const uint32_t Su = static_cast<uint32_t>(stride);
    dispatch1d_sync(@"k_build_slot_mask", Ku, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:off_x atIndex:0];
        [enc setBuffer:bm offset:off_m atIndex:1];
        [enc setBytes:&Ou length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:4];
    });
}

namespace {

NSString* const kActivationSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

inline float silu_scalar(float v) {
    return v / (1.0f + exp(-v));
}
inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanh(u));
}

kernel void k_silu_forward_fp32(device const float* x [[buffer(0)]],
                                device float*       y [[buffer(1)]],
                                constant uint& n      [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = silu_scalar(x[i]);
}
kernel void k_silu_forward_fp16(device const half* x [[buffer(0)]],
                                device half*       y [[buffer(1)]],
                                constant uint& n     [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(silu_scalar(float(x[i])));
}
kernel void k_gelu_forward_fp32(device const float* x [[buffer(0)]],
                                device float*       y [[buffer(1)]],
                                constant uint& n      [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = gelu_tanh_scalar(x[i]);
}
kernel void k_gelu_forward_fp16(device const half* x [[buffer(0)]],
                                device half*       y [[buffer(1)]],
                                constant uint& n     [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(gelu_tanh_scalar(float(x[i])));
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kActivationSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_silu_fp32, @"k_silu_forward_fp32")
DEF_PSO(pso_silu_fp16, @"k_silu_forward_fp16")
DEF_PSO(pso_gelu_fp32, @"k_gelu_forward_fp32")
DEF_PSO(pso_gelu_fp16, @"k_gelu_forward_fp16")
#undef DEF_PSO

void launch_activation_unary(id<MTLComputePipelineState> pso,
                             const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const uint32_t n = static_cast<uint32_t>(x.size());
    if (n == 0) return;
    id<MTLBuffer> bx = buffer_for(x);
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void silu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_activation_unary(x.dtype == Dtype::FP16 ? pso_silu_fp16() : pso_silu_fp32(),
                            x, y);
}
void gelu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    launch_activation_unary(x.dtype == Dtype::FP16 ? pso_gelu_fp16() : pso_gelu_fp32(),
                            x, y);
}

} // namespace brotensor
