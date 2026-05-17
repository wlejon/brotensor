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

// add_inplace forward decls (FP16 dispatcher lives further down with the
// other FP16-extension kernels).
namespace { void launch_fp16_add_inplace(GpuTensor&, const GpuTensor&, uint32_t);
             void launch_fp16_scale_inplace(GpuTensor&, float, uint32_t); }

void add_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        if (x.dtype != Dtype::FP16) {
            throw std::runtime_error("add_inplace_gpu: dtype mismatch");
        }
        launch_fp16_add_inplace(y, x, n);
        return;
    }
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
    if (y.dtype == Dtype::FP16) {
        launch_fp16_scale_inplace(y, s, n);
        return;
    }
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

namespace {

NSString* const kFp16ExtSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanh(u));
}

kernel void k_add_inplace_fp16(device half*       y [[buffer(0)]],
                               device const half* x [[buffer(1)]],
                               constant uint& n     [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) + float(x[i]));
}
kernel void k_scale_inplace_fp16(device half*       y [[buffer(0)]],
                                 constant float& s    [[buffer(1)]],
                                 constant uint& n     [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) * s);
}
kernel void k_mul_inplace_fp32(device float*       y [[buffer(0)]],
                               device const float* x [[buffer(1)]],
                               constant uint& n      [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] *= x[i];
}
kernel void k_mul_inplace_fp16(device half*       y [[buffer(0)]],
                               device const half* x [[buffer(1)]],
                               constant uint& n     [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) * float(x[i]));
}

// Y(B, D) = X_a(B, D) * gelu(X_b(B, D)) over X(B, 2D).
kernel void k_geglu_forward_fp16(device const half* X [[buffer(0)]],
                                 device half*       Y [[buffer(1)]],
                                 constant uint& B     [[buffer(2)]],
                                 constant uint& D     [[buffer(3)]],
                                 uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a = float(X[b * two_d + d]);
    float gv_raw = float(X[b * two_d + D + d]);
    Y[idx] = half(a * gelu_tanh_scalar(gv_raw));
}

kernel void k_causal_mask_row(device float* mask  [[buffer(0)]],
                              constant uint& L    [[buffer(1)]],
                              constant uint& q    [[buffer(2)]],
                              uint k [[thread_position_in_grid]]) {
    if (k >= L) return;
    mask[k] = (k <= q) ? 1.0f : 0.0f;
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kFp16ExtSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_add_inplace_fp16,   @"k_add_inplace_fp16")
DEF_PSO(pso_scale_inplace_fp16, @"k_scale_inplace_fp16")
DEF_PSO(pso_mul_inplace_fp32,   @"k_mul_inplace_fp32")
DEF_PSO(pso_mul_inplace_fp16,   @"k_mul_inplace_fp16")
DEF_PSO(pso_geglu_fp16,         @"k_geglu_forward_fp16")
DEF_PSO(pso_causal_mask_row,    @"k_causal_mask_row")
#undef DEF_PSO

void launch_1d(id<MTLComputePipelineState> pso, NSUInteger n,
               void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void launch_fp16_add_inplace(GpuTensor& y, const GpuTensor& x, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso_add_inplace_fp16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void launch_fp16_scale_inplace(GpuTensor& y, float s, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger oy = buffer_offset_for(y);
    launch_1d(pso_scale_inplace_fp16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

} // namespace

void mul_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("mul_inplace_gpu: shape/dtype mismatch");
    }
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso = (y.dtype == Dtype::FP16) ?
        pso_mul_inplace_fp16() : pso_mul_inplace_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void geglu_forward_gpu(const GpuTensor& X, GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("geglu_forward_gpu: X must be FP16");
    }
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != Dtype::FP16) {
        Y.resize(B, D, Dtype::FP16);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    const uint32_t Bu = B, Du = D;
    launch_1d(pso_geglu_fp16(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
    });
}

void build_causal_mask_row_gpu(int L, int q, GpuTensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    id<MTLBuffer> bm = buffer_for(mask);
    const NSUInteger om = buffer_offset_for(mask);
    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t qu = static_cast<uint32_t>(q);
    launch_1d(pso_causal_mask_row(), Lu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bm offset:om atIndex:0];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&qu length:sizeof(uint32_t) atIndex:2];
    });
}

} // namespace brotensor
