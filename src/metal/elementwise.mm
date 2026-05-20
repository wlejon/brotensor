#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::dispatch1d_sync;
using metal_impl::new_command_buffer;

namespace {

void launch_unary(NSString* name,
                  const Tensor& in, Tensor& out) {
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
                        const Tensor& a, const Tensor& dY,
                        Tensor& dX, int rows, int cols) {
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

void relu_forward(const Tensor& x, Tensor& y) {
    launch_unary(@"k_relu_forward", x, y);
}
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_binary_back(@"k_relu_backward", x, dY, dX, x.rows, x.cols);
}
void tanh_forward(const Tensor& x, Tensor& y) {
    launch_unary(@"k_tanh_forward", x, y);
}
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_binary_back(@"k_tanh_backward", y, dY, dX, y.rows, y.cols);
}
void sigmoid_forward(const Tensor& x, Tensor& y) {
    launch_unary(@"k_sigmoid_forward", x, y);
}
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_binary_back(@"k_sigmoid_backward", y, dY, dX, y.rows, y.cols);
}

// add_inplace forward decls (FP16 dispatcher lives further down with the
// other FP16-extension kernels).
namespace { void launch_fp16_add_inplace(Tensor&, const Tensor&, uint32_t);
             void launch_fp16_scale_inplace(Tensor&, float, uint32_t); }

void add_inplace(Tensor& y, const Tensor& x) {
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

// Forward decls for FP16 PSOs defined later in this TU.
namespace {
id<MTLComputePipelineState> pso_add_scalar_fp16();
id<MTLComputePipelineState> pso_scale_fp16();
id<MTLComputePipelineState> pso_clamp_fp32();
id<MTLComputePipelineState> pso_clamp_fp16();
}

namespace {
void dispatch_scalar_inplace(id<MTLComputePipelineState> pso,
                             id<MTLBuffer> by, NSUInteger off,
                             float s, uint32_t n) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:by offset:off atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}
} // namespace

void add_scalar_inplace(Tensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    if (y.dtype == Dtype::FP16) {
        dispatch_scalar_inplace(pso_add_scalar_fp16(), by, off_y, s, n);
        return;
    }
    dispatch1d_sync(@"k_add_scalar_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void scale_inplace(Tensor& y, float s) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        launch_fp16_scale_inplace(y, s, n);
        return;
    }
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    if (y.dtype == Dtype::FP16) {
        dispatch_scalar_inplace(pso_scale_fp16(), by, off_y, s, n);
        return;
    }
    dispatch1d_sync(@"k_scale_inplace", n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

void clamp(Tensor& y, float lo, float hi) {
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger off_y = buffer_offset_for(y);
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_clamp_fp16() : pso_clamp_fp32();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:by offset:off_y atIndex:0];
        [enc setBytes:&lo length:sizeof(float) atIndex:1];
        [enc setBytes:&hi length:sizeof(float) atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask) {
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

// MSL has no built-in erf; Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).
inline float erf_approx(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    float sign_x = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t  = 1.0f / (1.0f + p * ax);
    float y  = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign_x * y;
}
#define erf(x) erf_approx(x)

inline float silu_scalar(float v) {
    return v / (1.0f + exp(-v));
}
inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanh(u));
}
inline float quick_gelu_scalar(float v) {
    return v / (1.0f + exp(-1.702f * v));
}
inline float silu_grad_scalar(float v) {
    float s = 1.0f / (1.0f + exp(-v));
    return s * (1.0f + v * (1.0f - s));
}
inline float gelu_tanh_grad_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    float t = tanh(u);
    float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}
inline float quick_gelu_grad_scalar(float v) {
    float s = 1.0f / (1.0f + exp(-1.702f * v));
    return s + v * 1.702f * s * (1.0f - s);
}
inline float gelu_exact_scalar(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erf(v * kInvSqrt2));
}
inline float gelu_exact_grad_scalar(float v) {
    constexpr float kInvSqrt2   = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    float cdf_term = 0.5f * (1.0f + erf(v * kInvSqrt2));
    float pdf      = kInvSqrt2Pi * exp(-0.5f * v * v);
    return cdf_term + v * pdf;
}

kernel void k_silu_backward_fp32(device const float* x  [[buffer(0)]],
                                 device const float* dY [[buffer(1)]],
                                 device float*       dX [[buffer(2)]],
                                 constant uint& n       [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = dY[i] * silu_grad_scalar(x[i]);
}
kernel void k_silu_backward_fp16(device const half* x  [[buffer(0)]],
                                 device const half* dY [[buffer(1)]],
                                 device half*       dX [[buffer(2)]],
                                 constant uint& n      [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = half(float(dY[i]) * silu_grad_scalar(float(x[i])));
}
kernel void k_gelu_backward_fp32(device const float* x  [[buffer(0)]],
                                 device const float* dY [[buffer(1)]],
                                 device float*       dX [[buffer(2)]],
                                 constant uint& n       [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = dY[i] * gelu_tanh_grad_scalar(x[i]);
}
kernel void k_gelu_backward_fp16(device const half* x  [[buffer(0)]],
                                 device const half* dY [[buffer(1)]],
                                 device half*       dX [[buffer(2)]],
                                 constant uint& n      [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = half(float(dY[i]) * gelu_tanh_grad_scalar(float(x[i])));
}
kernel void k_quick_gelu_backward_fp32(device const float* x  [[buffer(0)]],
                                       device const float* dY [[buffer(1)]],
                                       device float*       dX [[buffer(2)]],
                                       constant uint& n       [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = dY[i] * quick_gelu_grad_scalar(x[i]);
}
kernel void k_quick_gelu_backward_fp16(device const half* x  [[buffer(0)]],
                                       device const half* dY [[buffer(1)]],
                                       device half*       dX [[buffer(2)]],
                                       constant uint& n      [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = half(float(dY[i]) * quick_gelu_grad_scalar(float(x[i])));
}
kernel void k_gelu_exact_forward_fp32(device const float* x [[buffer(0)]],
                                      device float*       y [[buffer(1)]],
                                      constant uint& n      [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = gelu_exact_scalar(x[i]);
}
kernel void k_gelu_exact_forward_fp16(device const half* x [[buffer(0)]],
                                      device half*       y [[buffer(1)]],
                                      constant uint& n     [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(gelu_exact_scalar(float(x[i])));
}
kernel void k_gelu_exact_backward_fp32(device const float* x  [[buffer(0)]],
                                       device const float* dY [[buffer(1)]],
                                       device float*       dX [[buffer(2)]],
                                       constant uint& n       [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = dY[i] * gelu_exact_grad_scalar(x[i]);
}
kernel void k_gelu_exact_backward_fp16(device const half* x  [[buffer(0)]],
                                       device const half* dY [[buffer(1)]],
                                       device half*       dX [[buffer(2)]],
                                       constant uint& n      [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = half(float(dY[i]) * gelu_exact_grad_scalar(float(x[i])));
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
kernel void k_quick_gelu_forward_fp32(device const float* x [[buffer(0)]],
                                      device float*       y [[buffer(1)]],
                                      constant uint& n      [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = quick_gelu_scalar(x[i]);
}
kernel void k_quick_gelu_forward_fp16(device const half* x [[buffer(0)]],
                                      device half*       y [[buffer(1)]],
                                      constant uint& n     [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(quick_gelu_scalar(float(x[i])));
}
kernel void k_add_scalar_inplace_fp16(device half* y [[buffer(0)]],
                                      constant float& s [[buffer(1)]],
                                      constant uint& n  [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) + s);
}
kernel void k_scale_inplace_fp16(device half* y [[buffer(0)]],
                                 constant float& s [[buffer(1)]],
                                 constant uint& n  [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) * s);
}
kernel void k_clamp_fp32(device float* y [[buffer(0)]],
                         constant float& lo [[buffer(1)]],
                         constant float& hi [[buffer(2)]],
                         constant uint&  n  [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = y[i];
    v = max(v, lo);
    v = min(v, hi);
    y[i] = v;
}
kernel void k_clamp_fp16(device half* y [[buffer(0)]],
                         constant float& lo [[buffer(1)]],
                         constant float& hi [[buffer(2)]],
                         constant uint&  n  [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = float(y[i]);
    v = max(v, lo);
    v = min(v, hi);
    y[i] = half(v);
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
DEF_PSO(pso_quick_gelu_fp32, @"k_quick_gelu_forward_fp32")
DEF_PSO(pso_quick_gelu_fp16, @"k_quick_gelu_forward_fp16")
DEF_PSO(pso_silu_bwd_fp32,       @"k_silu_backward_fp32")
DEF_PSO(pso_silu_bwd_fp16,       @"k_silu_backward_fp16")
DEF_PSO(pso_gelu_bwd_fp32,       @"k_gelu_backward_fp32")
DEF_PSO(pso_gelu_bwd_fp16,       @"k_gelu_backward_fp16")
DEF_PSO(pso_quick_gelu_bwd_fp32, @"k_quick_gelu_backward_fp32")
DEF_PSO(pso_quick_gelu_bwd_fp16, @"k_quick_gelu_backward_fp16")
DEF_PSO(pso_gelu_exact_fp32,     @"k_gelu_exact_forward_fp32")
DEF_PSO(pso_gelu_exact_fp16,     @"k_gelu_exact_forward_fp16")
DEF_PSO(pso_gelu_exact_bwd_fp32, @"k_gelu_exact_backward_fp32")
DEF_PSO(pso_gelu_exact_bwd_fp16, @"k_gelu_exact_backward_fp16")
DEF_PSO(pso_add_scalar_fp16,  @"k_add_scalar_inplace_fp16")
DEF_PSO(pso_scale_fp16,       @"k_scale_inplace_fp16")
DEF_PSO(pso_clamp_fp32,       @"k_clamp_fp32")
DEF_PSO(pso_clamp_fp16,       @"k_clamp_fp16")
#undef DEF_PSO

void launch_activation_unary(id<MTLComputePipelineState> pso,
                             const Tensor& x, Tensor& y) {
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
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void silu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(x.dtype == Dtype::FP16 ? pso_silu_fp16() : pso_silu_fp32(),
                            x, y);
}
void gelu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(x.dtype == Dtype::FP16 ? pso_gelu_fp16() : pso_gelu_fp32(),
                            x, y);
}
void quick_gelu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(x.dtype == Dtype::FP16 ? pso_quick_gelu_fp16() : pso_quick_gelu_fp32(),
                            x, y);
}

namespace {
void launch_activation_bwd(id<MTLComputePipelineState> pso,
                           const Tensor& x, const Tensor& dY,
                           Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const uint32_t n = static_cast<uint32_t>(x.size());
    if (n == 0) return;
    id<MTLBuffer> bx  = buffer_for(x);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ox  = buffer_offset_for(x);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}
} // namespace

void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(x.dtype == Dtype::FP16 ? pso_silu_bwd_fp16() : pso_silu_bwd_fp32(),
                          x, dY, dX);
}
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(x.dtype == Dtype::FP16 ? pso_gelu_bwd_fp16() : pso_gelu_bwd_fp32(),
                          x, dY, dX);
}
void quick_gelu_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX) {
    launch_activation_bwd(x.dtype == Dtype::FP16 ? pso_quick_gelu_bwd_fp16()
                                                  : pso_quick_gelu_bwd_fp32(),
                          x, dY, dX);
}
void gelu_exact_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(x.dtype == Dtype::FP16 ? pso_gelu_exact_fp16()
                                                    : pso_gelu_exact_fp32(),
                            x, y);
}
void gelu_exact_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX) {
    launch_activation_bwd(x.dtype == Dtype::FP16 ? pso_gelu_exact_bwd_fp16()
                                                  : pso_gelu_exact_bwd_fp32(),
                          x, dY, dX);
}

namespace {

NSString* const kFp16ExtSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// MSL has no built-in erf; Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).
inline float erf_approx(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    float sign_x = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t  = 1.0f / (1.0f + p * ax);
    float y  = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign_x * y;
}
#define erf(x) erf_approx(x)

inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanh(u));
}
inline float gelu_tanh_grad_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    float t = tanh(u);
    float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}
inline float gelu_exact_scalar(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + erf(v * kInvSqrt2));
}
inline float gelu_exact_grad_scalar(float v) {
    constexpr float kInvSqrt2   = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    float cdf_term = 0.5f * (1.0f + erf(v * kInvSqrt2));
    float pdf      = kInvSqrt2Pi * exp(-0.5f * v * v);
    return cdf_term + v * pdf;
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
kernel void k_geglu_forward_fp32(device const float* X [[buffer(0)]],
                                 device float*       Y [[buffer(1)]],
                                 constant uint& B      [[buffer(2)]],
                                 constant uint& D      [[buffer(3)]],
                                 uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a = X[b * two_d + d];
    float gv_raw = X[b * two_d + D + d];
    Y[idx] = a * gelu_tanh_scalar(gv_raw);
}
kernel void k_geglu_backward_fp32(device const float* X  [[buffer(0)]],
                                  device const float* dY [[buffer(1)]],
                                  device float*       dX [[buffer(2)]],
                                  constant uint& B       [[buffer(3)]],
                                  constant uint& D       [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a       = X[b * two_d + d];
    float bh      = X[b * two_d + D + d];
    float dy      = dY[idx];
    float g       = gelu_tanh_scalar(bh);
    float gprime  = gelu_tanh_grad_scalar(bh);
    dX[b * two_d + d]     = dy * g;
    dX[b * two_d + D + d] = dy * a * gprime;
}
kernel void k_geglu_backward_fp16(device const half* X  [[buffer(0)]],
                                  device const half* dY [[buffer(1)]],
                                  device half*       dX [[buffer(2)]],
                                  constant uint& B      [[buffer(3)]],
                                  constant uint& D      [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a       = float(X[b * two_d + d]);
    float bh      = float(X[b * two_d + D + d]);
    float dy      = float(dY[idx]);
    float g       = gelu_tanh_scalar(bh);
    float gprime  = gelu_tanh_grad_scalar(bh);
    dX[b * two_d + d]     = half(dy * g);
    dX[b * two_d + D + d] = half(dy * a * gprime);
}

kernel void k_geglu_exact_forward_fp32(device const float* X [[buffer(0)]],
                                       device float*       Y [[buffer(1)]],
                                       constant uint& B      [[buffer(2)]],
                                       constant uint& D      [[buffer(3)]],
                                       uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a = X[b * two_d + d];
    float gv_raw = X[b * two_d + D + d];
    Y[idx] = a * gelu_exact_scalar(gv_raw);
}
kernel void k_geglu_exact_forward_fp16(device const half* X [[buffer(0)]],
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
    Y[idx] = half(a * gelu_exact_scalar(gv_raw));
}
kernel void k_geglu_exact_backward_fp32(device const float* X  [[buffer(0)]],
                                        device const float* dY [[buffer(1)]],
                                        device float*       dX [[buffer(2)]],
                                        constant uint& B       [[buffer(3)]],
                                        constant uint& D       [[buffer(4)]],
                                        uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a       = X[b * two_d + d];
    float bh      = X[b * two_d + D + d];
    float dy      = dY[idx];
    float g       = gelu_exact_scalar(bh);
    float gprime  = gelu_exact_grad_scalar(bh);
    dX[b * two_d + d]     = dy * g;
    dX[b * two_d + D + d] = dy * a * gprime;
}
kernel void k_geglu_exact_backward_fp16(device const half* X  [[buffer(0)]],
                                        device const half* dY [[buffer(1)]],
                                        device half*       dX [[buffer(2)]],
                                        constant uint& B      [[buffer(3)]],
                                        constant uint& D      [[buffer(4)]],
                                        uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a       = float(X[b * two_d + d]);
    float bh      = float(X[b * two_d + D + d]);
    float dy      = float(dY[idx]);
    float g       = gelu_exact_scalar(bh);
    float gprime  = gelu_exact_grad_scalar(bh);
    dX[b * two_d + d]     = half(dy * g);
    dX[b * two_d + D + d] = half(dy * a * gprime);
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
DEF_PSO(pso_geglu_fp32,         @"k_geglu_forward_fp32")
DEF_PSO(pso_geglu_bwd_fp32,     @"k_geglu_backward_fp32")
DEF_PSO(pso_geglu_bwd_fp16,     @"k_geglu_backward_fp16")
DEF_PSO(pso_geglu_exact_fp32,     @"k_geglu_exact_forward_fp32")
DEF_PSO(pso_geglu_exact_fp16,     @"k_geglu_exact_forward_fp16")
DEF_PSO(pso_geglu_exact_bwd_fp32, @"k_geglu_exact_backward_fp32")
DEF_PSO(pso_geglu_exact_bwd_fp16, @"k_geglu_exact_backward_fp16")
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
        ::brotensor::metal_impl::submit(cmd);
    }
}

void launch_fp16_add_inplace(Tensor& y, const Tensor& x, uint32_t n) {
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

void launch_fp16_scale_inplace(Tensor& y, float s, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger oy = buffer_offset_for(y);
    launch_1d(pso_scale_inplace_fp16(), n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBytes:&s length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

} // namespace

void mul_inplace(Tensor& y, const Tensor& x) {
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

void geglu_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_fp16() : pso_geglu_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
    });
}

void geglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_backward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx  = buffer_for(X);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ox  = buffer_offset_for(X);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_bwd_fp16() : pso_geglu_bwd_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

void geglu_exact_forward(const Tensor& X, Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_exact_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_exact_fp16() : pso_geglu_exact_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
    });
}

void geglu_exact_backward(const Tensor& X, const Tensor& dY,
                          Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_exact_backward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(B) * static_cast<uint32_t>(D);
    if (total == 0) return;
    id<MTLBuffer> bx  = buffer_for(X);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ox  = buffer_offset_for(X);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    const uint32_t Bu = B, Du = D;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_geglu_exact_bwd_fp16() : pso_geglu_exact_bwd_fp32();
    launch_1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
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

// ─── cast: FP32 <-> FP16 dtype conversion ──────────────────────────────────

namespace {

NSString* const kCastSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_cast_f2h(device const float* s [[buffer(0)]],
                       device half*        d [[buffer(1)]],
                       constant uint&      n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    d[i] = half(s[i]);
}

kernel void k_cast_h2f(device const half* s [[buffer(0)]],
                       device float*      d [[buffer(1)]],
                       constant uint&     n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    d[i] = float(s[i]);
}
)msl";

id<MTLComputePipelineState> pso_cast_f2h() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_f2h"); });
    return pso;
}
id<MTLComputePipelineState> pso_cast_h2f() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_h2f"); });
    return pso;
}

} // namespace

void cast(const Tensor& src, Tensor& dst, Dtype out_dtype) {
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const uint32_t n = static_cast<uint32_t>(src.size());
    if (n == 0) return;
    id<MTLBuffer> bs = buffer_for(src);
    id<MTLBuffer> bd = buffer_for(dst);
    const NSUInteger os = buffer_offset_for(src);
    const NSUInteger od = buffer_offset_for(dst);

    if (src.dtype == out_dtype) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:bs sourceOffset:os
                        toBuffer:bd destinationOffset:od
                            size:src.bytes()];
            [blit endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    id<MTLComputePipelineState> pso;
    if (src.dtype == Dtype::FP32 && out_dtype == Dtype::FP16) {
        pso = pso_cast_f2h();
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::FP32) {
        pso = pso_cast_h2f();
    } else {
        throw std::runtime_error(
            "cast: unsupported dtype pair (Metal supports FP32<->FP16)");
    }
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bs offset:os atIndex:0];
        [enc setBuffer:bd offset:od atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
