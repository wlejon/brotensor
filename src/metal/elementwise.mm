#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::dispatch1d_sync;
using metal_impl::new_command_buffer;

// relu/tanh/sigmoid dispatch on dtype like the silu/gelu family below —
// their wrappers live after the kActivationSrc DEF_PSOs.

// add_inplace forward decls (FP16 dispatcher lives further down with the
// other FP16-extension kernels).
namespace { void launch_fp16_add_inplace(Tensor&, const Tensor&, uint32_t);
             void launch_fp16_scale_inplace(Tensor&, float, uint32_t);
             void launch_bf16_add_inplace(Tensor&, const Tensor&, uint32_t);
             void launch_bf16_scale_inplace(Tensor&, float, uint32_t); }

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
    if (y.dtype == Dtype::BF16) {
        if (x.dtype != Dtype::BF16) {
            throw std::runtime_error("add_inplace_gpu: dtype mismatch");
        }
        launch_bf16_add_inplace(y, x, n);
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

// Forward decls for FP16/BF16 PSOs defined later in this TU.
namespace {
id<MTLComputePipelineState> pso_add_scalar_fp16();
id<MTLComputePipelineState> pso_add_scalar_bf16();
id<MTLComputePipelineState> pso_scale_fp16();
id<MTLComputePipelineState> pso_clamp_fp32();
id<MTLComputePipelineState> pso_clamp_fp16();
id<MTLComputePipelineState> pso_clamp_bf16();
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
    if (y.dtype == Dtype::BF16) {
        dispatch_scalar_inplace(pso_add_scalar_bf16(), by, off_y, s, n);
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
    if (y.dtype == Dtype::BF16) {
        launch_bf16_scale_inplace(y, s, n);
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
        (y.dtype == Dtype::BF16) ? pso_clamp_bf16()
      : (y.dtype == Dtype::FP16) ? pso_clamp_fp16()
      : pso_clamp_fp32();
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
    if (mask.rows != K || mask.cols != 1 || mask.dtype != Dtype::FP32) mask.resize(K, 1, Dtype::FP32);
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
// Metal's builtin tanh() returns NaN for x ≳ 45 (it computes exp(2x)/(exp(2x)+1)
// internally and exp(2x) overflows FP32). Clamp to ±9 where tanh saturates to
// ±1 within FP32 epsilon. Used by every tanh site below.
inline float safe_tanh(float x) { return tanh(clamp(x, -9.0f, 9.0f)); }
inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + safe_tanh(u));
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
    float t = safe_tanh(u);
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
kernel void k_silu_backward_bf16(device const bfloat* x  [[buffer(0)]],
                                 device const bfloat* dY [[buffer(1)]],
                                 device bfloat*       dX [[buffer(2)]],
                                 constant uint& n        [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = bfloat(float(dY[i]) * silu_grad_scalar(float(x[i])));
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
kernel void k_gelu_backward_bf16(device const bfloat* x  [[buffer(0)]],
                                 device const bfloat* dY [[buffer(1)]],
                                 device bfloat*       dX [[buffer(2)]],
                                 constant uint& n        [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = bfloat(float(dY[i]) * gelu_tanh_grad_scalar(float(x[i])));
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
kernel void k_quick_gelu_backward_bf16(device const bfloat* x  [[buffer(0)]],
                                       device const bfloat* dY [[buffer(1)]],
                                       device bfloat*       dX [[buffer(2)]],
                                       constant uint& n        [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = bfloat(float(dY[i]) * quick_gelu_grad_scalar(float(x[i])));
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
kernel void k_gelu_exact_forward_bf16(device const bfloat* x [[buffer(0)]],
                                      device bfloat*       y [[buffer(1)]],
                                      constant uint& n       [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(gelu_exact_scalar(float(x[i])));
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
kernel void k_gelu_exact_backward_bf16(device const bfloat* x  [[buffer(0)]],
                                       device const bfloat* dY [[buffer(1)]],
                                       device bfloat*       dX [[buffer(2)]],
                                       constant uint& n        [[buffer(3)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dX[i] = bfloat(float(dY[i]) * gelu_exact_grad_scalar(float(x[i])));
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
kernel void k_silu_forward_bf16(device const bfloat* x [[buffer(0)]],
                                device bfloat*       y [[buffer(1)]],
                                constant uint& n       [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(silu_scalar(float(x[i])));
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
kernel void k_gelu_forward_bf16(device const bfloat* x [[buffer(0)]],
                                device bfloat*       y [[buffer(1)]],
                                constant uint& n       [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(gelu_tanh_scalar(float(x[i])));
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
kernel void k_quick_gelu_forward_bf16(device const bfloat* x [[buffer(0)]],
                                      device bfloat*       y [[buffer(1)]],
                                      constant uint& n       [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(quick_gelu_scalar(float(x[i])));
}
kernel void k_add_scalar_inplace_fp16(device half* y [[buffer(0)]],
                                      constant float& s [[buffer(1)]],
                                      constant uint& n  [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) + s);
}
kernel void k_add_scalar_inplace_bf16(device bfloat* y [[buffer(0)]],
                                      constant float& s [[buffer(1)]],
                                      constant uint& n  [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(float(y[i]) + s);
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
kernel void k_clamp_bf16(device bfloat* y [[buffer(0)]],
                         constant float& lo [[buffer(1)]],
                         constant float& hi [[buffer(2)]],
                         constant uint&  n  [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = float(y[i]);
    v = max(v, lo);
    v = min(v, hi);
    y[i] = bfloat(v);
}

// ── relu / tanh / sigmoid (all dtypes; math in FP32) ─────────────────────
// tanh/sigmoid backward take the FORWARD OUTPUT y as first argument, same as
// kernels.mm and CUDA.

inline float relu_scalar(float v)    { return v > 0.0f ? v : 0.0f; }
inline float sigmoid_scalar(float v) { return 1.0f / (1.0f + exp(-v)); }

#define EW_UNARY(NAME, T, FN)                                                 \
kernel void NAME(device const T* x [[buffer(0)]],                             \
                 device T*       y [[buffer(1)]],                             \
                 constant uint& n  [[buffer(2)]],                             \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= n) return;                                                       \
    y[i] = T(FN(float(x[i])));                                                \
}

#define EW_UNARY_BWD(NAME, T, GRAD_EXPR)                                      \
kernel void NAME(device const T* a  [[buffer(0)]],                            \
                 device const T* dY [[buffer(1)]],                            \
                 device T*       dX [[buffer(2)]],                            \
                 constant uint& n   [[buffer(3)]],                            \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= n) return;                                                       \
    float av = float(a[i]);                                                   \
    float dy = float(dY[i]);                                                  \
    dX[i] = T(GRAD_EXPR);                                                     \
}

EW_UNARY(k_relu_forward_fp32, float,  relu_scalar)
EW_UNARY(k_relu_forward_fp16, half,   relu_scalar)
EW_UNARY(k_relu_forward_bf16, bfloat, relu_scalar)
EW_UNARY(k_tanh_forward_fp32, float,  safe_tanh)
EW_UNARY(k_tanh_forward_fp16, half,   safe_tanh)
EW_UNARY(k_tanh_forward_bf16, bfloat, safe_tanh)
EW_UNARY(k_sigmoid_forward_fp32, float,  sigmoid_scalar)
EW_UNARY(k_sigmoid_forward_fp16, half,   sigmoid_scalar)
EW_UNARY(k_sigmoid_forward_bf16, bfloat, sigmoid_scalar)

EW_UNARY_BWD(k_relu_backward_fp32, float,  av > 0.0f ? dy : 0.0f)
EW_UNARY_BWD(k_relu_backward_fp16, half,   av > 0.0f ? dy : 0.0f)
EW_UNARY_BWD(k_relu_backward_bf16, bfloat, av > 0.0f ? dy : 0.0f)
EW_UNARY_BWD(k_tanh_backward_fp32, float,  dy * (1.0f - av * av))
EW_UNARY_BWD(k_tanh_backward_fp16, half,   dy * (1.0f - av * av))
EW_UNARY_BWD(k_tanh_backward_bf16, bfloat, dy * (1.0f - av * av))
EW_UNARY_BWD(k_sigmoid_backward_fp32, float,  dy * av * (1.0f - av))
EW_UNARY_BWD(k_sigmoid_backward_fp16, half,   dy * av * (1.0f - av))
EW_UNARY_BWD(k_sigmoid_backward_bf16, bfloat, dy * av * (1.0f - av))
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
DEF_PSO(pso_silu_bf16, @"k_silu_forward_bf16")
DEF_PSO(pso_gelu_fp32, @"k_gelu_forward_fp32")
DEF_PSO(pso_gelu_fp16, @"k_gelu_forward_fp16")
DEF_PSO(pso_gelu_bf16, @"k_gelu_forward_bf16")
DEF_PSO(pso_quick_gelu_fp32, @"k_quick_gelu_forward_fp32")
DEF_PSO(pso_quick_gelu_fp16, @"k_quick_gelu_forward_fp16")
DEF_PSO(pso_quick_gelu_bf16, @"k_quick_gelu_forward_bf16")
DEF_PSO(pso_silu_bwd_fp32,       @"k_silu_backward_fp32")
DEF_PSO(pso_silu_bwd_fp16,       @"k_silu_backward_fp16")
DEF_PSO(pso_silu_bwd_bf16,       @"k_silu_backward_bf16")
DEF_PSO(pso_gelu_bwd_fp32,       @"k_gelu_backward_fp32")
DEF_PSO(pso_gelu_bwd_fp16,       @"k_gelu_backward_fp16")
DEF_PSO(pso_gelu_bwd_bf16,       @"k_gelu_backward_bf16")
DEF_PSO(pso_quick_gelu_bwd_fp32, @"k_quick_gelu_backward_fp32")
DEF_PSO(pso_quick_gelu_bwd_fp16, @"k_quick_gelu_backward_fp16")
DEF_PSO(pso_quick_gelu_bwd_bf16, @"k_quick_gelu_backward_bf16")
DEF_PSO(pso_gelu_exact_fp32,     @"k_gelu_exact_forward_fp32")
DEF_PSO(pso_gelu_exact_fp16,     @"k_gelu_exact_forward_fp16")
DEF_PSO(pso_gelu_exact_bf16,     @"k_gelu_exact_forward_bf16")
DEF_PSO(pso_gelu_exact_bwd_fp32, @"k_gelu_exact_backward_fp32")
DEF_PSO(pso_gelu_exact_bwd_fp16, @"k_gelu_exact_backward_fp16")
DEF_PSO(pso_gelu_exact_bwd_bf16, @"k_gelu_exact_backward_bf16")
DEF_PSO(pso_add_scalar_fp16,  @"k_add_scalar_inplace_fp16")
DEF_PSO(pso_add_scalar_bf16,  @"k_add_scalar_inplace_bf16")
DEF_PSO(pso_scale_fp16,       @"k_scale_inplace_fp16")
DEF_PSO(pso_clamp_fp32,       @"k_clamp_fp32")
DEF_PSO(pso_clamp_fp16,       @"k_clamp_fp16")
DEF_PSO(pso_clamp_bf16,       @"k_clamp_bf16")
DEF_PSO(pso_relu_fp32,        @"k_relu_forward_fp32")
DEF_PSO(pso_relu_fp16,        @"k_relu_forward_fp16")
DEF_PSO(pso_relu_bf16,        @"k_relu_forward_bf16")
DEF_PSO(pso_tanh_fp32,        @"k_tanh_forward_fp32")
DEF_PSO(pso_tanh_fp16,        @"k_tanh_forward_fp16")
DEF_PSO(pso_tanh_bf16,        @"k_tanh_forward_bf16")
DEF_PSO(pso_sigmoid_fp32,     @"k_sigmoid_forward_fp32")
DEF_PSO(pso_sigmoid_fp16,     @"k_sigmoid_forward_fp16")
DEF_PSO(pso_sigmoid_bf16,     @"k_sigmoid_forward_bf16")
DEF_PSO(pso_relu_bwd_fp32,    @"k_relu_backward_fp32")
DEF_PSO(pso_relu_bwd_fp16,    @"k_relu_backward_fp16")
DEF_PSO(pso_relu_bwd_bf16,    @"k_relu_backward_bf16")
DEF_PSO(pso_tanh_bwd_fp32,    @"k_tanh_backward_fp32")
DEF_PSO(pso_tanh_bwd_fp16,    @"k_tanh_backward_fp16")
DEF_PSO(pso_tanh_bwd_bf16,    @"k_tanh_backward_bf16")
DEF_PSO(pso_sigmoid_bwd_fp32, @"k_sigmoid_backward_fp32")
DEF_PSO(pso_sigmoid_bwd_fp16, @"k_sigmoid_backward_fp16")
DEF_PSO(pso_sigmoid_bwd_bf16, @"k_sigmoid_backward_bf16")
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
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_silu_fp16()
      : x.dtype == Dtype::BF16 ? pso_silu_bf16()
      : pso_silu_fp32(), x, y);
}
void gelu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_gelu_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_bf16()
      : pso_gelu_fp32(), x, y);
}
void quick_gelu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_quick_gelu_fp16()
      : x.dtype == Dtype::BF16 ? pso_quick_gelu_bf16()
      : pso_quick_gelu_fp32(), x, y);
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
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_silu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_silu_bwd_bf16()
      : pso_silu_bwd_fp32(), x, dY, dX);
}
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_gelu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_bwd_bf16()
      : pso_gelu_bwd_fp32(), x, dY, dX);
}
void quick_gelu_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_quick_gelu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_quick_gelu_bwd_bf16()
      : pso_quick_gelu_bwd_fp32(), x, dY, dX);
}
void gelu_exact_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_gelu_exact_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_exact_bf16()
      : pso_gelu_exact_fp32(), x, y);
}
void gelu_exact_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_gelu_exact_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_gelu_exact_bwd_bf16()
      : pso_gelu_exact_bwd_fp32(), x, dY, dX);
}
void relu_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_relu_fp16()
      : x.dtype == Dtype::BF16 ? pso_relu_bf16()
      : pso_relu_fp32(), x, y);
}
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        x.dtype == Dtype::FP16 ? pso_relu_bwd_fp16()
      : x.dtype == Dtype::BF16 ? pso_relu_bwd_bf16()
      : pso_relu_bwd_fp32(), x, dY, dX);
}
void tanh_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_tanh_fp16()
      : x.dtype == Dtype::BF16 ? pso_tanh_bf16()
      : pso_tanh_fp32(), x, y);
}
// tanh/sigmoid backward take the forward OUTPUT y, not x (same as CUDA).
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        y.dtype == Dtype::FP16 ? pso_tanh_bwd_fp16()
      : y.dtype == Dtype::BF16 ? pso_tanh_bwd_bf16()
      : pso_tanh_bwd_fp32(), y, dY, dX);
}
void sigmoid_forward(const Tensor& x, Tensor& y) {
    launch_activation_unary(
        x.dtype == Dtype::FP16 ? pso_sigmoid_fp16()
      : x.dtype == Dtype::BF16 ? pso_sigmoid_bf16()
      : pso_sigmoid_fp32(), x, y);
}
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    launch_activation_bwd(
        y.dtype == Dtype::FP16 ? pso_sigmoid_bwd_fp16()
      : y.dtype == Dtype::BF16 ? pso_sigmoid_bwd_bf16()
      : pso_sigmoid_bwd_fp32(), y, dY, dX);
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

// Metal's builtin tanh() returns NaN for x ≳ 45 (it computes exp(2x)/(exp(2x)+1)
// internally and exp(2x) overflows FP32). Clamp to ±9 where tanh saturates to
// ±1 within FP32 epsilon. Used by every tanh site below.
inline float safe_tanh(float x) { return tanh(clamp(x, -9.0f, 9.0f)); }
inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + safe_tanh(u));
}
inline float gelu_tanh_grad_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    float t = safe_tanh(u);
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
kernel void k_add_inplace_bf16(device bfloat*       y [[buffer(0)]],
                               device const bfloat* x [[buffer(1)]],
                               constant uint& n       [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(float(y[i]) + float(x[i]));
}
kernel void k_scale_inplace_fp16(device half*       y [[buffer(0)]],
                                 constant float& s    [[buffer(1)]],
                                 constant uint& n     [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) * s);
}
kernel void k_scale_inplace_bf16(device bfloat*       y [[buffer(0)]],
                                 constant float& s      [[buffer(1)]],
                                 constant uint& n       [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(float(y[i]) * s);
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
kernel void k_mul_inplace_bf16(device bfloat*       y [[buffer(0)]],
                               device const bfloat* x [[buffer(1)]],
                               constant uint& n       [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(float(y[i]) * float(x[i]));
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
kernel void k_geglu_forward_bf16(device const bfloat* X [[buffer(0)]],
                                 device bfloat*       Y [[buffer(1)]],
                                 constant uint& B       [[buffer(2)]],
                                 constant uint& D       [[buffer(3)]],
                                 uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a = float(X[b * two_d + d]);
    float gv_raw = float(X[b * two_d + D + d]);
    Y[idx] = bfloat(a * gelu_tanh_scalar(gv_raw));
}
kernel void k_geglu_backward_bf16(device const bfloat* X  [[buffer(0)]],
                                  device const bfloat* dY [[buffer(1)]],
                                  device bfloat*       dX [[buffer(2)]],
                                  constant uint& B        [[buffer(3)]],
                                  constant uint& D        [[buffer(4)]],
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
    dX[b * two_d + d]     = bfloat(dy * g);
    dX[b * two_d + D + d] = bfloat(dy * a * gprime);
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
kernel void k_geglu_exact_forward_bf16(device const bfloat* X [[buffer(0)]],
                                       device bfloat*       Y [[buffer(1)]],
                                       constant uint& B       [[buffer(2)]],
                                       constant uint& D       [[buffer(3)]],
                                       uint idx [[thread_position_in_grid]]) {
    uint total = B * D;
    if (idx >= total) return;
    uint b = idx / D;
    uint d = idx % D;
    uint two_d = 2u * D;
    float a = float(X[b * two_d + d]);
    float gv_raw = float(X[b * two_d + D + d]);
    Y[idx] = bfloat(a * gelu_exact_scalar(gv_raw));
}
kernel void k_geglu_exact_backward_bf16(device const bfloat* X  [[buffer(0)]],
                                        device const bfloat* dY [[buffer(1)]],
                                        device bfloat*       dX [[buffer(2)]],
                                        constant uint& B        [[buffer(3)]],
                                        constant uint& D        [[buffer(4)]],
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
    dX[b * two_d + d]     = bfloat(dy * g);
    dX[b * two_d + D + d] = bfloat(dy * a * gprime);
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
DEF_PSO(pso_add_inplace_bf16,   @"k_add_inplace_bf16")
DEF_PSO(pso_scale_inplace_fp16, @"k_scale_inplace_fp16")
DEF_PSO(pso_scale_inplace_bf16, @"k_scale_inplace_bf16")
DEF_PSO(pso_mul_inplace_fp32,   @"k_mul_inplace_fp32")
DEF_PSO(pso_mul_inplace_fp16,   @"k_mul_inplace_fp16")
DEF_PSO(pso_mul_inplace_bf16,   @"k_mul_inplace_bf16")
DEF_PSO(pso_geglu_fp16,         @"k_geglu_forward_fp16")
DEF_PSO(pso_geglu_fp32,         @"k_geglu_forward_fp32")
DEF_PSO(pso_geglu_bf16,         @"k_geglu_forward_bf16")
DEF_PSO(pso_geglu_bwd_fp32,     @"k_geglu_backward_fp32")
DEF_PSO(pso_geglu_bwd_fp16,     @"k_geglu_backward_fp16")
DEF_PSO(pso_geglu_bwd_bf16,     @"k_geglu_backward_bf16")
DEF_PSO(pso_geglu_exact_fp32,     @"k_geglu_exact_forward_fp32")
DEF_PSO(pso_geglu_exact_fp16,     @"k_geglu_exact_forward_fp16")
DEF_PSO(pso_geglu_exact_bf16,     @"k_geglu_exact_forward_bf16")
DEF_PSO(pso_geglu_exact_bwd_fp32, @"k_geglu_exact_backward_fp32")
DEF_PSO(pso_geglu_exact_bwd_fp16, @"k_geglu_exact_backward_fp16")
DEF_PSO(pso_geglu_exact_bwd_bf16, @"k_geglu_exact_backward_bf16")
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

void launch_bf16_add_inplace(Tensor& y, const Tensor& x, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso_add_inplace_bf16(), n, ^(id<MTLComputeCommandEncoder> enc) {
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

void launch_bf16_scale_inplace(Tensor& y, float s, uint32_t n) {
    id<MTLBuffer> by = buffer_for(y);
    const NSUInteger oy = buffer_offset_for(y);
    launch_1d(pso_scale_inplace_bf16(), n, ^(id<MTLComputeCommandEncoder> enc) {
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
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_mul_inplace_fp16()
      : (y.dtype == Dtype::BF16) ? pso_mul_inplace_bf16()
      : pso_mul_inplace_fp32();
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
        (X.dtype == Dtype::FP16) ? pso_geglu_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_bf16()
      : pso_geglu_fp32();
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
        (X.dtype == Dtype::FP16) ? pso_geglu_bwd_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_bwd_bf16()
      : pso_geglu_bwd_fp32();
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
        (X.dtype == Dtype::FP16) ? pso_geglu_exact_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_exact_bf16()
      : pso_geglu_exact_fp32();
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
        (X.dtype == Dtype::FP16) ? pso_geglu_exact_bwd_fp16()
      : (X.dtype == Dtype::BF16) ? pso_geglu_exact_bwd_bf16()
      : pso_geglu_exact_bwd_fp32();
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

kernel void k_cast_f2b(device const float* s [[buffer(0)]],
                       device bfloat*      d [[buffer(1)]],
                       constant uint&      n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    d[i] = bfloat(s[i]);
}

kernel void k_cast_b2f(device const bfloat* s [[buffer(0)]],
                       device float*        d [[buffer(1)]],
                       constant uint&       n [[buffer(2)]],
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
id<MTLComputePipelineState> pso_cast_f2b() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_f2b"); });
    return pso;
}
id<MTLComputePipelineState> pso_cast_b2f() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kCastSrc, @"k_cast_b2f"); });
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

    // Ensure BF16 is handled properly (src.bytes() uses dtype_size_bytes).
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
    } else if (src.dtype == Dtype::FP32 && out_dtype == Dtype::BF16) {
        pso = pso_cast_f2b();
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP32) {
        pso = pso_cast_b2f();
    } else {
        throw std::runtime_error(
            "cast: unsupported dtype pair (Metal supports FP32<->FP16, FP32<->BF16)");
    }
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bs offset:os atIndex:0];
        [enc setBuffer:bd offset:od atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
    });
}

// ─── axpby_inplace / add_channel_bias_inplace / threshold_u8 ──────────────
//
// CUDA-only until now (see src/metal/register.mm's parity-gap comment); these
// three are plain elementwise/broadcast kernels, so they follow the same
// compile_pipeline + launch_1d pattern as the FP16-extension block above.

namespace {

NSString* const kExtra2Src = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_axpby_inplace_fp32(device float*       y [[buffer(0)]],
                                 device const float* x [[buffer(1)]],
                                 constant float& a     [[buffer(2)]],
                                 constant float& b     [[buffer(3)]],
                                 constant uint&  n     [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = a * y[i] + b * x[i];
}
kernel void k_axpby_inplace_fp16(device half*       y [[buffer(0)]],
                                 device const half* x [[buffer(1)]],
                                 constant float& a    [[buffer(2)]],
                                 constant float& b    [[buffer(3)]],
                                 constant uint&  n    [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(a * float(y[i]) + b * float(x[i]));
}
kernel void k_axpby_inplace_bf16(device bfloat*       y [[buffer(0)]],
                                 device const bfloat* x [[buffer(1)]],
                                 constant float& a      [[buffer(2)]],
                                 constant float& b      [[buffer(3)]],
                                 constant uint&  n      [[buffer(4)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(a * float(y[i]) + b * float(x[i]));
}

kernel void k_add_channel_bias_inplace_fp32(device float*       y    [[buffer(0)]],
                                            device const float* bias [[buffer(1)]],
                                            constant uint& L         [[buffer(2)]],
                                            constant uint& n         [[buffer(3)]],
                                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] += bias[i / L];
}
kernel void k_add_channel_bias_inplace_fp16(device half*       y    [[buffer(0)]],
                                            device const half* bias [[buffer(1)]],
                                            constant uint& L        [[buffer(2)]],
                                            constant uint& n        [[buffer(3)]],
                                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) + float(bias[i / L]));
}
kernel void k_add_channel_bias_inplace_bf16(device bfloat*       y    [[buffer(0)]],
                                            device const bfloat* bias [[buffer(1)]],
                                            constant uint& L          [[buffer(2)]],
                                            constant uint& n          [[buffer(3)]],
                                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(float(y[i]) + float(bias[i / L]));
}

kernel void k_threshold_u8_fp32(device const float* x [[buffer(0)]],
                                constant float& t     [[buffer(1)]],
                                device char*    y     [[buffer(2)]],
                                constant uint&  n     [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = (x[i] > t) ? char(1) : char(0);
}
kernel void k_threshold_u8_fp16(device const half* x [[buffer(0)]],
                                constant float& t    [[buffer(1)]],
                                device char*    y    [[buffer(2)]],
                                constant uint&  n    [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = (float(x[i]) > t) ? char(1) : char(0);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kExtra2Src, FN); }); \
        return pso; \
    }
DEF_PSO(pso_axpby_fp32, @"k_axpby_inplace_fp32")
DEF_PSO(pso_axpby_fp16, @"k_axpby_inplace_fp16")
DEF_PSO(pso_axpby_bf16, @"k_axpby_inplace_bf16")
DEF_PSO(pso_add_channel_bias_fp32, @"k_add_channel_bias_inplace_fp32")
DEF_PSO(pso_add_channel_bias_fp16, @"k_add_channel_bias_inplace_fp16")
DEF_PSO(pso_add_channel_bias_bf16, @"k_add_channel_bias_inplace_bf16")
DEF_PSO(pso_threshold_u8_fp32, @"k_threshold_u8_fp32")
DEF_PSO(pso_threshold_u8_fp16, @"k_threshold_u8_fp16")
#undef DEF_PSO

} // namespace

void axpby_inplace(Tensor& y, const Tensor& x, float a, float b) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("axpby_inplace: shape/dtype mismatch");
    }
    const uint32_t n = static_cast<uint32_t>(y.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_axpby_fp16()
      : (y.dtype == Dtype::BF16) ? pso_axpby_bf16()
      : pso_axpby_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bx = buffer_for(x);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ox = buffer_offset_for(x);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBytes:&a length:sizeof(float) atIndex:2];
        [enc setBytes:&b length:sizeof(float) atIndex:3];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:4];
    });
}

void add_channel_bias_inplace(Tensor& y, const Tensor& bias, int C, int L) {
    const uint32_t n = static_cast<uint32_t>(C) * static_cast<uint32_t>(L);
    if (n == 0) return;
    if (y.dtype != bias.dtype) {
        throw std::runtime_error("add_channel_bias_inplace: dtype mismatch");
    }
    if (static_cast<uint32_t>(y.size()) != n) {
        throw std::runtime_error("add_channel_bias_inplace: y size != C*L");
    }
    id<MTLComputePipelineState> pso =
        (y.dtype == Dtype::FP16) ? pso_add_channel_bias_fp16()
      : (y.dtype == Dtype::BF16) ? pso_add_channel_bias_bf16()
      : pso_add_channel_bias_fp32();
    id<MTLBuffer> by = buffer_for(y);
    id<MTLBuffer> bb = buffer_for(bias);
    const NSUInteger oy = buffer_offset_for(y);
    const NSUInteger ob = buffer_offset_for(bias);
    const uint32_t Lu = static_cast<uint32_t>(L);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:by offset:oy atIndex:0];
        [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&n  length:sizeof(uint32_t) atIndex:3];
    });
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("threshold_u8: X must be FP32 or FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::INT8) {
        Y.resize(X.rows, X.cols, Dtype::INT8);
    }
    const uint32_t n = static_cast<uint32_t>(X.size());
    if (n == 0) return;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_threshold_u8_fp16() : pso_threshold_u8_fp32();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    launch_1d(pso, n, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBytes:&t length:sizeof(float) atIndex:1];
        [enc setBuffer:by offset:oy atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
    });
}

} // namespace brotensor::detail::metal
