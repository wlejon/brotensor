#pragma once

#import <Foundation/Foundation.h>

namespace brotensor::detail::metal {
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
// FP32 y += FP16 x: a half-precision branch re-entering an FP32 residual.
kernel void k_add_inplace_f32_f16(device float*      y [[buffer(0)]],
                                  device const half* x [[buffer(1)]],
                                  constant uint& n     [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] += float(x[i]);
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

kernel void k_add_row_bias_inplace_fp32(device float*       y    [[buffer(0)]],
                                        device const float* bias [[buffer(1)]],
                                        constant uint& D         [[buffer(2)]],
                                        constant uint& n         [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] += bias[i % D];
}
kernel void k_add_row_bias_inplace_fp16(device half*       y    [[buffer(0)]],
                                        device const half* bias [[buffer(1)]],
                                        constant uint& D        [[buffer(2)]],
                                        constant uint& n        [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = half(float(y[i]) + float(bias[i % D]));
}
kernel void k_add_row_bias_inplace_bf16(device bfloat*       y    [[buffer(0)]],
                                        device const bfloat* bias [[buffer(1)]],
                                        constant uint& D          [[buffer(2)]],
                                        constant uint& n          [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = bfloat(float(y[i]) + float(bias[i % D]));
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

} // namespace
} // namespace brotensor::detail::metal
