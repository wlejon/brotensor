// ─── CPU activation ops (CHUNK 2) ──────────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU activation kernels in
// src/cuda/elementwise.cu — kernel math reproduced verbatim, FP32 path only.
//
//   silu        — x * sigmoid(x)
//   gelu        — tanh-approximation GELU (PyTorch approximate="tanh")
//   gelu_exact  — erf-based GELU (PyTorch approximate="none")
//   quick_gelu  — x * sigmoid(1.702 * x) (OpenAI CLIP)
//
// Each has a matching backward op. Outputs are sized to mirror the GPU op
// (resize if shape/dtype differs); backward writes dX (overwrite, not
// accumulate — matches the GPU which writes dX[i] directly).

#include <brotensor/tensor.h>

#include <cmath>

namespace brotensor::detail::cpu {

namespace {

inline float silu_scalar(float v) {
    return v / (1.0f + std::exp(-v));
}

inline float silu_grad_scalar(float v) {
    // d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
    const float s = 1.0f / (1.0f + std::exp(-v));
    return s * (1.0f + v * (1.0f - s));
}

inline float gelu_tanh_scalar(float v) {
    // GELU with tanh approximation (matches PyTorch's approximate="tanh").
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(u));
}

inline float gelu_tanh_grad_scalar(float v) {
    // Derivative of gelu_tanh_scalar w.r.t. v.
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    const float t = std::tanh(u);
    const float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}

inline float gelu_exact_scalar(float v) {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
}

inline float gelu_exact_grad_scalar(float v) {
    // d/dx [0.5*x*(1+erf(x/√2))] = 0.5*(1+erf(x/√2)) + x*φ(x).
    constexpr float kInvSqrt2   = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f; // 1/sqrt(2π)
    const float cdf_term = 0.5f * (1.0f + std::erf(v * kInvSqrt2));
    const float pdf      = kInvSqrt2Pi * std::exp(-0.5f * v * v);
    return cdf_term + v * pdf;
}

inline float quick_gelu_scalar(float v) {
    // OpenAI CLIP's QuickGELU: x * sigmoid(1.702 * x).
    return v / (1.0f + std::exp(-1.702f * v));
}

inline float quick_gelu_grad_scalar(float v) {
    // d/dx [x * sigmoid(1.702*x)] = s + x * 1.702 * s * (1 - s).
    const float s = 1.0f / (1.0f + std::exp(-1.702f * v));
    return s + v * 1.702f * s * (1.0f - s);
}

} // namespace

// ─── silu ──────────────────────────────────────────────────────────────────

void silu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = silu_scalar(xp[i]);
}

void silu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] * silu_grad_scalar(xp[i]);
}

// ─── gelu (tanh approximation) ─────────────────────────────────────────────

void gelu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = gelu_tanh_scalar(xp[i]);
}

void gelu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] * gelu_tanh_grad_scalar(xp[i]);
}

// ─── gelu_exact (erf-based) ────────────────────────────────────────────────

void gelu_exact_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = gelu_exact_scalar(xp[i]);
}

void gelu_exact_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY,
                         ::brotensor::Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] * gelu_exact_grad_scalar(xp[i]);
}

// ─── quick_gelu ────────────────────────────────────────────────────────────

void quick_gelu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = quick_gelu_scalar(xp[i]);
}

void quick_gelu_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY,
                         ::brotensor::Tensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] * quick_gelu_grad_scalar(xp[i]);
}

} // namespace brotensor::detail::cpu
