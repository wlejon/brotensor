// ─── CPU GEGLU ops (CHUNK 2) ───────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GEGLU kernels in
// src/cuda/elementwise.cu — kernel math reproduced verbatim, FP32 path only.
//
//   X is (B, 2D); split along the last dim into A = X[:, :D] (the value half)
//   and B_half = X[:, D:] (the gate half).
//     forward:  Y(B, D)  = A * gelu(B_half)
//     backward: dX(B,2D): dX[:, :D] = dY * gelu(B_half)
//                         dX[:, D:] = dY * A * gelu'(B_half)
//
// Two variants: tanh-approximation GELU and exact (erf-based) GELU. Backward
// overwrites dX (the GPU kernel writes both halves directly).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>

namespace brotensor::detail::cpu {

namespace {

inline float gelu_tanh_scalar(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(u));
}

// Backward needs both gelu(v) and gelu'(v) for the same v; both are
// algebraically derivable from a single std::tanh evaluation, so compute it
// once here instead of calling gelu_tanh_scalar plus a separate gradient
// function (which would redo the tanh).
inline void gelu_tanh_value_grad(float v, float& g, float& gprime) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    const float t = std::tanh(u);
    const float dudx = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    g = 0.5f * v * (1.0f + t);
    gprime = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudx;
}

inline float gelu_exact_scalar(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
}

// Backward needs both gelu(v) and gelu'(v) for the same v; both share the
// same std::erf evaluation (cdf_term), so compute it once here instead of
// calling gelu_exact_scalar plus a separate gradient function.
inline void gelu_exact_value_grad(float v, float& g, float& gprime) {
    constexpr float kInvSqrt2   = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    const float cdf_term = 0.5f * (1.0f + std::erf(v * kInvSqrt2));
    g = v * cdf_term;
    const float pdf = kInvSqrt2Pi * std::exp(-0.5f * v * v);
    gprime = cdf_term + v * pdf;
}

template <typename GeluFn>
void geglu_forward_impl(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                        const char* op, GeluFn gelu) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error(std::string(op) + ": X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D) Y.resize(B, D);
    const int total = B * D;
    if (total == 0) return;
    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();
    const int two_d = 2 * D;
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a      = Xp[b * two_d + d];
            const float gv_raw = Xp[b * two_d + D + d];
            Yp[b * D + d] = a * gelu(gv_raw);
        }
    }
}

template <typename GeluValueGradFn>
void geglu_backward_impl(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& dY,
                         ::brotensor::Tensor& dX,
                         const char* op, GeluValueGradFn gelu_value_grad) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error(std::string(op) + ": X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D) dX.resize(B, 2 * D);
    const int total = B * D;
    if (total == 0) return;
    const float* Xp = X.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    const int two_d = 2 * D;
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float a  = Xp[b * two_d + d];
            const float bh = Xp[b * two_d + D + d];
            const float dy = dYp[b * D + d];
            float g, gprime;
            gelu_value_grad(bh, g, gprime);
            dXp[b * two_d + d]     = dy * g;
            dXp[b * two_d + D + d] = dy * a * gprime;
        }
    }
}

} // namespace

void geglu_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    geglu_forward_impl(X, Y, "geglu_forward", gelu_tanh_scalar);
}

void geglu_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,
                    ::brotensor::Tensor& dX) {
    geglu_backward_impl(X, dY, dX, "geglu_backward", gelu_tanh_value_grad);
}

void geglu_exact_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    geglu_forward_impl(X, Y, "geglu_exact_forward", gelu_exact_scalar);
}

void geglu_exact_backward(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dX) {
    geglu_backward_impl(X, dY, dX, "geglu_exact_backward", gelu_exact_value_grad);
}

} // namespace brotensor::detail::cpu
