// ─── CPU log / exp / round elementwise ops (CHUNK 6, family G) ─────────────
//
// FP32 scalar host implementations, modelled on silu_forward/backward in
// activations.cpp. CPU is FP32-only.
//
//   log_forward / log_backward   y = log(x);   dX = dY / x
//   exp_forward / exp_backward   y = exp(x);   dX = dY * exp(x)
//   round_forward                y = round-half-to-even(x)  (torch.round)
//   round_backward               straight-through estimator: dX = dY
//
// log_forward / log_backward: the caller owns the x > 0 precondition. These
// ops do NOT guard the input — for x <= 0 they return the IEEE result
// (log(0) = -inf, log(<0) = NaN; 1/x for the backward) so a mis-clamped
// pipeline fails loudly. No floor is applied.
//
// All elementwise: the output is resized + dtype-set to match the input; the
// input and output may alias. None of these ops has a learnable parameter, so
// every backward OVERWRITES dX (it does not accumulate).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " +
                                 name + " must be FP32 (CPU backend is "
                                 "FP32-only)");
    }
}

} // namespace

// ─── log ───────────────────────────────────────────────────────────────────

void log_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    check_fp32(x, "log_forward", "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::log(xp[i]);
}

void log_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    check_fp32(x, "log_backward", "x");
    check_fp32(dY, "log_backward", "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != Dtype::FP32) {
        dX.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] / xp[i];
}

// ─── exp ───────────────────────────────────────────────────────────────────

void exp_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    check_fp32(x, "exp_forward", "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::exp(xp[i]);
}

void exp_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    check_fp32(x, "exp_backward", "x");
    check_fp32(dY, "exp_backward", "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != Dtype::FP32) {
        dX.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    // Read all inputs before writing so an in-place dX==dY alias is safe.
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] * std::exp(xp[i]);
}

// ─── round ─────────────────────────────────────────────────────────────────

void round_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    check_fp32(x, "round_forward", "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    // std::nearbyint = round-half-to-even under the default FE_TONEAREST
    // rounding mode — matches torch.round / numpy.round.
    for (int i = 0; i < n; ++i) yp[i] = std::nearbyint(xp[i]);
}

void round_backward(const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX) {
    check_fp32(dY, "round_backward", "dY");
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != Dtype::FP32) {
        dX.resize(dY.rows, dY.cols, Dtype::FP32);
    }
    const int n = dY.size();
    if (n == 0) return;
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    // Straight-through estimator: round() has zero gradient a.e. and is
    // non-differentiable at the half-integers, so we pass dY straight
    // through unchanged (identity) to keep gradients flowing.
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i];
}

} // namespace brotensor::detail::cpu
