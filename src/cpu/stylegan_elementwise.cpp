// ─── CPU StyleGAN3 synthesis-input primitives ──────────────────────────────
//
// FP32 scalar host implementations. CPU is FP32-only.
//
//   sin/cos     Fourier features for SynthesisInput (sin(2π·…)).
//   rsqrt       reciprocal sqrt backing modulation-demod / pixel-norm.
//   pixel_norm  RMS-over-channel normalisation for the mapping network.
//
// sin/cos/rsqrt are elementwise: outputs resized + dtype-set to match the
// input, x/y and dX/dY may alias, and the backward OVERWRITES dX (no learnable
// parameters). pixel_norm operates per row over the trailing (cols) axis.
//
// rsqrt: the caller owns the x > 0 precondition (no guard — rsqrt(0)=+inf,
// rsqrt(<0)=NaN), matching log/exp in log_exp_round.cpp.

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

// ─── sin ─────────────────────────────────────────────────────────────────────

void sin_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    check_fp32(x, "sin_forward", "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::sin(xp[i]);
}

void sin_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    check_fp32(x, "sin_backward", "x");
    check_fp32(dY, "sin_backward", "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != Dtype::FP32) {
        dX.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    for (int i = 0; i < n; ++i) dxp[i] = dyp[i] * std::cos(xp[i]);
}

// ─── cos ─────────────────────────────────────────────────────────────────────

void cos_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    check_fp32(x, "cos_forward", "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::cos(xp[i]);
}

void cos_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    check_fp32(x, "cos_backward", "x");
    check_fp32(dY, "cos_backward", "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != Dtype::FP32) {
        dX.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    for (int i = 0; i < n; ++i) dxp[i] = -dyp[i] * std::sin(xp[i]);
}

// ─── rsqrt ───────────────────────────────────────────────────────────────────

void rsqrt_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    check_fp32(x, "rsqrt_forward", "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = 1.0f / std::sqrt(xp[i]);
}

void rsqrt_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                    ::brotensor::Tensor& dX) {
    check_fp32(y, "rsqrt_backward", "y");
    check_fp32(dY, "rsqrt_backward", "dY");
    if (dX.rows != y.rows || dX.cols != y.cols || dX.dtype != Dtype::FP32) {
        dX.resize(y.rows, y.cols, Dtype::FP32);
    }
    const int n = y.size();
    if (n == 0) return;
    const float* yp = y.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    // y = x^{-1/2}  ⇒  dy/dx = -1/2 x^{-3/2} = -1/2 y^3.
    for (int i = 0; i < n; ++i) dxp[i] = -0.5f * dyp[i] * yp[i] * yp[i] * yp[i];
}

// ─── pixel_norm ──────────────────────────────────────────────────────────────

void pixel_norm_forward(const ::brotensor::Tensor& X, float eps,
                        ::brotensor::Tensor& Y) {
    check_fp32(X, "pixel_norm_forward", "X");
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::FP32) {
        Y.resize(X.rows, X.cols, Dtype::FP32);
    }
    const int R = X.rows, C = X.cols;
    if (R == 0 || C == 0) return;
    const float* xp = X.host_f32();
    float* yp = Y.host_f32_mut();
    for (int r = 0; r < R; ++r) {
        const float* xr = xp + static_cast<size_t>(r) * C;
        float* yr = yp + static_cast<size_t>(r) * C;
        float ss = 0.0f;
        for (int c = 0; c < C; ++c) ss += xr[c] * xr[c];
        const float rinv = 1.0f / std::sqrt(ss / static_cast<float>(C) + eps);
        for (int c = 0; c < C; ++c) yr[c] = xr[c] * rinv;
    }
}

void pixel_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& dY, float eps,
                         ::brotensor::Tensor& dX) {
    check_fp32(X, "pixel_norm_backward", "X");
    check_fp32(dY, "pixel_norm_backward", "dY");
    if (dX.rows != X.rows || dX.cols != X.cols || dX.dtype != Dtype::FP32) {
        dX.resize(X.rows, X.cols, Dtype::FP32);
    }
    const int R = X.rows, C = X.cols;
    if (R == 0 || C == 0) return;
    const float* xp = X.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();   // overwrite — dX may alias dY
    const float invC = 1.0f / static_cast<float>(C);
    for (int r = 0; r < R; ++r) {
        const float* xr = xp + static_cast<size_t>(r) * C;
        const float* dyr = dyp + static_cast<size_t>(r) * C;
        float* dxr = dxp + static_cast<size_t>(r) * C;
        float ss = 0.0f, s = 0.0f;
        for (int c = 0; c < C; ++c) {
            ss += xr[c] * xr[c];
            s  += dyr[c] * xr[c];
        }
        const float rinv = 1.0f / std::sqrt(ss * invC + eps);
        const float r3s  = rinv * rinv * rinv * s * invC;
        // Read s/ss before writing so an in-place dX==dY alias is safe.
        for (int c = 0; c < C; ++c) dxr[c] = rinv * dyr[c] - r3s * xr[c];
    }
}

} // namespace brotensor::detail::cpu
