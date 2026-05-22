// ─── CPU vocoder / codec activations (brosoundml CHUNK 4, family C) ─────────
//
// FP32 scalar host implementations of the genuinely-new vocoder/codec
// activation ops:
//   snake_forward / snake_backward         — BigVGAN / DAC snake + snakebeta
//   elu_forward / elu_backward             — EnCodec ELU
//   leaky_relu_forward / leaky_relu_backward — HiFi-GAN leaky ReLU
//
// ── Layout ──────────────────────────────────────────────────────────────────
//   snake is per-channel over an NCL tensor: element (n, c, l) at flat index
//   (n*C + c)*L + l. alpha / beta carry one scalar per channel c, broadcast
//   across the (n, l) plane — the group_norm per-channel handling.
//   elu / leaky_relu are plain elementwise — shape is irrelevant.
//
// ── Accumulation (matches the group_norm_backward contract) ─────────────────
//   snake_forward            — Y  OVERWRITTEN.
//   snake_backward           — dX OVERWRITTEN; dAlpha / dBeta ACCUMULATE (+=).
//   elu / leaky_relu forward — y  OVERWRITTEN.
//   elu / leaky_relu backward— dX OVERWRITTEN (no learnable params).
//
// CPU is FP32-only; alpha / beta / negative_slope arithmetic is FP32.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void require_fp32(const char* op, const ::brotensor::Tensor& t,
                  const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
    }
}

// Sign-preserving floor on a reciprocal denominator: keeps |d| >= 1e-9 so a
// near-zero alpha/beta degrades gracefully instead of producing NaN/Inf.
inline float guard_denom(float d) {
    constexpr float kMin = 1e-9f;
    if (d >= 0.0f) return d < kMin ? kMin : d;
    return d > -kMin ? -kMin : d;
}

} // namespace

// ─── snake ──────────────────────────────────────────────────────────────────

void snake_forward(const ::brotensor::Tensor& X,
                   const ::brotensor::Tensor& alpha,
                   const ::brotensor::Tensor* beta,
                   int N, int C, int L,
                   ::brotensor::Tensor& Y) {
    require_fp32("snake_forward", X, "X");
    require_fp32("snake_forward", alpha, "alpha");
    if (beta) require_fp32("snake_forward", *beta, "beta");
    if (N < 0 || C < 0 || L < 0) {
        fail("snake_forward", "N, C, L must be non-negative");
    }
    const long long cols = static_cast<long long>(C) * L;
    if (X.rows != N || X.cols != cols) {
        fail("snake_forward", "X must be shaped (N, C*L)");
    }
    if (alpha.size() != C) {
        fail("snake_forward", "alpha must have C elements");
    }
    if (beta && beta->size() != C) {
        fail("snake_forward", "beta must have C elements");
    }
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, static_cast<int>(cols), X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    const float* ap = alpha.host_f32();
    const float* bp = beta ? beta->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float a = ap[c];
            const float denom = guard_denom(bp ? bp[c] : a);
            const float r = 1.0f / denom;
            const float* x_row = Xp + (static_cast<long long>(n) * C + c) * L;
            float*       y_row = Yp + (static_cast<long long>(n) * C + c) * L;
            for (int l = 0; l < L; ++l) {
                const float x = x_row[l];
                const float s = std::sin(a * x);
                y_row[l] = x + r * s * s;          // overwrite
            }
        }
    }
}

void snake_backward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& alpha,
                    const ::brotensor::Tensor* beta,
                    const ::brotensor::Tensor& dY,
                    int N, int C, int L,
                    ::brotensor::Tensor& dX,
                    ::brotensor::Tensor& dAlpha,
                    ::brotensor::Tensor* dBeta) {
    require_fp32("snake_backward", X, "X");
    require_fp32("snake_backward", alpha, "alpha");
    require_fp32("snake_backward", dY, "dY");
    require_fp32("snake_backward", dAlpha, "dAlpha");
    if (beta) require_fp32("snake_backward", *beta, "beta");
    if (dBeta) require_fp32("snake_backward", *dBeta, "dBeta");
    if ((beta == nullptr) != (dBeta == nullptr)) {
        fail("snake_backward",
             "dBeta must be non-null exactly when beta is non-null");
    }
    if (N < 0 || C < 0 || L < 0) {
        fail("snake_backward", "N, C, L must be non-negative");
    }
    const long long cols = static_cast<long long>(C) * L;
    if (X.rows != N || X.cols != cols) {
        fail("snake_backward", "X must be shaped (N, C*L)");
    }
    if (dY.rows != N || dY.cols != cols) {
        fail("snake_backward", "dY must be shaped (N, C*L)");
    }
    if (alpha.size() != C) {
        fail("snake_backward", "alpha must have C elements");
    }
    if (beta && beta->size() != C) {
        fail("snake_backward", "beta must have C elements");
    }
    if (dAlpha.rows != C || dAlpha.cols != 1) {
        fail("snake_backward", "dAlpha must be (C, 1)");
    }
    if (dBeta && (dBeta->rows != C || dBeta->cols != 1)) {
        fail("snake_backward", "dBeta must be (C, 1)");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype) {
        dX.resize(N, static_cast<int>(cols), X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp  = X.host_f32();
    const float* ap  = alpha.host_f32();
    const float* bp  = beta ? beta->host_f32() : nullptr;
    const float* dYp = dY.host_f32();
    float* dXp  = dX.host_f32_mut();
    float* dAp  = dAlpha.host_f32_mut();
    float* dBp  = dBeta ? dBeta->host_f32_mut() : nullptr;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float a     = ap[c];
            const float denom = guard_denom(bp ? bp[c] : a);
            const float r     = 1.0f / denom;
            const float* x_row  = Xp  + (static_cast<long long>(n) * C + c) * L;
            const float* dy_row = dYp + (static_cast<long long>(n) * C + c) * L;
            float*       dx_row = dXp + (static_cast<long long>(n) * C + c) * L;

            float dalpha_acc = 0.0f;
            float dbeta_acc  = 0.0f;
            for (int l = 0; l < L; ++l) {
                const float x  = x_row[l];
                const float dy = dy_row[l];
                const float s  = std::sin(a * x);
                const float co = std::cos(a * x);
                const float sc = s * co;             // sin*cos = 0.5*sin(2ax)

                // dy/dx = 1 + 2*a*r*s*c
                dx_row[l] = dy * (1.0f + 2.0f * a * r * sc);   // overwrite

                // dy/dalpha (frequency term) = 2*r*x*s*c
                dalpha_acc += dy * (2.0f * r * x * sc);

                if (bp) {
                    // snakebeta: dy/dbeta = -r^2 * s^2.
                    dbeta_acc += dy * (-r * r * s * s);
                } else {
                    // plain snake: denom == alpha, so alpha also drives the
                    // reciprocal — add the -r^2*s^2 term into dAlpha too.
                    dalpha_acc += dy * (-r * r * s * s);
                }
            }
            dAp[c] += dalpha_acc;                    // accumulate
            if (dBp) dBp[c] += dbeta_acc;            // accumulate
        }
    }
}

// ─── elu ────────────────────────────────────────────────────────────────────

void elu_forward(const ::brotensor::Tensor& x, float alpha,
                 ::brotensor::Tensor& y) {
    require_fp32("elu_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float v = xp[i];
        yp[i] = v > 0.0f ? v : alpha * (std::exp(v) - 1.0f);
    }
}

void elu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  float alpha, ::brotensor::Tensor& dX) {
    require_fp32("elu_backward", x, "x");
    require_fp32("elu_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp  = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float v = xp[i];
        const float g = v > 0.0f ? 1.0f : alpha * std::exp(v);
        dxp[i] = dyp[i] * g;                          // overwrite
    }
}

// ─── leaky_relu ─────────────────────────────────────────────────────────────

void leaky_relu_forward(const ::brotensor::Tensor& x, float negative_slope,
                        ::brotensor::Tensor& y) {
    require_fp32("leaky_relu_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float v = xp[i];
        yp[i] = v > 0.0f ? v : negative_slope * v;
    }
}

void leaky_relu_backward(const ::brotensor::Tensor& x,
                         const ::brotensor::Tensor& dY,
                         float negative_slope, ::brotensor::Tensor& dX) {
    require_fp32("leaky_relu_backward", x, "x");
    require_fp32("leaky_relu_backward", dY, "dY");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype) {
        dX.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    const float* xp  = x.host_f32();
    const float* dyp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float g = xp[i] > 0.0f ? 1.0f : negative_slope;
        dxp[i] = dyp[i] * g;                          // overwrite
    }
}

} // namespace brotensor::detail::cpu
