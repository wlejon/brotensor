// ─── CPU bias_act (StyleGAN3) ───────────────────────────────────────────────
//
// Fused per-channel bias + activation + gain + clamp. FP32 reference mirroring
// NVlabs `_bias_act_ref`. CPU is FP32-only.
//
//   X: (N, C*HW)  — HW is the flattened spatial size; channel c owns the
//                   contiguous block [c*HW, (c+1)*HW) within each row.
//   b: (C,1) or null.   act: 0 = linear, 1 = lrelu.   clamp < 0 ⇒ no clamp.
//
// Forward:  t = X + b[c];  y = act(t);  y *= gain;  if clamp>=0: clip(±clamp).
// Backward: recompute t; dt = dY*gain*act'(t)*(clamp active ? 0 : 1);
//           dX = dt (overwrite);  dB[c] += Σ dt (accumulate — caller zeros).
//
// The clamp gradient mask uses the PRE-clamp value y_pre = gain*act(t): where
// |y_pre| > clamp the output saturated, so the gradient is zero there.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

constexpr int ACT_LINEAR = 0;
constexpr int ACT_LRELU  = 1;

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " +
                                 name + " must be FP32 (CPU backend is "
                                 "FP32-only)");
    }
}

inline void check_act(int act, const char* op) {
    if (act != ACT_LINEAR && act != ACT_LRELU) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": act must be 0 (linear) or 1 (lrelu)");
    }
}

inline float apply_act(float t, int act, float alpha) {
    if (act == ACT_LRELU) return t > 0.0f ? t : alpha * t;
    return t;  // linear
}

inline float act_grad(float t, int act, float alpha) {
    if (act == ACT_LRELU) return t > 0.0f ? 1.0f : alpha;
    return 1.0f;  // linear
}

} // namespace

void bias_act_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, ::brotensor::Tensor& Y) {
    check_fp32(X, "bias_act_forward", "X");
    if (b) check_fp32(*b, "bias_act_forward", "b");
    check_act(act, "bias_act_forward");
    const int cols = C * HW;
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("bias_act_forward: X shape mismatch");
    }
    if (b && b->size() != C) {
        throw std::runtime_error("bias_act_forward: b must have C elements");
    }
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    const float* Bp = b ? b->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float bias_v = Bp ? Bp[c] : 0.0f;
            const size_t base = (static_cast<size_t>(n) * C + c) * HW;
            for (int k = 0; k < HW; ++k) {
                float y = apply_act(Xp[base + k] + bias_v, act, alpha) * gain;
                if (clamp >= 0.0f) {
                    if (y < -clamp) y = -clamp;
                    else if (y > clamp) y = clamp;
                }
                Yp[base + k] = y;
            }
        }
    }
}

void bias_act_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp,
                       ::brotensor::Tensor& dX, ::brotensor::Tensor* dB) {
    check_fp32(dY, "bias_act_backward", "dY");
    check_fp32(X, "bias_act_backward", "X");
    if (b) check_fp32(*b, "bias_act_backward", "b");
    if (dB) check_fp32(*dB, "bias_act_backward", "dB");
    check_act(act, "bias_act_backward");
    const int cols = C * HW;
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("bias_act_backward: X shape mismatch");
    }
    if (dY.rows != N || dY.cols != cols) {
        throw std::runtime_error("bias_act_backward: dY shape mismatch");
    }
    if (b && b->size() != C) {
        throw std::runtime_error("bias_act_backward: b must have C elements");
    }
    if (dB && dB->size() != C) {
        throw std::runtime_error("bias_act_backward: dB must have C elements");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* dYp = dY.host_f32();
    const float* Xp = X.host_f32();
    const float* Bp = b ? b->host_f32() : nullptr;
    float* dXp = dX.host_f32_mut();
    float* dBp = dB ? dB->host_f32_mut() : nullptr;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float bias_v = Bp ? Bp[c] : 0.0f;
            const size_t base = (static_cast<size_t>(n) * C + c) * HW;
            float db_acc = 0.0f;
            for (int k = 0; k < HW; ++k) {
                const float t = Xp[base + k] + bias_v;
                float dt = dYp[base + k] * gain * act_grad(t, act, alpha);
                if (clamp >= 0.0f) {
                    const float y_pre = gain * apply_act(t, act, alpha);
                    if (y_pre < -clamp || y_pre > clamp) dt = 0.0f;
                }
                dXp[base + k] = dt;   // overwrite
                db_acc += dt;
            }
            if (dBp) dBp[c] += db_acc;   // accumulate — caller zeros
        }
    }
}

} // namespace brotensor::detail::cpu
