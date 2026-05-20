// ─── CPU SwiGLU ops (CHUNK 2) ──────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/swiglu.cu — kernel math
// reproduced verbatim, FP32 path only.
//
//   X is (B, 2D); split along the last dim into A = X[:, :D] (the value half
//   that is gated through silu) and B_half = X[:, D:] (the linear half).
//     forward:  Y(B, D)  = silu(A) * B_half
//     backward: dX(B,2D): dX[:, :D] = dY * B_half * silu'(A)
//                         dX[:, D:] = dY * silu(A)
//
// Backward overwrites dX (the GPU kernel writes both halves directly).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>

namespace brotensor::detail::cpu {

namespace {

inline float silu_scalar(float v) {
    return v / (1.0f + std::exp(-v));
}

inline float silu_grad_scalar(float v) {
    const float s = 1.0f / (1.0f + std::exp(-v));
    return s * (1.0f + v * (1.0f - s));
}

} // namespace

void swiglu_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_forward: X.cols must be even (2*D)");
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
            const float a  = Xp[b * two_d + d];
            const float bh = Xp[b * two_d + D + d];
            Yp[b * D + d] = silu_scalar(a) * bh;
        }
    }
}

void swiglu_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_backward: X.cols must be even (2*D)");
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
            const float s  = silu_scalar(a);
            const float sp = silu_grad_scalar(a);
            dXp[b * two_d + d]     = dy * bh * sp;
            dXp[b * two_d + D + d] = dy * s;
        }
    }
}

} // namespace brotensor::detail::cpu
