// ─── CPU L2 normalization over the channel axis (NCHW) ──────────────────────
//
// FP32 scalar host implementation of l2_normalize_nchw_forward. For every
// spatial position (n, h, w), normalizes the length-C channel vector to unit
// L2 norm with an epsilon floor on the divisor:
//   Y[n,c,h,w] = X[n,c,h,w] / max(sqrt(sum_c X[n,c,h,w]^2), eps)
// The per-pixel vector normalize used by surface-normal / direction-field
// models (DSINE pred-normal normalize), feature-map L2 norm, and cosine-sim
// preprocessing. Distinct from l2_norm_forward (gated-deltanet per-head, last
// dim of a (L, H*D) layout) — this one is NCHW with the unit axis = channels.
//
// Reduction accumulates in double for stability; output is FP32.
//
// ── ACCUMULATION ──  Y OVERWRITTEN. X and Y may alias. Inference-only: no
//                     backward (the bro pipeline never trains through this).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32)
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
}

} // namespace

void l2_normalize_nchw_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W,
                               float eps,
                               ::brotensor::Tensor& Y) {
    const char* op = "l2_normalize_nchw_forward";
    check_fp32(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");
    if (Y.rows != N || Y.cols != C * H * W || Y.dtype != Dtype::FP32)
        Y.resize(N, C * H * W, Dtype::FP32);
    if (N == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();
    const int HW = H * W;
    const double epsd = static_cast<double>(eps);

    for (int n = 0; n < N; ++n) {
        const float* x_img = Xp + static_cast<long>(n) * C * HW;
        float* y_img = Yp + static_cast<long>(n) * C * HW;
        for (int p = 0; p < HW; ++p) {
            double ss = 0.0;
            for (int c = 0; c < C; ++c) {
                const double v = x_img[static_cast<long>(c) * HW + p];
                ss += v * v;
            }
            const double inv = 1.0 / std::max(std::sqrt(ss), epsd);
            for (int c = 0; c < C; ++c)
                y_img[static_cast<long>(c) * HW + p] =
                    static_cast<float>(x_img[static_cast<long>(c) * HW + p] * inv);
        }
    }
}

} // namespace brotensor::detail::cpu
