// ─── CPU L2-norm (per-head, last-dim) ──────────────────────────────────────
//
// FP32-only host implementation. Used by Gated DeltaNet to L2-normalise q and
// k per head before the recurrence — distinct from rms_norm (sum vs. mean of
// squares, no learnable gamma, no sqrt(d) factor).
//
// Layout: (L, num_heads * head_dim), row-major; head h occupies columns
// [h*head_dim, (h+1)*head_dim). Same convention as rope_forward and rms_norm.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

inline void check_shape(const ::brotensor::Tensor& t,
                        int head_dim, int num_heads,
                        const char* op, const char* name) {
    if (head_dim <= 0) {
        throw std::runtime_error(std::string(op) + ": head_dim must be positive");
    }
    if (num_heads <= 0) {
        throw std::runtime_error(std::string(op) + ": num_heads must be positive");
    }
    if (t.cols != num_heads * head_dim) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 ".cols != num_heads * head_dim");
    }
}

} // namespace

void l2_norm_forward(const ::brotensor::Tensor& X,
                     int head_dim, int num_heads, float eps,
                     ::brotensor::Tensor& Y) {
    check_fp32(X, "l2_norm_forward", "X");
    check_shape(X, head_dim, num_heads, "l2_norm_forward", "X");
    const int L = X.rows;
    const int D = X.cols;
    if (Y.rows != L || Y.cols != D || Y.dtype != Dtype::FP32) {
        Y.resize(L, D, Dtype::FP32);
    }
    if (L == 0 || D == 0) return;

    const float* xp = X.host_f32();
    float* yp = Y.host_f32_mut();

    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < num_heads; ++h) {
            const int off = r * D + h * head_dim;
            // sum of squares
            float sumsq = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                const float v = xp[off + d];
                sumsq += v * v;
            }
            const float inv = 1.0f / std::sqrt(sumsq + eps);
            for (int d = 0; d < head_dim; ++d) {
                yp[off + d] = xp[off + d] * inv;
            }
        }
    }
}

void l2_norm_backward(const ::brotensor::Tensor& X,
                      int head_dim, int num_heads, float eps,
                      const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX) {
    check_fp32(X,  "l2_norm_backward", "X");
    check_fp32(dY, "l2_norm_backward", "dY");
    check_shape(X,  head_dim, num_heads, "l2_norm_backward", "X");
    check_shape(dY, head_dim, num_heads, "l2_norm_backward", "dY");
    if (dY.rows != X.rows) {
        throw std::runtime_error("l2_norm_backward: dY.rows != X.rows");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (dX.rows != L || dX.cols != D || dX.dtype != Dtype::FP32) {
        dX.resize(L, D, Dtype::FP32);
    }
    if (L == 0 || D == 0) return;

    const float* xp  = X.host_f32();
    const float* gp  = dY.host_f32();
    float* dxp = dX.host_f32_mut();

    // dX_d = n * dY_d - x_d * n^3 * sum_d' (x_d' * dY_d')
    //      = n * (dY_d - x_d * n^2 * dot(x, dY))
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < num_heads; ++h) {
            const int off = r * D + h * head_dim;
            float sumsq = 0.0f;
            float dot   = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                const float v  = xp[off + d];
                const float gv = gp[off + d];
                sumsq += v * v;
                dot   += v * gv;
            }
            const float n2 = 1.0f / (sumsq + eps);     // n^2
            const float n  = std::sqrt(n2);            // n
            const float c  = dot * n2;                  // x . dY / (||x||^2 + eps)
            for (int d = 0; d < head_dim; ++d) {
                dxp[off + d] = n * (gp[off + d] - xp[off + d] * c);
            }
        }
    }
}

} // namespace brotensor::detail::cpu
