// ─── CPU 1D resampling ops (CHUNK 6, family E) ─────────────────────────────
//
// FP32 scalar host implementations. Arbitrary-scale resampling along the
// length axis of an NCL audio tensor — the 1D, arbitrary-ratio analogue of
// the fixed-2x NCHW resample ops in resample.cpp. CPU is FP32-only.
//
// Memory layout (NCL flat — consistent with conv1d.cpp / snake):
//   X / Y / dX / dY : ((n * C + c) * L) + l
//   resample1d_forward : (N, C, L_in)  -> (N, C, L_out)
//
// Sampling convention — PyTorch align_corners=False:
//   src = (dst + 0.5) * (L_in / L_out) - 0.5
//   nearest : Y[dst] = X[ clamp(round_half_to_even(src), 0, L_in-1) ]
//   linear  : s  = clamp(src, 0, L_in-1),  x0 = floor(s),
//             x1 = min(x0+1, L_in-1),  f = s - x0
//             Y[dst] = (1-f) * X[x0] + f * X[x1]
//
// ACCUMULATION:
//   resample1d_forward  — Y  OVERWRITTEN.
//   resample1d_backward — dX OVERWRITTEN (zero-then-scatter; resampling has no
//                         learnable parameters, so the adjoint overwrites dX).

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

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline void check_args(const char* op, int N, int C, int L_in, int L_out,
                       int mode) {
    if (N < 0 || C < 0 || L_in < 0 || L_out < 0) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": N, C, L_in, L_out must be non-negative");
    }
    if (mode != 0 && mode != 1) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": mode must be 0 (nearest) or 1 (linear)");
    }
    if (L_out > 0 && L_in == 0) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": L_in must be > 0 when L_out > 0");
    }
}

} // namespace

// ─── Forward ───────────────────────────────────────────────────────────────

void resample1d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int L_in, int L_out, int mode,
                        ::brotensor::Tensor& Y) {
    check_fp32(X, "resample1d_forward", "X");
    check_args("resample1d_forward", N, C, L_in, L_out, mode);

    const int cols = C * L_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    const double scale = static_cast<double>(L_in) /
                         static_cast<double>(L_out);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int xbase = (n * C + c) * L_in;
            const int ybase = (n * C + c) * L_out;
            for (int dst = 0; dst < L_out; ++dst) {
                const double src = (dst + 0.5) * scale - 0.5;
                if (mode == 0) {
                    // nearest — round-half-to-even, then clamp.
                    const int idx = clampi(
                        static_cast<int>(std::nearbyint(src)), 0, L_in - 1);
                    Yp[ybase + dst] = Xp[xbase + idx];
                } else {
                    // linear — clamp src into range, then split into taps.
                    double s = src;
                    if (s < 0.0) s = 0.0;
                    if (s > L_in - 1) s = L_in - 1;
                    const int x0 = static_cast<int>(std::floor(s));
                    const int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
                    const float f = static_cast<float>(s - x0);
                    Yp[ybase + dst] = (1.0f - f) * Xp[xbase + x0] +
                                      f * Xp[xbase + x1];
                }
            }
        }
    }
}

// ─── Backward ──────────────────────────────────────────────────────────────

void resample1d_backward(const ::brotensor::Tensor& dY,
                         int N, int C, int L_in, int L_out, int mode,
                         ::brotensor::Tensor& dX) {
    check_fp32(dY, "resample1d_backward", "dY");
    check_args("resample1d_backward", N, C, L_in, L_out, mode);

    const int cols_in = C * L_in;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Adjoint: zero dX, then scatter each output gradient onto the input
    // position(s) it sampled — with the same weights as the forward pass.
    const int total_in = N * cols_in;
    for (int i = 0; i < total_in; ++i) dXp[i] = 0.0f;

    if (L_out == 0) return;

    const double scale = static_cast<double>(L_in) /
                         static_cast<double>(L_out);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int xbase = (n * C + c) * L_in;
            const int ybase = (n * C + c) * L_out;
            for (int dst = 0; dst < L_out; ++dst) {
                const double src = (dst + 0.5) * scale - 0.5;
                const float g = dYp[ybase + dst];
                if (mode == 0) {
                    const int idx = clampi(
                        static_cast<int>(std::nearbyint(src)), 0, L_in - 1);
                    dXp[xbase + idx] += g;
                } else {
                    double s = src;
                    if (s < 0.0) s = 0.0;
                    if (s > L_in - 1) s = L_in - 1;
                    const int x0 = static_cast<int>(std::floor(s));
                    const int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
                    const float f = static_cast<float>(s - x0);
                    dXp[xbase + x0] += (1.0f - f) * g;
                    dXp[xbase + x1] += f * g;
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
