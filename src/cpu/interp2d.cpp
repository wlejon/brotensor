// ─── CPU arbitrary-scale 2D resample ────────────────────────────────────────
//
// FP32 scalar host implementations. The general 2D counterpart to the fixed-2x
// upsample_*_2x ops in resample.cpp — supports any (H_in, W_in) -> (H_out,
// W_out) on an NCHW tensor, with nearest / bilinear / bicubic modes. CPU is
// FP32-only.
//
// Memory layout (NCHW flat — matches resample.cpp):
//   X / Y / dX / dY : ((n * C + c) * H + h) * W + w
//
// Sampling convention — PyTorch align_corners=False / half-pixel (matches the
// existing upsample_bilinear_2x exactly when (H_out, W_out) == (2H, 2W)):
//   src_y = (oh + 0.5) * (H_in / H_out) - 0.5
//   src_x = (ow + 0.5) * (W_in / W_out) - 0.5
//
//   nearest  : Y[oh,ow] = X[clamp(round_half_to_even(src), 0, dim-1)]
//   bilinear : 2x2 tap weighted blend (border-clamped indices)
//   bicubic  : 4x4 cubic-convolution tap, border-clamped — forward only.
//              mode 2 uses a = -0.5 (Catmull-Rom, matches PIL/Pillow BICUBIC);
//              mode 3 uses a = -0.75 (matches torch.nn.functional.interpolate
//              mode="bicubic" and OpenCV). The two differ only in that constant.
//
// ACCUMULATION:
//   interp2d_forward  — Y  OVERWRITTEN.
//   interp2d_backward — dX OVERWRITTEN (zero-then-scatter; resampling has no
//                       learnable parameters, so the adjoint overwrites dX).
//
// Identity check: if (H_out, W_out) == (H_in, W_in) and mode == 0 or 1, the
// op is the identity (within rounding) — the test relies on this.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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

inline void check_args(const char* op,
                       int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode,
                       bool allow_bicubic) {
    if (N < 0 || C < 0 || H_in < 0 || W_in < 0 ||
        H_out < 0 || W_out < 0) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": N, C, H_in, W_in, H_out, W_out must be "
                                 "non-negative");
    }
    const int max_mode = allow_bicubic ? 3 : 1;
    if (mode < 0 || mode > max_mode) {
        const char* msg = allow_bicubic
            ? ": mode must be 0 (nearest), 1 (bilinear), 2 (bicubic a=-0.5, "
              "PIL), or 3 (bicubic a=-0.75, torch)"
            : ": mode must be 0 (nearest) or 1 (bilinear) — bicubic "
              "backward is not implemented";
        throw std::runtime_error(std::string("brotensor: ") + op + msg);
    }
    if ((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": input spatial dims must be > 0 when "
                                 "output spatial dims are > 0");
    }
}

// Keys cubic-convolution kernel with coefficient `a`. a = -0.5 is Catmull-Rom
// (matches PIL/Pillow BICUBIC); a = -0.75 matches PyTorch
// interpolate(mode="bicubic") and OpenCV. |t| in [0,1] uses the first branch,
// |t| in [1,2] the second, otherwise 0.
inline float cubic_keys(float t, float a) {
    const float at = t < 0.0f ? -t : t;
    if (at < 1.0f) {
        return ((a + 2.0f) * at - (a + 3.0f)) * at * at + 1.0f;
    }
    if (at < 2.0f) {
        // a*t^3 - 5a*t^2 + 8a*t - 4a, Horner in |t|.
        return ((a * at - 5.0f * a) * at + 8.0f * a) * at - 4.0f * a;
    }
    return 0.0f;
}

} // namespace

// ─── Forward ───────────────────────────────────────────────────────────────

// Corner-aligned source coordinate: out pixel `o` maps to o*(in-1)/(out-1),
// with the degenerate out==1 case pinned to 0 (torch align_corners=True).
inline double align_corners_src(int o, int in_dim, int out_dim) {
    if (out_dim <= 1) return 0.0;
    return static_cast<double>(o) * static_cast<double>(in_dim - 1) /
           static_cast<double>(out_dim - 1);
}

// Shared forward worker for both the half-pixel (align_corners=False) and the
// corner-aligned (align_corners=True) resample — they differ only in the
// source-coordinate mapping, so the tap math below is identical.
static void interp2d_forward_impl(const ::brotensor::Tensor& X,
                                  int N, int C, int H_in, int W_in,
                                  int H_out, int W_out, int mode, bool align,
                                  ::brotensor::Tensor& Y, const char* op) {
    check_fp32(X, op, "X");
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/true);

    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    // Half-pixel scale (unused on the align path, which reads from in-1/out-1).
    const double sy = static_cast<double>(H_in) / static_cast<double>(H_out);
    const double sx = static_cast<double>(W_in) / static_cast<double>(W_out);

    // Per-row / per-column tap tables. src_y/src_x and the derived taps depend
    // only on (oh) or (ow) — never on (n, c) — so compute them once here
    // instead of redoing it inside the (n, c) loop below.
    struct RowNearest { int iy; };
    struct ColNearest { int ix; };
    struct RowBilinear { int y0, y1; float fy; };
    struct ColBilinear { int x0, x1; float fx; };
    struct RowBicubic { int iy[4]; float wy[4]; };
    struct ColBicubic { int ix[4]; float wx[4]; };

    std::vector<RowNearest> row_nearest;
    std::vector<ColNearest> col_nearest;
    std::vector<RowBilinear> row_bilinear;
    std::vector<ColBilinear> col_bilinear;
    std::vector<RowBicubic> row_bicubic;
    std::vector<ColBicubic> col_bicubic;

    if (mode == 0) {
        row_nearest.resize(H_out);
        for (int oh = 0; oh < H_out; ++oh) {
            const double src_y = align ? align_corners_src(oh, H_in, H_out)
                                       : (oh + 0.5) * sy - 0.5;
            row_nearest[oh].iy = clampi(
                static_cast<int>(std::nearbyint(src_y)), 0, H_in - 1);
        }
        col_nearest.resize(W_out);
        for (int ow = 0; ow < W_out; ++ow) {
            const double src_x = align ? align_corners_src(ow, W_in, W_out)
                                       : (ow + 0.5) * sx - 0.5;
            col_nearest[ow].ix = clampi(
                static_cast<int>(std::nearbyint(src_x)), 0, W_in - 1);
        }
    } else if (mode == 1) {
        row_bilinear.resize(H_out);
        for (int oh = 0; oh < H_out; ++oh) {
            const double src_y = align ? align_corners_src(oh, H_in, H_out)
                                       : (oh + 0.5) * sy - 0.5;
            const int y0 = static_cast<int>(std::floor(src_y));
            row_bilinear[oh].fy = static_cast<float>(src_y - y0);
            row_bilinear[oh].y0 = clampi(y0,     0, H_in - 1);
            row_bilinear[oh].y1 = clampi(y0 + 1, 0, H_in - 1);
        }
        col_bilinear.resize(W_out);
        for (int ow = 0; ow < W_out; ++ow) {
            const double src_x = align ? align_corners_src(ow, W_in, W_out)
                                       : (ow + 0.5) * sx - 0.5;
            const int x0 = static_cast<int>(std::floor(src_x));
            col_bilinear[ow].fx = static_cast<float>(src_x - x0);
            col_bilinear[ow].x0 = clampi(x0,     0, W_in - 1);
            col_bilinear[ow].x1 = clampi(x0 + 1, 0, W_in - 1);
        }
    } else {
        // bicubic — 4x4 cubic-convolution, border-clamped.
        // mode 2: a=-0.5 (PIL); mode 3: a=-0.75 (torch).
        const float a = (mode == 3) ? -0.75f : -0.5f;
        row_bicubic.resize(H_out);
        for (int oh = 0; oh < H_out; ++oh) {
            const double src_y = align ? align_corners_src(oh, H_in, H_out)
                                       : (oh + 0.5) * sy - 0.5;
            const int y0 = static_cast<int>(std::floor(src_y));
            const float fy = static_cast<float>(src_y - y0);
            for (int k = 0; k < 4; ++k) {
                row_bicubic[oh].wy[k] = cubic_keys(fy - (k - 1), a);
                row_bicubic[oh].iy[k] = clampi(y0 + k - 1, 0, H_in - 1);
            }
        }
        col_bicubic.resize(W_out);
        for (int ow = 0; ow < W_out; ++ow) {
            const double src_x = align ? align_corners_src(ow, W_in, W_out)
                                       : (ow + 0.5) * sx - 0.5;
            const int x0 = static_cast<int>(std::floor(src_x));
            const float fx = static_cast<float>(src_x - x0);
            for (int k = 0; k < 4; ++k) {
                col_bicubic[ow].wx[k] = cubic_keys(fx - (k - 1), a);
                col_bicubic[ow].ix[k] = clampi(x0 + k - 1, 0, W_in - 1);
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int xbase = (n * C + c) * H_in * W_in;
            const int ybase = (n * C + c) * H_out * W_out;

            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    if (mode == 0) {
                        // nearest — round_half_to_even then clamp.
                        const int iy = row_nearest[oh].iy;
                        const int ix = col_nearest[ow].ix;
                        Yp[ybase + oh * W_out + ow] =
                            Xp[xbase + iy * W_in + ix];
                    } else if (mode == 1) {
                        // bilinear — 2x2 tap, border-clamped.
                        const RowBilinear& r = row_bilinear[oh];
                        const ColBilinear& cx = col_bilinear[ow];
                        const float v00 = Xp[xbase + r.y0 * W_in + cx.x0];
                        const float v01 = Xp[xbase + r.y0 * W_in + cx.x1];
                        const float v10 = Xp[xbase + r.y1 * W_in + cx.x0];
                        const float v11 = Xp[xbase + r.y1 * W_in + cx.x1];
                        const float top = v00 + (v01 - v00) * cx.fx;
                        const float bot = v10 + (v11 - v10) * cx.fx;
                        Yp[ybase + oh * W_out + ow] =
                            top + (bot - top) * r.fy;
                    } else {
                        // bicubic — 4x4 cubic-convolution, border-clamped.
                        const RowBicubic& r = row_bicubic[oh];
                        const ColBicubic& cx = col_bicubic[ow];
                        float acc = 0.0f;
                        for (int j = 0; j < 4; ++j) {
                            float row = 0.0f;
                            for (int i = 0; i < 4; ++i) {
                                row += cx.wx[i] * Xp[xbase + r.iy[j] * W_in + cx.ix[i]];
                            }
                            acc += r.wy[j] * row;
                        }
                        Yp[ybase + oh * W_out + ow] = acc;
                    }
                }
            }
        }
    }
}

void interp2d_forward(const ::brotensor::Tensor& X,
                      int N, int C, int H_in, int W_in,
                      int H_out, int W_out, int mode,
                      ::brotensor::Tensor& Y) {
    interp2d_forward_impl(X, N, C, H_in, W_in, H_out, W_out, mode,
                          /*align=*/false, Y, "interp2d_forward");
}

void interp2d_align_corners_forward(const ::brotensor::Tensor& X,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out, int mode,
                                    ::brotensor::Tensor& Y) {
    interp2d_forward_impl(X, N, C, H_in, W_in, H_out, W_out, mode,
                          /*align=*/true, Y, "interp2d_align_corners_forward");
}

// ─── Backward ──────────────────────────────────────────────────────────────

void interp2d_backward(const ::brotensor::Tensor& dY,
                       int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode,
                       ::brotensor::Tensor& dX) {
    check_fp32(dY, "interp2d_backward", "dY");
    check_args("interp2d_backward", N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/false);

    const int cols_in = C * H_in * W_in;
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

    if (H_out == 0 || W_out == 0) return;

    const double sy = static_cast<double>(H_in) / static_cast<double>(H_out);
    const double sx = static_cast<double>(W_in) / static_cast<double>(W_out);

    // Per-row / per-column tap tables — same rationale as the forward pass:
    // src_y/src_x and the derived taps depend only on (oh)/(ow), not (n, c).
    struct RowNearest { int iy; };
    struct ColNearest { int ix; };
    struct RowBilinear { int y0, y1; float fy; };
    struct ColBilinear { int x0, x1; float fx; };

    std::vector<RowNearest> row_nearest;
    std::vector<ColNearest> col_nearest;
    std::vector<RowBilinear> row_bilinear;
    std::vector<ColBilinear> col_bilinear;

    if (mode == 0) {
        row_nearest.resize(H_out);
        for (int oh = 0; oh < H_out; ++oh) {
            const double src_y = (oh + 0.5) * sy - 0.5;
            row_nearest[oh].iy = clampi(
                static_cast<int>(std::nearbyint(src_y)), 0, H_in - 1);
        }
        col_nearest.resize(W_out);
        for (int ow = 0; ow < W_out; ++ow) {
            const double src_x = (ow + 0.5) * sx - 0.5;
            col_nearest[ow].ix = clampi(
                static_cast<int>(std::nearbyint(src_x)), 0, W_in - 1);
        }
    } else {
        row_bilinear.resize(H_out);
        for (int oh = 0; oh < H_out; ++oh) {
            const double src_y = (oh + 0.5) * sy - 0.5;
            const int y0 = static_cast<int>(std::floor(src_y));
            row_bilinear[oh].fy = static_cast<float>(src_y - y0);
            row_bilinear[oh].y0 = clampi(y0,     0, H_in - 1);
            row_bilinear[oh].y1 = clampi(y0 + 1, 0, H_in - 1);
        }
        col_bilinear.resize(W_out);
        for (int ow = 0; ow < W_out; ++ow) {
            const double src_x = (ow + 0.5) * sx - 0.5;
            const int x0 = static_cast<int>(std::floor(src_x));
            col_bilinear[ow].fx = static_cast<float>(src_x - x0);
            col_bilinear[ow].x0 = clampi(x0,     0, W_in - 1);
            col_bilinear[ow].x1 = clampi(x0 + 1, 0, W_in - 1);
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int xbase = (n * C + c) * H_in * W_in;
            const int ybase = (n * C + c) * H_out * W_out;

            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    const float g = dYp[ybase + oh * W_out + ow];

                    if (mode == 0) {
                        const int iy = row_nearest[oh].iy;
                        const int ix = col_nearest[ow].ix;
                        dXp[xbase + iy * W_in + ix] += g;
                    } else {
                        const RowBilinear& r = row_bilinear[oh];
                        const ColBilinear& cx = col_bilinear[ow];
                        const float w00 = (1.0f - r.fy) * (1.0f - cx.fx);
                        const float w01 = (1.0f - r.fy) * cx.fx;
                        const float w10 = r.fy        * (1.0f - cx.fx);
                        const float w11 = r.fy        * cx.fx;
                        dXp[xbase + r.y0 * W_in + cx.x0] += w00 * g;
                        dXp[xbase + r.y0 * W_in + cx.x1] += w01 * g;
                        dXp[xbase + r.y1 * W_in + cx.x0] += w10 * g;
                        dXp[xbase + r.y1 * W_in + cx.x1] += w11 * g;
                    }
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
