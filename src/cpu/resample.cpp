// ─── CPU resample ops (CHUNK 4) ────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/resample.cu — FP32 path
// only (CPU is FP32-only). 2x spatial nearest / bilinear upsample, 2x average
// downsample, plus their backward passes. All tensors NCHW row-major.
//
// Memory layout (matches the GPU exactly):
//   X / Y / dX / dY : NCHW — ((n*C + c) * H + h) * W + w
//   Upsample:   H_out = 2*H, W_out = 2*W.
//   Downsample: H_out = H/2, W_out = W/2 (H, W must be even).
//
// Bilinear sampling convention (matches the GPU kernel verbatim — HALF-PIXEL,
// NOT align-corners):
//   src_y = (oh + 0.5) * 0.5 - 0.5   (scale 0.5 because output is 2x input)
//   src_x = (ow + 0.5) * 0.5 - 0.5
//   y0 = floor(src_y), fy = src_y - y0  (then border-clamped indices)
//   value = bilinear interp of the 4 clamped neighbours.
//
// ACCUMULATION (matches the GPU kernels):
//   upsample_nearest_2x          — Y  OVERWRITTEN.
//   upsample_bilinear_2x         — Y  OVERWRITTEN.
//   downsample_avg_2x            — Y  OVERWRITTEN.
//   upsample_nearest_2x_backward — dX OVERWRITTEN (gather of 4 dY values).
//   upsample_bilinear_2x_backward— dX OVERWRITTEN (GPU memsets dX to 0 then
//                                  atomic-scatters; net effect = overwrite).
//   downsample_avg_2x_backward   — dX OVERWRITTEN.

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
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

// ─── Forward ───────────────────────────────────────────────────────────────

void upsample_nearest_2x(const ::brotensor::Tensor& X,
                         int N, int C, int H, int W,
                         ::brotensor::Tensor& Y) {
    check_fp32(X, "upsample_nearest_2x", "X");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                const int ih = oh / 2;
                for (int ow = 0; ow < W_out; ++ow) {
                    const int iw = ow / 2;
                    Yp[((n * C + c) * H_out + oh) * W_out + ow] =
                        Xp[((n * C + c) * H + ih) * W + iw];
                }
            }
        }
    }
}

void upsample_bilinear_2x(const ::brotensor::Tensor& X,
                          int N, int C, int H, int W,
                          ::brotensor::Tensor& Y) {
    check_fp32(X, "upsample_bilinear_2x", "X");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    // src_y/src_x and the derived taps depend only on (oh)/(ow), never on
    // (n, c) — precompute once instead of redoing it inside the (n, c) loop.
    struct RowTap { int y0, y1; float fy; };
    struct ColTap { int x0, x1; float fx; };
    std::vector<RowTap> row_tap(H_out);
    for (int oh = 0; oh < H_out; ++oh) {
        const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
        const int y0 = static_cast<int>(std::floor(src_y));
        row_tap[oh].fy = src_y - y0;
        row_tap[oh].y0 = clampi(y0,     0, H - 1);
        row_tap[oh].y1 = clampi(y0 + 1, 0, H - 1);
    }
    std::vector<ColTap> col_tap(W_out);
    for (int ow = 0; ow < W_out; ++ow) {
        const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
        const int x0 = static_cast<int>(std::floor(src_x));
        col_tap[ow].fx = src_x - x0;
        col_tap[ow].x0 = clampi(x0,     0, W - 1);
        col_tap[ow].x1 = clampi(x0 + 1, 0, W - 1);
    }

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int base = (n * C + c) * H;
            for (int oh = 0; oh < H_out; ++oh) {
                const RowTap& r = row_tap[oh];
                for (int ow = 0; ow < W_out; ++ow) {
                    const ColTap& cx = col_tap[ow];
                    const float v00 = Xp[(base + r.y0) * W + cx.x0];
                    const float v01 = Xp[(base + r.y0) * W + cx.x1];
                    const float v10 = Xp[(base + r.y1) * W + cx.x0];
                    const float v11 = Xp[(base + r.y1) * W + cx.x1];
                    const float top = v00 + (v01 - v00) * cx.fx;
                    const float bot = v10 + (v11 - v10) * cx.fx;
                    Yp[((n * C + c) * H_out + oh) * W_out + ow] =
                        top + (bot - top) * r.fy;
                }
            }
        }
    }
}

void downsample_avg_2x(const ::brotensor::Tensor& X,
                       int N, int C, int H, int W,
                       ::brotensor::Tensor& Y) {
    check_fp32(X, "downsample_avg_2x", "X");
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x: H and W must be even");
    }
    const int H_out = H / 2, W_out = W / 2;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                const int ih = oh * 2;
                for (int ow = 0; ow < W_out; ++ow) {
                    const int iw = ow * 2;
                    const int b = ((n * C + c) * H + ih) * W + iw;
                    Yp[((n * C + c) * H_out + oh) * W_out + ow] =
                        0.25f * (Xp[b] + Xp[b + 1] +
                                 Xp[b + W] + Xp[b + W + 1]);
                }
            }
        }
    }
}

// ─── Backward ──────────────────────────────────────────────────────────────

void upsample_nearest_2x_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  ::brotensor::Tensor& dX) {
    check_fp32(dY, "upsample_nearest_2x_backward", "dY");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // One accumulation per input pixel; gather the 4 contributing dY values.
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ih = 0; ih < H; ++ih) {
                for (int iw = 0; iw < W; ++iw) {
                    float s = 0.0f;
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            s += dYp[((n * C + c) * H_out + 2 * ih + a) *
                                     W_out + 2 * iw + b];
                        }
                    }
                    dXp[((n * C + c) * H + ih) * W + iw] = s;   // overwrite
                }
            }
        }
    }
}

void upsample_bilinear_2x_backward(const ::brotensor::Tensor& dY,
                                   int N, int C, int H, int W,
                                   ::brotensor::Tensor& dX) {
    check_fp32(dY, "upsample_bilinear_2x_backward", "dY");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // GPU memsets dX to 0 then atomic-scatters — net effect is OVERWRITE.
    const int total_in = N * cols_in;
    for (int i = 0; i < total_in; ++i) dXp[i] = 0.0f;

    // Same per-row / per-column tap hoist as the forward pass.
    struct RowTap { int y0, y1; float fy; };
    struct ColTap { int x0, x1; float fx; };
    std::vector<RowTap> row_tap(H_out);
    for (int oh = 0; oh < H_out; ++oh) {
        const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
        const int y0 = static_cast<int>(std::floor(src_y));
        row_tap[oh].fy = src_y - y0;
        row_tap[oh].y0 = clampi(y0,     0, H - 1);
        row_tap[oh].y1 = clampi(y0 + 1, 0, H - 1);
    }
    std::vector<ColTap> col_tap(W_out);
    for (int ow = 0; ow < W_out; ++ow) {
        const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
        const int x0 = static_cast<int>(std::floor(src_x));
        col_tap[ow].fx = src_x - x0;
        col_tap[ow].x0 = clampi(x0,     0, W - 1);
        col_tap[ow].x1 = clampi(x0 + 1, 0, W - 1);
    }

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int base = (n * C + c) * H;
            for (int oh = 0; oh < H_out; ++oh) {
                const RowTap& r = row_tap[oh];
                for (int ow = 0; ow < W_out; ++ow) {
                    const ColTap& cx = col_tap[ow];
                    const float w00 = (1.0f - r.fy) * (1.0f - cx.fx);
                    const float w01 = (1.0f - r.fy) * cx.fx;
                    const float w10 = r.fy * (1.0f - cx.fx);
                    const float w11 = r.fy * cx.fx;
                    const float g =
                        dYp[((n * C + c) * H_out + oh) * W_out + ow];
                    dXp[(base + r.y0) * W + cx.x0] += w00 * g;
                    dXp[(base + r.y0) * W + cx.x1] += w01 * g;
                    dXp[(base + r.y1) * W + cx.x0] += w10 * g;
                    dXp[(base + r.y1) * W + cx.x1] += w11 * g;
                }
            }
        }
    }
}

void downsample_avg_2x_backward(const ::brotensor::Tensor& dY,
                                int N, int C, int H, int W,
                                ::brotensor::Tensor& dX) {
    check_fp32(dY, "downsample_avg_2x_backward", "dY");
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x_backward: H and W must be even");
    }
    const int H_out = H / 2, W_out = W / 2;
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // One read per input pixel; each maps to a single output pixel, scale 1/4.
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ih = 0; ih < H; ++ih) {
                const int oh = ih / 2;
                for (int iw = 0; iw < W; ++iw) {
                    const int ow = iw / 2;
                    dXp[((n * C + c) * H + ih) * W + iw] =
                        0.25f * dYp[((n * C + c) * H_out + oh) * W_out + ow];
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
