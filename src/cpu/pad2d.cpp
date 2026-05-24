// ─── CPU 2D padding ─────────────────────────────────────────────────────────
//
// FP32 scalar host implementation of pad2d_forward / pad2d_backward — the
// image (NCHW) analogue of pad1d in conv1d.cpp. Same mode convention:
//   0 = zero, 1 = reflect (no edge repeat; requires pad < H/W on that axis),
//   2 = replicate (clamp to edge sample).
//
// Memory layout (NCHW flat — matches resample.cpp / interp2d.cpp):
//   X / dX : ((n*C + c)*H     + h)*W + w
//   Y / dY : ((n*C + c)*H_pad + h)*W_pad + w
//   with H_pad = H + pad_top + pad_bottom,  W_pad = W + pad_left + pad_right.
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   pad2d_forward  — Y  OVERWRITTEN.
//   pad2d_backward — dX OVERWRITTEN. Adjoint = scatter each output gradient
//                     onto the input sample it read; for reflect / replicate
//                     several output positions may collapse onto the same
//                     input position and those gradients sum.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) +
                 " must be FP32 (CPU backend is FP32-only)");
    }
}

// Map an output position p in [0, L_pad) along one axis to a source index in
// [0, L) for the given mode, or -1 for a zero-padded slot. Verbatim copy of
// the pad1d_src helper in conv1d.cpp — the per-axis logic is identical.
inline int pad_src(int p, int L, int pad_left, int mode) {
    const int rel = p - pad_left;
    if (rel >= 0 && rel < L) return rel;     // interior
    if (mode == 0) return -1;                // zero
    if (mode == 2) return rel < 0 ? 0 : L - 1;  // replicate (clamp)
    // mode == 1: reflect without repeating the edge sample (numpy 'reflect').
    if (L == 1) return 0;
    int q = rel;
    const int period = 2 * (L - 1);
    q %= period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

inline void check_args(const char* op,
                       int N, int C, int H, int W,
                       int pad_top, int pad_bottom,
                       int pad_left, int pad_right, int mode) {
    if (N < 0 || C < 1 || H < 1 || W < 1) {
        fail(op, "C/H/W must be >=1 and N >=0");
    }
    if (pad_top < 0 || pad_bottom < 0 ||
        pad_left < 0 || pad_right < 0) {
        fail(op, "pad counts must be >=0");
    }
    if (mode < 0 || mode > 2) {
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    }
    if (mode == 1) {
        if (pad_top >= H || pad_bottom >= H) {
            fail(op, "reflect padding requires pad_top and pad_bottom < H");
        }
        if (pad_left >= W || pad_right >= W) {
            fail(op, "reflect padding requires pad_left and pad_right < W");
        }
    }
}

} // namespace

// ─── pad2d_forward ─────────────────────────────────────────────────────────

void pad2d_forward(const ::brotensor::Tensor& X,
                   int N, int C, int H, int W,
                   int pad_top, int pad_bottom,
                   int pad_left, int pad_right, int mode,
                   ::brotensor::Tensor& Y) {
    const char* op = "pad2d_forward";
    check_fp32(X, op, "X");
    check_args(op, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode);
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must be (N, C*H*W)");
    }
    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    const int cols_out = C * H_pad * W_pad;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0 || C == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan =
                Xp + (static_cast<long>(n) * C + c) * H * W;
            float* y_chan =
                Yp + (static_cast<long>(n) * C + c) * H_pad * W_pad;
            for (int p = 0; p < H_pad; ++p) {
                const int src_h = pad_src(p, H, pad_top, mode);
                float* y_row = y_chan + static_cast<long>(p) * W_pad;
                if (src_h < 0) {
                    // Whole row is zero-padded.
                    for (int q = 0; q < W_pad; ++q) y_row[q] = 0.0f;
                    continue;
                }
                const float* x_row = x_chan + static_cast<long>(src_h) * W;
                for (int q = 0; q < W_pad; ++q) {
                    const int src_w = pad_src(q, W, pad_left, mode);
                    y_row[q] = src_w < 0 ? 0.0f : x_row[src_w];
                }
            }
        }
    }
}

// ─── pad2d_backward ────────────────────────────────────────────────────────

void pad2d_backward(const ::brotensor::Tensor& dY,
                    int N, int C, int H, int W,
                    int pad_top, int pad_bottom,
                    int pad_left, int pad_right, int mode,
                    ::brotensor::Tensor& dX) {
    const char* op = "pad2d_backward";
    check_fp32(dY, op, "dY");
    check_args(op, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode);
    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    if (dY.rows != N || dY.cols != C * H_pad * W_pad) {
        fail(op, "dY shape must be (N, C*(H+pt+pb)*(W+pl+pr))");
    }
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || C == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Zero dX, then scatter each output gradient onto its source input pixel.
    const long total_in = static_cast<long>(N) * cols_in;
    for (long i = 0; i < total_in; ++i) dXp[i] = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* dy_chan =
                dYp + (static_cast<long>(n) * C + c) * H_pad * W_pad;
            float* dx_chan =
                dXp + (static_cast<long>(n) * C + c) * H * W;
            for (int p = 0; p < H_pad; ++p) {
                const int src_h = pad_src(p, H, pad_top, mode);
                if (src_h < 0) continue;
                const float* dy_row = dy_chan + static_cast<long>(p) * W_pad;
                float* dx_row = dx_chan + static_cast<long>(src_h) * W;
                for (int q = 0; q < W_pad; ++q) {
                    const int src_w = pad_src(q, W, pad_left, mode);
                    if (src_w >= 0) dx_row[src_w] += dy_row[q];
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
