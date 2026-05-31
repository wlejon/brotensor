// ─── CPU 2D neighborhood unfold (im2col, spatial-preserving) ────────────────
//
// FP32 scalar host implementation of unfold2d_forward. For every output pixel
// it gathers the kH×kW window around the corresponding input position into a
// dedicated channel block — the "keep the spatial grid, add a neighbor axis"
// flavour of im2col (DSINE NRN propagation, neighborhood attention, guided /
// bilateral filtering), as opposed to torch.nn.Unfold's column-collapse form.
//
// Layout (NCHW flat, matches pad2d.cpp / interp2d.cpp):
//   X : ((n*C + c)*H + h)*W + w
//   Y : ((n*C + (c*kK + k))*H_out + oy)*W_out + ox
//   with kK = kH*kW, k = ky*kW + kx, and
//     H_out = (H + pad_top + pad_bottom - kH)/stride_h + 1   (W_out analogous).
//   Y[n, c, k, oy, ox] = X[n, c, oy*stride_h - pad_top + ky,
//                                ox*stride_w - pad_left + kx]
//   with out-of-range source positions resolved by `mode`:
//     0 = zero, 1 = reflect (no edge repeat), 2 = replicate (clamp to edge).
//   For stride 1 and pad (kH-1)/2 this is the same-size neighborhood unfold
//   (H_out == H, W_out == W) DSINE's get_unfold uses with kH=kW=5, mode=2.
//
// ── ACCUMULATION ──  Y OVERWRITTEN. Inference-only: no backward (the bro
//                     pipeline never trains through this), so no adjoint slot.

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
    if (t.dtype != Dtype::FP32)
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
}

// Map an output-window position to a source index in [0, L), or -1 for a
// zero-padded slot. Identical convention to pad2d's pad_src.
inline int unf_src(int coord, int L, int mode) {
    if (coord >= 0 && coord < L) return coord;
    if (mode == 0) return -1;                       // zero
    if (mode == 2) return coord < 0 ? 0 : L - 1;    // replicate (clamp)
    if (L == 1) return 0;                           // reflect, degenerate
    int q = coord;
    const int period = 2 * (L - 1);
    q %= period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

} // namespace

void unfold2d_forward(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W,
                      int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_top, int pad_bottom,
                      int pad_left, int pad_right,
                      int mode,
                      ::brotensor::Tensor& Y) {
    const char* op = "unfold2d_forward";
    check_fp32(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH/kW must be >=1");
    if (stride_h < 1 || stride_w < 1) fail(op, "stride must be >=1");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0)
        fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2)
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    const int W_out = (W + pad_left + pad_right - kW) / stride_w + 1;
    if (H_out < 1 || W_out < 1)
        fail(op, "kernel/padding/stride yield empty output");
    const int kK = kH * kW;
    const int cols_out = C * kK * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32)
        Y.resize(N, cols_out, Dtype::FP32);
    if (N == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan = Xp + (static_cast<long>(n) * C + c) * H * W;
            for (int ky = 0; ky < kH; ++ky) {
                for (int kx = 0; kx < kW; ++kx) {
                    const int k = ky * kW + kx;
                    float* y_blk = Yp +
                        ((static_cast<long>(n) * C + c) * kK + k) *
                        H_out * W_out;
                    for (int oy = 0; oy < H_out; ++oy) {
                        const int sy = unf_src(oy * stride_h - pad_top + ky,
                                               H, mode);
                        float* y_row = y_blk + static_cast<long>(oy) * W_out;
                        if (sy < 0) {
                            for (int ox = 0; ox < W_out; ++ox) y_row[ox] = 0.0f;
                            continue;
                        }
                        const float* x_row = x_chan + static_cast<long>(sy) * W;
                        for (int ox = 0; ox < W_out; ++ox) {
                            const int sx = unf_src(ox * stride_w - pad_left + kx,
                                                   W, mode);
                            y_row[ox] = sx < 0 ? 0.0f : x_row[sx];
                        }
                    }
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
