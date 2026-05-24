// ─── CPU window partition / reverse ─────────────────────────────────────────
//
// FP32 scalar host implementations of the SAM-style windowed-attention
// layout pair. Both ops are pure rearrangements (no math) and are exact
// inverses of each other — neither has a separate _backward op; callers
// apply the other op to map gradients back through the layout change.
//
// ── Layout ─────────────────────────────────────────────────────────────────
//   Input NCHW:    X(N, C*H*W) at ((n*C + c)*H + h)*W + w
//   Windowed:      Y(N*nw_h*nw_w, C*window*window)
//                  row index = n*nw_h*nw_w + nh*nw_w + nw
//                  within-row = (c*window + lh)*window + lw
//   With nw_h = H/window, nw_w = W/window, and (h, w) = (nh*window + lh,
//   nw*window + lw).
//
// This is the NCHW analogue of SAM's NHWC window_partition. We keep
// channels-contiguous within each window so subsequent attention sees the
// expected (B*nw, C, window, window) layout without an extra transpose.
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   Both ops OVERWRITE the output. No accumulation — they're pure copies.

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

inline void check_args(const char* op, int N, int C, int H, int W, int window) {
    if (N < 0 || C < 1 || H < 1 || W < 1) {
        fail(op, "C/H/W must be >=1 and N >=0");
    }
    if (window < 1) {
        fail(op, "window must be >=1");
    }
    if (H % window != 0 || W % window != 0) {
        fail(op, "H and W must be multiples of window (use pad2d first if "
                 "the input doesn't align)");
    }
}

} // namespace

void window_partition_forward(const ::brotensor::Tensor& X,
                              int N, int C, int H, int W, int window,
                              ::brotensor::Tensor& Y) {
    const char* op = "window_partition_forward";
    check_fp32(X, op, "X");
    check_args(op, N, C, H, W, window);
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must be (N, C*H*W)");
    }
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int B_out = N * nw_h * nw_w;
    const int cols_out = C * window * window;
    if (Y.rows != B_out || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(B_out, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int nh = 0; nh < nw_h; ++nh) {
            for (int nw = 0; nw < nw_w; ++nw) {
                const int b_out = (n * nw_h + nh) * nw_w + nw;
                float* y_row =
                    Yp + static_cast<long>(b_out) * cols_out;
                for (int c = 0; c < C; ++c) {
                    const float* x_chan =
                        Xp + (static_cast<long>(n) * C + c) * H * W;
                    for (int lh = 0; lh < window; ++lh) {
                        const int h = nh * window + lh;
                        const float* x_row =
                            x_chan + static_cast<long>(h) * W + nw * window;
                        float* y_block =
                            y_row + (static_cast<long>(c) * window + lh)
                                  * window;
                        for (int lw = 0; lw < window; ++lw) {
                            y_block[lw] = x_row[lw];
                        }
                    }
                }
            }
        }
    }
}

void window_reverse_forward(const ::brotensor::Tensor& X,
                            int N, int C, int H, int W, int window,
                            ::brotensor::Tensor& Y) {
    const char* op = "window_reverse_forward";
    check_fp32(X, op, "X");
    check_args(op, N, C, H, W, window);
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int B_in = N * nw_h * nw_w;
    const int cols_in = C * window * window;
    if (X.rows != B_in || X.cols != cols_in) {
        fail(op, "X shape must be (N*nw_h*nw_w, C*window*window)");
    }
    const int cols_out = C * H * W;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int nh = 0; nh < nw_h; ++nh) {
            for (int nw = 0; nw < nw_w; ++nw) {
                const int b_in = (n * nw_h + nh) * nw_w + nw;
                const float* x_row =
                    Xp + static_cast<long>(b_in) * cols_in;
                for (int c = 0; c < C; ++c) {
                    float* y_chan =
                        Yp + (static_cast<long>(n) * C + c) * H * W;
                    for (int lh = 0; lh < window; ++lh) {
                        const int h = nh * window + lh;
                        float* y_row =
                            y_chan + static_cast<long>(h) * W + nw * window;
                        const float* x_block =
                            x_row + (static_cast<long>(c) * window + lh)
                                  * window;
                        for (int lw = 0; lw < window; ++lw) {
                            y_row[lw] = x_block[lw];
                        }
                    }
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
