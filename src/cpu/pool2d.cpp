// ─── CPU 2D pooling: adaptive_avg_pool2d + max_pool2d ──────────────────────
//
// FP32 scalar host implementations for the two pooling primitives needed by
// modern vision encoders / detectors:
//
//   * adaptive_avg_pool2d — output (H_out, W_out) is the runtime parameter;
//     each output pixel averages a variable-size input region defined by
//     PyTorch's adaptive formula (floor / ceil at the boundaries). Used by
//     SegFormer / Mask2Former decoder aggregation and detection heads.
//
//   * max_pool2d — standard kernel/stride/pad max pool. Padding pixels are
//     treated as -inf so they never win. Forward returns Y and a per-output
//     INT32 flat-spatial Idx into the per-channel HxW plane; backward uses
//     Idx to scatter dY without rescanning the kernel.
//
// Memory layout (NCHW flat):
//   X / dX : ((n*C + c)*H     + h)*W     + w
//   Y / dY : ((n*C + c)*H_out + h)*W_out + w
//   Idx     same layout as Y, INT32.
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   adaptive_avg_pool2d_forward  — Y  OVERWRITTEN.
//   adaptive_avg_pool2d_backward — dX OVERWRITTEN (zero-then-scatter; many
//                                  output regions overlap the same input
//                                  pixel and their contributions sum).
//   max_pool2d_forward           — Y  OVERWRITTEN, Idx OVERWRITTEN.
//   max_pool2d_backward          — dX OVERWRITTEN (zero-then-scatter; with
//                                  stride < kernel size overlapping kernels
//                                  may pick the same input pixel from
//                                  multiple outputs — those dY values sum).

#include <brotensor/tensor.h>

#include <cmath>
#include <limits>
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

// PyTorch adaptive-pool window endpoints for axis len L -> L_out.
inline void adaptive_window(int o, int L, int L_out, int& start, int& end) {
    // start = floor(o * L / L_out),  end = ceil((o+1) * L / L_out)
    start = (o * L) / L_out;
    end   = ((o + 1) * L + L_out - 1) / L_out;
    if (end > L) end = L;
    if (start < 0) start = 0;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  adaptive_avg_pool2d
// ═══════════════════════════════════════════════════════════════════════════

void adaptive_avg_pool2d_forward(const ::brotensor::Tensor& X,
                                 int N, int C, int H, int W,
                                 int H_out, int W_out,
                                 ::brotensor::Tensor& Y) {
    const char* op = "adaptive_avg_pool2d_forward";
    check_fp32(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 1 || W_out < 1)
        fail(op, "H_out and W_out must be >= 1");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0) return;

    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan =
                Xp + (static_cast<long>(n) * C + c) * H * W;
            float* y_chan =
                Yp + (static_cast<long>(n) * C + c) * H_out * W_out;
            for (int oh = 0; oh < H_out; ++oh) {
                int h0, h1;
                adaptive_window(oh, H, H_out, h0, h1);
                for (int ow = 0; ow < W_out; ++ow) {
                    int w0, w1;
                    adaptive_window(ow, W, W_out, w0, w1);
                    const int area = (h1 - h0) * (w1 - w0);
                    double acc = 0.0;
                    for (int h = h0; h < h1; ++h) {
                        const float* row = x_chan + static_cast<long>(h) * W;
                        for (int w = w0; w < w1; ++w) acc += row[w];
                    }
                    y_chan[static_cast<long>(oh) * W_out + ow] =
                        static_cast<float>(acc / area);
                }
            }
        }
    }
}

void adaptive_avg_pool2d_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out,
                                  ::brotensor::Tensor& dX) {
    const char* op = "adaptive_avg_pool2d_backward";
    check_fp32(dY, op, "dY");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 1 || W_out < 1)
        fail(op, "H_out and W_out must be >= 1");
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0) return;

    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    const long total_in = static_cast<long>(N) * cols_in;
    for (long i = 0; i < total_in; ++i) dXp[i] = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* dy_chan =
                dYp + (static_cast<long>(n) * C + c) * H_out * W_out;
            float* dx_chan =
                dXp + (static_cast<long>(n) * C + c) * H * W;
            for (int oh = 0; oh < H_out; ++oh) {
                int h0, h1;
                adaptive_window(oh, H, H_out, h0, h1);
                for (int ow = 0; ow < W_out; ++ow) {
                    int w0, w1;
                    adaptive_window(ow, W, W_out, w0, w1);
                    const int area = (h1 - h0) * (w1 - w0);
                    const float g =
                        dy_chan[static_cast<long>(oh) * W_out + ow] /
                        static_cast<float>(area);
                    for (int h = h0; h < h1; ++h) {
                        float* row = dx_chan + static_cast<long>(h) * W;
                        for (int w = w0; w < w1; ++w) row[w] += g;
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  max_pool2d
// ═══════════════════════════════════════════════════════════════════════════

void max_pool2d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        ::brotensor::Tensor& Y, ::brotensor::Tensor& Idx) {
    const char* op = "max_pool2d_forward";
    check_fp32(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH and kW must be >= 1");
    if (stride_h < 1 || stride_w < 1) fail(op, "strides must be >= 1");
    if (pad_h < 0 || pad_w < 0) fail(op, "pads must be >= 0");
    if (kH > H + 2 * pad_h || kW > W + 2 * pad_w)
        fail(op, "kernel larger than padded input");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (Idx.rows != N || Idx.cols != cols_out || Idx.dtype != Dtype::INT32) {
        Idx.resize(N, cols_out, Dtype::INT32);
    }
    if (N == 0) return;

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    const float* Xp = X.host_f32();
    float* Yp = Y.host_f32_mut();
    int32_t* Ip = static_cast<int32_t*>(Idx.data);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* x_chan =
                Xp + (static_cast<long>(n) * C + c) * H * W;
            float* y_chan =
                Yp + (static_cast<long>(n) * C + c) * H_out * W_out;
            int32_t* i_chan =
                Ip + (static_cast<long>(n) * C + c) * H_out * W_out;
            for (int oh = 0; oh < H_out; ++oh) {
                const int h_base = oh * stride_h - pad_h;
                for (int ow = 0; ow < W_out; ++ow) {
                    const int w_base = ow * stride_w - pad_w;
                    float best_v = NEG_INF;
                    int32_t best_i = -1;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = h_base + kh;
                        if (ih < 0 || ih >= H) continue;
                        const float* row =
                            x_chan + static_cast<long>(ih) * W;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = w_base + kw;
                            if (iw < 0 || iw >= W) continue;
                            const float v = row[iw];
                            if (v > best_v) {
                                best_v = v;
                                best_i = ih * W + iw;
                            }
                        }
                    }
                    const long o = static_cast<long>(oh) * W_out + ow;
                    y_chan[o] = best_v;
                    i_chan[o] = best_i;
                }
            }
        }
    }
}

void max_pool2d_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& Idx,
                         int N, int C, int H, int W,
                         int H_out, int W_out,
                         ::brotensor::Tensor& dX) {
    const char* op = "max_pool2d_backward";
    check_fp32(dY, op, "dY");
    if (Idx.dtype != Dtype::INT32)
        fail(op, "Idx must be INT32 (as produced by max_pool2d_forward)");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 0 || W_out < 0)
        fail(op, "H_out and W_out must be >= 0");
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");
    if (Idx.rows != N || Idx.cols != C * H_out * W_out)
        fail(op, "Idx shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0) return;

    const float* dYp = dY.host_f32();
    const int32_t* Ip = static_cast<const int32_t*>(Idx.data);
    float* dXp = dX.host_f32_mut();

    const long total_in = static_cast<long>(N) * cols_in;
    for (long i = 0; i < total_in; ++i) dXp[i] = 0.0f;

    if (H_out == 0 || W_out == 0) return;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* dy_chan =
                dYp + (static_cast<long>(n) * C + c) * H_out * W_out;
            const int32_t* i_chan =
                Ip + (static_cast<long>(n) * C + c) * H_out * W_out;
            float* dx_chan =
                dXp + (static_cast<long>(n) * C + c) * H * W;
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    const long o = static_cast<long>(oh) * W_out + ow;
                    const int32_t idx = i_chan[o];
                    if (idx < 0) continue;  // degenerate: no valid pixel.
                    dx_chan[idx] += dy_chan[o];
                }
            }
        }
    }
}

} // namespace brotensor::detail::cpu
