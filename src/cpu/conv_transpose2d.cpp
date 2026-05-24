// ─── CPU 2D transposed convolution ──────────────────────────────────────────
//
// FP32 scalar host implementations of:
//   conv_transpose2d_forward / _backward_input / _backward_weight / _backward_bias
//
// The 2D counterpart of conv_transpose1d in conv1d.cpp, with H and W
// independently rescaled. The "learned upsample" primitive — SAM's mask
// decoder uses two 4x conv-transposes back-to-back; DPT depth heads use
// a 2x conv-transpose; many segmentation decoders rely on these instead
// of the cheaper bilinear upsample.
//
// ── Layout (NCHW) ───────────────────────────────────────────────────────────
//   X  / dX : (N, C_in*H*W)        flat index ((n*C_in + c)*H + h)*W + w
//   Y  / dY : (N, C_out*H_out*W_out)
//   Wt / dWt: (C_in, (C_out/groups)*kH*kW)  — input-channel-major
//             flat index (c_in*Cg_out + oc_local) * (kH*kW) + (kh*kW + kw)
//   bias    : (C_out, 1) or null
//
// ── Accumulation (matches the conv2d / conv_transpose1d contract) ──────────
//   *_forward / *_backward_input — output OVERWRITTEN.
//   _backward_weight / _bias      — dWt / dB ACCUMULATE (+=); caller zeros
//                                   them first.
//
// Output spatial dims (torch ConvTranspose2d):
//   H_out = (H-1)*stride_h - 2*pad_h + dil_h*(kH-1) + output_padding_h + 1
//   W_out = (W-1)*stride_w - 2*pad_w + dil_w*(kW-1) + output_padding_w + 1

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

void require_fp32(const char* op, const ::brotensor::Tensor& t,
                  const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
    }
}

inline int convt2d_out(int L, int stride, int padding, int output_padding,
                       int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

void check_geometry(const char* op, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int output_padding_h, int output_padding_w,
                    int dil_h, int dil_w) {
    if (kH < 1 || kW < 1 || stride_h < 1 || stride_w < 1
        || dil_h < 1 || dil_w < 1 || pad_h < 0 || pad_w < 0
        || output_padding_h < 0 || output_padding_w < 0) {
        fail(op, "kH/kW/stride/dilation >=1 and pad/output_padding >=0");
    }
    if (output_padding_h >= stride_h && output_padding_h >= dil_h) {
        fail(op, "output_padding_h must be < stride_h or < dil_h");
    }
    if (output_padding_w >= stride_w && output_padding_w >= dil_w) {
        fail(op, "output_padding_w must be < stride_w or < dil_w");
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_forward
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              ::brotensor::Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    require_fp32(op, X, "X");
    require_fp32(op, Wt, "Wt");
    if (bias) require_fp32(op, *bias, "bias");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h,
                                  dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w,
                                  dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");

    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must be (N, C_in*H*W)");
    }
    if (bias && (bias->rows != C_out || bias->cols != 1)) {
        fail(op, "bias shape must be (C_out, 1)");
    }

    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, out_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || out_cols == 0) return;

    const float* Xp = X.host_f32();
    const float* Wp = Wt.host_f32();
    const float* Bp = bias ? bias->host_f32() : nullptr;
    float* Yp = Y.host_f32_mut();

    // Seed every output pixel with its channel bias.
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            const float bv = Bp ? Bp[oc] : 0.0f;
            float* y_chan =
                Yp + (static_cast<long>(n) * C_out + oc) * H_out * W_out;
            for (int o = 0; o < H_out * W_out; ++o) y_chan[o] = bv;
        }
    }

    // Scatter-add. Input pixel (n, c_in, h, w) reaches output pixel
    //   ho = h*stride_h - pad_h + kh*dil_h
    //   wo = w*stride_w - pad_w + kw*dil_w
    // in each output channel of c_in's group.
    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            const int oc_base = g * Cg_out;
            const float* x_chan =
                Xp + (static_cast<long>(n) * C_in + c_in) * H * W;
            for (int h = 0; h < H; ++h) {
                const int ho_origin = h * stride_h - pad_h;
                for (int w = 0; w < W; ++w) {
                    const float xv = x_chan[static_cast<long>(h) * W + w];
                    if (xv == 0.0f) continue;
                    const int wo_origin = w * stride_w - pad_w;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ho = ho_origin + kh * dil_h;
                        if (ho < 0 || ho >= H_out) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int wo = wo_origin + kw * dil_w;
                            if (wo < 0 || wo >= W_out) continue;
                            for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                                const int oc = oc_base + oc_local;
                                const int w_idx =
                                    (c_in * Cg_out + oc_local) * kHW
                                    + kh * kW + kw;
                                Yp[(static_cast<long>(n) * C_out + oc)
                                   * H_out * W_out + ho * W_out + wo]
                                    += xv * Wp[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_backward_input
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     ::brotensor::Tensor& dX) {
    const char* op = "conv_transpose2d_backward_input";
    require_fp32(op, Wt, "Wt");
    require_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h,
                                  dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w,
                                  dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kHW) {
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols
        || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;

    const float* Wp  = Wt.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();

    // Adjoint of the transposed-conv scatter is a plain gather conv.
    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            const int oc_base = g * Cg_out;
            for (int h = 0; h < H; ++h) {
                const int ho_origin = h * stride_h - pad_h;
                for (int w = 0; w < W; ++w) {
                    const int wo_origin = w * stride_w - pad_w;
                    float acc = 0.0f;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ho = ho_origin + kh * dil_h;
                        if (ho < 0 || ho >= H_out) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int wo = wo_origin + kw * dil_w;
                            if (wo < 0 || wo >= W_out) continue;
                            for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
                                const int oc = oc_base + oc_local;
                                const int w_idx =
                                    (c_in * Cg_out + oc_local) * kHW
                                    + kh * kW + kw;
                                const long dy_idx =
                                    (static_cast<long>(n) * C_out + oc)
                                    * H_out * W_out + ho * W_out + wo;
                                acc += dYp[dy_idx] * Wp[w_idx];
                            }
                        }
                    }
                    dXp[(static_cast<long>(n) * C_in + c_in) * H * W
                        + h * W + w] = acc;
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_backward_weight
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      ::brotensor::Tensor& dWt) {
    const char* op = "conv_transpose2d_backward_weight";
    require_fp32(op, X, "X");
    require_fp32(op, dY, "dY");
    require_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = convt2d_out(H, stride_h, pad_h, output_padding_h,
                                  dil_h, kH);
    const int W_out = convt2d_out(W, stride_w, pad_w, output_padding_w,
                                  dil_w, kW);
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (dWt.rows != C_in || dWt.cols != Cg_out * kHW) {
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kH*kW)");
    }
    if (X.rows != N || X.cols != C_in * H * W) {
        fail(op, "X shape must be (N, C_in*H*W)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    if (C_in == 0 || Cg_out == 0 || kHW == 0) return;

    const float* Xp  = X.host_f32();
    const float* dYp = dY.host_f32();
    float* dWp = dWt.host_f32_mut();

    // One accumulation per weight element; += into dWt (caller zeroed it).
    for (int c_in = 0; c_in < C_in; ++c_in) {
        const int g = c_in / Cg_in;
        const int oc_base = g * Cg_out;
        for (int oc_local = 0; oc_local < Cg_out; ++oc_local) {
            const int oc = oc_base + oc_local;
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    float acc = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        const float* x_chan =
                            Xp + (static_cast<long>(n) * C_in + c_in) * H * W;
                        const float* dy_chan =
                            dYp + (static_cast<long>(n) * C_out + oc)
                                * H_out * W_out;
                        for (int h = 0; h < H; ++h) {
                            const int ho = h * stride_h - pad_h + kh * dil_h;
                            if (ho < 0 || ho >= H_out) continue;
                            for (int w = 0; w < W; ++w) {
                                const int wo =
                                    w * stride_w - pad_w + kw * dil_w;
                                if (wo < 0 || wo >= W_out) continue;
                                acc += x_chan[h * W + w]
                                     * dy_chan[ho * W_out + wo];
                            }
                        }
                    }
                    dWp[(c_in * Cg_out + oc_local) * kHW + kh * kW + kw]
                        += acc;
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_backward_bias
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    require_fp32(op, dY, "dY");
    require_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) {
        fail(op, "dB shape must be (C_out, 1)");
    }
    if (dY.rows != N || dY.cols != C_out * H_out * W_out) {
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    const float* dYp = dY.host_f32();
    float* dBp = dB.host_f32_mut();

    // Per-output-channel sum over (N, H_out, W_out); += into dB.
    for (int oc = 0; oc < C_out; ++oc) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* dy_chan =
                dYp + (static_cast<long>(n) * C_out + oc) * H_out * W_out;
            for (int i = 0; i < H_out * W_out; ++i) acc += dy_chan[i];
        }
        dBp[oc] += acc;
    }
}

} // namespace brotensor::detail::cpu
