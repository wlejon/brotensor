// Metal 2D transposed convolution backed by Brass MIR kernels:
//   conv_transpose2d_forward         — GEMM + MI_COL2IM2D (vision.mm), fallback MirKind::ConvDirect (mode 1)
//   conv_transpose2d_backward_input  — MirKind::ConvDirect (mode 0)
//   conv_transpose2d_backward_weight — ordinary conv2d_backward_weight (+ bottom/right X pad when cropped)
//   conv_transpose2d_backward_bias   — conv2d_backward_bias (MirKind::Gnorm GN_BIAS_GRAD)

#include "mtl_mir_cache.h"

#include <brotensor/ops/spatial.h>
#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

bool conv_transpose2d_gemm(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int sh, int sw, int ph, int pw,
                           int dh, int dw, int groups, int Ho, int Wo, Tensor& Y);
void conv2d_backward_weight(const Tensor& X, const Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups, Tensor& dWt);
void conv2d_backward_bias(const Tensor& dY, int N, int C_out, int H_out, int W_out, Tensor& dB);

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32");
}

void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

void check_geometry(const char* op, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int output_padding_h, int output_padding_w, int dil_h, int dil_w) {
    if (kH < 1 || kW < 1 || stride_h < 1 || stride_w < 1 || dil_h < 1 || dil_w < 1 ||
        pad_h < 0 || pad_w < 0 || output_padding_h < 0 || output_padding_w < 0) {
        fail(op, "kH/kW/stride/dilation >=1 and pad/output_padding >=0");
    }
    if (output_padding_h >= stride_h && output_padding_h >= dil_h) {
        fail(op, "output_padding_h must be < stride_h or < dil_h");
    }
    if (output_padding_w >= stride_w && output_padding_w >= dil_w) {
        fail(op, "output_padding_w must be < stride_w or < dil_w");
    }
}

inline int convt2d_out(int L, int stride, int padding, int output_padding, int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1) + output_padding + 1;
}

}  // namespace

void conv_transpose2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                              int N, int C_in, int H, int W, int C_out, int kH, int kW,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups, Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        fail(op, "X must be FP32, FP16 or BF16");
    }
    if (Wt.dtype != X.dtype) fail(op, "Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) fail(op, "bias dtype must match X");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);
    const int Ho = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int Wo = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (Ho <= 0 || Wo <= 0) fail(op, "non-positive output spatial size");
    const int Cg_out = C_out / groups;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kH * kW)
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    if (X.rows != N || X.cols != C_in * H * W)
        fail(op, "X shape must be (N, C_in*H*W)");
    if (bias && (bias->rows != C_out || bias->cols != 1))
        fail(op, "bias shape must be (C_out, 1)");
    const int out_cols = C_out * Ho * Wo;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) Y.resize(N, out_cols, X.dtype);
    if (N == 0 || out_cols == 0) return;
    if (conv_transpose2d_gemm(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                              stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                              groups, Ho, Wo, Y)) {
        return;
    }
    mir_dispatch_conv_direct(
        1u, X.dtype, op,
        X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, Ho, Wo,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv_transpose2d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int H, int W, int C_out, int kH, int kW,
                                     int stride_h, int stride_w, int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups, Tensor& dX) {
    const char* op = "conv_transpose2d_backward_input";
    req_fp32(op, Wt, "Wt");
    req_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);
    const int Ho = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int Wo = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (Ho <= 0 || Wo <= 0) fail(op, "non-positive output spatial size");
    const int Cg_out = C_out / groups;
    if (Wt.rows != C_in || Wt.cols != Cg_out * kH * kW)
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    if (dY.rows != N || dY.cols != C_out * Ho * Wo)
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, in_cols, Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;
    mir_dispatch_conv_direct(
        0u, Dtype::FP32, op,
        dY.data, Wt.data, dX.data, nullptr,
        N, C_out, 1, Ho, Wo, C_in, 1, H, W,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv_transpose2d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int H, int W, int C_out, int kH, int kW,
                                      int stride_h, int stride_w, int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups, Tensor& dWt) {
    const char* op = "conv_transpose2d_backward_weight";
    req_fp32(op, X, "X");
    req_fp32(op, dY, "dY");
    req_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);
    const int Ho = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    const int Wo = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (Ho <= 0 || Wo <= 0) fail(op, "non-positive output spatial size");
    const int Cg_out = C_out / groups;
    const int kHW = kH * kW;
    if (dWt.rows != C_in || dWt.cols != Cg_out * kHW)
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kH*kW)");
    if (X.rows != N || X.cols != C_in * H * W)
        fail(op, "X shape must be (N, C_in*H*W)");
    if (dY.rows != N || dY.cols != C_out * Ho * Wo)
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    if (C_in == 0 || Cg_out == 0 || kHW == 0 || N == 0) return;

    const int Hc = (Ho + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int Wc = (Wo + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (Hc == H && Wc == W) {
        conv2d_backward_weight(dY, X, N, C_out, Ho, Wo, C_in, kH, kW,
                               stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
        return;
    }
    Tensor Xp = Tensor::empty_on(X.device, N, C_in * Hc * Wc, X.dtype);
    ::brotensor::pad2d_forward(X, N, C_in, H, W, 0, Hc - H, 0, Wc - W, 0, Xp);
    conv2d_backward_weight(dY, Xp, N, C_out, Ho, Wo, C_in, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
}

void conv_transpose2d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int H_out, int W_out, Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    req_fp32(op, dY, "dY");
    req_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1) fail(op, "dB shape must be (C_out, 1)");
    if (dY.rows != N || dY.cols != C_out * H_out * W_out)
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

}  // namespace brotensor::detail::metal
