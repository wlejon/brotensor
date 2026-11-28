#pragma once

// Shared host-side geometry/dtype validation and output resizing for
// 2D transposed convolution across CUDA, Vulkan, and Metal.

#include <brotensor/tensor.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brotensor::gpu_common {

[[noreturn]] inline void convt2d_fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void convt2d_need(const char* op, bool ok, const char* what) {
    if (!ok) convt2d_fail(op, what);
}

inline void convt2d_require_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        convt2d_fail(op, std::string(name) + " must be FP32");
    }
}

inline void convt2d_check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        convt2d_fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

inline int convt2d_out(int L, int stride, int padding, int output_padding, int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1) + output_padding + 1;
}

inline void convt2d_check_geometry(const char* op, int kH, int kW,
                                   int stride_h, int stride_w, int pad_h, int pad_w,
                                   int output_padding_h, int output_padding_w,
                                   int dil_h, int dil_w) {
    if (kH < 1 || kW < 1 || stride_h < 1 || stride_w < 1 || dil_h < 1 || dil_w < 1 ||
        pad_h < 0 || pad_w < 0 || output_padding_h < 0 || output_padding_w < 0) {
        convt2d_fail(op, "kH/kW/stride/dilation >=1 and pad/output_padding >=0");
    }
    if (output_padding_h >= stride_h && output_padding_h >= dil_h) {
        convt2d_fail(op, "output_padding_h must be < stride_h or < dil_h");
    }
    if (output_padding_w >= stride_w && output_padding_w >= dil_w) {
        convt2d_fail(op, "output_padding_w must be < stride_w or < dil_w");
    }
}

struct ConvT2dDims {
    int Cg_out = 0;
    int Ho = 0;
    int Wo = 0;
    int kHW = 0;
    int out_cols = 0;
};

inline ConvT2dDims convt2d_dims(const char* op, int H, int W, int C_in, int C_out,
                                int kH, int kW, int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int output_padding_h, int output_padding_w,
                                int dil_h, int dil_w, int groups) {
    convt2d_check_groups(op, C_in, C_out, groups);
    convt2d_check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                           output_padding_h, output_padding_w, dil_h, dil_w);
    ConvT2dDims d{};
    d.Cg_out = C_out / groups;
    d.Ho = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    d.Wo = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    if (d.Ho <= 0 || d.Wo <= 0) convt2d_fail(op, "non-positive output spatial size");
    d.kHW = kH * kW;
    const long long out_cols64 = static_cast<long long>(C_out) * d.Ho * d.Wo;
    if (out_cols64 > 0x7fffffffLL) convt2d_fail(op, "output too large");
    d.out_cols = static_cast<int>(out_cols64);
    return d;
}

inline ConvT2dDims prepare_convt2d_forward(const char* op,
                                           const Tensor& X, const Tensor& Wt, const Tensor* bias,
                                           int N, int C_in, int H, int W, int C_out,
                                           int kH, int kW, int stride_h, int stride_w,
                                           int pad_h, int pad_w,
                                           int output_padding_h, int output_padding_w,
                                           int dil_h, int dil_w, int groups, Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        convt2d_fail(op, "X must be FP32, FP16 or BF16");
    }
    if (Wt.dtype != X.dtype) convt2d_fail(op, "Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) convt2d_fail(op, "bias dtype must match X");
    const ConvT2dDims d = convt2d_dims(
        op, H, W, C_in, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
        output_padding_h, output_padding_w, dil_h, dil_w, groups);
    convt2d_need(op, Wt.rows == C_in && Wt.cols == d.Cg_out * d.kHW,
                 "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    convt2d_need(op, X.rows == N && X.cols == C_in * H * W, "X shape must be (N, C_in*H*W)");
    convt2d_need(op, !bias || (bias->rows == C_out && bias->cols == 1) || bias->size() == C_out,
                 "bias shape must be (C_out, 1)");
    if (Y.rows != N || Y.cols != d.out_cols || Y.dtype != X.dtype) {
        Y.resize(N, d.out_cols, X.dtype);
    }
    return d;
}

inline void convt2d_crop_input_dims(int Ho, int Wo, int kH, int kW,
                                    int stride_h, int stride_w, int pad_h, int pad_w,
                                    int dil_h, int dil_w, int& Hc, int& Wc) {
    Hc = (Ho + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    Wc = (Wo + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
}

}  // namespace brotensor::gpu_common
