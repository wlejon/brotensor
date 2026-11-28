// Metal 2D transposed convolution backed by Brass MIR kernels:
//   conv_transpose2d_forward         — GEMM + MI_COL2IM2D (vision.mm), fallback MirKind::ConvDirect (mode 1)
//   conv_transpose2d_backward_input  — MirKind::ConvDirect (mode 0)
//   conv_transpose2d_backward_weight — ordinary conv2d_backward_weight (+ bottom/right X pad when cropped)
//   conv_transpose2d_backward_bias   — conv2d_backward_bias (MirKind::Gnorm GN_BIAS_GRAD)

#include "mtl_mir_cache.h"
#include "../gpu_common/conv_transpose2d.h"

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

void conv_transpose2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                              int N, int C_in, int H, int W, int C_out, int kH, int kW,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups, Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    const gpu_common::ConvT2dDims d = gpu_common::prepare_convt2d_forward(
        op, X, Wt, bias, N, C_in, H, W, C_out, kH, kW, stride_h, stride_w,
        pad_h, pad_w, output_padding_h, output_padding_w, dil_h, dil_w, groups, Y);
    if (N == 0 || d.out_cols == 0) return;
    if (conv_transpose2d_gemm(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                              stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                              groups, d.Ho, d.Wo, Y)) {
        return;
    }
    mir_dispatch_conv_direct(
        1u, X.dtype, op,
        X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, d.Ho, d.Wo,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv_transpose2d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int H, int W, int C_out, int kH, int kW,
                                     int stride_h, int stride_w, int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups, Tensor& dX) {
    const char* op = "conv_transpose2d_backward_input";
    gpu_common::convt2d_require_fp32(op, Wt, "Wt");
    gpu_common::convt2d_require_fp32(op, dY, "dY");
    const gpu_common::ConvT2dDims d = gpu_common::convt2d_dims(
        op, H, W, C_in, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
        output_padding_h, output_padding_w, dil_h, dil_w, groups);
    gpu_common::convt2d_need(op, Wt.rows == C_in && Wt.cols == d.Cg_out * d.kHW,
                             "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    gpu_common::convt2d_need(op, dY.rows == N && dY.cols == d.out_cols,
                             "dY shape must be (N, C_out*H_out*W_out)");
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, in_cols, Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;
    mir_dispatch_conv_direct(
        0u, Dtype::FP32, op,
        dY.data, Wt.data, dX.data, nullptr,
        N, C_out, 1, d.Ho, d.Wo, C_in, 1, H, W,
        1, kH, kW, 1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

void conv_transpose2d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int H, int W, int C_out, int kH, int kW,
                                      int stride_h, int stride_w, int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups, Tensor& dWt) {
    const char* op = "conv_transpose2d_backward_weight";
    gpu_common::convt2d_require_fp32(op, X, "X");
    gpu_common::convt2d_require_fp32(op, dY, "dY");
    gpu_common::convt2d_require_fp32(op, dWt, "dWt");
    const gpu_common::ConvT2dDims d = gpu_common::convt2d_dims(
        op, H, W, C_in, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
        output_padding_h, output_padding_w, dil_h, dil_w, groups);
    gpu_common::convt2d_need(op, dWt.rows == C_in && dWt.cols == d.Cg_out * d.kHW,
                             "dWt shape must be (C_in, (C_out/groups)*kH*kW)");
    gpu_common::convt2d_need(op, X.rows == N && X.cols == C_in * H * W, "X shape must be (N, C_in*H*W)");
    gpu_common::convt2d_need(op, dY.rows == N && dY.cols == d.out_cols,
                             "dY shape must be (N, C_out*H_out*W_out)");
    if (C_in == 0 || d.Cg_out == 0 || d.kHW == 0 || N == 0) return;

    int Hc = 0, Wc = 0;
    gpu_common::convt2d_crop_input_dims(d.Ho, d.Wo, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, Hc, Wc);
    if (Hc == H && Wc == W) {
        conv2d_backward_weight(dY, X, N, C_out, d.Ho, d.Wo, C_in, kH, kW,
                               stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
        return;
    }
    Tensor Xp = Tensor::empty_on(X.device, N, C_in * Hc * Wc, X.dtype);
    ::brotensor::pad2d_forward(X, N, C_in, H, W, 0, Hc - H, 0, Wc - W, 0, Xp);
    conv2d_backward_weight(dY, Xp, N, C_out, d.Ho, d.Wo, C_in, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
}

void conv_transpose2d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int H_out, int W_out, Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    gpu_common::convt2d_require_fp32(op, dY, "dY");
    gpu_common::convt2d_require_fp32(op, dB, "dB");
    gpu_common::convt2d_need(op, dB.rows == C_out && dB.cols == 1, "dB shape must be (C_out, 1)");
    gpu_common::convt2d_need(op, dY.rows == N && dY.cols == C_out * H_out * W_out,
                             "dY shape must be (N, C_out*H_out*W_out)");
    conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

}  // namespace brotensor::detail::metal
