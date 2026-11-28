// CUDA 2D transposed convolution forward + backward ops backed by
// direct/implicit-GEMM Brass MIR conv2d kernels.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include "../gpu_common/conv_transpose2d.h"
#include "cuda_driver.h"

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Forward declarations of intra-backend helpers.
bool conv_transpose2d_f32_gemm(const float* X, const float* Wt, const float* bias, float* Y,
                               int N, int Cin, int H, int W, int Cout, int kH, int kW, int sh,
                               int sw, int ph, int pw, int dh, int dw, int Ho, int Wo,
                               CUstream stream);

void dispatch_conv_direct(std::uint32_t mode, ::brotensor::Dtype dt,
                          const void* x, const void* w, void* y, const void* bias,
                          int N, int Cin, int T, int H, int Wd,
                          int Cout, int To, int Ho, int Wo,
                          int kT, int kH, int kW,
                          int st, int sh, int sw,
                          int pt, int ph, int pw,
                          int dt_, int dh, int dw, int groups);

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y);

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt);

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB);

void pad2d_forward(const ::brotensor::Tensor& X, int N, int C, int H, int W,
                   int pad_top, int pad_bottom, int pad_left, int pad_right,
                   int mode, ::brotensor::Tensor& Y);

void slice2d_forward(const ::brotensor::Tensor& X, int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out, ::brotensor::Tensor& Y);

namespace {

inline CUstream cur_stream() {
    return reinterpret_cast<CUstream>(::brotensor::cuda_current_stream());
}

}  // namespace

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
    const gpu_common::ConvT2dDims d = gpu_common::prepare_convt2d_forward(
        op, X, Wt, bias, N, C_in, H, W, C_out, kH, kW, stride_h, stride_w,
        pad_h, pad_w, output_padding_h, output_padding_w, dil_h, dil_w, groups, Y);
    if (N == 0 || d.out_cols == 0) return;

    if (X.dtype == ::brotensor::Dtype::FP32 && groups == 1 &&
        conv_transpose2d_f32_gemm(static_cast<const float*>(X.data),
                                  static_cast<const float*>(Wt.data),
                                  bias ? static_cast<const float*>(bias->data) : nullptr,
                                  static_cast<float*>(Y.data), N, C_in, H, W, C_out,
                                  kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h,
                                  dil_w, d.Ho, d.Wo, cur_stream())) {
        return;
    }
    dispatch_conv_direct(
        1u, X.dtype, X.data, Wt.data, Y.data, bias ? bias->data : nullptr,
        N, C_in, 1, H, W, C_out, 1, d.Ho, d.Wo, 1, kH, kW,
        1, stride_h, stride_w, 0, pad_h, pad_w, 1, dil_h, dil_w, groups);
}

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
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;

    int Hc = 0, Wc = 0;
    gpu_common::convt2d_crop_input_dims(d.Ho, d.Wo, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, Hc, Wc);
    if (Hc == H && Wc == W) {
        conv2d_forward(dY, Wt, nullptr, N, C_out, d.Ho, d.Wo, C_in, kH, kW,
                       stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dX);
        return;
    }
    ::brotensor::Tensor full = ::brotensor::Tensor::empty_on(dY.device, N, C_in * Hc * Wc, dY.dtype);
    conv2d_forward(dY, Wt, nullptr, N, C_out, d.Ho, d.Wo, C_in, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, full);
    slice2d_forward(full, N, C_in, Hc, Wc, 0, 0, H, W, dX);
}

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
    ::brotensor::Tensor Xp = ::brotensor::Tensor::empty_on(X.device, N, C_in * Hc * Wc, X.dtype);
    pad2d_forward(X, N, C_in, H, W, 0, Hc - H, 0, Wc - W, 0, Xp);
    conv2d_backward_weight(dY, Xp, N, C_out, d.Ho, d.Wo, C_in, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, dWt);
}

void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    gpu_common::convt2d_require_fp32(op, dY, "dY");
    gpu_common::convt2d_require_fp32(op, dB, "dB");
    gpu_common::convt2d_need(op, dB.rows == C_out && dB.cols == 1, "dB shape must be (C_out, 1)");
    gpu_common::convt2d_need(op, dY.rows == N && dY.cols == C_out * H_out * W_out,
                             "dY shape must be (N, C_out*H_out*W_out)");
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;
    conv2d_backward_bias(dY, N, C_out, H_out, W_out, dB);
}

void fill_cuda_vtable_conv_transpose2d(::brotensor::detail::OpsVTable& v) {
    v.conv_transpose2d_forward         = &conv_transpose2d_forward;
    v.conv_transpose2d_backward_input  = &conv_transpose2d_backward_input;
    v.conv_transpose2d_backward_weight = &conv_transpose2d_backward_weight;
    v.conv_transpose2d_backward_bias   = &conv_transpose2d_backward_bias;
}

}  // namespace brotensor::detail::cuda
