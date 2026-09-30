#pragma once

#include <brotensor/tensor.h>

namespace brotensor::detail::hip {

// ─── 1D convolution family (conv1d.hip) ───
void conv_transpose1d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, ::brotensor::Tensor& Y);

void conv_transpose1d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding, int output_padding,
                                     int dilation, int groups, ::brotensor::Tensor& dX);

void conv_transpose1d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding, int output_padding,
                                      int dilation, int groups, ::brotensor::Tensor& dWt);

void conv_transpose1d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int L_out,
                                    ::brotensor::Tensor& dB);

void causal_conv1d_update(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& Wt,
                          const ::brotensor::Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          ::brotensor::Tensor& state, ::brotensor::Tensor& Y);

void pad1d_forward(const ::brotensor::Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode, ::brotensor::Tensor& Y);

void pad1d_backward(const ::brotensor::Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode, ::brotensor::Tensor& dX);

// ─── 2D transposed convolution (conv_transpose.hip) ───
void conv_transpose2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C_in, int H, int W, int C_out,
                              int kH, int kW, int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              ::brotensor::Tensor& Y);

void conv_transpose2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C_in, int H, int W, int C_out,
                                     int kH, int kW, int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     ::brotensor::Tensor& dX);

void conv_transpose2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      int N, int C_in, int H, int W, int C_out,
                                      int kH, int kW, int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      ::brotensor::Tensor& dWt);

void conv_transpose2d_backward_bias(const ::brotensor::Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    ::brotensor::Tensor& dB);

// ─── 3D convolution (conv3d.hip) ───
void conv3d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w, int groups,
                    ::brotensor::Tensor& Y);

void conv3d_int8w_fp16_forward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W_int8,
                               const ::brotensor::Tensor& scales,
                               const ::brotensor::Tensor* bias,
                               int N, int C_in, int T, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w, int groups,
                               ::brotensor::Tensor& Y);

// ─── Deformable convolution (deform_conv.hip) ───
void deform_conv2d_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& offset,
                           const ::brotensor::Tensor* mask,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int stride_h, int stride_w,
                           int pad_h, int pad_w, int dil_h, int dil_w,
                           int groups, int deform_groups,
                           ::brotensor::Tensor& Y);

// ─── Direct and Depthwise 2D Convolutions (conv_direct.hip) ───
void depthwise_conv2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Wt,
                              const ::brotensor::Tensor* bias,
                              int N, int C, int H, int W,
                              int kH, int kW, int H_out, int W_out,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int dil_h, int dil_w, long long total,
                              ::brotensor::Tensor& Y);

void depthwise_conv2d_backward_input(const ::brotensor::Tensor& Wt,
                                     const ::brotensor::Tensor& dY,
                                     int N, int C, int H, int W,
                                     int kH, int kW, int H_out, int W_out,
                                     int stride_h, int stride_w, int pad_h, int pad_w,
                                     int dil_h, int dil_w, long long total,
                                     ::brotensor::Tensor& dX);

void depthwise_conv2d_backward_weight(const ::brotensor::Tensor& X,
                                      const ::brotensor::Tensor& dY,
                                      float* dWt_scratch,
                                      int N, int C, int H, int W,
                                      int kH, int kW, int H_out, int W_out,
                                      int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dil_h, int dil_w, int total);

void conv2d_direct_forward(const ::brotensor::Tensor& X,
                           const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor* bias,
                           int N, int C_in, int H, int W, int C_out,
                           int kH, int kW, int H_out, int W_out,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, int Cg_in, int Cg_out,
                           long long total, ::brotensor::Tensor& Y);

void conv2d_direct_backward_input(const ::brotensor::Tensor& Wt,
                                  const ::brotensor::Tensor& dY,
                                  int N, int C_in, int H, int W, int C_out,
                                  int kH, int kW, int H_out, int W_out,
                                  int stride_h, int stride_w, int pad_h, int pad_w,
                                  int dil_h, int dil_w, int groups, int Cg_in, int Cg_out,
                                  long long total, ::brotensor::Tensor& dX);

void conv2d_direct_backward_weight(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& dY,
                                   float* dWt_scratch,
                                   int N, int C_in, int H, int W, int C_out,
                                   int kH, int kW, int H_out, int W_out,
                                   int stride_h, int stride_w, int pad_h, int pad_w,
                                   int dil_h, int dil_w, int groups, int Cg_in, int Cg_out,
                                   long long total);

} // namespace brotensor::detail::hip
