#pragma once

#include <brotensor/tensor.h>

namespace brotensor::detail::hip {

// SwiGLU & GEGLU
void swiglu_forward(const Tensor& X, Tensor& Y);
void swiglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX);
void geglu_forward(const Tensor& X, Tensor& Y);
void geglu_backward(const Tensor& X, const Tensor& dY, Tensor& dX);
void geglu_exact_forward(const Tensor& X, Tensor& Y);
void geglu_exact_backward(const Tensor& X, const Tensor& dY, Tensor& dX);

// Bias + Act
void bias_act_forward(const Tensor& X, const Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, Tensor& Y);
void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp, Tensor& dX, Tensor* dB);

// Reductions
void sum_rows(const Tensor& X, Tensor& Y);
void sum_cols(const Tensor& X, Tensor& Y);
void argmax_rows(const Tensor& X, Tensor& Idx);
void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts);
void masked_mean_pool_forward(const Tensor& X, const float* d_mask, Tensor& y);
void masked_mean_pool_backward(const Tensor& dY, const float* d_mask, int K, Tensor& dX);

} // namespace brotensor::detail::hip
