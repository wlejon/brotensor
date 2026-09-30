#pragma once

#include <brotensor/tensor.h>

namespace brotensor::detail::hip {

// RMSNorm
void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y);
void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma);

// GroupNorm
void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups, float eps, Tensor& Y);
void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta);

// L2 Norm
void l2_norm_forward(const Tensor& X, int head_dim, int num_heads, float eps, Tensor& Y);
void l2_norm_backward(const Tensor& X, int head_dim, int num_heads, float eps,
                      const Tensor& dY, Tensor& dX);
void l2_normalize_nchw_forward(const Tensor& X, int N, int C, int H, int W, float eps, Tensor& Y);

// BatchNorm
void batch_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        Tensor& running_mean, Tensor& running_var,
                        int N, int C, int H, int W, float eps, float momentum,
                        Tensor& Y, Tensor& saved_mean, Tensor& saved_rstd);
void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                          const Tensor& running_mean, const Tensor& running_var,
                          int N, int C, int H, int W, float eps, Tensor& Y);
void batch_norm_backward(const Tensor& X, const Tensor& gamma,
                         const Tensor& saved_mean, const Tensor& saved_rstd,
                         const Tensor& dY, int N, int C, int H, int W,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta);

} // namespace brotensor::detail::hip
