#pragma once

#include <brotensor/tensor.h>

namespace brotensor::detail::hip {

void window_partition_forward(const ::brotensor::Tensor& X,
                              int N, int C, int H, int W, int window,
                              ::brotensor::Tensor& Y);

void window_reverse_forward(const ::brotensor::Tensor& X,
                            int N, int C, int H, int W, int window,
                            ::brotensor::Tensor& Y);

void convex_upsample_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             ::brotensor::Tensor& Y);

} // namespace brotensor::detail::hip
