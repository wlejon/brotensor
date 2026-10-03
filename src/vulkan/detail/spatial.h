#pragma once

// Internal entry points of the spatial family, shared by the op files that
// compose them (ops_conv.cpp, ops_gnorm.cpp, ops_diffusion.cpp's ResBlock)
// and by the tests / benchmark. Design: docs/vulkan.md "Convolution".

#include "device.h"

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::vulkan {

// One 2D convolution, NCHW / OIHW, all operands in `dt`. y is written
// (accum = false) or accumulated into (y += conv, accum = true).
struct Conv2dArgs {
    std::uint64_t x = 0, w = 0, bias = 0, y = 0;   // device addresses; bias 0 = none
    ::brotensor::Dtype dt = ::brotensor::Dtype::FP32;
    int n = 1, cin = 0, h = 0, wd = 0, cout = 0, kh = 1, kw = 1;
    int sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1, groups = 1;
    bool accum = false;
    std::uint64_t scale = 0;   // nonzero: w is INT8 (C_out, K) with these per-channel FP32 scales (dt FP16)
    const char* op = "conv2d_forward";
};

void conv2d(DeviceCtx& d, const Conv2dArgs& a);

// Which kernel conv2d() uses for `a`: "coopmat", "simt" or "direct".
const char* conv2d_path(DeviceCtx& d, const Conv2dArgs& a);

// Test / benchmark hook, process-wide: 0 = automatic, 1 = never the
// cooperative-matrix kernel (SIMT implicit GEMM), 2 = always the direct
// kernel (also conv_transpose2d's direct gather instead of its GEMM form).
void set_conv_override(int mode);

// GroupNorm over (n, c, hw) in `dt`, optionally followed by SiLU; y may be x.
void group_norm(DeviceCtx& d, std::uint64_t x, std::uint64_t gamma, std::uint64_t beta, std::uint64_t y,
                ::brotensor::Dtype dt, int n, int c, int hw, int groups, float eps, bool silu);

}  // namespace brotensor::detail::vulkan
