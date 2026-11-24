#pragma once

// Backend-neutral brass MIR kernels for GroupNorm, BatchNorm inference/train,
// NCHW per-pixel L2 normalize, and Conv2D backward bias gradient.

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kGnormBlock = 256;

// `gnorm`: params (16): (ptr x, ptr y, ptr g, ptr b, ptr part, ptr rm, ptr rv,
//                        i32 tile, i32 hw, i32 cpg, i32 groups,
//                        i32 splits, i32 chunk, i32 total, i32 c, f32 eps).
brass::Function* build_gnorm_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, bool silu, int lanes);

// `bnorm`: params (16): (ptr x, ptr y, ptr g, ptr b, ptr rm, ptr rv, ptr sm, ptr sr,
//                        ptr dy, ptr dg, ptr db, i32 n, i32 c, i32 hw, f32 eps, f32 momentum).
brass::Function* build_bnorm_mir(brass::Module& mod, std::uint32_t mode);

}  // namespace brotensor::kernels
