#pragma once

// Backend-neutral brass MIR kernel for RoPE (THETA, TABLE, PERHEAD, PACKED,
// MROPE), matching src/vulkan/shaders/rope.comp + math_acc.glsl accuracy.

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kRopeBlock = 256;

// Params (19): (ptr x, ptr y, ptr ct, ptr st, ptr ch, ptr sh, ptr cw, ptr sw,
//               ptr pt, ptr ph, ptr pw, i32 rows, i32 heads, i32 hd,
//               i32 dt, i32 dh, i32 dw, i32 offset, f32 ln_base).
brass::Function* build_rope_mir(brass::Module& mod, std::uint32_t mode, bool bwd, Dtype dt);

}  // namespace brotensor::kernels
