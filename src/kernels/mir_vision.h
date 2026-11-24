#pragma once

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

inline constexpr std::uint32_t kVisionBlock = 256;

// Builds a brass::Function for MI_UPFIRDN, MI_MODW*, MI_COL2IM2D, MI_DEFORM_COL.
// aux0: flip (UPFIRDN), demodulate (MODW), has_bias (COL2IM2D), has_offset (DEFORM_COL)
// aux1: from_dcoef (MODW), cols_f32 (COL2IM2D), has_mask (DEFORM_COL)
brass::Function* build_vision_mir(brass::Module& mod, std::uint32_t op, Dtype dt, bool aux0, bool aux1);

}  // namespace brotensor::kernels
