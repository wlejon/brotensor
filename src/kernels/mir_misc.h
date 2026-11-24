#pragma once

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

inline constexpr std::uint32_t kMiscBlock = 256;

// Builds a brass::Function for MI_MEAN_POOL .. MI_BIAS_ACT_DB from op_codes.h.
brass::Function* build_misc_mir(brass::Module& mod, std::uint32_t op, Dtype dt, bool has_aux);

}  // namespace brotensor::kernels
