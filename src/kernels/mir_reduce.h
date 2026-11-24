#pragma once

// Backend-neutral brass MIR kernels for row/column reductions (`sum_rows`,
// `argmax_rows`, `sum_cols`). Replaces src/vulkan/shaders/{reduce_rows,sum_cols}.comp
// and parallel CUDA/Metal kernels cleanly via a single source.

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kReduceBlock = 256;

// Builds `sum_rows`: params (ptr x, ptr y, i32 m, i32 n).
brass::Function* build_sum_rows_mir(brass::Module& mod, Dtype dt, int lanes);

// Builds `argmax_rows`: params (ptr x, ptr y, i32 m, i32 n).
brass::Function* build_argmax_rows_mir(brass::Module& mod, Dtype dt, bool out_i32, int lanes);

// Builds `sum_cols`: params (ptr x, ptr y, i32 m, i32 n).
brass::Function* build_sum_cols_mir(brass::Module& mod, Dtype dt, int lanes);

}  // namespace brotensor::kernels
