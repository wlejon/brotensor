#pragma once

// Backend-neutral brass MIR kernel for deterministic bitonic-sort scatter-add
// (scatter_rows_add / embedding_lookup_backward).

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kScatterBlock = 256;

// Params (11): (ptr idx, ptr keys, ptr keys_out, ptr src, ptr dst,
//               i32 m, i32 r, i32 c, i32 p, i32 k, i32 j).
brass::Function* build_scatter_add_mir(brass::Module& mod, std::uint32_t mode, Dtype dt);

}  // namespace brotensor::kernels
