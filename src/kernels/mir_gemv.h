#pragma once

// Skinny decode GEMV MIR builder replacing src/vulkan/shaders/gemv.comp:
//   Y(MB, *) from X(MB, K) and W(N, K) (NT layout, MB in 1..8)
//   Params: (ptr a, ptr b, ptr c, ptr bias,
//            i32 m, i32 n, i32 k, i32 lda, i32 ldb, i32 ldc, i32 half_n)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

brass::Function* build_gemv_mir(brass::Module& mod, Dtype dt_x, Dtype dt_w, Dtype dt_y,
                                std::uint32_t mb, std::uint32_t epi, std::uint32_t act,
                                bool vec, bool has_bias);

}  // namespace brotensor::kernels
