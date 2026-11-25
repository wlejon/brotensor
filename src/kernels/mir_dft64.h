#pragma once

// FP64 direct DFT MIR builder replacing:
//   dft64.comp (BASIS_R2C / BASIS_C2R / BASIS_C2C, L <= 2048, double-precision twiddles & sums)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// Params:
//   ptrs[0..2]: x, y, win
//   ints[3..13]: sa, sc, R, batch, cols, lda, rows, ldc, L, tofs, sgn
//   flts[14]: scale
brass::Function* build_dft64_mir(brass::Module& mod, std::uint32_t mode,
                                 bool has_win, bool has_weights);

}  // namespace brotensor::kernels
