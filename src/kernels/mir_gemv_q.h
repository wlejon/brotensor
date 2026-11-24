#pragma once

// Quantized GEMV (1..8 activation rows) MIR builder replacing:
//   gemv_q.comp (QF_INT8 / QF_FP8 / QF_Q8_0 / QF_Q4K / QF_Q6K)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// Params: (ptr x, ptr w, ptr y, ptr bias, ptr scale, i32 n, i32 k, i32 ldx, i32 ldy, i32 rowbytes)
brass::Function* build_gemv_q_mir(brass::Module& mod, Dtype dt, std::uint32_t fmt,
                                  std::uint32_t mb, std::uint32_t lpr, bool vec, bool has_bias,
                                  std::uint32_t unr, std::uint32_t nr,
                                  std::uint32_t act, std::uint32_t epi);

}  // namespace brotensor::kernels
