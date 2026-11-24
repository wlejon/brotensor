#pragma once

// Quantization / scaling helper MIR builders replacing:
//   gemm_rowscale.comp  (per-row power-of-two A exponents for CoopMat BF16/FP32)
//   wq_rows.comp        (quantize_rows / dequantize_rows INT8 & FP8 E4M3)
//   dequant.comp        (QF_INT8 / QF_FP8 / QF_Q8_0 / QF_Q4K / QF_Q6K -> FP16)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// Params: (ptr a, ptr out, i32 m, i32 k, i32 lda, i32 sa)
brass::Function* build_gemm_rowscale_mir(brass::Module& mod, Dtype dt, bool ta, bool vec);

// mode: 0 = quantize_rows, 1 = dequantize_rows; fp8: true = FP8 E4M3, false = INT8.
// Params: (ptr w, ptr q, ptr scale, i32 rows, i32 k)
brass::Function* build_wq_rows_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, bool fp8);

// fmt: QF_INT8(1), QF_FP8(2), QF_Q8_0(3), QF_Q4K(4), QF_Q6K(5); vec: K % 8 == 0 & aligned.
// Params: (ptr w, ptr y, ptr scale, i32 rows, i32 k, i32 rowbytes)
brass::Function* build_dequant_mir(brass::Module& mod, std::uint32_t fmt, bool vec);

}  // namespace brotensor::kernels
