#pragma once

// Shared SIMT tiled GEMM & implicit-GEMM 2D convolution MIR builders replacing:
//   gemm_simt.comp  (tiled FP32-acc GEMM: 64x64/4x4 or 128x128/8x8, WG=256)
//   conv_simt.comp  (implicit-GEMM 2D conv: 64x64/4x4 or 128x128/8x8, WG=256)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// Params: (ptr a, ptr b, ptr c, ptr bias,
//          i32 m, i32 n, i32 k, i32 lda, i32 ldb, i32 ldc, i32 sa, i32 sb, i32 sc)
brass::Function* build_gemm_simt_mir(brass::Module& mod, Dtype dt_a, Dtype dt_b, Dtype dt_c, Dtype dt_bias,
                                     bool big128, std::uint32_t epi, std::uint32_t act,
                                     bool ta, bool nb, bool has_bias);

// Params: (ptr x, ptr w, ptr y, ptr bias,
//          i32 m, i32 n, i32 k, i32 h, i32 wd, i32 wo, i32 cin, i32 cout, i32 groups)
brass::Function* build_conv_simt_mir(brass::Module& mod, Dtype dt, bool big128, bool accum, bool has_bias,
                                     int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw);

}  // namespace brotensor::kernels
