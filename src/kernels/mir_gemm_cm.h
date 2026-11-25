#pragma once

// 16x16x16 cooperative-matrix GEMM MIR builder replacing:
//   gemm_cm.comp (all 6 dtype variants: f16, bf16, f16_c32, bf16_c32, f32a_w16, f32a_wbf16)
//   supporting all 4 tile configs, TA/NB layouts, SCALE_A per-row power-of-two scaling,
//   QB quantized weights (INT8/FP8/Q8_0/Q4_K/Q6_K), DIRECT fragment epilogue, and
//   shared-scratch general epilogue (STORE/ACCUM/GEGLU/SWIGLU + bias + LACT_*).

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct GemmCmTile {
    std::uint32_t bm, bn, bk, wm, wn, wg;
};

constexpr GemmCmTile kGemmCmTiles[4] = {
    {256u, 128u, 32u, 64u, 64u, 256u},
    {128u, 128u, 32u, 64u, 64u, 128u},
    {128u,  64u, 32u, 64u, 32u, 128u},
    { 64u,  64u, 32u, 32u, 32u, 128u},
};

struct GemmCmSpec {
    Dtype dt_a = Dtype::FP16;
    Dtype dt_b = Dtype::FP16;
    Dtype dt_c = Dtype::FP16;
    std::uint32_t cfg_idx = 1;  // index into kGemmCmTiles (0..3)
    std::uint32_t epi = 0;      // EPI_*
    std::uint32_t act = 0;      // LACT_*
    std::uint32_t qb = 0;       // 0 or QF_*
    bool ta = false;
    bool nb = false;
    bool vec = true;
    bool direct = false;
    bool has_bias = false;
    bool scale_a = false;
    bool swizzle = false;       // 2D L2 slab swizzle (group_m = 4, disables krot; requires tm0==tn0==0)
};

// Params (18, or 19 with residual ptr at [6] when epi==EPI_RESIDUAL_ADD):
//   ptrs[0..5]: a, b, c, bias, scale, ascale [, residual]
//   ints[...]: m, n, k, lda, ldb, ldc, sa, sb, sc, half_n, tm0, tn0
brass::Function* build_gemm_cm_mir(brass::Module& mod, const GemmCmSpec& spec);

}  // namespace brotensor::kernels
