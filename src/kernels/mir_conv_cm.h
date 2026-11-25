#pragma once

// 16x16x16 cooperative-matrix 2D implicit-GEMM convolution MIR builder replacing:
//   conv_cm.comp (FP16 / BF16, optional INT8 weights QA=QF_INT8, BVEC=0/1/2,
//   DIRECT fragment epilogue & shared-scratch epilogue, bias, accum, act).

#include "mir_gemm_cm.h"

#include <cstdint>

namespace brotensor::kernels {

struct ConvCmSpec {
    Dtype dt = Dtype::FP16;
    std::uint32_t cfg_idx = 1;  // index into kGemmCmTiles (0, 2, 3)
    std::uint32_t act = 0;      // LACT_*
    std::uint32_t bvec = 0;     // 0: 8x u16, 1: 1x1 aligned i32x4, 2: stride-1 shifted 2x i32x4
    std::uint32_t qa = 0;       // 0 or QF_INT8
    bool avec = true;
    bool direct = false;
    bool has_bias = false;
    bool rowch = false;
    bool accum = false;
    int kh = 3, kw = 3, sh = 1, sw = 1, ph = 1, pw = 1, dh = 1, dw = 1;
};

// Params (16):
//   ptrs[0..4]: x, w, y, bias, scale
//   ints[5..15]: m, n, k, h, wd, wo, cin, cout, groups, tm0, tn0
brass::Function* build_conv_cm_mir(brass::Module& mod, const ConvCmSpec& spec);

}  // namespace brotensor::kernels
