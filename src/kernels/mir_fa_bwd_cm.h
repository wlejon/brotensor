#pragma once

// 16x16x16 cooperative-matrix FlashAttention backward MIR builder replacing:
//   fa_bwd_cm.comp (FP16 / BF16, PASS 0 statistics, PASS 1 dK/dV, PASS 2 dQ,
//   FA_MODE_{ROWS, VARLEN, PACKED}, optional key mask, VEC=0/1, AREG=0/1,
//   both 16-lane clustered-layout fast path and layout-agnostic path).

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct FaBwdCmSpec {
    Dtype dt = Dtype::FP16;
    std::uint32_t hd = 64;
    std::uint32_t nsg = 2;    // 2 or 4 (BO = 16 * nsg, WG = 32 * nsg)
    std::uint32_t bn = 16;    // 16 or 32
    std::uint32_t mode = 0;   // FA_MODE_*
    std::uint32_t pass = 0;   // 0: stats, 1: dK/dV, 2: dQ
    bool has_mask = false;
    bool vec = true;
    bool areg = true;
    bool agnostic = false;
};

// Params (21):
//   ptrs[0..9]: q, k, v, g, mask, aux, aux2, stats, o0, o1
//   ints[10..18]: lq, lk, ldq, ldk, ldg, ldo, window, causal, nseq
//   flts[19..20]: scale (log2(e)/sqrt(hd)), oscale (1/sqrt(hd))
brass::Function* build_fa_bwd_cm_mir(brass::Module& mod, const FaBwdCmSpec& spec);

}  // namespace brotensor::kernels
