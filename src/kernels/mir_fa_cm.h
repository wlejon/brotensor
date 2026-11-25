#pragma once

// 16x16x16 cooperative-matrix FlashAttention forward MIR builder replacing:
//   fa_cm.comp (FP16 / BF16, FA_MODE_{ROWS, VARLEN, PACKED, LEN}, optional mask,
//   VEC=0/1, DIRECT fragment store & shared-scratch store, SPLIT-K partials).

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct FaCmSpec {
    Dtype dt = Dtype::FP16;
    std::uint32_t hd = 64;
    std::uint32_t bc = 16;   // 16, 32, or 64
    std::uint32_t nsg = 2;   // 1, 2, or 4 (WG = nsg * 32, BR = nsg * 16)
    std::uint32_t mode = 0;  // FA_MODE_*
    bool has_mask = false;
    bool vec = true;
    bool direct = false;
    bool split = false;
    bool has_softcap = false;
    bool head_swizzle = false; // Launch with ctaid_x = head, ctaid_y = qb (breaks PoT L1/L2 set aliasing)
};

// Params (23):
//   ptrs[0..7]: q, k, v, o, mask, aux, aux2, part
//   ints[8..20]: lq, lk, ldq, ldk, ldo, group, window, causal, nseq, chunk, hq, qb0, q_offset
//   flts[21..22]: scale (log2(e) / sqrt(HD)), softcap
brass::Function* build_fa_cm_mir(brass::Module& mod, const FaCmSpec& spec);

}  // namespace brotensor::kernels
