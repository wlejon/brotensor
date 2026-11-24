#pragma once

// FlashAttention row-parallel split-K forward MIR builder replacing:
//   fa_rows.comp (FP32 / FP16 / BF16, ROWS / VARLEN / PACKED / LEN, GQA, mask, softcap, maskpos)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

inline constexpr std::uint32_t kFaRowsBlock = 128;

// Params mirror FaPush (23 args = 124B):
//   ptrs[0..7]: q, k, v, o, mask, aux, aux2, part
//   ints[8..20]: lq, lk, ldq, ldk, ldo, group, window, causal, nseq, chunk, hq, qb0, q_offset
//   flts[21..22]: scale, softcap
brass::Function* build_fa_rows_mir(brass::Module& mod, Dtype dt, std::uint32_t hd, std::uint32_t gqa_g,
                                   std::uint32_t mode, bool has_mask, bool split,
                                   bool has_softcap, bool vec, bool has_maskpos);

}  // namespace brotensor::kernels
