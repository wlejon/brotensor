#pragma once

// FlashAttention FMA backward MIR builder replacing:
//   fa_bwd.comp (FP32 / FP16 / BF16, PASS 0 = stats + dQ, PASS 1 = dK + dV)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// Params mirror BwdPush (21 args = 124B <= 128B):
//   ptrs[0..9]: q, k, v, g, mask, aux, aux2, stats, o0, o1
//   ints[10..18]: lq, lk, ldq, ldk, ldg, ldo, window, causal, nseq
//   flts[19..20]: scale (log2(e)/sqrt(hd)), oscale (1/sqrt(hd))
brass::Function* build_fa_bwd_mir(brass::Module& mod, Dtype dt,
                                  std::uint32_t nt, std::uint32_t hd, std::uint32_t hdp,
                                  std::uint32_t br, std::uint32_t bc,
                                  std::uint32_t mode, bool has_mask, std::uint32_t pass);

}  // namespace brotensor::kernels
