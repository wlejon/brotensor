#pragma once

// Small / subgroup-reduction attention MIR builders replacing:
//   attn_aux.comp      (AUX_ROW_GATE, AUX_HEAD_MEAN)
//   attn_bwd.comp      (row softmax backward in place over dP)
//   fa_combine.comp    (split-key flash-attention partial merge)
//   attn_softmax.comp  (dense masked/biased/relpos attention row softmax)

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// mode: 0 = AUX_ROW_GATE, 1 = AUX_HEAD_MEAN.
// Params: (ptr x, ptr y, ptr mask, i32 rows, i32 n)
brass::Function* build_attn_aux_mir(brass::Module& mod, std::uint32_t mode, Dtype out_dt, bool has_mask);

// Params: (ptr p, ptr dp, ptr kmask, ptr qmask, i32 rows, i32 lq, i32 lk, f32 scale)
brass::Function* build_attn_bwd_mir(brass::Module& mod, bool has_kmask, bool has_qmask);

// Params: (ptr part, ptr o, i32 entries, i32 nsplit, i32 hd, i32 hq, i32 ldo)
brass::Function* build_fa_combine_mir(brass::Module& mod, Dtype out_dt);

// bias_mode: 0 = none, 1 = additive bias, 2 = SAM decomposed rel-pos (bias=Bh, mask=Bw).
// Params: (ptr s, ptr p, ptr bias, ptr mask, ptr qmask,
//          i32 rows, i32 lk, i32 lq, i32 h0, i32 lds, i32 ldp, i32 q0, i32 bhs, f32 scale)
brass::Function* build_attn_softmax_mir(brass::Module& mod, Dtype out_dt, std::uint32_t bias_mode,
                                        bool has_mask, bool has_qmask, bool mask_ge);

}  // namespace brotensor::kernels
