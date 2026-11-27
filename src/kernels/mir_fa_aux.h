#pragma once

// Auxiliary flash-attention forward/backward MIR kernels replacing helper kernels in
// flash_attention.cu, flash_attention_fused.cu, and flash_attention_backward.cu.

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// op=0: fa_split_combine      (dt=FP16|BF16)
// op=1: fa_gather_head        (dt=FP16|BF16)
// op=2: fa_gather_head_T      (dt=FP16|BF16, flags&1=dup)
// op=3: fa_scatter_head       (dt=FP16|BF16)
// op=4: fa_softmax_rows_f32   (dt=FP16|BF16, flags&1=has_mask)
// op=5: bw_scatter            (dt=FP16|BF16, flags&1=has_sbits)
// op=6: bw_acc                (dt=FP32,      flags&1=has_sbits, flags&2=first)
// op=7: bw_row_stats          (dt=FP32,      flags&1=causal, flags&2=has_mask)
// op=8: bw_ds_tile            (dt=FP16|BF16, flags&1=causal, flags&2=has_mask)
brass::Function* build_fa_aux_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint8_t flags);

}  // namespace brotensor::kernels
