#pragma once

// FP32 flash_attention_varlen_forward MIR builder for head_dim=64, non-causal
// replacing flash_attention_varlen_f32.cu.

#include "../jit/gpu_mir_emit_detail.h"

namespace brotensor::kernels {

// Params: ptrs=(Q, K, V, cu_q, cu_k, O), ints=(total_q, B, D), flts=(q_scale_log2)
// Needs 67,584 bytes dynamic shared memory passed at launch.
brass::Function* build_fa_varlen_f32_mir(brass::Module& mod);

}  // namespace brotensor::kernels
