#pragma once

// Packed variable-length bidirectional attention MIR builder for head_dim=64
// (FP16 / BF16, mma.sync.m16n8k16 + ldmatrix + cp.async + warp-union skip)
// replacing flash_attention_packed.cu.

#include "../jit/gpu_mir_emit_detail.h"

namespace brotensor::kernels {

// Params: ptrs=(qkv, bounds, out), ints=(L, H, hw), flts=(scale_log2)
brass::Function* build_fa_packed_mma_mir(brass::Module& mod, Dtype dt);

}  // namespace brotensor::kernels
