#pragma once

// Register-only FlashAttention-2 MIR builder (mma.sync.m16n8k16 + ldmatrix +
// cp.async + ex2 online softmax, zero S/P shared memory) replacing
// flash_attention_fused.cu.

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

struct FaFusedMmaSpec {
    Dtype dt = Dtype::FP16;
    std::uint32_t hd = 64u;
    bool causal = false;
    bool has_mask = false;
    bool is_split = false;
};

inline std::uint32_t fa_fused_pad_hd(std::uint32_t hd) noexcept {
    return ((hd + 15u) / 16u) * 16u;
}

inline std::uint32_t fa_fused_bc(std::uint32_t hd) noexcept {
    return (fa_fused_pad_hd(hd) <= 64u) ? 64u : 32u;
}

// BR = 64 (128 threads, 4 warps) across all supported head dims keeps static
// shared memory <= 43 KB and eliminates S/P shared memory traffic.
inline constexpr std::uint32_t kFaFusedBr = 64u;
inline constexpr std::uint32_t kFaFusedThreads = 128u;

// Params:
//   ptrs[0..6]:  Q, K, V, mask, O, part_o, part_ml
//   ints[7..14]: Lq, Lk, D, Dkv, group, splits, qtiles, heads
//   flts[15]:    scale_log2
brass::Function* build_fa_fused_mma_mir(brass::Module& mod, const FaFusedMmaSpec& spec);

}  // namespace brotensor::kernels
