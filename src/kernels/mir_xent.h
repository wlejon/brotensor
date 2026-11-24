#pragma once

// Single-source MIR builder for FP32 segmented softmax cross-entropy and
// BCE-with-logits fused loss/gradient kernels.

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (logits, target, mask, off, probs, dlog, loss, rows, n, heads, pos_weight).
brass::Function* build_xent_mir(brass::Module& mod, std::uint32_t mode, bool has_mask, bool has_off);

}  // namespace brotensor::kernels
