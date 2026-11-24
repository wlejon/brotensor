#pragma once

// Single-source MIR builder for the Qwen3.5 gated delta rule scan.

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (q, k, v, a, b, loga, s, o, L, H, dk, dv, ldq, ldv).
brass::Function* build_delta_rule_mir(brass::Module& mod, std::uint32_t lpr, std::uint32_t ne);

}  // namespace brotensor::kernels
