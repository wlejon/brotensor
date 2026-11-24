#pragma once

// Single-source MIR builder for bit-exact NCHW/row pure-gather remaps
// (pad, slice, 2x nearest upsample, unfold, merge2, pshuf, patch, win_part/rev,
// gather_rows, scatter_rows).

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (x, y, idx, c, cin, hi, wi, ho, wo, a0, a1, a2, a3, a4, a5, a6).
brass::Function* build_remap_mir(brass::Module& mod, std::uint32_t mode, int elem_bytes);

}  // namespace brotensor::kernels
