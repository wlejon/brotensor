#pragma once

// Grouped-query decode attention MIR builder replacing decode_gqa.cu.

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// op=0: partials_kernel (128T)
//   flags: bits[0..1] = hd_code (0:64, 1:128, 2:256),
//          bits[2..3] = g_code  (0:1,  1:2,   2:4, 3:8),
//          bit[4]     = rc_code (0:1,  1:8/G).
//   Params: ptrs=(Q, K, V, d_len, ws), ints=(cap, Dkv, n_splits, nq, R, r0), flts=(inv_sqrt_log2)
// op=1: reduce_kernel (128T)
//   flags: 0 (unused).
//   Params: ptrs=(ws, O), ints=(hd, n_splits), flts=()
brass::Function* build_decode_gqa_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint8_t flags);

}  // namespace brotensor::kernels
