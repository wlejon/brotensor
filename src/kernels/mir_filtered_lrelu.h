#pragma once

// Backend-neutral Brass MIR builder for filtered_lrelu (polyphase up-FIR,
// leaky ReLU + gain + clamp into shared memory, dense down-FIR).

#include "../jit/gpu_mir_emit_detail.h"

#include <cstdint>

namespace brotensor::kernels {

// mode == 0: build_polyu(ptr fu, ptr poly, i32 fuH, i32 fuW, i32 up, i32 JyMax, i32 JxMax)
// mode == 1: build_polyd(ptr fd, ptr poly, i32 fdH, i32 fdW)
// mode == 2: filtered_lrelu_kernel(ptr X, ptr poly_u, ptr poly_d, ptr b, ptr Y,
//                                  i32 C, i32 H, i32 W, i32 Hout, i32 Wout, i32 px0, i32 py0,
//                                  f32 up_gain, f32 gain, f32 slope, f32 clamp_v)
// `aux` packs for mode == 2:
//   bits [0..7]   up    (>= 1)
//   bits [8..15]  down  (>= 1)
//   bits [16..23] JyMax (>= 1)
//   bits [24..31] JxMax (>= 1)
//   bits [32..39] fdH   (>= 1)
//   bits [40..47] fdW   (>= 1)
brass::Function* build_filtered_lrelu_mir(brass::Module& mod, std::uint32_t mode, Dtype dt,
                                          bool has_bias, bool has_clamp, std::uint64_t aux);

}  // namespace brotensor::kernels
