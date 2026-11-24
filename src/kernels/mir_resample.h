#pragma once

// Single-source MIR builder for NCHW resampling, adaptive pooling, max
// pooling, and convex upsampling forward kernels, plus shared rational coordinate
// helpers reused by mir_resample_bwd.cpp.

#include "mir_common.h"

#include <utility>

namespace brotensor::kernels {

struct FloorDiv {
    Value* q_i32;
    Value* rem_i64;
};

FloorDiv emit_floor_div(Ctx& c, Value* num_i64, Value* den_i64);
std::pair<Value*, Value*> emit_src_rational(Ctx& c, Value* o_i32, Value* in_dim, Value* out_dim, bool align_corners);

// Params: (x, y, aux, c, hi, wi, ho, wo, kh, kw, sh, sw, ph, pw).
brass::Function* build_resample_mir(brass::Module& mod, std::uint32_t mode, Dtype dt,
                                    bool align_corners, int imode);

}  // namespace brotensor::kernels
