#pragma once

// Single-source MIR builder for NCHW spatial adjoints: interp2d backward,
// pad2d backward, adaptive_avg_pool2d backward, and max_pool2d backward.

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (dy, dx, aux, planes, hi, wi, ho, wo, pt, pl).
brass::Function* build_resample_bwd_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, std::uint32_t imode);

}  // namespace brotensor::kernels
