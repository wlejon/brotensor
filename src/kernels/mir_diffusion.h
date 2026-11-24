#pragma once

// Single-source MIR builders for diffusion sampler steps, image
// preprocessing, timestep embedding, and Philox 4x32-10 noise.

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (x, e, p, y, z, n, dim, half, hw, c, k0, k1, k2, k3).
brass::Function* build_sampler_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, Dtype dti, int lanes = 1);

// Params: (y, key_lo, key_hi, ctr_lo, ctr_hi, n_lo, n_hi, p, lo, hi).
brass::Function* build_philox_mir(brass::Module& mod, std::uint32_t mode);

}  // namespace brotensor::kernels
