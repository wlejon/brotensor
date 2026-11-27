#pragma once

#include "mir_common.h"

#include <cstdint>

namespace brotensor::kernels {

// Linear backward kernels:
//   op 0: dX[j] = sum_i W[i, j] * dY[i]         (block=128, shared dytile[128])
//   op 1: dW[i, j] += dY[i] * x[j]              (block=256)
//   op 2: dB[i] += dY[i]                        (block=256)
Function* build_linear_bwd_mir(Module& mod, std::uint32_t op, Dtype dt);

}  // namespace brotensor::kernels
