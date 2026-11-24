#pragma once

// Single-source MIR builder for FP32 LSTM training pointwise steps (forward &
// backward pre-activation gradients / cell carry).

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (gates, c, y, bhh, c0, dy, dz, dh, dc, b, h, row0).
brass::Function* build_lstm_mir(brass::Module& mod, std::uint32_t mode, bool has_bhh, bool has_c0);

}  // namespace brotensor::kernels
