#pragma once

// Single-source MIR builder for per-row exact top-k selection and
// segment softmax statistics across FP32, FP16, and BF16 inputs.

#include "mir_common.h"

namespace brotensor::kernels {

// Params: (x, vals, idx, off, rows, cols, k, nseg).
brass::Function* build_topk_mir(brass::Module& mod, std::uint32_t mode, Dtype dt);

}  // namespace brotensor::kernels
