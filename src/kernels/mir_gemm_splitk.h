#pragma once

#include "mir_common.h"

#include <cstdint>

namespace brotensor::kernels {

struct GemmSplitkSpec {
    Dtype dt = Dtype::FP16;
    std::uint8_t epi = 0;  // 0=store, 1=accum, 2=geglu, 3=swiglu
    std::uint8_t act = 0;  // 0..5 (used when epi <= 1)
    bool has_bias = false;
};

Function* build_gemm_splitk_reduce_mir(Module& mod, const GemmSplitkSpec& spec);

}  // namespace brotensor::kernels
