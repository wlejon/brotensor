// Shared Q8_0 layout constants. Q8_0 has no nested structure to decode —
// each block is { fp16 d; int8 qs[32]; } = 34 bytes total. Both q8_0.cu
// (dequant + GEMV) and q8_0_wmma.cu (fused WMMA GEMM) include this header
// just to keep the constants single-sourced.
//
// Decoded value: y[i] = __half2float(d) * qs[i].

#pragma once
#include <cstdint>

namespace brotensor::detail::cuda::q8_0 {

constexpr int kBlockBytes = 34;
constexpr int kBlockElems = 32;
constexpr int kDOffset    = 0;
constexpr int kQsOffset   = 2;

}  // namespace brotensor::detail::cuda::q8_0
