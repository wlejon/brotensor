#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

inline constexpr std::uint32_t kSpectralBlock = 256;

// Builds a brass::Function for SP_BASIS .. SP_CX_POLAR from op_codes.h.
// aux0: BASIS_* mode (SP_BASIS) or center (SP_STFT_ADJ).
// aux1: weights (SP_BASIS); aux2: has_window (SP_BASIS).
brass::Function* build_spectral_mir(brass::Module& mod, std::uint32_t op, std::uint32_t aux0, bool aux1, bool aux2);

}  // namespace brotensor::kernels
