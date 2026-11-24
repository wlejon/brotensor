#pragma once

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

inline constexpr std::uint32_t kAudioBlock = 256;

// Builds a brass::Function for AU_SNAKE .. AU_COL2IM from op_codes.h.
// aux0: has_beta (AU_SNAKE*), pad mode 0..2 (AU_PAD*), resample mode 0..1
//       (AU_RESAMPLE*), has_bias (AU_CAUSAL, AU_COL2IM).
// aux1: cols_f32 (AU_COL2IM).
brass::Function* build_audio_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint32_t aux0, bool aux1);

}  // namespace brotensor::kernels
