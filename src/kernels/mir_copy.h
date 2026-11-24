#pragma once

// Backend-neutral brass MIR kernels for byte fill, strided 2D copy, and
// 32x32 shared-memory batched transpose.

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kFillBlock = 64;
constexpr std::uint32_t kCopyBlock = 256;
constexpr std::uint32_t kTransposeBlock = 256;

// Builds `fill_bytes`: params (ptr dst, i32 n, i32 value).
brass::Function* build_fill_bytes_mir(brass::Module& mod);

// Builds `copy2d`: params (ptr src, ptr dst, i32 src_pitch, i32 dst_pitch, i32 width, i32 height).
// `elem_bytes` in {1, 2, 4}; `lanes` = 1, or 4 (elem_bytes==4) / 8 (elem_bytes==2) when aligned.
brass::Function* build_copy2d_mir(brass::Module& mod, int elem_bytes, int lanes);

// Builds `transpose`: params (ptr src, ptr dst, i32 batch, i32 rows, i32 cols).
// `elem_bytes` in {2, 4}. Dispatches as 1D 256-thread blocks (32x8 virtual tile).
brass::Function* build_transpose_mir(brass::Module& mod, int elem_bytes);

}  // namespace brotensor::kernels
