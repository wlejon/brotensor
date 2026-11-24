#pragma once

// Backend-neutral brass MIR kernel builders for elementwise unary, binary,
// activation backward, broadcast bias, dtype cast, and GLU operations.
// Replaces the per-backend GLSL/CUDA/Metal kernels cleanly via a single source.

#include <brotensor/tensor.h>

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

constexpr std::uint32_t kEwBlock = 256;

// Builds `unary`: params (ptr x, ptr y, i32 n, f32 a, f32 b).
// `op_code` matches UOP_* from vulkan/shaders/op_codes.h.
brass::Function* build_unary_mir(brass::Module& mod, std::uint32_t op_code, Dtype dt, int lanes);

// Builds `binary`: params (ptr y, ptr x, i32 n, f32 a, f32 b).
// `op_code` matches BOP_* from vulkan/shaders/op_codes.h.
brass::Function* build_binary_mir(brass::Module& mod, std::uint32_t op_code, Dtype dt_y, Dtype dt_x, int lanes);

// Builds `act_bwd`: params (ptr p, ptr dy, ptr dx, i32 n, f32 a).
// `op_code` matches GOP_* from vulkan/shaders/op_codes.h.
brass::Function* build_act_bwd_mir(brass::Module& mod, std::uint32_t op_code, Dtype dt, int lanes);

// Builds `bias`: params (ptr y, ptr bias, i32 n, i32 d, i32 l).
// `mode` matches BIAS_ROW / BIAS_CHANNEL.
brass::Function* build_bias_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, int lanes = 1);

// Builds `cast`: params (ptr src, ptr dst, i32 n).
brass::Function* build_cast_mir(brass::Module& mod, Dtype src_dt, Dtype dst_dt, int lanes);

// Builds `glu`: params (ptr x, ptr dy, ptr y, i32 n, i32 d).
// `op_code` matches GLU_* from vulkan/shaders/op_codes.h.
brass::Function* build_glu_mir(brass::Module& mod, std::uint32_t op_code, Dtype dt_in, Dtype dt_out, int lanes);

// Builds `rowvec`: params (ptr x, ptr v, ptr w, ptr y, i32 n, i32 d).
// `op_code` matches ROWVEC_MODULATE / ROWVEC_MUL from vulkan/shaders/op_codes.h.
brass::Function* build_rowvec_mir(brass::Module& mod, std::uint32_t op_code, Dtype dt, int lanes);

}  // namespace brotensor::kernels
