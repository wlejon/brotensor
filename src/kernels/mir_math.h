#pragma once

// Shared FP32 math & workgroup reduction MIR helpers matching
// common.glsl / math_acc.glsl accuracy across SPIR-V, PTX, and MSL targets.

#include "../jit/gpu_mir_emit_detail.h"

#include <utility>

namespace brotensor::kernels {

using jit::gpu_mir::Ctx;
using brass::Value;

Value* emit_tanh(Ctx& c, Value* x);
Value* emit_erf(Ctx& c, Value* x);
Value* emit_sigmoid(Ctx& c, Value* x);
Value* emit_gelu_tanh(Ctx& c, Value* x);
Value* emit_gelu_tanh_grad(Ctx& c, Value* x);
Value* emit_gelu_exact(Ctx& c, Value* x);
Value* emit_round_even(Ctx& c, Value* x);

Value* emit_exp(Ctx& c, Value* x);
Value* emit_log(Ctx& c, Value* x);
Value* emit_sin(Ctx& c, Value* x);
Value* emit_cos(Ctx& c, Value* x);
Value* emit_rsqrt(Ctx& c, Value* x);

// High-accuracy math (within ~1 ulp of glibc expf/sinf/cosf across large ranges).
Value* emit_div_rn(Ctx& c, Value* a, Value* b);
Value* emit_exp_acc(Ctx& c, Value* t);
std::pair<Value*, Value*> emit_sincos_acc(Ctx& c, Value* x);

// 256-thread block reductions (8x32 slices, safe across Wave32 & Wave64).
// `scratch8` is `shared_alloc_f32(8)` (and `s_idx8` is `shared_alloc_i32(8)`).
// Returns uniform block total/winner in all threads and ends with a uniform sync().
Value* emit_block_sum_f32(Ctx& c, Value* scratch8, Value* v);
Value* emit_block_max_f32(Ctx& c, Value* scratch8, Value* v);
Value* emit_block_argmax_f32(Ctx& c, Value* s_val8, Value* s_idx8, Value* best_v, Value* best_i);

}  // namespace brotensor::kernels
