#pragma once

// Shared FP32 math MIR helpers matching src/vulkan/shaders/common.glsl
// accuracy across SPIR-V, PTX, and MSL targets.

#include "../jit/gpu_mir_emit_detail.h"

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

}  // namespace brotensor::kernels
