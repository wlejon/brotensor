#pragma once

// Shared Brass MIR helpers for 16x16x16 cooperative-matrix kernels
// (mir_gemm_cm, mir_conv_cm, mir_fa_cm, mir_fa_bwd_cm):
//   - i32x4 vector zero / select & FP16/BF16/FP32 tile packing
//   - 8-element quantized weight tile load + FP16 decode (INT8, FP8, Q8_0, Q4_K, Q6_K)
//   - elementwise cooperative-matrix accumulator loops & 16-lane clustered shuffles

#include "mir_common.h"

#include <array>
#include <cstdint>
#include <functional>

namespace brotensor::kernels {

struct QRawMir {
    Value* a = nullptr;      // i32x4: Q4_K header (or unused)
    Value* w0 = nullptr;     // i32: INT8/FP8/Q4_K low 4 bytes
    Value* w1 = nullptr;     // i32: INT8/FP8/Q4_K high 4 bytes
    std::array<Value*, 4> s0{};  // Q8_0 quants / Q6_K ql (4 x u16 as i32)
    std::array<Value*, 4> s1{};  // Q6_K qh (4 x u16 as i32)
    Value* c = nullptr;      // i32: INT8/FP8 scale bits (or Q8_0 d u16 / Q6_K sc_pair u16)
    Value* d = nullptr;      // i32: Q6_K d u16
};

Value* emit_v4_zero(Ctx& c);
Value* emit_v4_pack(Ctx& c, Value* w0, Value* w1, Value* w2, Value* w3);
Value* emit_v4_select(Ctx& c, Value* cond, Value* a, Value* b);
Value* emit_pack_f32x8_to_f16_v4(Ctx& c, const std::array<Value*, 8>& v);
Value* emit_bf16x8_to_f16x8(Ctx& c, Value* v4);
Value* emit_bf16x8_to_f16x8_scaled(Ctx& c, Value* v4, const std::array<Value*, 8>& mul);
Value* emit_f32x8_to_f16x8_scaled(Ctx& c, Value* v0, Value* v1, const std::array<Value*, 8>& mul);

// Load 8 consecutive quantized weights starting at column `k` of `row_ptr` (byte address).
QRawMir emit_qload8_mir(Ctx& c, std::uint32_t fmt, bool vec, Value* row_ptr,
                        Value* scale_ptr, Value* ridx, Value* k, Value* kmax);

// Decode QRawMir into 8 FP16 values packed in an i32x4.
// Optional `s_fp8` (256-entry f32 shared LUT) speeds up QF_FP8 decode when non-null.
Value* emit_qdecode8_f16_mir(Ctx& c, std::uint32_t fmt, const QRawMir& q, Value* k,
                             Value* s_fp8 = nullptr);

// Iterate over every local element of an accumulator fragment (`0..coop_acc_len(acc)`).
void emit_coop_acc_map(Ctx& c, Value* acc, const std::function<Value*(Value* idx, Value* val)>& fn);

// Subgroup-local 16-lane clustered reductions and 32-lane votes built purely on shfl_bfly.
Value* emit_cluster16_max_f32(Ctx& c, Value* v);
Value* emit_cluster16_add_f32(Ctx& c, Value* v);
Value* emit_cluster16_min_i32(Ctx& c, Value* v);
Value* emit_cluster16_max_i32(Ctx& c, Value* v);
Value* emit_subgroup32_any_i32(Ctx& c, Value* cond);
Value* emit_subgroup32_all_i32(Ctx& c, Value* cond);

}  // namespace brotensor::kernels
