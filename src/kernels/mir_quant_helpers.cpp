#include "mir_quant_helpers.h"

#include "mir_common.h"
#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <stdexcept>
#include <string>

namespace brotensor::kernels {

namespace {

using jit::gpu_mir::bits_to_f32;
using jit::gpu_mir::f32_to_bits;

Value* umax_i32(Ctx& c, Value* a, Value* b) {
    return c.b.build_select(c.b.build_ugt(a, b), a, b);
}

Value* finite_or_0(Ctx& c, Value* u) {
    return c.b.build_select(c.b.build_ult(u, c.i32(0x7f800000)), u, c.i32(0));
}

Value* abs_bits_elem(Ctx& c, Dtype dt, Value* base, Value* idx) {
    Value* u = nullptr;
    if (dt == Dtype::BF16) {
        Value* raw = c.kb.load_u16(ptr_at(c, base, idx, 2));
        u = c.b.build_and(c.b.build_shl(raw, c.i32(16)), c.i32(0x7fffffff));
    } else if (dt == Dtype::FP32) {
        u = c.b.build_and(c.kb.load_i32(ptr_at(c, base, idx, 4)), c.i32(0x7fffffff));
    } else {
        Value* fv = c.kb.fabs(load_elem_dt(c, dt, base, idx));
        u = f32_to_bits(c, fv);
    }
    return finite_or_0(c, u);
}

Value* exponent_for(Ctx& c, Value* u) {
    Value* raw = c.b.build_sub(c.b.build_lshr(u, c.i32(23)), c.i32(141));
    Value* clamped = clamp_i32(c, raw, c.i32(-100), c.i32(100));
    return c.b.build_select(c.b.build_eq(u, c.i32(0)), c.i32(0), clamped);
}

Value* s8_byte_to_f32(Ctx& c, Value* b) {
    return c.kb.i32_to_f32(c.b.build_ashr(c.b.build_shl(b, c.i32(24)), c.i32(24)));
}

// Decode OCP FP8 E4M3 (low 7-bit magnitude + bit 7 sign) to FP32.
Value* fp8_decode_u32(Ctx& c, Value* b) {
    Value* v = c.b.build_and(b, c.i32(0x7f));
    Value* norm = bits_to_f32(c, c.b.build_add(c.b.build_shl(v, c.i32(20)), c.i32(0x3c000000)));
    Value* sub = c.b.build_mul(c.kb.u32_to_f32(v), c.f32(0.001953125f));
    Value* mag = c.b.build_select(c.b.build_ult(v, c.i32(8)), sub, norm);
    Value* is_nan = c.b.build_eq(v, c.i32(0x7f));
    mag = c.b.build_select(is_nan, bits_to_f32(c, c.i32(0x7fc00000)), mag);
    Value* sgn = c.b.build_shl(c.b.build_and(b, c.i32(0x80)), c.i32(24));
    return bits_to_f32(c, c.b.build_or(f32_to_bits(c, mag), sgn));
}

Value* fp8_encode_f32(Ctx& c, Value* v) {
    Value* bits = f32_to_bits(c, v);
    Value* mag_bits = c.b.build_and(bits, c.i32(0x7fffffff));
    Value* is_nan = c.b.build_ugt(mag_bits, c.i32(0x7f800000));
    Value* sign = c.b.build_shl(c.b.build_lshr(bits, c.i32(31)), c.i32(7));
    Value* a = c.kb.fabs(v);

    Value* lo = c.i32(0);
    Value* hi = c.i32(126);
    for (int step = 0; step < 7; ++step) {
        Value* mid = c.b.build_lshr(c.b.build_add(lo, hi), c.i32(1));
        Value* le = c.b.build_sle(fp8_decode_u32(c, mid), a);
        lo = c.b.build_select(le, mid, lo);
        hi = c.b.build_select(le, hi, mid);
    }
    Value* dl = c.b.build_sub(a, fp8_decode_u32(c, lo));
    Value* dh = c.b.build_sub(fp8_decode_u32(c, hi), a);
    Value* even_pick = c.b.build_select(c.b.build_ne(c.b.build_and(lo, c.i32(1)), c.i32(0)), hi, lo);
    Value* code = c.b.build_select(c.b.build_slt(dl, dh), lo,
                                   c.b.build_select(c.b.build_slt(dh, dl), hi, even_pick));
    Value* sat = c.b.build_select(c.b.build_sge(a, c.f32(448.0f)), c.i32(0x7e), code);
    return c.b.build_select(is_nan, c.i32(0x7f), c.b.build_or(sign, sat));
}

// Correctly rounded a / b using pinned NR + FMA exact remainder on candidate +-1 ULP.
Value* div_cr_f32(Ctx& c, Value* a, Value* b, Value* zero_mask) {
    auto pin_f32 = [&](Value* v) {
        return bits_to_f32(c, c.b.build_xor(f32_to_bits(c, v), zero_mask));
    };
    Value* r = pin_f32(c.b.build_sdiv(c.f32(1.0f), b));
    Value* q0 = pin_f32(c.b.build_mul(a, r));
    Value* q1 = pin_f32(c.b.build_fma_f32(c.b.build_fma_f32(c.b.build_neg(q0), b, a), r, q0));
    Value* q = pin_f32(c.b.build_fma_f32(c.b.build_fma_f32(c.b.build_neg(q1), b, a), r, q1));
    Value* qb = f32_to_bits(c, q);
    Value* dn = pin_f32(bits_to_f32(c, c.b.build_sub(qb, c.i32(1))));
    Value* up = pin_f32(bits_to_f32(c, c.b.build_add(qb, c.i32(1))));
    auto rem_abs = [&](Value* cand) -> Value* {
        return c.kb.fabs(c.b.build_fma_f32(c.b.build_neg(cand), b, a));
    };
    Value* e = rem_abs(q);
    Value* ed = rem_abs(dn);
    Value* eu = rem_abs(up);
    Value* best = c.b.build_select(c.b.build_slt(ed, e), dn, q);
    Value* me = c.kb.fmin(e, ed);
    best = c.b.build_select(c.b.build_slt(eu, me), up, best);
    Value* zero_in = c.b.build_or(
        c.b.build_eq(a, c.f32(0.0f)),
        c.b.build_eq(c.b.build_and(qb, c.i32(0x7fffffff)), c.i32(0)));
    return c.b.build_select(zero_in, a, best);
}

Value* sel_word3(Ctx& c, Value* widx, Value* w0, Value* w1, Value* w2) {
    return c.b.build_select(c.b.build_eq(widx, c.i32(0)), w0,
                            c.b.build_select(c.b.build_eq(widx, c.i32(1)), w1, w2));
}

Value* extract_byte(Ctx& c, Value* word, Value* bidx) {
    return c.b.build_and(c.b.build_lshr(word, c.b.build_shl(c.b.build_and(bidx, c.i32(3)), c.i32(3))), c.i32(0xff));
}

}  // namespace

brass::Function* build_gemm_rowscale_mir(brass::Module& mod, Dtype dt, bool ta, bool vec) {
    const std::string name = std::string("mir_gemm_rowscale_") +
                             (dt == Dtype::FP32 ? "f32" : dt == Dtype::FP16 ? "f16" : "bf16") +
                             (ta ? "_ta" : "") + (vec ? "_v" : "");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0]=a, p[1]=out, p[2]=m, p[3]=k, p[4]=lda, p[5]=sa
    const int esz = elem_bytes(dt);
    Value* z = kb.ctaid_z();
    Value* A = ptr_at(c, p[0], c.b.build_mul(z, p[5]), esz);
    Value* t = kb.tid_x();
    Value* g = c.b.build_add(kb.ctaid_x(), c.b.build_mul(kb.ctaid_y(), kb.nctaid_x()));

    if (ta) {
        Value* m = c.b.build_add(c.b.build_mul(g, c.i32(256)), t);
        kb.if_then(c.b.build_ult(m, p[2]), [&] {
            Value* mx = kb.for_range_reduce(
                c.i32(0), p[3], c.i32(1), c.i32(0),
                [&](Value* k, Value* cur) -> Value* {
                    return umax_i32(c, cur, abs_bits_elem(c, dt, A, c.b.build_add(c.b.build_mul(k, p[4]), m)));
                });
            kb.store_i32(ptr_at(c, p[1], c.b.build_add(c.b.build_mul(z, p[2]), m), 4), exponent_for(c, mx));
        });
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    Value* s_red = kb.shared_alloc_i32(256);
    Value* row = c.b.build_add(c.b.build_mul(g, c.i32(4)), c.b.build_lshr(t, c.i32(6)));
    Value* lane = c.b.build_and(t, c.i32(63));
    Value* mx = emit_if_else_values(
        c, c.b.build_ult(row, p[2]), {Type::i32()},
        [&]() -> std::vector<Value*> {
            if (vec && dt != Dtype::FP16) {
                Value* per = c.i32(dt == Dtype::BF16 ? 8 : 4);
                Value* row_base = c.b.build_udiv(c.b.build_mul(row, p[4]), per);
                Value* nchunks = c.b.build_udiv(p[3], per);
                return {kb.for_range_reduce(
                    lane, nchunks, c.i32(64), c.i32(0),
                    [&](Value* ch, Value* cur) -> Value* {
                        Value* v4 = kb.vload(Type::i32x4(), ptr_at(c, A, c.b.build_add(row_base, ch), 16));
                        Value* res = cur;
                        for (std::uint32_t j = 0; j < 4; ++j) {
                            Value* w = c.b.build_vextract_lane(v4, j);
                            if (dt == Dtype::BF16) {
                                Value* lo = finite_or_0(c, c.b.build_and(c.b.build_shl(w, c.i32(16)), c.i32(0x7fffffff)));
                                Value* hi = finite_or_0(c, c.b.build_and(w, c.i32(0x7fff0000)));
                                res = umax_i32(c, res, umax_i32(c, lo, hi));
                            } else {
                                res = umax_i32(c, res, finite_or_0(c, c.b.build_and(w, c.i32(0x7fffffff))));
                            }
                        }
                        return res;
                    })};
            }
            Value* rbase = c.b.build_mul(row, p[4]);
            return {kb.for_range_reduce(
                lane, p[3], c.i32(64), c.i32(0),
                [&](Value* k, Value* cur) -> Value* {
                    return umax_i32(c, cur, abs_bits_elem(c, dt, A, c.b.build_add(rbase, k)));
                })};
        },
        [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];

    kb.shared_store_i32_indexed(s_red, t, mx);
    kb.sync();
    for (int s = 32; s > 0; s >>= 1) {
        kb.if_then(c.b.build_ult(lane, c.i32(s)), [&] {
            Value* v = umax_i32(c, kb.shared_load_i32_indexed(s_red, t),
                                kb.shared_load_i32_indexed(s_red, c.b.build_add(t, c.i32(s))));
            kb.shared_store_i32_indexed(s_red, t, v);
        });
        kb.sync();
    }
    kb.if_then(c.b.build_and(c.b.build_eq(lane, c.i32(0)), c.b.build_ult(row, p[2])), [&] {
        Value* best = kb.shared_load_i32_indexed(s_red, t);
        kb.store_i32(ptr_at(c, p[1], c.b.build_add(c.b.build_mul(z, p[2]), row), 4), exponent_for(c, best));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_wq_rows_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, bool fp8) {
    const std::string name = std::string("mir_wq_rows_") + (mode == 0u ? "q_" : "dq_") +
                             (dt == Dtype::FP32 ? "f32" : dt == Dtype::FP16 ? "f16" : "bf16") +
                             (fp8 ? "_fp8" : "_i8");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0]=w, p[1]=q, p[2]=scale, p[3]=rows, p[4]=k

    if (mode == 1u) {
        Value* total = c.b.build_mul(p[3], p[4]);
        kb.for_range(kb.global_tid_x(), total, grid_stride_ew(c), [&](Value* i) {
            Value* b = kb.load_u8(ptr_at(c, p[1], i, 1));
            Value* code = fp8 ? fp8_decode_u32(c, b) : s8_byte_to_f32(c, b);
            Value* sc = kb.load_f32(ptr_at(c, p[2], c.b.build_udiv(i, p[4]), 4));
            store_elem_dt(c, dt, p[0], i, c.b.build_mul(code, sc));
        });
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    Value* row = c.b.build_add(kb.ctaid_x(), c.b.build_mul(kb.ctaid_y(), kb.nctaid_x()));
    Value* tid = kb.tid_x();
    Value* s8 = kb.shared_alloc_f32(8);
    Value* wg = c.i32(static_cast<int>(jit::gpu_mir::kEwBlock));

    Value* mx_local = emit_if_else_values(
        c, c.b.build_ult(row, p[3]), {Type::f32()},
        [&]() -> std::vector<Value*> {
            Value* base = c.b.build_mul(row, p[4]);
            return {kb.for_range_reduce(
                tid, p[4], wg, c.f32(0.0f),
                [&](Value* col, Value* cur) -> Value* {
                    return c.kb.fmax(cur, c.kb.fabs(load_elem_dt(c, dt, p[0], c.b.build_add(base, col))));
                })};
        },
        [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];

    Value* mx = emit_block_max_f32(c, s8, mx_local);
    Value* zero_mask = c.b.build_select(c.b.build_eq(p[4], c.i32(0)), c.i32(1), c.i32(0));
    kb.if_then(c.b.build_ult(row, p[3]), [&] {
        Value* base = c.b.build_mul(row, p[4]);
        Value* scale = div_cr_f32(c, mx, c.f32(fp8 ? 448.0f : 127.0f), zero_mask);
        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            kb.store_f32(ptr_at(c, p[2], row, 4), scale);
        });
        Value* pos_sc = c.b.build_sgt(scale, c.f32(0.0f));
        kb.for_range(tid, p[4], wg, [&](Value* col) {
            Value* idx = c.b.build_add(base, col);
            Value* wv = load_elem_dt(c, dt, p[0], idx);
            Value* qv = emit_if_else_values(
                c, pos_sc, {Type::f32()},
                [&]() -> std::vector<Value*> { return {div_cr_f32(c, wv, scale, zero_mask)}; },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            Value* byte = fp8 ? fp8_encode_f32(c, qv)
                              : c.b.build_and(kb.f32_to_i32(c.kb.clamp_f32(emit_round_even(c, qv),
                                                                           c.f32(-127.0f), c.f32(127.0f))),
                                              c.i32(0xff));
            kb.store_u8(ptr_at(c, p[1], idx, 1), byte);
        });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_dequant_mir(brass::Module& mod, std::uint32_t fmt, bool vec) {
    const std::string name = std::string("mir_dequant_f") + std::to_string(fmt) + (vec ? "_v" : "");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0]=w, p[1]=y, p[2]=scale, p[3]=rows, p[4]=k, p[5]=rowbytes

    if (!vec) {
        Value* total = c.b.build_mul(p[3], p[4]);
        kb.for_range(kb.global_tid_x(), total, grid_stride_ew(c), [&](Value* i) {
            Value* r = c.b.build_udiv(i, p[4]);
            Value* col = c.b.build_umod(i, p[4]);
            Value* s = kb.load_f32(ptr_at(c, p[2], r, 4));
            Value* b = kb.load_u8(c.b.build_add(p[0], c.b.build_add(c.b.build_mul(c.b.build_zext_i64(r),
                                                                                   c.b.build_zext_i64(p[5])),
                                                                    c.b.build_zext_i64(col))));
            Value* dv = (fmt == QF_FP8) ? fp8_decode_u32(c, b) : s8_byte_to_f32(c, b);
            store_elem_dt(c, Dtype::FP16, p[1], i, c.b.build_mul(dv, s));
        });
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    Value* per = c.b.build_lshr(p[4], c.i32(3));
    Value* total = c.b.build_mul(p[3], per);
    kb.for_range(kb.global_tid_x(), total, grid_stride_ew(c), [&](Value* i) {
        Value* r = c.b.build_udiv(i, per);
        Value* k = c.b.build_shl(c.b.build_umod(i, per), c.i32(3));
        Value* row_ptr = c.b.build_add(p[0], c.b.build_mul(c.b.build_zext_i64(r), c.b.build_zext_i64(p[5])));
        std::vector<Value*> w(8, c.f32(0.0f));

        if (fmt == QF_INT8 || fmt == QF_FP8) {
            Value* q64 = kb.load_i64(c.b.build_add(row_ptr, c.b.build_zext_i64(k)));
            Value* qlo = c.b.build_trunc_i32(q64);
            Value* qhi = c.b.build_trunc_i32(c.b.build_lshr(q64, kb.const_i64(32)));
            Value* s = kb.load_f32(ptr_at(c, p[2], r, 4));
            for (int e = 0; e < 8; ++e) {
                Value* b = c.b.build_and(c.b.build_lshr(e < 4 ? qlo : qhi, c.i32((e & 3) * 8)), c.i32(0xff));
                Value* dv = (fmt == QF_FP8) ? fp8_decode_u32(c, b) : s8_byte_to_f32(c, b);
                w[static_cast<std::size_t>(e)] = c.b.build_mul(dv, s);
            }
        } else if (fmt == QF_Q8_0) {
            Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(k, c.i32(5))),
                                                              kb.const_i64(34)));
            Value* d = kb.f16_to_f32(kb.load_u16(blk));
            Value* q64 = kb.load_i64(c.b.build_add(blk, c.b.build_zext_i64(c.b.build_add(c.i32(2), c.b.build_and(k, c.i32(31))))));
            Value* qlo = c.b.build_trunc_i32(q64);
            Value* qhi = c.b.build_trunc_i32(c.b.build_lshr(q64, kb.const_i64(32)));
            for (int e = 0; e < 8; ++e) {
                Value* b = c.b.build_and(c.b.build_lshr(e < 4 ? qlo : qhi, c.i32((e & 3) * 8)), c.i32(0xff));
                w[static_cast<std::size_t>(e)] = c.b.build_mul(s8_byte_to_f32(c, b), d);
            }
        } else if (fmt == QF_Q4K) {
            Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(k, c.i32(8))),
                                                              kb.const_i64(144)));
            Value* hdr = kb.vload(Type::i32x4(), blk);
            Value* a0 = c.b.build_vextract_lane(hdr, 0);
            Value* s0 = c.b.build_vextract_lane(hdr, 1);
            Value* s1 = c.b.build_vextract_lane(hdr, 2);
            Value* s2 = c.b.build_vextract_lane(hdr, 3);
            Value* j = c.b.build_lshr(c.b.build_and(k, c.i32(255)), c.i32(5));
            Value* b_off = c.b.build_add(c.i32(16), c.b.build_add(c.b.build_shl(c.b.build_lshr(j, c.i32(1)), c.i32(5)),
                                                                  c.b.build_and(k, c.i32(31))));
            Value* nib64 = kb.load_i64(c.b.build_add(blk, c.b.build_zext_i64(b_off)));
            Value* blo = c.b.build_trunc_i32(nib64);
            Value* bhi = c.b.build_trunc_i32(c.b.build_lshr(nib64, kb.const_i64(32)));

            Value* low_j = c.b.build_ult(j, c.i32(4));
            Value* j4 = c.b.build_add(j, c.i32(4));
            Value* jm4 = c.b.build_sub(j, c.i32(4));
            Value* bs_j = extract_byte(c, sel_word3(c, c.b.build_lshr(j, c.i32(2)), s0, s1, s2), j);
            Value* bs_j4 = extract_byte(c, sel_word3(c, c.b.build_lshr(j4, c.i32(2)), s0, s1, s2), j4);
            Value* bs_jm4 = extract_byte(c, sel_word3(c, c.b.build_lshr(jm4, c.i32(2)), s0, s1, s2), jm4);
            Value* sc = c.b.build_select(
                low_j, c.b.build_and(bs_j, c.i32(63)),
                c.b.build_or(c.b.build_and(bs_j4, c.i32(15)), c.b.build_shl(c.b.build_lshr(bs_jm4, c.i32(6)), c.i32(4))));
            Value* mn = c.b.build_select(
                low_j, c.b.build_and(bs_j4, c.i32(63)),
                c.b.build_or(c.b.build_lshr(bs_j4, c.i32(4)), c.b.build_shl(c.b.build_lshr(bs_j, c.i32(6)), c.i32(4))));

            Value* d = kb.f16_to_f32(c.b.build_and(a0, c.i32(0xffff)));
            Value* dmin = kb.f16_to_f32(c.b.build_lshr(a0, c.i32(16)));
            Value* ws = c.b.build_mul(kb.u32_to_f32(sc), d);
            Value* wm = c.b.build_mul(kb.u32_to_f32(mn), dmin);
            Value* sh = c.b.build_shl(c.b.build_and(j, c.i32(1)), c.i32(2));
            for (int e = 0; e < 8; ++e) {
                Value* bw = e < 4 ? blo : bhi;
                Value* q4 = c.b.build_and(c.b.build_lshr(bw, c.b.build_add(sh, c.i32((e & 3) * 8))), c.i32(0x0f));
                w[static_cast<std::size_t>(e)] = c.b.build_fma_f32(ws, kb.u32_to_f32(q4), c.b.build_neg(wm));
            }
        } else if (fmt == QF_Q6K) {
            Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(k, c.i32(8))),
                                                              kb.const_i64(210)));
            Value* e0 = c.b.build_and(k, c.i32(255));
            Value* g = c.b.build_lshr(e0, c.i32(7));
            Value* quad = c.b.build_lshr(c.b.build_and(e0, c.i32(127)), c.i32(5));
            Value* l0 = c.b.build_and(e0, c.i32(31));
            Value* ql_off = c.b.build_add(c.b.build_add(c.b.build_shl(g, c.i32(6)),
                                                        c.b.build_shl(c.b.build_and(quad, c.i32(1)), c.i32(5))),
                                          l0);
            Value* qh_off = c.b.build_add(c.i32(128), c.b.build_add(c.b.build_shl(g, c.i32(5)), l0));
            Value* sc_off = c.b.build_add(c.i32(192), c.b.build_add(c.b.build_shl(g, c.i32(3)),
                                                                    c.b.build_shl(quad, c.i32(1))));
            Value* ql64 = kb.load_i64(c.b.build_add(blk, c.b.build_zext_i64(ql_off)));
            Value* qh64 = kb.load_i64(c.b.build_add(blk, c.b.build_zext_i64(qh_off)));
            Value* bx = kb.load_u16(c.b.build_add(blk, c.b.build_zext_i64(sc_off)));
            Value* d = kb.f16_to_f32(kb.load_u16(c.b.build_add(blk, kb.const_i64(208))));
            Value* sbyte = c.b.build_and(c.b.build_lshr(bx, c.b.build_shl(c.b.build_lshr(l0, c.i32(4)), c.i32(3))),
                                         c.i32(0xff));
            Value* s = c.b.build_mul(d, s8_byte_to_f32(c, sbyte));
            Value* ql_sh = c.b.build_select(c.b.build_uge(quad, c.i32(2)), c.i32(4), c.i32(0));
            Value* qh_sh = c.b.build_shl(quad, c.i32(1));
            Value* qlo = c.b.build_trunc_i32(ql64);
            Value* qhi = c.b.build_trunc_i32(c.b.build_lshr(ql64, kb.const_i64(32)));
            Value* hlo = c.b.build_trunc_i32(qh64);
            Value* hhi = c.b.build_trunc_i32(c.b.build_lshr(qh64, kb.const_i64(32)));
            for (int e = 0; e < 8; ++e) {
                Value* bw = e < 4 ? qlo : qhi;
                Value* hw = e < 4 ? hlo : hhi;
                Value* bsh = c.i32((e & 3) * 8);
                Value* nib = c.b.build_and(c.b.build_lshr(c.b.build_lshr(bw, bsh), ql_sh), c.i32(0x0f));
                Value* hb = c.b.build_shl(c.b.build_and(c.b.build_lshr(c.b.build_lshr(hw, bsh), qh_sh), c.i32(0x03)),
                                          c.i32(4));
                Value* q6 = c.b.build_sub(c.kb.u32_to_f32(c.b.build_or(nib, hb)), c.f32(32.0f));
                w[static_cast<std::size_t>(e)] = c.b.build_mul(s, q6);
            }
        } else {
            throw std::runtime_error("brotensor: mir_dequant: unknown fmt");
        }
        store_lanes(c, Dtype::FP16, 8, ptr_at(c, p[1], i, 16), w);
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
