#include "mir_fa_aux.h"

#include "mir_common.h"
#include "mir_math.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr float kPScale = 16384.0f;

Value* decode_half_raw(Ctx& c, Dtype dt, Value* raw16) {
    return (dt == Dtype::FP16) ? c.kb.f16_to_f32(raw16) : c.kb.bf16_to_f32(raw16);
}

Value* encode_half_raw(Ctx& c, Dtype dt, Value* v) {
    return (dt == Dtype::FP16) ? c.kb.f32_to_f16(v) : c.kb.f32_to_bf16(v);
}

// Compute power-of-two scale and inverse scale from *sbits:
// frexpf(b, &e) where b > 0 && isfinite(b) => e = exp_raw - 126,
// 14 - e = 140 - exp_raw clamped to [-120, 120].
std::pair<Value*, Value*> emit_bw_ds_scale_and_inv(Ctx& c, Value* sbits) {
    Value* u = c.kb.load_i32(sbits);
    Value* exp_raw = c.b.build_and(c.b.build_lshr(u, c.i32(23)), c.i32(0xff));
    Value* positive = c.b.build_sgt(u, c.i32(0));
    Value* finite = c.b.build_slt(exp_raw, c.i32(255));
    Value* valid = c.b.build_and(positive, finite);
    Value* shift = clamp_i32(c, c.b.build_sub(c.i32(140), exp_raw), c.i32(-120), c.i32(120));
    Value* scale_bits = c.b.build_shl(c.b.build_add(shift, c.i32(127)), c.i32(23));
    Value* inv_bits = c.b.build_shl(c.b.build_sub(c.i32(127), shift), c.i32(23));
    Value* scale = c.b.build_select(valid, c.b.build_bitcast_f32_i32(scale_bits), c.f32(1.0f));
    Value* inv = c.b.build_select(valid, c.b.build_bitcast_f32_i32(inv_bits), c.f32(1.0f));
    return {scale, inv};
}

brass::Function* build_split_combine(brass::Module& mod, Dtype dt) {
    const std::string name = std::string("mir_fa_split_combine_") + (dt == Dtype::FP16 ? "f16" : "bf16");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:part_o 1:part_ml 2:O | 3:Lq 4:D 5:heads 6:hd 7:splits
    Value* part_o = p[0];
    Value* part_ml = p[1];
    Value* O = p[2];
    Value* Lq = p[3];
    Value* D = p[4];
    Value* heads = p[5];
    Value* hd = p[6];
    Value* splits = p[7];

    Value* row = kb.ctaid_x();
    Value* head = kb.ctaid_y();
    Value* b = kb.ctaid_z();
    Value* bs_base = c.b.build_mul(b, splits);
    Value* head2 = c.b.build_shl(head, c.i32(1));
    Value* head_col = c.b.build_mul(head, hd);

    Value* m = kb.for_range_reduce(
        c.i32(0), splits, c.i32(1), c.f32(-1e30f),
        [&](Value* s, Value* cur) -> Value* {
            Value* pr = c.b.build_add(c.b.build_mul(c.b.build_add(bs_base, s), Lq), row);
            Value* idx = c.b.build_add(c.b.build_mul(c.b.build_mul(pr, heads), c.i32(2)), head2);
            Value* mv = kb.load_f32(ptr_at(c, part_ml, idx, 4));
            Value* lv = kb.load_f32(ptr_at(c, part_ml, c.b.build_add(idx, c.i32(1)), 4));
            return c.b.build_select(c.b.build_sgt(lv, c.f32(0.0f)), kb.fmax(cur, mv), cur);
        });
    Value* l = kb.for_range_reduce(
        c.i32(0), splits, c.i32(1), c.f32(0.0f),
        [&](Value* s, Value* cur) -> Value* {
            Value* pr = c.b.build_add(c.b.build_mul(c.b.build_add(bs_base, s), Lq), row);
            Value* idx = c.b.build_add(c.b.build_mul(c.b.build_mul(pr, heads), c.i32(2)), head2);
            Value* mv = kb.load_f32(ptr_at(c, part_ml, idx, 4));
            Value* lv = kb.load_f32(ptr_at(c, part_ml, c.b.build_add(idx, c.i32(1)), 4));
            Value* ok = c.b.build_sgt(lv, c.f32(0.0f));
            Value* term = c.b.build_mul(kb.ex2_approx(c.b.build_sub(mv, m)), lv);
            return c.b.build_add(cur, c.b.build_select(ok, term, c.f32(0.0f)));
        });
    Value* inv = c.b.build_select(
        c.b.build_sgt(l, c.f32(0.0f)), kb.div_approx(c.f32(1.0f), l), c.f32(0.0f));
    Value* o_base = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(b, Lq), row), D), head_col);

    kb.for_range(kb.tid_x(), hd, c.i32(128), [&](Value* col) {
        Value* acc = kb.for_range_reduce(
            c.i32(0), splits, c.i32(1), c.f32(0.0f),
            [&](Value* s, Value* cur) -> Value* {
                Value* pr = c.b.build_add(c.b.build_mul(c.b.build_add(bs_base, s), Lq), row);
                Value* ml_idx = c.b.build_add(c.b.build_mul(c.b.build_mul(pr, heads), c.i32(2)), head2);
                Value* mv = kb.load_f32(ptr_at(c, part_ml, ml_idx, 4));
                Value* lv = kb.load_f32(ptr_at(c, part_ml, c.b.build_add(ml_idx, c.i32(1)), 4));
                Value* ok = c.b.build_sgt(lv, c.f32(0.0f));
                Value* w = c.b.build_select(ok, kb.ex2_approx(c.b.build_sub(mv, m)), c.f32(0.0f));
                Value* pv = kb.load_f32(ptr_at(c, part_o, c.b.build_add(c.b.build_add(c.b.build_mul(pr, D), head_col), col), 4));
                return c.b.build_fma_f32(w, pv, cur);
            });
        store_elem_dt(c, dt, O, c.b.build_add(o_base, col), c.b.build_mul(acc, inv));
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_gather(brass::Module& mod, Dtype dt) {
    const std::string name = std::string("mir_fa_gather_") + (dt == Dtype::FP16 ? "f16" : "bf16");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:X 1:Y | 2:ldX 3:col0 4:rows 5:hd 6:rows_pad 7:ldY
    Value* total = c.b.build_mul(p[6], p[7]);
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, total), [&] {
        Value* r = c.b.build_udiv(idx, p[7]);
        Value* d = c.b.build_umod(idx, p[7]);
        Value* ok = c.b.build_and(c.b.build_slt(r, p[4]), c.b.build_slt(d, p[5]));
        Value* x_idx = c.b.build_add(c.b.build_add(c.b.build_mul(r, p[2]), p[3]), d);
        Value* raw = kb.load_u16(c.b.build_select(ok, ptr_at(c, p[0], x_idx, 2), p[0]));
        kb.store_u16(ptr_at(c, p[1], idx, 2), c.b.build_select(ok, raw, c.i32(0)));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_gather_T(brass::Module& mod, Dtype dt, bool dup) {
    const std::string name = std::string("mir_fa_gather_T_") + (dt == Dtype::FP16 ? "f16" : "bf16") +
                             (dup ? "_dup" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:X 1:Y | 2:ldX 3:col0 4:rows 5:hd 6:rows_pad 7:hdp 8:ldY
    Value* total = c.b.build_mul(p[7], p[6]);
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, total), [&] {
        Value* d = c.b.build_udiv(idx, p[6]);
        Value* r = c.b.build_umod(idx, p[6]);
        Value* ok = c.b.build_and(c.b.build_slt(r, p[4]), c.b.build_slt(d, p[5]));
        Value* x_idx = c.b.build_add(c.b.build_add(c.b.build_mul(r, p[2]), p[3]), d);
        Value* raw = kb.load_u16(c.b.build_select(ok, ptr_at(c, p[0], x_idx, 2), p[0]));
        Value* v = c.b.build_select(ok, raw, c.i32(0));
        Value* y_idx = c.b.build_add(c.b.build_mul(d, p[8]), r);
        kb.store_u16(ptr_at(c, p[1], y_idx, 2), v);
        if (dup) {
            kb.store_u16(ptr_at(c, p[1], c.b.build_add(y_idx, p[6]), 2), v);
        }
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_scatter_head(brass::Module& mod, Dtype dt) {
    const std::string name = std::string("mir_fa_scatter_head_") + (dt == Dtype::FP16 ? "f16" : "bf16");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:Y 1:Out | 2:L 3:D 4:col0 5:hd 6:ldY
    Value* total = c.b.build_mul(p[2], p[5]);
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, total), [&] {
        Value* l = c.b.build_udiv(idx, p[5]);
        Value* d = c.b.build_umod(idx, p[5]);
        Value* y_idx = c.b.build_add(c.b.build_mul(l, p[6]), d);
        Value* o_idx = c.b.build_add(c.b.build_add(c.b.build_mul(l, p[3]), p[4]), d);
        kb.store_u16(ptr_at(c, p[1], o_idx, 2), kb.load_u16(ptr_at(c, p[0], y_idx, 2)));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_softmax_rows_f32(brass::Module& mod, Dtype dt, bool has_mask) {
    const std::string name = std::string("mir_fa_softmax_rows_") +
                             (dt == Dtype::FP16 ? "f16" : "bf16") + (has_mask ? "_m" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:S 1:P 2:mask | 3:Lk 4:ld | 5:scale
    Value* s8 = kb.shared_alloc_f32(8);
    Value* row_off = c.b.build_mul(kb.ctaid_x(), p[4]);
    Value* srow = ptr_at(c, p[0], row_off, 4);
    Value* prow = ptr_at(c, p[1], row_off, 2);
    Value* tid = kb.tid_x();

    auto key_ok = [&](Value* k) -> Value* {
        if (!has_mask) return c.kb.const_i32(1);
        return c.b.build_sgt(kb.load_f32(ptr_at(c, p[2], k, 4)), c.f32(0.5f));
    };

    Value* local_max = kb.for_range_reduce(
        tid, p[3], c.i32(256), c.f32(-1e30f),
        [&](Value* k, Value* cur) -> Value* {
            Value* sv = c.b.build_mul(kb.load_f32(ptr_at(c, srow, k, 4)), p[5]);
            return has_mask ? c.b.build_select(key_ok(k), kb.fmax(cur, sv), cur) : kb.fmax(cur, sv);
        });
    Value* mx = emit_block_max_f32(c, s8, local_max);
    Value* empty = c.b.build_sle(mx, c.f32(-1e29f));

    Value* local_sum = kb.for_range_reduce(
        tid, p[3], c.i32(256), c.f32(0.0f),
        [&](Value* k, Value* cur) -> Value* {
            Value* ev = emit_exp(c, c.b.build_sub(c.b.build_mul(kb.load_f32(ptr_at(c, srow, k, 4)), p[5]), mx));
            Value* ok = has_mask ? key_ok(k) : c.i32(1);
            return c.b.build_add(cur, c.b.build_select(c.b.build_ne(ok, c.i32(0)), ev, c.f32(0.0f)));
        });
    Value* sum = emit_block_sum_f32(c, s8, local_sum);
    Value* valid_row = c.b.build_and(c.b.build_not(empty), c.b.build_sgt(sum, c.f32(0.0f)));
    Value* inv = c.b.build_select(valid_row, emit_div_rn(c, c.f32(1.0f), sum), c.f32(0.0f));

    kb.for_range(tid, p[4], c.i32(256), [&](Value* k) {
        Value* in_k = c.b.build_slt(k, p[3]);
        Value* safe_k = c.b.build_select(in_k, k, c.i32(0));
        Value* ev = emit_exp(c, c.b.build_sub(c.b.build_mul(kb.load_f32(ptr_at(c, srow, safe_k, 4)), p[5]), mx));
        Value* ok = has_mask ? c.b.build_and(in_k, c.b.build_sgt(kb.load_f32(ptr_at(c, p[2], safe_k, 4)), c.f32(0.5f)))
                             : in_k;
        Value* val = c.b.build_select(ok, c.b.build_mul(ev, inv), c.f32(0.0f));
        store_elem_dt(c, dt, prow, k, val);
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_bw_scatter(brass::Module& mod, Dtype dt, bool has_sbits) {
    const std::string name = std::string("mir_fa_bw_scatter_") + (dt == Dtype::FP16 ? "f16" : "bf16") +
                             (has_sbits ? "_sb" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:Y 1:Out 2:sbits | 3:ldY 4:rows 5:hd 6:ldO 7:col0
    Value* total = c.b.build_mul(p[4], p[5]);
    Value* idx = kb.global_tid_x();
    Value* inv = has_sbits ? emit_bw_ds_scale_and_inv(c, p[2]).second : c.f32(1.0f);
    kb.if_then(c.b.build_slt(idx, total), [&] {
        Value* r = c.b.build_udiv(idx, p[5]);
        Value* d = c.b.build_umod(idx, p[5]);
        Value* yv = kb.load_f32(ptr_at(c, p[0], c.b.build_add(c.b.build_mul(r, p[3]), d), 4));
        Value* o_idx = c.b.build_add(c.b.build_add(c.b.build_mul(r, p[6]), p[7]), d);
        store_elem_dt(c, dt, p[1], o_idx, c.b.build_mul(yv, inv));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_bw_acc(brass::Module& mod, bool has_sbits, bool first) {
    const std::string name = std::string("mir_fa_bw_acc") + (has_sbits ? "_sb" : "") + (first ? "_1st" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:acc 1:src 2:sbits | 3:n | 4:inv_const
    Value* inv = has_sbits ? emit_bw_ds_scale_and_inv(c, p[2]).second : p[4];
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, p[3]), [&] {
        Value* ptr_acc = ptr_at(c, p[0], idx, 4);
        Value* v = c.b.build_mul(kb.load_f32(ptr_at(c, p[1], idx, 4)), inv);
        kb.store_f32(ptr_acc, first ? v : c.b.build_add(kb.load_f32(ptr_acc), v));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_bw_row_stats(brass::Module& mod, bool causal, bool has_mask) {
    const std::string name = std::string("mir_fa_bw_row_stats") + (causal ? "_c" : "") + (has_mask ? "_m" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(),
        Type::f32(), Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:S 1:dP 2:mask 3:stats 4:bound_bits | 5:ld 6:Lk 7:q0 | 8:scale_log2 9:scale
    Value* s8 = kb.shared_alloc_f32(8);
    Value* i = kb.ctaid_x();
    Value* row_off = c.b.build_mul(i, p[5]);
    Value* s = ptr_at(c, p[0], row_off, 4);
    Value* dp = ptr_at(c, p[1], row_off, 4);
    Value* qa = c.b.build_add(p[7], i);
    Value* kend = causal ? smin_i32(c, p[6], c.b.build_add(qa, c.i32(1))) : p[6];
    Value* tid = kb.tid_x();

    Value* local_mx = kb.for_range_reduce(
        tid, kend, c.i32(256), c.f32(-1e30f),
        [&](Value* k, Value* cur) -> Value* {
            Value* sv = c.b.build_mul(kb.load_f32(ptr_at(c, s, k, 4)), p[8]);
            if (!has_mask) return kb.fmax(cur, sv);
            Value* ok = c.b.build_sgt(kb.load_f32(ptr_at(c, p[2], k, 4)), c.f32(0.5f));
            return c.b.build_select(ok, kb.fmax(cur, sv), cur);
        });
    Value* mx = emit_block_max_f32(c, s8, local_mx);
    Value* has_mx = c.b.build_sgt(mx, c.f32(-1e29f));

    auto acc3 = emit_if_else_values(
        c, has_mx, {Type::f32(), Type::f32(), Type::f32()},
        [&]() -> std::vector<Value*> {
            return kb.for_range_reduce_n(
                tid, kend, c.i32(256), {c.f32(0.0f), c.f32(0.0f), c.f32(0.0f)},
                [&](Value* k, const std::vector<Value*>& cur) -> std::vector<Value*> {
                    Value* ok = has_mask ? c.b.build_sgt(kb.load_f32(ptr_at(c, p[2], k, 4)), c.f32(0.5f))
                                         : c.i32(1);
                    Value* sv = kb.load_f32(ptr_at(c, s, k, 4));
                    Value* dpv = kb.load_f32(ptr_at(c, dp, k, 4));
                    Value* ev = kb.ex2_approx(c.b.build_sub(c.b.build_mul(sv, p[8]), mx));
                    if (!has_mask) {
                        return {
                            c.b.build_add(cur[0], ev),
                            c.b.build_fma_f32(ev, dpv, cur[1]),
                            kb.fmax(cur[2], kb.fabs(dpv)),
                        };
                    }
                    Value* b = c.b.build_ne(ok, c.i32(0));
                    return {
                        c.b.build_select(b, c.b.build_add(cur[0], ev), cur[0]),
                        c.b.build_select(b, c.b.build_fma_f32(ev, dpv, cur[1]), cur[1]),
                        c.b.build_select(b, kb.fmax(cur[2], kb.fabs(dpv)), cur[2]),
                    };
                });
        },
        [&]() -> std::vector<Value*> { return {c.f32(0.0f), c.f32(0.0f), c.f32(0.0f)}; });

    Value* l = emit_block_sum_f32(c, s8, acc3[0]);
    Value* a = emit_block_sum_f32(c, s8, acc3[1]);
    Value* g = emit_block_max_f32(c, s8, acc3[2]);

    kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
        Value* any = c.b.build_sgt(l, c.f32(0.0f));
        Value* dq = c.b.build_select(any, emit_div_rn(c, a, l), c.f32(0.0f));
        Value* inv_l = c.b.build_select(any, emit_div_rn(c, c.f32(1.0f), l), c.f32(0.0f));
        Value* i3 = c.b.build_mul(i, c.i32(3));
        kb.store_f32(ptr_at(c, p[3], i3, 4), c.b.build_select(any, mx, c.f32(0.0f)));
        kb.store_f32(ptr_at(c, p[3], c.b.build_add(i3, c.i32(1)), 4), inv_l);
        kb.store_f32(ptr_at(c, p[3], c.b.build_add(i3, c.i32(2)), 4), dq);

        Value* bval = c.b.build_select(any, c.b.build_mul(p[9], c.b.build_add(g, kb.fabs(dq))), c.f32(0.0f));
        Value* bu = c.b.build_bitcast_i32_f32(bval);
        Value* fin = c.b.build_ult(c.b.build_and(bu, c.i32(0x7f800000)), c.i32(0x7f800000));
        kb.if_then(fin, [&] { kb.atom_max_u32(p[4], bu); });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_bw_ds_tile(brass::Module& mod, Dtype dt, bool causal, bool has_mask) {
    const std::string name = std::string("mir_fa_bw_ds_") + (dt == Dtype::FP16 ? "f16" : "bf16") +
                             (causal ? "_c" : "") + (has_mask ? "_m" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::f32(), Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:S 1:dP 2:stats 3:sbits 4:mask 5:dS 6:dST 7:PT |
    //    8:ldS 9:m 10:m_pad 11:Lk 12:Lk_pad 13:q0 | 14:scale_log2 15:scale
    Value* tp = kb.shared_alloc_f32(32 * 33);
    Value* tds = kb.shared_alloc_f32(32 * 33);

    Value* j0 = c.b.build_shl(kb.ctaid_x(), c.i32(5));
    Value* i0 = c.b.build_shl(kb.ctaid_y(), c.i32(5));
    Value* tid = kb.tid_x();
    Value* tx = c.b.build_and(tid, c.i32(31));
    Value* ty = c.b.build_lshr(tid, c.i32(5));

    Value* j = c.b.build_add(j0, tx);
    Value* in_lk = c.b.build_slt(j, p[11]);
    Value* jon = in_lk;
    if (has_mask) {
        Value* safe_j = c.b.build_select(in_lk, j, c.i32(0));
        jon = c.b.build_and(in_lk, c.b.build_sgt(kb.load_f32(ptr_at(c, p[4], safe_j, 4)), c.f32(0.5f)));
    }
    Value* dsc = c.b.build_mul(emit_bw_ds_scale_and_inv(c, p[3]).first, p[15]);
    Value* lkp2 = c.b.build_shl(p[12], c.i32(1));
    Value* mp2 = c.b.build_shl(p[10], c.i32(1));

    for (int step = 0; step < 4; ++step) {
        Value* r = c.b.build_add(ty, c.i32(step * 8));
        Value* i = c.b.build_add(i0, r);
        Value* in_m = c.b.build_slt(i, p[9]);
        Value* attend = c.b.build_and(in_m, jon);
        if (causal) {
            attend = c.b.build_and(attend, c.b.build_sle(j, c.b.build_add(p[13], i)));
        }
        auto pds = emit_if_else_values(
            c, attend, {Type::f32(), Type::f32()},
            [&]() -> std::vector<Value*> {
                Value* i3 = c.b.build_mul(i, c.i32(3));
                Value* st0 = kb.load_f32(ptr_at(c, p[2], i3, 4));
                Value* st1 = kb.load_f32(ptr_at(c, p[2], c.b.build_add(i3, c.i32(1)), 4));
                Value* st2 = kb.load_f32(ptr_at(c, p[2], c.b.build_add(i3, c.i32(2)), 4));
                Value* pos = c.b.build_sgt(st1, c.f32(0.0f));
                Value* o = c.b.build_add(c.b.build_mul(i, p[8]), j);
                Value* sv = kb.load_f32(ptr_at(c, p[0], o, 4));
                Value* dpv = kb.load_f32(ptr_at(c, p[1], o, 4));
                Value* pv = c.b.build_mul(kb.ex2_approx(c.b.build_sub(c.b.build_mul(sv, p[14]), st0)), st1);
                Value* dsv = c.b.build_mul(c.b.build_mul(pv, c.b.build_sub(dpv, st2)), dsc);
                return {
                    c.b.build_select(pos, pv, c.f32(0.0f)),
                    c.b.build_select(pos, dsv, c.f32(0.0f)),
                };
            },
            [&]() -> std::vector<Value*> { return {c.f32(0.0f), c.f32(0.0f)}; });

        kb.if_then(c.b.build_and(in_m, c.b.build_slt(j, p[12])), [&] {
            Value* ds = pds[1];
            Value* hi_raw = encode_half_raw(c, dt, ds);
            Value* hi_f = decode_half_raw(c, dt, hi_raw);
            Value* lo_raw = encode_half_raw(c, dt, c.b.build_sub(ds, hi_f));
            Value* ds_idx = c.b.build_add(c.b.build_mul(i, lkp2), j);
            kb.store_u16(ptr_at(c, p[5], ds_idx, 2), hi_raw);
            kb.store_u16(ptr_at(c, p[5], c.b.build_add(ds_idx, p[12]), 2), lo_raw);
        });
        Value* sm_idx = c.b.build_add(c.b.build_mul(r, c.i32(33)), tx);
        kb.shared_store_f32_indexed(tp, sm_idx, pds[0]);
        kb.shared_store_f32_indexed(tds, sm_idx, pds[1]);
    }
    kb.sync();

    for (int step = 0; step < 4; ++step) {
        Value* r = c.b.build_add(ty, c.i32(step * 8));
        Value* jj = c.b.build_add(j0, r);
        Value* ii = c.b.build_add(i0, tx);
        kb.if_then(c.b.build_and(c.b.build_slt(jj, p[12]), c.b.build_slt(ii, p[10])), [&] {
            Value* sm_idx = c.b.build_add(c.b.build_mul(tx, c.i32(33)), r);
            Value* pv = kb.shared_load_f32_indexed(tp, sm_idx);
            Value* ds = kb.shared_load_f32_indexed(tds, sm_idx);
            kb.store_u16(
                ptr_at(c, p[7], c.b.build_add(c.b.build_mul(jj, p[10]), ii), 2),
                encode_half_raw(c, dt, c.b.build_mul(pv, c.f32(kPScale))));
            Value* hi_raw = encode_half_raw(c, dt, ds);
            Value* hi_f = decode_half_raw(c, dt, hi_raw);
            Value* lo_raw = encode_half_raw(c, dt, c.b.build_sub(ds, hi_f));
            Value* dst_idx = c.b.build_add(c.b.build_mul(jj, mp2), ii);
            kb.store_u16(ptr_at(c, p[6], dst_idx, 2), hi_raw);
            kb.store_u16(ptr_at(c, p[6], c.b.build_add(dst_idx, p[10]), 2), lo_raw);
        });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace

brass::Function* build_fa_aux_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint8_t flags) {
    switch (op) {
        case 0u: return build_split_combine(mod, dt);
        case 1u: return build_gather(mod, dt);
        case 2u: return build_gather_T(mod, dt, (flags & 1u) != 0);
        case 3u: return build_scatter_head(mod, dt);
        case 4u: return build_softmax_rows_f32(mod, dt, (flags & 1u) != 0);
        case 5u: return build_bw_scatter(mod, dt, (flags & 1u) != 0);
        case 6u: return build_bw_acc(mod, (flags & 1u) != 0, (flags & 2u) != 0);
        case 7u: return build_bw_row_stats(mod, (flags & 1u) != 0, (flags & 2u) != 0);
        case 8u: return build_bw_ds_tile(mod, dt, (flags & 1u) != 0, (flags & 2u) != 0);
        default: throw std::runtime_error("build_fa_aux_mir: unsupported op");
    }
}

}  // namespace brotensor::kernels
