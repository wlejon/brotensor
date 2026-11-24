#include "mir_audio.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* clamp_i32(Ctx& c, Value* v, Value* lo, Value* hi) {
    Value* a = c.b.build_select(c.b.build_slt(v, lo), lo, v);
    return c.b.build_select(c.b.build_sgt(a, hi), hi, a);
}

Value* guard_denom(Ctx& c, Value* d) {
    Value* pos_min = c.f32(1e-9f);
    Value* neg_min = c.f32(-1e-9f);
    Value* is_pos = c.b.build_sge(d, c.f32(0.0f));
    Value* p_clamp = c.b.build_select(c.b.build_slt(d, pos_min), pos_min, d);
    Value* n_clamp = c.b.build_select(c.b.build_sgt(d, neg_min), neg_min, d);
    return c.b.build_select(is_pos, p_clamp, n_clamp);
}

Value* pad1d_src(Ctx& c, Value* p, Value* L, Value* left, std::uint32_t mode) {
    Value* rel = c.b.build_sub(p, left);
    Value* in_mid = c.b.build_and(c.b.build_sge(rel, c.i32(0)), c.b.build_slt(rel, L));
    if (mode == 0u) {
        return c.b.build_select(in_mid, rel, c.i32(-1));
    }
    if (mode == 2u) {
        Value* cl = c.b.build_select(c.b.build_slt(rel, c.i32(0)), c.i32(0), c.b.build_sub(L, c.i32(1)));
        return c.b.build_select(in_mid, rel, cl);
    }
    // Reflect (mode == 1)
    return emit_if_else_values(
        c, in_mid, {Type::i32()},
        [&]() -> std::vector<Value*> { return {rel}; },
        [&]() -> std::vector<Value*> {
            Value* l_one = c.b.build_eq(L, c.i32(1));
            return emit_if_else_values(
                c, l_one, {Type::i32()},
                [&]() -> std::vector<Value*> { return {c.i32(0)}; },
                [&]() -> std::vector<Value*> {
                    Value* period = c.b.build_shl(c.b.build_sub(L, c.i32(1)), c.i32(1));
                    Value* q0 = c.b.build_smod(rel, period);
                    Value* q = c.b.build_select(c.b.build_slt(q0, c.i32(0)), c.b.build_add(q0, period), q0);
                    return {c.b.build_select(c.b.build_slt(q, L), q, c.b.build_sub(period, q))};
                });
        })[0];
}

struct ResampleTap {
    Value* x0;
    Value* x1;
    Value* f;
};

ResampleTap resample1d_src(Ctx& c, Value* dst, Value* lin, Value* lout, std::uint32_t mode) {
    Value* lin64 = c.b.build_zext_i64(lin);
    Value* lout64 = c.b.build_zext_i64(lout);
    Value* dst64 = c.b.build_zext_i64(dst);
    Value* two_d_p1 = c.b.build_add(c.b.build_shl(dst64, c.kb.const_i64(1)), c.kb.const_i64(1));
    Value* num = c.b.build_sub(c.b.build_mul(two_d_p1, lin64), lout64);
    Value* den = c.b.build_shl(lout64, c.kb.const_i64(1));

    if (mode == 0u) {
        Value* is_neg = c.b.build_slt(num, c.kb.const_i64(0));
        Value* abs_n = c.b.build_select(is_neg, c.b.build_neg(num), num);
        Value* q_neg = c.b.build_neg(c.b.build_udiv(
            c.b.build_sub(c.b.build_add(abs_n, den), c.kb.const_i64(1)), den));
        Value* q_pos = c.b.build_udiv(c.b.build_select(is_neg, c.kb.const_i64(0), num), den);
        Value* q = c.b.build_select(is_neg, q_neg, q_pos);
        Value* r2 = c.b.build_shl(c.b.build_sub(num, c.b.build_mul(q, den)), c.kb.const_i64(1));
        Value* gt = c.b.build_sgt(r2, den);
        Value* tie_odd = c.b.build_and(c.b.build_eq(r2, den),
                                       c.b.build_ne(c.b.build_and(q, c.kb.const_i64(1)), c.kb.const_i64(0)));
        Value* q_rnd = c.b.build_select(c.b.build_or(gt, tie_odd), c.b.build_add(q, c.kb.const_i64(1)), q);
        Value* x0 = clamp_i32(c, c.b.build_trunc_i32(q_rnd), c.i32(0), c.b.build_sub(lin, c.i32(1)));
        return {x0, x0, c.f32(0.0f)};
    }

    Value* le_zero = c.b.build_sle(num, c.kb.const_i64(0));
    Value* lin_m1 = c.b.build_sub(lin, c.i32(1));
    Value* x1_zero = c.b.build_select(c.b.build_slt(c.i32(1), lin_m1), c.i32(1), lin_m1);
    auto res = emit_if_else_values(
        c, le_zero, {Type::i32(), Type::i32(), Type::f32()},
        [&]() -> std::vector<Value*> { return {c.i32(0), x1_zero, c.f32(0.0f)}; },
        [&]() -> std::vector<Value*> {
            Value* q = c.b.build_udiv(num, den);
            Value* past = c.b.build_sge(q, c.b.build_sub(lin64, c.kb.const_i64(1)));
            return emit_if_else_values(
                c, past, {Type::i32(), Type::i32(), Type::f32()},
                [&]() -> std::vector<Value*> { return {lin_m1, lin_m1, c.f32(0.0f)}; },
                [&]() -> std::vector<Value*> {
                    Value* x0 = c.b.build_trunc_i32(q);
                    Value* rem = c.b.build_sub(num, c.b.build_mul(q, den));
                    Value* f = emit_div_rn(c, c.b.build_sitofp_f32_i64(rem), c.b.build_sitofp_f32_i64(den));
                    return {x0, c.b.build_add(x0, c.i32(1)), f};
                });
        });
    return {res[0], res[1], res[2]};
}

void build_snake(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool has_beta, bool bwd) {
    if (!bwd) {
        // p: X(0) alpha(1) beta(2) Y(3) C(4) L(5) total(6)
        Value* C = p[4];
        Value* L = p[5];
        emit_vec_grid_stride(c, p[6], 1, kAudioBlock, [&](Value* i) {
            Value* ch = c.b.build_umod(c.b.build_udiv(i, L), C);
            Value* al = c.kb.load_f32(ptr_at(c, p[1], ch, 4));
            Value* d_raw = has_beta ? c.kb.load_f32(ptr_at(c, p[2], ch, 4)) : al;
            Value* r = c.b.build_sdiv(c.f32(1.0f), guard_denom(c, d_raw));
            Value* x = load_elem_dt(c, dt, p[0], i);
            auto [s, co] = emit_sincos_acc(c, c.b.build_mul(al, x));
            (void)co;
            Value* y = c.b.build_add(x, c.b.build_mul(r, c.b.build_mul(s, s)));
            store_elem_dt(c, dt, p[3], i, y);
        });
        return;
    }
    // bwd: X(0) alpha(1) beta(2) dY(3) dX(4) dAlpha(5) dBeta(6) C(7) L(8) N(9)
    Value* C = p[7];
    Value* L = p[8];
    Value* per = c.b.build_mul(p[9], L);
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kAudioBlock));
    Value* s_red = c.kb.shared_alloc_f32(8);

    c.kb.for_range(c.kb.ctaid_x(), C, c.kb.nctaid_x(), [&](Value* ch) {
        Value* al = c.kb.load_f32(ptr_at(c, p[1], ch, 4));
        Value* d_raw = has_beta ? c.kb.load_f32(ptr_at(c, p[2], ch, 4)) : al;
        Value* r = c.b.build_sdiv(c.f32(1.0f), guard_denom(c, d_raw));
        auto sums = c.kb.for_range_reduce_n(
            tid, per, block, {c.f32(0.0f), c.f32(0.0f)}, [&](Value* j, const std::vector<Value*>& cur) {
                Value* n = c.b.build_udiv(j, L);
                Value* l = c.b.build_sub(j, c.b.build_mul(n, L));
                Value* idx = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, C), ch), L), l);
                Value* x = load_elem_dt(c, dt, p[0], idx);
                Value* dy = load_elem_dt(c, dt, p[3], idx);
                auto [s, co] = emit_sincos_acc(c, c.b.build_mul(al, x));
                Value* sc = c.b.build_mul(s, co);
                Value* two_al_r_sc = c.b.build_mul(c.b.build_mul(c.f32(2.0f), al), c.b.build_mul(r, sc));
                store_elem_dt(c, dt, p[4], idx, c.b.build_mul(dy, c.b.build_add(c.f32(1.0f), two_al_r_sc)));
                Value* da_inc = c.b.build_mul(dy, c.b.build_mul(c.b.build_mul(c.f32(2.0f), r), c.b.build_mul(x, sc)));
                Value* t = c.b.build_mul(dy, c.b.build_mul(c.b.build_neg(c.b.build_mul(r, r)), c.b.build_mul(s, s)));
                Value* n_da = c.b.build_add(cur[0], has_beta ? da_inc : c.b.build_add(da_inc, t));
                Value* n_db = has_beta ? c.b.build_add(cur[1], t) : cur[1];
                return std::vector<Value*>{n_da, n_db};
            });
        Value* tot_a = emit_block_sum_f32(c, s_red, sums[0]);
        Value* tot_b = has_beta ? emit_block_sum_f32(c, s_red, sums[1]) : nullptr;
        c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            Value* ap = ptr_at(c, p[5], ch, 4);
            c.kb.store_f32(ap, c.b.build_add(c.kb.load_f32(ap), tot_a));
            if (has_beta) {
                Value* bp = ptr_at(c, p[6], ch, 4);
                c.kb.store_f32(bp, c.b.build_add(c.kb.load_f32(bp), tot_b));
            }
        });
    });
}

void build_pad1d(Ctx& c, const std::vector<Value*>& p, Dtype dt, std::uint32_t mode, bool bwd) {
    // p: src(0) dst(1) L(2) left(3) L_pad(4) rows(5)
    Value* L = p[2];
    Value* left = p[3];
    Value* L_pad = p[4];
    Value* rows = p[5];
    if (!bwd) {
        emit_vec_grid_stride(c, c.b.build_mul(rows, L_pad), 1, kAudioBlock, [&](Value* i) {
            Value* row = c.b.build_udiv(i, L_pad);
            Value* pos = c.b.build_sub(i, c.b.build_mul(row, L_pad));
            Value* s = pad1d_src(c, pos, L, left, mode);
            Value* val = emit_if_else_values(
                c, c.b.build_sge(s, c.i32(0)), {Type::f32()},
                [&]() -> std::vector<Value*> {
                    return {load_elem_dt(c, dt, p[0], c.b.build_add(c.b.build_mul(row, L), s))};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            store_elem_dt(c, dt, p[1], i, val);
        });
        return;
    }
    emit_vec_grid_stride(c, c.b.build_mul(rows, L), 1, kAudioBlock, [&](Value* i) {
        Value* row = c.b.build_udiv(i, L);
        Value* l = c.b.build_sub(i, c.b.build_mul(row, L));
        Value* base = c.b.build_mul(row, L_pad);
        if (mode == 1u) {
            Value* has_l = c.b.build_and(c.b.build_sge(l, c.i32(1)), c.b.build_sle(l, left));
            Value* a0 = emit_if_else_values(
                c, has_l, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    return {load_elem_dt(c, dt, p[0], c.b.build_add(base, c.b.build_sub(left, l)))};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            Value* a1 = c.b.build_add(a0, load_elem_dt(c, dt, p[0], c.b.build_add(base, c.b.build_add(left, l))));
            Value* pr = c.b.build_sub(c.b.build_add(left, c.b.build_shl(c.b.build_sub(L, c.i32(1)), c.i32(1))), l);
            Value* has_r = c.b.build_and(
                c.b.build_sgt(L, c.i32(1)),
                c.b.build_and(c.b.build_sge(pr, c.b.build_add(left, L)), c.b.build_slt(pr, L_pad)));
            Value* a2 = emit_if_else_values(
                c, has_r, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    return {c.b.build_add(a1, load_elem_dt(c, dt, p[0], c.b.build_add(base, pr)))};
                },
                [&]() -> std::vector<Value*> { return {a1}; })[0];
            store_elem_dt(c, dt, p[1], i, a2);
        } else if (mode == 2u) {
            Value* a0 = emit_if_else_values(
                c, c.b.build_eq(l, c.i32(0)), {Type::f32()},
                [&]() -> std::vector<Value*> {
                    return {c.kb.for_range_reduce(c.i32(0), left, c.i32(1), c.f32(0.0f), [&](Value* q, Value* cur) {
                        return c.b.build_add(cur, load_elem_dt(c, dt, p[0], c.b.build_add(base, q)));
                    })};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            Value* a1 = c.b.build_add(a0, load_elem_dt(c, dt, p[0], c.b.build_add(base, c.b.build_add(left, l))));
            Value* a2 = emit_if_else_values(
                c, c.b.build_eq(l, c.b.build_sub(L, c.i32(1))), {Type::f32()},
                [&]() -> std::vector<Value*> {
                    return {c.kb.for_range_reduce(
                        c.b.build_add(left, L), L_pad, c.i32(1), a1, [&](Value* q, Value* cur) {
                            return c.b.build_add(cur, load_elem_dt(c, dt, p[0], c.b.build_add(base, q)));
                        })};
                },
                [&]() -> std::vector<Value*> { return {a1}; })[0];
            store_elem_dt(c, dt, p[1], i, a2);
        } else {
            store_elem_dt(c, dt, p[1], i, load_elem_dt(c, dt, p[0], c.b.build_add(base, c.b.build_add(left, l))));
        }
    });
}

void build_resample1d(Ctx& c, const std::vector<Value*>& p, Dtype dt, std::uint32_t mode, bool bwd) {
    // p: src(0) dst(1) L_in(2) L_out(3) rows(4)
    Value* lin = p[2];
    Value* lout = p[3];
    Value* rows = p[4];
    if (!bwd) {
        emit_vec_grid_stride(c, c.b.build_mul(rows, lout), 1, kAudioBlock, [&](Value* i) {
            Value* row = c.b.build_udiv(i, lout);
            Value* dst = c.b.build_sub(i, c.b.build_mul(row, lout));
            ResampleTap t = resample1d_src(c, dst, lin, lout, mode);
            Value* xb = c.b.build_mul(row, lin);
            Value* v0 = load_elem_dt(c, dt, p[0], c.b.build_add(xb, t.x0));
            Value* y = v0;
            if (mode != 0u) {
                Value* v1 = load_elem_dt(c, dt, p[0], c.b.build_add(xb, t.x1));
                y = c.b.build_add(c.b.build_mul(c.b.build_sub(c.f32(1.0f), t.f), v0), c.b.build_mul(t.f, v1));
            }
            store_elem_dt(c, dt, p[1], i, y);
        });
        return;
    }
    Value* two_lout64 = c.b.build_shl(c.b.build_zext_i64(lout), c.kb.const_i64(1));
    Value* two_lin64 = c.b.build_shl(c.b.build_zext_i64(lin), c.kb.const_i64(1));
    Value* ldiff64 = c.b.build_sub(c.b.build_zext_i64(lout), c.b.build_zext_i64(lin));
    emit_vec_grid_stride(c, c.b.build_mul(rows, lin), 1, kAudioBlock, [&](Value* t) {
        Value* row = c.b.build_udiv(t, lin);
        Value* idx = c.b.build_sub(t, c.b.build_mul(row, lin));
        Value* idx64 = c.b.build_zext_i64(idx);
        Value* lo_num = c.b.build_add(c.b.build_mul(two_lout64, c.b.build_sub(idx64, c.kb.const_i64(1))), ldiff64);
        Value* hi_num = c.b.build_add(c.b.build_mul(two_lout64, c.b.build_add(idx64, c.kb.const_i64(1))), ldiff64);
        Value* dlo0 = c.b.build_sub(c.b.build_trunc_i32(c.b.build_sdiv(lo_num, two_lin64)), c.i32(1));
        Value* dhi0 = c.b.build_add(c.b.build_trunc_i32(c.b.build_sdiv(hi_num, two_lin64)), c.i32(1));
        Value* dlo1 = c.b.build_select(c.b.build_eq(idx, c.i32(0)), c.i32(0), dlo0);
        Value* dhi1 = c.b.build_select(c.b.build_eq(idx, c.b.build_sub(lin, c.i32(1))),
                                       c.b.build_sub(lout, c.i32(1)), dhi0);
        Value* dlo = c.b.build_select(c.b.build_sgt(dlo1, c.i32(0)), dlo1, c.i32(0));
        Value* lout_m1 = c.b.build_sub(lout, c.i32(1));
        Value* dhi = c.b.build_select(c.b.build_slt(dhi1, lout_m1), dhi1, lout_m1);
        Value* yb = c.b.build_mul(row, lout);
        Value* acc = c.kb.for_range_reduce(
            dlo, c.b.build_add(dhi, c.i32(1)), c.i32(1), c.f32(0.0f), [&](Value* dst, Value* cur) {
                ResampleTap tap = resample1d_src(c, dst, lin, lout, mode);
                Value* g = load_elem_dt(c, dt, p[0], c.b.build_add(yb, dst));
                if (mode == 0u) {
                    return c.b.build_select(c.b.build_eq(tap.x0, idx), c.b.build_add(cur, g), cur);
                }
                Value* c1 = c.b.build_select(
                    c.b.build_eq(tap.x0, idx),
                    c.b.build_add(cur, c.b.build_mul(c.b.build_sub(c.f32(1.0f), tap.f), g)), cur);
                return c.b.build_select(
                    c.b.build_eq(tap.x1, idx),
                    c.b.build_add(c1, c.b.build_mul(tap.f, g)), c1);
            });
        store_elem_dt(c, dt, p[1], t, acc);
    });
}

void build_causal_conv1d(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool has_bias) {
    // p: X(0) W(1) bias(2) state(3) Y(4) C(5) L(6) kL(7) dil(8) rows(9)
    Value* C = p[5];
    Value* L = p[6];
    Value* kl = p[7];
    Value* dil = p[8];
    Value* hist = c.b.build_mul(c.b.build_sub(kl, c.i32(1)), dil);
    emit_vec_grid_stride(c, p[9], 1, kAudioBlock, [&](Value* row) {
        Value* ch = c.b.build_umod(row, C);
        Value* sb = c.b.build_mul(row, hist);
        Value* xb = c.b.build_mul(row, L);
        Value* bv = has_bias ? load_elem_dt(c, dt, p[2], ch) : c.f32(0.0f);
        Value* w_base = c.b.build_mul(ch, kl);
        auto read_src = [&](Value* idx) {
            return emit_if_else_values(
                c, c.b.build_ult(idx, hist), {Type::f32()},
                [&]() -> std::vector<Value*> { return {load_elem_dt(c, dt, p[3], c.b.build_add(sb, idx))}; },
                [&]() -> std::vector<Value*> {
                    return {load_elem_dt(c, dt, p[0], c.b.build_add(xb, c.b.build_sub(idx, hist)))};
                })[0];
        };
        c.kb.for_range(c.i32(0), L, c.i32(1), [&](Value* t) {
            Value* acc = c.kb.for_range_reduce(c.i32(0), kl, c.i32(1), bv, [&](Value* k, Value* cur) {
                Value* v = read_src(c.b.build_add(t, c.b.build_mul(k, dil)));
                Value* w = load_elem_dt(c, dt, p[1], c.b.build_add(w_base, k));
                return c.b.build_add(cur, c.b.build_mul(w, v));
            });
            store_elem_dt(c, dt, p[4], c.b.build_add(xb, t), acc);
        });
        c.kb.for_range(c.i32(0), hist, c.i32(1), [&](Value* s) {
            store_elem_dt(c, dt, p[3], c.b.build_add(sb, s), read_src(c.b.build_add(L, s)));
        });
    });
}

void build_convt1d_bwd_w(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    // p: X(0) dY(1) dW(2) N(3) Cin(4) L(5) Cout(6) kL(7) st(8) pad(9) dil(10) Cgin(11) Cgout(12) Lout(13)
    Value* Cin = p[4];
    Value* L = p[5];
    Value* Cout = p[6];
    Value* kL = p[7];
    Value* st = p[8];
    Value* pad = p[9];
    Value* dil = p[10];
    Value* Cgin = p[11];
    Value* Cgout = p[12];
    Value* Lout = p[13];
    Value* total = c.b.build_mul(c.b.build_mul(Cin, Cgout), kL);
    Value* per = c.b.build_mul(p[3], L);
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kAudioBlock));
    Value* s_red = c.kb.shared_alloc_f32(8);

    c.kb.for_range(c.kb.ctaid_x(), total, c.kb.nctaid_x(), [&](Value* w) {
        Value* tmp = c.b.build_udiv(w, kL);
        Value* k = c.b.build_sub(w, c.b.build_mul(tmp, kL));
        Value* cin = c.b.build_udiv(tmp, Cgout);
        Value* ocl = c.b.build_sub(tmp, c.b.build_mul(cin, Cgout));
        Value* g = c.b.build_udiv(cin, Cgin);
        Value* oc = c.b.build_add(c.b.build_mul(g, Cgout), ocl);
        Value* k_dil = c.b.build_mul(k, dil);
        Value* acc = c.kb.for_range_reduce(tid, per, block, c.f32(0.0f), [&](Value* j, Value* cur) {
            Value* n = c.b.build_udiv(j, L);
            Value* l = c.b.build_sub(j, c.b.build_mul(n, L));
            Value* lo = c.b.build_add(c.b.build_sub(c.b.build_mul(l, st), pad), k_dil);
            Value* inb = c.b.build_and(c.b.build_sge(lo, c.i32(0)), c.b.build_slt(lo, Lout));
            return emit_if_else_values(
                c, inb, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* xi = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, Cin), cin), L), l);
                    Value* yi = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, Cout), oc), Lout), lo);
                    return {c.b.build_add(cur, c.b.build_mul(load_elem_dt(c, dt, p[0], xi),
                                                            load_elem_dt(c, dt, p[1], yi)))};
                },
                [&]() -> std::vector<Value*> { return {cur}; })[0];
        });
        Value* sum = emit_block_sum_f32(c, s_red, acc);
        c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            store_elem_dt(c, dt, p[2], w, c.b.build_add(load_elem_dt(c, dt, p[2], w), sum));
        });
    });
}

void build_fsq(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    // p: x(0) levels(1) quant(2) packed(3) N(4) D(5)
    Value* D = p[5];
    emit_vec_grid_stride(c, p[4], 1, kAudioBlock, [&](Value* n) {
        Value* base = c.b.build_mul(n, D);
        Value* packed = c.kb.for_range_reduce(
            c.i32(0), D, c.i32(1), c.kb.const_i64(0), [&](Value* it, Value* cur) {
                Value* dd = c.b.build_sub(c.b.build_sub(D, c.i32(1)), it);
                Value* i = c.b.build_add(base, dd);
                Value* L = c.kb.load_i32(ptr_at(c, p[1], dd, 4));
                Value* lm1 = c.b.build_sub(L, c.i32(1));
                Value* flm1 = c.kb.i32_to_f32(lm1);
                Value* h = c.b.build_mul(flm1, c.f32(0.5f));
                Value* v = c.kb.clamp_f32(load_elem_dt(c, dt, p[0], i), c.f32(-1.0f), c.f32(1.0f));
                Value* t = c.b.build_mul(c.b.build_mul(c.b.build_add(v, c.f32(1.0f)), c.f32(0.5f)), flm1);
                Value* fl = c.b.build_floor_f32(t);
                Value* rnd_up = c.b.build_sge(c.b.build_sub(t, fl), c.f32(0.5f));
                Value* idx0 = c.b.build_add(c.b.build_fptosi_i32_f32(fl),
                                            c.b.build_select(rnd_up, c.i32(1), c.i32(0)));
                Value* idx = clamp_i32(c, idx0, c.i32(0), lm1);
                Value* idx_f = c.kb.i32_to_f32(idx);
                Value* zero_mask = c.b.build_select(c.b.build_eq(L, c.i32(0)), c.i32(1), c.i32(0));
                auto pin_f32 = [&](Value* v) {
                    return jit::gpu_mir::bits_to_f32(
                        c, c.b.build_xor(jit::gpu_mir::f32_to_bits(c, v), zero_mask));
                };
                // Keep NIR from cancelling -(a * (1/h)) * h + a or folding - 1.0 into the NR correction.
                Value* r = pin_f32(c.b.build_sdiv(c.f32(1.0f), h));
                Value* q = pin_f32(c.b.build_mul(idx_f, r));
                Value* e = c.b.build_fma_f32(c.b.build_neg(q), h, idx_f);
                Value* qv = pin_f32(c.b.build_fma_f32(e, r, q));
                Value* qo = c.b.build_sub(qv, c.f32(1.0f));
                store_elem_dt(c, dt, p[2], i, qo);
                return c.b.build_add(c.b.build_mul(cur, c.b.build_sext_i64(L)), c.b.build_sext_i64(idx));
            });
        c.kb.store_i32(ptr_at(c, p[3], n, 4), c.b.build_trunc_i32(packed));
    });
}

void build_col2im1d(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool has_bias, bool cols_f32) {
    // p: cols(0) bias(1) Y(2) N(3) Cout(4) L(5) kL(6) st(7) pad(8) dil(9) Cgout(10) Lout(11)
    Value* Cout = p[4];
    Value* L = p[5];
    Value* kL = p[6];
    Value* st = p[7];
    Value* pad = p[8];
    Value* dil = p[9];
    Value* Cgout = p[10];
    Value* Lout = p[11];
    Value* groups = c.b.build_udiv(Cout, Cgout);
    Value* total = c.b.build_mul(c.b.build_mul(p[3], Cout), Lout);

    emit_vec_grid_stride(c, total, 1, kAudioBlock, [&](Value* i) {
        Value* noc = c.b.build_udiv(i, Lout);
        Value* lo = c.b.build_sub(i, c.b.build_mul(noc, Lout));
        Value* n = c.b.build_udiv(noc, Cout);
        Value* oc = c.b.build_sub(noc, c.b.build_mul(n, Cout));
        Value* g = c.b.build_udiv(oc, Cgout);
        Value* ocl = c.b.build_sub(oc, c.b.build_mul(g, Cgout));
        Value* rowb = c.b.build_mul(c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, groups), g), Cgout), ocl),
                                    kL);
        Value* init = has_bias ? load_elem_dt(c, dt, p[1], oc) : c.f32(0.0f);
        Value* acc = c.kb.for_range_reduce(c.i32(0), kL, c.i32(1), init, [&](Value* k, Value* cur) {
            Value* t = c.b.build_sub(c.b.build_add(lo, pad), c.b.build_mul(k, dil));
            return emit_if_else_values(
                c, c.b.build_sge(t, c.i32(0)), {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* l = c.b.build_udiv(t, st);
                    Value* div_ok = c.b.build_eq(c.b.build_sub(t, c.b.build_mul(l, st)), c.i32(0));
                    Value* in_l = c.b.build_and(div_ok, c.b.build_slt(l, L));
                    return emit_if_else_values(
                        c, in_l, {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* ci = c.b.build_add(c.b.build_mul(c.b.build_add(rowb, k), L), l);
                            Value* v = cols_f32 ? c.kb.load_f32(ptr_at(c, p[0], ci, 4))
                                                : load_elem_dt(c, dt, p[0], ci);
                            return {c.b.build_add(cur, v)};
                        },
                        [&]() -> std::vector<Value*> { return {cur}; });
                },
                [&]() -> std::vector<Value*> { return {cur}; })[0];
        });
        store_elem_dt(c, dt, p[2], i, acc);
    });
}

void build_vq(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    // p: x(0) codebook(1) quantized(2) indices(3) N(4) D(5) K(6)
    Value* N = p[4];
    Value* D = p[5];
    Value* K = p[6];
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kAudioBlock));
    Value* s_val = c.kb.shared_alloc_f32(8);
    Value* s_idx = c.kb.shared_alloc_i32(8);
    Value* inv_bi = c.i32(-1);

    c.kb.for_range(c.kb.ctaid_x(), N, c.kb.nctaid_x(), [&](Value* n) {
        Value* x_base = c.b.build_mul(n, D);
        auto best = c.kb.for_range_reduce_n(
            tid, K, block, {c.f32(-3.4028235e38f), inv_bi}, [&](Value* k, const std::vector<Value*>& st) {
                Value* c_base = c.b.build_mul(k, D);
                Value* d2 = c.kb.for_range_reduce(c.i32(0), D, c.i32(1), c.f32(0.0f), [&](Value* j, Value* acc) {
                    Value* df = c.b.build_sub(load_elem_dt(c, dt, p[0], c.b.build_add(x_base, j)),
                                              load_elem_dt(c, dt, p[1], c.b.build_add(c_base, j)));
                    return c.b.build_add(acc, c.b.build_mul(df, df));
                });
                Value* neg_d2 = c.b.build_neg(d2);
                Value* win = c.b.build_or(c.b.build_eq(st[1], inv_bi), c.b.build_sgt(neg_d2, st[0]));
                return std::vector<Value*>{c.b.build_select(win, neg_d2, st[0]), c.b.build_select(win, k, st[1])};
            });
        Value* bi_safe = c.b.build_select(c.b.build_eq(best[1], inv_bi), c.i32(-2), best[1]);
        Value* win = emit_block_argmax_f32(c, s_val, s_idx, best[0], bi_safe);
        Value* kb = c.b.build_select(c.b.build_uge(win, K), c.i32(0), win);
        c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] { c.kb.store_i32(ptr_at(c, p[3], n, 4), kb); });
        Value* kb_base = c.b.build_mul(kb, D);
        c.kb.for_range(tid, D, block, [&](Value* j) {
            store_elem_dt(c, dt, p[2], c.b.build_add(x_base, j),
                          load_elem_dt(c, dt, p[1], c.b.build_add(kb_base, j)));
        });
        c.kb.sync();
    });
}

}  // namespace

brass::Function* build_audio_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint32_t aux0, bool aux1) {
    std::vector<Type> params;
    switch (op) {
        case AU_SNAKE:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_SNAKE_BWD:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_PAD:
        case AU_PAD_BWD:
            params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_RESAMPLE:
        case AU_RESAMPLE_BWD:
            params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_CAUSAL:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_CONVT_BWD_W:
            params = {Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_FSQ:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()};
            break;
        case AU_VQ:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case AU_COL2IM:
            params = {Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        default:
            return nullptr;
    }

    std::string name = "mir_audio_" + std::to_string(op) + "_" + std::to_string(static_cast<int>(dt)) + "_" +
                       std::to_string(aux0) + "_" + std::to_string(aux1 ? 1 : 0);
    Function* fn = mod.create_function(name, Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    switch (op) {
        case AU_SNAKE:
        case AU_SNAKE_BWD:
            build_snake(c, p, dt, aux0 != 0u, op == AU_SNAKE_BWD);
            break;
        case AU_PAD:
        case AU_PAD_BWD:
            build_pad1d(c, p, dt, aux0, op == AU_PAD_BWD);
            break;
        case AU_RESAMPLE:
        case AU_RESAMPLE_BWD:
            build_resample1d(c, p, dt, aux0, op == AU_RESAMPLE_BWD);
            break;
        case AU_CAUSAL:
            build_causal_conv1d(c, p, dt, aux0 != 0u);
            break;
        case AU_CONVT_BWD_W:
            build_convt1d_bwd_w(c, p, dt);
            break;
        case AU_FSQ:
            build_fsq(c, p, dt);
            break;
        case AU_VQ:
            build_vq(c, p, dt);
            break;
        case AU_COL2IM:
            build_col2im1d(c, p, dt, aux0 != 0u, aux1);
            break;
        default:
            break;
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
