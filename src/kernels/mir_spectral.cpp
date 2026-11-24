#include "mir_spectral.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

std::pair<Value*, Value*> emit_twiddle(Ctx& c, Value* m, Value* L) {
    Value* m64 = c.b.build_zext_i64(m);
    Value* L64 = c.b.build_zext_i64(L);
    Value* m4 = c.b.build_shl(m64, c.kb.const_i64(2));
    Value* q = c.b.build_trunc_i32(c.b.build_udiv(m4, L64));
    Value* r0 = c.b.build_trunc_i32(c.b.build_sub(m4, c.b.build_mul(c.b.build_zext_i64(q), L64)));
    Value* swp = c.b.build_ugt(c.b.build_shl(r0, c.i32(1)), L);
    Value* r = c.b.build_select(swp, c.b.build_sub(L, r0), r0);
    Value* x = c.b.build_mul(c.kb.u32_to_f32(r),
                             c.b.build_sdiv(c.f32(1.57079632679489662f), c.kb.u32_to_f32(L)));
    auto [sx, cx] = emit_sincos_acc(c, x);
    Value* cq = c.b.build_select(swp, sx, cx);
    Value* sq = c.b.build_select(swp, cx, sx);
    Value* is0 = c.b.build_eq(q, c.i32(0));
    Value* is1 = c.b.build_eq(q, c.i32(1));
    Value* is2 = c.b.build_eq(q, c.i32(2));
    Value* ncq = c.b.build_neg(cq);
    Value* nsq = c.b.build_neg(sq);
    Value* cos_v = c.b.build_select(is0, cq, c.b.build_select(is1, nsq, c.b.build_select(is2, ncq, sq)));
    Value* sin_v = c.b.build_select(is0, sq, c.b.build_select(is1, cq, c.b.build_select(is2, nsq, ncq)));
    return {cos_v, sin_v};
}

Value* hermite_wk(Ctx& c, Value* k, Value* L) {
    Value* k0 = c.b.build_eq(k, c.i32(0));
    Value* even = c.b.build_eq(c.b.build_and(L, c.i32(1)), c.i32(0));
    Value* nyq = c.b.build_and(even, c.b.build_eq(c.b.build_shl(k, c.i32(1)), L));
    return c.b.build_select(c.b.build_or(k0, nyq), c.f32(1.0f), c.f32(2.0f));
}

Value* emit_atan2_acc(Ctx& c, Value* y, Value* x) {
    Value* ax = c.b.build_fabs_f32(x);
    Value* ay = c.b.build_fabs_f32(y);
    Value* xbits = jit::gpu_mir::f32_to_bits(c, x);
    Value* ybits = jit::gpu_mir::f32_to_bits(c, y);
    Value* x_neg = c.b.build_ne(c.b.build_lshr(xbits, c.i32(31)), c.i32(0));
    Value* y_neg = c.b.build_ne(c.b.build_lshr(ybits, c.i32(31)), c.i32(0));
    Value* both_zero = c.b.build_and(c.b.build_eq(ax, c.f32(0.0f)), c.b.build_eq(ay, c.f32(0.0f)));
    Value* pi = c.f32(3.14159265358979324f);
    Value* half_pi = c.f32(1.57079632679489662f);

    return emit_if_else_values(
        c, both_zero, {Type::f32()},
        [&]() -> std::vector<Value*> {
            Value* r0 = c.b.build_select(x_neg, pi, c.f32(0.0f));
            return {c.b.build_select(y_neg, c.b.build_neg(r0), r0)};
        },
        [&]() -> std::vector<Value*> {
            Value* sw = c.b.build_sgt(ay, ax);
            Value* a0 = c.b.build_select(sw, emit_div_rn(c, ax, ay), emit_div_rn(c, ay, ax));
            Value* oct = c.b.build_sgt(a0, c.f32(0.414213562373095f));
            Value* a = c.b.build_select(
                oct, emit_div_rn(c, c.b.build_sub(a0, c.f32(1.0f)), c.b.build_add(a0, c.f32(1.0f))), a0);
            Value* off = c.b.build_select(oct, c.f32(0.785398163397448310f), c.f32(0.0f));
            Value* z = c.b.build_mul(a, a);
            Value* p = c.f32(8.05374449538e-2f);
            p = c.b.build_sub(c.b.build_mul(p, z), c.f32(1.38776856032e-1f));
            p = c.b.build_add(c.b.build_mul(p, z), c.f32(1.99777106478e-1f));
            p = c.b.build_sub(c.b.build_mul(p, z), c.f32(3.33329491539e-1f));
            Value* r = c.b.build_add(c.b.build_mul(c.b.build_mul(p, z), a), a);
            r = c.b.build_add(r, off);
            r = c.b.build_select(sw, c.b.build_sub(half_pi, r), r);
            r = c.b.build_select(x_neg, c.b.build_sub(pi, r), r);
            return {c.b.build_select(y_neg, c.b.build_neg(r), r)};
        })[0];
}

std::pair<Value*, Value*> cover_range(Ctx& c, Value* p, Value* hop, Value* pad_lo, Value* win, Value* frames) {
    Value* t = c.b.build_sub(p, pad_lo);
    Value* div_t = c.b.build_sdiv(t, hop);
    Value* fr_m1 = c.b.build_sub(frames, c.i32(1));
    Value* min_f1 = c.b.build_select(c.b.build_slt(div_t, fr_m1), div_t, fr_m1);
    Value* f1 = c.b.build_select(c.b.build_slt(t, c.i32(0)), c.i32(-1), min_f1);
    Value* lo = c.b.build_add(c.b.build_sub(t, win), c.i32(1));
    Value* lo_div = c.b.build_sdiv(c.b.build_sub(c.b.build_add(lo, hop), c.i32(1)), hop);
    Value* f0 = c.b.build_select(c.b.build_sle(lo, c.i32(0)), c.i32(0), lo_div);
    return {f0, f1};
}

void build_basis(Ctx& c, const std::vector<Value*>& p, std::uint32_t mode, bool weights, bool has_win) {
    // p: win(0) B(1) rows(2) cols(3) L(4) tofs(5) row0(6) sgn(7) scale(8)
    Value* cols = p[3];
    Value* L = p[4];
    Value* tofs = p[5];
    Value* row0 = p[6];
    Value* sg = c.kb.i32_to_f32(p[7]);
    Value* scale = p[8];
    Value* L64 = c.b.build_zext_i64(L);
    Value* total = c.b.build_mul(p[2], cols);

    emit_vec_grid_stride(c, total, 1, kSpectralBlock, [&](Value* t) {
        Value* o0 = c.b.build_udiv(t, cols);
        Value* i = c.b.build_sub(t, c.b.build_mul(o0, cols));
        Value* o = c.b.build_add(o0, row0);
        Value *k = nullptr, *n = nullptr, *widx = nullptr, *oc = nullptr, *ic = nullptr;
        if (mode == BASIS_R2C) {
            k = c.b.build_lshr(o, c.i32(1));
            oc = c.b.build_and(o, c.i32(1));
            n = c.b.build_add(i, tofs);
            widx = i;
            ic = c.i32(0);
        } else if (mode == BASIS_C2R) {
            k = c.b.build_lshr(i, c.i32(1));
            ic = c.b.build_and(i, c.i32(1));
            n = c.b.build_add(o, tofs);
            widx = o;
            oc = c.i32(0);
        } else {
            k = c.b.build_lshr(o, c.i32(1));
            oc = c.b.build_and(o, c.i32(1));
            n = c.b.build_lshr(i, c.i32(1));
            ic = c.b.build_and(i, c.i32(1));
            widx = c.i32(0);
        }
        Value* km = c.b.build_umod(k, L);
        Value* nm = c.b.build_umod(n, L);
        Value* m = c.b.build_trunc_i32(c.b.build_umod(c.b.build_mul(c.b.build_zext_i64(km), c.b.build_zext_i64(nm)), L64));
        auto [cv, sv] = emit_twiddle(c, m, L);
        Value* s_sv = c.b.build_mul(sg, sv);
        Value* oc0 = c.b.build_eq(oc, c.i32(0));
        Value* ic0 = c.b.build_eq(ic, c.i32(0));
        Value* v_oc0 = c.b.build_select(ic0, cv, c.b.build_neg(s_sv));
        Value* v_oc1 = c.b.build_select(ic0, s_sv, cv);
        Value* v = c.b.build_mul(c.b.build_select(oc0, v_oc0, v_oc1), scale);
        if (weights) v = c.b.build_mul(v, hermite_wk(c, k, L));
        if (has_win) v = c.b.build_mul(v, c.kb.load_f32(ptr_at(c, p[0], widx, 4)));
        c.kb.store_f32(ptr_at(c, p[1], t, 4), v);
    });
}

void build_ola_or_istft_adj(Ctx& c, const std::vector<Value*>& p, bool is_ola) {
    // p: a(0) win(1) d(2) N(3) sig_len(4) frames(5) hop(6) pad_lo(7) win_len(8) shift(9) padded_len(10)
    Value* sig_len = p[4];
    Value* frames = p[5];
    Value* hop = p[6];
    Value* pad_lo = p[7];
    Value* win_len = p[8];
    Value* shift = p[9];
    Value* per = is_ola ? sig_len : p[10];
    Value* total = c.b.build_mul(p[3], per);

    emit_vec_grid_stride(c, total, 1, kSpectralBlock, [&](Value* t) {
        Value* b = c.b.build_udiv(t, per);
        Value* rem = c.b.build_sub(t, c.b.build_mul(b, per));
        Value* pos = is_ola ? c.b.build_add(rem, shift) : rem;
        Value* n = c.b.build_sub(pos, shift);
        Value* oob = is_ola ? c.i32(0) : c.b.build_or(c.b.build_slt(n, c.i32(0)), c.b.build_sge(n, sig_len));

        Value* out = emit_if_else_values(
            c, oob, {Type::f32()},
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; },
            [&]() -> std::vector<Value*> {
                auto [f0, f1] = cover_range(c, pos, hop, pad_lo, win_len, frames);
                auto res = c.kb.for_range_reduce_n(
                    f0, c.b.build_add(f1, c.i32(1)), c.i32(1), {c.f32(0.0f), c.f32(0.0f)},
                    [&](Value* f, const std::vector<Value*>& cur) {
                        Value* j = c.b.build_sub(c.b.build_sub(pos, c.b.build_mul(f, hop)), pad_lo);
                        Value* w = c.kb.load_f32(ptr_at(c, p[1], j, 4));
                        Value* env = c.b.build_add(cur[0], c.b.build_mul(w, w));
                        Value* acc = cur[1];
                        if (is_ola) {
                            Value* a_idx = c.b.build_add(
                                c.b.build_mul(c.b.build_add(c.b.build_mul(b, frames), f), win_len), j);
                            acc = c.b.build_add(acc, c.kb.load_f32(ptr_at(c, p[0], a_idx, 4)));
                        }
                        return std::vector<Value*>{env, acc};
                    });
                Value* env = res[0];
                Value* num = is_ola ? res[1] : c.kb.load_f32(ptr_at(c, p[0], c.b.build_add(c.b.build_mul(b, sig_len), n), 4));
                return {c.b.build_select(c.b.build_sgt(env, c.f32(1e-10f)), emit_div_rn(c, num, env), c.f32(0.0f))};
            })[0];
        c.kb.store_f32(ptr_at(c, p[2], t, 4), out);
    });
}

void build_stft_adj(Ctx& c, const std::vector<Value*>& p, bool center) {
    // p: G(0) dSignal(1) N(2) sig_len(3) frames(4) hop(5) pad_lo(6) win_len(7) half_nfft(8) padded_len(9)
    Value* L = p[3];
    Value* frames = p[4];
    Value* hop = p[5];
    Value* pad_lo = p[6];
    Value* win_len = p[7];
    Value* half = p[8];
    Value* plen = p[9];
    Value* total = c.b.build_mul(p[2], L);

    emit_vec_grid_stride(c, total, 1, kSpectralBlock, [&](Value* t) {
        Value* b = c.b.build_udiv(t, L);
        Value* s = c.b.build_sub(t, c.b.build_mul(b, L));
        auto accum_at = [&](Value* pos, Value* acc_in) {
            auto [f0, f1] = cover_range(c, pos, hop, pad_lo, win_len, frames);
            return c.kb.for_range_reduce(
                f0, c.b.build_add(f1, c.i32(1)), c.i32(1), acc_in, [&](Value* f, Value* cur) {
                    Value* j = c.b.build_sub(c.b.build_sub(pos, c.b.build_mul(f, hop)), pad_lo);
                    Value* idx = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(b, frames), f), win_len), j);
                    return c.b.build_add(cur, c.kb.load_f32(ptr_at(c, p[0], idx, 4)));
                });
        };
        if (!center) {
            c.kb.store_f32(ptr_at(c, p[1], t, 4), accum_at(s, c.f32(0.0f)));
            return;
        }
        Value* per = c.b.build_shl(c.b.build_sub(L, c.i32(1)), c.i32(1));
        Value* l_one = c.b.build_eq(L, c.i32(1));
        std::array<Value*, 8> p_vals{};
        std::array<Value*, 8> v_flags{};
        Value* acc = c.f32(0.0f);
        for (int idx = 0; idx < 8; ++idx) {
            const int k = (idx >> 1) - 1;
            const int sg = idx & 1;
            Value* q = c.b.build_add(sg == 0 ? s : c.b.build_neg(s), c.b.build_mul(c.i32(k), per));
            Value* pc = c.b.build_add(q, half);
            p_vals[static_cast<std::size_t>(idx)] = pc;
            Value* valid = c.b.build_and(c.b.build_sge(pc, c.i32(0)), c.b.build_slt(pc, plen));
            if (k != 0 || sg != 0) valid = c.b.build_and(valid, c.b.build_eq(l_one, c.i32(0)));
            for (int prev = 0; prev < idx; ++prev) {
                Value* same = c.b.build_and(v_flags[static_cast<std::size_t>(prev)],
                                            c.b.build_eq(p_vals[static_cast<std::size_t>(prev)], pc));
                valid = c.b.build_and(valid, c.b.build_eq(same, c.i32(0)));
            }
            v_flags[static_cast<std::size_t>(idx)] = valid;
            acc = emit_if_else_values(
                c, valid, {Type::f32()},
                [&]() -> std::vector<Value*> { return {accum_at(pc, acc)}; },
                [&]() -> std::vector<Value*> { return {acc}; })[0];
        }
        c.kb.store_f32(ptr_at(c, p[1], t, 4), acc);
    });
}

void build_complex(Ctx& c, const std::vector<Value*>& p, std::uint32_t op) {
    // a(0) b(1) c_dy(2) d(3) e(4) n(5)
    emit_vec_grid_stride(c, p[5], 1, kSpectralBlock, [&](Value* i) {
        Value* i2 = c.b.build_shl(i, c.i32(1));
        Value* i2p1 = c.b.build_add(i2, c.i32(1));
        if (op == SP_CX_MUL) {
            Value* ar = c.kb.load_f32(ptr_at(c, p[0], i2, 4));
            Value* ai = c.kb.load_f32(ptr_at(c, p[0], i2p1, 4));
            Value* br = c.kb.load_f32(ptr_at(c, p[1], i2, 4));
            Value* bi = c.kb.load_f32(ptr_at(c, p[1], i2p1, 4));
            c.kb.store_f32(ptr_at(c, p[3], i2, 4), c.b.build_sub(c.b.build_mul(ar, br), c.b.build_mul(ai, bi)));
            c.kb.store_f32(ptr_at(c, p[3], i2p1, 4), c.b.build_add(c.b.build_mul(ar, bi), c.b.build_mul(ai, br)));
        } else if (op == SP_CX_MUL_BWD) {
            Value* ar = c.kb.load_f32(ptr_at(c, p[0], i2, 4));
            Value* ai = c.kb.load_f32(ptr_at(c, p[0], i2p1, 4));
            Value* br = c.kb.load_f32(ptr_at(c, p[1], i2, 4));
            Value* bi = c.kb.load_f32(ptr_at(c, p[1], i2p1, 4));
            Value* gr = c.kb.load_f32(ptr_at(c, p[2], i2, 4));
            Value* gi = c.kb.load_f32(ptr_at(c, p[2], i2p1, 4));
            Value* dar_p = ptr_at(c, p[3], i2, 4);
            Value* dai_p = ptr_at(c, p[3], i2p1, 4);
            Value* dbr_p = ptr_at(c, p[4], i2, 4);
            Value* dbi_p = ptr_at(c, p[4], i2p1, 4);
            c.kb.store_f32(dar_p, c.b.build_add(c.kb.load_f32(dar_p),
                                                c.b.build_add(c.b.build_mul(gr, br), c.b.build_mul(gi, bi))));
            c.kb.store_f32(dai_p, c.b.build_add(c.kb.load_f32(dai_p),
                                                c.b.build_sub(c.b.build_mul(gi, br), c.b.build_mul(gr, bi))));
            c.kb.store_f32(dbr_p, c.b.build_add(c.kb.load_f32(dbr_p),
                                                c.b.build_add(c.b.build_mul(gr, ar), c.b.build_mul(gi, ai))));
            c.kb.store_f32(dbi_p, c.b.build_add(c.kb.load_f32(dbi_p),
                                                c.b.build_sub(c.b.build_mul(gi, ar), c.b.build_mul(gr, ai))));
        } else if (op == SP_CX_ABS || op == SP_CX_ABS_BWD) {
            Value* re = c.kb.load_f32(ptr_at(c, p[0], i2, 4));
            Value* im = c.kb.load_f32(ptr_at(c, p[0], i2p1, 4));
            Value* mag = c.b.build_sqrt_f32(c.b.build_add(c.b.build_mul(re, re), c.b.build_mul(im, im)));
            if (op == SP_CX_ABS) {
                c.kb.store_f32(ptr_at(c, p[3], i, 4), mag);
            } else {
                Value* pos = c.b.build_sgt(mag, c.f32(0.0f));
                Value* inv = emit_if_else_values(
                    c, pos, {Type::f32()},
                    [&]() -> std::vector<Value*> { return {emit_div_rn(c, c.kb.load_f32(ptr_at(c, p[1], i, 4)), mag)}; },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                c.kb.store_f32(ptr_at(c, p[3], i2, 4), c.b.build_select(pos, c.b.build_mul(re, inv), c.f32(0.0f)));
                c.kb.store_f32(ptr_at(c, p[3], i2p1, 4), c.b.build_select(pos, c.b.build_mul(im, inv), c.f32(0.0f)));
            }
        } else if (op == SP_CX_ANGLE) {
            Value* re = c.kb.load_f32(ptr_at(c, p[0], i2, 4));
            Value* im = c.kb.load_f32(ptr_at(c, p[0], i2p1, 4));
            c.kb.store_f32(ptr_at(c, p[3], i, 4), emit_atan2_acc(c, im, re));
        } else {
            Value* m = c.kb.load_f32(ptr_at(c, p[0], i, 4));
            auto [sv, cv] = emit_sincos_acc(c, c.kb.load_f32(ptr_at(c, p[1], i, 4)));
            c.kb.store_f32(ptr_at(c, p[3], i2, 4), c.b.build_mul(m, cv));
            c.kb.store_f32(ptr_at(c, p[3], i2p1, 4), c.b.build_mul(m, sv));
        }
    });
}

}  // namespace

brass::Function* build_spectral_mir(brass::Module& mod, std::uint32_t op, std::uint32_t aux0, bool aux1, bool aux2) {
    std::vector<Type> params;
    switch (op) {
        case SP_BASIS:
            params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32(), Type::i32(), Type::f32()};
            break;
        case SP_OLA:
        case SP_ISTFT_ADJ:
            params = {Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case SP_STFT_ADJ:
            params = {Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case SP_CX_MUL:
        case SP_CX_MUL_BWD:
        case SP_CX_ABS:
        case SP_CX_ABS_BWD:
        case SP_CX_ANGLE:
        case SP_CX_POLAR:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()};
            break;
        default:
            return nullptr;
    }

    std::string name = "mir_spectral_" + std::to_string(op) + "_" + std::to_string(aux0) + "_" +
                       std::to_string(aux1 ? 1 : 0) + "_" + std::to_string(aux2 ? 1 : 0);
    Function* fn = mod.create_function(name, Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    switch (op) {
        case SP_BASIS: build_basis(c, p, aux0, aux1, aux2); break;
        case SP_OLA:
        case SP_ISTFT_ADJ: build_ola_or_istft_adj(c, p, op == SP_OLA); break;
        case SP_STFT_ADJ: build_stft_adj(c, p, aux0 != 0u); break;
        default: build_complex(c, p, op); break;
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
