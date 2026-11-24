#include "mir_math.h"

#include <cstddef>

namespace brotensor::kernels {

namespace {

template <std::size_t N>
Value* horner(Ctx& c, Value* x, const float (&coeffs)[N]) {
    Value* acc = c.f32(coeffs[N - 1]);
    for (std::size_t i = N - 1; i > 0; --i) {
        acc = c.b.build_fma_f32(acc, x, c.f32(coeffs[i - 1]));
    }
    return acc;
}

}  // namespace

Value* emit_exp(Ctx& c, Value* x) { return c.b.build_call("expf", brass::Type::f32(), {x}); }
Value* emit_log(Ctx& c, Value* x) { return c.b.build_call("logf", brass::Type::f32(), {x}); }
Value* emit_sin(Ctx& c, Value* x) { return c.b.build_call("sinf", brass::Type::f32(), {x}); }
Value* emit_cos(Ctx& c, Value* x) { return c.b.build_call("cosf", brass::Type::f32(), {x}); }
Value* emit_sqrt(Ctx& c, Value* x) { return c.b.build_call("sqrtf", brass::Type::f32(), {x}); }
Value* emit_rsqrt(Ctx& c, Value* x) { return c.b.build_call("rsqrtf", brass::Type::f32(), {x}); }
Value* emit_round_even(Ctx& c, Value* x) { return c.b.build_round_f32(x); }

Value* emit_sigmoid(Ctx& c, Value* x) {
    Value* nx = c.kb.clamp_f32(c.b.build_neg(x), c.f32(-88.0f), c.f32(88.0f));
    Value* d = c.b.build_add(c.f32(1.0f), emit_exp(c, nx));
    return c.b.build_sdiv(c.f32(1.0f), d);
}

Value* emit_tanh(Ctx& c, Value* x) {
    Value* ax = c.kb.fabs(x);
    Value* x2 = c.b.build_mul(x, x);
    static constexpr float kMac[] = {
        1.0f, -1.0f / 3.0f, 2.0f / 15.0f, -17.0f / 315.0f, 62.0f / 2835.0f, -1382.0f / 155925.0f};
    Value* small = c.b.build_mul(x, horner(c, x2, kMac));

    Value* neg_2ax = c.kb.fmax(c.b.build_mul(c.f32(-2.0f), ax), c.f32(-88.0f));
    Value* t = emit_exp(c, neg_2ax);
    Value* mag = c.b.build_sdiv(c.b.build_sub(c.f32(1.0f), t), c.b.build_add(c.f32(1.0f), t));
    Value* large = c.b.build_select(c.b.build_slt(x, c.f32(0.0f)), c.b.build_neg(mag), mag);
    return c.b.build_select(c.b.build_slt(ax, c.f32(0.25f)), small, large);
}

Value* emit_erf(Ctx& c, Value* x) {
    Value* ax = c.kb.fmin(c.kb.fabs(x), c.f32(20.0f));
    Value* x2 = c.b.build_mul(ax, ax);
    static constexpr float kErfMac[] = {
        1.0f, -1.0f / 3.0f, 0.1f, -1.0f / 42.0f, 1.0f / 216.0f, -1.0f / 1320.0f};
    Value* small = c.b.build_mul(c.b.build_mul(c.f32(1.1283791670955126f), ax), horner(c, x2, kErfMac));

    Value* t = c.b.build_sdiv(c.f32(1.0f), c.b.build_fma_f32(ax, c.f32(0.5f), c.f32(1.0f)));
    static constexpr float kErfcPoly[] = {
        -1.26551223f,  1.00002368f,  0.37409196f,  0.09678418f, -0.18628806f,
         0.27886807f, -1.13520398f,  1.48851587f, -0.82215223f,  0.17087277f};
    Value* p = horner(c, t, kErfcPoly);
    Value* erfc_exp = emit_exp(c, c.b.build_sub(p, x2));
    Value* large = c.b.build_sub(c.f32(1.0f), c.b.build_mul(t, erfc_exp));
    Value* mag = c.b.build_select(c.b.build_slt(ax, c.f32(0.5f)), small, large);
    return c.b.build_select(c.b.build_slt(x, c.f32(0.0f)), c.b.build_neg(mag), mag);
}

Value* emit_gelu_tanh(Ctx& c, Value* x) {
    constexpr float kGeluC = 0.7978845608028654f;
    Value* xc = c.kb.clamp_f32(x, c.f32(-20.0f), c.f32(20.0f));
    Value* xc3 = c.b.build_mul(c.b.build_mul(xc, xc), xc);
    Value* inner = c.b.build_fma_f32(xc3, c.f32(0.044715f), xc);
    Value* u2 = c.b.build_mul(c.f32(2.0f * kGeluC), inner);
    return c.b.build_mul(x, emit_sigmoid(c, u2));
}

Value* emit_gelu_tanh_grad(Ctx& c, Value* x) {
    constexpr float kGeluC = 0.7978845608028654f;
    Value* xc = c.kb.clamp_f32(x, c.f32(-20.0f), c.f32(20.0f));
    Value* x2 = c.b.build_mul(xc, xc);
    Value* inner = c.b.build_fma_f32(c.b.build_mul(x2, xc), c.f32(0.044715f), xc);
    Value* s = emit_sigmoid(c, c.b.build_mul(c.f32(2.0f * kGeluC), inner));
    Value* d_inner = c.b.build_mul(c.f32(kGeluC), c.b.build_fma_f32(x2, c.f32(3.0f * 0.044715f), c.f32(1.0f)));
    Value* s1s = c.b.build_mul(s, c.b.build_sub(c.f32(1.0f), s));
    return c.b.build_fma_f32(c.b.build_mul(c.f32(2.0f), xc), c.b.build_mul(s1s, d_inner), s);
}

Value* emit_gelu_exact(Ctx& c, Value* x) {
    Value* erf_v = emit_erf(c, c.b.build_mul(x, c.f32(0.70710678118654752440f)));
    return c.b.build_mul(c.b.build_mul(c.f32(0.5f), x), c.b.build_add(c.f32(1.0f), erf_v));
}

Value* emit_div_rn(Ctx& c, Value* a, Value* b_val) {
    Value* r = c.b.build_sdiv(c.f32(1.0f), b_val);
    Value* q = c.b.build_mul(a, r);
    return c.b.build_fma_f32(c.b.build_fma_f32(c.b.build_neg(q), b_val, a), r, q);
}

Value* emit_exp_acc(Ctx& c, Value* t) {
    Value* n = c.b.build_round_f32(c.b.build_mul(t, c.f32(1.44269504088896341f)));
    Value* neg_n = c.b.build_neg(n);
    Value* r = c.b.build_fma_f32(neg_n, c.f32(0.693145751953125f), t);
    r = c.b.build_fma_f32(neg_n, c.f32(1.428606765330187045e-06f), r);
    static constexpr float kExpPoly[] = {
        1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f};
    Value* p = horner(c, r, kExpPoly);
    return c.kb.ldexp_f32(p, c.kb.f32_to_i32(n));
}

std::pair<Value*, Value*> emit_sincos_acc(Ctx& c, Value* x) {
    Value* j = c.b.build_round_f32(c.b.build_mul(x, c.f32(0.636619772367581343f)));
    Value* neg_j = c.b.build_neg(j);
    Value* y = c.b.build_fma_f32(neg_j, c.f32(1.57079637050628662f), x);
    y = c.b.build_fma_f32(neg_j, c.f32(-4.37113882867379e-08f), y);
    y = c.b.build_fma_f32(neg_j, c.f32(-1.71512451e-15f), y);
    Value* z = c.b.build_mul(y, y);
    static constexpr float kSinPoly[] = {-1.6666654611e-1f, 8.3321608736e-3f, -1.9515295891e-4f};
    static constexpr float kCosPoly[] = {4.166664568298827e-2f, -1.388731625493765e-3f, 2.443315711809948e-5f};
    Value* sp = c.b.build_fma_f32(c.b.build_mul(y, z), horner(c, z, kSinPoly), y);
    Value* cp = c.b.build_fma_f32(c.b.build_mul(z, z), horner(c, z, kCosPoly),
                                  c.b.build_fma_f32(c.f32(-0.5f), z, c.f32(1.0f)));
    Value* nsp = c.b.build_neg(sp);
    Value* ncp = c.b.build_neg(cp);
    Value* q = c.b.build_and(c.kb.f32_to_i32(j), c.i32(3));
    Value* is0 = c.b.build_eq(q, c.i32(0));
    Value* is1 = c.b.build_eq(q, c.i32(1));
    Value* is2 = c.b.build_eq(q, c.i32(2));
    Value* sin_v = c.b.build_select(is0, sp, c.b.build_select(is1, cp, c.b.build_select(is2, nsp, ncp)));
    Value* cos_v = c.b.build_select(is0, cp, c.b.build_select(is1, nsp, c.b.build_select(is2, ncp, sp)));
    return {sin_v, cos_v};
}

namespace {

Value* emit_block_reduce_32x8(Ctx& c, Value* scratch8, Value* v, float ident, bool is_max) {
    auto comb = [&](Value* a, Value* b_val) -> Value* {
        return is_max ? c.kb.fmax(a, b_val) : c.b.build_add(a, b_val);
    };
    for (std::uint32_t m = 16; m >= 1; m >>= 1) {
        v = comb(v, c.kb.shfl_bfly_f32(v, m));
    }
    Value* tid = c.kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_and(c.b.build_lshr(tid, c.i32(5)), c.i32(7));
    c.kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] {
        c.kb.shared_store_f32_indexed(scratch8, warp, v);
    });
    c.kb.sync();
    Value* loaded = c.kb.shared_load_f32_indexed(scratch8, c.b.build_and(lane, c.i32(7)));
    Value* mine = c.b.build_select(c.b.build_ult(lane, c.i32(8)), loaded, c.f32(ident));
    for (std::uint32_t m = 16; m >= 1; m >>= 1) {
        mine = comb(mine, c.kb.shfl_bfly_f32(mine, m));
    }
    c.kb.sync();
    return mine;
}

}  // namespace

Value* emit_block_sum_f32(Ctx& c, Value* scratch8, Value* v) {
    return emit_block_reduce_32x8(c, scratch8, v, 0.0f, false);
}

Value* emit_block_max_f32(Ctx& c, Value* scratch8, Value* v) {
    return emit_block_reduce_32x8(c, scratch8, v, -3.4028234663852886e38f, true);
}

Value* emit_block_argmax_f32(Ctx& c, Value* s_val8, Value* s_idx8, Value* best_v, Value* best_i) {
    auto better = [&](Value* ov, Value* oi, Value* bv, Value* bi) -> Value* {
        Value* gt = c.b.build_sgt(ov, bv);
        Value* eq_tie = c.b.build_and(c.b.build_eq(ov, bv), c.b.build_ult(oi, bi));
        return c.b.build_or(gt, eq_tie);
    };
    for (std::uint32_t m = 16; m >= 1; m >>= 1) {
        Value* ov = c.kb.shfl_bfly_f32(best_v, m);
        Value* oi = c.kb.shfl_bfly_i32(best_i, m);
        Value* take = better(ov, oi, best_v, best_i);
        best_v = c.b.build_select(take, ov, best_v);
        best_i = c.b.build_select(take, oi, best_i);
    }
    Value* tid = c.kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_and(c.b.build_lshr(tid, c.i32(5)), c.i32(7));
    c.kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] {
        c.kb.shared_store_f32_indexed(s_val8, warp, best_v);
        c.kb.shared_store_i32_indexed(s_idx8, warp, best_i);
    });
    c.kb.sync();
    Value* bv = c.kb.shared_load_f32_indexed(s_val8, c.i32(0));
    Value* bi = c.kb.shared_load_i32_indexed(s_idx8, c.i32(0));
    for (int w = 1; w < 8; ++w) {
        Value* wv = c.kb.shared_load_f32_indexed(s_val8, c.i32(w));
        Value* wi = c.kb.shared_load_i32_indexed(s_idx8, c.i32(w));
        Value* take = better(wv, wi, bv, bi);
        bv = c.b.build_select(take, wv, bv);
        bi = c.b.build_select(take, wi, bi);
    }
    c.kb.sync();
    return bi;
}

}  // namespace brotensor::kernels
