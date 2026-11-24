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

}  // namespace brotensor::kernels
