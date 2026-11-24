#include "mir_delta.h"

#include "mir_math.h"

#include <algorithm>
#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* group_sum(Ctx& c, Value* v, std::uint32_t lpr) {
    for (std::uint32_t off = lpr >> 1; off > 0; off >>= 1) {
        v = c.b.build_add(v, c.kb.shfl_bfly_f32(v, off));
    }
    return v;
}

Value* softplus_stable(Ctx& c, Value* x) {
    Value* e = emit_exp(c, c.b.build_neg(c.kb.fabs(x)));
    Value* small = c.b.build_mul(e, c.b.build_sub(c.f32(1.0f), c.b.build_mul(c.f32(0.5f), e)));
    Value* large = emit_log(c, c.b.build_add(c.f32(1.0f), e));
    Value* tail = c.b.build_select(c.b.build_slt(e, c.f32(1e-4f)), small, large);
    return c.b.build_add(c.kb.fmax(x, c.f32(0.0f)), tail);
}

Value* sigmoid_stable(Ctx& c, Value* x) {
    Value* pos = c.b.build_sdiv(c.f32(1.0f), c.b.build_add(c.f32(1.0f), emit_exp(c, c.b.build_neg(x))));
    Value* ep = emit_exp(c, x);
    Value* neg = c.b.build_sdiv(ep, c.b.build_add(c.f32(1.0f), ep));
    return c.b.build_select(c.b.build_sge(x, c.f32(0.0f)), pos, neg);
}

}  // namespace

brass::Function* build_delta_rule_mir(brass::Module& mod, std::uint32_t lpr, std::uint32_t ne) {
    const std::uint32_t count = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(ne, 16u));
    auto* fn = mod.create_function(
        "mir_delta_" + std::to_string(lpr) + "_" + std::to_string(count), Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *q = p[0], *k = p[1], *v = p[2], *a = p[3], *b = p[4], *loga = p[5], *s = p[6], *o = p[7];
    Value *L = p[8], *H = p[9], *dk = p[10], *dv = p[11], *ldq = p[12], *ldv = p[13];

    Value* lpr_v = c.i32(static_cast<int>(lpr));
    Value* lane = c.b.build_and(c.kb.tid_x(), c.i32(static_cast<int>(lpr - 1u)));
    Value* row = c.b.build_udiv(c.kb.global_tid_x(), lpr_v);
    Value* total_rows = c.b.build_mul(H, dv);
    Value* active = c.b.build_ult(row, total_rows);

    Value* safe_row = c.b.build_select(active, row, c.i32(0));
    Value* h = c.b.build_udiv(safe_row, dv);
    Value* vr = c.b.build_umod(safe_row, dv);
    Value* sbase = c.b.build_mul(safe_row, dk);

    std::vector<Value*> S_init(count);
    for (std::uint32_t e = 0; e < count; ++e) {
        Value* j = c.b.build_add(lane, c.i32(static_cast<int>(e * lpr)));
        Value* ok = c.b.build_and(active, c.b.build_ult(j, dk));
        S_init[e] = emit_if_else_values(
            c, ok, {Type::f32()},
            [&]() -> std::vector<Value*> { return {c.kb.load_f32(ptr_at(c, s, c.b.build_add(sbase, j), 4))}; },
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
    }

    Value* expA = emit_exp(c, c.kb.load_f32(ptr_at(c, loga, h, 4)));
    Value* h_dk = c.b.build_mul(h, dk);
    Value* h_dv = c.b.build_mul(h, dv);
    Value* is_lane0 = c.b.build_and(active, c.b.build_eq(lane, c.i32(0)));

    auto S_fin = c.kb.for_range_reduce_n(
        c.i32(0), L, c.i32(1), S_init, [&](Value* t, const std::vector<Value*>& S_cur) -> std::vector<Value*> {
            Value* qb = c.b.build_add(c.b.build_mul(t, ldq), h_dk);
            Value* th_idx = c.b.build_add(c.b.build_mul(t, H), h);
            Value* v_idx = c.b.build_add(c.b.build_add(c.b.build_mul(t, ldv), h_dv), vr);

            std::vector<Value*> kv(count), qv(count);
            Value* dot = c.f32(0.0f);
            for (std::uint32_t e = 0; e < count; ++e) {
                Value* j = c.b.build_add(lane, c.i32(static_cast<int>(e * lpr)));
                Value* ok = c.b.build_and(active, c.b.build_ult(j, dk));
                auto pair = emit_if_else_values(
                    c, ok, {Type::f32(), Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* idx = c.b.build_add(qb, j);
                        return {c.kb.load_f32(ptr_at(c, k, idx, 4)), c.kb.load_f32(ptr_at(c, q, idx, 4))};
                    },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f), c.f32(0.0f)}; });
                kv[e] = pair[0];
                qv[e] = pair[1];
                dot = c.b.build_add(dot, c.b.build_mul(S_cur[e], kv[e]));
            }

            Value* ar = c.kb.load_f32(ptr_at(c, a, th_idx, 4));
            Value* br = c.kb.load_f32(ptr_at(c, b, th_idx, 4));
            Value* alpha = emit_exp(c, c.b.build_neg(c.b.build_mul(softplus_stable(c, ar), expA)));
            Value* beta = sigmoid_stable(c, br);

            Value* dot_alpha = c.b.build_mul(group_sum(c, dot, lpr), alpha);
            Value* v_t = emit_if_else_values(
                c, active, {Type::f32()},
                [&]() -> std::vector<Value*> { return {c.kb.load_f32(ptr_at(c, v, v_idx, 4))}; },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            Value* sc = c.b.build_mul(beta, c.b.build_sub(v_t, dot_alpha));

            std::vector<Value*> S_next(count);
            Value* o_acc = c.f32(0.0f);
            for (std::uint32_t e = 0; e < count; ++e) {
                S_next[e] = c.b.build_add(c.b.build_mul(alpha, S_cur[e]), c.b.build_mul(sc, kv[e]));
                o_acc = c.b.build_add(o_acc, c.b.build_mul(S_next[e], qv[e]));
            }
            Value* o_sum = group_sum(c, o_acc, lpr);
            c.kb.if_then(is_lane0, [&] { c.kb.store_f32(ptr_at(c, o, v_idx, 4), o_sum); });
            return S_next;
        });

    for (std::uint32_t e = 0; e < count; ++e) {
        Value* j = c.b.build_add(lane, c.i32(static_cast<int>(e * lpr)));
        Value* ok = c.b.build_and(active, c.b.build_ult(j, dk));
        c.kb.if_then(ok, [&] { c.kb.store_f32(ptr_at(c, s, c.b.build_add(sbase, j), 4), S_fin[e]); });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
