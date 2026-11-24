#include "mir_misc.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* count_valid_rows(Ctx& c, Value* mask, Value* K, bool has_mask) {
    if (!has_mask) return K;
    return c.kb.for_range_reduce(c.i32(0), K, c.i32(1), c.i32(0), [&](Value* k, Value* cur) {
        Value* ok = c.b.build_sge(c.kb.load_f32(ptr_at(c, mask, k, 4)), c.f32(0.5f));
        return c.b.build_add(cur, c.b.build_select(ok, c.i32(1), c.i32(0)));
    });
}

Value* row_valid(Ctx& c, Value* mask, Value* k, bool has_mask) {
    if (!has_mask) return c.i32(1);
    return c.b.build_sge(c.kb.load_f32(ptr_at(c, mask, k, 4)), c.f32(0.5f));
}

Value* act_eval(Ctx& c, Value* t, Value* is_lrelu, Value* alpha) {
    Value* neg_or_zero = c.b.build_eq(c.b.build_sgt(t, c.f32(0.0f)), c.i32(0));
    return c.b.build_select(c.b.build_and(is_lrelu, neg_or_zero), c.b.build_mul(alpha, t), t);
}

Value* bias_act_grad_eval(Ctx& c, Value* dy, Value* t, Value* is_lrelu, Value* alpha, Value* gain, Value* clamp_v) {
    Value* neg_or_zero = c.b.build_eq(c.b.build_sgt(t, c.f32(0.0f)), c.i32(0));
    Value* gr = c.b.build_select(c.b.build_and(is_lrelu, neg_or_zero), alpha, c.f32(1.0f));
    Value* dt = c.b.build_mul(c.b.build_mul(dy, gain), gr);
    Value* has_clamp = c.b.build_sge(clamp_v, c.f32(0.0f));
    Value* yp = c.b.build_mul(gain, act_eval(c, t, is_lrelu, alpha));
    Value* sat = c.b.build_or(c.b.build_slt(yp, c.b.build_neg(clamp_v)), c.b.build_sgt(yp, clamp_v));
    return c.b.build_select(c.b.build_and(has_clamp, sat), c.f32(0.0f), dt);
}

void build_mean_pool(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool has_mask, bool bwd) {
    Value* K = p[3];
    Value* D = p[4];
    Value* nv = count_valid_rows(c, p[1], K, has_mask);
    Value* inv = c.b.build_select(c.b.build_ugt(nv, c.i32(0)),
                                  c.b.build_sdiv(c.f32(1.0f), c.kb.u32_to_f32(nv)), c.f32(0.0f));
    if (!bwd) {
        emit_vec_grid_stride(c, D, 1, kMiscBlock, [&](Value* j) {
            Value* acc = c.kb.for_range_reduce(c.i32(0), K, c.i32(1), c.f32(0.0f), [&](Value* k, Value* cur) {
                Value* idx = c.b.build_add(c.b.build_mul(k, D), j);
                if (!has_mask) return c.b.build_add(cur, load_elem_dt(c, dt, p[0], idx));
                return emit_if_else_values(
                    c, row_valid(c, p[1], k, true), {Type::f32()},
                    [&]() -> std::vector<Value*> { return {c.b.build_add(cur, load_elem_dt(c, dt, p[0], idx))}; },
                    [&]() -> std::vector<Value*> { return {cur}; })[0];
            });
            store_elem_dt(c, dt, p[2], j, c.b.build_mul(acc, inv));
        });
    } else {
        Value* total = c.b.build_mul(K, D);
        Value* has_nv = c.b.build_ugt(nv, c.i32(0));
        emit_vec_grid_stride(c, total, 1, kMiscBlock, [&](Value* i) {
            Value* k = c.b.build_udiv(i, D);
            Value* j = c.b.build_sub(i, c.b.build_mul(k, D));
            Value* ok = has_mask ? c.b.build_and(has_nv, row_valid(c, p[1], k, true)) : has_nv;
            Value* val = emit_if_else_values(
                c, ok, {Type::f32()},
                [&]() -> std::vector<Value*> { return {c.b.build_mul(load_elem_dt(c, dt, p[0], j), inv)}; },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            store_elem_dt(c, dt, p[2], i, val);
        });
    }
}

void build_count_above(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    Value* R = p[2];
    Value* C = p[3];
    Value* t_lo = p[4];
    Value* t_hi = p[5];
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kMiscBlock));
    Value* s_red = c.kb.shared_alloc_f32(8);
    c.kb.for_range(c.kb.ctaid_x(), R, c.kb.nctaid_x(), [&](Value* r) {
        Value* base = c.b.build_mul(r, C);
        auto cnts = c.kb.for_range_reduce_n(
            tid, C, block, {c.i32(0), c.i32(0)}, [&](Value* j, const std::vector<Value*>& cur) {
                Value* v = load_elem_dt(c, dt, p[0], c.b.build_add(base, j));
                Value* nlo = c.b.build_add(cur[0], c.b.build_select(c.b.build_sgt(v, t_lo), c.i32(1), c.i32(0)));
                Value* nhi = c.b.build_add(cur[1], c.b.build_select(c.b.build_sgt(v, t_hi), c.i32(1), c.i32(0)));
                return std::vector<Value*>{nlo, nhi};
            });
        Value* slo = emit_block_sum_f32(c, s_red, c.kb.u32_to_f32(cnts[0]));
        Value* shi = emit_block_sum_f32(c, s_red, c.kb.u32_to_f32(cnts[1]));
        c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            Value* o = c.b.build_shl(r, c.i32(1));
            c.kb.store_i32(ptr_at(c, p[1], o, 4), c.b.build_fptosi_i32_f32(slo));
            c.kb.store_i32(ptr_at(c, p[1], c.b.build_add(o, c.i32(1)), 4), c.b.build_fptosi_i32_f32(shi));
        });
    });
}

void build_moments(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    Value* Lq = p[3];
    Value* Lk = p[4];
    Value* w_lat = p[5];
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kMiscBlock));
    Value* s_red = c.kb.shared_alloc_f32(8);
    c.kb.for_range(c.kb.ctaid_x(), Lk, c.kb.nctaid_x(), [&](Value* k) {
        auto acc = c.kb.for_range_reduce_n(
            tid, Lq, block, {c.f32(0.0f), c.f32(0.0f), c.f32(0.0f)}, [&](Value* q, const std::vector<Value*>& cur) {
                Value* v = load_elem_dt(c, dt, p[0], c.b.build_add(c.b.build_mul(q, Lk), k));
                Value* row = c.b.build_udiv(q, w_lat);
                Value* col = c.b.build_sub(q, c.b.build_mul(row, w_lat));
                return std::vector<Value*>{
                    c.b.build_add(cur[0], v),
                    c.b.build_add(cur[1], c.b.build_mul(c.kb.u32_to_f32(row), v)),
                    c.b.build_add(cur[2], c.b.build_mul(c.kb.u32_to_f32(col), v)),
                };
            });
        Value* am = emit_block_sum_f32(c, s_red, acc[0]);
        Value* ay = emit_block_sum_f32(c, s_red, acc[1]);
        Value* ax = emit_block_sum_f32(c, s_red, acc[2]);
        c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            c.kb.store_f32(ptr_at(c, p[1], k, 4), am);
            Value* ok = c.b.build_sgt(am, c.f32(1e-8f));
            Value* inv = c.b.build_select(ok, c.b.build_sdiv(c.f32(1.0f), am), c.f32(0.0f));
            Value* cy = c.b.build_select(ok, c.b.build_mul(ay, inv), c.f32(0.0f));
            Value* cx = c.b.build_select(ok, c.b.build_mul(ax, inv), c.f32(0.0f));
            Value* o = c.b.build_shl(k, c.i32(1));
            c.kb.store_f32(ptr_at(c, p[2], o, 4), cy);
            c.kb.store_f32(ptr_at(c, p[2], c.b.build_add(o, c.i32(1)), 4), cx);
        });
    });
}

void build_bias_act(Ctx& c, const std::vector<Value*>& p, std::uint32_t op, Dtype dt, bool has_bias) {
    if (op == MI_BIAS_ACT_DB) {
        // p: dY(0), X(1), bias(2), dB(3), N(4), C(5), HW(6), act(7), alpha(8), gain(9), clamp(10)
        Value* N = p[4];
        Value* C = p[5];
        Value* HW = p[6];
        Value* is_lrelu = c.b.build_eq(p[7], c.i32(1));
        Value* per = c.b.build_mul(N, HW);
        Value* tid = c.kb.tid_x();
        Value* block = c.i32(static_cast<int>(kMiscBlock));
        Value* s_red = c.kb.shared_alloc_f32(8);
        c.kb.for_range(c.kb.ctaid_x(), C, c.kb.nctaid_x(), [&](Value* ch) {
            Value* bias = has_bias ? load_elem_dt(c, dt, p[2], ch) : c.f32(0.0f);
            Value* acc = c.kb.for_range_reduce(tid, per, block, c.f32(0.0f), [&](Value* j, Value* cur) {
                Value* n = c.b.build_udiv(j, HW);
                Value* rem = c.b.build_sub(j, c.b.build_mul(n, HW));
                Value* idx = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, C), ch), HW), rem);
                Value* t = c.b.build_add(load_elem_dt(c, dt, p[1], idx), bias);
                Value* g = bias_act_grad_eval(c, load_elem_dt(c, dt, p[0], idx), t, is_lrelu, p[8], p[9], p[10]);
                return c.b.build_add(cur, g);
            });
            Value* sum = emit_block_sum_f32(c, s_red, acc);
            c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                Value* old_db = load_elem_dt(c, dt, p[3], ch);
                store_elem_dt(c, dt, p[3], ch, c.b.build_add(old_db, sum));
            });
        });
        return;
    }
    const bool fwd = (op == MI_BIAS_ACT);
    // fwd: a X(0), b bias(1), c Y(2), total(3), C(4), HW(5), act(6), alpha(7), gain(8), clamp(9)
    // bwd: a dY(0), b X(1), c bias(2), d dX(3), total(4), C(5), HW(6), act(7), alpha(8), gain(9), clamp(10)
    const std::size_t off = fwd ? 0 : 1;
    Value* xb = fwd ? p[0] : p[1];
    Value* bb = fwd ? p[1] : p[2];
    Value* out_ptr = p[2 + off];
    Value* total = p[3 + off];
    Value* C = p[4 + off];
    Value* HW = p[5 + off];
    Value* is_lrelu = c.b.build_eq(p[6 + off], c.i32(1));
    Value* alpha = p[7 + off];
    Value* gain = p[8 + off];
    Value* clamp_v = p[9 + off];
    Value* has_clamp = c.b.build_sge(clamp_v, c.f32(0.0f));
    emit_vec_grid_stride(c, total, 1, kMiscBlock, [&](Value* i) {
        Value* ch = c.b.build_umod(c.b.build_udiv(i, HW), C);
        Value* bval = has_bias ? load_elem_dt(c, dt, bb, ch) : c.f32(0.0f);
        Value* t = c.b.build_add(load_elem_dt(c, dt, xb, i), bval);
        if (fwd) {
            Value* y = c.b.build_mul(act_eval(c, t, is_lrelu, alpha), gain);
            Value* yc = c.kb.fmin(c.kb.fmax(y, c.b.build_neg(clamp_v)), clamp_v);
            store_elem_dt(c, dt, out_ptr, i, c.b.build_select(has_clamp, yc, y));
        } else {
            Value* dy = load_elem_dt(c, dt, p[0], i);
            store_elem_dt(c, dt, out_ptr, i, bias_act_grad_eval(c, dy, t, is_lrelu, alpha, gain, clamp_v));
        }
    });
}

}  // namespace

brass::Function* build_misc_mir(brass::Module& mod, std::uint32_t op, Dtype dt, bool has_aux) {
    std::vector<Type> params;
    switch (op) {
        case MI_MEAN_POOL:
        case MI_MEAN_POOL_BWD:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()};
            break;
        case MI_SLOT_MASK:
            params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case MI_CAUSAL_ROW:
            params = {Type::ptr(), Type::i32(), Type::i32()};
            break;
        case MI_THRESHOLD_U8:
            params = {Type::ptr(), Type::ptr(), Type::i32(), Type::f32()};
            break;
        case MI_COUNT_ABOVE:
            params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::f32(), Type::f32()};
            break;
        case MI_XAVIER:
            params = {Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::f32()};
            break;
        case MI_SGD:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::f32(), Type::f32()};
            break;
        case MI_ADAM:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(),
                      Type::f32(), Type::f32(), Type::f32(), Type::f32(), Type::f32(), Type::f32()};
            break;
        case MI_MSE_SAMPLE:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()};
            break;
        case MI_MSE_BWD:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::f32()};
            break;
        case MI_MSE_SUM:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()};
            break;
        case MI_MOMENTS:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case MI_BIAS_ACT:
            params = {Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::f32(), Type::f32(), Type::f32()};
            break;
        case MI_BIAS_ACT_BWD:
        case MI_BIAS_ACT_DB:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::f32(), Type::f32(), Type::f32()};
            break;
        default:
            return nullptr;
    }

    std::string name = "mir_misc_" + std::to_string(op) + "_" + std::to_string(static_cast<int>(dt)) + "_" +
                       std::to_string(has_aux ? 1 : 0);
    Function* fn = mod.create_function(name, Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    switch (op) {
        case MI_MEAN_POOL:
        case MI_MEAN_POOL_BWD:
            build_mean_pool(c, p, dt, has_aux, op == MI_MEAN_POOL_BWD);
            break;
        case MI_SLOT_MASK:
            emit_vec_grid_stride(c, p[2], 1, kMiscBlock, [&](Value* k) {
                Value* idx = c.b.build_add(p[3], c.b.build_mul(k, p[4]));
                Value* ok = c.b.build_sgt(kb.load_f32(ptr_at(c, p[0], idx, 4)), c.f32(0.5f));
                kb.store_f32(ptr_at(c, p[1], k, 4), c.b.build_select(ok, c.f32(1.0f), c.f32(0.0f)));
            });
            break;
        case MI_CAUSAL_ROW:
            emit_vec_grid_stride(c, p[1], 1, kMiscBlock, [&](Value* k) {
                Value* ok = c.b.build_sle(k, p[2]);
                kb.store_f32(ptr_at(c, p[0], k, 4), c.b.build_select(ok, c.f32(1.0f), c.f32(0.0f)));
            });
            break;
        case MI_THRESHOLD_U8:
            emit_vec_grid_stride(c, p[2], 1, kMiscBlock, [&](Value* i) {
                Value* ok = c.b.build_sgt(load_elem_dt(c, dt, p[0], i), p[3]);
                kb.store_u8(ptr_at(c, p[1], i, 1), c.b.build_select(ok, c.i32(1), c.i32(0)));
            });
            break;
        case MI_COUNT_ABOVE:
            build_count_above(c, p, dt);
            break;
        case MI_XAVIER: {
            Value* base = c.b.build_or(c.b.build_zext_i64(p[2]),
                                       c.b.build_shl(c.b.build_zext_i64(p[3]), kb.const_i64(32)));
            Value* mul_k = kb.const_i64(static_cast<std::int64_t>(0x9E3779B97F4A7C15ULL));
            emit_vec_grid_stride(c, p[1], 1, kMiscBlock, [&](Value* i) {
                Value* z = c.b.build_add(base, c.b.build_mul(c.b.build_zext_i64(c.b.build_add(i, c.i32(1))), mul_k));
                Value* u = emit_splitmix64_mix_u01(c, z);
                Value* val = c.b.build_mul(c.b.build_sub(c.b.build_mul(u, c.f32(2.0f)), c.f32(1.0f)), p[4]);
                kb.store_f32(ptr_at(c, p[0], i, 4), val);
            });
            break;
        }
        case MI_SGD:
            emit_vec_grid_stride(c, p[3], 1, kMiscBlock, [&](Value* i) {
                Value* pp = ptr_at(c, p[0], i, 4);
                Value* vp = ptr_at(c, p[2], i, 4);
                Value* nv = c.b.build_add(c.b.build_mul(p[5], kb.load_f32(vp)), kb.load_f32(ptr_at(c, p[1], i, 4)));
                kb.store_f32(vp, nv);
                kb.store_f32(pp, c.b.build_sub(kb.load_f32(pp), c.b.build_mul(p[4], nv)));
            });
            break;
        case MI_ADAM:
            emit_vec_grid_stride(c, p[4], 1, kMiscBlock, [&](Value* i) {
                Value* pp = ptr_at(c, p[0], i, 4);
                Value* mp = ptr_at(c, p[2], i, 4);
                Value* vp = ptr_at(c, p[3], i, 4);
                Value* g = kb.load_f32(ptr_at(c, p[1], i, 4));
                Value* b1 = p[8];
                Value* b2 = p[9];
                Value* omb1 = c.b.build_sub(c.f32(1.0f), b1);
                Value* omb2 = c.b.build_sub(c.f32(1.0f), b2);
                Value* m = c.b.build_add(c.b.build_mul(b1, kb.load_f32(mp)), c.b.build_mul(omb1, g));
                Value* v = c.b.build_add(c.b.build_mul(b2, kb.load_f32(vp)),
                                         c.b.build_mul(c.b.build_mul(omb2, g), g));
                kb.store_f32(mp, m);
                kb.store_f32(vp, v);
                Value* num = c.b.build_mul(p[7], c.b.build_mul(m, p[5]));
                Value* den = c.b.build_add(c.b.build_sqrt_f32(c.b.build_mul(v, p[6])), p[10]);
                kb.store_f32(pp, c.b.build_sub(kb.load_f32(pp), c.b.build_sdiv(num, den)));
            });
            break;
        case MI_MSE_SAMPLE:
            emit_vec_grid_stride(c, p[4], 1, kMiscBlock, [&](Value* i) {
                Value* dd = c.b.build_sub(kb.load_f32(ptr_at(c, p[0], i, 4)), kb.load_f32(ptr_at(c, p[1], i, 4)));
                kb.store_f32(ptr_at(c, p[2], i, 4), dd);
                kb.store_f32(ptr_at(c, p[3], i, 4), c.b.build_mul(c.f32(0.5f), c.b.build_mul(dd, dd)));
            });
            break;
        case MI_MSE_BWD:
            emit_vec_grid_stride(c, p[3], 1, kMiscBlock, [&](Value* i) {
                Value* dd = c.b.build_sub(kb.load_f32(ptr_at(c, p[0], i, 4)), kb.load_f32(ptr_at(c, p[1], i, 4)));
                kb.store_f32(ptr_at(c, p[2], i, 4), c.b.build_mul(p[4], dd));
            });
            break;
        case MI_MSE_SUM: {
            Value* tid = kb.tid_x();
            Value* s_red = kb.shared_alloc_f32(8);
            Value* acc = kb.for_range_reduce(tid, p[3], c.i32(static_cast<int>(kMiscBlock)), c.f32(0.0f),
                                             [&](Value* i, Value* cur) {
                                                 Value* dd = c.b.build_sub(kb.load_f32(ptr_at(c, p[0], i, 4)),
                                                                           kb.load_f32(ptr_at(c, p[1], i, 4)));
                                                 return c.b.build_add(cur, c.b.build_mul(dd, dd));
                                             });
            Value* sum = emit_block_sum_f32(c, s_red, acc);
            kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] { kb.store_f32(p[2], sum); });
            break;
        }
        case MI_MOMENTS:
            build_moments(c, p, dt);
            break;
        case MI_BIAS_ACT:
        case MI_BIAS_ACT_BWD:
        case MI_BIAS_ACT_DB:
            build_bias_act(c, p, op, dt, has_aux);
            break;
        default:
            break;
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
