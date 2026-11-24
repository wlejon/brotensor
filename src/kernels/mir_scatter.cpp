#include "mir_scatter.h"
#include "mir_common.h"

#include "../vulkan/shaders/op_codes.h"

#include <cstdint>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::int64_t kNoneKey = -1LL;  // 0xffffffffffffffff

Value* pack_u32_pair(Ctx& c, Value* hi32, Value* lo32) {
    Value* h64 = c.b.build_shl(c.b.build_zext_i64(hi32), c.kb.const_i64(32));
    return c.b.build_or(h64, c.b.build_zext_i64(lo32));
}

Value* pair_gt(Ctx& c, Value* ha, Value* la, Value* hb, Value* lb) {
    Value* hi_gt = c.b.build_ugt(ha, hb);
    Value* hi_eq = c.b.build_eq(ha, hb);
    return c.b.build_or(hi_gt, c.b.build_and(hi_eq, c.b.build_ugt(la, lb)));
}

void build_sa_keys(Ctx& c, const std::vector<Value*>& p) {
    Value* m = p[5];
    Value* r = p[6];
    Value* p_cnt = p[8];
    Value* none = c.kb.const_i64(kNoneKey);
    Value* inv_hi = c.i32(static_cast<int>(0xffffffffu));
    emit_vec_grid_stride(c, p_cnt, 1, kScatterBlock, [&](Value* i) {
        auto kv = emit_if_else_values(
            c, c.b.build_ult(i, m), {Type::i64()},
            [&]() -> std::vector<Value*> {
                Value* row = c.kb.load_i32(ptr_at(c, p[0], i, 4));
                Value* valid = c.b.build_and(c.b.build_sge(row, c.i32(0)), c.b.build_ult(row, r));
                Value* hi = c.b.build_select(valid, row, inv_hi);
                return {pack_u32_pair(c, hi, i)};
            },
            [&]() -> std::vector<Value*> { return {none}; });
        c.kb.store_i64(ptr_at(c, p[1], i, 8), kv[0]);
    });
}

void build_sa_local(Ctx& c, const std::vector<Value*>& p) {
    Value* p_cnt = p[8];
    Value* tid = c.kb.tid_x();
    Value* sk_hi = c.kb.shared_alloc_i32(512);
    Value* sk_lo = c.kb.shared_alloc_i32(512);
    Value* none = c.kb.const_i64(kNoneKey);

    for (int s = 0; s < 2; ++s) {
        Value* idx = (s == 0) ? tid : c.b.build_add(tid, c.i32(256));
        auto kv = emit_if_else_values(
            c, c.b.build_ult(idx, p_cnt), {Type::i64()},
            [&]() -> std::vector<Value*> { return {c.kb.load_i64(ptr_at(c, p[1], idx, 8))}; },
            [&]() -> std::vector<Value*> { return {none}; });
        Value* hi = c.b.build_trunc_i32(c.b.build_lshr(kv[0], c.kb.const_i64(32)));
        Value* lo = c.b.build_trunc_i32(kv[0]);
        c.kb.shared_store_i32_indexed(sk_hi, idx, hi);
        c.kb.shared_store_i32_indexed(sk_lo, idx, lo);
    }
    c.kb.sync();

    for (int k = 2; k <= 512; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int s = 0; s < 2; ++s) {
                Value* i = (s == 0) ? tid : c.b.build_add(tid, c.i32(256));
                Value* l = c.b.build_xor(i, c.i32(j));
                c.kb.if_then(c.b.build_ugt(l, i), [&] {
                    Value* ha = c.kb.shared_load_i32_indexed(sk_hi, i);
                    Value* la = c.kb.shared_load_i32_indexed(sk_lo, i);
                    Value* hb = c.kb.shared_load_i32_indexed(sk_hi, l);
                    Value* lb = c.kb.shared_load_i32_indexed(sk_lo, l);
                    Value* up = c.b.build_eq(c.b.build_and(i, c.i32(k)), c.i32(0));
                    Value* gt = pair_gt(c, ha, la, hb, lb);
                    Value* do_swap = c.b.build_eq(gt, up);
                    c.kb.shared_store_i32_indexed(sk_hi, i, c.b.build_select(do_swap, hb, ha));
                    c.kb.shared_store_i32_indexed(sk_lo, i, c.b.build_select(do_swap, lb, la));
                    c.kb.shared_store_i32_indexed(sk_hi, l, c.b.build_select(do_swap, ha, hb));
                    c.kb.shared_store_i32_indexed(sk_lo, l, c.b.build_select(do_swap, la, lb));
                });
            }
            c.kb.sync();
        }
    }

    for (int s = 0; s < 2; ++s) {
        Value* idx = (s == 0) ? tid : c.b.build_add(tid, c.i32(256));
        c.kb.if_then(c.b.build_ult(idx, p_cnt), [&] {
            Value* hi = c.kb.shared_load_i32_indexed(sk_hi, idx);
            Value* lo = c.kb.shared_load_i32_indexed(sk_lo, idx);
            c.kb.store_i64(ptr_at(c, p[1], idx, 8), pack_u32_pair(c, hi, lo));
        });
    }
}

void build_sa_step(Ctx& c, const std::vector<Value*>& p) {
    Value* p_cnt = p[8];
    Value* k_v = p[9];
    Value* j_v = p[10];
    emit_vec_grid_stride(c, p_cnt, 1, kScatterBlock, [&](Value* i) {
        Value* l = c.b.build_xor(i, j_v);
        Value* a = c.kb.load_i64(ptr_at(c, p[1], i, 8));
        Value* b_val = c.kb.load_i64(ptr_at(c, p[1], l, 8));
        Value* up = c.b.build_eq(c.b.build_and(i, k_v), c.i32(0));
        Value* take_min = c.b.build_eq(c.b.build_ult(i, l), up);
        Value* lo = c.b.build_select(c.b.build_ult(a, b_val), a, b_val);
        Value* hi = c.b.build_select(c.b.build_ugt(a, b_val), a, b_val);
        c.kb.store_i64(ptr_at(c, p[2], i, 8), c.b.build_select(take_min, lo, hi));
    });
}

void build_sa_sum(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    const int esz = elem_bytes(dt);
    Value* m = p[5];
    Value* r = p[6];
    Value* cols = p[7];
    Value* col = c.b.build_add(c.b.build_mul(c.kb.ctaid_x(), c.i32(static_cast<int>(kScatterBlock))),
                               c.kb.tid_x());
    Value* col_ok = c.b.build_ult(col, cols);

    c.kb.for_range(c.kb.ctaid_y(), m, c.kb.nctaid_y(), [&](Value* pos) {
        Value* key = c.kb.load_i64(ptr_at(c, p[1], pos, 8));
        Value* row = c.b.build_trunc_i32(c.b.build_lshr(key, c.kb.const_i64(32)));
        Value* in_r = c.b.build_ult(row, r);
        auto first_res = emit_if_else_values(
            c, c.b.build_ugt(pos, c.i32(0)), {Type::i32()},
            [&]() -> std::vector<Value*> {
                Value* pk = c.kb.load_i64(ptr_at(c, p[1], c.b.build_sub(pos, c.i32(1)), 8));
                Value* pr = c.b.build_trunc_i32(c.b.build_lshr(pk, c.kb.const_i64(32)));
                return {c.b.build_select(c.b.build_ne(pr, row), c.i32(1), c.i32(0))};
            },
            [&]() -> std::vector<Value*> { return {c.i32(1)}; });
        Value* active = c.b.build_and(c.b.build_and(in_r, col_ok), c.b.build_ne(first_res[0], c.i32(0)));
        c.kb.if_then(active, [&] {
            Value* o = c.b.build_add(c.b.build_mul(row, cols), col);
            Value* dst_ptr = ptr_at(c, p[4], o, esz);
            Value* acc0 = load_lanes(c, dt, 1, dst_ptr)[0];
            auto sum_st = c.kb.for_range_reduce_n(
                pos, m, c.i32(1), {acc0, c.i32(1)}, [&](Value* q, const std::vector<Value*>& cur) {
                    return emit_if_else_values(
                        c, c.b.build_ne(cur[1], c.i32(0)), {Type::f32(), Type::i32()},
                        [&]() -> std::vector<Value*> {
                            Value* kq = c.kb.load_i64(ptr_at(c, p[1], q, 8));
                            Value* rq = c.b.build_trunc_i32(c.b.build_lshr(kq, c.kb.const_i64(32)));
                            return emit_if_else_values(
                                c, c.b.build_eq(rq, row), {Type::f32(), Type::i32()},
                                [&]() -> std::vector<Value*> {
                                    Value* mq = c.b.build_trunc_i32(kq);
                                    Value* si = c.b.build_add(c.b.build_mul(mq, cols), col);
                                    Value* sv = load_lanes(c, dt, 1, ptr_at(c, p[3], si, esz))[0];
                                    return {c.b.build_add(cur[0], sv), c.i32(1)};
                                },
                                [&]() -> std::vector<Value*> { return {cur[0], c.i32(0)}; });
                        },
                        [&]() -> std::vector<Value*> { return cur; });
                });
            store_lanes(c, dt, 1, dst_ptr, {sum_st[0]});
        });
    });
}

}  // namespace

Function* build_scatter_add_mir(Module& mod, std::uint32_t mode, Dtype dt) {
    Function* fn = mod.create_function(
        "mir_scatter_add", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    switch (mode) {
        case SA_KEYS:  build_sa_keys(c, p); break;
        case SA_LOCAL: build_sa_local(c, p); break;
        case SA_STEP:  build_sa_step(c, p); break;
        default:       build_sa_sum(c, p, dt); break;
    }
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
