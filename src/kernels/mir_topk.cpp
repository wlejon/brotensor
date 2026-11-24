#include "mir_topk.h"

#include "mir_math.h"

#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* beats(Ctx& c, Value* a, Value* ia, Value* b, Value* ib) {
    return c.b.build_select(c.b.build_ne(a, b), c.b.build_sgt(a, b), c.b.build_ult(ia, ib));
}

std::pair<Value*, Value*> merge2(Ctx& c, Value* a1, Value* a2, Value* b1, Value* b2) {
    Value* b1_wins = c.b.build_sgt(b1, a1);
    return {c.b.build_select(b1_wins, b1, a1),
            c.b.build_select(b1_wins, c.kb.fmax(a1, b2), c.kb.fmax(a2, b1))};
}

}  // namespace

brass::Function* build_topk_mir(brass::Module& mod, std::uint32_t mode, Dtype dt) {
    const std::string name = "mir_topk_" + std::to_string(mode) + "_dt" + std::to_string(static_cast<int>(dt));
    auto* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *x = p[0], *vals = p[1], *idx = p[2], *off = p[3];
    Value *cols = p[5], *k = p[6], *nseg = p[7];
    Value* tid = c.kb.tid_x();

    if (mode == 0u) {
        Value* sv = c.kb.shared_alloc_f32(256);
        Value* r = c.kb.ctaid_y();
        Value* i = c.kb.global_tid_x();
        Value* base = c.b.build_mul(r, cols);
        Value* live = c.b.build_ult(i, cols);
        Value* v = emit_if_else_values(
            c, live, {Type::f32()},
            [&]() -> std::vector<Value*> { return {load_elem_dt(c, dt, x, c.b.build_add(base, i))}; },
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];

        Value* block_sz = c.i32(256);
        Value* rank = c.kb.for_range_reduce(c.i32(0), cols, block_sz, c.i32(0), [&](Value* j0, Value* r_acc) {
            c.kb.sync();
            Value* j = c.b.build_add(j0, tid);
            c.kb.if_then(c.b.build_ult(j, cols), [&] {
                c.kb.shared_store_f32_indexed(sv, tid, load_elem_dt(c, dt, x, c.b.build_add(base, j)));
            });
            c.kb.sync();
            Value* rem = c.b.build_sub(cols, j0);
            Value* n = c.b.build_select(c.b.build_ult(rem, block_sz), rem, block_sz);
            return c.kb.for_range_reduce(c.i32(0), n, c.i32(1), r_acc, [&](Value* t, Value* cur_r) {
                Value* svt = c.kb.shared_load_f32_indexed(sv, t);
                Value* win = beats(c, svt, c.b.build_add(j0, t), v, i);
                return c.b.build_add(cur_r, c.b.build_select(win, c.i32(1), c.i32(0)));
            });
        });

        c.kb.if_then(c.b.build_and(live, c.b.build_ult(rank, k)), [&] {
            Value* out_idx = c.b.build_add(c.b.build_mul(r, k), rank);
            c.kb.store_f32(ptr_at(c, vals, out_idx, 4), v);
            c.kb.store_i32(ptr_at(c, idx, out_idx, 4), i);
        });
    } else {
        Value* scratch8 = c.kb.shared_alloc_f32(8);
        Value* s1 = c.kb.shared_alloc_f32(256);
        Value* s2 = c.kb.shared_alloc_f32(256);
        Value* block_sz = c.i32(256);

        c.kb.for_range(c.kb.ctaid_x(), nseg, c.kb.nctaid_x(), [&](Value* s) {
            Value* raw_a = c.kb.load_i32(ptr_at(c, off, s, 4));
            Value* raw_b = c.kb.load_i32(ptr_at(c, off, c.b.build_add(s, c.i32(1)), 4));
            Value* a = clamp_i32(c, raw_a, c.i32(0), cols);
            Value* b = clamp_i32(c, raw_b, a, cols);
            Value* non_empty = c.b.build_sgt(b, a);
            Value* start_i = c.b.build_add(a, tid);

            Value* mx_loc = c.kb.for_range_reduce(
                start_i, b, block_sz, c.f32(-3.4028234663852886e38f),
                [&](Value* i, Value* mx) { return c.kb.fmax(mx, load_elem_dt(c, dt, x, i)); });
            Value* mx = emit_block_max_f32(c, scratch8, mx_loc);

            Value* sum_loc = c.kb.for_range_reduce(start_i, b, block_sz, c.f32(0.0f), [&](Value* i, Value* sm) {
                return c.b.build_add(sm, emit_exp(c, c.b.build_sub(load_elem_dt(c, dt, x, i), mx)));
            });
            Value* sum = emit_block_sum_f32(c, scratch8, sum_loc);
            Value* safe_sum = c.b.build_select(non_empty, sum, c.f32(1.0f));

            auto acc = c.kb.for_range_reduce_n(
                start_i, b, block_sz, {c.f32(0.0f), c.f32(0.0f), c.f32(0.0f)},
                [&](Value* i, const std::vector<Value*>& cur) -> std::vector<Value*> {
                    Value* pr = c.b.build_sdiv(
                        emit_exp(c, c.b.build_sub(load_elem_dt(c, dt, x, i), mx)), safe_sum);
                    Value* ent = c.b.build_sub(
                        cur[2], c.b.build_mul(pr, emit_log(c, c.kb.fmax(pr, c.f32(1e-9f)))));
                    auto [nt1, nt2] = merge2(c, cur[0], cur[1], pr, c.f32(0.0f));
                    return {nt1, nt2, ent};
                });

            Value* ent_tot = emit_block_sum_f32(c, scratch8, acc[2]);
            c.kb.shared_store_f32_indexed(s1, tid, acc[0]);
            c.kb.shared_store_f32_indexed(s2, tid, acc[1]);
            c.kb.sync();
            for (int w = 128; w > 0; w >>= 1) {
                c.kb.if_then(c.b.build_ult(tid, c.i32(w)), [&] {
                    Value* mate = c.b.build_add(tid, c.i32(w));
                    auto [na1, na2] = merge2(
                        c, c.kb.shared_load_f32_indexed(s1, tid), c.kb.shared_load_f32_indexed(s2, tid),
                        c.kb.shared_load_f32_indexed(s1, mate), c.kb.shared_load_f32_indexed(s2, mate));
                    c.kb.shared_store_f32_indexed(s1, tid, na1);
                    c.kb.shared_store_f32_indexed(s2, tid, na2);
                });
                c.kb.sync();
            }

            Value* o = c.b.build_mul(s, c.i32(4));
            c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                Value* top1 = c.kb.shared_load_f32_indexed(s1, c.i32(0));
                Value* top2 = c.kb.shared_load_f32_indexed(s2, c.i32(0));
                Value* len_f = c.kb.i32_to_f32(c.b.build_sub(b, a));
                Value* kk = c.kb.fmax(c.f32(2.0f), len_f);
                Value* out0 = c.b.build_select(non_empty, top1, c.f32(0.0f));
                Value* out1 = c.b.build_select(non_empty, c.b.build_sub(top1, top2), c.f32(0.0f));
                Value* out2 = c.b.build_select(non_empty, c.b.build_sdiv(ent_tot, emit_log(c, kk)), c.f32(0.0f));
                Value* out3 = c.b.build_select(non_empty, c.b.build_sdiv(kk, c.f32(255.0f)), c.f32(0.0f));
                store_elem_dt(c, dt, vals, o, out0);
                store_elem_dt(c, dt, vals, c.b.build_add(o, c.i32(1)), out1);
                store_elem_dt(c, dt, vals, c.b.build_add(o, c.i32(2)), out2);
                store_elem_dt(c, dt, vals, c.b.build_add(o, c.i32(3)), out3);
            });
            c.kb.sync();
        });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
