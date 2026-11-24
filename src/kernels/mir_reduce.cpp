#include "mir_reduce.h"
#include "mir_common.h"
#include "mir_math.h"

#include <vector>

namespace brotensor::kernels {

Function* build_sum_rows_mir(Module& mod, Dtype dt, int lanes) {
    Function* fn = mod.create_function(
        "mir_sum_rows", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz = elem_bytes(dt);
    Value* s_val = kb.shared_alloc_f32(8);
    Value* tid = kb.tid_x();
    Value* n_vecs = vec_count(c, p[3], lanes);
    Value* block = c.i32(static_cast<int>(kReduceBlock));

    kb.for_range(kb.ctaid_x(), p[2], kb.nctaid_x(), [&](Value* row) {
        Value* base = c.b.build_mul(row, p[3]);
        Value* acc = kb.for_range_reduce(tid, n_vecs, block, c.f32(0.0f), [&](Value* k, Value* cur) {
            Value* col = col_elem_idx(c, k, lanes);
            auto xs = load_lanes(c, dt, lanes, ptr_at(c, p[0], c.b.build_add(base, col), esz));
            for (int j = 0; j < lanes; ++j) {
                cur = c.b.build_add(cur, xs[static_cast<std::size_t>(j)]);
            }
            return cur;
        });
        Value* total = emit_block_sum_f32(c, s_val, acc);
        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            store_lanes(c, dt, 1, ptr_at(c, p[1], row, esz), {total});
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_argmax_rows_mir(Module& mod, Dtype dt, bool out_i32, int lanes) {
    Function* fn = mod.create_function(
        "mir_argmax_rows", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz = elem_bytes(dt);
    Value* s_val = kb.shared_alloc_f32(8);
    Value* s_idx = kb.shared_alloc_i32(8);
    Value* tid = kb.tid_x();
    Value* n_vecs = vec_count(c, p[3], lanes);
    Value* block = c.i32(static_cast<int>(kReduceBlock));

    kb.for_range(kb.ctaid_x(), p[2], kb.nctaid_x(), [&](Value* row) {
        Value* base = c.b.build_mul(row, p[3]);
        const std::vector<Value*> inits = {c.f32(-3.4028234663852886e38f), c.i32(0)};
        auto res = kb.for_range_reduce_n(tid, n_vecs, block, inits, [&](Value* k, const std::vector<Value*>& cur) {
            Value* bv = cur[0];
            Value* bi = cur[1];
            Value* col0 = col_elem_idx(c, k, lanes);
            auto xs = load_lanes(c, dt, lanes, ptr_at(c, p[0], c.b.build_add(base, col0), esz));
            for (int j = 0; j < lanes; ++j) {
                Value* vj = xs[static_cast<std::size_t>(j)];
                Value* col = (j == 0) ? col0 : c.b.build_add(col0, c.i32(j));
                Value* better = c.b.build_sgt(vj, bv);
                bv = c.b.build_select(better, vj, bv);
                bi = c.b.build_select(better, col, bi);
            }
            return std::vector<Value*>{bv, bi};
        });
        Value* win = emit_block_argmax_f32(c, s_val, s_idx, res[0], res[1]);
        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            Value* out_ptr = ptr_at(c, p[1], row, 4);
            if (out_i32) {
                c.b.build_store(Type::i32(), out_ptr, 0, win);
            } else {
                c.b.build_store(Type::f32(), out_ptr, 0, kb.i32_to_f32(win));
            }
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_sum_cols_mir(Module& mod, Dtype dt, int lanes) {
    Function* fn = mod.create_function(
        "mir_sum_cols", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz = elem_bytes(dt);

    emit_vec_grid_stride(c, p[3], lanes, kReduceBlock, [&](Value* k) {
        Value* col = col_elem_idx(c, k, lanes);
        std::vector<Value*> zeros(static_cast<std::size_t>(lanes), c.f32(0.0f));
        auto sums = kb.for_range_reduce_n(c.i32(0), p[2], c.i32(1), zeros, [&](Value* r, const std::vector<Value*>& cur) {
            Value* idx = c.b.build_add(c.b.build_mul(r, p[3]), col);
            auto xs = load_lanes(c, dt, lanes, ptr_at(c, p[0], idx, esz));
            std::vector<Value*> next(static_cast<std::size_t>(lanes));
            for (int j = 0; j < lanes; ++j) {
                const auto js = static_cast<std::size_t>(j);
                next[js] = c.b.build_add(cur[js], xs[js]);
            }
            return next;
        });
        store_lanes(c, dt, lanes, ptr_at(c, p[1], col, esz), sums);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
