#include "mir_linear_bwd.h"
#include "mir_common.h"

#include <stdexcept>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* load_one_f32(Ctx& c, Dtype dt, Value* base, Value* idx) {
    return load_elem_dt(c, dt, base, idx);
}

void store_one_f32(Ctx& c, Dtype dt, Value* base, Value* idx, Value* v) {
    store_elem_dt(c, dt, base, idx, v);
}

}  // namespace

Function* build_linear_bwd_mir(Module& mod, std::uint32_t op, Dtype dt) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("build_linear_bwd_mir: unsupported dtype");
    }

    if (op == 0u) {
        // p: W(0), dY(1), dX(2), out_dim(3), in_dim(4)
        Function* fn = mod.create_function(
            "mir_linear_bwd_dx", Type::void_type(),
            {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
        brass::codegen::KernelBuilder kb(mod, fn);
        Ctx c(kb);
        auto p = entry_params(c, fn);
        Value* dytile = kb.shared_alloc_f32(128u);
        Value* tid = kb.tid_x();
        Value* col = kb.global_tid_x();
        Value* out_dim = p[3];
        Value* in_dim = p[4];
        Value* col_ok = c.b.build_slt(col, in_dim);
        const int esz = elem_bytes(dt);
        const int shift = log2_exact(esz);

        Value* acc = kb.for_range_reduce(
            c.i32(0), out_dim, c.i32(128), c.f32(0.0f),
            [&](Value* t0, Value* cur) -> Value* {
                Value* rem = c.b.build_sub(out_dim, t0);
                Value* t_len = smin_i32(c, c.i32(128), rem);
                kb.if_then(c.b.build_slt(tid, t_len), [&] {
                    Value* v = load_one_f32(c, dt, p[1], c.b.build_add(t0, tid));
                    kb.shared_store_f32_indexed(dytile, tid, v);
                });
                kb.sync();
                Value* nxt = emit_if_else_values(
                    c, col_ok, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* w_ptr = ptr_at(
                            c, p[0],
                            c.b.build_add(c.b.build_mul(c.b.build_zext_i64(t0),
                                                        c.b.build_zext_i64(in_dim)),
                                          c.b.build_zext_i64(col)),
                            esz);
                        Value* in64_bytes = c.b.build_shl(c.b.build_zext_i64(in_dim), kb.const_i64(shift));
                        std::vector<Value*> r = kb.for_range_reduce_n(
                            c.i32(0), t_len, c.i32(1), {cur, w_ptr},
                            [&](Value* k, const std::vector<Value*>& st) -> std::vector<Value*> {
                                Value* wv = load_lanes(c, dt, 1, st[1])[0];
                                Value* dyv = kb.shared_load_f32_indexed(dytile, k);
                                Value* sum = c.b.build_fma(wv, dyv, st[0]);
                                return {sum, c.b.build_add(st[1], in64_bytes)};
                            });
                        return {r[0]};
                    },
                    [&]() -> std::vector<Value*> { return {cur}; })[0];
                kb.sync();
                return nxt;
            });

        kb.if_then(col_ok, [&] { store_one_f32(c, dt, p[2], col, acc); });
        c.b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        return fn;
    }

    if (op == 1u) {
        // p: dY(0), x(1), dW(2), out_dim(3), in_dim(4)
        Function* fn = mod.create_function(
            "mir_linear_bwd_dw", Type::void_type(),
            {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
        brass::codegen::KernelBuilder kb(mod, fn);
        Ctx c(kb);
        auto p = entry_params(c, fn);
        Value* idx = kb.global_tid_x();
        Value* in_dim = p[4];
        Value* total = c.b.build_mul(p[3], in_dim);
        kb.if_then(c.b.build_ult(idx, total), [&] {
            Value* i = c.b.build_udiv(idx, in_dim);
            Value* j = c.b.build_umod(idx, in_dim);
            Value* prev = load_one_f32(c, dt, p[2], idx);
            Value* dy = load_one_f32(c, dt, p[0], i);
            Value* xv = load_one_f32(c, dt, p[1], j);
            store_one_f32(c, dt, p[2], idx, c.b.build_fma(dy, xv, prev));
        });
        c.b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        return fn;
    }

    if (op == 2u) {
        // p: dY(0), dB(1), out_dim(2)
        Function* fn = mod.create_function(
            "mir_linear_bwd_db", Type::void_type(),
            {Type::ptr(), Type::ptr(), Type::i32()});
        brass::codegen::KernelBuilder kb(mod, fn);
        Ctx c(kb);
        auto p = entry_params(c, fn);
        Value* i = kb.global_tid_x();
        kb.if_then(c.b.build_ult(i, p[2]), [&] {
            Value* dy = load_one_f32(c, dt, p[0], i);
            Value* db = load_one_f32(c, dt, p[1], i);
            store_one_f32(c, dt, p[1], i, c.b.build_add(db, dy));
        });
        c.b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        return fn;
    }

    throw std::runtime_error("build_linear_bwd_mir: unknown op");
}

}  // namespace brotensor::kernels
