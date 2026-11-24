#include "mir_attn_small.h"

#include "mir_common.h"
#include "mir_math.h"

#include <string>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kAuxRowGate = 0;
constexpr std::uint32_t kAuxHeadMean = 1;

Value* valid_mask(Ctx& c, Value* mask_ptr, Value* idx, bool mask_ge) {
    Value* v = c.kb.load_f32(ptr_at(c, mask_ptr, idx, 4));
    return mask_ge ? c.b.build_sge(v, c.f32(0.5f)) : c.b.build_slt(c.f32(0.5f), v);
}

Value* eval_softmax_score(Ctx& c, std::uint32_t bias_mode, Value* s_ptr, Value* bias_ptr, Value* mask_ptr,
                          Value* so, Value* bo, Value* h, Value* q, Value* k, Value* bhs, Value* scale) {
    Value* sv = c.kb.load_f32(ptr_at(c, s_ptr, c.b.build_add(so, k), 4));
    Value* x = c.b.build_mul(sv, scale);
    if (bias_mode == 1u) {
        return c.b.build_add(x, c.kb.load_f32(ptr_at(c, bias_ptr, c.b.build_add(bo, k), 4)));
    }
    if (bias_mode == 2u) {
        Value* gw = c.b.build_and(bhs, c.i32(0xffff));
        Value* gh = c.b.build_lshr(bhs, c.i32(16));
        Value* rq = c.b.build_add(c.b.build_mul(h, c.b.build_mul(gh, gw)), q);
        Value* qh = c.b.build_udiv(q, gw);
        Value* qw = c.b.build_umod(q, gw);
        Value* kh = c.b.build_udiv(k, gw);
        Value* kw = c.b.build_umod(k, gw);
        Value* dh_idx = c.b.build_add(c.b.build_sub(qh, kh), c.b.build_sub(gh, c.i32(1)));
        Value* dw_idx = c.b.build_add(c.b.build_sub(qw, kw), c.b.build_sub(gw, c.i32(1)));
        Value* h_stride = c.b.build_sub(c.b.build_shl(gh, c.i32(1)), c.i32(1));
        Value* w_stride = c.b.build_sub(c.b.build_shl(gw, c.i32(1)), c.i32(1));
        Value* bh = c.kb.load_f32(ptr_at(c, bias_ptr, c.b.build_add(c.b.build_mul(rq, h_stride), dh_idx), 4));
        Value* bw = c.kb.load_f32(ptr_at(c, mask_ptr, c.b.build_add(c.b.build_mul(rq, w_stride), dw_idx), 4));
        return c.b.build_add(x, c.b.build_add(bh, bw));
    }
    return x;
}

}  // namespace

brass::Function* build_attn_aux_mir(brass::Module& mod, std::uint32_t mode, Dtype out_dt, bool has_mask) {
    const std::string name = std::string("mir_attn_aux_") +
                             (mode == kAuxRowGate ? "gate_" : "mean_") +
                             (out_dt == Dtype::FP32 ? "f32" : out_dt == Dtype::FP16 ? "f16" : "bf16") +
                             (has_mask ? "_m" : "");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0]=x, p[1]=y, p[2]=mask, p[3]=rows, p[4]=n
    Value* gid = kb.global_tid_x();
    Value* stride = grid_stride_ew(c);

    if (mode == kAuxRowGate) {
        Value* total = c.b.build_mul(p[3], p[4]);
        kb.for_range(gid, total, stride, [&](Value* i) {
            Value* xv = kb.load_f32(ptr_at(c, p[0], i, 4));
            Value* out = xv;
            if (has_mask) {
                Value* row = c.b.build_udiv(i, p[4]);
                Value* mv = kb.load_f32(ptr_at(c, p[2], row, 4));
                out = c.b.build_select(c.b.build_sge(mv, c.f32(0.5f)), xv, c.f32(0.0f));
            }
            store_elem_dt(c, out_dt, p[1], i, out);
        });
    } else if (mode == kAuxHeadMean) {
        Value* inv = emit_div_rn(c, c.f32(1.0f), kb.u32_to_f32(p[3]));
        kb.for_range(gid, p[4], stride, [&](Value* i) {
            Value* sum = kb.for_range_reduce(
                c.i32(0), p[3], c.i32(1), c.f32(0.0f),
                [&](Value* h, Value* acc) -> Value* {
                    Value* v = kb.load_f32(ptr_at(c, p[0], c.b.build_add(c.b.build_mul(h, p[4]), i), 4));
                    return c.b.build_add(acc, v);
                });
            store_elem_dt(c, out_dt, p[1], i, c.b.build_mul(sum, inv));
        });
    } else {
        throw std::runtime_error("brotensor: mir_attn_aux: unknown mode");
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_attn_bwd_mir(brass::Module& mod, bool has_kmask, bool has_qmask) {
    const std::string name = std::string("mir_attn_bwd") + (has_kmask ? "_km" : "") + (has_qmask ? "_qm" : "");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                                Type::i32(), Type::i32(), Type::i32(), Type::f32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0]=p, p[1]=dp, p[2]=kmask, p[3]=qmask, p[4]=rows, p[5]=lq, p[6]=lk, p[7]=scale
    Value* s8 = kb.shared_alloc_f32(8);
    Value* tid = kb.tid_x();
    Value* wg = c.i32(static_cast<int>(jit::gpu_mir::kEwBlock));

    kb.for_range(kb.ctaid_x(), p[4], kb.nctaid_x(), [&](Value* row) {
        Value* base = c.b.build_mul(row, p[6]);
        auto reduce_row = [&]() -> Value* {
            return kb.for_range_reduce(
                tid, p[6], wg, c.f32(0.0f),
                [&](Value* j, Value* cur) -> Value* {
                    Value* idx = c.b.build_add(base, j);
                    return c.b.build_fma_f32(kb.load_f32(ptr_at(c, p[1], idx, 4)),
                                             kb.load_f32(ptr_at(c, p[0], idx, 4)), cur);
                });
        };
        auto write_row = [&](Value* dot) {
            kb.for_range(tid, p[6], wg, [&](Value* j) {
                Value* idx = c.b.build_add(base, j);
                Value* pv = kb.load_f32(ptr_at(c, p[0], idx, 4));
                Value* dpv = kb.load_f32(ptr_at(c, p[1], idx, 4));
                Value* ds = c.b.build_mul(c.b.build_mul(pv, c.b.build_sub(dpv, dot)), p[7]);
                if (has_kmask) {
                    Value* masked = c.b.build_slt(kb.load_f32(ptr_at(c, p[2], j, 4)), c.f32(0.5f));
                    ds = c.b.build_select(masked, c.f32(0.0f), ds);
                }
                kb.store_f32(ptr_at(c, p[1], idx, 4), ds);
            });
        };

        if (!has_qmask) {
            Value* dot = emit_block_sum_f32(c, s8, reduce_row());
            write_row(dot);
            return;
        }
        Value* gated = c.b.build_slt(kb.load_f32(ptr_at(c, p[3], c.b.build_umod(row, p[5]), 4)), c.f32(0.5f));
        Value* acc = emit_if_else_values(
            c, gated, {Type::f32()},
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; },
            [&]() -> std::vector<Value*> { return {reduce_row()}; })[0];
        Value* dot = emit_block_sum_f32(c, s8, acc);
        emit_if_else_values(
            c, gated, {},
            [&]() -> std::vector<Value*> {
                kb.for_range(tid, p[6], wg, [&](Value* j) {
                    kb.store_f32(ptr_at(c, p[1], c.b.build_add(base, j), 4), c.f32(0.0f));
                });
                return {};
            },
            [&]() -> std::vector<Value*> {
                write_row(dot);
                return {};
            });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_fa_combine_mir(brass::Module& mod, Dtype out_dt) {
    const std::string name = std::string("mir_fa_combine_") +
                             (out_dt == Dtype::FP32 ? "f32" : out_dt == Dtype::FP16 ? "f16" : "bf16");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(),
                                Type::i32(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0]=part, p[1]=o, p[2]=entries, p[3]=nsplit, p[4]=hd, p[5]=hq, p[6]=ldo
    Value* w = c.b.build_add(p[4], c.i32(2));
    Value* tid = kb.tid_x();
    Value* wg = c.i32(static_cast<int>(jit::gpu_mir::kEwBlock));

    kb.for_range(kb.ctaid_x(), p[2], kb.nctaid_x(), [&](Value* e) {
        Value* base = c.b.build_mul(c.b.build_mul(e, p[3]), w);
        Value* M = kb.for_range_reduce(
            c.i32(0), p[3], c.i32(1), c.f32(-1e30f),
            [&](Value* s, Value* cur) -> Value* {
                Value* ms = kb.load_f32(ptr_at(c, p[0], c.b.build_add(c.b.build_add(base, c.b.build_mul(s, w)), p[4]), 4));
                return c.b.build_fmax_f32(cur, ms);
            });
        Value* L = kb.for_range_reduce(
            c.i32(0), p[3], c.i32(1), c.f32(0.0f),
            [&](Value* s, Value* cur) -> Value* {
                Value* off = c.b.build_add(base, c.b.build_mul(s, w));
                Value* ls = kb.load_f32(ptr_at(c, p[0], c.b.build_add(c.b.build_add(off, p[4]), c.i32(1)), 4));
                Value* ms = kb.load_f32(ptr_at(c, p[0], c.b.build_add(off, p[4]), 4));
                Value* term = c.b.build_mul(ls, emit_exp(c, c.b.build_sub(ms, M)));
                return c.b.build_add(cur, c.b.build_select(c.b.build_sgt(ls, c.f32(0.0f)), term, c.f32(0.0f)));
            });
        Value* inv = c.b.build_select(c.b.build_sgt(L, c.f32(0.0f)), emit_div_rn(c, c.f32(1.0f), L), c.f32(0.0f));
        Value* row = c.b.build_udiv(e, p[5]);
        Value* h = c.b.build_umod(e, p[5]);
        Value* dst_row = c.b.build_add(c.b.build_mul(row, p[6]), c.b.build_mul(h, p[4]));

        kb.for_range(tid, p[4], wg, [&](Value* d) {
            Value* acc = kb.for_range_reduce(
                c.i32(0), p[3], c.i32(1), c.f32(0.0f),
                [&](Value* s, Value* cur) -> Value* {
                    Value* off = c.b.build_add(base, c.b.build_mul(s, w));
                    Value* ls = kb.load_f32(ptr_at(c, p[0], c.b.build_add(c.b.build_add(off, p[4]), c.i32(1)), 4));
                    Value* ok = c.b.build_sgt(ls, c.f32(0.0f));
                    Value* inc = emit_if_else_values(
                        c, ok, {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* vs = kb.load_f32(ptr_at(c, p[0], c.b.build_add(off, d), 4));
                            Value* ms = kb.load_f32(ptr_at(c, p[0], c.b.build_add(off, p[4]), 4));
                            return {c.b.build_mul(vs, emit_exp(c, c.b.build_sub(ms, M)))};
                        },
                        [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                    return c.b.build_add(cur, inc);
                });
            store_elem_dt(c, out_dt, p[1], c.b.build_add(dst_row, d), c.b.build_mul(acc, inv));
        });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_attn_softmax_mir(brass::Module& mod, Dtype out_dt, std::uint32_t bias_mode,
                                        bool has_mask, bool has_qmask, bool mask_ge) {
    const std::string name = std::string("mir_attn_softmax_") +
                             (out_dt == Dtype::FP32 ? "f32" : out_dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_b" + std::to_string(bias_mode) +
                             (has_mask ? "_km" : "") + (has_qmask ? "_qm" : "") + (mask_ge ? "_ge" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0..4] = s, p, bias, mask, qmask
    // p[5..12] = rows, lk, lq, h0, lds, ldp, q0, bhs; p[13] = scale
    Value* s8 = kb.shared_alloc_f32(8);
    Value* tid = kb.tid_x();
    Value* wg = c.i32(static_cast<int>(jit::gpu_mir::kEwBlock));

    kb.for_range(kb.ctaid_x(), p[5], kb.nctaid_x(), [&](Value* r) {
        Value* q = c.b.build_add(p[11], c.b.build_umod(r, p[7]));
        Value* h = c.b.build_add(p[8], c.b.build_udiv(r, p[7]));
        Value* so = c.b.build_mul(r, p[9]);
        Value* po = c.b.build_mul(r, p[10]);
        Value* bo = c.b.build_add(c.b.build_mul(h, p[12]), c.b.build_mul(q, p[6]));

        auto reduce_max = [&]() -> Value* {
            return kb.for_range_reduce(
                tid, p[6], wg, c.f32(-1e30f),
                [&](Value* k, Value* cur) -> Value* {
                    if (!has_mask) {
                        Value* sc = eval_softmax_score(c, bias_mode, p[0], p[2], p[3], so, bo, h, q, k, p[12], p[13]);
                        return c.b.build_fmax_f32(cur, sc);
                    }
                    return emit_if_else_values(
                        c, valid_mask(c, p[3], k, mask_ge), {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* sc = eval_softmax_score(c, bias_mode, p[0], p[2], p[3], so, bo, h, q, k, p[12], p[13]);
                            return {c.b.build_fmax_f32(cur, sc)};
                        },
                        [&]() -> std::vector<Value*> { return {cur}; })[0];
                });
        };

        Value* mx_local = !has_qmask
                              ? reduce_max()
                              : emit_if_else_values(
                                    c, valid_mask(c, p[4], q, mask_ge), {Type::f32()},
                                    [&]() -> std::vector<Value*> { return {reduce_max()}; },
                                    [&]() -> std::vector<Value*> { return {c.f32(-1e30f)}; })[0];

        Value* mx = emit_block_max_f32(c, s8, mx_local);
        Value* empty = c.b.build_sle(mx, c.f32(-1e29f));

        Value* sum_local = emit_if_else_values(
            c, empty, {Type::f32()},
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; },
            [&]() -> std::vector<Value*> {
                return {kb.for_range_reduce(
                    tid, p[6], wg, c.f32(0.0f),
                    [&](Value* k, Value* cur) -> Value* {
                        if (!has_mask) {
                            Value* sc = eval_softmax_score(c, bias_mode, p[0], p[2], p[3], so, bo, h, q, k, p[12], p[13]);
                            return c.b.build_add(cur, emit_exp(c, c.b.build_sub(sc, mx)));
                        }
                        return emit_if_else_values(
                            c, valid_mask(c, p[3], k, mask_ge), {Type::f32()},
                            [&]() -> std::vector<Value*> {
                                Value* sc = eval_softmax_score(c, bias_mode, p[0], p[2], p[3], so, bo, h, q, k, p[12], p[13]);
                                return {c.b.build_add(cur, emit_exp(c, c.b.build_sub(sc, mx)))};
                            },
                            [&]() -> std::vector<Value*> { return {cur}; })[0];
                    })};
            })[0];

        Value* sum = emit_block_sum_f32(c, s8, sum_local);
        Value* inv = c.b.build_select(
            c.b.build_and(c.b.build_not(empty), c.b.build_sgt(sum, c.f32(0.0f))),
            emit_div_rn(c, c.f32(1.0f), sum),
            c.f32(0.0f));

        emit_if_else_values(
            c, empty, {},
            [&]() -> std::vector<Value*> {
                kb.for_range(tid, p[6], wg, [&](Value* k) {
                    store_elem_dt(c, out_dt, p[1], c.b.build_add(po, k), c.f32(0.0f));
                });
                return {};
            },
            [&]() -> std::vector<Value*> {
                kb.for_range(tid, p[6], wg, [&](Value* k) {
                    Value* pv;
                    if (!has_mask) {
                        Value* sc = eval_softmax_score(c, bias_mode, p[0], p[2], p[3], so, bo, h, q, k, p[12], p[13]);
                        pv = c.b.build_mul(emit_exp(c, c.b.build_sub(sc, mx)), inv);
                    } else {
                        pv = emit_if_else_values(
                            c, valid_mask(c, p[3], k, mask_ge), {Type::f32()},
                            [&]() -> std::vector<Value*> {
                                Value* sc = eval_softmax_score(c, bias_mode, p[0], p[2], p[3], so, bo, h, q, k, p[12], p[13]);
                                return {c.b.build_mul(emit_exp(c, c.b.build_sub(sc, mx)), inv)};
                            },
                            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                    }
                    store_elem_dt(c, out_dt, p[1], c.b.build_add(po, k), pv);
                });
                return {};
            });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
