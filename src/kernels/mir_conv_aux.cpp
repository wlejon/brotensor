#include "mir_conv_aux.h"

#include "mir_common.h"
#include "mir_math.h"
#include "mir_quant_helpers.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kAuxBlock = 256;

// Accumulates `delta` into `base[idx]` for FP32, FP16, or BF16 using 64-bit addressing.
void accum_elem_dt_i64(Ctx& c, Dtype dt, Value* base, Value* idx64, Value* delta) {
    Value* p = c.b.build_add(base, c.b.build_shl(idx64, c.kb.const_i64(dt == Dtype::FP32 ? 2 : 1)));
    Value* prev = load_lanes(c, dt, 1, p)[0];
    store_lanes(c, dt, 1, p, {c.b.build_add(prev, delta)});
}

Value* block_sum_256(Ctx& c, Value* s_buf, Value* tid, Value* local_acc) {
    c.kb.shared_store_f32_indexed(s_buf, tid, local_acc);
    c.kb.sync();
    for (int s = 128; s > 0; s >>= 1) {
        c.kb.if_then(c.b.build_slt(tid, c.i32(s)), [&] {
            Value* v = c.b.build_add(
                c.kb.shared_load_f32_indexed(s_buf, tid),
                c.kb.shared_load_f32_indexed(s_buf, c.b.build_add(tid, c.i32(s))));
            c.kb.shared_store_f32_indexed(s_buf, tid, v);
        });
        c.kb.sync();
    }
    return c.kb.shared_load_f32_indexed(s_buf, c.i32(0));
}

Function* build_bwd_weight(Module& mod, Dtype dt, bool block_per_elem) {
    std::vector<Type> params(3, Type::ptr());
    params.insert(params.end(), 17, Type::i32());
    // 0:X 1:dY 2:dWt | 3:N 4:Cin 5:H 6:W 7:Cout 8:kH 9:kW 10:Ho 11:Wo
    //                  12:sh 13:sw 14:ph 15:pw 16:dh 17:dw 18:Cgi 19:Cgo (total=block grid or 19:total)
    if (!block_per_elem) params.push_back(Type::i32());  // 20:total
    Function* fn = mod.create_function(
        block_per_elem ? "mir_conv2d_bwd_w_blk" : "mir_conv2d_bwd_w_thr",
        Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    Value* tid = kb.tid_x();
    Value* s_acc = block_per_elem ? kb.shared_alloc_f32(kAuxBlock) : nullptr;
    Value* idx = block_per_elem ? kb.ctaid_x() : kb.global_tid_x();
    int go = [&] {
        // Compute (c_out, c_in_local, kh, kw) -> shared between modes.
        auto body = [&] {
            Value* kw = c.b.build_umod(idx, p[9]);
            Value* t0 = c.b.build_udiv(idx, p[9]);
            Value* kh = c.b.build_umod(t0, p[8]);
            Value* t1 = c.b.build_udiv(t0, p[8]);
            Value* c_in_loc = c.b.build_umod(t1, p[18]);
            Value* c_out = c.b.build_udiv(t1, p[18]);
            Value* c_in = c.b.build_add(c.b.build_mul(c.b.build_udiv(c_out, p[19]), p[18]), c_in_loc);
            Value* spatial = c.b.build_mul(p[10], p[11]);
            Value* hw_in = c.b.build_mul(p[5], p[6]);
            Value* cin_hw = c.b.build_mul(p[4], hw_in);
            Value* cout_sp = c.b.build_mul(p[7], spatial);
            Value* kh_dil = c.b.build_sub(c.b.build_mul(kh, p[16]), p[14]);
            Value* kw_dil = c.b.build_sub(c.b.build_mul(kw, p[17]), p[15]);
            Value* sum = c.f32(0.0f);

            if (block_per_elem) {
                Value* M = c.b.build_mul(p[3], spatial);
                Value* local = kb.for_range_reduce(
                    tid, M, c.i32(kAuxBlock), c.f32(0.0f),
                    [&](Value* m, Value* cur) -> Value* {
                        Value* n = c.b.build_udiv(m, spatial);
                        Value* sp = c.b.build_umod(m, spatial);
                        Value* io = c.b.build_udiv(sp, p[11]);
                        Value* jo = c.b.build_umod(sp, p[11]);
                        Value* ih = c.b.build_add(c.b.build_mul(io, p[12]), kh_dil);
                        Value* iw = c.b.build_add(c.b.build_mul(jo, p[13]), kw_dil);
                        Value* ok = c.b.build_and(
                            c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, p[5])),
                            c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, p[6])));
                        return emit_if_else_values(
                            c, ok, {Type::f32()},
                            [&]() -> std::vector<Value*> {
                                Value* xi = c.b.build_add(
                                    c.b.build_add(c.b.build_mul(c.b.build_zext_i64(n), c.b.build_zext_i64(cin_hw)),
                                                  c.b.build_mul(c.b.build_zext_i64(c_in), c.b.build_zext_i64(hw_in))),
                                    c.b.build_zext_i64(c.b.build_add(c.b.build_mul(ih, p[6]), iw)));
                                Value* dyi = c.b.build_add(
                                    c.b.build_add(c.b.build_mul(c.b.build_zext_i64(n), c.b.build_zext_i64(cout_sp)),
                                                  c.b.build_mul(c.b.build_zext_i64(c_out), c.b.build_zext_i64(spatial))),
                                    c.b.build_zext_i64(sp));
                                const int esz_sh = (dt == Dtype::FP32) ? 2 : 1;
                                Value* xv = load_lanes(c, dt, 1, c.b.build_add(p[0], c.b.build_shl(xi, kb.const_i64(esz_sh))))[0];
                                Value* dv = load_lanes(c, dt, 1, c.b.build_add(p[1], c.b.build_shl(dyi, kb.const_i64(esz_sh))))[0];
                                return {c.b.build_fma_f32(dv, xv, cur)};
                            },
                            [&]() -> std::vector<Value*> { return {cur}; })[0];
                    });
                sum = block_sum_256(c, s_acc, tid, local);
                kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                    accum_elem_dt_i64(c, dt, p[2], c.b.build_zext_i64(idx), sum);
                });
            } else {
                const int esz_sh = (dt == Dtype::FP32) ? 2 : 1;
                sum = kb.for_range_reduce(
                    c.i32(0), p[3], c.i32(1), c.f32(0.0f),
                    [&](Value* n, Value* a0) -> Value* {
                        Value* xn = c.b.build_add(
                            c.b.build_mul(c.b.build_zext_i64(n), c.b.build_zext_i64(cin_hw)),
                            c.b.build_mul(c.b.build_zext_i64(c_in), c.b.build_zext_i64(hw_in)));
                        Value* dyn = c.b.build_add(
                            c.b.build_mul(c.b.build_zext_i64(n), c.b.build_zext_i64(cout_sp)),
                            c.b.build_mul(c.b.build_zext_i64(c_out), c.b.build_zext_i64(spatial)));
                        return kb.for_range_reduce(
                            c.i32(0), p[10], c.i32(1), a0,
                            [&](Value* io, Value* a1) -> Value* {
                                Value* ih = c.b.build_add(c.b.build_mul(io, p[12]), kh_dil);
                                Value* h_ok = c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, p[5]));
                                return emit_if_else_values(
                                    c, h_ok, {Type::f32()},
                                    [&]() -> std::vector<Value*> {
                                        Value* xrow = c.b.build_add(xn, c.b.build_zext_i64(c.b.build_mul(ih, p[6])));
                                        Value* dyrow = c.b.build_add(dyn, c.b.build_zext_i64(c.b.build_mul(io, p[11])));
                                        return {kb.for_range_reduce(
                                            c.i32(0), p[11], c.i32(1), a1,
                                            [&](Value* jo, Value* a2) -> Value* {
                                                Value* iw = c.b.build_add(c.b.build_mul(jo, p[13]), kw_dil);
                                                Value* w_ok = c.b.build_and(
                                                    c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, p[6]));
                                                return emit_if_else_values(
                                                    c, w_ok, {Type::f32()},
                                                    [&]() -> std::vector<Value*> {
                                                        Value* xi = c.b.build_add(xrow, c.b.build_zext_i64(iw));
                                                        Value* dyi = c.b.build_add(dyrow, c.b.build_zext_i64(jo));
                                                        Value* xv = load_lanes(c, dt, 1, c.b.build_add(p[0], c.b.build_shl(xi, kb.const_i64(esz_sh))))[0];
                                                        Value* dv = load_lanes(c, dt, 1, c.b.build_add(p[1], c.b.build_shl(dyi, kb.const_i64(esz_sh))))[0];
                                                        return {c.b.build_fma_f32(dv, xv, a2)};
                                                    },
                                                    [&]() -> std::vector<Value*> { return {a2}; })[0];
                                            })};
                                    },
                                    [&]() -> std::vector<Value*> { return {a1}; })[0];
                            });
                    });
                accum_elem_dt_i64(c, dt, p[2], c.b.build_zext_i64(idx), sum);
            }
        };
        if (block_per_elem) body();
        else kb.if_then(c.b.build_slt(idx, p[20]), body);
        return 0;
    }();
    (void)go;
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_bwd_bias(Module& mod, Dtype dt) {
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function("mir_conv2d_bwd_bias", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // 0:dY 1:dB | 2:N 3:C_out 4:Ho 5:Wo
    Value* s_acc = kb.shared_alloc_f32(kAuxBlock);
    Value* oc = kb.ctaid_x();
    Value* tid = kb.tid_x();
    Value* spatial = c.b.build_mul(p[4], p[5]);
    Value* total = c.b.build_mul(p[2], spatial);
    Value* cout_sp = c.b.build_mul(p[3], spatial);
    const int esz_sh = (dt == Dtype::FP32) ? 2 : 1;

    Value* local = kb.for_range_reduce(
        tid, total, c.i32(kAuxBlock), c.f32(0.0f),
        [&](Value* idx, Value* cur) -> Value* {
            Value* n = c.b.build_udiv(idx, spatial);
            Value* sp = c.b.build_umod(idx, spatial);
            Value* off = c.b.build_add(
                c.b.build_add(c.b.build_mul(c.b.build_zext_i64(n), c.b.build_zext_i64(cout_sp)),
                              c.b.build_mul(c.b.build_zext_i64(oc), c.b.build_zext_i64(spatial))),
                c.b.build_zext_i64(sp));
            Value* v = load_lanes(c, dt, 1, c.b.build_add(p[0], c.b.build_shl(off, kb.const_i64(esz_sh))))[0];
            return c.b.build_add(cur, v);
        });
    Value* sum = block_sum_256(c, s_acc, tid, local);
    kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
        accum_elem_dt_i64(c, dt, p[1], c.b.build_zext_i64(oc), sum);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_conv3d_int8w(Module& mod, bool has_bias) {
    std::vector<Type> params(5, Type::ptr());
    params.insert(params.end(), 24, Type::i32());
    // 0:X 1:W 2:scales 3:bias 4:Y
    // 5:N 6:Cin 7:Tin 8:H 9:Win 10:Cout 11:kT 12:kH 13:kW 14:To 15:Ho 16:Wo
    // 17:st 18:sh 19:sw 20:pt 21:ph 22:pw 23:dt 24:dh 25:dw 26:Cgi 27:Cgo 28:total
    Function* fn = mod.create_function("mir_conv3d_int8w_fp16", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, p[28]), [&] {
        Value* ow = c.b.build_umod(idx, p[16]);
        Value* t0 = c.b.build_udiv(idx, p[16]);
        Value* oh = c.b.build_umod(t0, p[15]);
        Value* t1 = c.b.build_udiv(t0, p[15]);
        Value* ot = c.b.build_umod(t1, p[14]);
        Value* t2 = c.b.build_udiv(t1, p[14]);
        Value* oc = c.b.build_umod(t2, p[10]);
        Value* n = c.b.build_udiv(t2, p[10]);

        Value* it0 = c.b.build_sub(c.b.build_mul(ot, p[17]), p[20]);
        Value* ih0 = c.b.build_sub(c.b.build_mul(oh, p[18]), p[21]);
        Value* iw0 = c.b.build_sub(c.b.build_mul(ow, p[19]), p[22]);
        Value* scale = kb.load_f32(ptr_at(c, p[2], oc, 4));
        Value* ic_abs_base = c.b.build_mul(c.b.build_udiv(oc, p[27]), p[26]);
        Value* khw = c.b.build_mul(p[12], p[13]);
        Value* kthw = c.b.build_mul(p[11], khw);
        Value* w_oc_base = c.b.build_mul(oc, c.b.build_mul(p[26], kthw));
        Value* hwi = c.b.build_mul(p[8], p[9]);
        Value* thwi = c.b.build_mul(p[7], hwi);
        Value* x_n_base = c.b.build_mul(n, c.b.build_mul(p[6], thwi));

        Value* acc = kb.for_range_reduce(
            c.i32(0), p[26], c.i32(1), c.f32(0.0f),
            [&](Value* ic_loc, Value* a0) -> Value* {
                Value* ic = c.b.build_add(ic_abs_base, ic_loc);
                Value* w_ic = c.b.build_add(w_oc_base, c.b.build_mul(ic_loc, kthw));
                Value* x_ic = c.b.build_add(x_n_base, c.b.build_mul(ic, thwi));
                return kb.for_range_reduce(
                    c.i32(0), p[11], c.i32(1), a0,
                    [&](Value* kt, Value* a1) -> Value* {
                        Value* it = c.b.build_add(it0, c.b.build_mul(kt, p[23]));
                        Value* t_ok = c.b.build_and(c.b.build_sge(it, c.i32(0)), c.b.build_slt(it, p[7]));
                        return emit_if_else_values(
                            c, t_ok, {Type::f32()},
                            [&]() -> std::vector<Value*> {
                                Value* w_kt = c.b.build_add(w_ic, c.b.build_mul(kt, khw));
                                Value* x_t = c.b.build_add(x_ic, c.b.build_mul(it, hwi));
                                return {kb.for_range_reduce(
                                    c.i32(0), p[12], c.i32(1), a1,
                                    [&](Value* kh, Value* a2) -> Value* {
                                        Value* ih = c.b.build_add(ih0, c.b.build_mul(kh, p[24]));
                                        Value* h_ok = c.b.build_and(
                                            c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, p[8]));
                                        return emit_if_else_values(
                                            c, h_ok, {Type::f32()},
                                            [&]() -> std::vector<Value*> {
                                                Value* w_kh = c.b.build_add(w_kt, c.b.build_mul(kh, p[13]));
                                                Value* x_h = c.b.build_add(x_t, c.b.build_mul(ih, p[9]));
                                                return {kb.for_range_reduce(
                                                    c.i32(0), p[13], c.i32(1), a2,
                                                    [&](Value* kw, Value* a3) -> Value* {
                                                        Value* iw = c.b.build_add(iw0, c.b.build_mul(kw, p[25]));
                                                        Value* w_ok = c.b.build_and(
                                                            c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, p[9]));
                                                        return emit_if_else_values(
                                                            c, w_ok, {Type::f32()},
                                                            [&]() -> std::vector<Value*> {
                                                                Value* xv = load_elem_dt(
                                                                    c, Dtype::FP16, p[0], c.b.build_add(x_h, iw));
                                                                Value* ub = kb.load_u8(
                                                                    ptr_at(c, p[1], c.b.build_add(w_kh, kw), 1));
                                                                Value* wv = c.b.build_mul(
                                                                    emit_s8_byte_to_f32(c, ub), scale);
                                                                return {c.b.build_fma_f32(xv, wv, a3)};
                                                            },
                                                            [&]() -> std::vector<Value*> { return {a3}; })[0];
                                                    })};
                                            },
                                            [&]() -> std::vector<Value*> { return {a2}; })[0];
                                    })};
                            },
                            [&]() -> std::vector<Value*> { return {a1}; })[0];
                    });
            });
        if (has_bias) acc = c.b.build_add(acc, load_elem_dt(c, Dtype::FP16, p[3], oc));
        store_elem_dt(c, Dtype::FP16, p[4], idx, acc);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_col2im_f32(Module& mod, bool has_bias) {
    std::vector<Type> params(3, Type::ptr());
    params.insert(params.end(), 13, Type::i32());
    // 0:cols 1:bias 2:Y | 3:N 4:Cout 5:H 6:W 7:kH 8:kW 9:sh 10:sw 11:ph 12:pw 13:dh 14:dw 15:Ho
    // (16:Wo added below)
    params.push_back(Type::i32());  // 16:Wo
    Function* fn = mod.create_function("mir_col2im_f32", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    Value* total = c.b.build_mul(
        c.b.build_mul(c.b.build_zext_i64(p[3]), c.b.build_zext_i64(p[4])),
        c.b.build_mul(c.b.build_zext_i64(p[15]), c.b.build_zext_i64(p[16])));
    Value* gid = c.b.build_zext_i64(kb.global_tid_x());
    Value* stride = c.b.build_zext_i64(grid_stride_ew(c));
    Value* Wo64 = c.b.build_zext_i64(p[16]);
    Value* Ho64 = c.b.build_zext_i64(p[15]);
    Value* Co64 = c.b.build_zext_i64(p[4]);
    Value* khw_hw = c.b.build_mul(
        c.b.build_mul(c.b.build_zext_i64(p[7]), c.b.build_zext_i64(p[8])),
        c.b.build_mul(c.b.build_zext_i64(p[5]), c.b.build_zext_i64(p[6])));

    kb.for_range(gid, total, stride, [&](Value* idx) {
        Value* ow = c.b.build_trunc_i32(c.b.build_umod(idx, Wo64));
        Value* t = c.b.build_udiv(idx, Wo64);
        Value* oh = c.b.build_trunc_i32(c.b.build_umod(t, Ho64));
        Value* t2 = c.b.build_udiv(t, Ho64);
        Value* oc = c.b.build_trunc_i32(c.b.build_umod(t2, Co64));
        Value* n = c.b.build_trunc_i32(c.b.build_udiv(t2, Co64));

        Value* init_acc = has_bias ? load_elem_dt(c, Dtype::FP32, p[1], oc) : c.f32(0.0f);
        Value* cbase_idx = c.b.build_mul(
            c.b.build_add(c.b.build_mul(c.b.build_zext_i64(n), Co64), c.b.build_zext_i64(oc)), khw_hw);
        Value* cbase = c.b.build_add(p[0], c.b.build_shl(cbase_idx, kb.const_i64(2)));
        Value* hw_in = c.b.build_mul(p[5], p[6]);

        Value* acc = kb.for_range_reduce(
            c.i32(0), p[7], c.i32(1), init_acc,
            [&](Value* kh, Value* a0) -> Value* {
                Value* th = c.b.build_sub(c.b.build_add(oh, p[11]), c.b.build_mul(kh, p[13]));
                Value* th_nn = c.b.build_sge(th, c.i32(0));
                return emit_if_else_values(
                    c, th_nn, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* ih = c.b.build_udiv(th, p[9]);
                        Value* th_ok = c.b.build_and(
                            c.b.build_eq(c.b.build_umod(th, p[9]), c.i32(0)),
                            c.b.build_slt(ih, p[5]));
                        return emit_if_else_values(
                            c, th_ok, {Type::f32()},
                            [&]() -> std::vector<Value*> {
                                return {kb.for_range_reduce(
                                    c.i32(0), p[8], c.i32(1), a0,
                                    [&](Value* kw, Value* a1) -> Value* {
                                        Value* tw = c.b.build_sub(c.b.build_add(ow, p[12]), c.b.build_mul(kw, p[14]));
                                        Value* tw_nn = c.b.build_sge(tw, c.i32(0));
                                        return emit_if_else_values(
                                            c, tw_nn, {Type::f32()},
                                            [&]() -> std::vector<Value*> {
                                                Value* iw = c.b.build_udiv(tw, p[10]);
                                                Value* tw_ok = c.b.build_and(
                                                    c.b.build_eq(c.b.build_umod(tw, p[10]), c.i32(0)),
                                                    c.b.build_slt(iw, p[6]));
                                                return emit_if_else_values(
                                                    c, tw_ok, {Type::f32()},
                                                    [&]() -> std::vector<Value*> {
                                                        Value* tap = c.b.build_add(c.b.build_mul(kh, p[8]), kw);
                                                        Value* off = c.b.build_add(
                                                            c.b.build_mul(c.b.build_add(c.b.build_mul(tap, p[5]), ih), p[6]), iw);
                                                        return {c.b.build_add(a1, kb.load_f32(ptr_at(c, cbase, off, 4)))};
                                                    },
                                                    [&]() -> std::vector<Value*> { return {a1}; });
                                            },
                                            [&]() -> std::vector<Value*> { return {a1}; })[0];
                                    })};
                            },
                            [&]() -> std::vector<Value*> { return {a0}; });
                    },
                    [&]() -> std::vector<Value*> { return {a0}; })[0];
            });
        kb.store_f32(c.b.build_add(p[2], c.b.build_shl(idx, kb.const_i64(2))), acc);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_resblock_shift(Module& mod, Dtype dt, bool has_N) {
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function("mir_resblock_add_shift", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // 0:Y 1:shift | 2:N 3:C 4:spatial
    Value* total = c.b.build_mul(p[2], c.b.build_mul(p[3], p[4]));
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, total), [&] {
        Value* t = c.b.build_udiv(idx, p[4]);
        Value* ch = c.b.build_umod(t, p[3]);
        Value* n = c.b.build_udiv(t, p[3]);
        Value* sidx = has_N ? c.b.build_add(c.b.build_mul(n, p[3]), ch) : ch;
        Value* yv = load_elem_dt(c, dt, p[0], idx);
        Value* sv = load_elem_dt(c, dt, p[1], sidx);
        store_elem_dt(c, dt, p[0], idx, c.b.build_add(yv, sv));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_resblock_sum_hw(Module& mod, Dtype dt) {
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function("mir_resblock_sum_hw", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // 0:dh2 1:d_shift | 2:N 3:C 4:spatial
    Value* s_buf = kb.shared_alloc_f32(kAuxBlock);
    Value* nc = kb.ctaid_x();
    Value* tid = kb.tid_x();
    kb.if_then(c.b.build_slt(nc, c.b.build_mul(p[2], p[3])), [&] {
        Value* rbase = c.b.build_mul(c.b.build_zext_i64(nc), c.b.build_zext_i64(p[4]));
        Value* row = c.b.build_add(p[0], c.b.build_shl(rbase, kb.const_i64(1)));
        Value* local = kb.for_range_reduce(
            tid, p[4], c.i32(kAuxBlock), c.f32(0.0f),
            [&](Value* i, Value* cur) -> Value* {
                return c.b.build_add(cur, load_elem_dt(c, dt, row, i));
            });
        Value* sum = block_sum_256(c, s_buf, tid, local);
        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            accum_elem_dt_i64(c, dt, p[1], c.b.build_zext_i64(nc), sum);
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_splitk_fold(Module& mod, Dtype dt) {
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function("mir_conv2d_splitk_fold", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // 0:part(f32) 1:dst(dt) | 2:n 3:S
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, p[2]), [&] {
        Value* n64 = c.b.build_zext_i64(p[2]);
        Value* idx64 = c.b.build_zext_i64(idx);
        Value* acc = kb.for_range_reduce(
            c.i32(0), p[3], c.i32(1), c.f32(0.0f),
            [&](Value* s, Value* cur) -> Value* {
                Value* off = c.b.build_add(c.b.build_mul(c.b.build_zext_i64(s), n64), idx64);
                Value* v = kb.load_f32(c.b.build_add(p[0], c.b.build_shl(off, kb.const_i64(2))));
                return c.b.build_add(cur, v);
            });
        accum_elem_dt_i64(c, dt, p[1], idx64, acc);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace

Function* build_conv_aux_mir(Module& mod, std::uint32_t op, Dtype dt, std::uint8_t flags) {
    switch (op) {
        case 0u: return build_bwd_weight(mod, dt, true);
        case 1u: return build_bwd_weight(mod, dt, false);
        case 2u: return build_bwd_bias(mod, dt);
        case 3u: return build_conv3d_int8w(mod, (flags & 1u) != 0);
        case 4u: return build_col2im_f32(mod, (flags & 1u) != 0);
        case 5u: return build_resblock_shift(mod, dt, (flags & 1u) != 0);
        case 6u: return build_resblock_sum_hw(mod, dt);
        case 7u: return build_splitk_fold(mod, dt);
        default: throw std::runtime_error("build_conv_aux_mir: unknown op");
    }
}

}  // namespace brotensor::kernels
