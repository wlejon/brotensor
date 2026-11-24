#include "mir_conv_direct.h"

#include "mir_common.h"

#include <string>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kConvDirectFwd = 0;
constexpr std::uint32_t kConvDirectTrans = 1;

Value* in_bounds(Ctx& c, Value* idx, Value* bound) {
    return c.b.build_and(c.b.build_sge(idx, c.i32(0)), c.b.build_slt(idx, bound));
}

Value* conv_direct_fwd(Ctx& c, Dtype dt, const std::vector<Value*>& p,
                       Value* ot, Value* oh, Value* ow, Value* oc,
                       Value* cgi, Value* khw, Value* kvol, Value* ivol, Value* xbase, Value* acc0) {
    // p: [0..3]=x,w,y,bias; [4..15]=n,cin,t,h,wd,cout,to,ho,wo,kt,kh,kw; [16..25]=st,sh,sw,pt,ph,pw,dt,dh,dw,groups
    Value* t0 = c.b.build_sub(c.b.build_mul(ot, p[16]), p[19]);
    Value* h0 = c.b.build_sub(c.b.build_mul(oh, p[17]), p[20]);
    Value* w0 = c.b.build_sub(c.b.build_mul(ow, p[18]), p[21]);
    return c.kb.for_range_reduce(
        c.i32(0), cgi, c.i32(1), acc0,
        [&](Value* icl, Value* acc_ic) -> Value* {
            Value* wb = c.b.build_mul(c.b.build_add(c.b.build_mul(oc, cgi), icl), kvol);
            Value* xb = c.b.build_add(xbase, c.b.build_mul(icl, ivol));
            return c.kb.for_range_reduce(
                c.i32(0), p[13], c.i32(1), acc_ic,
                [&](Value* a, Value* acc_t) -> Value* {
                    Value* it = c.b.build_add(t0, c.b.build_mul(a, p[22]));
                    return emit_if_else_values(
                        c, in_bounds(c, it, p[6]), {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* it_h = c.b.build_mul(it, p[7]);
                            Value* wa = c.b.build_add(wb, c.b.build_mul(a, khw));
                            Value* res = c.kb.for_range_reduce(
                                c.i32(0), p[14], c.i32(1), acc_t,
                                [&](Value* b, Value* acc_h) -> Value* {
                                    Value* ih = c.b.build_add(h0, c.b.build_mul(b, p[23]));
                                    return emit_if_else_values(
                                        c, in_bounds(c, ih, p[7]), {Type::f32()},
                                        [&]() -> std::vector<Value*> {
                                            Value* xr = c.b.build_add(xb, c.b.build_mul(c.b.build_add(it_h, ih), p[8]));
                                            Value* wr = c.b.build_add(wa, c.b.build_mul(b, p[15]));
                                            Value* r_w = c.kb.for_range_reduce(
                                                c.i32(0), p[15], c.i32(1), acc_h,
                                                [&](Value* kw_i, Value* acc_w) -> Value* {
                                                    Value* iw = c.b.build_add(w0, c.b.build_mul(kw_i, p[24]));
                                                    return emit_if_else_values(
                                                        c, in_bounds(c, iw, p[8]), {Type::f32()},
                                                        [&]() -> std::vector<Value*> {
                                                            Value* xv = load_elem_dt(c, dt, p[0], c.b.build_add(xr, iw));
                                                            Value* wv = load_elem_dt(c, dt, p[1], c.b.build_add(wr, kw_i));
                                                            return {c.b.build_fma_f32(xv, wv, acc_w)};
                                                        },
                                                        [&]() -> std::vector<Value*> { return {acc_w}; })[0];
                                                });
                                            return {r_w};
                                        },
                                        [&]() -> std::vector<Value*> { return {acc_h}; })[0];
                                });
                            return {res};
                        },
                        [&]() -> std::vector<Value*> { return {acc_t}; })[0];
                });
        });
}

Value* conv_direct_trans(Ctx& c, Dtype dt, const std::vector<Value*>& p,
                         Value* oh, Value* ow, Value* g, Value* ocl,
                         Value* cgi, Value* cgo, Value* khw, Value* ivol, Value* xbase, Value* acc0) {
    Value* g_base = c.b.build_mul(g, cgi);
    return c.kb.for_range_reduce(
        c.i32(0), p[14], c.i32(1), acc0,
        [&](Value* b, Value* acc_h) -> Value* {
            Value* nh = c.b.build_sub(c.b.build_add(oh, p[20]), c.b.build_mul(b, p[23]));
            Value* nh_pos = c.b.build_sge(nh, c.i32(0));
            Value* nh_div = c.b.build_eq(c.b.build_umod(nh, p[17]), c.i32(0));
            Value* ih = c.b.build_udiv(nh, p[17]);
            Value* h_ok = c.b.build_and(c.b.build_and(nh_pos, nh_div), c.b.build_slt(ih, p[7]));
            return emit_if_else_values(
                c, h_ok, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* ih_wd = c.b.build_mul(ih, p[8]);
                    Value* bk = c.b.build_mul(b, p[15]);
                    Value* r_w = c.kb.for_range_reduce(
                        c.i32(0), p[15], c.i32(1), acc_h,
                        [&](Value* kw_i, Value* acc_w) -> Value* {
                            Value* nw = c.b.build_sub(c.b.build_add(ow, p[21]), c.b.build_mul(kw_i, p[24]));
                            Value* nw_pos = c.b.build_sge(nw, c.i32(0));
                            Value* nw_div = c.b.build_eq(c.b.build_umod(nw, p[18]), c.i32(0));
                            Value* iw = c.b.build_udiv(nw, p[18]);
                            Value* w_ok = c.b.build_and(c.b.build_and(nw_pos, nw_div), c.b.build_slt(iw, p[8]));
                            return emit_if_else_values(
                                c, w_ok, {Type::f32()},
                                [&]() -> std::vector<Value*> {
                                    Value* xo = c.b.build_add(xbase, c.b.build_add(ih_wd, iw));
                                    Value* wk_off = c.b.build_add(bk, kw_i);
                                    Value* r_ic = c.kb.for_range_reduce(
                                        c.i32(0), cgi, c.i32(1), acc_w,
                                        [&](Value* icl, Value* acc_ic) -> Value* {
                                            Value* wi = c.b.build_add(
                                                c.b.build_mul(
                                                    c.b.build_add(c.b.build_mul(c.b.build_add(g_base, icl), cgo), ocl),
                                                    khw),
                                                wk_off);
                                            Value* xv = load_elem_dt(c, dt, p[0], c.b.build_add(xo, c.b.build_mul(icl, ivol)));
                                            Value* wv = load_elem_dt(c, dt, p[1], wi);
                                            return c.b.build_fma_f32(xv, wv, acc_ic);
                                        });
                                    return {r_ic};
                                },
                                [&]() -> std::vector<Value*> { return {acc_w}; })[0];
                        });
                    return {r_w};
                },
                [&]() -> std::vector<Value*> { return {acc_h}; })[0];
        });
}

}  // namespace

brass::Function* build_conv_direct_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, bool has_bias) {
    if (mode != kConvDirectFwd && mode != kConvDirectTrans) {
        throw std::runtime_error("brotensor: mir_conv_direct: unknown mode");
    }
    const std::string name = std::string("mir_conv_direct_") +
                             (mode == kConvDirectFwd ? "fwd_" : "trans_") +
                             (dt == Dtype::FP32 ? "f32" : dt == Dtype::FP16 ? "f16" : "bf16") +
                             (has_bias ? "_b" : "");
    std::vector<Type> params(26, Type::i32());
    params[0] = Type::ptr();
    params[1] = Type::ptr();
    params[2] = Type::ptr();
    params[3] = Type::ptr();
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    Value* plane = c.b.build_mul(p[10], c.b.build_mul(p[11], p[12]));
    Value* total = c.b.build_mul(p[4], c.b.build_mul(p[9], plane));
    Value* cgi = c.b.build_udiv(p[5], p[25]);
    Value* cgo = c.b.build_udiv(p[9], p[25]);
    Value* khw = c.b.build_mul(p[14], p[15]);
    Value* kvol = c.b.build_mul(p[13], khw);
    Value* ivol = c.b.build_mul(p[6], c.b.build_mul(p[7], p[8]));
    Value* hw_o = c.b.build_mul(p[12], p[11]);

    kb.for_range(kb.global_tid_x(), total, grid_stride_ew(c), [&](Value* idx) {
        Value* o = c.b.build_umod(idx, plane);
        Value* rest = c.b.build_udiv(idx, plane);
        Value* oc = c.b.build_umod(rest, p[9]);
        Value* img = c.b.build_udiv(rest, p[9]);
        Value* ow = c.b.build_umod(o, p[12]);
        Value* oh = c.b.build_umod(c.b.build_udiv(o, p[12]), p[11]);
        Value* ot = c.b.build_udiv(o, hw_o);
        Value* g = c.b.build_udiv(oc, cgo);
        Value* ocl = c.b.build_umod(oc, cgo);
        Value* acc0 = has_bias ? load_elem_dt(c, dt, p[3], oc) : c.f32(0.0f);
        Value* xbase = c.b.build_mul(c.b.build_add(c.b.build_mul(img, p[5]), c.b.build_mul(g, cgi)), ivol);

        Value* acc = (mode == kConvDirectFwd)
                         ? conv_direct_fwd(c, dt, p, ot, oh, ow, oc, cgi, khw, kvol, ivol, xbase, acc0)
                         : conv_direct_trans(c, dt, p, oh, ow, g, ocl, cgi, cgo, khw, ivol, xbase, acc0);
        store_elem_dt(c, dt, p[2], idx, acc);
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
