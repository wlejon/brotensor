#include "mir_vision.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* min_i32(Ctx& c, Value* a, Value* b) {
    return c.b.build_select(c.b.build_slt(a, b), a, b);
}

Value* max_i32(Ctx& c, Value* a, Value* b) {
    return c.b.build_select(c.b.build_sgt(a, b), a, b);
}

void build_upfirdn(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool flip) {
    // p: X(0) f(1) Y(2) NC(3) Hin(4) Win(5) Hout(6) Wout(7) fH(8) fW(9)
    //    upx(10) upy(11) downx(12) downy(13) px0(14) py0(15) gain(16)
    Value* NC = p[3];
    Value* Hin = p[4];
    Value* Win = p[5];
    Value* Hout = p[6];
    Value* Wout = p[7];
    Value* fH = p[8];
    Value* fW = p[9];
    Value* upx = p[10];
    Value* upy = p[11];
    Value* downx = p[12];
    Value* downy = p[13];
    Value* px0 = p[14];
    Value* py0 = p[15];
    Value* gain = p[16];
    Value* Hu = c.b.build_mul(Hin, upy);
    Value* Wu = c.b.build_mul(Win, upx);
    Value* plane = c.b.build_mul(Hout, Wout);
    Value* in_plane = c.b.build_mul(Hin, Win);
    Value* total = c.b.build_mul(NC, plane);

    emit_vec_grid_stride(c, total, 1, kVisionBlock, [&](Value* i) {
        Value* nc = c.b.build_udiv(i, plane);
        Value* rem = c.b.build_sub(i, c.b.build_mul(nc, plane));
        Value* oh = c.b.build_udiv(rem, Wout);
        Value* ow = c.b.build_sub(rem, c.b.build_mul(oh, Wout));
        Value* by = c.b.build_sub(c.b.build_mul(oh, downy), py0);
        Value* bx = c.b.build_sub(c.b.build_mul(ow, downx), px0);

        Value* kh0a = max_i32(c, c.i32(0), c.b.build_neg(by));
        Value* rh = c.b.build_umod(c.b.build_add(by, kh0a), upy);
        Value* kh0 = c.b.build_select(c.b.build_ne(rh, c.i32(0)),
                                      c.b.build_add(kh0a, c.b.build_sub(upy, rh)), kh0a);
        Value* kh1 = min_i32(c, fH, c.b.build_sub(Hu, by));

        Value* kw0a = max_i32(c, c.i32(0), c.b.build_neg(bx));
        Value* rw = c.b.build_umod(c.b.build_add(bx, kw0a), upx);
        Value* kw0 = c.b.build_select(c.b.build_ne(rw, c.i32(0)),
                                      c.b.build_add(kw0a, c.b.build_sub(upx, rw)), kw0a);
        Value* kw1 = min_i32(c, fW, c.b.build_sub(Wu, bx));

        Value* ib = c.b.build_mul(nc, in_plane);
        Value* acc = c.kb.for_range_reduce(kh0, kh1, upy, c.f32(0.0f), [&](Value* kh, Value* row_acc) {
            Value* iy = c.b.build_udiv(c.b.build_add(by, kh), upy);
            Value* fy = flip ? kh : c.b.build_sub(c.b.build_sub(fH, c.i32(1)), kh);
            Value* xr = c.b.build_add(ib, c.b.build_mul(iy, Win));
            Value* fr = c.b.build_mul(fy, fW);
            return c.kb.for_range_reduce(kw0, kw1, upx, row_acc, [&](Value* kw, Value* cur) {
                Value* ix = c.b.build_udiv(c.b.build_add(bx, kw), upx);
                Value* fx = flip ? kw : c.b.build_sub(c.b.build_sub(fW, c.i32(1)), kw);
                Value* xv = load_elem_dt(c, dt, p[0], c.b.build_add(xr, ix));
                Value* fv = load_elem_dt(c, dt, p[1], c.b.build_add(fr, fx));
                return c.b.build_add(cur, c.b.build_mul(xv, fv));
            });
        });
        store_elem_dt(c, dt, p[2], i, c.b.build_mul(acc, gain));
    });
}

void build_modw(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool demodulate, bool from_dcoef) {
    // p: W(0) s(1) Wn(2) dcoef(3) N(4) Cout(5) Cin(6) kk(7) eps(8)
    Value* N = p[4];
    Value* Cout = p[5];
    Value* Cin = p[6];
    Value* kk = p[7];
    Value* eps = p[8];
    Value* wk = c.b.build_mul(Cin, kk);
    Value* total = c.b.build_mul(N, Cout);
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kVisionBlock));
    Value* s_red = (!from_dcoef && demodulate) ? c.kb.shared_alloc_f32(8) : nullptr;

    c.kb.for_range(c.kb.ctaid_x(), total, c.kb.nctaid_x(), [&](Value* no) {
        Value* n = c.b.build_udiv(no, Cout);
        Value* o = c.b.build_sub(no, c.b.build_mul(n, Cout));
        Value* w_base = c.b.build_mul(o, wk);
        Value* s_base = c.b.build_mul(n, Cin);
        Value* wn_base = c.b.build_mul(no, wk);

        Value* dc = c.f32(1.0f);
        if (demodulate) {
            if (from_dcoef) {
                dc = c.kb.load_f32(ptr_at(c, p[3], no, 4));
            } else {
                Value* ss = c.kb.for_range_reduce(tid, wk, block, c.f32(0.0f), [&](Value* j, Value* cur) {
                    Value* ic = c.b.build_udiv(j, kk);
                    Value* w = c.b.build_mul(load_elem_dt(c, dt, p[0], c.b.build_add(w_base, j)),
                                             load_elem_dt(c, dt, p[1], c.b.build_add(s_base, ic)));
                    return c.b.build_add(cur, c.b.build_mul(w, w));
                });
                Value* sum_sq = emit_block_sum_f32(c, s_red, ss);
                dc = c.b.build_sdiv(c.f32(1.0f), c.b.build_sqrt_f32(c.b.build_add(sum_sq, eps)));
            }
        }
        c.kb.for_range(tid, wk, block, [&](Value* j) {
            Value* ic = c.b.build_udiv(j, kk);
            Value* w = c.b.build_mul(load_elem_dt(c, dt, p[0], c.b.build_add(w_base, j)),
                                     load_elem_dt(c, dt, p[1], c.b.build_add(s_base, ic)));
            Value* out = demodulate ? c.b.build_mul(w, dc) : w;
            store_elem_dt(c, dt, p[2], c.b.build_add(wn_base, j), out);
        });
        if (!from_dcoef) {
            c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] { c.kb.store_f32(ptr_at(c, p[3], no, 4), dc); });
            c.kb.sync();
        }
    });
}

void build_modw_bwd(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    // p: W(0) s(1) dw(2) dcoef(3) N(4) Cout(5) Cin(6) kk(7)
    Value* Cout = p[5];
    Value* Cin = p[6];
    Value* kk = p[7];
    Value* wk = c.b.build_mul(Cin, kk);
    Value* total = c.b.build_mul(p[4], Cout);
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kVisionBlock));
    Value* s_red = c.kb.shared_alloc_f32(8);

    c.kb.for_range(c.kb.ctaid_x(), total, c.kb.nctaid_x(), [&](Value* no) {
        Value* n = c.b.build_udiv(no, Cout);
        Value* o = c.b.build_sub(no, c.b.build_mul(n, Cout));
        Value* w_base = c.b.build_mul(o, wk);
        Value* s_base = c.b.build_mul(n, Cin);
        Value* dw_base = c.b.build_mul(no, wk);
        Value* g0 = c.kb.for_range_reduce(tid, wk, block, c.f32(0.0f), [&](Value* j, Value* cur) {
            Value* ic = c.b.build_udiv(j, kk);
            Value* wp = c.b.build_mul(load_elem_dt(c, dt, p[0], c.b.build_add(w_base, j)),
                                      load_elem_dt(c, dt, p[1], c.b.build_add(s_base, ic)));
            return c.b.build_add(cur, c.b.build_mul(c.kb.load_f32(ptr_at(c, p[2], c.b.build_add(dw_base, j), 4)), wp));
        });
        Value* g = emit_block_sum_f32(c, s_red, g0);
        Value* dc = c.kb.load_f32(ptr_at(c, p[3], no, 4));
        Value* gd3 = c.b.build_mul(c.b.build_mul(c.b.build_mul(g, dc), dc), dc);
        c.kb.for_range(tid, wk, block, [&](Value* j) {
            Value* ic = c.b.build_udiv(j, kk);
            Value* wp = c.b.build_mul(load_elem_dt(c, dt, p[0], c.b.build_add(w_base, j)),
                                      load_elem_dt(c, dt, p[1], c.b.build_add(s_base, ic)));
            Value* ptr = ptr_at(c, p[2], c.b.build_add(dw_base, j), 4);
            Value* dwn = c.b.build_sub(c.b.build_mul(c.kb.load_f32(ptr), dc), c.b.build_mul(gd3, wp));
            c.kb.store_f32(ptr, dwn);
        });
    });
}

void build_modw_ds_dw(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool is_dw) {
    // ds: W(0) dw(1) ds(2) N(3) Cout(4) Cin(5) kk(6)
    // dw: s(0) dw(1) dW(2) N(3) Cout(4) Cin(5) kk(6)
    Value* N = p[3];
    Value* Cout = p[4];
    Value* Cin = p[5];
    Value* kk = p[6];
    Value* wk = c.b.build_mul(Cin, kk);
    if (!is_dw) {
        emit_vec_grid_stride(c, c.b.build_mul(N, Cin), 1, kVisionBlock, [&](Value* ni) {
            Value* n = c.b.build_udiv(ni, Cin);
            Value* ic = c.b.build_sub(ni, c.b.build_mul(n, Cin));
            Value* nc_base = c.b.build_mul(n, Cout);
            Value* ik = c.b.build_mul(ic, kk);
            Value* acc = c.kb.for_range_reduce(c.i32(0), Cout, c.i32(1), c.f32(0.0f), [&](Value* o, Value* r_acc) {
                Value* dw_b = c.b.build_mul(c.b.build_add(nc_base, o), wk);
                Value* w_b = c.b.build_mul(o, wk);
                return c.kb.for_range_reduce(c.i32(0), kk, c.i32(1), r_acc, [&](Value* t, Value* cur) {
                    Value* j = c.b.build_add(ik, t);
                    Value* dwp = c.kb.load_f32(ptr_at(c, p[1], c.b.build_add(dw_b, j), 4));
                    Value* w = load_elem_dt(c, dt, p[0], c.b.build_add(w_b, j));
                    return c.b.build_add(cur, c.b.build_mul(dwp, w));
                });
            });
            store_elem_dt(c, dt, p[2], ni, acc);
        });
    } else {
        emit_vec_grid_stride(c, c.b.build_mul(Cout, wk), 1, kVisionBlock, [&](Value* oj) {
            Value* o = c.b.build_udiv(oj, wk);
            Value* j = c.b.build_sub(oj, c.b.build_mul(o, wk));
            Value* ic = c.b.build_udiv(j, kk);
            Value* acc = c.kb.for_range_reduce(c.i32(0), N, c.i32(1), c.f32(0.0f), [&](Value* n, Value* cur) {
                Value* idx_dw = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, Cout), o), wk), j);
                Value* idx_s = c.b.build_add(c.b.build_mul(n, Cin), ic);
                return c.b.build_add(cur, c.b.build_mul(c.kb.load_f32(ptr_at(c, p[1], idx_dw, 4)),
                                                       load_elem_dt(c, dt, p[0], idx_s)));
            });
            store_elem_dt(c, dt, p[2], oj, c.b.build_add(load_elem_dt(c, dt, p[2], oj), acc));
        });
    }
}

void build_col2im2d(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool has_bias, bool cols_f32) {
    // p: cols(0) bias(1) Y(2) N(3) Cout(4) H(5) W(6) kH(7) kW(8) sh(9) sw(10) ph(11) pw(12) dh(13) dw(14) Ho(15) Wo(16)
    Value* Cout = p[4];
    Value* H = p[5];
    Value* W = p[6];
    Value* kH = p[7];
    Value* kW = p[8];
    Value* sh = p[9];
    Value* sw = p[10];
    Value* ph = p[11];
    Value* pw = p[12];
    Value* dh = p[13];
    Value* dw = p[14];
    Value* Ho = p[15];
    Value* Wo = p[16];
    Value* plane = c.b.build_mul(Ho, Wo);
    Value* kk = c.b.build_mul(kH, kW);
    Value* hw = c.b.build_mul(H, W);
    Value* total = c.b.build_mul(c.b.build_mul(p[3], Cout), plane);

    emit_vec_grid_stride(c, total, 1, kVisionBlock, [&](Value* i) {
        Value* noc = c.b.build_udiv(i, plane);
        Value* px = c.b.build_sub(i, c.b.build_mul(noc, plane));
        Value* n = c.b.build_udiv(noc, Cout);
        Value* oc = c.b.build_sub(noc, c.b.build_mul(n, Cout));
        Value* oh = c.b.build_udiv(px, Wo);
        Value* ow = c.b.build_sub(px, c.b.build_mul(oh, Wo));
        Value* rowb = c.b.build_mul(noc, kk);
        Value* init = has_bias ? load_elem_dt(c, dt, p[1], oc) : c.f32(0.0f);

        Value* acc = c.kb.for_range_reduce(c.i32(0), kH, c.i32(1), init, [&](Value* kh, Value* r_acc) {
            Value* th = c.b.build_sub(c.b.build_add(oh, ph), c.b.build_mul(kh, dh));
            Value* th_nn = c.b.build_sge(th, c.i32(0));
            return emit_if_else_values(
                c, th_nn, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* ih = c.b.build_udiv(th, sh);
                    Value* th_div = c.b.build_eq(c.b.build_sub(th, c.b.build_mul(ih, sh)), c.i32(0));
                    Value* h_ok = c.b.build_and(th_div, c.b.build_slt(ih, H));
                    return emit_if_else_values(
                        c, h_ok, {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* row_k = c.b.build_add(rowb, c.b.build_mul(kh, kW));
                            Value* ih_w = c.b.build_mul(ih, W);
                            return {c.kb.for_range_reduce(c.i32(0), kW, c.i32(1), r_acc, [&](Value* kw, Value* cur) {
                                Value* tw = c.b.build_sub(c.b.build_add(ow, pw), c.b.build_mul(kw, dw));
                                Value* tw_nn = c.b.build_sge(tw, c.i32(0));
                                return emit_if_else_values(
                                    c, tw_nn, {Type::f32()},
                                    [&]() -> std::vector<Value*> {
                                        Value* iw = c.b.build_udiv(tw, sw);
                                        Value* tw_div = c.b.build_eq(c.b.build_sub(tw, c.b.build_mul(iw, sw)), c.i32(0));
                                        Value* w_ok = c.b.build_and(tw_div, c.b.build_slt(iw, W));
                                        return emit_if_else_values(
                                            c, w_ok, {Type::f32()},
                                            [&]() -> std::vector<Value*> {
                                                Value* ci = c.b.build_add(
                                                    c.b.build_mul(c.b.build_add(row_k, kw), hw),
                                                    c.b.build_add(ih_w, iw));
                                                Value* v = cols_f32 ? c.kb.load_f32(ptr_at(c, p[0], ci, 4))
                                                                    : load_elem_dt(c, dt, p[0], ci);
                                                return {c.b.build_add(cur, v)};
                                            },
                                            [&]() -> std::vector<Value*> { return {cur}; });
                                    },
                                    [&]() -> std::vector<Value*> { return {cur}; })[0];
                            })};
                        },
                        [&]() -> std::vector<Value*> { return {r_acc}; });
                },
                [&]() -> std::vector<Value*> { return {r_acc}; })[0];
        });
        store_elem_dt(c, dt, p[2], i, acc);
    });
}

Value* bilinear_at(Ctx& c, Dtype dt, Value* x_ptr, Value* base, Value* H, Value* W, Value* fH, Value* fW,
                   Value* h, Value* w) {
    Value* oob = c.b.build_or(
        c.b.build_or(c.b.build_sle(h, c.f32(-1.0f)), c.b.build_sle(fH, h)),
        c.b.build_or(c.b.build_sle(w, c.f32(-1.0f)), c.b.build_sle(fW, w)));
    return emit_if_else_values(
        c, oob, {Type::f32()},
        [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; },
        [&]() -> std::vector<Value*> {
            Value* hf = c.b.build_floor_f32(h);
            Value* wf = c.b.build_floor_f32(w);
            Value* hl = c.b.build_fptosi_i32_f32(hf);
            Value* wl = c.b.build_fptosi_i32_f32(wf);
            Value* hh = c.b.build_add(hl, c.i32(1));
            Value* wh = c.b.build_add(wl, c.i32(1));
            Value* lh = c.b.build_sub(h, hf);
            Value* lw = c.b.build_sub(w, wf);
            Value* ih = c.b.build_sub(c.f32(1.0f), lh);
            Value* iw = c.b.build_sub(c.f32(1.0f), lw);

            auto corner = [&](Value* cy, Value* cx, Value* in_y, Value* in_x) {
                return emit_if_else_values(
                    c, c.b.build_and(in_y, in_x), {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* idx = c.b.build_add(base, c.b.build_add(c.b.build_mul(cy, W), cx));
                        return {load_elem_dt(c, dt, x_ptr, idx)};
                    },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            };
            Value* hl_ok = c.b.build_sge(hl, c.i32(0));
            Value* wl_ok = c.b.build_sge(wl, c.i32(0));
            Value* hh_ok = c.b.build_slt(hh, H);
            Value* wh_ok = c.b.build_slt(wh, W);
            Value* v1 = corner(hl, wl, hl_ok, wl_ok);
            Value* v2 = corner(hl, wh, hl_ok, wh_ok);
            Value* v3 = corner(hh, wl, hh_ok, wl_ok);
            Value* v4 = corner(hh, wh, hh_ok, wh_ok);
            Value* t1 = c.b.build_mul(c.b.build_mul(ih, iw), v1);
            Value* t2 = c.b.build_mul(c.b.build_mul(ih, lw), v2);
            Value* t3 = c.b.build_mul(c.b.build_mul(lh, iw), v3);
            Value* t4 = c.b.build_mul(c.b.build_mul(lh, lw), v4);
            return {c.b.build_add(c.b.build_add(t1, t2), c.b.build_add(t3, t4))};
        })[0];
}

void build_deform_col(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool has_offset, bool has_mask) {
    // p: X(0) off(1) mask(2) col(3) Cgi(4) H(5) W(6) Wo(7) HoWo(8) kH(9) kW(10)
    //    sh_sw(11) ph_pw(12) dh_dw(13) Cin_dg(14) ic_first(15) p0(16) np(17)
    Value* H = p[5];
    Value* W = p[6];
    Value* Wo = p[7];
    Value* HoWo = p[8];
    Value* kW = p[10];
    Value* kk = c.b.build_mul(p[9], kW);
    Value* sh = c.b.build_and(p[11], c.i32(0xffff));
    Value* sw = c.b.build_lshr(p[11], c.i32(16));
    Value* ph = c.b.build_and(p[12], c.i32(0xffff));
    Value* pw = c.b.build_lshr(p[12], c.i32(16));
    Value* dh = c.b.build_and(p[13], c.i32(0xffff));
    Value* dw = c.b.build_lshr(p[13], c.i32(16));
    Value* Cin_dg = p[14];
    Value* ic_first = p[15];
    Value* p0 = p[16];
    Value* np = p[17];
    Value* hw = c.b.build_mul(H, W);
    Value* fH = has_offset ? c.kb.i32_to_f32(H) : nullptr;
    Value* fW = has_offset ? c.kb.i32_to_f32(W) : nullptr;
    Value* total = c.b.build_mul(c.b.build_mul(p[4], kk), np);

    emit_vec_grid_stride(c, total, 1, kVisionBlock, [&](Value* i) {
        Value* r = c.b.build_udiv(i, np);
        Value* sp = c.b.build_add(p0, c.b.build_sub(i, c.b.build_mul(r, np)));
        Value* icl = c.b.build_udiv(r, kk);
        Value* tap = c.b.build_sub(r, c.b.build_mul(icl, kk));
        Value* kh = c.b.build_udiv(tap, kW);
        Value* kw = c.b.build_sub(tap, c.b.build_mul(kh, kW));
        Value* oh = c.b.build_udiv(sp, Wo);
        Value* ow = c.b.build_sub(sp, c.b.build_mul(oh, Wo));
        Value* ic = c.b.build_add(ic_first, icl);
        Value* yi = c.b.build_add(c.b.build_sub(c.b.build_mul(oh, sh), ph), c.b.build_mul(kh, dh));
        Value* xi = c.b.build_add(c.b.build_sub(c.b.build_mul(ow, sw), pw), c.b.build_mul(kw, dw));
        Value* x_base = c.b.build_mul(ic, hw);

        if (!has_offset) {
            Value* inb = c.b.build_and(c.b.build_and(c.b.build_sge(yi, c.i32(0)), c.b.build_slt(yi, H)),
                                       c.b.build_and(c.b.build_sge(xi, c.i32(0)), c.b.build_slt(xi, W)));
            Value* val = emit_if_else_values(
                c, inb, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* idx = c.b.build_add(x_base, c.b.build_add(c.b.build_mul(yi, W), xi));
                    return {load_elem_dt(c, dt, p[0], idx)};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            store_elem_dt(c, dt, p[3], i, val);
        } else {
            Value* og = c.b.build_udiv(ic, Cin_dg);
            Value* og_tap = c.b.build_add(c.b.build_mul(og, kk), tap);
            Value* oy_idx = c.b.build_add(c.b.build_mul(c.b.build_shl(og_tap, c.i32(1)), HoWo), sp);
            Value* ox_idx = c.b.build_add(oy_idx, HoWo);
            Value* oy = load_elem_dt(c, dt, p[1], oy_idx);
            Value* ox = load_elem_dt(c, dt, p[1], ox_idx);
            Value* m = has_mask ? load_elem_dt(c, dt, p[2], c.b.build_add(c.b.build_mul(og_tap, HoWo), sp))
                                : c.f32(1.0f);
            Value* yf = c.b.build_add(c.kb.i32_to_f32(yi), oy);
            Value* xf = c.b.build_add(c.kb.i32_to_f32(xi), ox);
            Value* bv = bilinear_at(c, dt, p[0], x_base, H, W, fH, fW, yf, xf);
            store_elem_dt(c, dt, p[3], i, c.b.build_mul(m, bv));
        }
    });
}

}  // namespace

brass::Function* build_vision_mir(brass::Module& mod, std::uint32_t op, Dtype dt, bool aux0, bool aux1) {
    std::vector<Type> params;
    switch (op) {
        case MI_UPFIRDN:
            params = {Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::f32()};
            break;
        case MI_MODW:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::f32()};
            break;
        case MI_MODW_BWD:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case MI_MODW_DS:
        case MI_MODW_DW:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        case MI_COL2IM2D:
            params = {Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32()};
            break;
        case MI_DEFORM_COL:
            params = {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
                      Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()};
            break;
        default:
            return nullptr;
    }

    std::string name = "mir_vision_" + std::to_string(op) + "_" + std::to_string(static_cast<int>(dt)) + "_" +
                       std::to_string(aux0 ? 1 : 0) + "_" + std::to_string(aux1 ? 1 : 0);
    Function* fn = mod.create_function(name, Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    switch (op) {
        case MI_UPFIRDN: build_upfirdn(c, p, dt, aux0); break;
        case MI_MODW: build_modw(c, p, dt, aux0, aux1); break;
        case MI_MODW_BWD: build_modw_bwd(c, p, dt); break;
        case MI_MODW_DS:
        case MI_MODW_DW: build_modw_ds_dw(c, p, dt, op == MI_MODW_DW); break;
        case MI_COL2IM2D: build_col2im2d(c, p, dt, aux0, aux1); break;
        case MI_DEFORM_COL: build_deform_col(c, p, dt, aux0, aux1); break;
        default: break;
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
