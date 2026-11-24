#include "mir_resample.h"

#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <limits>
#include <string>

namespace brotensor::kernels {

FloorDiv emit_floor_div(Ctx& c, Value* num_i64, Value* den_i64) {
    Value* q = c.b.build_sdiv(num_i64, den_i64);
    Value* r = c.b.build_sub(num_i64, c.b.build_mul(q, den_i64));
    Value* neg = c.b.build_slt(r, c.kb.const_i64(0));
    Value* q_adj = c.b.build_select(neg, c.b.build_sub(q, c.kb.const_i64(1)), q);
    Value* rem = c.b.build_select(neg, c.b.build_add(r, den_i64), r);
    return {c.b.build_trunc_i32(q_adj), rem};
}

std::pair<Value*, Value*> emit_src_rational(Ctx& c, Value* o_i32, Value* in_dim, Value* out_dim, bool align_corners) {
    if (align_corners) {
        Value* gt1 = c.b.build_sgt(out_dim, c.i32(1));
        Value* num_gt = c.b.build_mul(c.b.build_zext_i64(o_i32),
                                      c.b.build_zext_i64(c.b.build_sub(in_dim, c.i32(1))));
        Value* den_gt = c.b.build_zext_i64(c.b.build_sub(out_dim, c.i32(1)));
        return {c.b.build_select(gt1, num_gt, c.kb.const_i64(0)),
                c.b.build_select(gt1, den_gt, c.kb.const_i64(1))};
    }
    Value* num = c.b.build_sub(
        c.b.build_mul(c.b.build_zext_i64(c.b.build_add(c.b.build_mul(o_i32, c.i32(2)), c.i32(1))),
                      c.b.build_zext_i64(in_dim)),
        c.b.build_zext_i64(out_dim));
    Value* den = c.b.build_mul(c.kb.const_i64(2), c.b.build_zext_i64(out_dim));
    return {num, den};
}

namespace {

constexpr int kResampleCpt = 16;

struct AxisTaps {
    std::array<Value*, 4> idx{};
    std::array<Value*, 4> w{};
};

AxisTaps emit_axis_taps(Ctx& c, Value* o, Value* in_dim, Value* out_dim, bool align_corners, int imode) {
    AxisTaps taps;
    auto [num, den] = emit_src_rational(c, o, in_dim, out_dim, align_corners);
    auto [i0, rem] = emit_floor_div(c, num, den);
    Value* last = c.b.build_sub(in_dim, c.i32(1));
    Value* zero = c.i32(0);

    if (imode == 0) {
        Value* twice = c.b.build_mul(rem, c.kb.const_i64(2));
        Value* gt = c.b.build_sgt(twice, den);
        Value* lt = c.b.build_slt(twice, den);
        Value* tie = c.b.build_add(i0, c.b.build_and(i0, c.i32(1)));
        Value* r = c.b.build_select(gt, c.b.build_add(i0, c.i32(1)), c.b.build_select(lt, i0, tie));
        taps.idx[0] = clamp_i32(c, r, zero, last);
        return taps;
    }

    Value* f = c.b.build_sdiv(c.b.build_sitofp_f32_i64(rem), c.b.build_sitofp_f32_i64(den));
    if (imode == 1) {
        taps.idx[0] = clamp_i32(c, i0, zero, last);
        taps.idx[1] = clamp_i32(c, c.b.build_add(i0, c.i32(1)), zero, last);
        taps.w[0] = f;
        return taps;
    }

    const float a = (imode == 3) ? -0.75f : -0.5f;
    for (int k = 0; k < 4; ++k) {
        Value* t = c.kb.fabs(c.b.build_sub(f, c.f32(static_cast<float>(k - 1))));
        Value* t2 = c.b.build_mul(t, t);
        Value* poly1 = c.b.build_add(
            c.b.build_mul(c.b.build_sub(c.b.build_mul(c.f32(a + 2.0f), t), c.f32(a + 3.0f)), t2),
            c.f32(1.0f));
        Value* poly2 = c.b.build_sub(
            c.b.build_mul(
                c.b.build_add(c.b.build_mul(c.b.build_sub(c.b.build_mul(c.f32(a), t), c.f32(5.0f * a)), t),
                              c.f32(8.0f * a)),
                t),
            c.f32(4.0f * a));
        taps.w[k] = c.b.build_select(
            c.b.build_slt(t, c.f32(1.0f)), poly1,
            c.b.build_select(c.b.build_slt(t, c.f32(2.0f)), poly2, c.f32(0.0f)));
        taps.idx[k] = clamp_i32(c, c.b.build_add(i0, c.i32(k - 1)), zero, last);
    }
    return taps;
}

Value* adaptive_lo(Ctx& c, Value* o, Value* in_dim, Value* out_dim) {
    return c.b.build_udiv(c.b.build_mul(o, in_dim), out_dim);
}

Value* adaptive_hi(Ctx& c, Value* o, Value* in_dim, Value* out_dim) {
    Value* num = c.b.build_add(c.b.build_mul(c.b.build_add(o, c.i32(1)), in_dim), c.b.build_sub(out_dim, c.i32(1)));
    return c.b.build_udiv(num, out_dim);
}

}  // namespace

brass::Function* build_resample_mir(brass::Module& mod, std::uint32_t mode, Dtype dt,
                                    bool align_corners, int imode) {
    const std::string name = "mir_resample_" + std::to_string(mode) + "_dt" +
                             std::to_string(static_cast<int>(dt)) + "_a" + std::to_string(align_corners ? 1 : 0) +
                             "_i" + std::to_string(imode);
    auto* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *x = p[0], *y = p[1], *aux = p[2];
    Value *c_dim = p[3], *hi = p[4], *wi = p[5], *ho = p[6], *wo = p[7];
    Value *kh = p[8], *kw = p[9], *sh = p[10], *sw = p[11], *ph = p[12], *pw = p[13];

    Value* plane_o = c.b.build_mul(ho, wo);
    Value* plane_i = c.b.build_mul(hi, wi);
    Value* img = c.kb.ctaid_z();
    Value* c0 = c.b.build_mul(c.kb.ctaid_y(), c.i32(kResampleCpt));
    Value* c1 = c.b.build_select(c.b.build_ult(c.b.build_add(c0, c.i32(kResampleCpt)), c_dim),
                                 c.b.build_add(c0, c.i32(kResampleCpt)), c_dim);

    c.kb.for_range(c.kb.global_tid_x(), plane_o, grid_stride_ew(c), [&](Value* pos) {
        Value* oh = c.b.build_udiv(pos, wo);
        Value* ow = c.b.build_umod(pos, wo);

        if (mode == RS_INTERP) {
            AxisTaps ty = emit_axis_taps(c, oh, hi, ho, align_corners, imode);
            AxisTaps tx = emit_axis_taps(c, ow, wi, wo, align_corners, imode);
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* xb = c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_i);
                Value* v = nullptr;
                if (imode == 0) {
                    v = load_elem_dt(c, dt, x,
                                     c.b.build_add(c.b.build_add(xb, c.b.build_mul(ty.idx[0], wi)), tx.idx[0]));
                } else if (imode == 1) {
                    Value* row0 = c.b.build_add(xb, c.b.build_mul(ty.idx[0], wi));
                    Value* row1 = c.b.build_add(xb, c.b.build_mul(ty.idx[1], wi));
                    Value* v00 = load_elem_dt(c, dt, x, c.b.build_add(row0, tx.idx[0]));
                    Value* v01 = load_elem_dt(c, dt, x, c.b.build_add(row0, tx.idx[1]));
                    Value* v10 = load_elem_dt(c, dt, x, c.b.build_add(row1, tx.idx[0]));
                    Value* v11 = load_elem_dt(c, dt, x, c.b.build_add(row1, tx.idx[1]));
                    Value* top = c.b.build_add(v00, c.b.build_mul(c.b.build_sub(v01, v00), tx.w[0]));
                    Value* bot = c.b.build_add(v10, c.b.build_mul(c.b.build_sub(v11, v10), tx.w[0]));
                    v = c.b.build_add(top, c.b.build_mul(c.b.build_sub(bot, top), ty.w[0]));
                } else {
                    v = c.f32(0.0f);
                    for (int j = 0; j < 4; ++j) {
                        Value* row_base = c.b.build_add(xb, c.b.build_mul(ty.idx[j], wi));
                        Value* row = c.f32(0.0f);
                        for (int i = 0; i < 4; ++i) {
                            Value* tap = load_elem_dt(c, dt, x, c.b.build_add(row_base, tx.idx[i]));
                            row = c.b.build_add(row, c.b.build_mul(tx.w[i], tap));
                        }
                        v = c.b.build_add(v, c.b.build_mul(ty.w[j], row));
                    }
                }
                Value* o = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_o), pos);
                store_elem_dt(c, dt, y, o, v);
            });
        } else if (mode == RS_DOWN_AVG2 || mode == RS_UP_NEAREST2_BWD) {
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* b = c.b.build_add(
                    c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_i),
                                  c.b.build_mul(c.b.build_mul(oh, c.i32(2)), wi)),
                    c.b.build_mul(ow, c.i32(2)));
                Value* v0 = load_elem_dt(c, dt, x, b);
                Value* v1 = load_elem_dt(c, dt, x, c.b.build_add(b, c.i32(1)));
                Value* v2 = load_elem_dt(c, dt, x, c.b.build_add(b, wi));
                Value* v3 = load_elem_dt(c, dt, x, c.b.build_add(c.b.build_add(b, wi), c.i32(1)));
                Value* s = c.b.build_add(c.b.build_add(c.b.build_add(v0, v1), v2), v3);
                if (mode == RS_DOWN_AVG2) s = c.b.build_mul(s, c.f32(0.25f));
                Value* o = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_o), pos);
                store_elem_dt(c, dt, y, o, s);
            });
        } else if (mode == RS_DOWN_AVG2_BWD) {
            Value* ip = c.b.build_add(c.b.build_mul(c.b.build_lshr(oh, c.i32(1)), wi), c.b.build_lshr(ow, c.i32(1)));
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* src = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_i), ip);
                Value* g = c.b.build_mul(load_elem_dt(c, dt, x, src), c.f32(0.25f));
                Value* o = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_o), pos);
                store_elem_dt(c, dt, y, o, g);
            });
        } else if (mode == RS_ADAPTIVE_AVG) {
            Value* h0 = adaptive_lo(c, oh, hi, ho);
            Value* h1 = adaptive_hi(c, oh, hi, ho);
            Value* w0 = adaptive_lo(c, ow, wi, wo);
            Value* w1 = adaptive_hi(c, ow, wi, wo);
            Value* area = c.b.build_mul(c.b.build_sub(h1, h0), c.b.build_sub(w1, w0));
            Value* inv_area = c.b.build_sdiv(c.f32(1.0f), c.kb.i32_to_f32(area));
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* xb = c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_i);
                Value* sum = c.kb.for_range_reduce(h0, h1, c.i32(1), c.f32(0.0f), [&](Value* h, Value* h_acc) {
                    Value* row_b = c.b.build_add(xb, c.b.build_mul(h, wi));
                    return c.kb.for_range_reduce(w0, w1, c.i32(1), h_acc, [&](Value* w, Value* w_acc) {
                        return c.b.build_add(w_acc, load_elem_dt(c, dt, x, c.b.build_add(row_b, w)));
                    });
                });
                Value* o = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_o), pos);
                store_elem_dt(c, dt, y, o, c.b.build_mul(sum, inv_area));
            });
        } else if (mode == RS_MAXPOOL) {
            Value* hb = c.b.build_sub(c.b.build_mul(oh, sh), ph);
            Value* wb = c.b.build_sub(c.b.build_mul(ow, sw), pw);
            Value* neg_inf = c.f32(-std::numeric_limits<float>::infinity());
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* xb = c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_i);
                auto res = c.kb.for_range_reduce_n(
                    c.i32(0), kh, c.i32(1), {neg_inf, c.i32(-1)},
                    [&](Value* a, const std::vector<Value*>& h_acc) -> std::vector<Value*> {
                        Value* ih = c.b.build_add(hb, a);
                        Value* h_ok = c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, hi));
                        return c.kb.for_range_reduce_n(
                            c.i32(0), kw, c.i32(1), h_acc,
                            [&](Value* b_idx, const std::vector<Value*>& w_acc) -> std::vector<Value*> {
                                Value* iw = c.b.build_add(wb, b_idx);
                                Value* w_ok = c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, wi));
                                return emit_if_else_values(
                                    c, c.b.build_and(h_ok, w_ok), {Type::f32(), Type::i32()},
                                    [&]() -> std::vector<Value*> {
                                        Value* lin = c.b.build_add(c.b.build_mul(ih, wi), iw);
                                        Value* v = load_elem_dt(c, dt, x, c.b.build_add(xb, lin));
                                        Value* better = c.b.build_sgt(v, w_acc[0]);
                                        return {c.b.build_select(better, v, w_acc[0]),
                                                c.b.build_select(better, lin, w_acc[1])};
                                    },
                                    [&]() -> std::vector<Value*> { return {w_acc[0], w_acc[1]}; });
                            });
                    });
                Value* o = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_o), pos);
                store_elem_dt(c, dt, y, o, res[0]);
                c.kb.store_i32(ptr_at(c, aux, o, 4), res[1]);
            });
        } else {
            Value* s = kh;
            Value* kk = c.b.build_mul(s, s);
            Value* py = c.b.build_udiv(oh, s);
            Value* sy = c.b.build_umod(oh, s);
            Value* px = c.b.build_udiv(ow, s);
            Value* sx = c.b.build_umod(ow, s);
            Value* pix = c.b.build_add(c.b.build_mul(py, wi), px);
            Value* sub = c.b.build_add(c.b.build_mul(sy, s), sx);
            Value* mb = c.b.build_mul(c.b.build_mul(img, c.b.build_mul(c.i32(9), kk)), plane_i);

            std::array<Value*, 9> m{};
            Value* mx = c.f32(-std::numeric_limits<float>::infinity());
            for (int j = 0; j < 9; ++j) {
                Value* midx = c.b.build_add(
                    c.b.build_add(mb, c.b.build_mul(c.b.build_add(c.b.build_mul(c.i32(j), kk), sub), plane_i)), pix);
                m[j] = load_elem_dt(c, dt, aux, midx);
                mx = c.kb.fmax(mx, m[j]);
            }
            Value* sum = c.f32(0.0f);
            for (int j = 0; j < 9; ++j) {
                m[j] = emit_exp(c, c.b.build_sub(m[j], mx));
                sum = c.b.build_add(sum, m[j]);
            }
            Value* inv = c.b.build_sdiv(c.f32(1.0f), sum);
            std::array<Value*, 9> nb{};
            Value* h_last = c.b.build_sub(hi, c.i32(1));
            Value* w_last = c.b.build_sub(wi, c.i32(1));
            for (int j = 0; j < 9; ++j) {
                Value* ny = clamp_i32(c, c.b.build_add(py, c.i32(j / 3 - 1)), c.i32(0), h_last);
                Value* nx = clamp_i32(c, c.b.build_add(px, c.i32(j % 3 - 1)), c.i32(0), w_last);
                nb[j] = c.b.build_add(c.b.build_mul(ny, wi), nx);
                m[j] = c.b.build_mul(m[j], inv);
            }
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* xb = c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_i);
                Value* acc = c.f32(0.0f);
                for (int j = 0; j < 9; ++j) {
                    acc = c.b.build_add(acc, c.b.build_mul(m[j], load_elem_dt(c, dt, x, c.b.build_add(xb, nb[j]))));
                }
                Value* o = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, c_dim), ch), plane_o), pos);
                store_elem_dt(c, dt, y, o, acc);
            });
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
