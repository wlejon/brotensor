#include "mir_resample_bwd.h"

#include "mir_resample.h"
#include "../vulkan/shaders/op_codes.h"

#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* interp_weight(Ctx& c, Value* o, Value* i, Value* in_dim, Value* out_dim, std::uint32_t imode) {
    auto [num, den] = emit_src_rational(c, o, in_dim, out_dim, false);
    auto [i0, rem] = emit_floor_div(c, num, den);
    Value* last = c.b.build_sub(in_dim, c.i32(1));
    Value* zero = c.i32(0);
    if (imode == 0) {
        Value* twice = c.b.build_mul(rem, c.kb.const_i64(2));
        Value* gt = c.b.build_sgt(twice, den);
        Value* lt = c.b.build_slt(twice, den);
        Value* tie = c.b.build_add(i0, c.b.build_and(i0, c.i32(1)));
        Value* r = c.b.build_select(gt, c.b.build_add(i0, c.i32(1)), c.b.build_select(lt, i0, tie));
        Value* hit = c.b.build_eq(clamp_i32(c, r, zero, last), i);
        return c.b.build_select(hit, c.f32(1.0f), c.f32(0.0f));
    }
    Value* f = c.b.build_sdiv(c.b.build_sitofp_f32_i64(rem), c.b.build_sitofp_f32_i64(den));
    Value* w0 = c.b.build_select(c.b.build_eq(clamp_i32(c, i0, zero, last), i), c.b.build_sub(c.f32(1.0f), f),
                                 c.f32(0.0f));
    Value* w1 = c.b.build_select(c.b.build_eq(clamp_i32(c, c.b.build_add(i0, c.i32(1)), zero, last), i), f,
                                 c.f32(0.0f));
    return c.b.build_add(w0, w1);
}

std::pair<Value*, Value*> interp_range(Ctx& c, Value* i, Value* in_dim, Value* out_dim) {
    Value* in2 = c.b.build_mul(c.kb.const_i64(2), c.b.build_zext_i64(in_dim));
    Value* in64 = c.b.build_zext_i64(in_dim);
    Value* out64 = c.b.build_zext_i64(out_dim);
    Value* i64_v = c.b.build_sext_i64(i);
    Value* lo_num = c.b.build_sub(
        c.b.build_mul(c.b.build_sub(c.b.build_mul(i64_v, c.kb.const_i64(2)), c.kb.const_i64(2)), out64), in64);
    Value* hi_num = c.b.build_sub(
        c.b.build_mul(c.b.build_add(c.b.build_mul(i64_v, c.kb.const_i64(2)), c.kb.const_i64(3)), out64), in64);
    Value* lo_raw = c.b.build_sub(c.b.build_trunc_i32(c.b.build_sdiv(lo_num, in2)), c.i32(1));
    Value* hi_raw = c.b.build_add(c.b.build_trunc_i32(c.b.build_sdiv(hi_num, in2)), c.i32(1));
    lo_raw = c.b.build_select(c.b.build_eq(i, c.i32(0)), c.i32(0), lo_raw);
    Value* last_out = c.b.build_sub(out_dim, c.i32(1));
    hi_raw = c.b.build_select(c.b.build_eq(i, c.b.build_sub(in_dim, c.i32(1))), last_out, hi_raw);
    return {c.b.build_select(c.b.build_sgt(lo_raw, c.i32(0)), lo_raw, c.i32(0)),
            c.b.build_select(c.b.build_slt(hi_raw, last_out), hi_raw, last_out)};
}

std::pair<Value*, Value*> adaptive_win(Ctx& c, Value* o, Value* in_dim, Value* out_dim) {
    Value* lo = c.b.build_udiv(c.b.build_mul(o, in_dim), out_dim);
    Value* hi_num = c.b.build_add(c.b.build_mul(c.b.build_add(o, c.i32(1)), in_dim),
                                  c.b.build_sub(out_dim, c.i32(1)));
    return {lo, c.b.build_udiv(hi_num, out_dim)};
}

std::vector<std::pair<Value*, Value*>> pad_runs(Ctx& c, Value* i, Value* len, Value* pad_lo, Value* lp,
                                                std::uint32_t pad_mode) {
    std::vector<std::pair<Value*, Value*>> runs;
    Value* interior = c.b.build_add(i, pad_lo);
    runs.emplace_back(interior, c.b.build_add(interior, c.i32(1)));
    if (pad_mode == 2) {
        Value* left_ok = c.b.build_and(c.b.build_eq(i, c.i32(0)), c.b.build_sgt(pad_lo, c.i32(0)));
        runs.emplace_back(c.i32(0), c.b.build_select(left_ok, pad_lo, c.i32(0)));
        Value* end_i = c.b.build_add(pad_lo, len);
        Value* right_ok = c.b.build_and(c.b.build_eq(i, c.b.build_sub(len, c.i32(1))), c.b.build_slt(end_i, lp));
        runs.emplace_back(end_i, c.b.build_select(right_ok, lp, end_i));
    } else if (pad_mode == 1) {
        Value* len_gt1 = c.b.build_sgt(len, c.i32(1));
        Value* left = c.b.build_sub(pad_lo, i);
        Value* left_ok = c.b.build_and(
            len_gt1, c.b.build_and(c.b.build_sge(i, c.i32(1)), c.b.build_sge(left, c.i32(0))));
        runs.emplace_back(left, c.b.build_select(left_ok, c.b.build_add(left, c.i32(1)), left));
        Value* right = c.b.build_sub(c.b.build_add(pad_lo, c.b.build_mul(c.b.build_sub(len, c.i32(1)), c.i32(2))), i);
        Value* right_ok = c.b.build_and(
            len_gt1,
            c.b.build_and(c.b.build_sle(i, c.b.build_sub(len, c.i32(2))), c.b.build_slt(right, lp)));
        runs.emplace_back(right, c.b.build_select(right_ok, c.b.build_add(right, c.i32(1)), right));
    }
    return runs;
}

}  // namespace

brass::Function* build_resample_bwd_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, std::uint32_t imode) {
    const std::string name = "mir_resample_bwd_" + std::to_string(mode) + "_dt" +
                             std::to_string(static_cast<int>(dt)) + "_i" + std::to_string(imode);
    auto* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *dy = p[0], *dx = p[1], *aux = p[2];
    Value *planes = p[3], *hi = p[4], *wi = p[5], *ho = p[6], *wo = p[7], *pt = p[8], *pl = p[9];

    Value* plane_i = c.b.build_mul(hi, wi);
    Value* plane_o = c.b.build_mul(ho, wo);

    if (mode == RB_MAXPOOL) {
        c.kb.for_range(c.kb.global_tid_x(), planes, grid_stride_ew(c), [&](Value* pl_idx) {
            Value* xb = c.b.build_mul(pl_idx, plane_i);
            Value* yb = c.b.build_mul(pl_idx, plane_o);
            c.kb.for_range(c.i32(0), plane_i, c.i32(1), [&](Value* i) {
                c.kb.store_f32(ptr_at(c, dx, c.b.build_add(xb, i), 4), c.f32(0.0f));
            });
            c.kb.for_range(c.i32(0), plane_o, c.i32(1), [&](Value* o) {
                Value* y_idx = c.b.build_add(yb, o);
                Value* idx = c.kb.load_i32(ptr_at(c, aux, y_idx, 4));
                Value* valid = c.b.build_and(c.b.build_sge(idx, c.i32(0)), c.b.build_slt(idx, plane_i));
                c.kb.if_then(valid, [&] {
                    Value* dst_ptr = ptr_at(c, dx, c.b.build_add(xb, idx), 4);
                    Value* g = load_elem_dt(c, dt, dy, y_idx);
                    c.kb.store_f32(dst_ptr, c.b.build_add(c.kb.load_f32(dst_ptr), g));
                });
            });
        });
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    Value* total = c.b.build_mul(planes, plane_i);
    c.kb.for_range(c.kb.global_tid_x(), total, grid_stride_ew(c), [&](Value* e) {
        Value* pl_idx = c.b.build_udiv(e, plane_i);
        Value* pos = c.b.build_umod(e, plane_i);
        Value* h = c.b.build_udiv(pos, wi);
        Value* w = c.b.build_umod(pos, wi);
        Value* yb = c.b.build_mul(pl_idx, plane_o);
        Value* acc = c.f32(0.0f);

        if (mode == RB_INTERP) {
            auto [ylo, yhi] = interp_range(c, h, hi, ho);
            auto [xlo, xhi] = interp_range(c, w, wi, wo);
            acc = c.kb.for_range_reduce(
                ylo, c.b.build_add(yhi, c.i32(1)), c.i32(1), c.f32(0.0f), [&](Value* oh, Value* h_acc) {
                    Value* wy = interp_weight(c, oh, h, hi, ho, imode);
                    Value* row_b = c.b.build_add(yb, c.b.build_mul(oh, wo));
                    return emit_if_else_values(
                        c, c.b.build_ne(wy, c.f32(0.0f)), {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* row = c.kb.for_range_reduce(
                                xlo, c.b.build_add(xhi, c.i32(1)), c.i32(1), c.f32(0.0f),
                                [&](Value* ow, Value* w_acc) {
                                    Value* wx = interp_weight(c, ow, w, wi, wo, imode);
                                    return emit_if_else_values(
                                        c, c.b.build_ne(wx, c.f32(0.0f)), {Type::f32()},
                                        [&]() -> std::vector<Value*> {
                                            Value* term = c.b.build_mul(
                                                wx, load_elem_dt(c, dt, dy, c.b.build_add(row_b, ow)));
                                            return {c.b.build_add(w_acc, term)};
                                        },
                                        [&]() -> std::vector<Value*> { return {w_acc}; })[0];
                                });
                            return {c.b.build_add(h_acc, c.b.build_mul(wy, row))};
                        },
                        [&]() -> std::vector<Value*> { return {h_acc}; })[0];
                });
        } else if (mode == RB_ADAPTIVE) {
            auto range1d = [&](Value* coord, Value* in_d, Value* out_d) -> std::pair<Value*, Value*> {
                Value* l = c.b.build_sub(c.b.build_udiv(c.b.build_mul(coord, out_d), in_d), c.i32(1));
                Value* r = c.b.build_udiv(
                    c.b.build_add(c.b.build_mul(c.b.build_add(coord, c.i32(1)), out_d), c.b.build_sub(in_d, c.i32(1))),
                    in_d);
                return {c.b.build_select(c.b.build_sgt(l, c.i32(0)), l, c.i32(0)),
                        c.b.build_select(c.b.build_slt(c.b.build_add(r, c.i32(1)), out_d),
                                         c.b.build_add(r, c.i32(1)), out_d)};
            };
            auto [ylo, yhi_end] = range1d(h, hi, ho);
            auto [xlo, xhi_end] = range1d(w, wi, wo);
            acc = c.kb.for_range_reduce(ylo, yhi_end, c.i32(1), c.f32(0.0f), [&](Value* oh, Value* h_acc) {
                auto [h0, h1] = adaptive_win(c, oh, hi, ho);
                Value* h_in = c.b.build_and(c.b.build_sge(h, h0), c.b.build_slt(h, h1));
                return emit_if_else_values(
                    c, h_in, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* dh = c.b.build_sub(h1, h0);
                        Value* row_b = c.b.build_add(yb, c.b.build_mul(oh, wo));
                        Value* next_h = c.kb.for_range_reduce(
                            xlo, xhi_end, c.i32(1), h_acc, [&](Value* ow, Value* w_acc) {
                                auto [w0, w1] = adaptive_win(c, ow, wi, wo);
                                Value* w_in = c.b.build_and(c.b.build_sge(w, w0), c.b.build_slt(w, w1));
                                return emit_if_else_values(
                                    c, w_in, {Type::f32()},
                                    [&]() -> std::vector<Value*> {
                                        Value* area = c.kb.i32_to_f32(c.b.build_mul(dh, c.b.build_sub(w1, w0)));
                                        Value* term = c.b.build_sdiv(
                                            load_elem_dt(c, dt, dy, c.b.build_add(row_b, ow)), area);
                                        return {c.b.build_add(w_acc, term)};
                                    },
                                    [&]() -> std::vector<Value*> { return {w_acc}; })[0];
                            });
                        return {next_h};
                    },
                    [&]() -> std::vector<Value*> { return {h_acc}; })[0];
            });
        } else {
            auto ry = pad_runs(c, h, hi, pt, ho, imode);
            auto rx = pad_runs(c, w, wi, pl, wo, imode);
            for (const auto& yr : ry) {
                acc = c.kb.for_range_reduce(yr.first, yr.second, c.i32(1), acc, [&](Value* oy, Value* y_acc) {
                    Value* row_b = c.b.build_add(yb, c.b.build_mul(oy, wo));
                    Value* cur = y_acc;
                    for (const auto& xr : rx) {
                        cur = c.kb.for_range_reduce(xr.first, xr.second, c.i32(1), cur, [&](Value* ox, Value* x_acc) {
                            return c.b.build_add(x_acc, load_elem_dt(c, dt, dy, c.b.build_add(row_b, ox)));
                        });
                    }
                    return cur;
                });
            }
        }
        store_elem_dt(c, dt, dx, e, acc);
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
