#include "mir_diffusion.h"

#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <functional>
#include <string>
#include <utility>

namespace brotensor::kernels {

std::pair<Value*, Value*> philox_words(Ctx& c, Value* k0, Value* k1, Value* sub64) {
    Value* c0 = c.b.build_trunc_i32(sub64);
    Value* c1 = c.b.build_trunc_i32(c.b.build_lshr(sub64, c.kb.const_i64(32)));
    Value* c2 = c.i32(0);
    Value* c3 = c.i32(0);
    Value* m0 = c.i32(static_cast<std::int32_t>(0xD2511F53u));
    Value* m1 = c.i32(static_cast<std::int32_t>(0xCD9E8D57u));
    Value* w0 = c.i32(static_cast<std::int32_t>(0x9E3779B9u));
    Value* w1 = c.i32(static_cast<std::int32_t>(0xBB67AE85u));
    for (int r = 0; r < 10; ++r) {
        Value* lo0 = c.b.build_mul(m0, c0);
        Value* hi0 = c.kb.mul_hi_u32(m0, c0);
        Value* lo1 = c.b.build_mul(m1, c2);
        Value* hi1 = c.kb.mul_hi_u32(m1, c2);
        Value* nc0 = c.b.build_xor(c.b.build_xor(hi1, c1), k0);
        Value* nc2 = c.b.build_xor(c.b.build_xor(hi0, c3), k1);
        c0 = nc0;
        c1 = lo1;
        c2 = nc2;
        c3 = lo0;
        if (r < 9) {
            k0 = c.b.build_add(k0, w0);
            k1 = c.b.build_add(k1, w1);
        }
    }
    return {c0, c1};
}

namespace {

Value* u32_to_u01(Ctx& c, Value* w) {
    Value* top24 = c.kb.u32_to_f32(c.b.build_lshr(w, c.i32(8)));
    return c.b.build_mul(top24, c.f32(1.0f / 16777216.0f));
}

Value* philox_normal(Ctx& c, Value* k0, Value* k1, Value* sub64) {
    auto [c0, c1] = philox_words(c, k0, k1, sub64);
    Value* u1 = c.b.build_sub(c.f32(1.0f), u32_to_u01(c, c0));
    Value* u2 = u32_to_u01(c, c1);
    Value* radius = emit_sqrt(c, c.b.build_mul(c.f32(-2.0f), emit_log_acc(c, u1)));
    Value* h = c.f32(6.28318548202514648f);
    Value* l = c.f32(-1.7484555314695172e-07f);
    Value* ph = c.b.build_mul(u2, h);
    Value* err = c.b.build_fma_f32(u2, h, c.b.build_neg(ph));
    Value* theta = c.b.build_add(ph, c.b.build_add(err, c.b.build_mul(u2, l)));
    auto [s, co] = emit_sincos_acc(c, theta);
    (void)s;
    return c.b.build_mul(radius, co);
}

void emit_ew_sampler(Ctx& c, Value* n, int lanes, const std::function<void(Value* base, int k)>& emit_k) {
    Value* gid = c.kb.global_tid_x();
    Value* stride = grid_stride_ew(c);
    if (lanes > 1) {
        Value* vec_end = c.b.build_and(n, c.i32(-lanes));
        c.kb.for_range(c.b.build_mul(gid, c.i32(lanes)), vec_end, c.b.build_mul(stride, c.i32(lanes)),
                       [&](Value* base) { emit_k(base, lanes); });
        c.kb.for_range(c.b.build_add(vec_end, gid), n, stride, [&](Value* idx) { emit_k(idx, 1); });
    } else {
        c.kb.for_range(gid, n, stride, [&](Value* idx) { emit_k(idx, 1); });
    }
}

}  // namespace

brass::Function* build_sampler_mir(brass::Module& mod, std::uint32_t mode, Dtype dt, Dtype dti, int lanes) {
    const std::string name = "mir_sampler_" + std::to_string(mode);
    auto* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::i32(), Type::f32(), Type::f32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *x = p[0], *e = p[1], *pbuf = p[2], *y = p[3], *z = p[4];
    Value *n = p[5], *dim = p[6], *half = p[7], *hw = p[8], *c_dim = p[9];
    Value *k0 = p[10], *k1 = p[11], *k2 = p[12], *k3 = p[13];
    const int es = elem_bytes(dt);

    if (mode == SMP_DDIM || mode == SMP_EULER || mode == SMP_DPMPP_2M) {
        emit_ew_sampler(c, n, lanes, [&](Value* base, int k) {
            auto xv = load_lanes(c, dt, k, ptr_at(c, x, base, es));
            auto ev = load_lanes(c, dt, k, ptr_at(c, e, base, es));
            std::vector<Value*> yv(static_cast<std::size_t>(k));
            std::vector<Value*> zv(static_cast<std::size_t>(k));
            if (mode == SMP_DDIM) {
                for (int i = 0; i < k; ++i) {
                    Value* x0 = c.b.build_mul(c.b.build_sub(xv[i], c.b.build_mul(k2, ev[i])), k1);
                    yv[i] = c.b.build_add(c.b.build_mul(k0, x0), c.b.build_mul(k3, ev[i]));
                }
            } else if (mode == SMP_EULER) {
                for (int i = 0; i < k; ++i) {
                    yv[i] = c.b.build_add(xv[i], c.b.build_mul(k0, ev[i]));
                }
            } else {
                auto pv = load_lanes(c, dt, k, ptr_at(c, pbuf, base, es));
                for (int i = 0; i < k; ++i) {
                    Value* x0 = c.b.build_sub(xv[i], c.b.build_mul(k0, ev[i]));
                    yv[i] = c.b.build_add(c.b.build_add(c.b.build_mul(k1, xv[i]), c.b.build_mul(k2, x0)),
                                          c.b.build_mul(k3, pv[i]));
                    zv[i] = x0;
                }
                store_lanes(c, dt, k, ptr_at(c, z, base, es), zv);
            }
            store_lanes(c, dt, k, ptr_at(c, y, base, es), yv);
        });
    } else if (mode == SMP_TIMESTEP_EMB) {
        Value* two_half = c.b.build_mul(half, c.i32(2));
        Value* safe_half = c.kb.i32_to_f32(c.b.build_select(c.b.build_eq(half, c.i32(0)), c.i32(1), half));
        Value* neg_k0 = c.b.build_neg(k0);
        c.kb.for_range(c.kb.global_tid_x(), n, grid_stride_ew(c), [&](Value* i) {
            Value* r = c.b.build_udiv(i, dim);
            Value* j = c.b.build_umod(i, dim);
            Value* lt_h = c.b.build_ult(j, half);
            Value* k = c.b.build_select(lt_h, j, c.b.build_sub(j, half));
            Value* freq = emit_exp_acc(c, emit_div_rn(c, c.b.build_mul(neg_k0, c.kb.u32_to_f32(k)), safe_half));
            Value* t = load_elem_dt(c, dti, x, r);
            auto [s, co] = emit_sincos_acc(c, c.b.build_mul(t, freq));
            Value* v = c.b.build_select(c.b.build_ult(j, two_half), c.b.build_select(lt_h, co, s), c.f32(0.0f));
            store_elem_dt(c, dt, y, i, v);
        });
    } else if (mode == SMP_IMAGE_NORM) {
        c.kb.for_range(c.kb.global_tid_x(), n, grid_stride_ew(c), [&](Value* i) {
            Value* ch = c.b.build_umod(c.b.build_udiv(i, hw), c_dim);
            Value* m = c.kb.load_f32(ptr_at(c, e, ch, 4));
            Value* inv = emit_div_rn(c, c.f32(1.0f), c.kb.load_f32(ptr_at(c, pbuf, ch, 4)));
            Value* xv = load_elem_dt(c, dt, x, i);
            store_elem_dt(c, dt, y, i, c.b.build_mul(c.b.build_sub(xv, m), inv));
        });
    } else {
        Value* hc = c.b.build_mul(hw, c_dim);
        c.kb.for_range(c.kb.global_tid_x(), n, grid_stride_ew(c), [&](Value* i) {
            Value* img = c.b.build_udiv(i, hc);
            Value* rem = c.b.build_umod(i, hc);
            Value* ch = c.b.build_udiv(rem, hw);
            Value* px = c.b.build_umod(rem, hw);
            Value* src_idx = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, hw), px), c_dim), ch);
            Value* vf = c.kb.u32_to_f32(c.kb.load_u8(c.b.build_add(x, c.b.build_zext_i64(src_idx))));
            Value* res = c.b.build_add(c.b.build_mul(vf, k0), k1);
            store_elem_dt(c, dt, y, i, res);
        });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_philox_mir(brass::Module& mod, std::uint32_t mode) {
    auto* fn = mod.create_function(
        "mir_philox_" + std::to_string(mode), Type::void_type(),
        {Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::i32(), Type::f32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *y = p[0], *k_lo = p[1], *k_hi = p[2], *c_lo = p[3], *c_hi = p[4];
    Value *n_lo = p[5], *n_hi = p[6], *prob = p[7], *lo = p[8], *hi = p[9];

    auto join_u64 = [&](Value* lo32, Value* hi32) -> Value* {
        return c.b.build_or(c.b.build_zext_i64(lo32), c.b.build_shl(c.b.build_zext_i64(hi32), c.kb.const_i64(32)));
    };
    Value* ctr_base = join_u64(c_lo, c_hi);
    Value* n64 = join_u64(n_lo, n_hi);
    Value* start64 = c.b.build_zext_i64(c.kb.global_tid_x());
    Value* stride64 = c.b.build_zext_i64(grid_stride_ew(c));

    c.kb.for_range(start64, n64, stride64, [&](Value* idx64) {
        Value* sub = c.b.build_add(ctr_base, idx64);
        Value* v = nullptr;
        if (mode == RNG_NORMAL) {
            v = philox_normal(c, k_lo, k_hi, sub);
        } else if (mode == RNG_UNIFORM) {
            v = u32_to_u01(c, philox_words(c, k_lo, k_hi, sub).first);
        } else if (mode == RNG_BERNOULLI) {
            Value* u = u32_to_u01(c, philox_words(c, k_lo, k_hi, sub).first);
            v = c.b.build_select(c.b.build_slt(u, prob), c.f32(1.0f), c.f32(0.0f));
        } else {
            auto st = c.kb.for_range_reduce_n(
                c.i32(0), c.i32(64), c.i32(1), {c.i32(0), c.f32(0.0f)},
                [&](Value* r_cur, const std::vector<Value*>& acc) -> std::vector<Value*> {
                    return emit_if_else_values(
                        c, c.b.build_eq(acc[0], c.i32(0)), {Type::i32(), Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* sub_r = c.b.build_add(sub, c.b.build_mul(c.b.build_zext_i64(r_cur), n64));
                            Value* draw = philox_normal(c, k_lo, k_hi, sub_r);
                            Value* ok = c.b.build_and(c.b.build_sge(draw, lo), c.b.build_sle(draw, hi));
                            return {c.b.build_select(ok, c.i32(1), c.i32(0)), draw};
                        },
                        [&]() -> std::vector<Value*> { return acc; });
                });
            v = c.kb.clamp_f32(st[1], lo, hi);
        }
        Value* byte_off = c.b.build_shl(idx64, c.kb.const_i64(2));
        c.kb.store_f32(c.b.build_add(y, byte_off), v);
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
