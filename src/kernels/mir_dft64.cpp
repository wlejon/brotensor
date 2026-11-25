#include "mir_dft64.h"

#include "mir_common.h"
#include "../vulkan/shaders/op_codes.h"

#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kMaxL = 2048;
constexpr std::uint32_t kMaxIn = 2050;

Value* f64(Ctx& c, double v) { return c.b.build_fconst_f64(v); }

Value* shared_alloc_f64(Ctx& c, std::uint32_t count) {
    return c.b.build_call("ptx_shared_alloc_f64", Type::ptr(), {c.i32(static_cast<int>(count))});
}

Value* shared_load_f64_idx(Ctx& c, Value* smem, Value* idx) {
    return c.b.build_call("ptx_shared_load_f64_indexed", Type::f64(), {smem, idx});
}

void shared_store_f64_idx(Ctx& c, Value* smem, Value* idx, Value* val) {
    c.b.build_call("ptx_shared_store_f64_indexed", Type::void_type(), {smem, idx, val});
}

Value* clamp_tw(Ctx& c, Value* idx) { return c.b.build_and(idx, c.i32(static_cast<int>(kMaxL - 1u))); }

Value* clamp_in(Ctx& c, Value* idx) {
    return c.b.build_select(c.b.build_ult(idx, c.i32(static_cast<int>(kMaxIn))), idx, c.i32(0));
}

// sin / cos of x in [0, pi / 4] to double rounding: Taylor through x^21 / x^20.
std::pair<Value*, Value*> sincos64(Ctx& c, Value* x) {
    Value* z = c.b.build_mul(x, x);
    Value* s = f64(c, 1.0 / 51090942171709440000.0);
    s = c.b.build_fma_f64(s, z, f64(c, -1.0 / 121645100408832000.0));
    s = c.b.build_fma_f64(s, z, f64(c, 1.0 / 355687428096000.0));
    s = c.b.build_fma_f64(s, z, f64(c, -1.0 / 1307674368000.0));
    s = c.b.build_fma_f64(s, z, f64(c, 1.0 / 6227020800.0));
    s = c.b.build_fma_f64(s, z, f64(c, -1.0 / 39916800.0));
    s = c.b.build_fma_f64(s, z, f64(c, 1.0 / 362880.0));
    s = c.b.build_fma_f64(s, z, f64(c, -1.0 / 5040.0));
    s = c.b.build_fma_f64(s, z, f64(c, 1.0 / 120.0));
    s = c.b.build_fma_f64(s, z, f64(c, -1.0 / 6.0));
    s = c.b.build_fma_f64(c.b.build_mul(x, z), s, x);

    Value* cs = f64(c, 1.0 / 2432902008176640000.0);
    cs = c.b.build_fma_f64(cs, z, f64(c, -1.0 / 6402373705728000.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, 1.0 / 20922789888000.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, -1.0 / 87178291200.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, 1.0 / 479001600.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, -1.0 / 3628800.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, 1.0 / 40320.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, -1.0 / 720.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, 1.0 / 24.0));
    cs = c.b.build_fma_f64(cs, z, f64(c, -0.5));
    cs = c.b.build_fma_f64(z, cs, f64(c, 1.0));
    return {s, cs};
}

// (cos, sin) of 2 * pi * m / L.
std::pair<Value*, Value*> twiddle64(Ctx& c, Value* m, Value* L) {
    Value* m4 = c.b.build_shl(m, c.i32(2));
    Value* q = c.b.build_udiv(m4, L);
    Value* r0 = c.b.build_sub(m4, c.b.build_mul(q, L));
    Value* swp = c.b.build_ugt(c.b.build_shl(r0, c.i32(1)), L);
    Value* r = c.b.build_select(swp, c.b.build_sub(L, r0), r0);
    Value* rd = c.b.build_sitofp_f64_i32(r);
    Value* ld = c.b.build_sitofp_f64_i32(L);
    Value* x = c.b.build_mul(rd, c.b.build_sdiv(f64(c, 1.5707963267948966192), ld));
    auto [sx, cx] = sincos64(c, x);
    Value* cq = c.b.build_select(swp, sx, cx);
    Value* sq = c.b.build_select(swp, cx, sx);
    Value* n_cq = c.b.build_neg(cq);
    Value* n_sq = c.b.build_neg(sq);
    Value* q0 = c.b.build_eq(q, c.i32(0));
    Value* q1 = c.b.build_eq(q, c.i32(1));
    Value* q2 = c.b.build_eq(q, c.i32(2));
    Value* re = c.b.build_select(q0, cq, c.b.build_select(q1, n_sq, c.b.build_select(q2, n_cq, sq)));
    Value* im = c.b.build_select(q0, sq, c.b.build_select(q1, cq, c.b.build_select(q2, n_sq, n_cq)));
    return {re, im};
}

Value* hermitian_wk(Ctx& c, Value* k, Value* L) {
    Value* is_dc = c.b.build_eq(k, c.i32(0));
    Value* even_L = c.b.build_eq(c.b.build_and(L, c.i32(1)), c.i32(0));
    Value* is_nyq = c.b.build_and(even_L, c.b.build_eq(c.b.build_shl(k, c.i32(1)), L));
    return c.b.build_select(c.b.build_or(is_dc, is_nyq), f64(c, 1.0), f64(c, 2.0));
}

}  // namespace

brass::Function* build_dft64_mir(brass::Module& mod, std::uint32_t mode,
                                 bool has_win, bool has_weights) {
    const std::string name = std::string("mir_dft64_m") + std::to_string(mode) +
                             (has_win ? "_w" : "") + (has_weights ? "_h" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: 0:x 1:y 2:win
    // ints: 3:sa 4:sc 5:R 6:batch 7:cols 8:lda 9:rows 10:ldc 11:L 12:tofs 13:sgn
    // flts: 14:scale

    Value* s_tw_re = shared_alloc_f64(c, kMaxL);
    Value* s_tw_im = shared_alloc_f64(c, kMaxL);
    Value* s_in = shared_alloc_f64(c, kMaxIn);

    Value* tid = kb.tid_x();
    Value* wg = c.i32(static_cast<int>(jit::gpu_mir::kEwBlock));
    Value* L = p[11];

    kb.for_range(tid, L, wg, [&](Value* m) {
        auto [re, im] = twiddle64(c, m, L);
        Value* cm = clamp_tw(c, m);
        shared_store_f64_idx(c, s_tw_re, cm, re);
        shared_store_f64_idx(c, s_tw_im, cm, im);
    });

    Value* sg = c.b.build_sitofp_f64_i32(p[13]);
    Value* scale = c.b.build_fpext_f64_f32(p[14]);
    Value* total = c.b.build_mul(p[5], p[6]);

    kb.for_range(kb.ctaid_x(), total, kb.nctaid_x(), [&](Value* row) {
        Value* z = c.b.build_udiv(row, p[5]);
        Value* r = c.b.build_umod(row, p[5]);
        Value* xo = c.b.build_add(c.b.build_mul(z, p[3]), c.b.build_mul(r, p[8]));
        Value* yo = c.b.build_add(c.b.build_mul(z, p[4]), c.b.build_mul(r, p[10]));
        kb.sync();

        kb.for_range(tid, p[7], wg, [&](Value* i) {
            Value* v = c.b.build_fpext_f64_f32(kb.load_f32(ptr_at(c, p[0], c.b.build_add(xo, i), 4)));
            if (mode == BASIS_R2C && has_win) {
                Value* wv = c.b.build_fpext_f64_f32(kb.load_f32(ptr_at(c, p[2], i, 4)));
                v = c.b.build_mul(v, wv);
            }
            if (mode == BASIS_C2R && has_weights) {
                v = c.b.build_mul(v, hermitian_wk(c, c.b.build_lshr(i, c.i32(1)), L));
            }
            shared_store_f64_idx(c, s_in, clamp_in(c, i), v);
        });
        kb.sync();

        if (mode == BASIS_C2R) {
            Value* nb = c.b.build_lshr(p[7], c.i32(1));
            kb.for_range(tid, p[9], wg, [&](Value* o) {
                Value* n = c.b.build_umod(c.b.build_add(o, p[12]), L);
                auto st = kb.for_range_reduce_n(
                    c.i32(0), nb, c.i32(1), {f64(c, 0.0), c.i32(0)},
                    [&](Value* k, const std::vector<Value*>& cur) -> std::vector<Value*> {
                        Value* cm = clamp_tw(c, cur[1]);
                        Value* tr = shared_load_f64_idx(c, s_tw_re, cm);
                        Value* ti = shared_load_f64_idx(c, s_tw_im, cm);
                        Value* k2 = c.b.build_shl(k, c.i32(1));
                        Value* sr = shared_load_f64_idx(c, s_in, clamp_in(c, k2));
                        Value* si = shared_load_f64_idx(c, s_in, clamp_in(c, c.b.build_add(k2, c.i32(1))));
                        Value* acc = c.b.build_fma_f64(sr, tr, cur[0]);
                        acc = c.b.build_sub(acc, c.b.build_mul(sg, c.b.build_mul(si, ti)));
                        Value* nm = c.b.build_add(cur[1], n);
                        nm = c.b.build_select(c.b.build_uge(nm, L), c.b.build_sub(nm, L), nm);
                        return {acc, nm};
                    });
                Value* out = c.b.build_mul(st[0], scale);
                if (has_win) {
                    out = c.b.build_mul(out, c.b.build_fpext_f64_f32(kb.load_f32(ptr_at(c, p[2], o, 4))));
                }
                kb.store_f32(ptr_at(c, p[1], c.b.build_add(yo, o), 4), c.b.build_fptrunc_f32_f64(out));
            });
        } else {
            const bool cx = (mode == BASIS_C2C);
            Value* nin = cx ? c.b.build_lshr(p[7], c.i32(1)) : p[7];
            Value* nk = c.b.build_lshr(c.b.build_add(p[9], c.i32(1)), c.i32(1));
            Value* tofs_m = cx ? c.i32(0) : c.b.build_umod(p[12], L);
            kb.for_range(tid, nk, wg, [&](Value* k) {
                Value* step = c.b.build_umod(k, L);
                Value* m0 = cx ? c.i32(0) : c.b.build_umod(c.b.build_mul(step, tofs_m), L);
                auto st = kb.for_range_reduce_n(
                    c.i32(0), nin, c.i32(1), {f64(c, 0.0), f64(c, 0.0), m0},
                    [&](Value* n, const std::vector<Value*>& cur) -> std::vector<Value*> {
                        Value* cm = clamp_tw(c, cur[2]);
                        Value* tr = shared_load_f64_idx(c, s_tw_re, cm);
                        Value* ti = shared_load_f64_idx(c, s_tw_im, cm);
                        Value* nm = c.b.build_add(cur[2], step);
                        nm = c.b.build_select(c.b.build_uge(nm, L), c.b.build_sub(nm, L), nm);
                        if (cx) {
                            Value* n2 = c.b.build_shl(n, c.i32(1));
                            Value* xr = shared_load_f64_idx(c, s_in, clamp_in(c, n2));
                            Value* xi = shared_load_f64_idx(c, s_in, clamp_in(c, c.b.build_add(n2, c.i32(1))));
                            Value* re = c.b.build_sub(c.b.build_fma_f64(xr, tr, cur[0]),
                                                      c.b.build_mul(sg, c.b.build_mul(xi, ti)));
                            Value* im = c.b.build_fma_f64(xi, tr,
                                                          c.b.build_fma_f64(sg, c.b.build_mul(xr, ti), cur[1]));
                            return {re, im, nm};
                        }
                        Value* xv = shared_load_f64_idx(c, s_in, clamp_in(c, n));
                        return {c.b.build_fma_f64(xv, tr, cur[0]), c.b.build_fma_f64(xv, ti, cur[1]), nm};
                    });
                Value* re = st[0];
                Value* im = cx ? st[1] : c.b.build_mul(st[1], sg);
                Value* f = has_weights ? c.b.build_mul(scale, hermitian_wk(c, k, L)) : scale;
                Value* k2 = c.b.build_shl(k, c.i32(1));
                kb.store_f32(ptr_at(c, p[1], c.b.build_add(yo, k2), 4),
                             c.b.build_fptrunc_f32_f64(c.b.build_mul(re, f)));
                Value* k2p1 = c.b.build_add(k2, c.i32(1));
                kb.if_then(c.b.build_ult(k2p1, p[9]), [&] {
                    kb.store_f32(ptr_at(c, p[1], c.b.build_add(yo, k2p1), 4),
                                 c.b.build_fptrunc_f32_f64(c.b.build_mul(im, f)));
                });
            });
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
