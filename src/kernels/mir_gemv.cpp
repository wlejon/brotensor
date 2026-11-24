#include "mir_gemv.h"

#include "mir_common.h"
#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

std::vector<Value*> load8_vec(Ctx& c, Dtype dt, Value* base, Value* elem_idx) {
    Value* p = ptr_at(c, base, elem_idx, elem_bytes(dt));
    if (dt == Dtype::FP32) {
        auto lo = load_lanes(c, Dtype::FP32, 4, p);
        auto hi = load_lanes(c, Dtype::FP32, 4, ptr_at(c, base, c.b.build_add(elem_idx, c.i32(4)), 4));
        lo.insert(lo.end(), hi.begin(), hi.end());
        return lo;
    }
    return load_lanes(c, dt, 8, p);
}

std::vector<Value*> load8_scalar(Ctx& c, Dtype dt, Value* base, Value* idx, Value* col, Value* ncols) {
    std::vector<Value*> out(8);
    for (int e = 0; e < 8; ++e) {
        Value* ok = c.b.build_ult(c.b.build_add(col, c.i32(e)), ncols);
        out[static_cast<std::size_t>(e)] = emit_if_else_values(
            c, ok, {Type::f32()},
            [&]() -> std::vector<Value*> { return {load_elem_dt(c, dt, base, c.b.build_add(idx, c.i32(e)))}; },
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
    }
    return out;
}

}  // namespace

brass::Function* build_gemv_mir(brass::Module& mod, Dtype dt_x, Dtype dt_w, Dtype dt_y,
                                std::uint32_t mb, std::uint32_t epi, std::uint32_t act,
                                bool vec, bool has_bias) {
    const bool glu = (epi == EPI_GEGLU || epi == EPI_SWIGLU);
    const std::uint32_t nw = glu ? 2u : 1u;
    const std::uint32_t unr = (mb * nw <= 8u) ? 4u : 2u;
    const std::string name = std::string("mir_gemv_m") + std::to_string(mb) +
                             "_e" + std::to_string(epi) + "_a" + std::to_string(act) +
                             (vec ? "_v" : "_s") + (has_bias ? "_b" : "");

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: a(0) b(1) c(2) bias(3) m(4) n(5) k(6) lda(7) ldb(8) ldc(9) half_n(10)

    Value* s_part = kb.shared_alloc_f32(8 * 16);
    Value* tid = kb.tid_x();
    Value* unit = c.b.build_udiv(tid, c.i32(64));
    Value* t = c.b.build_umod(tid, c.i32(64));
    Value* g = c.b.build_add(kb.ctaid_x(), c.b.build_mul(kb.ctaid_y(), kb.nctaid_x()));
    Value* j = c.b.build_add(c.b.build_shl(g, c.i32(2)), unit);
    Value* nout = (epi == EPI_SWIGLU) ? p[10] : (epi == EPI_GEGLU ? c.b.build_lshr(p[5], c.i32(1)) : p[5]);
    Value* live = c.b.build_ult(j, nout);

    Value* wrow[2];
    if (epi == EPI_GEGLU) {
        wrow[0] = c.b.build_shl(j, c.i32(1));
        wrow[1] = c.b.build_add(wrow[0], c.i32(1));
    } else if (epi == EPI_SWIGLU) {
        wrow[0] = j;
        wrow[1] = c.b.build_add(p[10], j);
    } else {
        wrow[0] = j;
        wrow[1] = j;
    }

    std::vector<Value*> acc_init(mb * nw, c.f32(0.0f));
    std::vector<Value*> acc = emit_if_else_values(
        c, live, std::vector<Type>(mb * nw, Type::f32()),
        [&]() -> std::vector<Value*> {
            if (vec) {
                Value* nch = c.b.build_lshr(p[6], c.i32(3));
                return kb.for_range_reduce_n(
                    t, nch, c.i32(static_cast<int>(64u * unr)), acc_init,
                    [&](Value* c0, const std::vector<Value*>& cur) -> std::vector<Value*> {
                        std::vector<Value*> ok(unr), cc(unr);
                        std::vector<std::vector<std::vector<Value*>>> wv(
                            unr, std::vector<std::vector<Value*>>(nw));
                        for (std::uint32_t u = 0; u < unr; ++u) {
                            Value* col_u = c.b.build_add(c0, c.i32(static_cast<int>(u * 64u)));
                            Value* in_k = c.b.build_ult(col_u, nch);
                            ok[u] = c.b.build_select(in_k, c.f32(1.0f), c.f32(0.0f));
                            cc[u] = c.b.build_select(in_k, col_u, c0);
                            for (std::uint32_t w = 0; w < nw; ++w) {
                                Value* idx = c.b.build_add(c.b.build_mul(wrow[w], p[8]),
                                                           c.b.build_shl(cc[u], c.i32(3)));
                                wv[u][w] = load8_vec(c, dt_w, p[1], idx);
                            }
                        }
                        std::vector<Value*> next = cur;
                        for (std::uint32_t r = 0; r < mb; ++r) {
                            Value* row_off = c.b.build_mul(c.i32(static_cast<int>(r)), p[7]);
                            for (std::uint32_t u = 0; u < unr; ++u) {
                                Value* aidx = c.b.build_add(row_off, c.b.build_shl(cc[u], c.i32(3)));
                                auto xv = load8_vec(c, dt_x, p[0], aidx);
                                for (std::uint32_t w = 0; w < nw; ++w) {
                                    Value* s = c.f32(0.0f);
                                    for (int e = 0; e < 8; ++e) {
                                        s = c.b.build_fma_f32(xv[e], wv[u][w][e], s);
                                    }
                                    std::size_t q = r * nw + w;
                                    next[q] = c.b.build_fma_f32(s, ok[u], next[q]);
                                }
                            }
                        }
                        return next;
                    });
            }
            return kb.for_range_reduce_n(
                c.b.build_shl(t, c.i32(3)), p[6], c.i32(64 * 8), acc_init,
                [&](Value* col, const std::vector<Value*>& cur) -> std::vector<Value*> {
                    std::vector<std::vector<Value*>> wv(nw);
                    for (std::uint32_t w = 0; w < nw; ++w) {
                        wv[w] = load8_scalar(c, dt_w, p[1], c.b.build_add(c.b.build_mul(wrow[w], p[8]), col), col, p[6]);
                    }
                    std::vector<Value*> next = cur;
                    for (std::uint32_t r = 0; r < mb; ++r) {
                        Value* aidx = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(r)), p[7]), col);
                        auto xv = load8_scalar(c, dt_x, p[0], aidx, col, p[6]);
                        for (std::uint32_t w = 0; w < nw; ++w) {
                            Value* s = c.f32(0.0f);
                            for (int e = 0; e < 8; ++e) {
                                s = c.b.build_fma_f32(xv[e], wv[w][e], s);
                            }
                            std::size_t q = r * nw + w;
                            next[q] = c.b.build_add(next[q], s);
                        }
                    }
                    return next;
                });
        },
        [&]() -> std::vector<Value*> { return acc_init; });

    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* w32 = c.b.build_and(c.b.build_lshr(tid, c.i32(5)), c.i32(1));
    Value* sg_base = c.b.build_shl(c.b.build_add(c.b.build_shl(unit, c.i32(1)), w32), c.i32(4));
    for (std::uint32_t q = 0; q < mb * nw; ++q) {
        Value* s = acc[q];
        for (std::uint32_t m = 16; m >= 1; m >>= 1) {
            s = c.b.build_add(s, kb.shfl_bfly_f32(s, m));
        }
        kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] {
            kb.shared_store_f32_indexed(s_part, c.b.build_add(sg_base, c.i32(static_cast<int>(q))), s);
        });
    }
    kb.sync();

    kb.if_then(c.b.build_and(c.b.build_eq(t, c.i32(0)), live), [&] {
        Value* u0_base = c.b.build_shl(unit, c.i32(5));
        Value* u1_base = c.b.build_add(u0_base, c.i32(16));
        for (std::uint32_t r = 0; r < mb; ++r) {
            Value* v[2] = {c.f32(0.0f), c.f32(0.0f)};
            for (std::uint32_t w = 0; w < nw; ++w) {
                Value* qv = c.i32(static_cast<int>(r * nw + w));
                Value* s = c.b.build_add(kb.shared_load_f32_indexed(s_part, c.b.build_add(u0_base, qv)),
                                         kb.shared_load_f32_indexed(s_part, c.b.build_add(u1_base, qv)));
                if (has_bias) {
                    s = c.b.build_add(s, load_elem_dt(c, dt_y, p[3], wrow[w]));
                }
                v[w] = emit_gemm_act(c, act, s);
            }
            Value* o = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(r)), p[9]), j);
            Value* out = v[0];
            if (epi == EPI_GEGLU) {
                out = c.b.build_mul(v[0], emit_gelu_exact(c, v[1]));
            } else if (epi == EPI_SWIGLU) {
                out = c.b.build_mul(c.b.build_mul(v[0], emit_sigmoid(c, v[0])), v[1]);
            } else if (epi == EPI_ACCUM) {
                out = c.b.build_add(v[0], load_elem_dt(c, dt_y, p[2], o));
            }
            store_elem_dt(c, dt_y, p[2], o, out);
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
