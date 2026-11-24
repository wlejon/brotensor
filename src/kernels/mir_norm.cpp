#include "mir_norm.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <functional>
#include <vector>

namespace brotensor::kernels {

namespace {

struct NormCtx {
    Ctx& c;
    const std::vector<Value*>& p;
    std::uint32_t mode;
    Dtype dt_x, dt_y, dt_g;
    int esz_x, esz_y, esz_g, lanes;
    NormMirFlags flags;
    Value* tid;
    Value* block;
    Value* n_vecs;
    Value* invd;
    Value* d_f;
    Value* eps_safe;
    Value* s_red;
};

Value* norm_rsqrt(Ctx& c, Value* arg) {
    return emit_rsqrt(c, c.kb.fmax(arg, c.f32(1e-12f)));
}

void build_norm_columns(NormCtx& n) {
    Ctx& c = n.c;
    const auto& p = n.p;
    Value* lx = c.b.build_and(n.tid, c.i32(31));
    Value* ly = c.b.build_and(c.b.build_lshr(n.tid, c.i32(5)), c.i32(7));
    Value* s_col0 = c.kb.shared_alloc_f32(8u * 33u);
    Value* s_col1 = (n.mode == NORM_COL_LN) ? c.kb.shared_alloc_f32(8u * 33u) : nullptr;
    Value* slot = c.b.build_add(c.b.build_mul(ly, c.i32(33)), lx);
    Value* step_c = c.b.build_shl(c.kb.nctaid_x(), c.i32(5));
    Value* start_c = c.b.build_shl(c.kb.ctaid_x(), c.i32(5));
    Value* rows = p[7];
    Value* d = p[8];

    c.kb.for_range(start_c, d, step_c, [&](Value* c0) {
        Value* col = c.b.build_add(c0, lx);
        Value* in_b = c.b.build_ult(col, d);
        Value* r_end = c.b.build_select(in_b, rows, c.i32(0));
        const std::vector<Value*> inits = (n.mode == NORM_COL_LN)
            ? std::vector<Value*>{c.f32(0.0f), c.f32(0.0f)}
            : std::vector<Value*>{c.f32(0.0f)};
        auto acc = c.kb.for_range_reduce_n(ly, r_end, c.i32(8), inits, [&](Value* r, const std::vector<Value*>& cur) {
            Value* idx = c.b.build_add(c.b.build_mul(r, d), col);
            Value* xv = load_lanes(c, n.dt_x, 1, ptr_at(c, p[0], idx, n.esz_x))[0];
            if (n.mode == NORM_COL_LN) {
                Value* xhv = load_lanes(c, n.dt_x, 1, ptr_at(c, p[4], idx, n.esz_x))[0];
                return std::vector<Value*>{c.b.build_fma_f32(xv, xhv, cur[0]), c.b.build_add(cur[1], xv)};
            }
            if (n.mode == NORM_COL_RMS) {
                Value* dy = load_lanes(c, n.dt_x, 1, ptr_at(c, p[4], idx, n.esz_x))[0];
                Value* rr = c.kb.load_f32(ptr_at(c, p[6], r, 4));
                return std::vector<Value*>{c.b.build_fma_f32(c.b.build_mul(dy, xv), rr, cur[0])};
            }
            return std::vector<Value*>{c.b.build_add(cur[0], xv)};
        });
        c.kb.shared_store_f32_indexed(s_col0, slot, acc[0]);
        if (n.mode == NORM_COL_LN) c.kb.shared_store_f32_indexed(s_col1, slot, acc[1]);
        c.kb.sync();
        c.kb.if_then(c.b.build_and(c.b.build_eq(ly, c.i32(0)), in_b), [&] {
            Value* t0 = c.kb.shared_load_f32_indexed(s_col0, lx);
            for (int k = 1; k < 8; ++k) {
                t0 = c.b.build_add(t0, c.kb.shared_load_f32_indexed(s_col0, c.b.build_add(c.i32(k * 33), lx)));
            }
            Value* gp = ptr_at(c, p[2], col, n.esz_g);
            store_lanes(c, n.dt_g, 1, gp, {c.b.build_add(load_lanes(c, n.dt_g, 1, gp)[0], t0)});
            if (n.mode == NORM_COL_LN) {
                Value* t1 = c.kb.shared_load_f32_indexed(s_col1, lx);
                for (int k = 1; k < 8; ++k) {
                    t1 = c.b.build_add(t1, c.kb.shared_load_f32_indexed(s_col1, c.b.build_add(c.i32(k * 33), lx)));
                }
                Value* bp = ptr_at(c, p[3], col, n.esz_g);
                store_lanes(c, n.dt_g, 1, bp, {c.b.build_add(load_lanes(c, n.dt_g, 1, bp)[0], t1)});
            }
        });
        c.kb.sync();
    });
}

void build_norm_ln(NormCtx& n, Value* row, Value* base) {
    Ctx& c = n.c;
    const auto& p = n.p;
    Value* sum = c.kb.for_range_reduce(n.tid, n.n_vecs, n.block, c.f32(0.0f), [&](Value* k, Value* a) {
        Value* col = col_elem_idx(c, k, n.lanes);
        for (Value* v : load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], c.b.build_add(base, col), n.esz_x))) {
            a = c.b.build_add(a, v);
        }
        return a;
    });
    Value* mean = c.b.build_mul(emit_block_sum_f32(c, n.s_red, sum), n.invd);
    Value* sq = c.kb.for_range_reduce(n.tid, n.n_vecs, n.block, c.f32(0.0f), [&](Value* k, Value* a) {
        Value* col = col_elem_idx(c, k, n.lanes);
        for (Value* v : load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], c.b.build_add(base, col), n.esz_x))) {
            Value* diff = c.b.build_sub(v, mean);
            a = c.b.build_fma_f32(diff, diff, a);
        }
        return a;
    });
    Value* var = c.b.build_mul(emit_block_sum_f32(c, n.s_red, sq), n.invd);
    Value* rstd = norm_rsqrt(c, c.b.build_add(var, n.eps_safe));
    if (n.flags.has_s0 || n.flags.has_s1) {
        c.kb.if_then(c.b.build_eq(n.tid, c.i32(0)), [&] {
            if (n.flags.has_s0) c.kb.store_f32(ptr_at(c, p[5], row, 4), mean);
            if (n.flags.has_s1) c.kb.store_f32(ptr_at(c, p[6], row, 4), rstd);
        });
    }
    c.kb.for_range(n.tid, n.n_vecs, n.block, [&](Value* k) {
        Value* col = col_elem_idx(c, k, n.lanes);
        Value* idx = c.b.build_add(base, col);
        auto xs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        std::vector<Value*> gs, bs;
        if (n.flags.has_g) gs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[2], col, n.esz_g));
        if (n.flags.has_b) bs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[3], col, n.esz_g));
        std::vector<Value*> xhs(static_cast<std::size_t>(n.lanes));
        std::vector<Value*> ys(static_cast<std::size_t>(n.lanes));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            xhs[js] = c.b.build_mul(c.b.build_sub(xs[js], mean), rstd);
            Value* scaled = gs.empty() ? xhs[js] : c.b.build_mul(gs[js], xhs[js]);
            ys[js] = bs.empty() ? scaled : c.b.build_add(scaled, bs[js]);
        }
        if (n.flags.has_xh) store_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x), xhs);
        store_lanes(c, n.dt_y, n.lanes, ptr_at(c, p[1], idx, n.esz_y), ys);
    });
}

void build_norm_rms_l2(NormCtx& n, Value* row, Value* base) {
    Ctx& c = n.c;
    const auto& p = n.p;
    Value* sq = c.kb.for_range_reduce(n.tid, n.n_vecs, n.block, c.f32(0.0f), [&](Value* k, Value* a) {
        Value* col = col_elem_idx(c, k, n.lanes);
        for (Value* v : load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], c.b.build_add(base, col), n.esz_x))) {
            a = c.b.build_fma_f32(v, v, a);
        }
        return a;
    });
    Value* q = emit_block_sum_f32(c, n.s_red, sq);
    Value* arg = (n.mode == NORM_RMS) ? c.b.build_fma_f32(q, n.invd, n.eps_safe) : c.b.build_add(q, n.eps_safe);
    Value* r = norm_rsqrt(c, arg);
    if (n.mode == NORM_RMS && n.flags.has_s1) {
        c.kb.if_then(c.b.build_eq(n.tid, c.i32(0)), [&] { c.kb.store_f32(ptr_at(c, p[6], row, 4), r); });
    }
    c.kb.for_range(n.tid, n.n_vecs, n.block, [&](Value* k) {
        Value* col = col_elem_idx(c, k, n.lanes);
        Value* idx = c.b.build_add(base, col);
        auto xs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        std::vector<Value*> gs;
        if (n.mode == NORM_RMS && n.flags.has_g) {
            gs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[2], col, n.esz_g));
        }
        std::vector<Value*> ys(static_cast<std::size_t>(n.lanes));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            Value* vr = c.b.build_mul(xs[js], r);
            ys[js] = (!gs.empty()) ? c.b.build_mul(vr, gs[js]) : vr;
        }
        store_lanes(c, n.dt_y, n.lanes, ptr_at(c, p[1], idx, n.esz_y), ys);
    });
}

void build_norm_ln_bwd(NormCtx& n, Value* row, Value* base) {
    Ctx& c = n.c;
    const auto& p = n.p;
    auto sums = c.kb.for_range_reduce_n(
        n.tid, n.n_vecs, n.block, {c.f32(0.0f), c.f32(0.0f)}, [&](Value* k, const std::vector<Value*>& cur) {
            Value* col = col_elem_idx(c, k, n.lanes);
            Value* idx = c.b.build_add(base, col);
            auto dys = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
            auto gs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[2], col, n.esz_g));
            auto xhs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x));
            Value* a1 = cur[0];
            Value* a2 = cur[1];
            for (int j = 0; j < n.lanes; ++j) {
                const auto js = static_cast<std::size_t>(j);
                Value* dxh = c.b.build_mul(dys[js], gs[js]);
                a1 = c.b.build_add(a1, dxh);
                a2 = c.b.build_fma_f32(dxh, xhs[js], a2);
            }
            return std::vector<Value*>{a1, a2};
        });
    Value* s1 = emit_block_sum_f32(c, n.s_red, sums[0]);
    Value* s2 = emit_block_sum_f32(c, n.s_red, sums[1]);
    Value* rstd = n.flags.has_s1 ? c.kb.load_f32(ptr_at(c, p[6], row, 4)) : p[10];
    Value* scale = c.b.build_mul(rstd, n.invd);
    c.kb.for_range(n.tid, n.n_vecs, n.block, [&](Value* k) {
        Value* col = col_elem_idx(c, k, n.lanes);
        Value* idx = c.b.build_add(base, col);
        auto dys = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        auto gs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[2], col, n.esz_g));
        auto xhs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x));
        std::vector<Value*> ys(static_cast<std::size_t>(n.lanes));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            Value* dxh = c.b.build_mul(dys[js], gs[js]);
            Value* inner = c.b.build_sub(c.b.build_sub(c.b.build_mul(n.d_f, dxh), s1),
                                         c.b.build_mul(xhs[js], s2));
            ys[js] = c.b.build_mul(scale, inner);
        }
        store_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[1], idx, n.esz_x), ys);
    });
}

void build_norm_rms_l2_bwd(NormCtx& n, Value* row, Value* base) {
    Ctx& c = n.c;
    const auto& p = n.p;
    const bool use_g = (n.mode == NORM_RMS_BWD && n.flags.has_g);
    auto sums = c.kb.for_range_reduce_n(
        n.tid, n.n_vecs, n.block, {c.f32(0.0f), c.f32(0.0f)}, [&](Value* k, const std::vector<Value*>& cur) {
            Value* col = col_elem_idx(c, k, n.lanes);
            Value* idx = c.b.build_add(base, col);
            auto vs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
            auto dys = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x));
            std::vector<Value*> gs;
            if (use_g) gs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[2], col, n.esz_g));
            Value* aq = cur[0];
            Value* ap = cur[1];
            for (int j = 0; j < n.lanes; ++j) {
                const auto js = static_cast<std::size_t>(j);
                aq = c.b.build_fma_f32(vs[js], vs[js], aq);
                Value* vd = c.b.build_mul(vs[js], dys[js]);
                ap = (!gs.empty()) ? c.b.build_fma_f32(vd, gs[js], ap) : c.b.build_add(ap, vd);
            }
            return std::vector<Value*>{aq, ap};
        });
    Value* q = emit_block_sum_f32(c, n.s_red, sums[0]);
    Value* p_tot = emit_block_sum_f32(c, n.s_red, sums[1]);
    Value* arg = (n.mode == NORM_RMS_BWD) ? c.b.build_fma_f32(q, n.invd, n.eps_safe) : c.b.build_add(q, n.eps_safe);
    Value* r = norm_rsqrt(c, arg);
    if (n.mode == NORM_RMS_BWD && n.flags.has_s1) {
        c.kb.if_then(c.b.build_eq(n.tid, c.i32(0)), [&] { c.kb.store_f32(ptr_at(c, p[6], row, 4), r); });
    }
    Value* rr = c.b.build_mul(r, r);
    Value* coeff = (n.mode == NORM_RMS_BWD) ? c.b.build_mul(c.b.build_mul(n.invd, rr), p_tot)
                                            : c.b.build_mul(rr, p_tot);
    c.kb.for_range(n.tid, n.n_vecs, n.block, [&](Value* k) {
        Value* col = col_elem_idx(c, k, n.lanes);
        Value* idx = c.b.build_add(base, col);
        auto vs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        auto dys = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x));
        std::vector<Value*> gs;
        if (use_g) gs = load_lanes(c, n.dt_g, n.lanes, ptr_at(c, p[2], col, n.esz_g));
        std::vector<Value*> ys(static_cast<std::size_t>(n.lanes));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            Value* gdy = (!gs.empty()) ? c.b.build_mul(gs[js], dys[js]) : dys[js];
            ys[js] = c.b.build_mul(r, c.b.build_sub(gdy, c.b.build_mul(vs[js], coeff)));
        }
        store_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[1], idx, n.esz_x), ys);
    });
}

void build_norm_softmax(NormCtx& n, Value* base) {
    Ctx& c = n.c;
    const auto& p = n.p;
    auto mask_off = [&](const std::vector<Value*>& ms, std::size_t j) -> Value* {
        return c.b.build_slt(ms[j], c.f32(0.5f));
    };
    Value* m_acc = c.kb.for_range_reduce(n.tid, n.n_vecs, n.block, c.f32(-1e30f), [&](Value* k, Value* a) {
        Value* idx = c.b.build_add(base, col_elem_idx(c, k, n.lanes));
        auto xs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        std::vector<Value*> ms;
        if (n.flags.has_s0) ms = load_lanes(c, Dtype::FP32, n.lanes, ptr_at(c, p[5], idx, 4));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            Value* mx = c.kb.fmax(a, xs[js]);
            a = n.flags.has_s0 ? c.b.build_select(mask_off(ms, js), a, mx) : mx;
        }
        return a;
    });
    Value* m = emit_block_max_f32(c, n.s_red, m_acc);
    Value* s_acc = c.kb.for_range_reduce(n.tid, n.n_vecs, n.block, c.f32(0.0f), [&](Value* k, Value* a) {
        Value* idx = c.b.build_add(base, col_elem_idx(c, k, n.lanes));
        auto xs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        std::vector<Value*> ms;
        if (n.flags.has_s0) ms = load_lanes(c, Dtype::FP32, n.lanes, ptr_at(c, p[5], idx, 4));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            Value* ex = emit_exp(c, c.b.build_sub(xs[js], m));
            a = c.b.build_add(a, n.flags.has_s0 ? c.b.build_select(mask_off(ms, js), c.f32(0.0f), ex) : ex);
        }
        return a;
    });
    Value* s = emit_block_sum_f32(c, n.s_red, s_acc);
    Value* inv = c.b.build_select(c.b.build_sgt(s, c.f32(0.0f)),
                                c.b.build_sdiv(c.f32(1.0f), c.kb.fmax(s, c.f32(1e-30f))), c.f32(0.0f));
    c.kb.for_range(n.tid, n.n_vecs, n.block, [&](Value* k) {
        Value* idx = c.b.build_add(base, col_elem_idx(c, k, n.lanes));
        auto xs = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        std::vector<Value*> ms;
        if (n.flags.has_s0) ms = load_lanes(c, Dtype::FP32, n.lanes, ptr_at(c, p[5], idx, 4));
        std::vector<Value*> ys(static_cast<std::size_t>(n.lanes));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            Value* val = c.b.build_mul(emit_exp(c, c.b.build_sub(xs[js], m)), inv);
            ys[js] = n.flags.has_s0 ? c.b.build_select(mask_off(ms, js), c.f32(0.0f), val) : val;
        }
        store_lanes(c, n.dt_y, n.lanes, ptr_at(c, p[1], idx, n.esz_y), ys);
    });
}

void build_norm_sm_bwd(NormCtx& n, Value* base) {
    Ctx& c = n.c;
    const auto& p = n.p;
    Value* dot = c.kb.for_range_reduce(n.tid, n.n_vecs, n.block, c.f32(0.0f), [&](Value* k, Value* a) {
        Value* idx = c.b.build_add(base, col_elem_idx(c, k, n.lanes));
        auto ps = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        auto dps = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            a = c.b.build_fma_f32(dps[js], ps[js], a);
        }
        return a;
    });
    dot = emit_block_sum_f32(c, n.s_red, dot);
    c.kb.for_range(n.tid, n.n_vecs, n.block, [&](Value* k) {
        Value* idx = c.b.build_add(base, col_elem_idx(c, k, n.lanes));
        auto ps = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[0], idx, n.esz_x));
        auto dps = load_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[4], idx, n.esz_x));
        std::vector<Value*> ys(static_cast<std::size_t>(n.lanes));
        for (int j = 0; j < n.lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            ys[js] = c.b.build_mul(ps[js], c.b.build_sub(dps[js], dot));
        }
        store_lanes(c, n.dt_x, n.lanes, ptr_at(c, p[1], idx, n.esz_x), ys);
    });
}

}  // namespace

Function* build_norm_mir(Module& mod, std::uint32_t mode, Dtype dt_x, Dtype dt_y,
                         Dtype dt_g, NormMirFlags flags, int lanes) {
    Function* fn = mod.create_function(
        "mir_norm", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* d_f = kb.fmax(kb.i32_to_f32(p[8]), c.f32(1.0f));
    NormCtx n{
        c, p, mode, dt_x, dt_y, dt_g,
        elem_bytes(dt_x), elem_bytes(dt_y), elem_bytes(dt_g), lanes, flags,
        kb.tid_x(), c.i32(static_cast<int>(kNormBlock)),
        vec_count(c, p[8], lanes),
        c.b.build_sdiv(c.f32(1.0f), d_f),
        d_f,
        kb.fmax(p[9], c.f32(1e-12f)),
        (mode < NORM_COL_LN) ? kb.shared_alloc_f32(8) : nullptr};

    if (mode >= NORM_COL_LN) {
        build_norm_columns(n);
    } else {
        kb.for_range(kb.ctaid_x(), p[7], kb.nctaid_x(), [&](Value* row) {
            Value* base = c.b.build_mul(row, p[8]);
            if (mode == NORM_LN)                              build_norm_ln(n, row, base);
            else if (mode == NORM_RMS || mode == NORM_L2)     build_norm_rms_l2(n, row, base);
            else if (mode == NORM_LN_BWD)                     build_norm_ln_bwd(n, row, base);
            else if (mode == NORM_RMS_BWD || mode == NORM_L2_BWD) build_norm_rms_l2_bwd(n, row, base);
            else if (mode == NORM_SOFTMAX)                    build_norm_softmax(n, base);
            else                                              build_norm_sm_bwd(n, base);
        });
    }
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
