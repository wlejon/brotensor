#include "mir_fa_bwd.h"

#include "mir_common.h"
#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr float kLog2e = 1.44269504088896340736f;

Value* alloc_stage(Ctx& c, Dtype dt, std::uint32_t count) {
    return (dt == Dtype::FP32) ? c.kb.shared_alloc_f32(count) : c.kb.shared_alloc_u16(count);
}

Value* load_stage_f32(Ctx& c, Dtype dt, Value* buf, Value* idx) {
    if (dt == Dtype::FP32) return c.kb.shared_load_f32_indexed(buf, idx);
    Value* raw = c.kb.shared_load_u16_indexed(buf, idx);
    return (dt == Dtype::FP16) ? c.kb.f16_to_f32(raw) : c.kb.bf16_to_f32(raw);
}

void stage_tile(Ctx& c, Dtype dt, Value* s_buf, Value* g_ptr, Value* g_ld,
                Value* r0, Value* r_limit, Value* head_col,
                std::uint32_t rows, std::uint32_t hd, std::uint32_t hdp, std::uint32_t nt) {
    Value* tid = c.kb.tid_x();
    const std::uint32_t per = (rows * hdp) / nt;
    const int esz = elem_bytes(dt);
    for (std::uint32_t e = 0; e < per; ++e) {
        Value* idx = c.b.build_add(c.i32(static_cast<int>(e * nt)), tid);
        Value* r = c.b.build_udiv(idx, c.i32(static_cast<int>(hdp)));
        Value* d = c.b.build_umod(idx, c.i32(static_cast<int>(hdp)));
        Value* gr = c.b.build_add(r0, r);
        Value* ok = c.b.build_and(c.b.build_ult(gr, r_limit), c.b.build_ult(d, c.i32(static_cast<int>(hd))));
        Value* elem_off = c.b.build_add(c.b.build_add(c.b.build_mul(gr, g_ld), head_col), d);
        if (dt == Dtype::FP32) {
            Value* val = emit_if_else_values(
                c, ok, {Type::f32()},
                [&]() -> std::vector<Value*> { return {c.kb.load_f32(ptr_at(c, g_ptr, elem_off, esz))}; },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            c.kb.shared_store_f32_indexed(s_buf, idx, val);
        } else {
            Value* raw = emit_if_else_values(
                c, ok, {Type::i32()},
                [&]() -> std::vector<Value*> { return {c.kb.load_u16(ptr_at(c, g_ptr, elem_off, esz))}; },
                [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
            c.kb.shared_store_u16_indexed(s_buf, idx, raw);
        }
    }
}

void stage_kval(Ctx& c, bool has_mask, Value* s_kval, Value* mask_ptr, Value* k0, Value* lk,
                std::uint32_t bc, std::uint32_t nt) {
    Value* tid = c.kb.tid_x();
    for (std::uint32_t e = 0; e < bc; e += nt) {
        Value* col = (e == 0) ? tid : c.b.build_add(c.i32(static_cast<int>(e)), tid);
        auto work = [&] {
            Value* gk = c.b.build_add(k0, col);
            Value* in_k = c.b.build_ult(gk, lk);
            Value* ok = in_k;
            if (has_mask) {
                Value* mv = emit_if_else_values(
                    c, in_k, {Type::f32()},
                    [&]() -> std::vector<Value*> { return {c.kb.load_f32(ptr_at(c, mask_ptr, gk, 4))}; },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                ok = c.b.build_and(in_k, c.b.build_slt(c.f32(0.5f), mv));
            }
            c.kb.shared_store_i32_indexed(s_kval, col, c.b.build_select(ok, c.i32(1), c.i32(0)));
        };
        if (e + nt <= bc) work();
        else c.kb.if_then(c.b.build_ult(col, c.i32(static_cast<int>(bc))), work);
    }
}

std::pair<Value*, Value*> emit_bw_interval(Ctx& c, std::uint32_t mode,
                                           Value* qi, Value* lk, Value* causal, Value* window,
                                           Value* nseq, Value* aux, Value* aux2) {
    if (mode == FA_MODE_VARLEN) {
        auto st = c.kb.for_range_reduce_n(
            c.i32(0), c.i32(24), c.i32(1), {c.i32(0), nseq},
            [&](Value* /*step*/, const std::vector<Value*>& cur) -> std::vector<Value*> {
                Value* slo = cur[0];
                Value* shi = cur[1];
                Value* active = c.b.build_slt(slo, shi);
                Value* mid = c.b.build_add(slo, c.b.build_ashr(c.b.build_sub(shi, slo), c.i32(1)));
                Value* safe_mid = c.b.build_select(active, mid, c.i32(0));
                Value* end_q = c.kb.load_i32(ptr_at(c, aux, c.b.build_add(safe_mid, c.i32(1)), 4));
                Value* go_right = c.b.build_sle(end_q, qi);
                Value* nlo = c.b.build_select(c.b.build_and(active, go_right), c.b.build_add(safe_mid, c.i32(1)), slo);
                Value* nhi = c.b.build_select(c.b.build_and(active, c.b.build_not(go_right)), safe_mid, shi);
                return {nlo, nhi};
            });
        Value* has_seq = c.b.build_sgt(nseq, c.i32(0));
        Value* b = c.b.build_select(has_seq, clamp_i32(c, st[0], c.i32(0), c.b.build_sub(nseq, c.i32(1))), c.i32(0));
        Value* qs = c.kb.load_i32(ptr_at(c, aux, b, 4));
        Value* qe = c.kb.load_i32(ptr_at(c, aux, c.b.build_add(b, c.i32(1)), 4));
        Value* ks = c.kb.load_i32(ptr_at(c, aux2, b, 4));
        Value* ke = c.kb.load_i32(ptr_at(c, aux2, c.b.build_add(b, c.i32(1)), 4));
        Value* in_seq = c.b.build_and(has_seq, c.b.build_and(c.b.build_sle(qs, qi), c.b.build_slt(qi, qe)));
        Value* lo = smax_i32(c, c.i32(0), ks);
        Value* hi_seq = smin_i32(c, lk, ke);
        Value* is_c = c.b.build_ne(causal, c.i32(0));
        Value* aq = c.b.build_add(ks, c.b.build_sub(qi, qs));
        Value* hi = c.b.build_select(is_c, smin_i32(c, hi_seq, c.b.build_add(aq, c.i32(1))), hi_seq);
        Value* has_win = c.b.build_sgt(window, c.i32(0));
        Value* wlo_c = smax_i32(c, lo, c.b.build_add(c.b.build_sub(aq, window), c.i32(1)));
        Value* hw = c.b.build_ashr(window, c.i32(1));
        Value* wlo_b = smax_i32(c, lo, c.b.build_sub(aq, hw));
        Value* whi_b = smin_i32(c, hi, c.b.build_add(c.b.build_add(aq, hw), c.i32(1)));
        lo = c.b.build_select(has_win, c.b.build_select(is_c, wlo_c, wlo_b), lo);
        hi = c.b.build_select(c.b.build_and(has_win, c.b.build_not(is_c)), whi_b, hi);
        lo = c.b.build_select(in_seq, lo, c.i32(0));
        hi = c.b.build_select(in_seq, smax_i32(c, lo, hi), c.i32(0));
        return {lo, hi};
    }
    if (mode == FA_MODE_PACKED) {
        Value* s0 = c.kb.load_i32(ptr_at(c, aux, c.b.build_shl(qi, c.i32(1)), 4));
        Value* s1 = c.kb.load_i32(ptr_at(c, aux, c.b.build_add(c.b.build_shl(qi, c.i32(1)), c.i32(1)), 4));
        Value* lo = smax_i32(c, c.i32(0), s0);
        Value* hi = smin_i32(c, lk, s1);
        Value* has_win = c.b.build_sgt(window, c.i32(0));
        Value* hw = c.b.build_ashr(window, c.i32(1));
        lo = c.b.build_select(has_win, smax_i32(c, lo, c.b.build_sub(qi, hw)), lo);
        hi = c.b.build_select(has_win, smin_i32(c, hi, c.b.build_add(c.b.build_add(qi, hw), c.i32(1))), hi);
        return {lo, smax_i32(c, lo, hi)};
    }
    // FA_MODE_ROWS
    Value* is_c = c.b.build_ne(causal, c.i32(0));
    Value* has_win = c.b.build_sgt(window, c.i32(0));
    Value* hw = c.b.build_ashr(window, c.i32(1));
    Value* hi_c = smin_i32(c, lk, c.b.build_add(qi, c.i32(1)));
    Value* hi_nc = c.b.build_select(has_win, smin_i32(c, lk, c.b.build_add(c.b.build_add(qi, hw), c.i32(1))), lk);
    Value* hi = c.b.build_select(is_c, hi_c, hi_nc);
    Value* lo_c = c.b.build_select(has_win, smax_i32(c, c.i32(0), c.b.build_add(c.b.build_sub(qi, window), c.i32(1))), c.i32(0));
    Value* lo_nc = c.b.build_select(has_win, smax_i32(c, c.i32(0), c.b.build_sub(qi, hw)), c.i32(0));
    Value* lo = c.b.build_select(is_c, lo_c, lo_nc);
    return {lo, smax_i32(c, lo, hi)};
}

std::pair<Value*, Value*> emit_bw_query_range(Ctx& c, std::uint32_t mode,
                                              Value* k0, Value* kend, Value* lq,
                                              Value* causal, Value* window, Value* nseq,
                                              Value* aux, Value* aux2) {
    if (mode == FA_MODE_VARLEN) {
        auto st0 = c.kb.for_range_reduce_n(
            c.i32(0), c.i32(24), c.i32(1), {c.i32(0), nseq},
            [&](Value* /*step*/, const std::vector<Value*>& cur) -> std::vector<Value*> {
                Value* slo = cur[0];
                Value* shi = cur[1];
                Value* active = c.b.build_slt(slo, shi);
                Value* mid = c.b.build_add(slo, c.b.build_ashr(c.b.build_sub(shi, slo), c.i32(1)));
                Value* safe_mid = c.b.build_select(active, mid, c.i32(0));
                Value* end_k = c.kb.load_i32(ptr_at(c, aux2, c.b.build_add(safe_mid, c.i32(1)), 4));
                Value* go_right = c.b.build_sle(end_k, k0);
                Value* nlo = c.b.build_select(c.b.build_and(active, go_right), c.b.build_add(safe_mid, c.i32(1)), slo);
                Value* nhi = c.b.build_select(c.b.build_and(active, c.b.build_not(go_right)), safe_mid, shi);
                return {nlo, nhi};
            });
        auto st1 = c.kb.for_range_reduce_n(
            c.i32(0), c.i32(24), c.i32(1), {c.i32(0), nseq},
            [&](Value* /*step*/, const std::vector<Value*>& cur) -> std::vector<Value*> {
                Value* slo = cur[0];
                Value* shi = cur[1];
                Value* active = c.b.build_slt(slo, shi);
                Value* mid = c.b.build_add(slo, c.b.build_ashr(c.b.build_sub(shi, slo), c.i32(1)));
                Value* safe_mid = c.b.build_select(active, mid, c.i32(0));
                Value* beg_k = c.kb.load_i32(ptr_at(c, aux2, safe_mid, 4));
                Value* go_right = c.b.build_slt(beg_k, kend);
                Value* nlo = c.b.build_select(c.b.build_and(active, go_right), c.b.build_add(safe_mid, c.i32(1)), slo);
                Value* nhi = c.b.build_select(c.b.build_and(active, c.b.build_not(go_right)), safe_mid, shi);
                return {nlo, nhi};
            });
        Value* has_seq = c.b.build_sgt(nseq, c.i32(0));
        Value* b0 = st0[0];
        Value* b1 = c.b.build_sub(st1[0], c.i32(1));
        Value* valid = c.b.build_and(has_seq, c.b.build_and(c.b.build_slt(b0, nseq), c.b.build_sle(b0, b1)));
        Value* sb0 = clamp_i32(c, b0, c.i32(0), smax_i32(c, c.i32(0), c.b.build_sub(nseq, c.i32(1))));
        Value* sb1 = clamp_i32(c, b1, c.i32(0), smax_i32(c, c.i32(0), c.b.build_sub(nseq, c.i32(1))));
        Value* qlo = clamp_i32(c, c.kb.load_i32(ptr_at(c, aux, sb0, 4)), c.i32(0), lq);
        Value* qhi = clamp_i32(c, c.kb.load_i32(ptr_at(c, aux, c.b.build_add(sb1, c.i32(1)), 4)), qlo, lq);
        return {c.b.build_select(valid, qlo, c.i32(0)), c.b.build_select(valid, qhi, c.i32(0))};
    }
    if (mode == FA_MODE_PACKED) {
        return {c.i32(0), lq};
    }
    // FA_MODE_ROWS
    Value* is_c = c.b.build_ne(causal, c.i32(0));
    Value* has_win = c.b.build_sgt(window, c.i32(0));
    Value* hw = c.b.build_ashr(window, c.i32(1));
    Value* qlo_c = smax_i32(c, c.i32(0), k0);
    Value* qlo_nc = c.b.build_select(has_win, smax_i32(c, c.i32(0), c.b.build_sub(k0, hw)), c.i32(0));
    Value* qlo = c.b.build_select(is_c, qlo_c, qlo_nc);
    Value* qhi_c = c.b.build_select(has_win, smin_i32(c, lq, c.b.build_add(kend, c.b.build_sub(window, c.i32(1)))), lq);
    Value* qhi_nc = c.b.build_select(has_win, smin_i32(c, lq, c.b.build_add(kend, hw)), lq);
    Value* qhi = c.b.build_select(is_c, qhi_c, qhi_nc);
    return {qlo, smax_i32(c, qlo, qhi)};
}

void compute_tile_sdp(Ctx& c, Dtype dt, Value* Qs, Value* Ks, Value* Gs, Value* Vs,
                      Value* Ps, Value* Ds, Value* scale,
                      std::uint32_t br, std::uint32_t bc, std::uint32_t hd, std::uint32_t hdp, std::uint32_t nt) {
    Value* tid = c.kb.tid_x();
    const std::uint32_t per = (br * bc) / nt;
    const std::uint32_t hd4 = hd & ~3u;
    for (std::uint32_t m = 0; m < per; ++m) {
        Value* cell = c.b.build_add(c.i32(static_cast<int>(m * nt)), tid);
        Value* r = c.b.build_udiv(cell, c.i32(static_cast<int>(bc)));
        Value* col = c.b.build_umod(cell, c.i32(static_cast<int>(bc)));
        Value* rb = c.b.build_mul(r, c.i32(static_cast<int>(hdp)));
        Value* cb = c.b.build_mul(col, c.i32(static_cast<int>(hdp)));

        std::vector<Value*> acc = {c.f32(0.0f), c.f32(0.0f)};
        if (hd4 > 0u) {
            acc = c.kb.for_range_reduce_n(
                c.i32(0), c.i32(static_cast<int>(hd4)), c.i32(4), acc,
                [&](Value* d, const std::vector<Value*>& cur) -> std::vector<Value*> {
                    Value* s = cur[0];
                    Value* dp = cur[1];
                    for (int e = 0; e < 4; ++e) {
                        Value* qi = c.b.build_add(rb, c.b.build_add(d, c.i32(e)));
                        Value* ki = c.b.build_add(cb, c.b.build_add(d, c.i32(e)));
                        Value* qv = load_stage_f32(c, dt, Qs, qi);
                        Value* kv = load_stage_f32(c, dt, Ks, ki);
                        Value* gv = load_stage_f32(c, dt, Gs, qi);
                        Value* vv = load_stage_f32(c, dt, Vs, ki);
                        s = c.b.build_fma_f32(qv, kv, s);
                        dp = c.b.build_fma_f32(gv, vv, dp);
                    }
                    return {s, dp};
                });
        }
        for (std::uint32_t d = hd4; d < hd; ++d) {
            Value* qi = c.b.build_add(rb, c.i32(static_cast<int>(d)));
            Value* ki = c.b.build_add(cb, c.i32(static_cast<int>(d)));
            acc[0] = c.b.build_fma_f32(load_stage_f32(c, dt, Qs, qi), load_stage_f32(c, dt, Ks, ki), acc[0]);
            acc[1] = c.b.build_fma_f32(load_stage_f32(c, dt, Gs, qi), load_stage_f32(c, dt, Vs, ki), acc[1]);
        }
        c.kb.shared_store_f32_indexed(Ps, cell, c.b.build_mul(acc[0], scale));
        c.kb.shared_store_f32_indexed(Ds, cell, acc[1]);
    }
}

void tree_reduce_min_max(Ctx& c, Value* s_ctrl, std::uint32_t nt) {
    Value* tid = c.kb.tid_x();
    for (std::uint32_t step = nt / 2u; step >= 1u; step >>= 1) {
        c.kb.sync();
        c.kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(step))), [&] {
            Value* o = c.b.build_add(tid, c.i32(static_cast<int>(step)));
            c.kb.shared_store_i32_indexed(
                s_ctrl, tid,
                smin_i32(c, c.kb.shared_load_i32_indexed(s_ctrl, tid), c.kb.shared_load_i32_indexed(s_ctrl, o)));
            Value* t2 = c.b.build_add(tid, c.i32(static_cast<int>(nt)));
            Value* o2 = c.b.build_add(o, c.i32(static_cast<int>(nt)));
            c.kb.shared_store_i32_indexed(
                s_ctrl, t2,
                smax_i32(c, c.kb.shared_load_i32_indexed(s_ctrl, t2), c.kb.shared_load_i32_indexed(s_ctrl, o2)));
        });
    }
    c.kb.sync();
}

void compute_ps_ds_weights(Ctx& c, Value* Ps, Value* Ds, Value* kval, Value* s_qlo, Value* s_qhi,
                           Value* s_lse, Value* s_D, Value* k0, std::uint32_t br, std::uint32_t bc,
                           std::uint32_t nt) {
    Value* tid = c.kb.tid_x();
    const std::uint32_t per = (br * bc) / nt;
    for (std::uint32_t m = 0; m < per; ++m) {
        Value* cell = c.b.build_add(c.i32(static_cast<int>(m * nt)), tid);
        Value* r = c.b.build_udiv(cell, c.i32(static_cast<int>(bc)));
        Value* col = c.b.build_umod(cell, c.i32(static_cast<int>(bc)));
        Value* gk = c.b.build_add(k0, col);
        Value* kv_ok = c.b.build_ne(c.kb.shared_load_i32_indexed(kval, col), c.i32(0));
        Value* lo_r = c.kb.shared_load_i32_indexed(s_qlo, r);
        Value* hi_r = c.kb.shared_load_i32_indexed(s_qhi, r);
        Value* in_r = c.b.build_and(c.b.build_sge(gk, lo_r), c.b.build_slt(gk, hi_r));
        Value* lse = c.kb.shared_load_f32_indexed(s_lse, r);
        Value* D_r = c.kb.shared_load_f32_indexed(s_D, r);
        Value* valid = c.b.build_and(c.b.build_and(kv_ok, in_r), c.b.build_slt(lse, c.f32(1e29f)));
        Value* sc = c.kb.shared_load_f32_indexed(Ps, cell);
        Value* dp = c.kb.shared_load_f32_indexed(Ds, cell);
        Value* pv = c.b.build_select(valid, c.kb.ex2_approx(c.b.build_sub(sc, lse)), c.f32(0.0f));
        Value* dsv = c.b.build_mul(pv, c.b.build_sub(dp, D_r));
        c.kb.shared_store_f32_indexed(Ps, cell, pv);
        c.kb.shared_store_f32_indexed(Ds, cell, dsv);
    }
}

}  // namespace

brass::Function* build_fa_bwd_mir(brass::Module& mod, Dtype dt,
                                  std::uint32_t nt, std::uint32_t hd, std::uint32_t hdp,
                                  std::uint32_t br, std::uint32_t bc,
                                  std::uint32_t mode, bool has_mask, std::uint32_t pass) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("brotensor: mir_fa_bwd: unsupported dtype");
    }
    if (nt == 0 || (nt & (nt - 1u)) != 0u || br == 0 || bc == 0 || hd == 0 || hdp < hd ||
        (br * hdp) % nt != 0u || (bc * hdp) % nt != 0u || (br * bc) % nt != 0u || nt < br) {
        throw std::runtime_error("brotensor: mir_fa_bwd: invalid tile geometry");
    }
    const std::string name = std::string("mir_fa_bwd_") +
                             (dt == Dtype::FP32 ? "f32" : dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_p" + std::to_string(pass) + "_nt" + std::to_string(nt) +
                             "_hd" + std::to_string(hd) + "_br" + std::to_string(br) +
                             "_bc" + std::to_string(bc) + "_m" + std::to_string(mode) +
                             (has_mask ? "_msk" : "");

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::f32(), Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: 0:q 1:k 2:v 3:g 4:mask 5:aux 6:aux2 7:stats 8:o0 9:o1
    // ints: 10:lq 11:lk 12:ldq 13:ldk 14:ldg 15:ldo 16:window 17:causal 18:nseq
    // flts: 19:scale 20:oscale

    Value* Qs = alloc_stage(c, dt, br * hdp);
    Value* Gs = alloc_stage(c, dt, br * hdp);
    Value* Ks = alloc_stage(c, dt, bc * hdp);
    Value* Vs = alloc_stage(c, dt, bc * hdp);
    Value* Ps = kb.shared_alloc_f32(br * bc);
    Value* Ds = kb.shared_alloc_f32(br * bc);
    Value* kval = kb.shared_alloc_i32(bc);
    Value* s_ctrl = kb.shared_alloc_i32(nt * 2u);
    Value* s_qlo = kb.shared_alloc_i32(br);
    Value* s_qhi = kb.shared_alloc_i32(br);
    Value* s_lse = kb.shared_alloc_f32(br);
    Value* s_D = kb.shared_alloc_f32(br);

    Value* tid = kb.tid_x();
    Value* h = kb.ctaid_y();
    Value* head_col = c.b.build_mul(h, c.i32(static_cast<int>(hd)));
    const int esz = elem_bytes(dt);

    if (pass == 0u) {
        Value* q0 = c.b.build_mul(kb.ctaid_x(), c.i32(static_cast<int>(br)));
        stage_tile(c, dt, Qs, p[0], p[12], q0, p[10], head_col, br, hd, hdp, nt);
        stage_tile(c, dt, Gs, p[3], p[14], q0, p[10], head_col, br, hd, hdp, nt);

        Value* def_lo = p[11];
        Value* def_hi = c.i32(0);
        kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(br))), [&] {
            Value* qi = c.b.build_add(q0, tid);
            Value* in_q = c.b.build_ult(qi, p[10]);
            auto [iv_lo, iv_hi] = emit_bw_interval(c, mode, qi, p[11], p[17], p[16], p[18], p[5], p[6]);
            Value* lo_r = c.b.build_select(in_q, iv_lo, c.i32(0));
            Value* hi_r = c.b.build_select(in_q, iv_hi, c.i32(0));
            kb.shared_store_i32_indexed(s_qlo, tid, lo_r);
            kb.shared_store_i32_indexed(s_qhi, tid, hi_r);
            Value* has_r = c.b.build_slt(lo_r, hi_r);
            kb.shared_store_i32_indexed(s_ctrl, tid, c.b.build_select(has_r, lo_r, def_lo));
            kb.shared_store_i32_indexed(
                s_ctrl, c.b.build_add(tid, c.i32(static_cast<int>(nt))),
                c.b.build_select(has_r, hi_r, def_hi));
        });
        kb.if_then(c.b.build_uge(tid, c.i32(static_cast<int>(br))), [&] {
            kb.shared_store_i32_indexed(s_ctrl, tid, def_lo);
            kb.shared_store_i32_indexed(s_ctrl, c.b.build_add(tid, c.i32(static_cast<int>(nt))), def_hi);
        });
        tree_reduce_min_max(c, s_ctrl, nt);

        Value* tile_lo = kb.shared_load_i32_indexed(s_ctrl, c.i32(0));
        Value* tile_hi = kb.shared_load_i32_indexed(s_ctrl, c.i32(static_cast<int>(nt)));

        // Sweep 1: online softmax statistics (m, l, D_num) per query row.
        std::vector<Value*> st_init = {c.f32(-1e30f), c.f32(0.0f), c.f32(0.0f)};
        std::vector<Value*> st = kb.for_range_reduce_n(
            tile_lo, tile_hi, c.i32(static_cast<int>(bc)), st_init,
            [&](Value* k0, const std::vector<Value*>& cur) -> std::vector<Value*> {
                stage_tile(c, dt, Ks, p[1], p[13], k0, p[11], head_col, bc, hd, hdp, nt);
                stage_tile(c, dt, Vs, p[2], p[13], k0, p[11], head_col, bc, hd, hdp, nt);
                stage_kval(c, has_mask, kval, p[4], k0, p[11], bc, nt);
                kb.sync();
                compute_tile_sdp(c, dt, Qs, Ks, Gs, Vs, Ps, Ds, p[19], br, bc, hd, hdp, nt);
                kb.sync();

                Value* is_row = c.b.build_and(c.b.build_ult(tid, c.i32(static_cast<int>(br))),
                                              c.b.build_ult(c.b.build_add(q0, tid), p[10]));
                std::vector<Value*> next = emit_if_else_values(
                    c, is_row, {Type::f32(), Type::f32(), Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* lo_r = kb.shared_load_i32_indexed(s_qlo, tid);
                        Value* hi_r = kb.shared_load_i32_indexed(s_qhi, tid);
                        Value* r_base = c.b.build_mul(tid, c.i32(static_cast<int>(bc)));
                        Value* m_tile = c.f32(-1e30f);
                        for (std::uint32_t col = 0; col < bc; ++col) {
                            Value* gk = c.b.build_add(k0, c.i32(static_cast<int>(col)));
                            Value* ok = c.b.build_and(
                                c.b.build_ne(kb.shared_load_i32_indexed(kval, c.i32(static_cast<int>(col))), c.i32(0)),
                                c.b.build_and(c.b.build_sge(gk, lo_r), c.b.build_slt(gk, hi_r)));
                            Value* sv = kb.shared_load_f32_indexed(Ps, c.b.build_add(r_base, c.i32(static_cast<int>(col))));
                            m_tile = c.b.build_select(ok, kb.fmax(m_tile, sv), m_tile);
                        }
                        Value* m_new = kb.fmax(cur[0], m_tile);
                        Value* has_any = c.b.build_sgt(m_new, c.f32(-1e29f));
                        Value* had_prev = c.b.build_sgt(cur[0], c.f32(-1e29f));
                        Value* alpha = c.b.build_select(
                            c.b.build_and(has_any, had_prev), kb.ex2_approx(c.b.build_sub(cur[0], m_new)), c.f32(0.0f));
                        Value* l_sum = c.f32(0.0f);
                        Value* d_sum = c.f32(0.0f);
                        for (std::uint32_t col = 0; col < bc; ++col) {
                            Value* gk = c.b.build_add(k0, c.i32(static_cast<int>(col)));
                            Value* ok = c.b.build_and(
                                c.b.build_ne(kb.shared_load_i32_indexed(kval, c.i32(static_cast<int>(col))), c.i32(0)),
                                c.b.build_and(c.b.build_sge(gk, lo_r), c.b.build_slt(gk, hi_r)));
                            Value* cidx = c.b.build_add(r_base, c.i32(static_cast<int>(col)));
                            Value* sv = kb.shared_load_f32_indexed(Ps, cidx);
                            Value* dpv = kb.shared_load_f32_indexed(Ds, cidx);
                            Value* pv = c.b.build_select(c.b.build_and(ok, has_any),
                                                         kb.ex2_approx(c.b.build_sub(sv, m_new)), c.f32(0.0f));
                            l_sum = c.b.build_add(l_sum, pv);
                            d_sum = c.b.build_fma_f32(pv, dpv, d_sum);
                        }
                        return {m_new, c.b.build_fma_f32(cur[1], alpha, l_sum), c.b.build_fma_f32(cur[2], alpha, d_sum)};
                    },
                    [&]() -> std::vector<Value*> { return cur; });
                kb.sync();
                return next;
            });

        kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(br))), [&] {
            Value* l_pos = c.b.build_sgt(st[1], c.f32(0.0f));
            Value* lse = c.b.build_select(
                l_pos, c.b.build_fma_f32(emit_log(c, c.b.build_select(l_pos, st[1], c.f32(1.0f))), c.f32(kLog2e), st[0]),
                c.f32(1e30f));
            Value* D = c.b.build_select(l_pos, emit_div_rn(c, st[2], c.b.build_select(l_pos, st[1], c.f32(1.0f))), c.f32(0.0f));
            kb.shared_store_f32_indexed(s_lse, tid, lse);
            kb.shared_store_f32_indexed(s_D, tid, D);
            Value* qi = c.b.build_add(q0, tid);
            kb.if_then(c.b.build_ult(qi, p[10]), [&] {
                Value* stat_idx = c.b.build_shl(c.b.build_add(c.b.build_mul(h, p[10]), qi), c.i32(1));
                kb.store_f32(ptr_at(c, p[7], stat_idx, 4), lse);
                kb.store_f32(ptr_at(c, p[7], c.b.build_add(stat_idx, c.i32(1)), 4), D);
            });
        });
        kb.sync();

        // Sweep 2: recompute S & dP and accumulate dQ.
        const std::uint32_t nq_e = (br * hdp) / nt;
        std::vector<Value*> dq_acc = kb.for_range_reduce_n(
            tile_lo, tile_hi, c.i32(static_cast<int>(bc)), std::vector<Value*>(nq_e, c.f32(0.0f)),
            [&](Value* k0, const std::vector<Value*>& cur) -> std::vector<Value*> {
                stage_tile(c, dt, Ks, p[1], p[13], k0, p[11], head_col, bc, hd, hdp, nt);
                stage_tile(c, dt, Vs, p[2], p[13], k0, p[11], head_col, bc, hd, hdp, nt);
                stage_kval(c, has_mask, kval, p[4], k0, p[11], bc, nt);
                kb.sync();
                compute_tile_sdp(c, dt, Qs, Ks, Gs, Vs, Ps, Ds, p[19], br, bc, hd, hdp, nt);
                kb.sync();
                compute_ps_ds_weights(c, Ps, Ds, kval, s_qlo, s_qhi, s_lse, s_D, k0, br, bc, nt);
                kb.sync();

                std::vector<Value*> next = cur;
                Value* tile_c = smin_i32(c, c.i32(static_cast<int>(bc)), c.b.build_sub(tile_hi, k0));
                for (std::uint32_t e = 0; e < nq_e; ++e) {
                    Value* idx = c.b.build_add(c.i32(static_cast<int>(e * nt)), tid);
                    Value* r = c.b.build_udiv(idx, c.i32(static_cast<int>(hdp)));
                    Value* d = c.b.build_umod(idx, c.i32(static_cast<int>(hdp)));
                    Value* rb = c.b.build_mul(r, c.i32(static_cast<int>(bc)));
                    next[e] = kb.for_range_reduce(
                        c.i32(0), tile_c, c.i32(1), cur[e],
                        [&](Value* col, Value* acc) -> Value* {
                            Value* dsv = kb.shared_load_f32_indexed(Ds, c.b.build_add(rb, col));
                            Value* kv = load_stage_f32(
                                c, dt, Ks, c.b.build_add(c.b.build_mul(col, c.i32(static_cast<int>(hdp))), d));
                            return c.b.build_fma_f32(dsv, kv, acc);
                        });
                }
                kb.sync();
                return next;
            });

        for (std::uint32_t e = 0; e < nq_e; ++e) {
            Value* idx = c.b.build_add(c.i32(static_cast<int>(e * nt)), tid);
            Value* r = c.b.build_udiv(idx, c.i32(static_cast<int>(hdp)));
            Value* d = c.b.build_umod(idx, c.i32(static_cast<int>(hdp)));
            Value* qr = c.b.build_add(q0, r);
            Value* ok = c.b.build_and(c.b.build_ult(qr, p[10]), c.b.build_ult(d, c.i32(static_cast<int>(hd))));
            kb.if_then(ok, [&] {
                Value* out_idx = c.b.build_add(c.b.build_add(c.b.build_mul(qr, p[15]), head_col), d);
                store_elem_dt(c, dt, p[8], out_idx, c.b.build_mul(dq_acc[e], p[20]));
            });
        }
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    // PASS 1: dK (o0), dV (o1)
    Value* k0 = c.b.build_mul(kb.ctaid_x(), c.i32(static_cast<int>(bc)));
    Value* kend = smin_i32(c, p[11], c.b.build_add(k0, c.i32(static_cast<int>(bc))));
    stage_tile(c, dt, Ks, p[1], p[13], k0, p[11], head_col, bc, hd, hdp, nt);
    stage_tile(c, dt, Vs, p[2], p[13], k0, p[11], head_col, bc, hd, hdp, nt);
    stage_kval(c, has_mask, kval, p[4], k0, p[11], bc, nt);
    kb.sync();

    auto [q_lo, q_hi] = emit_bw_query_range(c, mode, k0, kend, p[10], p[17], p[16], p[18], p[5], p[6]);
    const std::uint32_t nk_e = (bc * hdp) / nt;
    std::vector<Value*> kv_acc = kb.for_range_reduce_n(
        q_lo, q_hi, c.i32(static_cast<int>(br)), std::vector<Value*>(2u * nk_e, c.f32(0.0f)),
        [&](Value* q0, const std::vector<Value*>& cur) -> std::vector<Value*> {
            kb.shared_store_i32_indexed(s_ctrl, tid, c.i32(0));
            kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(br))), [&] {
                Value* qi = c.b.build_add(q0, tid);
                Value* in_q = c.b.build_ult(qi, p[10]);
                auto [iv_lo, iv_hi] = emit_bw_interval(c, mode, qi, p[11], p[17], p[16], p[18], p[5], p[6]);
                Value* lo_r = c.b.build_select(in_q, iv_lo, c.i32(0));
                Value* hi_r = c.b.build_select(in_q, iv_hi, c.i32(0));
                kb.shared_store_i32_indexed(s_qlo, tid, lo_r);
                kb.shared_store_i32_indexed(s_qhi, tid, hi_r);
                Value* ov = c.b.build_and(in_q, c.b.build_slt(smax_i32(c, k0, lo_r), smin_i32(c, kend, hi_r)));
                kb.shared_store_i32_indexed(s_ctrl, tid, c.b.build_select(ov, c.i32(1), c.i32(0)));
                Value* lse = emit_if_else_values(
                    c, in_q, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* si = c.b.build_shl(c.b.build_add(c.b.build_mul(h, p[10]), qi), c.i32(1));
                        return {kb.load_f32(ptr_at(c, p[7], si, 4))};
                    },
                    [&]() -> std::vector<Value*> { return {c.f32(1e30f)}; })[0];
                Value* Dv = emit_if_else_values(
                    c, in_q, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        Value* si = c.b.build_shl(c.b.build_add(c.b.build_mul(h, p[10]), qi), c.i32(1));
                        return {kb.load_f32(ptr_at(c, p[7], c.b.build_add(si, c.i32(1)), 4))};
                    },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                kb.shared_store_f32_indexed(s_lse, tid, lse);
                kb.shared_store_f32_indexed(s_D, tid, Dv);
            });
            for (std::uint32_t step = nt / 2u; step >= 1u; step >>= 1) {
                kb.sync();
                kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(step))), [&] {
                    Value* a = kb.shared_load_i32_indexed(s_ctrl, tid);
                    Value* b = kb.shared_load_i32_indexed(s_ctrl, c.b.build_add(tid, c.i32(static_cast<int>(step))));
                    kb.shared_store_i32_indexed(s_ctrl, tid, c.b.build_or(a, b));
                });
            }
            kb.sync();

            Value* any_q = c.b.build_ne(kb.shared_load_i32_indexed(s_ctrl, c.i32(0)), c.i32(0));
            kb.if_then(any_q, [&] {
                stage_tile(c, dt, Qs, p[0], p[12], q0, p[10], head_col, br, hd, hdp, nt);
                stage_tile(c, dt, Gs, p[3], p[14], q0, p[10], head_col, br, hd, hdp, nt);
            });
            kb.sync();

            kb.if_then(any_q, [&] { compute_tile_sdp(c, dt, Qs, Ks, Gs, Vs, Ps, Ds, p[19], br, bc, hd, hdp, nt); });
            kb.sync();

            kb.if_then(any_q, [&] { compute_ps_ds_weights(c, Ps, Ds, kval, s_qlo, s_qhi, s_lse, s_D, k0, br, bc, nt); });
            kb.sync();

            std::vector<Value*> next = emit_if_else_values(
                c, any_q, std::vector<Type>(2u * nk_e, Type::f32()),
                [&]() -> std::vector<Value*> {
                    std::vector<Value*> out = cur;
                    Value* tile_r = smin_i32(c, c.i32(static_cast<int>(br)), c.b.build_sub(q_hi, q0));
                    for (std::uint32_t e = 0; e < nk_e; ++e) {
                        Value* idx = c.b.build_add(c.i32(static_cast<int>(e * nt)), tid);
                        Value* col = c.b.build_udiv(idx, c.i32(static_cast<int>(hdp)));
                        Value* d = c.b.build_umod(idx, c.i32(static_cast<int>(hdp)));
                        auto res = kb.for_range_reduce_n(
                            c.i32(0), tile_r, c.i32(1), {cur[e], cur[nk_e + e]},
                            [&](Value* r, const std::vector<Value*>& acc) -> std::vector<Value*> {
                                Value* cell = c.b.build_add(c.b.build_mul(r, c.i32(static_cast<int>(bc))), col);
                                Value* pv = kb.shared_load_f32_indexed(Ps, cell);
                                Value* dsv = kb.shared_load_f32_indexed(Ds, cell);
                                Value* sidx = c.b.build_add(c.b.build_mul(r, c.i32(static_cast<int>(hdp))), d);
                                Value* qv = load_stage_f32(c, dt, Qs, sidx);
                                Value* gv = load_stage_f32(c, dt, Gs, sidx);
                                return {c.b.build_fma_f32(dsv, qv, acc[0]), c.b.build_fma_f32(pv, gv, acc[1])};
                            });
                        out[e] = res[0];
                        out[nk_e + e] = res[1];
                    }
                    return out;
                },
                [&]() -> std::vector<Value*> { return cur; });
            kb.sync();
            return next;
        });

    for (std::uint32_t e = 0; e < nk_e; ++e) {
        Value* idx = c.b.build_add(c.i32(static_cast<int>(e * nt)), tid);
        Value* col = c.b.build_udiv(idx, c.i32(static_cast<int>(hdp)));
        Value* d = c.b.build_umod(idx, c.i32(static_cast<int>(hdp)));
        Value* kr = c.b.build_add(k0, col);
        Value* ok = c.b.build_and(c.b.build_ult(kr, p[11]), c.b.build_ult(d, c.i32(static_cast<int>(hd))));
        kb.if_then(ok, [&] {
            Value* out_idx = c.b.build_add(c.b.build_add(c.b.build_mul(kr, p[15]), head_col), d);
            store_elem_dt(c, dt, p[8], out_idx, c.b.build_mul(kv_acc[e], p[20]));
            store_elem_dt(c, dt, p[9], out_idx, kv_acc[nk_e + e]);
        });
    }
    (void)esz;
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
