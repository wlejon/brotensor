#include "mir_fa_bwd_cm.h"

#include "mir_common.h"
#include "mir_coop_common.h"
#include "../vulkan/shaders/op_codes.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

std::pair<Value*, Value*> emit_bw_cm_interval(Ctx& c, std::uint32_t mode, Value* qi,
                                              Value* lq, Value* lk, Value* window, Value* causal,
                                              Value* nseq, Value* aux, Value* aux2) {
    auto res = emit_if_else_values(
        c, c.b.build_ult(qi, lq), {Type::i32(), Type::i32()},
        [&]() -> std::vector<Value*> {
            Value* a = c.i32(0);
            Value* b = lk;
            if (mode == FA_MODE_ROWS) {
                Value* is_c = c.b.build_ne(causal, c.i32(0));
                Value* has_w = c.b.build_ugt(window, c.i32(0));
                Value* b_c = smin_i32(c, lk, c.b.build_add(qi, c.i32(1)));
                Value* a_cw = smax_i32(c, c.i32(0), c.b.build_add(c.b.build_sub(qi, window), c.i32(1)));
                Value* a_c = c.b.build_select(has_w, a_cw, c.i32(0));
                Value* w2 = c.b.build_lshr(window, c.i32(1));
                Value* a_nc = c.b.build_select(has_w, smax_i32(c, c.i32(0), c.b.build_sub(qi, w2)), c.i32(0));
                Value* b_nc = c.b.build_select(has_w, smin_i32(c, lk, c.b.build_add(c.b.build_add(qi, w2), c.i32(1))), lk);
                a = c.b.build_select(is_c, a_c, a_nc);
                b = c.b.build_select(is_c, b_c, b_nc);
            } else if (mode == FA_MODE_VARLEN) {
                Value* valid = emit_if_else_values(
                    c, c.b.build_ugt(nseq, c.i32(0)), {Type::i32()},
                    [&]() -> std::vector<Value*> {
                        return {c.b.build_slt(qi, c.kb.load_i32(ptr_at(c, aux, nseq, 4)))};
                    },
                    [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                auto ab = emit_if_else_values(
                    c, valid, {Type::i32(), Type::i32()},
                    [&]() -> std::vector<Value*> {
                        auto bs = c.kb.for_range_reduce_n(
                            c.i32(0), c.i32(24), c.i32(1), {c.i32(0), nseq},
                            [&](Value*, const std::vector<Value*>& st) -> std::vector<Value*> {
                                Value* lo = st[0];
                                Value* hi = st[1];
                                Value* gt1 = c.b.build_ugt(c.b.build_sub(hi, lo), c.i32(1));
                                Value* mid = c.b.build_lshr(c.b.build_add(lo, hi), c.i32(1));
                                Value* le = c.b.build_sle(c.kb.load_i32(ptr_at(c, aux, mid, 4)), qi);
                                Value* nlo = c.b.build_select(c.b.build_and(gt1, le), mid, lo);
                                Value* nhi = c.b.build_select(c.b.build_and(gt1, c.b.build_xor(le, c.i32(1))), mid, hi);
                                return {nlo, nhi};
                            });
                        Value* s0 = bs[0];
                        Value* ka = c.kb.load_i32(ptr_at(c, aux2, s0, 4));
                        Value* kb_v = c.kb.load_i32(ptr_at(c, aux2, c.b.build_add(s0, c.i32(1)), 4));
                        Value* qstart = c.kb.load_i32(ptr_at(c, aux, s0, 4));
                        Value* kb_c = smin_i32(c, kb_v, c.b.build_add(c.b.build_add(ka, c.b.build_sub(qi, qstart)), c.i32(1)));
                        return {ka, c.b.build_select(c.b.build_ne(causal, c.i32(0)), kb_c, kb_v)};
                    },
                    [&]() -> std::vector<Value*> { return {c.i32(0), c.i32(0)}; });
                a = ab[0];
                b = ab[1];
            } else if (mode == FA_MODE_PACKED) {
                Value* q2 = c.b.build_shl(qi, c.i32(1));
                a = c.kb.load_i32(ptr_at(c, aux, q2, 4));
                b = c.kb.load_i32(ptr_at(c, aux, c.b.build_add(q2, c.i32(1)), 4));
                Value* has_w = c.b.build_ugt(window, c.i32(0));
                Value* w2 = c.b.build_lshr(window, c.i32(1));
                a = c.b.build_select(has_w, smax_i32(c, a, c.b.build_sub(qi, w2)), a);
                b = c.b.build_select(has_w, smin_i32(c, b, c.b.build_add(c.b.build_add(qi, w2), c.i32(1))), b);
            }
            a = clamp_i32(c, a, c.i32(0), lk);
            b = clamp_i32(c, b, a, lk);
            return {a, b};
        },
        [&]() -> std::vector<Value*> { return {c.i32(0), c.i32(0)}; });
    return {res[0], res[1]};
}

std::pair<Value*, Value*> emit_bw_cm_qrange(Ctx& c, std::uint32_t mode, Value* k0, Value* k1,
                                            Value* lq, Value* window, Value* causal, Value* nseq,
                                            Value* aux, Value* aux2) {
    if (mode == FA_MODE_ROWS) {
        Value* is_c = c.b.build_ne(causal, c.i32(0));
        Value* has_w = c.b.build_ugt(window, c.i32(0));
        Value* qhi_cw = smin_i32(c, lq, c.b.build_add(k1, window));
        Value* w2 = c.b.build_lshr(window, c.i32(1));
        Value* qlo_nc = c.b.build_select(has_w, smax_i32(c, c.i32(0), c.b.build_sub(k0, w2)), c.i32(0));
        Value* qhi_nc = c.b.build_select(has_w, smin_i32(c, lq, c.b.build_add(c.b.build_add(k1, w2), c.i32(1))), lq);
        return {c.b.build_select(is_c, k0, qlo_nc),
                c.b.build_select(is_c, c.b.build_select(has_w, qhi_cw, lq), qhi_nc)};
    }
    if (mode == FA_MODE_VARLEN) {
        auto find_seq = [&](Value* key) -> Value* {
            auto bs = c.kb.for_range_reduce_n(
                c.i32(0), c.i32(24), c.i32(1), {c.i32(0), nseq},
                [&](Value*, const std::vector<Value*>& st) -> std::vector<Value*> {
                    Value* lo = st[0];
                    Value* hi = st[1];
                    Value* gt1 = c.b.build_ugt(c.b.build_sub(hi, lo), c.i32(1));
                    Value* mid = c.b.build_lshr(c.b.build_add(lo, hi), c.i32(1));
                    Value* le = c.b.build_sle(c.kb.load_i32(ptr_at(c, aux2, mid, 4)), key);
                    Value* nlo = c.b.build_select(c.b.build_and(gt1, le), mid, lo);
                    Value* nhi = c.b.build_select(c.b.build_and(gt1, c.b.build_xor(le, c.i32(1))), mid, hi);
                    return {nlo, nhi};
                });
            return bs[0];
        };
        auto res = emit_if_else_values(
            c, c.b.build_ugt(nseq, c.i32(0)), {Type::i32(), Type::i32()},
            [&]() -> std::vector<Value*> {
                Value* sa = find_seq(k0);
                Value* sb = find_seq(k1);
                Value* qlo = clamp_i32(c, c.kb.load_i32(ptr_at(c, aux, sa, 4)), c.i32(0), lq);
                Value* qhi = clamp_i32(c, c.kb.load_i32(ptr_at(c, aux, c.b.build_add(sb, c.i32(1)), 4)), qlo, lq);
                return {qlo, qhi};
            },
            [&]() -> std::vector<Value*> { return {c.i32(0), c.i32(0)}; });
        return {res[0], res[1]};
    }
    return {c.i32(0), lq};
}

Value* load8_bwd(Ctx& c, Dtype dt, std::uint32_t hd, bool vec, Value* base, Value* off, Value* ok, Value* col_v4) {
    if (vec) {
        Value* idx = c.b.build_select(ok, c.b.build_add(off, c.b.build_shl(col_v4, c.i32(3))), c.i32(0));
        Value* w = c.kb.vload(Type::i32x4(), ptr_at(c, base, idx, 2));
        if (dt == Dtype::BF16) w = emit_bf16x8_to_f16x8(c, w);
        return emit_v4_select(c, ok, w, emit_v4_zero(c));
    }
    return emit_if_else_values(
        c, ok, {Type::i32x4()},
        [&]() -> std::vector<Value*> {
            Value* c0 = c.b.build_shl(col_v4, c.i32(3));
            std::array<Value*, 8> h{};
            for (int e = 0; e < 8; ++e) {
                Value* ce = c.b.build_add(c0, c.i32(e));
                h[static_cast<std::size_t>(e)] = emit_if_else_values(
                    c, c.b.build_ult(ce, c.i32(static_cast<int>(hd))), {Type::f32()},
                    [&]() -> std::vector<Value*> { return {load_elem_dt(c, dt, base, c.b.build_add(off, ce))}; },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            }
            return {emit_pack_f32x8_to_f16_v4(c, h)};
        },
        [&]() -> std::vector<Value*> { return {emit_v4_zero(c)}; })[0];
}

}  // namespace

brass::Function* build_fa_bwd_cm_mir(brass::Module& mod, const FaBwdCmSpec& s) {
    if ((s.nsg != 2u && s.nsg != 4u) || (s.bn != 16u && s.bn != 32u) || s.pass > 2u) {
        throw std::runtime_error("brotensor: mir_fa_bwd_cm: invalid geometry/pass");
    }
    const std::uint32_t HD = s.hd;
    const std::uint32_t DP = ((HD + 15u) / 16u) * 16u;
    const std::uint32_t ND = DP / 16u;
    const std::uint32_t DV = DP / 8u;
    const std::uint32_t DVR = (HD + 7u) / 8u;
    const std::uint32_t SD = DV + 1u;
    const std::uint32_t BO = s.nsg * 16u;
    const std::uint32_t WG = s.nsg * 32u;
    const std::uint32_t NJ = s.bn / 16u;
    const std::uint32_t HS = s.bn + 8u;
    const std::uint32_t HS_V4 = HS / 8u;
    const std::uint32_t FS = s.bn + 4u;
    const std::uint32_t KVO = s.areg ? 0u : (2u * BO * SD);
    const std::uint32_t NSM = 2u * (s.areg ? std::max(BO, s.bn) : (BO + s.bn)) * SD;
    const std::uint32_t NSF = (s.pass == 0u) ? std::max(256u * s.nsg, 2u * BO * FS) : (256u * s.nsg);
    const std::uint32_t NSH_V4 = (s.pass == 0u) ? 1u : ((s.pass == 1u ? 2u : 1u) * BO * HS_V4);

    const std::string name = std::string("mir_fa_bwd_cm_") + (s.dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_p" + std::to_string(s.pass) + "_hd" + std::to_string(HD) +
                             "_g" + std::to_string(s.nsg) + "_bn" + std::to_string(s.bn) +
                             "_m" + std::to_string(s.mode) + (s.has_mask ? "_mk" : "") +
                             (s.vec ? "_v" : "_s") + (s.areg ? "_ar" : "") + (s.agnostic ? "_ag" : "");

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

    Value* SM = kb.shared_alloc_i32x4(NSM);
    Value* SF = kb.shared_alloc_f32(NSF);
    Value* SH = (s.pass != 0u) ? kb.shared_alloc_i32x4(NSH_V4) : nullptr;
    Value* ilo = kb.shared_alloc_i32(BO + s.bn);
    Value* ihi = kb.shared_alloc_i32(BO + s.bn);
    Value* ist_x = kb.shared_alloc_f32(BO + s.bn);
    Value* ist_y = kb.shared_alloc_f32(BO + s.bn);
    Value* kok = kb.shared_alloc_f32(BO + s.bn);
    Value* s_lo = (s.pass != 1u) ? kb.shared_alloc_i32(WG) : nullptr;
    Value* s_hi = (s.pass != 1u) ? kb.shared_alloc_i32(WG) : nullptr;
    Value* sflag = (s.pass == 1u) ? kb.shared_alloc_i32(s.nsg) : nullptr;
    Value* s_fast = (!s.agnostic) ? kb.shared_alloc_i32(1) : nullptr;

    Value* tid = c.b.build_and(kb.tid_x(), c.i32(static_cast<int>(WG - 1u)));
    Value* sg = c.b.build_and(kb.subgroup_id(), c.i32(static_cast<int>(s.nsg - 1u)));
    Value* lane = c.b.build_and(kb.lane_id(), c.i32(31));
    Value* cl = c.b.build_and(lane, c.i32(15));
    Value* h = kb.ctaid_y();
    Value* col_h = c.b.build_mul(h, c.i32(static_cast<int>(HD)));
    Value* stride_sd = c.i32(static_cast<int>(SD * 8u));
    Value* stride_hs = c.i32(static_cast<int>(HS));

    auto stage = [&](std::uint32_t dst_v4, std::uint32_t nrows, Value* base, Value* ld, Value* row0, Value* rowend) {
        const std::uint32_t tot = nrows * DV;
        for (std::uint32_t i0 = 0; i0 < tot; i0 += WG) {
            Value* i = (i0 == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(i0)));
            auto body = [&] {
                Value* rr = c.b.build_udiv(i, c.i32(static_cast<int>(DV)));
                Value* cc = c.b.build_umod(i, c.i32(static_cast<int>(DV)));
                Value* row = c.b.build_add(row0, rr);
                Value* ok = c.b.build_and(c.b.build_ult(row, rowend), c.b.build_ult(cc, c.i32(static_cast<int>(DVR))));
                Value* v = load8_bwd(c, s.dt, HD, s.vec, base, c.b.build_add(c.b.build_mul(row, ld), col_h), ok, cc);
                Value* at = c.b.build_add(c.i32(static_cast<int>(dst_v4)),
                                          c.b.build_add(c.b.build_mul(rr, c.i32(static_cast<int>(SD))), cc));
                kb.shared_store_i32x4_indexed(SM, at, v);
            };
            if (i0 + WG <= tot) body();
            else kb.if_then(c.b.build_ult(i, c.i32(static_cast<int>(tot))), body);
        }
    };

    auto key_ok_f32 = [&](Value* key) -> Value* {
        if (!s.has_mask) return c.b.build_select(c.b.build_ult(key, p[11]), c.f32(1.0f), c.f32(0.0f));
        return emit_if_else_values(
            c, c.b.build_ult(key, p[11]), {Type::f32()},
            [&]() -> std::vector<Value*> {
                Value* mv = kb.load_f32(ptr_at(c, p[4], key, 4));
                return {c.b.build_select(c.b.build_sgt(mv, c.f32(0.5f)), c.f32(1.0f), c.f32(0.0f))};
            },
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
    };

    auto stat_at = [&](Value* q) -> Value* {
        return c.b.build_shl(c.b.build_add(c.b.build_mul(h, p[10]), q), c.i32(1));
    };

    auto store_tile = [&](Value* a_frag, Value* sbase_f, Value* out_base, Value* row0, Value* rowend,
                          std::uint32_t c0, Value* fac) {
        kb.coop_store_acc(a_frag, c.b.build_add(SF, c.b.build_shl(sbase_f, c.i32(2))), c.i32(16), false, false);
        kb.subgroup_barrier();
        for (int e0 = 0; e0 < 256; e0 += 32) {
            Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
            Value* row = c.b.build_add(row0, c.b.build_lshr(e, c.i32(4)));
            Value* col = c.b.build_add(c.i32(static_cast<int>(c0)), c.b.build_and(e, c.i32(15)));
            kb.if_then(c.b.build_and(c.b.build_ult(row, rowend), c.b.build_ult(col, c.i32(static_cast<int>(HD)))), [&] {
                Value* val = c.b.build_mul(kb.shared_load_f32_indexed(SF, c.b.build_add(sbase_f, e)), fac);
                store_elem_dt(c, s.dt, out_base, c.b.build_add(c.b.build_add(c.b.build_mul(row, p[15]), col_h), col), val);
            });
        }
        kb.subgroup_barrier();
    };

    // Populate IX table SF[0..255] = float(0..255).
    for (std::uint32_t e0 = 0; e0 < 256u; e0 += WG) {
        Value* e = (e0 == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(e0)));
        kb.shared_store_f32_indexed(SF, e, kb.u32_to_f32(e));
    }

    auto check_cluster_layout = [&](Value* IX) -> Value* {
        if (s.agnostic) return c.i32(0);
        Value* len_ok = c.b.build_eq(kb.coop_acc_len(IX), c.i32(8));
        Value* ok = emit_if_else_values(
            c, len_ok, {Type::i32()},
            [&]() -> std::vector<Value*> {
                Value* all_ok = c.i32(1);
                for (int i = 0; i < 8; ++i) {
                    Value* ev = kb.f32_to_u32(kb.coop_get_acc(IX, c.i32(i)));
                    Value* rw = c.b.build_lshr(ev, c.i32(4));
                    Value* col_ok = c.b.build_eq(c.b.build_and(ev, c.i32(15)), cl);
                    Value* row_eq = c.b.build_eq(emit_cluster16_min_i32(c, rw), emit_cluster16_max_i32(c, rw));
                    all_ok = c.b.build_and(all_ok, c.b.build_and(col_ok, row_eq));
                }
                return {all_ok};
            },
            [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
        Value* raw_fast = emit_subgroup32_all_i32(c, ok);
        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            kb.shared_store_i32_indexed(s_fast, c.i32(0), c.b.build_select(raw_fast, c.i32(1), c.i32(0)));
        });
        kb.sync();
        return c.b.build_ne(kb.shared_load_i32_indexed(s_fast, c.i32(0)), c.i32(0));
    };

    Value* IX = kb.coop_alloc_acc();
    Value* sbase = c.b.build_shl(sg, c.i32(8));  // sg * 256

    if (s.pass != 1u) {
        // ── Query-major: PASS 0 (stats) or PASS 2 (dQ) ──
        Value* r0 = c.b.build_mul(c.b.build_sub(c.b.build_sub(kb.nctaid_x(), c.i32(1)), kb.ctaid_x()),
                                  c.i32(static_cast<int>(BO)));
        auto row_int = emit_if_else_values(
            c, c.b.build_ult(tid, c.i32(static_cast<int>(BO))), {Type::i32(), Type::i32()},
            [&]() -> std::vector<Value*> {
                Value* q = c.b.build_add(r0, tid);
                auto [lo_v, hi_v] = emit_bw_cm_interval(c, s.mode, q, p[10], p[11], p[16], p[17], p[18], p[5], p[6]);
                kb.shared_store_i32_indexed(ilo, tid, lo_v);
                kb.shared_store_i32_indexed(ihi, tid, hi_v);
                if (s.pass == 2u) {
                    auto st_v = emit_if_else_values(
                        c, c.b.build_ult(q, p[10]), {Type::f32(), Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* se = stat_at(q);
                            return {kb.load_f32(ptr_at(c, p[7], se, 4)),
                                    kb.load_f32(ptr_at(c, p[7], c.b.build_add(se, c.i32(1)), 4))};
                        },
                        [&]() -> std::vector<Value*> { return {c.f32(0.0f), c.f32(0.0f)}; });
                    kb.shared_store_f32_indexed(ist_x, tid, st_v[0]);
                    kb.shared_store_f32_indexed(ist_y, tid, st_v[1]);
                }
                return {lo_v, hi_v};
            },
            [&]() -> std::vector<Value*> { return {c.i32(0), c.i32(0)}; });
        stage(0u, BO, p[0], p[12], r0, p[10]);
        stage(BO * SD, BO, p[3], p[14], r0, p[10]);

        Value* has_k = c.b.build_slt(row_int[0], row_int[1]);
        kb.shared_store_i32_indexed(s_lo, tid, c.b.build_select(has_k, row_int[0], c.i32(static_cast<int>(0x7fffffff))));
        kb.shared_store_i32_indexed(s_hi, tid, c.b.build_select(has_k, row_int[1], c.i32(0)));
        kb.sync();
        for (std::uint32_t st = WG / 2u; st >= 1u; st >>= 1) {
            kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(st))), [&] {
                Value* t2 = c.b.build_add(tid, c.i32(static_cast<int>(st)));
                kb.shared_store_i32_indexed(s_lo, tid, smin_i32(c, kb.shared_load_i32_indexed(s_lo, tid),
                                                                kb.shared_load_i32_indexed(s_lo, t2)));
                kb.shared_store_i32_indexed(s_hi, tid, smax_i32(c, kb.shared_load_i32_indexed(s_hi, tid),
                                                                kb.shared_load_i32_indexed(s_hi, t2)));
            });
            kb.sync();
        }

        kb.coop_load_acc(IX, SF, c.i32(16), false, false);
        std::vector<Value*> fq(ND), fg(ND);
        for (std::uint32_t dc = 0; dc < ND; ++dc) {
            fq[dc] = kb.coop_alloc_a();
            fg[dc] = kb.coop_alloc_a();
            if (s.areg) {
                Value* q_off = c.b.build_add(c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(SD))),
                                             c.i32(static_cast<int>(dc * 2u)));
                kb.coop_load_a(fq[dc], c.b.build_add(SM, c.b.build_shl(q_off, c.i32(4))), stride_sd, false);
                kb.coop_load_a(fg[dc], c.b.build_add(SM, c.b.build_shl(c.b.build_add(q_off, c.i32(static_cast<int>(BO * SD))), c.i32(4))),
                               stride_sd, false);
            }
        }
        Value* fast = check_cluster_layout(IX);
        std::array<Value*, 8> flo{}, fhi{}, fst_x{}, fst_y{};
        if (!s.agnostic) {
            for (int i = 0; i < 8; ++i) {
                Value* ri = emit_if_else_values(
                    c, fast, {Type::i32()},
                    [&]() -> std::vector<Value*> {
                        Value* r_local = c.b.build_and(c.b.build_lshr(kb.f32_to_u32(kb.coop_get_acc(IX, c.i32(i))), c.i32(4)),
                                                       c.i32(15));
                        return {c.b.build_add(c.b.build_shl(sg, c.i32(4)), r_local)};
                    },
                    [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                flo[static_cast<std::size_t>(i)] = kb.shared_load_i32_indexed(ilo, ri);
                fhi[static_cast<std::size_t>(i)] = kb.shared_load_i32_indexed(ihi, ri);
                fst_x[static_cast<std::size_t>(i)] = (s.pass == 2u) ? kb.shared_load_f32_indexed(ist_x, ri) : c.f32(-1e30f);
                fst_y[static_cast<std::size_t>(i)] = (s.pass == 2u) ? kb.shared_load_f32_indexed(ist_y, ri) : c.f32(0.0f);
            }
        }
        kb.sync();

        Value* kbeg = kb.shared_load_i32_indexed(s_lo, c.i32(0));
        Value* kend = kb.shared_load_i32_indexed(s_hi, c.i32(0));

        Value* fk_frag = kb.coop_alloc_b();
        Value* fv_frag = kb.coop_alloc_b();
        std::vector<Value*> s_fr(NJ), dp_fr(NJ);
        for (std::uint32_t j = 0; j < NJ; ++j) {
            s_fr[j] = kb.coop_alloc_acc();
            dp_fr[j] = kb.coop_alloc_acc();
        }
        auto tile_sdp = [&] {
            for (std::uint32_t j = 0; j < NJ; ++j) {
                kb.coop_fill_acc(s_fr[j], c.f32(0.0f));
                kb.coop_fill_acc(dp_fr[j], c.f32(0.0f));
                for (std::uint32_t dc = 0; dc < ND; ++dc) {
                    if (!s.areg) {
                        Value* q_off = c.b.build_add(c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(SD))),
                                                     c.i32(static_cast<int>(dc * 2u)));
                        kb.coop_load_a(fq[dc], c.b.build_add(SM, c.b.build_shl(q_off, c.i32(4))), stride_sd, false);
                        kb.coop_load_a(fg[dc], c.b.build_add(SM, c.b.build_shl(c.b.build_add(q_off, c.i32(static_cast<int>(BO * SD))), c.i32(4))),
                                       stride_sd, false);
                    }
                    Value* k_off = c.i32(static_cast<int>(KVO + j * 16u * SD + dc * 2u));
                    Value* v_off = c.i32(static_cast<int>(KVO + s.bn * SD + j * 16u * SD + dc * 2u));
                    kb.coop_load_b(fk_frag, c.b.build_add(SM, c.b.build_shl(k_off, c.i32(4))), stride_sd, true);
                    kb.coop_load_b(fv_frag, c.b.build_add(SM, c.b.build_shl(v_off, c.i32(4))), stride_sd, true);
                    kb.coop_mma(s_fr[j], fq[dc], fk_frag);
                    kb.coop_mma(dp_fr[j], fg[dc], fv_frag);
                }
            }
        };

        if (s.pass == 0u) {
            Value* r = c.b.build_lshr(lane, c.i32(1));
            Value* half_ = c.b.build_and(lane, c.i32(1));
            Value* rl = c.b.build_add(c.b.build_shl(sg, c.i32(4)), r);
            Value* c0 = c.b.build_mul(half_, c.i32(static_cast<int>(s.bn / 2u)));
            Value* alo = kb.shared_load_i32_indexed(ilo, rl);
            Value* ahi = kb.shared_load_i32_indexed(ihi, rl);
            Value* fsb = c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(FS)));
            Value* fdb = c.b.build_add(c.i32(static_cast<int>(BO * FS)), fsb);

            // Loop state: agnostic {m, l, dacc} (3) + clustered {m[8], l[8], d[8]} (24 when !agnostic)
            std::vector<Value*> init_st = {c.f32(-1e30f), c.f32(0.0f), c.f32(0.0f)};
            if (!s.agnostic) {
                for (int i = 0; i < 8; ++i) init_st.push_back(c.f32(-1e30f));
                for (int i = 0; i < 8; ++i) init_st.push_back(c.f32(0.0f));
                for (int i = 0; i < 8; ++i) init_st.push_back(c.f32(0.0f));
            }
            auto fin_st = kb.for_range_reduce_n(
                kbeg, kend, c.i32(static_cast<int>(s.bn)), init_st,
                [&](Value* kb_v, const std::vector<Value*>& st) -> std::vector<Value*> {
                    kb.sync();
                    stage(KVO, s.bn, p[1], p[13], kb_v, p[11]);
                    stage(KVO + s.bn * SD, s.bn, p[2], p[13], kb_v, p[11]);
                    kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(s.bn))), [&] {
                        kb.shared_store_f32_indexed(kok, tid, key_ok_f32(c.b.build_add(kb_v, tid)));
                    });
                    kb.sync();
                    tile_sdp();

                    if (s.agnostic) {
                        for (std::uint32_t j = 0; j < NJ; ++j) {
                            kb.coop_store_acc(s_fr[j], c.b.build_add(SF, c.b.build_shl(c.b.build_add(fsb, c.i32(static_cast<int>(j * 16u))), c.i32(2))),
                                              c.i32(static_cast<int>(FS)), false, false);
                            kb.coop_store_acc(dp_fr[j], c.b.build_add(SF, c.b.build_shl(c.b.build_add(fdb, c.i32(static_cast<int>(j * 16u))), c.i32(2))),
                                              c.i32(static_cast<int>(FS)), false, false);
                        }
                        kb.subgroup_barrier();
                        Value* row_off = c.b.build_mul(r, c.i32(static_cast<int>(FS)));
                        Value* mx = c.f32(-1e30f);
                        for (std::uint32_t ci = 0; ci < s.bn / 2u; ++ci) {
                            Value* col = c.b.build_add(c0, c.i32(static_cast<int>(ci)));
                            Value* key = c.b.build_add(kb_v, col);
                            Value* ok = c.b.build_and(c.b.build_and(c.b.build_sge(key, alo), c.b.build_slt(key, ahi)),
                                                      c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f)));
                            Value* sv = c.b.build_mul(kb.shared_load_f32_indexed(SF, c.b.build_add(c.b.build_add(fsb, row_off), col)), p[19]);
                            mx = c.b.build_select(ok, kb.fmax(mx, sv), mx);
                        }
                        mx = kb.fmax(mx, kb.shfl_bfly_f32(mx, 1u));
                        Value* mnew = kb.fmax(st[0], mx);
                        Value* sum = c.f32(0.0f);
                        Value* ds = c.f32(0.0f);
                        for (std::uint32_t ci = 0; ci < s.bn / 2u; ++ci) {
                            Value* col = c.b.build_add(c0, c.i32(static_cast<int>(ci)));
                            Value* key = c.b.build_add(kb_v, col);
                            Value* ok = c.b.build_and(c.b.build_and(c.b.build_sge(key, alo), c.b.build_slt(key, ahi)),
                                                      c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f)));
                            Value* sv = c.b.build_mul(kb.shared_load_f32_indexed(SF, c.b.build_add(c.b.build_add(fsb, row_off), col)), p[19]);
                            Value* pv = c.b.build_select(ok, kb.ex2_approx(c.b.build_sub(sv, mnew)), c.f32(0.0f));
                            sum = c.b.build_add(sum, pv);
                            ds = c.b.build_fma_f32(pv, kb.shared_load_f32_indexed(SF, c.b.build_add(c.b.build_add(fdb, row_off), col)), ds);
                        }
                        sum = c.b.build_add(sum, kb.shfl_bfly_f32(sum, 1u));
                        ds = c.b.build_add(ds, kb.shfl_bfly_f32(ds, 1u));
                        Value* alpha = kb.ex2_approx(c.b.build_sub(st[0], mnew));
                        return {mnew, c.b.build_fma_f32(st[1], alpha, sum), c.b.build_fma_f32(st[2], alpha, ds)};
                    }

                    // Uniform barrier outside fast/slow split so BVSolver/barrier prover always verifies.
                    kb.if_then(c.b.build_eq(fast, c.i32(0)), [&] {
                        for (std::uint32_t j = 0; j < NJ; ++j) {
                            kb.coop_store_acc(s_fr[j], c.b.build_add(SF, c.b.build_shl(c.b.build_add(fsb, c.i32(static_cast<int>(j * 16u))), c.i32(2))),
                                              c.i32(static_cast<int>(FS)), false, false);
                            kb.coop_store_acc(dp_fr[j], c.b.build_add(SF, c.b.build_shl(c.b.build_add(fdb, c.i32(static_cast<int>(j * 16u))), c.i32(2))),
                                              c.i32(static_cast<int>(FS)), false, false);
                        }
                    });
                    kb.subgroup_barrier();

                    return emit_if_else_values(
                        c, fast, std::vector<Type>(27, Type::f32()),
                        [&]() -> std::vector<Value*> {
                            std::vector<Value*> out = st;
                            for (int i = 0; i < 8; ++i) {
                                Value* mx = c.f32(-1e30f);
                                std::vector<Value*> sv(NJ), okv(NJ);
                                for (std::uint32_t j = 0; j < NJ; ++j) {
                                    Value* col = c.b.build_add(c.i32(static_cast<int>(j * 16u)), cl);
                                    Value* key = c.b.build_add(kb_v, col);
                                    Value* ok = c.b.build_and(
                                        c.b.build_and(c.b.build_sge(key, flo[static_cast<std::size_t>(i)]),
                                                      c.b.build_slt(key, fhi[static_cast<std::size_t>(i)])),
                                        c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f)));
                                    Value* s_sc = c.b.build_mul(kb.coop_get_acc(s_fr[j], c.i32(i)), p[19]);
                                    sv[j] = s_sc;
                                    okv[j] = ok;
                                    mx = c.b.build_select(ok, kb.fmax(mx, s_sc), mx);
                                }
                                Value* mnew = kb.fmax(st[3u + static_cast<std::size_t>(i)], emit_cluster16_max_f32(c, mx));
                                Value* sum = c.f32(0.0f);
                                Value* ds = c.f32(0.0f);
                                for (std::uint32_t j = 0; j < NJ; ++j) {
                                    Value* pv = c.b.build_select(okv[j], kb.ex2_approx(c.b.build_sub(sv[j], mnew)), c.f32(0.0f));
                                    sum = c.b.build_add(sum, pv);
                                    ds = c.b.build_fma_f32(pv, kb.coop_get_acc(dp_fr[j], c.i32(i)), ds);
                                }
                                Value* alpha = kb.ex2_approx(c.b.build_sub(st[3u + static_cast<std::size_t>(i)], mnew));
                                out[3u + static_cast<std::size_t>(i)] = mnew;
                                out[11u + static_cast<std::size_t>(i)] = c.b.build_fma_f32(st[11u + static_cast<std::size_t>(i)], alpha, emit_cluster16_add_f32(c, sum));
                                out[19u + static_cast<std::size_t>(i)] = c.b.build_fma_f32(st[19u + static_cast<std::size_t>(i)], alpha, emit_cluster16_add_f32(c, ds));
                            }
                            return out;
                        },
                        [&]() -> std::vector<Value*> {
                            std::vector<Value*> out = st;
                            Value* row_off = c.b.build_mul(r, c.i32(static_cast<int>(FS)));
                            Value* mx = c.f32(-1e30f);
                            for (std::uint32_t ci = 0; ci < s.bn / 2u; ++ci) {
                                Value* col = c.b.build_add(c0, c.i32(static_cast<int>(ci)));
                                Value* key = c.b.build_add(kb_v, col);
                                Value* ok = c.b.build_and(c.b.build_and(c.b.build_sge(key, alo), c.b.build_slt(key, ahi)),
                                                          c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f)));
                                Value* sv = c.b.build_mul(kb.shared_load_f32_indexed(SF, c.b.build_add(c.b.build_add(fsb, row_off), col)), p[19]);
                                mx = c.b.build_select(ok, kb.fmax(mx, sv), mx);
                            }
                            mx = kb.fmax(mx, kb.shfl_bfly_f32(mx, 1u));
                            Value* mnew = kb.fmax(st[0], mx);
                            Value* sum = c.f32(0.0f);
                            Value* ds = c.f32(0.0f);
                            for (std::uint32_t ci = 0; ci < s.bn / 2u; ++ci) {
                                Value* col = c.b.build_add(c0, c.i32(static_cast<int>(ci)));
                                Value* key = c.b.build_add(kb_v, col);
                                Value* ok = c.b.build_and(c.b.build_and(c.b.build_sge(key, alo), c.b.build_slt(key, ahi)),
                                                          c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f)));
                                Value* sv = c.b.build_mul(kb.shared_load_f32_indexed(SF, c.b.build_add(c.b.build_add(fsb, row_off), col)), p[19]);
                                Value* pv = c.b.build_select(ok, kb.ex2_approx(c.b.build_sub(sv, mnew)), c.f32(0.0f));
                                sum = c.b.build_add(sum, pv);
                                ds = c.b.build_fma_f32(pv, kb.shared_load_f32_indexed(SF, c.b.build_add(c.b.build_add(fdb, row_off), col)), ds);
                            }
                            sum = c.b.build_add(sum, kb.shfl_bfly_f32(sum, 1u));
                            ds = c.b.build_add(ds, kb.shfl_bfly_f32(ds, 1u));
                            Value* alpha = kb.ex2_approx(c.b.build_sub(st[0], mnew));
                            out[0] = mnew;
                            out[1] = c.b.build_fma_f32(st[1], alpha, sum);
                            out[2] = c.b.build_fma_f32(st[2], alpha, ds);
                            return out;
                        });
                });

            if (s.agnostic) {
                kb.if_then(c.b.build_eq(half_, c.i32(0)), [&] {
                    Value* any = c.b.build_sgt(fin_st[1], c.f32(0.0f));
                    Value* safe_l = c.b.build_select(any, fin_st[1], c.f32(1.0f));
                    kb.shared_store_f32_indexed(ist_x, rl, c.b.build_select(any, c.b.build_add(fin_st[0], kb.lg2_approx(safe_l)), c.f32(0.0f)));
                    kb.shared_store_f32_indexed(ist_y, rl, c.b.build_select(any, c.b.build_sdiv(fin_st[2], safe_l), c.f32(0.0f)));
                });
            } else {
                kb.if_then(fast, [&] {
                    kb.if_then(c.b.build_eq(cl, c.i32(0)), [&] {
                        for (int i = 0; i < 8; ++i) {
                            Value* fl_i = fin_st[11u + static_cast<std::size_t>(i)];
                            Value* any = c.b.build_sgt(fl_i, c.f32(0.0f));
                            Value* safe_l = c.b.build_select(any, fl_i, c.f32(1.0f));
                            Value* rr = c.b.build_add(c.b.build_shl(sg, c.i32(4)),
                                                      c.b.build_and(c.b.build_lshr(kb.f32_to_u32(kb.coop_get_acc(IX, c.i32(i))), c.i32(4)), c.i32(15)));
                            kb.shared_store_f32_indexed(ist_x, rr, c.b.build_select(any, c.b.build_add(fin_st[3u + static_cast<std::size_t>(i)], kb.lg2_approx(safe_l)), c.f32(0.0f)));
                            kb.shared_store_f32_indexed(ist_y, rr, c.b.build_select(any, c.b.build_sdiv(fin_st[19u + static_cast<std::size_t>(i)], safe_l), c.f32(0.0f)));
                        }
                    });
                });
                kb.if_then(c.b.build_and(c.b.build_eq(fast, c.i32(0)), c.b.build_eq(half_, c.i32(0))), [&] {
                    Value* any = c.b.build_sgt(fin_st[1], c.f32(0.0f));
                    Value* safe_l = c.b.build_select(any, fin_st[1], c.f32(1.0f));
                    kb.shared_store_f32_indexed(ist_x, rl, c.b.build_select(any, c.b.build_add(fin_st[0], kb.lg2_approx(safe_l)), c.f32(0.0f)));
                    kb.shared_store_f32_indexed(ist_y, rl, c.b.build_select(any, c.b.build_sdiv(fin_st[2], safe_l), c.f32(0.0f)));
                });
            }
            kb.subgroup_barrier();
            kb.if_then(c.b.build_and(c.b.build_eq(half_, c.i32(0)), c.b.build_ult(c.b.build_add(r0, rl), p[10])), [&] {
                Value* se = stat_at(c.b.build_add(r0, rl));
                kb.store_f32(ptr_at(c, p[7], se, 4), kb.shared_load_f32_indexed(ist_x, rl));
                kb.store_f32(ptr_at(c, p[7], c.b.build_add(se, c.i32(1)), 4), kb.shared_load_f32_indexed(ist_y, rl));
            });
            c.b.build_ret_void();
            return finalize_fn(fn);
        }

        // ── PASS 2: dQ += dS K ──
        Value* hbase_v4 = c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(HS_V4)));
        std::vector<Value*> acc(ND);
        for (std::uint32_t dc = 0; dc < ND; ++dc) {
            acc[dc] = kb.coop_alloc_acc();
            kb.coop_fill_acc(acc[dc], c.f32(0.0f));
        }
        Value* fs_frag = kb.coop_alloc_a();
        kb.for_range(kbeg, kend, c.i32(static_cast<int>(s.bn)), [&](Value* kb_v) {
            kb.sync();
            stage(KVO, s.bn, p[1], p[13], kb_v, p[11]);
            stage(KVO + s.bn * SD, s.bn, p[2], p[13], kb_v, p[11]);
            kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(s.bn))), [&] {
                kb.shared_store_f32_indexed(kok, tid, key_ok_f32(c.b.build_add(kb_v, tid)));
            });
            kb.sync();
            tile_sdp();

            for (std::uint32_t j = 0; j < NJ; ++j) {
                auto do_agnostic = [&] {
                    emit_coop_acc_map(c, s_fr[j], [&](Value* idx, Value* sv) {
                        Value* ev = kb.f32_to_u32(kb.coop_get_acc(IX, idx));
                        Value* rr = c.b.build_add(c.b.build_shl(sg, c.i32(4)), c.b.build_and(c.b.build_lshr(ev, c.i32(4)), c.i32(15)));
                        Value* col = c.b.build_add(c.i32(static_cast<int>(j * 16u)), c.b.build_and(ev, c.i32(15)));
                        Value* key = c.b.build_add(kb_v, col);
                        Value* ok = c.b.build_and(
                            c.b.build_and(c.b.build_sge(key, kb.shared_load_i32_indexed(ilo, rr)),
                                          c.b.build_slt(key, kb.shared_load_i32_indexed(ihi, rr))),
                            c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f)));
                        Value* pv = c.b.build_select(
                            ok, kb.ex2_approx(c.b.build_sub(c.b.build_mul(sv, p[19]), kb.shared_load_f32_indexed(ist_x, rr))),
                            c.f32(0.0f));
                        return c.b.build_mul(pv, c.b.build_sub(kb.coop_get_acc(dp_fr[j], idx), kb.shared_load_f32_indexed(ist_y, rr)));
                    });
                };
                if (s.agnostic) {
                    do_agnostic();
                } else {
                    kb.if_then(fast, [&] {
                        Value* col = c.b.build_add(c.i32(static_cast<int>(j * 16u)), cl);
                        Value* key = c.b.build_add(kb_v, col);
                        Value* kv = c.b.build_sgt(kb.shared_load_f32_indexed(kok, col), c.f32(0.5f));
                        for (int i = 0; i < 8; ++i) {
                            Value* ok = c.b.build_and(
                                kv, c.b.build_and(c.b.build_sge(key, flo[static_cast<std::size_t>(i)]),
                                                  c.b.build_slt(key, fhi[static_cast<std::size_t>(i)])));
                            Value* sv = kb.coop_get_acc(s_fr[j], c.i32(i));
                            Value* pv = c.b.build_select(
                                ok, kb.ex2_approx(c.b.build_sub(c.b.build_mul(sv, p[19]), fst_x[static_cast<std::size_t>(i)])),
                                c.f32(0.0f));
                            Value* ds = c.b.build_mul(pv, c.b.build_sub(kb.coop_get_acc(dp_fr[j], c.i32(i)), fst_y[static_cast<std::size_t>(i)]));
                            kb.coop_set_acc(s_fr[j], c.i32(i), ds);
                        }
                    });
                    kb.if_then(c.b.build_eq(fast, c.i32(0)), do_agnostic);
                }
                Value* dst_v4 = c.b.build_add(hbase_v4, c.i32(static_cast<int>(j * 2u)));
                kb.coop_store_acc(s_fr[j], c.b.build_add(SH, c.b.build_shl(dst_v4, c.i32(4))), stride_hs, false, true);
            }
            kb.subgroup_barrier();
            for (std::uint32_t j = 0; j < NJ; ++j) {
                Value* src_v4 = c.b.build_add(hbase_v4, c.i32(static_cast<int>(j * 2u)));
                kb.coop_load_a(fs_frag, c.b.build_add(SH, c.b.build_shl(src_v4, c.i32(4))), stride_hs, false);
                for (std::uint32_t dc = 0; dc < ND; ++dc) {
                    Value* k_off = c.i32(static_cast<int>(KVO + j * 16u * SD + dc * 2u));
                    kb.coop_load_b(fk_frag, c.b.build_add(SM, c.b.build_shl(k_off, c.i32(4))), stride_sd, false);
                    kb.coop_mma(acc[dc], fs_frag, fk_frag);
                }
            }
        });
        kb.sync();
        Value* row0 = c.b.build_add(r0, c.b.build_shl(sg, c.i32(4)));
        for (std::uint32_t dc = 0; dc < ND; ++dc) {
            store_tile(acc[dc], sbase, p[8], row0, p[10], dc * 16u, p[20]);
        }
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    // ── PASS 1, key-major: dK, dV ──
    Value* k0 = c.b.build_mul(kb.ctaid_x(), c.i32(static_cast<int>(BO)));
    Value* kr = c.b.build_shl(sg, c.i32(4));
    const std::uint32_t QB = KVO, GB = QB + s.bn * SD;
    stage(0u, BO, p[1], p[13], k0, p[11]);
    stage(BO * SD, BO, p[2], p[13], k0, p[11]);
    kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(BO))), [&] {
        kb.shared_store_f32_indexed(kok, c.b.build_add(c.i32(static_cast<int>(s.bn)), tid), key_ok_f32(c.b.build_add(k0, tid)));
    });
    Value* k1 = c.b.build_sub(smin_i32(c, c.b.build_add(k0, c.i32(static_cast<int>(BO))), p[11]), c.i32(1));
    auto [qlo, qhi] = emit_bw_cm_qrange(c, s.mode, k0, k1, p[10], p[16], p[17], p[18], p[5], p[6]);
    kb.sync();

    kb.coop_load_acc(IX, SF, c.i32(16), false, false);
    Value* fast = check_cluster_layout(IX);
    std::array<Value*, 8> k_ok_i{}, k_row_i{};
    if (!s.agnostic) {
        for (int i = 0; i < 8; ++i) {
            Value* r_local = emit_if_else_values(
                c, fast, {Type::i32()},
                [&]() -> std::vector<Value*> {
                    return {c.b.build_and(c.b.build_lshr(kb.f32_to_u32(kb.coop_get_acc(IX, c.i32(i))), c.i32(4)), c.i32(15))};
                },
                [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
            k_row_i[static_cast<std::size_t>(i)] = c.b.build_add(c.b.build_add(k0, kr), r_local);
            Value* kok_idx = c.b.build_add(c.i32(static_cast<int>(s.bn)), c.b.build_add(kr, r_local));
            k_ok_i[static_cast<std::size_t>(i)] = c.b.build_sgt(kb.shared_load_f32_indexed(kok, kok_idx), c.f32(0.5f));
        }
    }
    kb.sync();

    std::vector<Value*> ak(ND), av(ND), dk(ND), dv(ND);
    for (std::uint32_t dc = 0; dc < ND; ++dc) {
        ak[dc] = kb.coop_alloc_a();
        av[dc] = kb.coop_alloc_a();
        if (s.areg) {
            Value* off = c.b.build_add(c.b.build_mul(kr, c.i32(static_cast<int>(SD))), c.i32(static_cast<int>(dc * 2u)));
            kb.coop_load_a(ak[dc], c.b.build_add(SM, c.b.build_shl(off, c.i32(4))), stride_sd, false);
            kb.coop_load_a(av[dc], c.b.build_add(SM, c.b.build_shl(c.b.build_add(off, c.i32(static_cast<int>(BO * SD))), c.i32(4))),
                           stride_sd, false);
        }
        dk[dc] = kb.coop_alloc_acc();
        dv[dc] = kb.coop_alloc_acc();
        kb.coop_fill_acc(dk[dc], c.f32(0.0f));
        kb.coop_fill_acc(dv[dc], c.f32(0.0f));
    }
    Value* fq_b = kb.coop_alloc_b();
    Value* fg_b = kb.coop_alloc_b();
    Value* fp_a = kb.coop_alloc_a();
    Value* fd_a = kb.coop_alloc_a();
    std::vector<Value*> s_fr(NJ), dp_fr(NJ);
    for (std::uint32_t j = 0; j < NJ; ++j) {
        s_fr[j] = kb.coop_alloc_acc();
        dp_fr[j] = kb.coop_alloc_acc();
    }
    Value* pbase_v4 = c.b.build_mul(kr, c.i32(static_cast<int>(HS_V4)));
    Value* dbase_v4 = c.b.build_add(c.i32(static_cast<int>(BO * HS_V4)), pbase_v4);
    Value* qb_start = c.b.build_mul(c.b.build_sdiv(qlo, c.i32(static_cast<int>(s.bn))), c.i32(static_cast<int>(s.bn)));

    kb.for_range(qb_start, qhi, c.i32(static_cast<int>(s.bn)), [&](Value* qb) {
        kb.sync();
        stage(QB, s.bn, p[0], p[12], qb, p[10]);
        stage(GB, s.bn, p[3], p[14], qb, p[10]);
        Value* mine = emit_if_else_values(
            c, c.b.build_ult(tid, c.i32(static_cast<int>(s.bn))), {Type::i32()},
            [&]() -> std::vector<Value*> {
                Value* q = c.b.build_add(qb, tid);
                auto [rlo, rhi] = emit_bw_cm_interval(c, s.mode, q, p[10], p[11], p[16], p[17], p[18], p[5], p[6]);
                kb.shared_store_i32_indexed(ilo, tid, rlo);
                kb.shared_store_i32_indexed(ihi, tid, rhi);
                Value* sq = smin_i32(c, q, c.b.build_sub(p[10], c.i32(1)));
                Value* se = stat_at(sq);
                kb.shared_store_f32_indexed(ist_x, tid, kb.load_f32(ptr_at(c, p[7], se, 4)));
                kb.shared_store_f32_indexed(ist_y, tid, kb.load_f32(ptr_at(c, p[7], c.b.build_add(se, c.i32(1)), 4)));
                Value* hit = c.b.build_and(c.b.build_slt(rlo, rhi),
                                           c.b.build_and(c.b.build_sle(rlo, k1), c.b.build_sgt(rhi, k0)));
                return {hit};
            },
            [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
        Value* sg_any = emit_subgroup32_any_i32(c, mine);
        kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] {
            kb.shared_store_i32_indexed(sflag, sg, c.b.build_select(sg_any, c.i32(1), c.i32(0)));
        });
        kb.sync();
        Value* any = c.i32(0);
        for (std::uint32_t ss = 0; ss < s.nsg; ++ss) {
            any = c.b.build_or(any, kb.shared_load_i32_indexed(sflag, c.i32(static_cast<int>(ss))));
        }

        // Compute S^T, dP^T, P^T, dS^T only when any query in this block attends [k0, k1].
        // Keep subgroup_barrier outside the non-trivial branch or note `any` is workgroup-uniform
        // (every thread reads the same sflag[0..nsg-1] after kb.sync()).
        kb.if_then(c.b.build_ne(any, c.i32(0)), [&] {
            for (std::uint32_t j = 0; j < NJ; ++j) {
                kb.coop_fill_acc(s_fr[j], c.f32(0.0f));
                kb.coop_fill_acc(dp_fr[j], c.f32(0.0f));
            }
            for (std::uint32_t dc = 0; dc < ND; ++dc) {
                if (!s.areg) {
                    Value* off = c.b.build_add(c.b.build_mul(kr, c.i32(static_cast<int>(SD))), c.i32(static_cast<int>(dc * 2u)));
                    kb.coop_load_a(ak[dc], c.b.build_add(SM, c.b.build_shl(off, c.i32(4))), stride_sd, false);
                    kb.coop_load_a(av[dc], c.b.build_add(SM, c.b.build_shl(c.b.build_add(off, c.i32(static_cast<int>(BO * SD))), c.i32(4))),
                                   stride_sd, false);
                }
                for (std::uint32_t j = 0; j < NJ; ++j) {
                    Value* q_off = c.i32(static_cast<int>(QB + j * 16u * SD + dc * 2u));
                    Value* g_off = c.i32(static_cast<int>(GB + j * 16u * SD + dc * 2u));
                    kb.coop_load_b(fq_b, c.b.build_add(SM, c.b.build_shl(q_off, c.i32(4))), stride_sd, true);
                    kb.coop_load_b(fg_b, c.b.build_add(SM, c.b.build_shl(g_off, c.i32(4))), stride_sd, true);
                    kb.coop_mma(s_fr[j], ak[dc], fq_b);
                    kb.coop_mma(dp_fr[j], av[dc], fg_b);
                }
            }
            for (std::uint32_t j = 0; j < NJ; ++j) {
                auto do_agnostic = [&] {
                    Value* len = kb.coop_acc_len(s_fr[j]);
                    kb.for_range(c.i32(0), len, c.i32(1), [&](Value* idx) {
                        Value* ev = kb.f32_to_u32(kb.coop_get_acc(IX, idx));
                        Value* rr = c.b.build_and(c.b.build_lshr(ev, c.i32(4)), c.i32(15));
                        Value* col = c.b.build_add(c.i32(static_cast<int>(j * 16u)), c.b.build_and(ev, c.i32(15)));
                        Value* key = c.b.build_add(c.b.build_add(k0, kr), rr);
                        Value* kok_v = kb.shared_load_f32_indexed(kok, c.b.build_add(c.i32(static_cast<int>(s.bn)), c.b.build_add(kr, rr)));
                        Value* ok = c.b.build_and(
                            c.b.build_sgt(kok_v, c.f32(0.5f)),
                            c.b.build_and(c.b.build_sge(key, kb.shared_load_i32_indexed(ilo, col)),
                                          c.b.build_slt(key, kb.shared_load_i32_indexed(ihi, col))));
                        Value* sv = kb.coop_get_acc(s_fr[j], idx);
                        Value* pv = c.b.build_select(
                            ok, kb.ex2_approx(c.b.build_sub(c.b.build_mul(sv, p[19]), kb.shared_load_f32_indexed(ist_x, col))),
                            c.f32(0.0f));
                        Value* dsv = c.b.build_mul(pv, c.b.build_sub(kb.coop_get_acc(dp_fr[j], idx), kb.shared_load_f32_indexed(ist_y, col)));
                        kb.coop_set_acc(s_fr[j], idx, pv);
                        kb.coop_set_acc(dp_fr[j], idx, dsv);
                    });
                };
                if (s.agnostic) {
                    do_agnostic();
                } else {
                    kb.if_then(fast, [&] {
                        Value* col = c.b.build_add(c.i32(static_cast<int>(j * 16u)), cl);
                        Value* qx = kb.shared_load_f32_indexed(ist_x, col);
                        Value* qy = kb.shared_load_f32_indexed(ist_y, col);
                        Value* lo_c = kb.shared_load_i32_indexed(ilo, col);
                        Value* hi_c = kb.shared_load_i32_indexed(ihi, col);
                        for (int i = 0; i < 8; ++i) {
                            Value* key = k_row_i[static_cast<std::size_t>(i)];
                            Value* ok = c.b.build_and(k_ok_i[static_cast<std::size_t>(i)],
                                                      c.b.build_and(c.b.build_sge(key, lo_c), c.b.build_slt(key, hi_c)));
                            Value* sv = kb.coop_get_acc(s_fr[j], c.i32(i));
                            Value* pv = c.b.build_select(ok, kb.ex2_approx(c.b.build_sub(c.b.build_mul(sv, p[19]), qx)), c.f32(0.0f));
                            Value* dsv = c.b.build_mul(pv, c.b.build_sub(kb.coop_get_acc(dp_fr[j], c.i32(i)), qy));
                            kb.coop_set_acc(s_fr[j], c.i32(i), pv);
                            kb.coop_set_acc(dp_fr[j], c.i32(i), dsv);
                        }
                    });
                    kb.if_then(c.b.build_eq(fast, c.i32(0)), do_agnostic);
                }
                kb.coop_store_acc(s_fr[j], c.b.build_add(SH, c.b.build_shl(c.b.build_add(pbase_v4, c.i32(static_cast<int>(j * 2u))), c.i32(4))),
                                  stride_hs, false, true);
                kb.coop_store_acc(dp_fr[j], c.b.build_add(SH, c.b.build_shl(c.b.build_add(dbase_v4, c.i32(static_cast<int>(j * 2u))), c.i32(4))),
                                  stride_hs, false, true);
            }
        });
        kb.subgroup_barrier();
        kb.if_then(c.b.build_ne(any, c.i32(0)), [&] {
            for (std::uint32_t j = 0; j < NJ; ++j) {
                kb.coop_load_a(fp_a, c.b.build_add(SH, c.b.build_shl(c.b.build_add(pbase_v4, c.i32(static_cast<int>(j * 2u))), c.i32(4))),
                               stride_hs, false);
                kb.coop_load_a(fd_a, c.b.build_add(SH, c.b.build_shl(c.b.build_add(dbase_v4, c.i32(static_cast<int>(j * 2u))), c.i32(4))),
                               stride_hs, false);
                for (std::uint32_t dc = 0; dc < ND; ++dc) {
                    Value* g_off = c.i32(static_cast<int>(GB + j * 16u * SD + dc * 2u));
                    Value* q_off = c.i32(static_cast<int>(QB + j * 16u * SD + dc * 2u));
                    kb.coop_load_b(fg_b, c.b.build_add(SM, c.b.build_shl(g_off, c.i32(4))), stride_sd, false);
                    kb.coop_load_b(fq_b, c.b.build_add(SM, c.b.build_shl(q_off, c.i32(4))), stride_sd, false);
                    kb.coop_mma(dv[dc], fp_a, fg_b);
                    kb.coop_mma(dk[dc], fd_a, fq_b);
                }
            }
        });
    });

    kb.sync();
    Value* krow0 = c.b.build_add(k0, kr);
    for (std::uint32_t dc = 0; dc < ND; ++dc) {
        store_tile(dk[dc], sbase, p[8], krow0, p[11], dc * 16u, p[20]);
        store_tile(dv[dc], sbase, p[9], krow0, p[11], dc * 16u, c.f32(1.0f));
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
