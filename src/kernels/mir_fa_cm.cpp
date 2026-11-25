#include "mir_fa_cm.h"

#include "mir_common.h"
#include "mir_coop_common.h"
#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

std::pair<Value*, Value*> emit_fa_cm_interval(Ctx& c, std::uint32_t mode, Value* qi,
                                              Value* lq, Value* lk, Value* window, Value* causal,
                                              Value* nseq, Value* q_offset, Value* aux, Value* aux2) {
    auto res = emit_if_else_values(
        c, c.b.build_ult(qi, lq), {Type::i32(), Type::i32()},
        [&]() -> std::vector<Value*> {
            Value* a = c.i32(0);
            Value* b = lk;
            if (mode == FA_MODE_ROWS) {
                Value* aq = c.b.build_add(qi, q_offset);
                Value* is_c = c.b.build_ne(causal, c.i32(0));
                Value* has_w = c.b.build_ugt(window, c.i32(0));
                Value* b_c = smin_i32(c, lk, c.b.build_add(aq, c.i32(1)));
                Value* a_cw = smax_i32(c, c.i32(0), c.b.build_add(c.b.build_sub(aq, window), c.i32(1)));
                Value* a_c = c.b.build_select(has_w, a_cw, c.i32(0));
                Value* w2 = c.b.build_lshr(window, c.i32(1));
                Value* a_nc = c.b.build_select(has_w, smax_i32(c, c.i32(0), c.b.build_sub(aq, w2)), c.i32(0));
                Value* b_nc = c.b.build_select(has_w, smin_i32(c, lk, c.b.build_add(c.b.build_add(aq, w2), c.i32(1))), lk);
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
            } else if (mode == FA_MODE_LEN) {
                Value* flen = clamp_i32(c, c.kb.load_i32(aux), c.i32(0), lk);
                b = c.b.build_add(c.b.build_add(c.b.build_sub(flen, lq), c.i32(1)), qi);
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

Value* load8_fa_scalar(Ctx& c, Dtype dt, std::uint32_t hd, Value* base, Value* off, Value* ok, Value* c0) {
    return emit_if_else_values(
        c, ok, {Type::i32x4()},
        [&]() -> std::vector<Value*> {
            std::array<Value*, 8> h{};
            for (int e = 0; e < 8; ++e) {
                Value* col = c.b.build_add(c0, c.i32(e));
                h[static_cast<std::size_t>(e)] = emit_if_else_values(
                    c, c.b.build_ult(col, c.i32(static_cast<int>(hd))), {Type::f32()},
                    [&]() -> std::vector<Value*> { return {load_elem_dt(c, dt, base, c.b.build_add(off, col))}; },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            }
            return {emit_pack_f32x8_to_f16_v4(c, h)};
        },
        [&]() -> std::vector<Value*> { return {emit_v4_zero(c)}; })[0];
}

}  // namespace

brass::Function* build_fa_cm_mir(brass::Module& mod, const FaCmSpec& s) {
    if ((s.bc != 16u && s.bc != 32u && s.bc != 64u) || (s.nsg != 1u && s.nsg != 2u && s.nsg != 4u)) {
        throw std::runtime_error("brotensor: mir_fa_cm: invalid bc/nsg");
    }
    const std::uint32_t HD = s.hd;
    const std::uint32_t DP = ((HD + 15u) / 16u) * 16u;
    const std::uint32_t ND = DP / 16u;
    const std::uint32_t DV = DP / 8u;
    const std::uint32_t DVR = (HD + 7u) / 8u;
    const std::uint32_t SD = DV + 1u;
    const std::uint32_t BR = s.nsg * 16u;
    const std::uint32_t WG = s.nsg * 32u;
    const std::uint32_t NC = s.bc / 16u;
    const std::uint32_t SPH = s.bc + 8u;
    const std::uint32_t SPH_V4 = SPH / 8u;
    const std::uint32_t KVSZ = s.bc * SD;
    const std::uint32_t NKV = s.bc * DV;
    const std::uint32_t LKV = (NKV + WG - 1u) / WG;
    const std::uint32_t Q_KV_SLOTS = std::max(2u * KVSZ, BR * SD);

    const std::string name = std::string("mir_fa_cm_") + (s.dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_hd" + std::to_string(HD) + "_bc" + std::to_string(s.bc) +
                             "_g" + std::to_string(s.nsg) + "_m" + std::to_string(s.mode) +
                             (s.has_mask ? "_mk" : "") + (s.vec ? "_v" : "_s") +
                             (s.direct ? "_d" : "") + (s.split ? "_sp" : "");

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: 0:q 1:k 2:v 3:o 4:mask 5:aux 6:aux2 7:part
    // ints: 8:lq 9:lk 10:ldq 11:ldk 12:ldo 13:group 14:window 15:causal 16:nseq 17:chunk 18:hq 19:qb0 20:q_offset
    // flts: 21:scale

    Value* KV = kb.shared_alloc_i32x4(Q_KV_SLOTS);
    Value* Ss = kb.shared_alloc_f32(std::max(256u, BR * s.bc));
    Value* Ps = kb.shared_alloc_i32x4(BR * SPH_V4);
    Value* rowfac = kb.shared_alloc_f32(BR);
    Value* maskS = s.has_mask ? kb.shared_alloc_f32(s.bc) : nullptr;
    Value* s_lo = kb.shared_alloc_i32(WG);
    Value* s_hi = kb.shared_alloc_i32(WG);

    Value* tid = c.b.build_and(kb.tid_x(), c.i32(static_cast<int>(WG - 1u)));
    Value* sg = c.b.build_and(kb.subgroup_id(), c.i32(static_cast<int>(s.nsg - 1u)));
    Value* lane = c.b.build_and(kb.lane_id(), c.i32(31));

    Value* qb = c.b.build_add(p[19], c.b.build_sub(c.b.build_sub(kb.nctaid_x(), c.i32(1)), kb.ctaid_x()));
    Value* h = kb.ctaid_y();
    Value* hk = c.b.build_udiv(h, p[13]);
    Value* qoff = c.b.build_mul(h, c.i32(static_cast<int>(HD)));
    Value* koff = c.b.build_mul(hk, c.i32(static_cast<int>(HD)));

    Value* r = c.b.build_lshr(lane, c.i32(1));
    Value* half_ = c.b.build_and(lane, c.i32(1));
    Value* row_local = c.b.build_add(c.b.build_shl(sg, c.i32(4)), r);
    Value* qi = c.b.build_add(c.b.build_mul(qb, c.i32(static_cast<int>(BR))), row_local);
    auto [lo, hi] = emit_fa_cm_interval(c, s.mode, qi, p[8], p[9], p[14], p[15], p[16], p[20], p[5], p[6]);

    // Populate RI (row indices) in Ss[0..255] and stage Q into KV[0..BR*SD).
    for (std::uint32_t e0 = 0; e0 < 256u; e0 += WG) {
        Value* e = (e0 == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(e0)));
        kb.shared_store_f32_indexed(Ss, e, kb.u32_to_f32(c.b.build_lshr(e, c.i32(4))));
    }
    Value* z4 = emit_v4_zero(c);
    for (std::uint32_t i0 = 0; i0 < BR * DV; i0 += WG) {
        Value* i = (i0 == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(i0)));
        auto stage_q = [&] {
            Value* rr = c.b.build_udiv(i, c.i32(static_cast<int>(DV)));
            Value* col = c.b.build_umod(i, c.i32(static_cast<int>(DV)));
            Value* row = c.b.build_add(c.b.build_mul(qb, c.i32(static_cast<int>(BR))), rr);
            Value* ok = c.b.build_and(c.b.build_ult(row, p[8]), c.b.build_ult(col, c.i32(static_cast<int>(DVR))));
            Value* w = nullptr;
            if (s.vec) {
                Value* off = c.b.build_add(c.b.build_add(c.b.build_mul(row, p[10]), qoff), c.b.build_shl(col, c.i32(3)));
                Value* raw = kb.vload(Type::i32x4(), ptr_at(c, p[0], c.b.build_select(ok, off, c.i32(0)), 2));
                if (s.dt == Dtype::BF16) raw = emit_bf16x8_to_f16x8(c, raw);
                w = emit_v4_select(c, ok, raw, z4);
            } else {
                w = load8_fa_scalar(c, s.dt, HD, p[0], c.b.build_add(c.b.build_mul(row, p[10]), qoff), ok,
                                    c.b.build_shl(col, c.i32(3)));
            }
            kb.shared_store_i32x4_indexed(KV, c.b.build_add(c.b.build_mul(rr, c.i32(static_cast<int>(SD))), col), w);
        };
        if (i0 + WG <= BR * DV) stage_q();
        else kb.if_then(c.b.build_ult(i, c.i32(static_cast<int>(BR * DV))), stage_q);
    }

    // Uniform tree reduction for workgroup [kbeg, kend) across all rows.
    Value* has_keys = c.b.build_ult(lo, hi);
    kb.shared_store_i32_indexed(s_lo, tid, c.b.build_select(has_keys, lo, c.i32(static_cast<int>(0x7fffffff))));
    kb.shared_store_i32_indexed(s_hi, tid, c.b.build_select(has_keys, hi, c.i32(0)));
    kb.sync();
    for (std::uint32_t st = WG / 2u; st >= 1u; st >>= 1) {
        kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(st))), [&] {
            Value* t2 = c.b.build_add(tid, c.i32(static_cast<int>(st)));
            Value* a0 = kb.shared_load_i32_indexed(s_lo, tid);
            Value* a1 = kb.shared_load_i32_indexed(s_lo, t2);
            kb.shared_store_i32_indexed(s_lo, tid, smin_i32(c, a0, a1));
            Value* b0 = kb.shared_load_i32_indexed(s_hi, tid);
            Value* b1 = kb.shared_load_i32_indexed(s_hi, t2);
            kb.shared_store_i32_indexed(s_hi, tid, smax_i32(c, b0, b1));
        });
        kb.sync();
    }

    std::vector<Value*> fq(ND), acc(ND);
    Value* fk = kb.coop_alloc_b();
    Value* fv = kb.coop_alloc_b();
    Value* fp = kb.coop_alloc_a();
    Value* s_frag = kb.coop_alloc_acc();
    Value* RI = kb.coop_alloc_acc();
    Value* stride_sd = c.i32(static_cast<int>(SD * 8u));
    Value* stride_bc = c.i32(static_cast<int>(s.bc));
    Value* stride_sph = c.i32(static_cast<int>(SPH));

    for (std::uint32_t dc = 0; dc < ND; ++dc) {
        fq[dc] = kb.coop_alloc_a();
        Value* q_off = c.b.build_add(c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(SD))),
                                     c.i32(static_cast<int>(dc * 2u)));
        kb.coop_load_a(fq[dc], c.b.build_add(KV, c.b.build_shl(q_off, c.i32(4))), stride_sd, false);
        acc[dc] = kb.coop_alloc_acc();
        kb.coop_fill_acc(acc[dc], c.f32(0.0f));
    }
    kb.coop_load_acc(RI, Ss, c.i32(16), false, false);
    kb.sync();

    Value* kbeg = kb.shared_load_i32_indexed(s_lo, c.i32(0));
    Value* kend = kb.shared_load_i32_indexed(s_hi, c.i32(0));
    if (s.split) {
        kbeg = c.b.build_add(kbeg, c.b.build_mul(kb.ctaid_z(), p[17]));
        kend = smin_i32(c, kend, c.b.build_add(kbeg, p[17]));
    }

    struct KvLoaded {
        std::vector<Value*> rk, rv, vk;
        Value* mreg = nullptr;
    };
    auto load_kv = [&](Value* kb_v) -> KvLoaded {
        KvLoaded t;
        t.rk.resize(LKV);
        t.rv.resize(LKV);
        t.vk.resize(LKV);
        for (std::uint32_t j = 0; j < LKV; ++j) {
            Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
            Value* rr = c.b.build_udiv(i, c.i32(static_cast<int>(DV)));
            Value* col = c.b.build_umod(i, c.i32(static_cast<int>(DV)));
            Value* key = c.b.build_add(kb_v, rr);
            Value* ok = c.b.build_and(c.b.build_ult(i, c.i32(static_cast<int>(NKV))),
                                      c.b.build_and(c.b.build_ult(key, kend), c.b.build_ult(col, c.i32(static_cast<int>(DVR)))));
            if (s.vec) {
                Value* elem_off = c.b.build_add(c.b.build_add(c.b.build_mul(key, p[11]), koff), c.b.build_shl(col, c.i32(3)));
                Value* safe_off = c.b.build_select(ok, elem_off, c.i32(0));
                t.rk[j] = kb.vload(Type::i32x4(), ptr_at(c, p[1], safe_off, 2));
                t.rv[j] = kb.vload(Type::i32x4(), ptr_at(c, p[2], safe_off, 2));
                t.vk[j] = ok;
            } else {
                Value* row_off = c.b.build_add(c.b.build_mul(key, p[11]), koff);
                t.rk[j] = load8_fa_scalar(c, s.dt, HD, p[1], row_off, ok, c.b.build_shl(col, c.i32(3)));
                t.rv[j] = load8_fa_scalar(c, s.dt, HD, p[2], row_off, ok, c.b.build_shl(col, c.i32(3)));
                t.vk[j] = c.i32(1);
            }
        }
        if (s.has_mask) {
            Value* key = c.b.build_add(kb_v, tid);
            Value* mok = c.b.build_and(c.b.build_ult(tid, c.i32(static_cast<int>(s.bc))), c.b.build_ult(key, kend));
            t.mreg = emit_if_else_values(
                c, mok, {Type::f32()},
                [&]() -> std::vector<Value*> { return {kb.load_f32(ptr_at(c, p[4], key, 4))}; },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
        }
        return t;
    };

    auto store_kv = [&](const KvLoaded& t) {
        for (std::uint32_t j = 0; j < LKV; ++j) {
            Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
            auto put = [&] {
                Value* rr = c.b.build_udiv(i, c.i32(static_cast<int>(DV)));
                Value* col = c.b.build_umod(i, c.i32(static_cast<int>(DV)));
                Value* at = c.b.build_add(c.b.build_mul(rr, c.i32(static_cast<int>(SD))), col);
                Value* rk = t.rk[j];
                Value* rv = t.rv[j];
                if (s.vec) {
                    if (s.dt == Dtype::BF16) {
                        rk = emit_bf16x8_to_f16x8(c, rk);
                        rv = emit_bf16x8_to_f16x8(c, rv);
                    }
                    rk = emit_v4_select(c, t.vk[j], rk, z4);
                    rv = emit_v4_select(c, t.vk[j], rv, z4);
                }
                kb.shared_store_i32x4_indexed(KV, at, rk);
                kb.shared_store_i32x4_indexed(KV, c.b.build_add(at, c.i32(static_cast<int>(KVSZ))), rv);
            };
            if (j * WG + WG <= NKV) put();
            else kb.if_then(c.b.build_ult(i, c.i32(static_cast<int>(NKV))), put);
        }
        if (s.has_mask) {
            kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(s.bc))), [&] {
                kb.shared_store_f32_indexed(maskS, tid, t.mreg);
            });
        }
    };

    Value* srow = c.b.build_mul(row_local, c.i32(static_cast<int>(s.bc)));
    Value* prow_v4 = c.b.build_mul(row_local, c.i32(static_cast<int>(SPH_V4)));
    Value* c0 = c.b.build_mul(half_, c.i32(static_cast<int>(s.bc / 2u)));
    Value* c0_v4 = c.b.build_lshr(c0, c.i32(3));
    Value* sg_sbase = c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(s.bc)));
    Value* sg_pbase_v4 = c.b.build_mul(c.b.build_shl(sg, c.i32(4)), c.i32(static_cast<int>(SPH_V4)));

    auto ml = kb.for_range_reduce_n(
        kbeg, kend, c.i32(static_cast<int>(s.bc)), {c.f32(-1e30f), c.f32(0.0f)},
        [&](Value* kb_v, const std::vector<Value*>& st) -> std::vector<Value*> {
            Value* m = st[0];
            Value* l = st[1];
            store_kv(load_kv(kb_v));
            kb.sync();

            for (std::uint32_t j = 0; j < NC; ++j) {
                kb.coop_fill_acc(s_frag, c.f32(0.0f));
                for (std::uint32_t dc = 0; dc < ND; ++dc) {
                    Value* k_off = c.i32(static_cast<int>(j * 16u * SD + dc * 2u));
                    kb.coop_load_b(fk, c.b.build_add(KV, c.b.build_shl(k_off, c.i32(4))), stride_sd, true);
                    kb.coop_mma(s_frag, fq[dc], fk);
                }
                Value* s_dst = c.b.build_add(sg_sbase, c.i32(static_cast<int>(j * 16u)));
                kb.coop_store_acc(s_frag, c.b.build_add(Ss, c.b.build_shl(s_dst, c.i32(2))), stride_bc, false, false);
            }
            kb.subgroup_barrier();

            Value* mx = c.f32(-1e30f);
            for (std::uint32_t ci = 0; ci < s.bc / 2u; ++ci) {
                Value* col = c.b.build_add(c0, c.i32(static_cast<int>(ci)));
                Value* key = c.b.build_add(kb_v, col);
                Value* ok = c.b.build_and(c.b.build_uge(key, lo), c.b.build_ult(key, hi));
                if (s.has_mask) ok = c.b.build_and(ok, c.b.build_sgt(kb.shared_load_f32_indexed(maskS, col), c.f32(0.5f)));
                Value* sc = c.b.build_mul(kb.shared_load_f32_indexed(Ss, c.b.build_add(srow, col)), p[21]);
                mx = c.b.build_select(ok, kb.fmax(mx, sc), mx);
            }
            mx = kb.fmax(mx, kb.shfl_bfly_f32(mx, 1u));
            Value* mnew = kb.fmax(m, mx);
            Value* alpha = kb.ex2_approx(c.b.build_sub(m, mnew));

            Value* sum = c.f32(0.0f);
            for (std::uint32_t cg = 0; cg < (s.bc / 2u) / 8u; ++cg) {
                std::array<Value*, 8> pv{};
                for (int e = 0; e < 8; ++e) {
                    Value* col = c.b.build_add(c0, c.i32(static_cast<int>(cg * 8u + static_cast<std::uint32_t>(e))));
                    Value* key = c.b.build_add(kb_v, col);
                    Value* ok = c.b.build_and(c.b.build_uge(key, lo), c.b.build_ult(key, hi));
                    if (s.has_mask) ok = c.b.build_and(ok, c.b.build_sgt(kb.shared_load_f32_indexed(maskS, col), c.f32(0.5f)));
                    Value* sc = c.b.build_mul(kb.shared_load_f32_indexed(Ss, c.b.build_add(srow, col)), p[21]);
                    Value* pv_e = c.b.build_select(ok, kb.ex2_approx(c.b.build_sub(sc, mnew)), c.f32(0.0f));
                    pv[static_cast<std::size_t>(e)] = pv_e;
                    sum = c.b.build_add(sum, pv_e);
                }
                kb.shared_store_i32x4_indexed(
                    Ps, c.b.build_add(prow_v4, c.b.build_add(c0_v4, c.i32(static_cast<int>(cg)))),
                    emit_pack_f32x8_to_f16_v4(c, pv));
            }
            sum = c.b.build_add(sum, kb.shfl_bfly_f32(sum, 1u));
            Value* lnew = c.b.build_fma_f32(l, alpha, sum);
            kb.if_then(c.b.build_eq(half_, c.i32(0)), [&] {
                kb.shared_store_f32_indexed(rowfac, row_local, alpha);
            });
            kb.subgroup_barrier();

            Value* sg_rbase = c.b.build_shl(sg, c.i32(4));
            for (std::uint32_t dc = 0; dc < ND; ++dc) {
                emit_coop_acc_map(c, acc[dc], [&](Value* idx, Value* av) {
                    Value* ri = c.b.build_and(kb.f32_to_u32(kb.coop_get_acc(RI, idx)), c.i32(15));
                    return c.b.build_mul(av, kb.shared_load_f32_indexed(rowfac, c.b.build_add(sg_rbase, ri)));
                });
            }
            for (std::uint32_t j = 0; j < NC; ++j) {
                Value* p_off = c.b.build_add(sg_pbase_v4, c.i32(static_cast<int>(j * 2u)));
                kb.coop_load_a(fp, c.b.build_add(Ps, c.b.build_shl(p_off, c.i32(4))), stride_sph, false);
                for (std::uint32_t dc = 0; dc < ND; ++dc) {
                    Value* v_off = c.i32(static_cast<int>(KVSZ + j * 16u * SD + dc * 2u));
                    kb.coop_load_b(fv, c.b.build_add(KV, c.b.build_shl(v_off, c.i32(4))), stride_sd, false);
                    kb.coop_mma(acc[dc], fp, fv);
                }
            }
            kb.sync();
            return {mnew, lnew};
        });

    Value* m_fin = ml[0];
    Value* l_fin = ml[1];
    Value* row0 = c.b.build_add(c.b.build_mul(qb, c.i32(static_cast<int>(BR))), c.b.build_shl(sg, c.i32(4)));

    if (s.split) {
        Value* nsplit = kb.nctaid_z();
        Value* sp = kb.ctaid_z();
        Value* stride_p = c.i32(static_cast<int>(HD + 2u));
        kb.if_then(c.b.build_and(c.b.build_eq(half_, c.i32(0)), c.b.build_ult(qi, p[8])), [&] {
            Value* e = c.b.build_mul(
                c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(qi, p[18]), h), nsplit), sp), stride_p);
            kb.store_f32(ptr_at(c, p[7], c.b.build_add(e, c.i32(static_cast<int>(HD))), 4),
                         c.b.build_mul(m_fin, c.f32(0.69314718055994531f)));
            kb.store_f32(ptr_at(c, p[7], c.b.build_add(e, c.i32(static_cast<int>(HD + 1u))), 4), l_fin);
        });
        for (std::uint32_t dc = 0; dc < ND; ++dc) {
            kb.coop_store_acc(acc[dc], c.b.build_add(Ss, c.b.build_shl(sg_sbase, c.i32(2))), c.i32(16), false, false);
            kb.subgroup_barrier();
            for (int e0 = 0; e0 < 256; e0 += 32) {
                Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
                Value* rr = c.b.build_lshr(e, c.i32(4));
                Value* col = c.b.build_add(c.i32(static_cast<int>(dc * 16u)), c.b.build_and(e, c.i32(15)));
                Value* row = c.b.build_add(row0, rr);
                kb.if_then(c.b.build_and(c.b.build_ult(row, p[8]), c.b.build_ult(col, c.i32(static_cast<int>(HD)))), [&] {
                    Value* dst_e = c.b.build_add(
                        c.b.build_mul(c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(row, p[18]), h), nsplit), sp),
                                      stride_p),
                        col);
                    kb.store_f32(ptr_at(c, p[7], dst_e, 4), kb.shared_load_f32_indexed(Ss, c.b.build_add(sg_sbase, e)));
                });
            }
            kb.subgroup_barrier();
        }
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    kb.if_then(c.b.build_eq(half_, c.i32(0)), [&] {
        Value* inv_l = c.b.build_select(c.b.build_sgt(l_fin, c.f32(0.0f)),
                                        c.b.build_sdiv(c.f32(1.0f), c.b.build_select(c.b.build_sgt(l_fin, c.f32(0.0f)), l_fin, c.f32(1.0f))),
                                        c.f32(0.0f));
        kb.shared_store_f32_indexed(rowfac, row_local, inv_l);
    });
    kb.subgroup_barrier();

    if (s.direct) {
        Value* sg_rbase = c.b.build_shl(sg, c.i32(4));
        Value* o_row = c.b.build_add(c.b.build_mul(row0, p[12]), qoff);
        for (std::uint32_t dc = 0; dc < ND; ++dc) {
            emit_coop_acc_map(c, acc[dc], [&](Value* idx, Value* av) {
                Value* ri = c.b.build_and(kb.f32_to_u32(kb.coop_get_acc(RI, idx)), c.i32(15));
                return c.b.build_mul(av, kb.shared_load_f32_indexed(rowfac, c.b.build_add(sg_rbase, ri)));
            });
            kb.coop_store_acc(acc[dc], ptr_at(c, p[3], c.b.build_add(o_row, c.i32(static_cast<int>(dc * 16u))), 2),
                              p[12], false, true);
        }
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    Value* sg_rbase = c.b.build_shl(sg, c.i32(4));
    for (std::uint32_t dc = 0; dc < ND; ++dc) {
        kb.coop_store_acc(acc[dc], c.b.build_add(Ss, c.b.build_shl(sg_sbase, c.i32(2))), c.i32(16), false, false);
        kb.subgroup_barrier();
        for (int e0 = 0; e0 < 256; e0 += 32) {
            Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
            Value* rr = c.b.build_lshr(e, c.i32(4));
            Value* col = c.b.build_add(c.i32(static_cast<int>(dc * 16u)), c.b.build_and(e, c.i32(15)));
            Value* row = c.b.build_add(row0, rr);
            kb.if_then(c.b.build_and(c.b.build_ult(row, p[8]), c.b.build_ult(col, c.i32(static_cast<int>(HD)))), [&] {
                Value* val = c.b.build_mul(kb.shared_load_f32_indexed(Ss, c.b.build_add(sg_sbase, e)),
                                           kb.shared_load_f32_indexed(rowfac, c.b.build_add(sg_rbase, rr)));
                Value* dst = c.b.build_add(c.b.build_add(c.b.build_mul(row, p[12]), qoff), col);
                store_elem_dt(c, s.dt, p[3], dst, val);
            });
        }
        kb.subgroup_barrier();
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
