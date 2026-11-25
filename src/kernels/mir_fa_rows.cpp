#include "mir_fa_rows.h"

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

std::pair<Value*, Value*> emit_fa_interval(Ctx& c, std::uint32_t mode,
                                           Value* qi, Value* lk, Value* causal, Value* window,
                                           Value* q_offset, Value* lq, Value* nseq, Value* aux, Value* aux2) {
    if (mode == FA_MODE_VARLEN) {
        auto st = c.kb.for_range_reduce_n(
            c.i32(0), c.i32(24), c.i32(1), {c.i32(0), nseq},
            [&](Value* /*step*/, const std::vector<Value*>& cur) -> std::vector<Value*> {
                Value* slo = cur[0];
                Value* shi = cur[1];
                Value* active = c.b.build_slt(slo, shi);
                Value* diff = c.b.build_sub(shi, slo);
                Value* mid = c.b.build_add(slo, c.b.build_ashr(diff, c.i32(1)));
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
        Value* half_w = c.b.build_ashr(window, c.i32(1));
        Value* wlo_b = smax_i32(c, lo, c.b.build_sub(aq, half_w));
        Value* whi_b = smin_i32(c, hi, c.b.build_add(c.b.build_add(aq, half_w), c.i32(1)));
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
    if (mode == FA_MODE_LEN) {
        Value* dlen = clamp_i32(c, c.kb.load_i32(aux), c.i32(0), lk);
        Value* pq = c.b.build_add(c.b.build_sub(dlen, lq), qi);
        Value* hi = clamp_i32(c, c.b.build_add(pq, c.i32(1)), c.i32(0), lk);
        Value* lo = c.b.build_select(
            c.b.build_sgt(window, c.i32(0)),
            smax_i32(c, c.i32(0), c.b.build_add(c.b.build_sub(pq, window), c.i32(1))), c.i32(0));
        return {lo, smax_i32(c, lo, hi)};
    }
    // FA_MODE_ROWS
    Value* pq = c.b.build_add(q_offset, qi);
    Value* is_c = c.b.build_ne(causal, c.i32(0));
    Value* has_win = c.b.build_sgt(window, c.i32(0));
    Value* hw = c.b.build_ashr(window, c.i32(1));
    Value* hi_c = clamp_i32(c, c.b.build_add(pq, c.i32(1)), c.i32(0), lk);
    Value* hi_nc = c.b.build_select(has_win, clamp_i32(c, c.b.build_add(c.b.build_add(pq, hw), c.i32(1)), c.i32(0), lk), lk);
    Value* hi = c.b.build_select(is_c, hi_c, hi_nc);
    Value* lo_c = c.b.build_select(has_win, smax_i32(c, c.i32(0), c.b.build_add(c.b.build_sub(pq, window), c.i32(1))), c.i32(0));
    Value* lo_nc = c.b.build_select(has_win, smax_i32(c, c.i32(0), c.b.build_sub(pq, hw)), c.i32(0));
    Value* lo = c.b.build_select(is_c, lo_c, lo_nc);
    return {lo, smax_i32(c, lo, hi)};
}

}  // namespace

brass::Function* build_fa_rows_mir(brass::Module& mod, Dtype dt, std::uint32_t hd, std::uint32_t gqa_g,
                                   std::uint32_t mode, bool has_mask, bool split,
                                   bool has_softcap, bool vec, bool has_maskpos) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("brotensor: mir_fa_rows: unsupported dtype");
    }
    if (hd == 0 || hd > 8192u || gqa_g == 0 || gqa_g > 16u) {
        throw std::runtime_error("brotensor: mir_fa_rows: bad head_dim or G");
    }
    const std::uint32_t hds = (hd + 127u) / 128u;
    if (gqa_g * hds > 16u || gqa_g * hd > 8192u) {
        throw std::runtime_error("brotensor: mir_fa_rows: G * HDS / shared limit exceeded");
    }
    const std::string name = std::string("mir_fa_rows_") +
                             (dt == Dtype::FP32 ? "f32" : dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_hd" + std::to_string(hd) + "_g" + std::to_string(gqa_g) +
                             "_m" + std::to_string(mode) + (has_mask ? "_msk" : "") +
                             (split ? "_sp" : "") + (has_softcap ? "_sc" : "") +
                             (vec ? "_v" : "") + (has_maskpos ? "_mp" : "");

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::f32(), Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: 0:q 1:k 2:v 3:o 4:mask 5:aux 6:aux2 7:part
    // ints: 8:lq 9:lk 10:ldq 11:ldk 12:ldo 13:group 14:window 15:causal 16:nseq 17:chunk 18:hq 19:qb0 20:q_offset
    // flts: 21:scale 22:softcap

    const int esz = elem_bytes(dt);
    const bool vec_qk_multi = vec && (hd == 64u || hd == 128u);
    const std::uint32_t qk_kpw = !vec_qk_multi ? 1u : (hd == 64u ? 4u : 2u);
    const std::uint32_t qk_lanes_per_k = 32u / qk_kpw; // 8 (hd=64), 16 (hd=128), 32 (fallback)
    const bool vec_pv = vec && (hd == 64u || hd == 128u) && gqa_g <= 4u;
    const std::uint32_t pv_cols_v4 = hd / 8u;           // 8 (hd=64) or 16 (hd=128)
    const std::uint32_t pv_k_step = vec_pv ? (128u / pv_cols_v4) : 1u; // 16 or 8

    Value* sQ = kb.shared_alloc_f32(gqa_g * hd);
    Value* sS = kb.shared_alloc_f32(gqa_g * 128u);
    Value* s_valid = kb.shared_alloc_i32(128u);
    Value* s_red = kb.shared_alloc_f32(2u * gqa_g * 4u);
    Value* sO = vec_pv ? kb.shared_alloc_f32(pv_k_step * hd) : nullptr;
    Value* s_mp = has_maskpos ? kb.shared_alloc_i32(4u) : nullptr;

    Value* tid = kb.tid_x();
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* sp = kb.ctaid_x();
    Value* qi = c.b.build_add(p[19], kb.ctaid_y());
    Value* hg = kb.ctaid_z();
    Value* qh0 = c.b.build_mul(hg, c.i32(static_cast<int>(gqa_g)));
    Value* hkv = c.b.build_udiv(qh0, p[13]);

    // Load Q * scale -> sQ[g, d]
    Value* q_row_base = c.b.build_add(c.b.build_mul(qi, p[10]), c.b.build_mul(qh0, c.i32(static_cast<int>(hd))));
    for (std::uint32_t sl = 0; sl < hds; ++sl) {
        Value* d = c.b.build_add(c.i32(static_cast<int>(sl * 128u)), tid);
        auto load_q_slot = [&] {
            for (std::uint32_t g = 0; g < gqa_g; ++g) {
                Value* qidx = c.b.build_add(c.b.build_add(q_row_base, c.i32(static_cast<int>(g * hd))), d);
                Value* qv = c.b.build_mul(load_elem_dt(c, dt, p[0], qidx), p[21]);
                kb.shared_store_f32_indexed(sQ, c.b.build_add(c.i32(static_cast<int>(g * hd)), d), qv);
            }
        };
        if ((sl + 1u) * 128u <= hd) load_q_slot();
        else kb.if_then(c.b.build_ult(d, c.i32(static_cast<int>(hd))), load_q_slot);
    }
    kb.sync();

    auto [iv_lo, iv_hi] = has_maskpos
        ? std::pair<Value*, Value*>{c.i32(0), p[9]}
        : emit_fa_interval(c, mode, qi, p[9], p[15], p[14], p[20], p[8], p[16], p[5], p[6]);
    Value* lo = iv_lo;
    Value* hi = iv_hi;

    if (has_maskpos && has_mask) {
        Value* best = kb.for_range_reduce(
            c.b.build_add(lo, tid), hi, c.i32(128), c.i32(-1),
            [&](Value* kk, Value* cur) -> Value* {
                Value* mv = kb.load_f32(ptr_at(c, p[4], kk, 4));
                return c.b.build_select(c.b.build_slt(c.f32(0.5f), mv), smax_i32(c, cur, kk), cur);
            });
        for (std::uint32_t m = 16; m >= 1; m >>= 1) best = smax_i32(c, best, kb.shfl_bfly_i32(best, m));
        kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] { kb.shared_store_i32_indexed(s_mp, warp, best); });
        kb.sync();
        Value* b01 = smax_i32(c, kb.shared_load_i32_indexed(s_mp, c.i32(0)), kb.shared_load_i32_indexed(s_mp, c.i32(1)));
        Value* b23 = smax_i32(c, kb.shared_load_i32_indexed(s_mp, c.i32(2)), kb.shared_load_i32_indexed(s_mp, c.i32(3)));
        Value* last_k = smax_i32(c, b01, b23);
        kb.sync();
        Value* has_pos = c.b.build_sge(last_k, c.i32(0));
        lo = c.b.build_select(
            has_pos, smax_i32(c, lo, c.b.build_add(c.b.build_sub(last_k, p[14]), c.i32(1))), lo);
        hi = c.b.build_select(has_pos, hi, lo);
    }

    Value* tile_lo = lo;
    Value* tile_hi = hi;
    if (split) {
        Value* sp_lo = c.b.build_mul(sp, p[17]);
        Value* sp_hi = c.b.build_add(sp_lo, p[17]);
        tile_lo = smax_i32(c, lo, sp_lo);
        tile_hi = smax_i32(c, tile_lo, smin_i32(c, hi, sp_hi));
    }

    // State vector:
    //   When vec_pv: [0 .. 8*G-1] = o_vec8, else [0 .. G*HDS-1] = o_reg
    //   then [n_o .. n_o+G-1] = m, [n_o+G .. n_o+2G-1] = l
    const std::size_t n_o = vec_pv ? (gqa_g * 8u) : (gqa_g * hds);
    const std::size_t m_off = n_o;
    const std::size_t l_off = n_o + gqa_g;
    std::vector<Value*> init(n_o + 2u * gqa_g, c.f32(0.0f));
    for (std::uint32_t g = 0; g < gqa_g; ++g) init[m_off + g] = c.f32(-1e30f);

    Value* kv_head_off = c.b.build_mul(hkv, c.i32(static_cast<int>(hd)));
    Value* softcap_inv = has_softcap ? emit_div_rn(c, c.f32(1.0f), p[22]) : nullptr;
    Value* pv_kslot = vec_pv ? c.b.build_udiv(tid, c.i32(static_cast<int>(pv_cols_v4))) : nullptr;
    Value* pv_dv4 = vec_pv ? c.b.build_umod(tid, c.i32(static_cast<int>(pv_cols_v4))) : nullptr;

    std::vector<Value*> state = kb.for_range_reduce_n(
        tile_lo, tile_hi, c.i32(128), init,
        [&](Value* t0, const std::vector<Value*>& cur) -> std::vector<Value*> {
            Value* tile_n = smin_i32(c, c.i32(128), c.b.build_sub(tile_hi, t0));
            Value* in_t = c.b.build_ult(tid, tile_n);
            Value* valid_k = in_t;
            if (has_mask) {
                Value* mv = emit_if_else_values(
                    c, in_t, {Type::f32()},
                    [&]() -> std::vector<Value*> { return {kb.load_f32(ptr_at(c, p[4], c.b.build_add(t0, tid), 4))}; },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                valid_k = c.b.build_and(in_t, c.b.build_slt(c.f32(0.5f), mv));
            }
            kb.shared_store_i32_indexed(s_valid, tid, c.b.build_select(valid_k, c.i32(1), c.i32(0)));
            kb.sync();

            // 1. Q * K^T scores: 4 warps x qk_kpw keys per warp (full 32-lane utilization when vec_qk_multi).
            const std::uint32_t qk_step = 4u * qk_kpw;
            Value* qk_sub = vec_qk_multi
                ? c.b.build_udiv(lane, c.i32(static_cast<int>(qk_lanes_per_k)))
                : c.i32(0);
            Value* qk_lane = vec_qk_multi
                ? c.b.build_and(lane, c.i32(static_cast<int>(qk_lanes_per_k - 1u)))
                : lane;
            Value* qk_warp_k = c.b.build_add(c.b.build_mul(warp, c.i32(static_cast<int>(qk_kpw))), qk_sub);

            kb.for_range(c.i32(0), tile_n, c.i32(static_cast<int>(qk_step)), [&](Value* ks_base) {
                Value* ks = c.b.build_add(ks_base, qk_warp_k);
                Value* in_ks = c.b.build_ult(ks, tile_n);
                Value* safe_ks = c.b.build_select(in_ks, c.b.build_and(ks, c.i32(127)), c.i32(0));
                Value* k_ok = c.b.build_and(
                    in_ks, c.b.build_ne(kb.shared_load_i32_indexed(s_valid, safe_ks), c.i32(0)));
                std::vector<Value*> dots = emit_if_else_values(
                    c, k_ok, std::vector<Type>(gqa_g, Type::f32()),
                    [&]() -> std::vector<Value*> {
                        Value* kg = c.b.build_add(t0, safe_ks);
                        Value* k_row = c.b.build_add(c.b.build_mul(kg, p[11]), kv_head_off);
                        std::vector<Value*> d_init(gqa_g, c.f32(0.0f));
                        if (vec_qk_multi) {
                            Value* db = c.b.build_shl(qk_lane, c.i32(3));
                            Value* kp = ptr_at(c, p[1], c.b.build_add(k_row, db), esz);
                            std::vector<Value*> kv;
                            if (dt == Dtype::FP32) {
                                kv = load_lanes(c, Dtype::FP32, 4, kp);
                                auto kv2 = load_lanes(c, Dtype::FP32, 4, c.b.build_add(kp, kb.const_i64(16)));
                                kv.insert(kv.end(), kv2.begin(), kv2.end());
                            } else {
                                kv = load_lanes(c, dt, 8, kp);
                            }
                            for (std::uint32_t g = 0; g < gqa_g; ++g) {
                                Value* s_off = c.b.build_add(c.i32(static_cast<int>(g * hd)), db);
                                for (int e = 0; e < 8; ++e) {
                                    Value* qv = kb.shared_load_f32_indexed(sQ, c.b.build_add(s_off, c.i32(e)));
                                    d_init[g] = c.b.build_fma_f32(qv, kv[e], d_init[g]);
                                }
                            }
                            return d_init;
                        }
                        if (vec) {
                            return kb.for_range_reduce_n(
                                qk_lane, c.i32(static_cast<int>(hd / 8u)), c.i32(32), d_init,
                                [&](Value* dc_chunk, const std::vector<Value*>& dc) -> std::vector<Value*> {
                                    Value* db = c.b.build_shl(dc_chunk, c.i32(3));
                                    std::vector<Value*> dn = dc;
                                    Value* kp = ptr_at(c, p[1], c.b.build_add(k_row, db), esz);
                                    std::vector<Value*> kv;
                                    if (dt == Dtype::FP32) {
                                        kv = load_lanes(c, Dtype::FP32, 4, kp);
                                        auto kv2 = load_lanes(c, Dtype::FP32, 4, c.b.build_add(kp, kb.const_i64(16)));
                                        kv.insert(kv.end(), kv2.begin(), kv2.end());
                                    } else {
                                        kv = load_lanes(c, dt, 8, kp);
                                    }
                                    for (std::uint32_t g = 0; g < gqa_g; ++g) {
                                        Value* s_off = c.b.build_add(c.i32(static_cast<int>(g * hd)), db);
                                        for (int e = 0; e < 8; ++e) {
                                            Value* qv = kb.shared_load_f32_indexed(sQ, c.b.build_add(s_off, c.i32(e)));
                                            dn[g] = c.b.build_fma_f32(qv, kv[e], dn[g]);
                                        }
                                    }
                                    return dn;
                                });
                        }
                        return kb.for_range_reduce_n(
                            qk_lane, c.i32(static_cast<int>(hd)), c.i32(32), d_init,
                            [&](Value* d, const std::vector<Value*>& dc) -> std::vector<Value*> {
                                std::vector<Value*> dn = dc;
                                Value* kv = load_elem_dt(c, dt, p[1], c.b.build_add(k_row, d));
                                for (std::uint32_t g = 0; g < gqa_g; ++g) {
                                    Value* qv = kb.shared_load_f32_indexed(
                                        sQ, c.b.build_add(c.i32(static_cast<int>(g * hd)), d));
                                    dn[g] = c.b.build_fma_f32(qv, kv, dn[g]);
                                }
                                return dn;
                            });
                    },
                    [&]() -> std::vector<Value*> { return std::vector<Value*>(gqa_g, c.f32(0.0f)); });

                for (std::uint32_t g = 0; g < gqa_g; ++g) {
                    Value* s = dots[g];
                    for (std::uint32_t m = qk_lanes_per_k >> 1; m >= 1; m >>= 1) {
                        s = c.b.build_add(s, kb.shfl_bfly_f32(s, m));
                    }
                    if (has_softcap) {
                        s = c.b.build_mul(p[22], emit_tanh(c, c.b.build_mul(s, softcap_inv)));
                    }
                    kb.if_then(c.b.build_and(c.b.build_eq(qk_lane, c.i32(0)), in_ks), [&] {
                        Value* sval = c.b.build_select(k_ok, s, c.f32(-1e30f));
                        kb.shared_store_f32_indexed(sS, c.b.build_add(c.i32(static_cast<int>(g * 128u)), safe_ks), sval);
                    });
                }
            });
            kb.sync();

            // 2. Batched per-head online softmax & store probabilities in sS (only 2 barriers for all G heads).
            Value* self_valid = c.b.build_ne(kb.shared_load_i32_indexed(s_valid, tid), c.i32(0));
            std::vector<Value*> sc_v(gqa_g);
            for (std::uint32_t g = 0; g < gqa_g; ++g) {
                Value* sidx = c.b.build_add(c.i32(static_cast<int>(g * 128u)), tid);
                Value* sc = c.b.build_select(self_valid, kb.shared_load_f32_indexed(sS, sidx), c.f32(-1e30f));
                sc_v[g] = sc;
                Value* mx = sc;
                for (std::uint32_t m = 16; m >= 1; m >>= 1) mx = c.kb.fmax(mx, kb.shfl_bfly_f32(mx, m));
                kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] {
                    kb.shared_store_f32_indexed(s_red, c.b.build_add(c.i32(static_cast<int>(g * 4u)), warp), mx);
                });
            }
            kb.sync();

            std::vector<Value*> next = cur;
            for (std::uint32_t g = 0; g < gqa_g; ++g) {
                const int rb = static_cast<int>(g * 4u);
                Value* m01 = kb.fmax(kb.shared_load_f32_indexed(s_red, c.i32(rb + 0)),
                                     kb.shared_load_f32_indexed(s_red, c.i32(rb + 1)));
                Value* m23 = kb.fmax(kb.shared_load_f32_indexed(s_red, c.i32(rb + 2)),
                                     kb.shared_load_f32_indexed(s_red, c.i32(rb + 3)));
                Value* m_tile = kb.fmax(m01, m23);
                Value* m_prev = cur[m_off + g];
                Value* m_new = c.kb.fmax(m_prev, m_tile);
                Value* has_any = c.b.build_sgt(m_new, c.f32(-1e29f));
                Value* had_prev = c.b.build_sgt(m_prev, c.f32(-1e29f));
                Value* alpha = c.b.build_select(
                    c.b.build_and(has_any, had_prev),
                    kb.ex2_approx(c.b.build_mul(c.b.build_sub(m_prev, m_new), c.f32(kLog2e))),
                    c.f32(0.0f));
                Value* pv = c.b.build_select(
                    c.b.build_and(has_any, self_valid),
                    kb.ex2_approx(c.b.build_mul(c.b.build_sub(sc_v[g], m_new), c.f32(kLog2e))),
                    c.f32(0.0f));
                Value* sidx = c.b.build_add(c.i32(static_cast<int>(g * 128u)), tid);
                kb.shared_store_f32_indexed(sS, sidx, pv);
                Value* wsum = pv;
                for (std::uint32_t m = 16; m >= 1; m >>= 1) wsum = c.b.build_add(wsum, kb.shfl_bfly_f32(wsum, m));
                kb.if_then(c.b.build_eq(lane, c.i32(0)), [&] {
                    kb.shared_store_f32_indexed(
                        s_red, c.b.build_add(c.i32(static_cast<int>((gqa_g + g) * 4u)), warp), wsum);
                });
                next[m_off + g] = m_new;
                next[l_off + g] = alpha; // stash alpha temporarily until l_tile sync finishes
            }
            kb.sync();

            for (std::uint32_t g = 0; g < gqa_g; ++g) {
                const int sb = static_cast<int>((gqa_g + g) * 4u);
                Value* s01 = c.b.build_add(kb.shared_load_f32_indexed(s_red, c.i32(sb + 0)),
                                           kb.shared_load_f32_indexed(s_red, c.i32(sb + 1)));
                Value* s23 = c.b.build_add(kb.shared_load_f32_indexed(s_red, c.i32(sb + 2)),
                                           kb.shared_load_f32_indexed(s_red, c.i32(sb + 3)));
                Value* l_tile = c.b.build_add(s01, s23);
                Value* alpha = next[l_off + g];
                next[l_off + g] = c.b.build_fma_f32(cur[l_off + g], alpha, l_tile);
                if (vec_pv) {
                    for (int e = 0; e < 8; ++e) {
                        next[g * 8u + static_cast<std::size_t>(e)] =
                            c.b.build_mul(cur[g * 8u + static_cast<std::size_t>(e)], alpha);
                    }
                } else {
                    for (std::uint32_t sl = 0; sl < hds; ++sl) {
                        next[g * hds + sl] = c.b.build_mul(cur[g * hds + sl], alpha);
                    }
                }
            }

            // 3. P * V accumulation: read V[t0 + ks, d] across all G query heads.
            std::vector<Value*> o_init(next.begin(), next.begin() + static_cast<std::ptrdiff_t>(n_o));
            std::vector<Value*> o_updated = vec_pv
                ? kb.for_range_reduce_n(
                      c.i32(0), tile_n, c.i32(static_cast<int>(pv_k_step)), o_init,
                      [&](Value* ks_base, const std::vector<Value*>& o_cur) -> std::vector<Value*> {
                          Value* ks = c.b.build_add(ks_base, pv_kslot);
                          Value* in_k = c.b.build_ult(ks, tile_n);
                          Value* safe_ks = c.b.build_select(in_k, c.b.build_and(ks, c.i32(127)), c.i32(0));
                          Value* v_ok = c.b.build_and(
                              in_k, c.b.build_ne(kb.shared_load_i32_indexed(s_valid, safe_ks), c.i32(0)));
                          return emit_if_else_values(
                              c, v_ok, std::vector<Type>(n_o, Type::f32()),
                              [&]() -> std::vector<Value*> {
                                  Value* v_row = c.b.build_add(
                                      c.b.build_mul(c.b.build_add(t0, safe_ks), p[11]), kv_head_off);
                                  Value* vp = ptr_at(c, p[2], c.b.build_add(v_row, c.b.build_shl(pv_dv4, c.i32(3))), esz);
                                  std::vector<Value*> vv;
                                  if (dt == Dtype::FP32) {
                                      vv = load_lanes(c, Dtype::FP32, 4, vp);
                                      auto vv2 = load_lanes(c, Dtype::FP32, 4, c.b.build_add(vp, kb.const_i64(16)));
                                      vv.insert(vv.end(), vv2.begin(), vv2.end());
                                  } else {
                                      vv = load_lanes(c, dt, 8, vp);
                                  }
                                  std::vector<Value*> o_next = o_cur;
                                  for (std::uint32_t g = 0; g < gqa_g; ++g) {
                                      Value* pv = kb.shared_load_f32_indexed(
                                          sS, c.b.build_add(c.i32(static_cast<int>(g * 128u)), safe_ks));
                                      for (int e = 0; e < 8; ++e) {
                                          const std::size_t idx = g * 8u + static_cast<std::size_t>(e);
                                          o_next[idx] = c.b.build_fma_f32(pv, vv[static_cast<std::size_t>(e)], o_next[idx]);
                                      }
                                  }
                                  return o_next;
                              },
                              [&]() -> std::vector<Value*> { return o_cur; });
                      })
                : kb.for_range_reduce_n(
                      c.i32(0), tile_n, c.i32(1), o_init,
                      [&](Value* ks, const std::vector<Value*>& o_cur) -> std::vector<Value*> {
                          Value* v_ok = c.b.build_ne(kb.shared_load_i32_indexed(s_valid, ks), c.i32(0));
                          return emit_if_else_values(
                              c, v_ok, std::vector<Type>(n_o, Type::f32()),
                              [&]() -> std::vector<Value*> {
                                  Value* v_row = c.b.build_add(c.b.build_mul(c.b.build_add(t0, ks), p[11]), kv_head_off);
                                  std::vector<Value*> pv(gqa_g);
                                  for (std::uint32_t g = 0; g < gqa_g; ++g) {
                                      pv[g] = kb.shared_load_f32_indexed(
                                          sS, c.b.build_add(c.i32(static_cast<int>(g * 128u)), ks));
                                  }
                                  std::vector<Value*> o_next = o_cur;
                                  for (std::uint32_t sl = 0; sl < hds; ++sl) {
                                      Value* d = c.b.build_add(c.i32(static_cast<int>(sl * 128u)), tid);
                                      Value* vv = ((sl + 1u) * 128u <= hd)
                                                      ? load_elem_dt(c, dt, p[2], c.b.build_add(v_row, d))
                                                      : emit_if_else_values(
                                                            c, c.b.build_ult(d, c.i32(static_cast<int>(hd))), {Type::f32()},
                                                            [&]() -> std::vector<Value*> {
                                                                return {load_elem_dt(c, dt, p[2], c.b.build_add(v_row, d))};
                                                            },
                                                            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                                      for (std::uint32_t g = 0; g < gqa_g; ++g) {
                                          o_next[g * hds + sl] = c.b.build_fma_f32(pv[g], vv, o_next[g * hds + sl]);
                                      }
                                  }
                                  return o_next;
                              },
                              [&]() -> std::vector<Value*> { return o_cur; });
                      });
            for (std::size_t i = 0; i < n_o; ++i) next[i] = o_updated[i];
            kb.sync();
            return next;
        });

    std::vector<Value*> final_o(gqa_g * hds, c.f32(0.0f));
    if (vec_pv) {
        Value* slot_base = c.b.build_add(
            c.b.build_mul(pv_kslot, c.i32(static_cast<int>(hd))),
            c.b.build_shl(pv_dv4, c.i32(3)));
        for (std::uint32_t g = 0; g < gqa_g; ++g) {
            for (int e = 0; e < 8; ++e) {
                kb.shared_store_f32_indexed(
                    sO, c.b.build_add(slot_base, c.i32(e)), state[g * 8u + static_cast<std::size_t>(e)]);
            }
            kb.sync();
            Value* acc_d = c.f32(0.0f);
            if (hd == 128u) {
                for (std::uint32_t k = 0; k < pv_k_step; ++k) {
                    acc_d = c.b.build_add(
                        acc_d, kb.shared_load_f32_indexed(sO, c.b.build_add(c.i32(static_cast<int>(k * hd)), tid)));
                }
            } else {
                Value* d_safe = c.b.build_select(c.b.build_ult(tid, c.i32(static_cast<int>(hd))), tid, c.i32(0));
                for (std::uint32_t k = 0; k < pv_k_step; ++k) {
                    acc_d = c.b.build_add(
                        acc_d, kb.shared_load_f32_indexed(sO, c.b.build_add(c.i32(static_cast<int>(k * hd)), d_safe)));
                }
            }
            final_o[g * hds] = acc_d;
            if (g + 1u < gqa_g) kb.sync();
        }
    } else {
        for (std::size_t i = 0; i < gqa_g * hds; ++i) final_o[i] = state[i];
    }

    if (split) {
        Value* nsplit = kb.nctaid_x();
        Value* pw = c.i32(static_cast<int>(hd + 2u));
        for (std::uint32_t g = 0; g < gqa_g; ++g) {
            Value* qh = c.b.build_add(qh0, c.i32(static_cast<int>(g)));
            Value* entry = c.b.build_add(c.b.build_mul(qi, p[18]), qh);
            Value* row_idx = c.b.build_add(c.b.build_mul(entry, nsplit), sp);
            Value* row_base = c.b.build_mul(row_idx, pw);
            for (std::uint32_t sl = 0; sl < hds; ++sl) {
                Value* d = c.b.build_add(c.i32(static_cast<int>(sl * 128u)), tid);
                auto store_p = [&] {
                    kb.store_f32(ptr_at(c, p[7], c.b.build_add(row_base, d), 4), final_o[g * hds + sl]);
                };
                if ((sl + 1u) * 128u <= hd) store_p();
                else kb.if_then(c.b.build_ult(d, c.i32(static_cast<int>(hd))), store_p);
            }
            kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                kb.store_f32(ptr_at(c, p[7], c.b.build_add(row_base, c.i32(static_cast<int>(hd))), 4),
                             state[m_off + g]);
                kb.store_f32(ptr_at(c, p[7], c.b.build_add(row_base, c.i32(static_cast<int>(hd + 1u))), 4),
                             state[l_off + g]);
            });
        }
    } else {
        Value* o_row_base = c.b.build_add(c.b.build_mul(qi, p[12]), c.b.build_mul(qh0, c.i32(static_cast<int>(hd))));
        for (std::uint32_t g = 0; g < gqa_g; ++g) {
            Value* lv = state[l_off + g];
            Value* inv = c.b.build_select(c.b.build_sgt(lv, c.f32(0.0f)), emit_div_rn(c, c.f32(1.0f), lv), c.f32(0.0f));
            for (std::uint32_t sl = 0; sl < hds; ++sl) {
                Value* d = c.b.build_add(c.i32(static_cast<int>(sl * 128u)), tid);
                auto store_o = [&] {
                    Value* oidx = c.b.build_add(c.b.build_add(o_row_base, c.i32(static_cast<int>(g * hd))), d);
                    store_elem_dt(c, dt, p[3], oidx, c.b.build_mul(final_o[g * hds + sl], inv));
                };
                if ((sl + 1u) * 128u <= hd) store_o();
                else kb.if_then(c.b.build_ult(d, c.i32(static_cast<int>(hd))), store_o);
            }
        }
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
