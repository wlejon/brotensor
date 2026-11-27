#include "mir_fa_fused_mma.h"

#include "mir_common.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* ptr_idx_scaled(Ctx& c, Value* base, Value* elem_idx, int esz) {
    return ptr_at(c, base, elem_idx, esz);
}

Value* pack_pair_i32(Ctx& c, Dtype dt, Value* v0, Value* v1) {
    if (dt == Dtype::FP16) return c.kb.pack_f16x2(v0, v1);
    Value* b0 = c.b.build_and(c.kb.f32_to_bf16(v0), c.i32(0xffff));
    Value* b1 = c.b.build_shl(c.kb.f32_to_bf16(v1), c.i32(16));
    return c.b.build_or(b0, b1);
}

Value* mma_sync_f32(Ctx& c, Dtype dt, Value* a4, Value* b0, Value* b1, Value* c4) {
    const char* sym = (dt == Dtype::FP16) ? "ptx_mma_m16n8k16_f16_f32" : "ptx_mma_m16n8k16_bf16_f32";
    return c.b.build_call(sym, Type::f32x4(), {a4, b0, b1, c4});
}

void emit_load_tile_cp_async(Ctx& c, Value* smem_base_i32, Value* g_base_ptr,
                             Value* row0, Value* row_limit, Value* ld, Value* col0,
                             int rows, int hd_pad, int lds, int segs_real, Value* safe_ptr) {
    Value* tid = c.kb.tid_x();
    const int nthreads = static_cast<int>(kFaFusedThreads);
    const int segs = hd_pad / 8;
    const int total = rows * segs;
    const int per = (total + nthreads - 1) / nthreads;
    const bool need_guard = (per * nthreads != total);

    for (int i = 0; i < per; ++i) {
        Value* idx = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * nthreads));
        auto body = [&] {
            Value* r = c.b.build_udiv(idx, c.i32(segs));
            Value* seg = c.b.build_umod(idx, c.i32(segs));
            Value* gr = c.b.build_add(row0, r);
            Value* r_ok = c.b.build_slt(gr, row_limit);
            Value* s_ok = (segs_real < segs)
                ? c.b.build_and(r_ok, c.b.build_slt(seg, c.i32(segs_real)))
                : r_ok;
            Value* dst = c.b.build_add(
                smem_base_i32,
                c.b.build_shl(c.b.build_add(c.b.build_mul(r, c.i32(lds)), c.b.build_shl(seg, c.i32(3))),
                              c.i32(1)));
            Value* g_elem = c.b.build_add(c.b.build_add(c.b.build_mul(gr, ld), col0),
                                          c.b.build_shl(seg, c.i32(3)));
            Value* g_ptr = ptr_idx_scaled(c, g_base_ptr, g_elem, 2);
            Value* safe_g = c.b.build_select(s_ok, g_ptr, safe_ptr);
            Value* sz = c.b.build_select(s_ok, c.i32(16), c.i32(0));
            c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, safe_g, sz});
        };
        if (need_guard) {
            c.kb.if_then(c.b.build_slt(idx, c.i32(total)), body);
        } else {
            body();
        }
    }
}

}  // namespace

brass::Function* build_fa_fused_mma_mir(brass::Module& mod, const FaFusedMmaSpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_fa_fused_mma_mir: only FP16/BF16 supported");
    }
    if (spec.hd == 0u || (spec.hd % 8u) != 0u || spec.hd > 160u) {
        throw std::runtime_error("build_fa_fused_mma_mir: unsupported head_dim");
    }
    const int HD = static_cast<int>(spec.hd);
    const int PAD = static_cast<int>(fa_fused_pad_hd(spec.hd));
    const int BC = static_cast<int>(fa_fused_bc(spec.hd));
    const int BR = static_cast<int>(kFaFusedBr);
    const int LDS = PAD + 8;
    const int KT = PAD / 16;
    const int NT = BC / 8;
    const int KC_PV = BC / 16;
    const int SEGS_REAL = HD / 8;
    const int stage_elems = BC * LDS;
    const int stage_bytes = stage_elems * 2;
    const int kv_total_elems = 4 * stage_elems;  // sK[2] + sV[2]

    const std::string name = std::string("mir_fa_fused_") +
                             (spec.dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_hd" + std::to_string(HD) +
                             (spec.causal ? "_c" : "") +
                             (spec.has_mask ? "_m" : "") +
                             (spec.is_split ? "_sp" : "");

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:Q 1:K 2:V 3:mask 4:O 5:part_o 6:part_ml
    //    7:Lq 8:Lk 9:D 10:Dkv 11:group 12:splits 13:qtiles 14:heads | 15:scale_log2
    Value* Q_in = p[0];
    Value* K_in = p[1];
    Value* V_in = p[2];
    Value* mask_in = p[3];
    Value* O_in = p[4];
    Value* part_o = p[5];
    Value* part_ml = p[6];
    Value* Lq = p[7];
    Value* Lk = p[8];
    Value* D = p[9];
    Value* Dkv = p[10];
    Value* group = p[11];
    Value* splits = p[12];
    Value* qtiles = p[13];
    Value* heads = p[14];
    Value* scale_log2 = p[15];

    Value* smem = kb.shared_alloc_u16(static_cast<std::uint32_t>(kv_total_elems));
    Value* sK_base = c.b.build_trunc_i32(smem);
    Value* sV_base = c.b.build_add(sK_base, c.i32(2 * stage_bytes));
    Value* sMask = spec.has_mask ? kb.shared_alloc_f32(static_cast<std::uint32_t>(BC)) : nullptr;

    Value* bz = kb.ctaid_z();
    Value* split = spec.is_split ? c.b.build_umod(bz, splits) : c.i32(0);
    Value* b_idx = spec.is_split ? c.b.build_udiv(bz, splits) : bz;

    Value* Q = ptr_idx_scaled(c, Q_in, c.b.build_mul(b_idx, c.b.build_mul(Lq, D)), 2);
    Value* K = ptr_idx_scaled(c, K_in, c.b.build_mul(b_idx, c.b.build_mul(Lk, Dkv)), 2);
    Value* V = ptr_idx_scaled(c, V_in, c.b.build_mul(b_idx, c.b.build_mul(Lk, Dkv)), 2);
    Value* O = ptr_idx_scaled(c, O_in, c.b.build_mul(b_idx, c.b.build_mul(Lq, D)), 2);
    Value* mask_b = spec.has_mask ? ptr_idx_scaled(c, mask_in, c.b.build_mul(b_idx, Lk), 4) : mask_in;

    Value* head = kb.ctaid_y();
    Value* head_off = c.b.build_mul(head, c.i32(HD));
    Value* kv_off = c.b.build_mul(c.b.build_udiv(head, group), c.i32(HD));

    Value* qtile = spec.causal
        ? c.b.build_sub(c.b.build_sub(qtiles, c.i32(1)), kb.ctaid_x())
        : kb.ctaid_x();
    Value* q0 = c.b.build_mul(qtile, c.i32(BR));

    Value* tid = kb.tid_x();
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t = c.b.build_and(lane, c.i32(3));
    Value* r0 = c.b.build_add(c.b.build_add(q0, c.b.build_shl(warp, c.i32(4))), g);
    Value* r1 = c.b.build_add(r0, c.i32(8));

    // Stage Q into sK_base (aliased as sQ) once, load Q_frag via ldmatrix.x4, then sync before KV.
    emit_load_tile_cp_async(c, sK_base, Q, q0, Lq, D, head_off, BR, PAD, LDS, SEGS_REAL, Q);
    c.b.build_call("ptx_cp_async_commit", Type::void_type());
    c.b.build_call("ptx_cp_async_wait_all", Type::void_type());
    kb.sync();

    std::vector<Value*> q_frag(static_cast<std::size_t>(KT));
    Value* q_lrow = c.b.build_add(c.b.build_shl(warp, c.i32(4)), c.b.build_and(lane, c.i32(15)));
    Value* q_lcol0 = c.b.build_shl(c.b.build_lshr(lane, c.i32(4)), c.i32(3));
    for (int kc = 0; kc < KT; ++kc) {
        Value* col = c.b.build_add(c.i32(kc * 16), q_lcol0);
        Value* q_addr = c.b.build_add(
            sK_base, c.b.build_shl(c.b.build_add(c.b.build_mul(q_lrow, c.i32(LDS)), col), c.i32(1)));
        q_frag[static_cast<std::size_t>(kc)] = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {q_addr});
    }
    kb.sync();

    // Key range & tile count for this CTA.
    Value* kv_end = spec.causal ? smin_i32(c, Lk, c.b.build_add(q0, c.i32(BR))) : Lk;
    Value* kv_begin = c.i32(0);
    if (spec.is_split) {
        Value* total_tiles = c.b.build_udiv(c.b.build_add(kv_end, c.i32(BC - 1)), c.i32(BC));
        Value* tps = c.b.build_udiv(c.b.build_sub(c.b.build_add(total_tiles, splits), c.i32(1)), splits);
        kv_begin = c.b.build_mul(c.b.build_mul(split, tps), c.i32(BC));
        kv_end = smin_i32(c, kv_end, c.b.build_add(kv_begin, c.b.build_mul(tps, c.i32(BC))));
    }
    Value* span_k = smax_i32(c, c.b.build_sub(kv_end, kv_begin), c.i32(0));
    Value* n_tiles = c.b.build_select(
        c.b.build_slt(kv_begin, kv_end),
        c.b.build_udiv(c.b.build_add(span_k, c.i32(BC - 1)), c.i32(BC)),
        c.i32(0));

    // Precompute ldmatrix offsets for K and V relative to stage 0 (stage 1 = +stage_bytes).
    Value* mi = c.b.build_lshr(lane, c.i32(3));
    Value* k_lrow0 = c.b.build_add(c.b.build_shl(c.b.build_lshr(mi, c.i32(1)), c.i32(3)),
                                   c.b.build_and(lane, c.i32(7)));
    Value* k_lcol0 = c.b.build_shl(c.b.build_and(mi, c.i32(1)), c.i32(3));
    Value* v_lrow0 = c.b.build_add(c.b.build_shl(c.b.build_and(mi, c.i32(1)), c.i32(3)),
                                   c.b.build_and(lane, c.i32(7)));
    Value* v_lcol0 = c.b.build_shl(c.b.build_lshr(mi, c.i32(1)), c.i32(3));

    std::vector<Type> loop_types(static_cast<std::size_t>(4 + 2 * KT), Type::f32());
    for (int dt_i = 0; dt_i < 2 * KT; ++dt_i) loop_types[static_cast<std::size_t>(4 + dt_i)] = Type::f32x4();
    std::vector<Value*> loop_inits(loop_types.size());
    loop_inits[0] = c.f32(-1e30f);
    loop_inits[1] = c.f32(-1e30f);
    loop_inits[2] = c.f32(0.0f);
    loop_inits[3] = c.f32(0.0f);
    for (int dt_i = 0; dt_i < 2 * KT; ++dt_i) loop_inits[static_cast<std::size_t>(4 + dt_i)] = c.b.build_vzero(Type::f32x4());

    auto res_vec = emit_if_else_values(
        c, c.b.build_sgt(n_tiles, c.i32(0)), loop_types,
        [&]() -> std::vector<Value*> {
            emit_load_tile_cp_async(c, sK_base, K, kv_begin, kv_end, Dkv, kv_off, BC, PAD, LDS, SEGS_REAL, K);
            emit_load_tile_cp_async(c, sV_base, V, kv_begin, kv_end, Dkv, kv_off, BC, PAD, LDS, SEGS_REAL, V);
            c.b.build_call("ptx_cp_async_commit", Type::void_type());

            return kb.for_range_reduce_n(
                c.i32(0), n_tiles, c.i32(1), loop_inits,
                [&](Value* j, const std::vector<Value*>& st) -> std::vector<Value*> {
                    Value* buf = c.b.build_and(j, c.i32(1));
                    Value* cur_k = c.b.build_add(sK_base, c.b.build_mul(buf, c.i32(stage_bytes)));
                    Value* cur_v = c.b.build_add(sV_base, c.b.build_mul(buf, c.i32(stage_bytes)));

                    Value* has_next = c.b.build_slt(c.b.build_add(j, c.i32(1)), n_tiles);
                    emit_if_else_values(
                        c, has_next, {},
                        [&]() -> std::vector<Value*> {
                            Value* nxt_buf = c.b.build_xor(buf, c.i32(1));
                            Value* nxt_k = c.b.build_add(sK_base, c.b.build_mul(nxt_buf, c.i32(stage_bytes)));
                            Value* nxt_v = c.b.build_add(sV_base, c.b.build_mul(nxt_buf, c.i32(stage_bytes)));
                            Value* nk = c.b.build_add(kv_begin, c.b.build_mul(c.b.build_add(j, c.i32(1)), c.i32(BC)));
                            emit_load_tile_cp_async(c, nxt_k, K, nk, kv_end, Dkv, kv_off, BC, PAD, LDS, SEGS_REAL, K);
                            emit_load_tile_cp_async(c, nxt_v, V, nk, kv_end, Dkv, kv_off, BC, PAD, LDS, SEGS_REAL, V);
                            c.b.build_call("ptx_cp_async_commit", Type::void_type());
                            c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(1)});
                            return {};
                        },
                        [&]() -> std::vector<Value*> {
                            c.b.build_call("ptx_cp_async_wait_all", Type::void_type());
                            return {};
                        });

                    Value* kb_row = c.b.build_add(kv_begin, c.b.build_mul(j, c.i32(BC)));
                    if (spec.has_mask) {
                        kb.if_then(c.b.build_slt(tid, c.i32(BC)), [&] {
                            Value* gk = c.b.build_add(kb_row, tid);
                            Value* ok = c.b.build_slt(gk, kv_end);
                            Value* mptr = c.b.build_select(ok, ptr_idx_scaled(c, mask_b, gk, 4), mask_b);
                            Value* mv = c.b.build_select(ok, kb.load_f32(mptr), c.f32(0.0f));
                            kb.shared_store_f32_indexed(sMask, tid, mv);
                        });
                    }
                    kb.sync();

                    // S = Q * K^T (NT f32x4 accumulators).
                    std::vector<Value*> s_acc(static_cast<std::size_t>(NT), c.b.build_vzero(Type::f32x4()));
                    for (int kc = 0; kc < KT; ++kc) {
                        Value* qf = q_frag[static_cast<std::size_t>(kc)];
                        Value* col_k = c.b.build_add(c.i32(kc * 16), k_lcol0);
                        for (int np = 0; np < BC / 16; ++np) {
                            Value* row_k = c.b.build_add(c.i32(np * 16), k_lrow0);
                            Value* k_addr = c.b.build_add(
                                cur_k, c.b.build_shl(c.b.build_add(c.b.build_mul(row_k, c.i32(LDS)), col_k), c.i32(1)));
                            Value* b4 = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {k_addr});
                            Value* b0 = c.b.build_vextract_lane(b4, 0);
                            Value* b1 = c.b.build_vextract_lane(b4, 1);
                            Value* b2 = c.b.build_vextract_lane(b4, 2);
                            Value* b3 = c.b.build_vextract_lane(b4, 3);
                            const auto n0 = static_cast<std::size_t>(2 * np);
                            const auto n1 = n0 + 1u;
                            s_acc[n0] = mma_sync_f32(c, spec.dt, qf, b0, b1, s_acc[n0]);
                            s_acc[n1] = mma_sync_f32(c, spec.dt, qf, b2, b3, s_acc[n1]);
                        }
                    }

                    // Scale + mask + online softmax in registers.
                    Value* mx0 = c.f32(-1e30f);
                    Value* mx1 = c.f32(-1e30f);
                    std::vector<std::array<Value*, 4>> s_elem(static_cast<std::size_t>(NT));
                    std::vector<std::array<Value*, 4>> s_valid(static_cast<std::size_t>(NT));
                    Value* key_t2 = c.b.build_shl(t, c.i32(1));

                    for (int nt = 0; nt < NT; ++nt) {
                        const auto ntu = static_cast<std::size_t>(nt);
                        Value* key0 = c.b.build_add(kb_row, c.b.build_add(c.i32(nt * 8), key_t2));
                        for (int e = 0; e < 2; ++e) {
                            Value* kk = (e == 0) ? key0 : c.b.build_add(key0, c.i32(1));
                            Value* in_k = c.b.build_slt(kk, kv_end);
                            if (spec.has_mask) {
                                Value* mcol = c.b.build_add(c.i32(nt * 8 + e), key_t2);
                                Value* mv = kb.shared_load_f32_indexed(sMask, mcol);
                                in_k = c.b.build_and(in_k, c.b.build_sgt(mv, c.f32(0.5f)));
                            }
                            Value* ok0 = spec.causal ? c.b.build_and(in_k, c.b.build_sle(kk, r0)) : in_k;
                            Value* ok1 = spec.causal ? c.b.build_and(in_k, c.b.build_sle(kk, r1)) : in_k;

                            Value* v0 = c.b.build_mul(c.b.build_vextract_lane(s_acc[ntu], static_cast<std::uint32_t>(e)), scale_log2);
                            Value* v1 = c.b.build_mul(c.b.build_vextract_lane(s_acc[ntu], static_cast<std::uint32_t>(2 + e)), scale_log2);
                            Value* m0v = c.b.build_select(ok0, v0, c.f32(-1e30f));
                            Value* m1v = c.b.build_select(ok1, v1, c.f32(-1e30f));
                            s_elem[ntu][static_cast<std::size_t>(e)] = m0v;
                            s_elem[ntu][static_cast<std::size_t>(2 + e)] = m1v;
                            s_valid[ntu][static_cast<std::size_t>(e)] = ok0;
                            s_valid[ntu][static_cast<std::size_t>(2 + e)] = ok1;
                            mx0 = kb.fmax(mx0, m0v);
                            mx1 = kb.fmax(mx1, m1v);
                        }
                    }

                    for (int off = 1; off <= 2; off <<= 1) {
                        mx0 = kb.fmax(mx0, kb.shfl_bfly_f32(mx0, static_cast<std::uint32_t>(off)));
                        mx1 = kb.fmax(mx1, kb.shfl_bfly_f32(mx1, static_cast<std::uint32_t>(off)));
                    }

                    Value* m0_old = st[0];
                    Value* m1_old = st[1];
                    Value* l0_old = st[2];
                    Value* l1_old = st[3];
                    Value* has0 = c.b.build_sgt(mx0, c.f32(-1e29f));
                    Value* has1 = c.b.build_sgt(mx1, c.f32(-1e29f));
                    Value* mn0 = c.b.build_select(has0, kb.fmax(m0_old, mx0), m0_old);
                    Value* mn1 = c.b.build_select(has1, kb.fmax(m1_old, mx1), m1_old);
                    Value* prev0 = c.b.build_sgt(m0_old, c.f32(-1e29f));
                    Value* prev1 = c.b.build_sgt(m1_old, c.f32(-1e29f));
                    Value* a0 = c.b.build_select(
                        c.b.build_and(prev0, has0), kb.ex2_approx(c.b.build_sub(m0_old, mn0)), c.f32(1.0f));
                    Value* a1 = c.b.build_select(
                        c.b.build_and(prev1, has1), kb.ex2_approx(c.b.build_sub(m1_old, mn1)), c.f32(1.0f));

                    Value* ps0 = c.f32(0.0f);
                    Value* ps1 = c.f32(0.0f);
                    for (int nt = 0; nt < NT; ++nt) {
                        const auto ntu = static_cast<std::size_t>(nt);
                        for (int e = 0; e < 2; ++e) {
                            const auto eu = static_cast<std::size_t>(e);
                            Value* p0 = c.b.build_select(
                                c.b.build_and(has0, s_valid[ntu][eu]),
                                kb.ex2_approx(c.b.build_sub(s_elem[ntu][eu], mn0)), c.f32(0.0f));
                            Value* p1 = c.b.build_select(
                                c.b.build_and(has1, s_valid[ntu][2u + eu]),
                                kb.ex2_approx(c.b.build_sub(s_elem[ntu][2u + eu], mn1)), c.f32(0.0f));
                            s_elem[ntu][eu] = p0;
                            s_elem[ntu][2u + eu] = p1;
                            ps0 = c.b.build_add(ps0, p0);
                            ps1 = c.b.build_add(ps1, p1);
                        }
                    }

                    std::vector<Value*> nxt(st.size());
                    nxt[0] = mn0;
                    nxt[1] = mn1;
                    nxt[2] = c.b.build_fma_f32(l0_old, a0, ps0);
                    nxt[3] = c.b.build_fma_f32(l1_old, a1, ps1);

                    // Rescale O accumulators in place.
                    Value* rescale01 = c.b.build_vzero(Type::f32x4());
                    rescale01 = c.b.build_vinsert_lane(rescale01, a0, 0);
                    rescale01 = c.b.build_vinsert_lane(rescale01, a0, 1);
                    rescale01 = c.b.build_vinsert_lane(rescale01, a1, 2);
                    rescale01 = c.b.build_vinsert_lane(rescale01, a1, 3);
                    for (int dt_i = 0; dt_i < 2 * KT; ++dt_i) {
                        const auto idx = static_cast<std::size_t>(4 + dt_i);
                        nxt[idx] = kb.vmul(st[idx], rescale01);
                    }

                    // O += P * V directly from register-packed P.
                    for (int kc = 0; kc < KC_PV; ++kc) {
                        const auto k2 = static_cast<std::size_t>(2 * kc);
                        Value* a_p = c.b.build_vzero(Type::i32x4());
                        a_p = c.b.build_vinsert_lane(a_p, pack_pair_i32(c, spec.dt, s_elem[k2][0], s_elem[k2][1]), 0);
                        a_p = c.b.build_vinsert_lane(a_p, pack_pair_i32(c, spec.dt, s_elem[k2][2], s_elem[k2][3]), 1);
                        a_p = c.b.build_vinsert_lane(a_p, pack_pair_i32(c, spec.dt, s_elem[k2 + 1u][0], s_elem[k2 + 1u][1]), 2);
                        a_p = c.b.build_vinsert_lane(a_p, pack_pair_i32(c, spec.dt, s_elem[k2 + 1u][2], s_elem[k2 + 1u][3]), 3);

                        Value* row_v = c.b.build_add(c.i32(kc * 16), v_lrow0);
                        for (int dp = 0; dp < KT; ++dp) {
                            Value* col_v = c.b.build_add(c.i32(dp * 16), v_lcol0);
                            Value* v_addr = c.b.build_add(
                                cur_v, c.b.build_shl(c.b.build_add(c.b.build_mul(row_v, c.i32(LDS)), col_v), c.i32(1)));
                            Value* b4 = c.b.build_call("ptx_ldmatrix_x4_trans", Type::i32x4(), {v_addr});
                            const auto o0 = static_cast<std::size_t>(4 + 2 * dp);
                            const auto o1 = o0 + 1u;
                            nxt[o0] = mma_sync_f32(c, spec.dt, a_p,
                                                   c.b.build_vextract_lane(b4, 0),
                                                   c.b.build_vextract_lane(b4, 1), nxt[o0]);
                            nxt[o1] = mma_sync_f32(c, spec.dt, a_p,
                                                   c.b.build_vextract_lane(b4, 2),
                                                   c.b.build_vextract_lane(b4, 3), nxt[o1]);
                        }
                    }
                    kb.sync();
                    return nxt;
                });
        },
        [&]() -> std::vector<Value*> { return loop_inits; });

    Value* m0_f = res_vec[0];
    Value* m1_f = res_vec[1];
    Value* l0_f = res_vec[2];
    Value* l1_f = res_vec[3];
    for (int off = 1; off <= 2; off <<= 1) {
        l0_f = c.b.build_add(l0_f, kb.shfl_bfly_f32(l0_f, static_cast<std::uint32_t>(off)));
        l1_f = c.b.build_add(l1_f, kb.shfl_bfly_f32(l1_f, static_cast<std::uint32_t>(off)));
    }

    Value* t2 = c.b.build_shl(t, c.i32(1));
    if (spec.is_split) {
        Value* prow0 = c.b.build_add(c.b.build_mul(bz, Lq), r0);
        Value* prow1 = c.b.build_add(c.b.build_mul(bz, Lq), r1);
        Value* out0_base = c.b.build_add(c.b.build_mul(prow0, D), head_off);
        Value* out1_base = c.b.build_add(c.b.build_mul(prow1, D), head_off);

        for (int dt_i = 0; dt_i < 2 * KT; ++dt_i) {
            Value* col = c.b.build_add(c.i32(dt_i * 8), t2);
            Value* c_ok = c.b.build_slt(col, c.i32(HD));
            Value* ov = res_vec[static_cast<std::size_t>(4 + dt_i)];
            kb.if_then(c.b.build_and(c_ok, c.b.build_slt(r0, Lq)), [&] {
                Value* o_idx = c.b.build_add(out0_base, col);
                kb.store_f32(ptr_idx_scaled(c, part_o, o_idx, 4), c.b.build_vextract_lane(ov, 0));
                kb.store_f32(ptr_idx_scaled(c, part_o, c.b.build_add(o_idx, c.i32(1)), 4), c.b.build_vextract_lane(ov, 1));
            });
            kb.if_then(c.b.build_and(c_ok, c.b.build_slt(r1, Lq)), [&] {
                Value* o_idx = c.b.build_add(out1_base, col);
                kb.store_f32(ptr_idx_scaled(c, part_o, o_idx, 4), c.b.build_vextract_lane(ov, 2));
                kb.store_f32(ptr_idx_scaled(c, part_o, c.b.build_add(o_idx, c.i32(1)), 4), c.b.build_vextract_lane(ov, 3));
            });
        }
        kb.if_then(c.b.build_eq(t, c.i32(0)), [&] {
            kb.if_then(c.b.build_slt(r0, Lq), [&] {
                Value* ml0 = c.b.build_shl(c.b.build_add(c.b.build_mul(prow0, heads), head), c.i32(1));
                kb.store_f32(ptr_idx_scaled(c, part_ml, ml0, 4), m0_f);
                kb.store_f32(ptr_idx_scaled(c, part_ml, c.b.build_add(ml0, c.i32(1)), 4), l0_f);
            });
            kb.if_then(c.b.build_slt(r1, Lq), [&] {
                Value* ml1 = c.b.build_shl(c.b.build_add(c.b.build_mul(prow1, heads), head), c.i32(1));
                kb.store_f32(ptr_idx_scaled(c, part_ml, ml1, 4), m1_f);
                kb.store_f32(ptr_idx_scaled(c, part_ml, c.b.build_add(ml1, c.i32(1)), 4), l1_f);
            });
        });
    } else {
        Value* inv0 = c.b.build_select(c.b.build_sgt(l0_f, c.f32(0.0f)), kb.div_approx(c.f32(1.0f), l0_f), c.f32(0.0f));
        Value* inv1 = c.b.build_select(c.b.build_sgt(l1_f, c.f32(0.0f)), kb.div_approx(c.f32(1.0f), l1_f), c.f32(0.0f));
        Value* r0_ok = c.b.build_slt(r0, Lq);
        Value* r1_ok = c.b.build_slt(r1, Lq);
        Value* o0_base = c.b.build_add(c.b.build_mul(r0, D), head_off);
        Value* o1_base = c.b.build_add(c.b.build_mul(r1, D), head_off);

        for (int dt_i = 0; dt_i < 2 * KT; ++dt_i) {
            Value* col = c.b.build_add(c.i32(dt_i * 8), t2);
            Value* c_ok = c.b.build_slt(col, c.i32(HD));
            Value* ov = res_vec[static_cast<std::size_t>(4 + dt_i)];
            Value* p0 = pack_pair_i32(
                c, spec.dt,
                c.b.build_mul(c.b.build_vextract_lane(ov, 0), inv0),
                c.b.build_mul(c.b.build_vextract_lane(ov, 1), inv0));
            Value* p1 = pack_pair_i32(
                c, spec.dt,
                c.b.build_mul(c.b.build_vextract_lane(ov, 2), inv1),
                c.b.build_mul(c.b.build_vextract_lane(ov, 3), inv1));
            kb.if_then(c.b.build_and(c_ok, r0_ok), [&] {
                kb.store_i32(ptr_idx_scaled(c, O, c.b.build_add(o0_base, col), 2), p0);
            });
            kb.if_then(c.b.build_and(c_ok, r1_ok), [&] {
                kb.store_i32(ptr_idx_scaled(c, O, c.b.build_add(o1_base, col), 2), p1);
            });
        }
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
