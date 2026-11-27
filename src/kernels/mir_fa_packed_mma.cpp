#include "mir_fa_packed_mma.h"

#include "mir_common.h"

#include <array>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kHd = 64;
constexpr int kBq = 64;
constexpr int kBk = 64;
constexpr int kLds = kHd + 8;
constexpr int kThreads = 128;

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

std::pair<Value*, Value*> row_range(Ctx& c, Value* bounds, Value* r, Value* L, Value* hw) {
    auto res = emit_if_else_values(
        c, c.b.build_slt(r, L), {Type::i32(), Type::i32()},
        [&]() -> std::vector<Value*> {
            Value* r2 = c.b.build_shl(r, c.i32(1));
            Value* s = c.kb.load_i32(ptr_at(c, bounds, r2, 4));
            Value* e = c.kb.load_i32(ptr_at(c, bounds, c.b.build_add(r2, c.i32(1)), 4));
            Value* has_w = c.b.build_sge(hw, c.i32(0));
            Value* lo = c.b.build_select(has_w, smax_i32(c, s, c.b.build_sub(r, hw)), s);
            Value* hi = c.b.build_select(has_w, smin_i32(c, e, c.b.build_add(c.b.build_add(r, hw), c.i32(1))), e);
            return {lo, hi};
        },
        [&]() -> std::vector<Value*> { return {c.i32(0), c.i32(0)}; });
    return {res[0], res[1]};
}

void load_tile_cp_async(Ctx& c, Value* dst_smem_i32, Value* src, Value* ld, Value* col0,
                        Value* row0, Value* row_end) {
    Value* tid = c.kb.tid_x();
    for (int i = 0; i < 4; ++i) {
        Value* chunk = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * kThreads));
        Value* r = c.b.build_lshr(chunk, c.i32(3));
        Value* ch = c.b.build_and(chunk, c.i32(7));
        Value* ch8 = c.b.build_shl(ch, c.i32(3));
        Value* gr = c.b.build_add(row0, r);
        Value* ok = c.b.build_slt(gr, row_end);
        Value* g_idx = c.b.build_add(c.b.build_add(c.b.build_mul(gr, ld), col0), ch8);
        Value* g_ptr = c.b.build_select(ok, ptr_at(c, src, g_idx, 2), src);
        Value* s_off = c.b.build_shl(c.b.build_add(c.b.build_mul(r, c.i32(kLds)), ch8), c.i32(1));
        Value* dst = c.b.build_add(dst_smem_i32, s_off);
        Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
        c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, g_ptr, sz});
    }
}

}  // namespace

brass::Function* build_fa_packed_mma_mir(brass::Module& mod, Dtype dt) {
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("build_fa_packed_mma_mir: only FP16/BF16 supported");
    }
    const std::string name = std::string("mir_fa_packed_hd64_") + (dt == Dtype::FP16 ? "f16" : "bf16");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:qkv 1:bounds 2:out | 3:L 4:H 5:hw | 6:scale_log2
    Value* qkv = p[0];
    Value* bounds = p[1];
    Value* out = p[2];
    Value* L = p[3];
    Value* H = p[4];
    Value* hw = p[5];
    Value* scale_log2 = p[6];

    const int stage_elems = kBk * kLds;
    const int stage_bytes = stage_elems * 2;
    Value* smem = kb.shared_alloc_u16(static_cast<std::uint32_t>(4 * stage_elems));
    Value* sK_base = c.b.build_trunc_i32(smem);
    Value* sV_base = c.b.build_add(sK_base, c.i32(2 * stage_bytes));
    Value* s_lo = kb.shared_alloc_i32(1);
    Value* s_hi = kb.shared_alloc_i32(1);

    Value* D = c.b.build_mul(H, c.i32(kHd));
    Value* ld = c.b.build_mul(D, c.i32(3));
    Value* h = kb.ctaid_y();
    Value* head_off = c.b.build_mul(h, c.i32(kHd));
    Value* q0 = c.b.build_mul(kb.ctaid_x(), c.i32(kBq));
    Value* tid = kb.tid_x();
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t = c.b.build_and(lane, c.i32(3));

    kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
        kb.shared_store_i32(s_lo, c.i32(INT_MAX));
        kb.shared_store_i32(s_hi, c.i32(INT_MIN));
    });
    kb.sync();

    kb.if_then(c.b.build_slt(tid, c.i32(kBq)), [&] {
        auto [lo, hi] = row_range(c, bounds, c.b.build_add(q0, tid), L, hw);
        kb.if_then(c.b.build_slt(lo, hi), [&] {
            kb.atom_shared_min_i32(s_lo, lo);
            kb.atom_shared_max_i32(s_hi, hi);
        });
    });

    Value* r0 = c.b.build_add(c.b.build_add(q0, c.b.build_shl(warp, c.i32(4))), g);
    Value* r1 = c.b.build_add(r0, c.i32(8));
    auto [lo0, hi0] = row_range(c, bounds, r0, L, hw);
    auto [lo1, hi1] = row_range(c, bounds, r1, L, hw);

    Value* wlo0 = c.b.build_select(c.b.build_slt(lo0, hi0), lo0, c.i32(INT_MAX));
    Value* wlo1 = c.b.build_select(c.b.build_slt(lo1, hi1), lo1, c.i32(INT_MAX));
    Value* whi0 = c.b.build_select(c.b.build_slt(lo0, hi0), hi0, c.i32(INT_MIN));
    Value* whi1 = c.b.build_select(c.b.build_slt(lo1, hi1), hi1, c.i32(INT_MIN));
    Value* wlo = smin_i32(c, wlo0, wlo1);
    Value* whi = smax_i32(c, whi0, whi1);
    for (int off = 16; off > 0; off >>= 1) {
        wlo = smin_i32(c, wlo, kb.shfl_bfly_i32(wlo, static_cast<std::uint32_t>(off)));
        whi = smax_i32(c, whi, kb.shfl_bfly_i32(whi, static_cast<std::uint32_t>(off)));
    }

    // Stage Q into sK_base alias and read Q_frag via ldmatrix.
    load_tile_cp_async(c, sK_base, qkv, ld, head_off, q0, L);
    c.b.build_call("ptx_cp_async_commit", Type::void_type());
    c.b.build_call("ptx_cp_async_wait_all", Type::void_type());
    kb.sync();

    Value* kmin = kb.shared_load_i32(s_lo);
    Value* kmax = kb.shared_load_i32(s_hi);
    kb.if_then(c.b.build_slt(kmin, kmax), [&] {
        std::array<Value*, 4> qf{};
        Value* q_lrow = c.b.build_add(c.b.build_shl(warp, c.i32(4)), c.b.build_and(lane, c.i32(15)));
        Value* q_lcol0 = c.b.build_shl(c.b.build_lshr(lane, c.i32(4)), c.i32(3));
        for (int kc = 0; kc < 4; ++kc) {
            Value* col = c.b.build_add(c.i32(kc * 16), q_lcol0);
            Value* q_addr = c.b.build_add(
                sK_base, c.b.build_shl(c.b.build_add(c.b.build_mul(q_lrow, c.i32(kLds)), col), c.i32(1)));
            qf[static_cast<std::size_t>(kc)] = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {q_addr});
        }
        kb.sync();

        Value* k_col0 = c.b.build_add(D, head_off);
        Value* v_col0 = c.b.build_add(c.b.build_shl(D, c.i32(1)), head_off);
        Value* ntiles = c.b.build_udiv(c.b.build_sub(c.b.build_add(c.b.build_sub(kmax, kmin), c.i32(kBk - 1)),
                                                     c.i32(0)),
                                       c.i32(kBk));

        load_tile_cp_async(c, sK_base, qkv, ld, k_col0, kmin, kmax);
        load_tile_cp_async(c, sV_base, qkv, ld, v_col0, kmin, kmax);
        c.b.build_call("ptx_cp_async_commit", Type::void_type());

        Value* mi = c.b.build_lshr(lane, c.i32(3));
        Value* k_lrow0 = c.b.build_add(c.b.build_shl(c.b.build_lshr(mi, c.i32(1)), c.i32(3)),
                                       c.b.build_and(lane, c.i32(7)));
        Value* k_lcol0 = c.b.build_shl(c.b.build_and(mi, c.i32(1)), c.i32(3));
        Value* v_lrow0 = c.b.build_add(c.b.build_shl(c.b.build_and(mi, c.i32(1)), c.i32(3)),
                                       c.b.build_and(lane, c.i32(7)));
        Value* v_lcol0 = c.b.build_shl(c.b.build_lshr(mi, c.i32(1)), c.i32(3));

        std::vector<Type> loop_types(12, Type::f32());
        for (int i = 0; i < 8; ++i) loop_types[static_cast<std::size_t>(4 + i)] = Type::f32x4();
        std::vector<Value*> inits(12);
        inits[0] = c.f32(-1e30f);
        inits[1] = c.f32(-1e30f);
        inits[2] = c.f32(0.0f);
        inits[3] = c.f32(0.0f);
        for (int i = 0; i < 8; ++i) inits[static_cast<std::size_t>(4 + i)] = c.b.build_vzero(Type::f32x4());

        auto out_st = kb.for_range_reduce_n(
            c.i32(0), ntiles, c.i32(1), inits,
            [&](Value* j, const std::vector<Value*>& st) -> std::vector<Value*> {
                Value* buf = c.b.build_and(j, c.i32(1));
                Value* cur_k = c.b.build_add(sK_base, c.b.build_mul(buf, c.i32(stage_bytes)));
                Value* cur_v = c.b.build_add(sV_base, c.b.build_mul(buf, c.i32(stage_bytes)));

                Value* has_next = c.b.build_slt(c.b.build_add(j, c.i32(1)), ntiles);
                emit_if_else_values(
                    c, has_next, {},
                    [&]() -> std::vector<Value*> {
                        Value* nb = c.b.build_add(kmin, c.b.build_mul(c.b.build_add(j, c.i32(1)), c.i32(kBk)));
                        Value* nxt_buf = c.b.build_xor(buf, c.i32(1));
                        Value* nxt_k = c.b.build_add(sK_base, c.b.build_mul(nxt_buf, c.i32(stage_bytes)));
                        Value* nxt_v = c.b.build_add(sV_base, c.b.build_mul(nxt_buf, c.i32(stage_bytes)));
                        load_tile_cp_async(c, nxt_k, qkv, ld, k_col0, nb, kmax);
                        load_tile_cp_async(c, nxt_v, qkv, ld, v_col0, nb, kmax);
                        c.b.build_call("ptx_cp_async_commit", Type::void_type());
                        c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(1)});
                        return {};
                    },
                    [&]() -> std::vector<Value*> {
                        c.b.build_call("ptx_cp_async_wait_all", Type::void_type());
                        return {};
                    });
                kb.sync();

                Value* kb_row = c.b.build_add(kmin, c.b.build_mul(j, c.i32(kBk)));
                Value* warp_active = c.b.build_and(
                    c.b.build_slt(kb_row, whi),
                    c.b.build_sgt(c.b.build_add(kb_row, c.i32(kBk)), wlo));

                auto nxt = emit_if_else_values(
                    c, warp_active, loop_types,
                    [&]() -> std::vector<Value*> {
                        std::array<Value*, 8> s_acc{};
                        s_acc.fill(c.b.build_vzero(Type::f32x4()));
                        for (int kc = 0; kc < 4; ++kc) {
                            Value* qv = qf[static_cast<std::size_t>(kc)];
                            Value* col_k = c.b.build_add(c.i32(kc * 16), k_lcol0);
                            for (int np = 0; np < 4; ++np) {
                                Value* row_k = c.b.build_add(c.i32(np * 16), k_lrow0);
                                Value* k_addr = c.b.build_add(
                                    cur_k, c.b.build_shl(c.b.build_add(c.b.build_mul(row_k, c.i32(kLds)), col_k), c.i32(1)));
                                Value* b4 = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {k_addr});
                                const auto s0 = static_cast<std::size_t>(2 * np);
                                const auto s1 = s0 + 1u;
                                s_acc[s0] = mma_sync_f32(c, dt, qv,
                                                         c.b.build_vextract_lane(b4, 0),
                                                         c.b.build_vextract_lane(b4, 1), s_acc[s0]);
                                s_acc[s1] = mma_sync_f32(c, dt, qv,
                                                         c.b.build_vextract_lane(b4, 2),
                                                         c.b.build_vextract_lane(b4, 3), s_acc[s1]);
                            }
                        }

                        Value* mx0 = c.f32(-1e30f);
                        Value* mx1 = c.f32(-1e30f);
                        std::array<std::array<Value*, 4>, 8> sv{};
                        std::array<std::array<Value*, 4>, 8> sok{};
                        Value* t2 = c.b.build_shl(t, c.i32(1));
                        for (int nt = 0; nt < 8; ++nt) {
                            const auto ntu = static_cast<std::size_t>(nt);
                            Value* key0 = c.b.build_add(kb_row, c.b.build_add(c.i32(nt * 8), t2));
                            for (int e = 0; e < 2; ++e) {
                                Value* kk = (e == 0) ? key0 : c.b.build_add(key0, c.i32(1));
                                Value* ok0 = c.b.build_and(c.b.build_sge(kk, lo0), c.b.build_slt(kk, hi0));
                                Value* ok1 = c.b.build_and(c.b.build_sge(kk, lo1), c.b.build_slt(kk, hi1));
                                Value* v0 = c.b.build_mul(c.b.build_vextract_lane(s_acc[ntu], static_cast<std::uint32_t>(e)), scale_log2);
                                Value* v1 = c.b.build_mul(c.b.build_vextract_lane(s_acc[ntu], static_cast<std::uint32_t>(2 + e)), scale_log2);
                                sv[ntu][static_cast<std::size_t>(e)] = c.b.build_select(ok0, v0, c.f32(-1e30f));
                                sv[ntu][static_cast<std::size_t>(2 + e)] = c.b.build_select(ok1, v1, c.f32(-1e30f));
                                sok[ntu][static_cast<std::size_t>(e)] = ok0;
                                sok[ntu][static_cast<std::size_t>(2 + e)] = ok1;
                                mx0 = kb.fmax(mx0, sv[ntu][static_cast<std::size_t>(e)]);
                                mx1 = kb.fmax(mx1, sv[ntu][static_cast<std::size_t>(2 + e)]);
                            }
                        }
                        for (int off = 1; off <= 2; off <<= 1) {
                            mx0 = kb.fmax(mx0, kb.shfl_bfly_f32(mx0, static_cast<std::uint32_t>(off)));
                            mx1 = kb.fmax(mx1, kb.shfl_bfly_f32(mx1, static_cast<std::uint32_t>(off)));
                        }
                        Value* has0 = c.b.build_sgt(mx0, c.f32(-1e29f));
                        Value* has1 = c.b.build_sgt(mx1, c.f32(-1e29f));
                        Value* mn0 = c.b.build_select(has0, kb.fmax(st[0], mx0), st[0]);
                        Value* mn1 = c.b.build_select(has1, kb.fmax(st[1], mx1), st[1]);
                        Value* prev0 = c.b.build_sgt(st[0], c.f32(-1e29f));
                        Value* prev1 = c.b.build_sgt(st[1], c.f32(-1e29f));
                        Value* a0 = c.b.build_select(
                            c.b.build_and(prev0, has0), kb.ex2_approx(c.b.build_sub(st[0], mn0)), c.f32(1.0f));
                        Value* a1 = c.b.build_select(
                            c.b.build_and(prev1, has1), kb.ex2_approx(c.b.build_sub(st[1], mn1)), c.f32(1.0f));

                        Value* ps0 = c.f32(0.0f);
                        Value* ps1 = c.f32(0.0f);
                        for (int nt = 0; nt < 8; ++nt) {
                            const auto ntu = static_cast<std::size_t>(nt);
                            for (int e = 0; e < 2; ++e) {
                                const auto eu = static_cast<std::size_t>(e);
                                Value* p0 = c.b.build_select(
                                    c.b.build_and(has0, sok[ntu][eu]),
                                    kb.ex2_approx(c.b.build_sub(sv[ntu][eu], mn0)), c.f32(0.0f));
                                Value* p1 = c.b.build_select(
                                    c.b.build_and(has1, sok[ntu][2u + eu]),
                                    kb.ex2_approx(c.b.build_sub(sv[ntu][2u + eu], mn1)), c.f32(0.0f));
                                sv[ntu][eu] = p0;
                                sv[ntu][2u + eu] = p1;
                                ps0 = c.b.build_add(ps0, p0);
                                ps1 = c.b.build_add(ps1, p1);
                            }
                        }

                        std::vector<Value*> nst(12);
                        nst[0] = mn0;
                        nst[1] = mn1;
                        nst[2] = c.b.build_fma_f32(st[2], a0, ps0);
                        nst[3] = c.b.build_fma_f32(st[3], a1, ps1);
                        Value* rscale = c.b.build_vzero(Type::f32x4());
                        rscale = c.b.build_vinsert_lane(rscale, a0, 0);
                        rscale = c.b.build_vinsert_lane(rscale, a0, 1);
                        rscale = c.b.build_vinsert_lane(rscale, a1, 2);
                        rscale = c.b.build_vinsert_lane(rscale, a1, 3);
                        for (int i = 0; i < 8; ++i) nst[static_cast<std::size_t>(4 + i)] = kb.vmul(st[static_cast<std::size_t>(4 + i)], rscale);

                        for (int kc = 0; kc < 4; ++kc) {
                            const auto k2 = static_cast<std::size_t>(2 * kc);
                            Value* ap = c.b.build_vzero(Type::i32x4());
                            ap = c.b.build_vinsert_lane(ap, pack_pair_i32(c, dt, sv[k2][0], sv[k2][1]), 0);
                            ap = c.b.build_vinsert_lane(ap, pack_pair_i32(c, dt, sv[k2][2], sv[k2][3]), 1);
                            ap = c.b.build_vinsert_lane(ap, pack_pair_i32(c, dt, sv[k2 + 1u][0], sv[k2 + 1u][1]), 2);
                            ap = c.b.build_vinsert_lane(ap, pack_pair_i32(c, dt, sv[k2 + 1u][2], sv[k2 + 1u][3]), 3);
                            Value* row_v = c.b.build_add(c.i32(kc * 16), v_lrow0);
                            for (int dp = 0; dp < 4; ++dp) {
                                Value* col_v = c.b.build_add(c.i32(dp * 16), v_lcol0);
                                Value* v_addr = c.b.build_add(
                                    cur_v, c.b.build_shl(c.b.build_add(c.b.build_mul(row_v, c.i32(kLds)), col_v), c.i32(1)));
                                Value* b4 = c.b.build_call("ptx_ldmatrix_x4_trans", Type::i32x4(), {v_addr});
                                const auto o0 = static_cast<std::size_t>(4 + 2 * dp);
                                const auto o1 = o0 + 1u;
                                nst[o0] = mma_sync_f32(c, dt, ap,
                                                       c.b.build_vextract_lane(b4, 0),
                                                       c.b.build_vextract_lane(b4, 1), nst[o0]);
                                nst[o1] = mma_sync_f32(c, dt, ap,
                                                       c.b.build_vextract_lane(b4, 2),
                                                       c.b.build_vextract_lane(b4, 3), nst[o1]);
                            }
                        }
                        return nst;
                    },
                    [&]() -> std::vector<Value*> { return st; });

                kb.sync();
                return nxt;
            });

        Value* l0 = out_st[2];
        Value* l1 = out_st[3];
        for (int off = 1; off <= 2; off <<= 1) {
            l0 = c.b.build_add(l0, kb.shfl_bfly_f32(l0, static_cast<std::uint32_t>(off)));
            l1 = c.b.build_add(l1, kb.shfl_bfly_f32(l1, static_cast<std::uint32_t>(off)));
        }
        Value* inv0 = c.b.build_select(c.b.build_sgt(l0, c.f32(0.0f)), kb.div_approx(c.f32(1.0f), l0), c.f32(0.0f));
        Value* inv1 = c.b.build_select(c.b.build_sgt(l1, c.f32(0.0f)), kb.div_approx(c.f32(1.0f), l1), c.f32(0.0f));
        Value* r0_ok = c.b.build_slt(r0, L);
        Value* r1_ok = c.b.build_slt(r1, L);
        Value* t2 = c.b.build_shl(t, c.i32(1));
        Value* o0_base = c.b.build_add(c.b.build_mul(r0, D), head_off);
        Value* o1_base = c.b.build_add(c.b.build_mul(r1, D), head_off);
        for (int dt_i = 0; dt_i < 8; ++dt_i) {
            Value* col = c.b.build_add(c.i32(dt_i * 8), t2);
            Value* ov = out_st[static_cast<std::size_t>(4 + dt_i)];
            Value* p0 = pack_pair_i32(
                c, dt,
                c.b.build_mul(c.b.build_vextract_lane(ov, 0), inv0),
                c.b.build_mul(c.b.build_vextract_lane(ov, 1), inv0));
            Value* p1 = pack_pair_i32(
                c, dt,
                c.b.build_mul(c.b.build_vextract_lane(ov, 2), inv1),
                c.b.build_mul(c.b.build_vextract_lane(ov, 3), inv1));
            kb.if_then(r0_ok, [&] { kb.store_i32(ptr_at(c, out, c.b.build_add(o0_base, col), 2), p0); });
            kb.if_then(r1_ok, [&] { kb.store_i32(ptr_at(c, out, c.b.build_add(o1_base, col), 2), p1); });
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
