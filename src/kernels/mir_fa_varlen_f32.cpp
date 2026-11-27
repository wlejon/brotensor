#include "mir_fa_varlen_f32.h"

#include "mir_common.h"

#include <array>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kBQ = 64;
constexpr int kBK = 64;
constexpr int kHD = 64;
constexpr int kKS = kHD + 4;
constexpr int kThreads = 256;

using jit::gpu_mir::bits_to_f32;
using jit::gpu_mir::f32_to_bits;

Value* f32x4_to_i32x4(Ctx& c, Value* v) {
    Value* r = c.b.build_vzero(Type::i32x4());
    for (std::uint32_t j = 0; j < 4u; ++j) {
        r = c.b.build_vinsert_lane(r, f32_to_bits(c, c.b.build_vextract_lane(v, j)), j);
    }
    return r;
}

Value* i32x4_to_f32x4(Ctx& c, Value* v) {
    Value* r = c.b.build_vzero(Type::f32x4());
    for (std::uint32_t j = 0; j < 4u; ++j) {
        r = c.b.build_vinsert_lane(r, bits_to_f32(c, c.b.build_vextract_lane(v, j)), j);
    }
    return r;
}

Value* load_f32x4_masked(Ctx& c, Value* cond, Value* ptr, Value* safe_ptr, Value* scale_opt = nullptr) {
    Value* p = c.b.build_select(cond, ptr, safe_ptr);
    Value* v = c.kb.vload_f32x4(p);
    Value* out = c.b.build_vzero(Type::f32x4());
    for (std::uint32_t j = 0; j < 4u; ++j) {
        Value* elem = c.b.build_vextract_lane(v, j);
        if (scale_opt) elem = c.b.build_mul(elem, scale_opt);
        out = c.b.build_vinsert_lane(out, c.b.build_select(cond, elem, c.f32(0.0f)), j);
    }
    return out;
}

Value* load_i32x4_masked(Ctx& c, Value* cond, Value* ptr, Value* safe_ptr) {
    Value* p = c.b.build_select(cond, ptr, safe_ptr);
    Value* v = c.kb.vload(Type::i32x4(), p);
    Value* out = c.b.build_vzero(Type::i32x4());
    for (std::uint32_t j = 0; j < 4u; ++j) {
        Value* elem = c.b.build_vextract_lane(v, j);
        out = c.b.build_vinsert_lane(out, c.b.build_select(cond, elem, c.i32(0)), j);
    }
    return out;
}

Value* fma4_f32x4(Ctx& c, Value* a, Value* b, Value* acc) {
    Value* p = c.kb.vmul(a, b);
    for (std::uint32_t j = 0; j < 4u; ++j) {
        acc = c.b.build_add(acc, c.b.build_vextract_lane(p, j));
    }
    return acc;
}

}  // namespace

brass::Function* build_fa_varlen_f32_mir(brass::Module& mod) {
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function("mir_fa_varlen_f32_hd64", Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:Q 1:K 2:V 3:cu_q 4:cu_k 5:O | 6:total_q 7:B 8:D | 9:q_scale
    Value* Q = p[0];
    Value* K = p[1];
    Value* V = p[2];
    Value* cu_q = p[3];
    Value* cu_k = p[4];
    Value* O = p[5];
    Value* total_q = p[6];
    Value* B = p[7];
    Value* D = p[8];
    Value* q_scale = p[9];

    Value* smem_dyn = c.b.build_call("ptx_shared_dynamic", Type::ptr());
    Value* Qs = smem_dyn;
    Value* Ks = ptr_at(c, Qs, c.i32(kBQ * kHD), 4);
    Value* Vs = ptr_at(c, Ks, c.i32(kBK * kKS), 4);
    Value* Ps = Ks;
    Value* s_kb = kb.shared_alloc_i32(kBQ);
    Value* s_ke = kb.shared_alloc_i32(kBQ);

    Value* tid = kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* tx = c.b.build_and(lane, c.i32(15));
    Value* ty = c.b.build_add(c.b.build_shl(c.b.build_lshr(tid, c.i32(5)), c.i32(1)),
                              c.b.build_lshr(lane, c.i32(4)));
    Value* q0 = c.b.build_mul(kb.ctaid_x(), c.i32(kBQ));
    Value* hoff = c.b.build_mul(kb.ctaid_y(), c.i32(kHD));

    kb.if_then(c.b.build_slt(tid, c.i32(kBQ)), [&] {
        Value* r = c.b.build_add(q0, tid);
        auto kbe = emit_if_else_values(
            c, c.b.build_slt(r, total_q), {Type::i32(), Type::i32()},
            [&]() -> std::vector<Value*> {
                auto bs = kb.for_range_reduce_n(
                    c.i32(0), c.i32(24), c.i32(1), {c.i32(0), c.b.build_sub(B, c.i32(1))},
                    [&](Value*, const std::vector<Value*>& st) -> std::vector<Value*> {
                        Value* lo = st[0];
                        Value* hi = st[1];
                        Value* lt = c.b.build_slt(lo, hi);
                        Value* mid = c.b.build_ashr(c.b.build_add(c.b.build_add(lo, hi), c.i32(1)), c.i32(1));
                        Value* safe_mid = c.b.build_select(lt, mid, lo);
                        Value* le = c.b.build_sle(kb.load_i32(ptr_at(c, cu_q, safe_mid, 4)), r);
                        Value* nlo = c.b.build_select(c.b.build_and(lt, le), mid, lo);
                        Value* nhi = c.b.build_select(c.b.build_and(lt, c.b.build_not(le)),
                                                      c.b.build_sub(mid, c.i32(1)), hi);
                        return {nlo, nhi};
                    });
                Value* lo = bs[0];
                return {
                    kb.load_i32(ptr_at(c, cu_k, lo, 4)),
                    kb.load_i32(ptr_at(c, cu_k, c.b.build_add(lo, c.i32(1)), 4)),
                };
            },
            [&]() -> std::vector<Value*> { return {c.i32(0), c.i32(0)}; });
        kb.shared_store_i32_indexed(s_kb, tid, kbe[0]);
        kb.shared_store_i32_indexed(s_ke, tid, kbe[1]);
    });

    for (int step = 0; step < (kBQ * (kHD / 4)) / kThreads; ++step) {
        Value* idx = c.b.build_add(tid, c.i32(step * kThreads));
        Value* r = c.b.build_lshr(idx, c.i32(4));
        Value* d4 = c.b.build_and(idx, c.i32(15));
        Value* gr = c.b.build_add(q0, r);
        Value* ok = c.b.build_slt(gr, total_q);
        Value* g_off = c.b.build_add(
            c.b.build_add(c.b.build_mul(gr, D), hoff),
            c.b.build_shl(d4, c.i32(2)));
        Value* qv = load_f32x4_masked(c, ok, ptr_at(c, Q, g_off, 4), Q, q_scale);
        kb.shared_store_i32x4_indexed(Qs, idx, f32x4_to_i32x4(c, qv));
    }
    kb.sync();

    Value* last = c.b.build_sub(smin_i32(c, c.i32(kBQ), c.b.build_sub(total_q, q0)), c.i32(1));
    Value* kmin = kb.shared_load_i32_indexed(s_kb, c.i32(0));
    Value* kmax = kb.shared_load_i32_indexed(s_ke, smax_i32(c, last, c.i32(0)));

    std::array<Value*, 4> kb_r{}, ke_r{};
    std::vector<Value*> init_st;
    init_st.reserve(24);
    for (int i = 0; i < 4; ++i) {
        Value* ri = c.b.build_add(c.b.build_shl(ty, c.i32(2)), c.i32(i));
        kb_r[static_cast<std::size_t>(i)] = kb.shared_load_i32_indexed(s_kb, ri);
        ke_r[static_cast<std::size_t>(i)] = kb.shared_load_i32_indexed(s_ke, ri);
        init_st.push_back(c.f32(-1e30f));  // m[i]
        init_st.push_back(c.f32(0.0f));    // l[i]
    }
    for (int ij = 0; ij < 16; ++ij) init_st.push_back(c.f32(0.0f));

    auto final_st = kb.for_range_reduce_n(
        kmin, kmax, c.i32(kBK), init_st,
        [&](Value* k0, const std::vector<Value*>& cur) -> std::vector<Value*> {
            kb.sync();
            for (int step = 0; step < (kBK * (kHD / 4)) / kThreads; ++step) {
                Value* idx = c.b.build_add(tid, c.i32(step * kThreads));
                Value* row = c.b.build_lshr(idx, c.i32(4));
                Value* d4 = c.b.build_and(idx, c.i32(15));
                Value* g = c.b.build_add(k0, row);
                Value* ok = c.b.build_slt(g, kmax);
                Value* g_off = c.b.build_add(
                    c.b.build_add(c.b.build_mul(g, D), hoff),
                    c.b.build_shl(d4, c.i32(2)));
                Value* kvec = load_i32x4_masked(c, ok, ptr_at(c, K, g_off, 4), K);
                Value* vvec = load_i32x4_masked(c, ok, ptr_at(c, V, g_off, 4), V);
                Value* ks_v4_idx = c.b.build_add(c.b.build_mul(row, c.i32(kKS / 4)), d4);
                kb.shared_store_i32x4_indexed(Ks, ks_v4_idx, kvec);
                kb.shared_store_i32x4_indexed(Vs, idx, vvec);
            }
            kb.sync();

            std::array<std::array<Value*, 4>, 4> s{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = c.f32(0.0f);

            for (int d4 = 0; d4 < kHD / 4; ++d4) {
                std::array<Value*, 4> qa{}, kv{};
                for (int i = 0; i < 4; ++i) {
                    Value* q_row = c.b.build_add(c.b.build_shl(ty, c.i32(2)), c.i32(i));
                    Value* q_v4_idx = c.b.build_add(c.b.build_mul(q_row, c.i32(kHD / 4)), c.i32(d4));
                    qa[static_cast<std::size_t>(i)] =
                        i32x4_to_f32x4(c, kb.shared_load_i32x4_indexed(Qs, q_v4_idx));
                }
                for (int j = 0; j < 4; ++j) {
                    Value* k_row = c.b.build_add(tx, c.i32(16 * j));
                    Value* k_v4_idx = c.b.build_add(c.b.build_mul(k_row, c.i32(kKS / 4)), c.i32(d4));
                    kv[static_cast<std::size_t>(j)] =
                        i32x4_to_f32x4(c, kb.shared_load_i32x4_indexed(Ks, k_v4_idx));
                }
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                            fma4_f32x4(c, qa[static_cast<std::size_t>(i)], kv[static_cast<std::size_t>(j)],
                                       s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }

            std::vector<Value*> nxt = cur;
            for (int i = 0; i < 4; ++i) {
                const auto iu = static_cast<std::size_t>(i);
                Value* mx = c.f32(-1e30f);
                for (int j = 0; j < 4; ++j) {
                    const auto ju = static_cast<std::size_t>(j);
                    Value* g = c.b.build_add(k0, c.b.build_add(tx, c.i32(16 * j)));
                    Value* ok = c.b.build_and(c.b.build_sge(g, kb_r[iu]), c.b.build_slt(g, ke_r[iu]));
                    Value* sv = c.b.build_select(ok, s[iu][ju], c.f32(-1e30f));
                    s[iu][ju] = sv;
                    mx = kb.fmax(mx, sv);
                }
                for (int off = 1; off < 16; off <<= 1) {
                    mx = kb.fmax(mx, kb.shfl_bfly_f32(mx, static_cast<std::uint32_t>(off)));
                }
                Value* has_tile = c.b.build_sgt(mx, c.f32(-1e29f));
                Value* m_old = nxt[iu * 2];
                Value* m_new = c.b.build_select(has_tile, kb.fmax(m_old, mx), m_old);
                Value* had_prev = c.b.build_sgt(m_old, c.f32(-1e29f));
                Value* alpha = c.b.build_select(
                    c.b.build_and(had_prev, has_tile),
                    kb.ex2_approx(c.b.build_sub(m_old, m_new)),
                    c.f32(1.0f));
                Value* sum = c.f32(0.0f);
                for (int j = 0; j < 4; ++j) {
                    const auto ju = static_cast<std::size_t>(j);
                    Value* pv = kb.ex2_approx(c.b.build_sub(s[iu][ju], m_new));
                    Value* p_ok = c.b.build_and(has_tile, c.b.build_sgt(s[iu][ju], c.f32(-1e29f)));
                    Value* p_safe = c.b.build_select(p_ok, pv, c.f32(0.0f));
                    s[iu][ju] = p_safe;
                    sum = c.b.build_add(sum, p_safe);
                }
                for (int off = 1; off < 16; off <<= 1) {
                    sum = c.b.build_add(sum, kb.shfl_bfly_f32(sum, static_cast<std::uint32_t>(off)));
                }
                nxt[iu * 2] = m_new;
                nxt[iu * 2 + 1] = c.b.build_fma_f32(nxt[iu * 2 + 1], alpha, sum);
                for (int j = 0; j < 4; ++j) {
                    const std::size_t o_idx = 8u + iu * 4u + static_cast<std::size_t>(j);
                    nxt[o_idx] = c.b.build_mul(nxt[o_idx], alpha);
                }
            }

            kb.sync();
            Value* q_row0 = c.b.build_shl(ty, c.i32(2));
            for (int i = 0; i < 4; ++i) {
                Value* prow = c.b.build_add(q_row0, c.i32(i));
                for (int j = 0; j < 4; ++j) {
                    Value* pidx = c.b.build_add(c.b.build_mul(prow, c.i32(kBK)),
                                                c.b.build_add(tx, c.i32(16 * j)));
                    kb.shared_store_f32_indexed(Ps, pidx, s[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
                }
            }
            kb.sync();

            for (int k4 = 0; k4 < kBK / 4; ++k4) {
                std::array<Value*, 4> pa{}, vb{};
                for (int i = 0; i < 4; ++i) {
                    Value* prow = c.b.build_add(q_row0, c.i32(i));
                    Value* p_v4 = c.b.build_add(c.b.build_mul(prow, c.i32(kBK / 4)), c.i32(k4));
                    pa[static_cast<std::size_t>(i)] =
                        i32x4_to_f32x4(c, kb.shared_load_i32x4_indexed(Ps, p_v4));
                }
                for (int e = 0; e < 4; ++e) {
                    Value* vrow = c.i32(k4 * 4 + e);
                    Value* v_v4 = c.b.build_add(c.b.build_mul(vrow, c.i32(kHD / 4)), tx);
                    vb[static_cast<std::size_t>(e)] =
                        i32x4_to_f32x4(c, kb.shared_load_i32x4_indexed(Vs, v_v4));
                }
                for (int i = 0; i < 4; ++i) {
                    const std::size_t o_base = 8u + static_cast<std::size_t>(i) * 4u;
                    Value* pv4 = pa[static_cast<std::size_t>(i)];
                    for (int e = 0; e < 4; ++e) {
                        Value* pe = c.b.build_vextract_lane(pv4, static_cast<std::uint32_t>(e));
                        Value* vbe = vb[static_cast<std::size_t>(e)];
                        for (int j = 0; j < 4; ++j) {
                            nxt[o_base + static_cast<std::size_t>(j)] = c.b.build_fma_f32(
                                pe, c.b.build_vextract_lane(vbe, static_cast<std::uint32_t>(j)),
                                nxt[o_base + static_cast<std::size_t>(j)]);
                        }
                    }
                }
            }
            return nxt;
        });

    Value* tx4 = c.b.build_shl(tx, c.i32(2));
    for (int i = 0; i < 4; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* r = c.b.build_add(q0, c.b.build_add(c.b.build_shl(ty, c.i32(2)), c.i32(i)));
        kb.if_then(c.b.build_slt(r, total_q), [&] {
            Value* lv = final_st[iu * 2 + 1];
            Value* inv = c.b.build_select(
                c.b.build_sgt(lv, c.f32(0.0f)),
                kb.div_approx(c.f32(1.0f), lv),
                c.f32(0.0f));
            Value* vec = c.b.build_vzero(Type::f32x4());
            for (int j = 0; j < 4; ++j) {
                Value* ov = c.b.build_mul(final_st[8u + iu * 4u + static_cast<std::size_t>(j)], inv);
                vec = c.b.build_vinsert_lane(vec, ov, static_cast<std::uint32_t>(j));
            }
            Value* o_off = c.b.build_add(c.b.build_add(c.b.build_mul(r, D), hoff), tx4);
            kb.vstore_f32x4(ptr_at(c, O, o_off, 4), vec);
        });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
