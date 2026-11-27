#include "mir_decode_gqa.h"

#include "mir_common.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kBlock = 128;
constexpr int kWarps = kBlock / 32;
constexpr int kMaxSplits = 64;

std::pair<Value*, Value*> unpack_word_dt(Ctx& c, Dtype dt, Value* w) {
    Value* lo_bits = c.b.build_and(w, c.i32(0xffff));
    Value* hi_bits = c.b.build_lshr(w, c.i32(16));
    if (dt == Dtype::FP16) {
        return {c.kb.f16_to_f32(lo_bits), c.kb.f16_to_f32(hi_bits)};
    }
    return {c.kb.bf16_to_f32(lo_bits), c.kb.bf16_to_f32(hi_bits)};
}

std::array<Value*, 8> load8_f32_safe(Ctx& c, Dtype dt, Value* ptr, Value* safe_ptr, Value* valid) {
    Value* p = valid ? c.b.build_select(valid, ptr, safe_ptr) : ptr;
    Value* v4 = c.b.build_call("ptx_ldg_i32x4", Type::i32x4(), {p});
    std::array<Value*, 8> out{};
    for (int i = 0; i < 4; ++i) {
        auto [a, b] = unpack_word_dt(c, dt, c.b.build_vextract_lane(v4, static_cast<std::uint32_t>(i)));
        out[static_cast<std::size_t>(2 * i)] = a;
        out[static_cast<std::size_t>(2 * i + 1)] = b;
    }
    return out;
}

Value* pack_pair_i32(Ctx& c, Dtype dt, Value* v0, Value* v1) {
    if (dt == Dtype::FP16) return c.kb.pack_f16x2(v0, v1);
    Value* b0 = c.b.build_and(c.kb.f32_to_bf16(v0), c.i32(0xffff));
    Value* b1 = c.b.build_shl(c.kb.f32_to_bf16(v1), c.i32(16));
    return c.b.build_or(b0, b1);
}

Value* ptr_idx(Ctx& c, Value* base, Value* elem_idx, int esz) {
    return ptr_at(c, base, elem_idx, esz);
}

brass::Function* build_partials(brass::Module& mod, Dtype dt, std::uint8_t flags) {
    static constexpr int kHds[3] = {64, 128, 256};
    static constexpr int kGs[4] = {1, 2, 4, 8};
    const int hd_code = static_cast<int>(flags & 3u);
    const int g_code = static_cast<int>((flags >> 2) & 3u);
    const int rc_code = static_cast<int>((flags >> 4) & 1u);
    if (hd_code > 2) throw std::runtime_error("mir_decode_gqa: invalid hd_code");

    const int HD = kHds[hd_code];
    const int G = kGs[g_code];
    const int RC = (rc_code != 0) ? (8 / G) : 1;
    const int NG = G * RC;
    const int LPK = HD / 8;
    const int KPS = 32 / LPK;
    const int STRIDE = kWarps * KPS;

    const std::string name = std::string("mir_decode_gqa_p_") +
                             (dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_hd" + std::to_string(HD) + "_g" + std::to_string(G) +
                             "_rc" + std::to_string(RC);
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::f32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:Q 1:K 2:V 3:d_len 4:ws | 5:cap 6:Dkv 7:n_splits 8:nq 9:R 10:r0 | 11:inv_sqrt_log2
    Value* Q = p[0];
    Value* K = p[1];
    Value* V = p[2];
    Value* ws = p[4];
    Value* cap = p[5];
    Value* Dkv = p[6];
    Value* n_splits = p[7];
    Value* nq = p[8];
    Value* R = p[9];
    Value* r0 = p[10];
    Value* inv_sqrt_log2 = p[11];

    Value* spl = kb.ctaid_x();
    Value* kvh = kb.ctaid_y();
    Value* tid = kb.tid_x();
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* sub = c.b.build_udiv(lane, c.i32(LPK));
    Value* sl = c.b.build_umod(lane, c.i32(LPK));

    Value* len_raw = kb.load_i32(p[3]);
    Value* len = clamp_i32(c, len_raw, c.i32(0), cap);
    Value* chunk = c.b.build_udiv(c.b.build_sub(c.b.build_add(len, n_splits), c.i32(1)), n_splits);
    Value* k_begin = c.b.build_mul(spl, chunk);
    Value* k_end = smin_i32(c, len, c.b.build_add(k_begin, chunk));

    Value* kvh_g = c.b.build_mul(kvh, c.i32(G));
    Value* sl8 = c.b.build_shl(sl, c.i32(3));
    Value* nq_hd = c.b.build_mul(nq, c.i32(HD));

    std::vector<std::array<Value*, 8>> q_vecs(static_cast<std::size_t>(NG));
    std::vector<Value*> lim(static_cast<std::size_t>(NG));
    for (int gi = 0; gi < NG; ++gi) {
        const auto giu = static_cast<std::size_t>(gi);
        Value* rr = c.b.build_add(r0, c.i32(gi / G));
        Value* valid_r = c.b.build_slt(rr, R);
        Value* lim_v = c.b.build_add(c.b.build_add(c.b.build_sub(len, R), c.i32(1)), rr);
        lim[giu] = c.b.build_select(valid_r, lim_v, c.i32(0));
        Value* h = c.b.build_add(kvh_g, c.i32(gi % G));
        Value* q_off = c.b.build_add(
            c.b.build_mul(rr, nq_hd),
            c.b.build_add(c.b.build_mul(h, c.i32(HD)), sl8));
        auto q_raw = load8_f32_safe(c, dt, ptr_idx(c, Q, q_off, 2), Q, valid_r);
        for (int i = 0; i < 8; ++i) {
            Value* scaled = c.b.build_mul(q_raw[static_cast<std::size_t>(i)], inv_sqrt_log2);
            q_vecs[giu][static_cast<std::size_t>(i)] = c.b.build_select(valid_r, scaled, c.f32(0.0f));
        }
    }

    Value* col = c.b.build_add(c.b.build_mul(kvh, c.i32(HD)), sl8);
    Value* init_base = c.b.build_add(k_begin, c.b.build_mul(warp, c.i32(KPS)));

    std::vector<Value*> init_st;
    init_st.reserve(static_cast<std::size_t>(NG * 10));
    for (int g = 0; g < NG; ++g) {
        init_st.push_back(c.f32(-1e30f));  // m[g]
        init_st.push_back(c.f32(0.0f));    // l[g]
        for (int i = 0; i < 8; ++i) init_st.push_back(c.f32(0.0f));
    }

    auto st_out = kb.for_range_reduce_n(
        init_base, k_end, c.i32(2 * STRIDE), init_st,
        [&](Value* base, const std::vector<Value*>& cur) -> std::vector<Value*> {
            Value* k0 = c.b.build_add(base, sub);
            Value* k1 = c.b.build_add(c.b.build_add(base, c.i32(STRIDE)), sub);
            Value* v0 = c.b.build_slt(k0, k_end);
            Value* v1 = c.b.build_slt(k1, k_end);

            Value* off0 = c.b.build_add(c.b.build_mul(k0, Dkv), col);
            Value* off1 = c.b.build_add(c.b.build_mul(k1, Dkv), col);
            auto ka = load8_f32_safe(c, dt, ptr_idx(c, K, off0, 2), K, v0);
            auto va = load8_f32_safe(c, dt, ptr_idx(c, V, off0, 2), V, v0);
            auto kb_v = load8_f32_safe(c, dt, ptr_idx(c, K, off1, 2), K, v1);
            auto vb = load8_f32_safe(c, dt, ptr_idx(c, V, off1, 2), V, v1);

            std::vector<Value*> nxt = cur;
            for (int step = 0; step < 2; ++step) {
                Value* valid = (step == 0) ? v0 : v1;
                const auto& kk = (step == 0) ? ka : kb_v;
                const auto& vv = (step == 0) ? va : vb;
                Value* kidx = (step == 0) ? k0 : k1;
                for (int g = 0; g < NG; ++g) {
                    const std::size_t off = static_cast<std::size_t>(g * 10);
                    Value* s = c.f32(0.0f);
                    for (int i = 0; i < 8; ++i) {
                        s = c.b.build_fma_f32(
                            q_vecs[static_cast<std::size_t>(g)][static_cast<std::size_t>(i)],
                            kk[static_cast<std::size_t>(i)], s);
                    }
                    s = c.b.build_select(valid, s, c.f32(0.0f));
                    for (int red_off = LPK / 2; red_off > 0; red_off >>= 1) {
                        s = c.b.build_add(s, kb.shfl_bfly_f32(s, static_cast<std::uint32_t>(red_off)));
                    }
                    Value* attend = c.b.build_and(valid, c.b.build_slt(kidx, lim[static_cast<std::size_t>(g)]));
                    Value* m_cur = nxt[off + 0];
                    Value* l_cur = nxt[off + 1];
                    Value* mn = kb.fmax(m_cur, s);
                    Value* c_scale = kb.ex2_approx(c.b.build_sub(m_cur, mn));
                    Value* p_val = kb.ex2_approx(c.b.build_sub(s, mn));
                    Value* l_new = c.b.build_fma_f32(l_cur, c_scale, p_val);
                    nxt[off + 0] = c.b.build_select(attend, mn, m_cur);
                    nxt[off + 1] = c.b.build_select(attend, l_new, l_cur);
                    for (int i = 0; i < 8; ++i) {
                        Value* ai = nxt[off + 2 + static_cast<std::size_t>(i)];
                        Value* a_new = c.b.build_fma_f32(
                            p_val, vv[static_cast<std::size_t>(i)], c.b.build_mul(ai, c_scale));
                        nxt[off + 2 + static_cast<std::size_t>(i)] = c.b.build_select(attend, a_new, ai);
                    }
                }
            }
            return nxt;
        });

    // In-warp merge across key slots (lanes sharing `sl`).
    for (int off = LPK; off < 32; off <<= 1) {
        const auto mask = static_cast<std::uint32_t>(off);
        for (int g = 0; g < NG; ++g) {
            const std::size_t base = static_cast<std::size_t>(g * 10);
            Value* m1 = st_out[base + 0];
            Value* l1 = st_out[base + 1];
            Value* m2 = kb.shfl_bfly_f32(m1, mask);
            Value* l2 = kb.shfl_bfly_f32(l1, mask);
            Value* mn = kb.fmax(m1, m2);
            Value* ca = kb.ex2_approx(c.b.build_sub(m1, mn));
            Value* cb = kb.ex2_approx(c.b.build_sub(m2, mn));
            st_out[base + 0] = mn;
            st_out[base + 1] = c.b.build_fma_f32(l2, cb, c.b.build_mul(l1, ca));
            for (int i = 0; i < 8; ++i) {
                Value* a1 = st_out[base + 2 + static_cast<std::size_t>(i)];
                Value* a2 = kb.shfl_bfly_f32(a1, mask);
                st_out[base + 2 + static_cast<std::size_t>(i)] =
                    c.b.build_fma_f32(a2, cb, c.b.build_mul(a1, ca));
            }
        }
    }

    // Cross-warp merge via shared memory, writing [m, l, acc[HD]] to ws.
    Value* s_ml = kb.shared_alloc_f32(static_cast<std::uint32_t>(kWarps * NG * 2));
    Value* s_acc = kb.shared_alloc_f32(static_cast<std::uint32_t>(kWarps * NG * HD));
    kb.if_then(c.b.build_eq(sub, c.i32(0)), [&] {
        Value* ml_base = c.b.build_mul(warp, c.i32(NG * 2));
        Value* acc_base = c.b.build_mul(warp, c.i32(NG * HD));
        for (int g = 0; g < NG; ++g) {
            const std::size_t sb = static_cast<std::size_t>(g * 10);
            Value* gml = c.b.build_add(ml_base, c.i32(g * 2));
            kb.if_then(c.b.build_eq(sl, c.i32(0)), [&] {
                kb.shared_store_f32_indexed(s_ml, gml, st_out[sb + 0]);
                kb.shared_store_f32_indexed(s_ml, c.b.build_add(gml, c.i32(1)), st_out[sb + 1]);
            });
            Value* gacc = c.b.build_add(c.b.build_add(acc_base, c.i32(g * HD)), sl8);
            for (int i = 0; i < 8; ++i) {
                kb.shared_store_f32_indexed(
                    s_acc, c.b.build_add(gacc, c.i32(i)),
                    st_out[sb + 2 + static_cast<std::size_t>(i)]);
            }
        }
    });
    kb.sync();

    const int rec_w = HD + 2;
    const int total = NG * rec_w;
    kb.for_range(tid, c.i32(total), c.i32(kBlock), [&](Value* idx) {
        Value* g = c.b.build_udiv(idx, c.i32(rec_w));
        Value* e = c.b.build_umod(idx, c.i32(rec_w));
        Value* rr = c.b.build_add(r0, c.b.build_udiv(g, c.i32(G)));
        kb.if_then(c.b.build_slt(rr, R), [&] {
            Value* g_ml = c.b.build_shl(g, c.i32(1));
            Value* m = c.f32(-1e30f);
            std::array<Value*, kWarps> wm{};
            for (int w = 0; w < kWarps; ++w) {
                Value* v = kb.shared_load_f32_indexed(
                    s_ml, c.b.build_add(c.i32(w * NG * 2), g_ml));
                wm[static_cast<std::size_t>(w)] = v;
                m = kb.fmax(m, v);
            }
            Value* g_acc_e = c.b.build_add(c.b.build_mul(g, c.i32(HD)), c.b.build_sub(e, c.i32(2)));
            Value* is_e0 = c.b.build_eq(e, c.i32(0));
            Value* is_e1 = c.b.build_eq(e, c.i32(1));
            Value* safe_acc_e = c.b.build_select(c.b.build_ult(e, c.i32(2)), c.i32(0), g_acc_e);
            Value* sum = c.f32(0.0f);
            for (int w = 0; w < kWarps; ++w) {
                Value* cw = kb.ex2_approx(c.b.build_sub(wm[static_cast<std::size_t>(w)], m));
                Value* lv = kb.shared_load_f32_indexed(
                    s_ml, c.b.build_add(c.i32(w * NG * 2 + 1), g_ml));
                Value* av = kb.shared_load_f32_indexed(
                    s_acc, c.b.build_add(c.i32(w * NG * HD), safe_acc_e));
                Value* val = c.b.build_select(is_e0, c.f32(0.0f), c.b.build_select(is_e1, lv, av));
                sum = c.b.build_fma_f32(cw, val, sum);
            }
            Value* out_v = c.b.build_select(is_e0, m, sum);
            Value* h = c.b.build_add(
                c.b.build_add(c.b.build_mul(rr, nq), kvh_g),
                c.b.build_umod(g, c.i32(G)));
            Value* ws_idx = c.b.build_add(
                c.b.build_mul(c.b.build_add(c.b.build_mul(h, n_splits), spl), c.i32(rec_w)), e);
            kb.store_f32(ptr_idx(c, ws, ws_idx, 4), out_v);
        });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_reduce(brass::Module& mod, Dtype dt) {
    const std::string name = std::string("mir_decode_gqa_r_") + (dt == Dtype::FP16 ? "f16" : "bf16");
    std::vector<Type> params = {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()};
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: 0:ws 1:O 2:hd 3:n_splits
    Value* ws = p[0];
    Value* O = p[1];
    Value* hd = p[2];
    Value* n_splits = p[3];

    Value* sm = kb.shared_alloc_f32(kMaxSplits);
    Value* sl = kb.shared_alloc_f32(kMaxSplits);
    Value* sw = kb.shared_alloc_f32(kMaxSplits);

    Value* h = kb.ctaid_x();
    Value* tid = kb.tid_x();
    Value* rec = c.b.build_add(hd, c.i32(2));
    Value* h_off = c.b.build_mul(c.b.build_mul(h, n_splits), rec);
    Value* base = ptr_idx(c, ws, h_off, 4);

    kb.for_range(tid, n_splits, c.i32(kBlock), [&](Value* s) {
        Value* sr = c.b.build_mul(s, rec);
        kb.shared_store_f32_indexed(sm, s, kb.load_f32(ptr_idx(c, base, sr, 4)));
        kb.shared_store_f32_indexed(sl, s, kb.load_f32(ptr_idx(c, base, c.b.build_add(sr, c.i32(1)), 4)));
    });
    kb.sync();

    Value* m = kb.for_range_reduce(
        c.i32(0), n_splits, c.i32(1), c.f32(-1e30f),
        [&](Value* s, Value* cur) -> Value* {
            return kb.fmax(cur, kb.shared_load_f32_indexed(sm, s));
        });
    kb.for_range(tid, n_splits, c.i32(kBlock), [&](Value* s) {
        Value* mv = kb.shared_load_f32_indexed(sm, s);
        kb.shared_store_f32_indexed(sw, s, kb.ex2_approx(c.b.build_sub(mv, m)));
    });
    kb.sync();

    Value* den = kb.for_range_reduce(
        c.i32(0), n_splits, c.i32(1), c.f32(0.0f),
        [&](Value* s, Value* cur) -> Value* {
            return c.b.build_fma_f32(
                kb.shared_load_f32_indexed(sw, s),
                kb.shared_load_f32_indexed(sl, s), cur);
        });
    Value* inv = c.b.build_select(
        c.b.build_sgt(den, c.f32(0.0f)),
        kb.div_approx(c.f32(1.0f), den),
        c.f32(0.0f));

    Value* o_row = ptr_idx(c, O, c.b.build_mul(h, hd), 2);
    kb.for_range(tid, hd, c.i32(kBlock), [&](Value* d) {
        Value* num = kb.for_range_reduce(
            c.i32(0), n_splits, c.i32(1), c.f32(0.0f),
            [&](Value* s, Value* cur) -> Value* {
                Value* w = kb.shared_load_f32_indexed(sw, s);
                Value* val = kb.load_f32(
                    ptr_idx(c, base, c.b.build_add(c.b.build_add(c.b.build_mul(s, rec), c.i32(2)), d), 4));
                return c.b.build_fma_f32(w, val, cur);
            });
        Value* res = c.b.build_mul(num, inv);
        store_elem_dt(c, dt, o_row, d, res);
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace

brass::Function* build_decode_gqa_mir(brass::Module& mod, std::uint32_t op, Dtype dt, std::uint8_t flags) {
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("build_decode_gqa_mir: only FP16/BF16 supported");
    }
    if (op == 0u) return build_partials(mod, dt, flags);
    if (op == 1u) return build_reduce(mod, dt);
    throw std::runtime_error("build_decode_gqa_mir: unsupported op");
}

}  // namespace brotensor::kernels
