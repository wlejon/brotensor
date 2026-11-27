#include "mir_conv_mma.h"

#include "mir_common.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kConvMmaBk = 32u;
constexpr std::uint32_t kConvMmaStages = 2u;

struct ConvShape { int kh, kw, ph, pw, sh, sw; };
constexpr ConvShape kConvShapes[7] = {
    {3, 3, 1, 1, 1, 1},  // 0: 3x3 p1 s1
    {1, 1, 0, 0, 1, 1},  // 1: 1x1 p0 s1
    {3, 3, 1, 1, 2, 2},  // 2: 3x3 p1 s2
    {3, 3, 0, 0, 1, 1},  // 3: 3x3 p0 s1
    {7, 7, 3, 3, 1, 1},  // 4: 7x7 p3 s1
    {7, 7, 0, 0, 1, 1},  // 5: 7x7 p0 s1
    {5, 5, 2, 2, 1, 1},  // 6: 5x5 p2 s1
};

Value* swz_bytes(Ctx& c, Value* row, Value* chunk) {
    Value* row_off = c.b.build_shl(row, c.i32(6));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    return c.b.build_or(row_off, c.b.build_shl(c.b.build_xor(chunk, swz), c.i32(4)));
}

Value* swz_chunk(Ctx& c, Value* row, Value* chunk) {
    Value* row_off = c.b.build_shl(row, c.i32(2));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    return c.b.build_or(row_off, c.b.build_xor(chunk, swz));
}

Value* make_i32x4(Ctx& c, Value* w0, Value* w1, Value* w2, Value* w3) {
    Value* v = c.b.build_vzero(Type::i32x4());
    v = c.b.build_vinsert_lane(v, w0, 0);
    v = c.b.build_vinsert_lane(v, w1, 1);
    v = c.b.build_vinsert_lane(v, w2, 2);
    return c.b.build_vinsert_lane(v, w3, 3);
}

Value* pack_u16_pair(Ctx& c, Value* lo, Value* hi) {
    return c.b.build_or(c.b.build_and(lo, c.i32(0xffff)), c.b.build_shl(hi, c.i32(16)));
}

Value* ptr_2d_i64(Ctx& c, Value* base, Value* row, Value* ld, Value* col) {
    Value* elem = c.b.build_add(
        c.b.build_mul(c.b.build_zext_i64(row), c.b.build_zext_i64(ld)),
        c.b.build_zext_i64(col));
    return c.b.build_add(base, c.b.build_shl(elem, c.kb.const_i64(1)));
}

}  // namespace

Function* build_conv2d_mma_mir(Module& mod, const Conv2dMmaSpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_conv2d_mma_mir: only FP16/BF16 supported");
    }
    if (spec.shape_id >= 7u) {
        throw std::runtime_error("build_conv2d_mma_mir: invalid shape_id");
    }
    const ConvShape& sh = kConvShapes[spec.shape_id];
    const int khw_sz = sh.kh * sh.kw;
    const Conv2dMmaCfg& cfg = kConv2dMmaCfgs[std::min<std::uint8_t>(spec.cfg, 1u)];
    const int bm = static_cast<int>(cfg.bm);
    const int bn = static_cast<int>(cfg.bn);
    const int warps_n = static_cast<int>(cfg.warps_n);
    const int threads = static_cast<int>(cfg.threads);
    const int wm = bm / static_cast<int>(cfg.warps_m);
    const int wn = bn / warps_n;
    const int mt = wm / 16;  // 2
    const int nt = wn / 8;   // 4
    const int num_accs = mt * nt;
    const int a_rows_thr = 2;  // each warp of 32 lanes covers 2 rows x 8 cols per stage
    const int b_chunks = (bn * 4) / threads;
    const int a_stage_bytes = bm * static_cast<int>(kConvMmaBk) * 2;
    const int b_stage_bytes = bn * static_cast<int>(kConvMmaBk) * 2;
    const int a_stage_chunks = bm * 4;
    const int sB_chunk_base = static_cast<int>(kConvMmaStages) * a_stage_chunks;
    const int b_stage_chunks = bn * 4;

    std::vector<Type> param_types(4, Type::ptr());
    param_types.insert(param_types.end(), 10, Type::i32());
    Function* fn = mod.create_function("mir_conv2d_mma", Type::void_type(), param_types);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: X(0) Wt(1) bias(2) Y(3) | N(4) C_in(5) H(6) W(7) C_out(8) H_out(9) W_out(10) M_tot(11) HW_out(12) K_tot(13)
    Value* X = p[0];
    Value* Wt = p[1];
    Value* bias = p[2];
    Value* Y = p[3];
    Value* C_in = p[5];
    Value* H = p[6];
    Value* W = p[7];
    Value* C_out = p[8];
    Value* W_out = p[10];
    Value* M_tot = p[11];
    Value* HW_out = p[12];
    Value* K_tot = p[13];

    Value* smem = kb.shared_alloc_u16(kConvMmaStages * (cfg.bm + cfg.bn) * kConvMmaBk);
    Value* sA_base = c.b.build_trunc_i32(smem);
    Value* sB_base = c.b.build_add(sA_base, c.i32(static_cast<int>(kConvMmaStages) * a_stage_bytes));

    Value* tid = kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* w_m = c.b.build_udiv(warp, c.i32(warps_n));
    Value* w_n = c.b.build_umod(warp, c.i32(warps_n));
    Value* m0 = c.b.build_mul(kb.ctaid_y(), c.i32(bm));
    Value* n0 = c.b.build_mul(kb.ctaid_x(), c.i32(bn));

    Value* HW_in = c.b.build_mul(H, W);
    Value* CHW_in = c.b.build_mul(C_in, HW_in);
    Value* KT = c.b.build_lshr(c.b.build_add(K_tot, c.i32(31)), c.i32(5));

    // Coalesced NCHW A-tile layout:
    // Every warp reads 2 rows (32 consecutive spatial pixels) at a warp-uniform ch in 0..3.
    Value* a_ch = c.b.build_and(warp, c.i32(3));
    Value* a_w_sub = (spec.cfg == 0u) ? c.i32(0) : c.b.build_lshr(warp, c.i32(2));
    Value* a_base_row = (spec.cfg == 0u)
        ? lane
        : c.b.build_add(c.b.build_shl(a_w_sub, c.i32(6)), lane);

    struct ARowInfo {
        Value* chunk_off;
        Value* m_ok;
        Value* is_int;
        Value* x_n;
        Value* ih0;
        Value* iw0;
        Value* x_fast;
    };
    std::array<ARowInfo, 2> ar{};
    for (int r = 0; r < a_rows_thr; ++r) {
        Value* row = (r == 0) ? a_base_row : c.b.build_add(a_base_row, c.i32(32));
        Value* m_g = c.b.build_add(m0, row);
        Value* m_ok = c.b.build_slt(m_g, M_tot);
        Value* safe_m = c.b.build_select(m_ok, m_g, c.i32(0));
        Value* n_b = c.b.build_udiv(safe_m, HW_out);
        Value* sp = c.b.build_umod(safe_m, HW_out);
        Value* oh = c.b.build_udiv(sp, W_out);
        Value* ow = c.b.build_umod(sp, W_out);
        Value* ih0 = (sh.kh == 1 && sh.sh == 1 && sh.ph == 0)
            ? oh
            : c.b.build_sub(c.b.build_mul(oh, c.i32(sh.sh)), c.i32(sh.ph));
        Value* iw0 = (sh.kw == 1 && sh.sw == 1 && sh.pw == 0)
            ? ow
            : c.b.build_sub(c.b.build_mul(ow, c.i32(sh.sw)), c.i32(sh.pw));
        Value* in_h_ok = c.b.build_and(c.b.build_sge(ih0, c.i32(0)),
                                       c.b.build_sle(c.b.build_add(ih0, c.i32(sh.kh)), H));
        Value* in_w_ok = c.b.build_and(c.b.build_sge(iw0, c.i32(0)),
                                       c.b.build_sle(c.b.build_add(iw0, c.i32(sh.kw)), W));
        Value* is_int = (sh.kh == 1 && sh.kw == 1 && sh.ph == 0 && sh.pw == 0)
            ? m_ok
            : c.b.build_and(m_ok, c.b.build_and(in_h_ok, in_w_ok));
        Value* x_n = c.b.build_add(X, c.b.build_shl(
            c.b.build_mul(c.b.build_zext_i64(n_b), c.b.build_zext_i64(CHW_in)), kb.const_i64(1)));
        Value* hw_off = c.b.build_add(c.b.build_mul(ih0, W), iw0);
        Value* safe_hw = c.b.build_select(is_int, hw_off, c.i32(0));
        Value* x_fast = c.b.build_add(x_n, c.b.build_shl(c.b.build_zext_i64(safe_hw), kb.const_i64(1)));
        ar[static_cast<std::size_t>(r)] = {swz_chunk(c, row, a_ch), m_ok, is_int, x_n, ih0, iw0, x_fast};
    }

    // Precompute B descriptors (cp.async when aligned_k8, guarded u16 loads otherwise).
    Value* lane_ch_k = c.b.build_shl(c.b.build_and(tid, c.i32(3)), c.i32(3));
    std::vector<Value*> b_s_off(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_chunk_off(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_g_base(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_row_ok(static_cast<std::size_t>(b_chunks));
    for (int i = 0; i < b_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        b_s_off[iu] = c.b.build_add(sB_base, swz_bytes(c, row, ch));
        b_chunk_off[iu] = c.b.build_add(c.i32(sB_chunk_base), swz_chunk(c, row, ch));
        Value* oc = c.b.build_add(n0, row);
        b_row_ok[iu] = c.b.build_slt(oc, C_out);
        b_g_base[iu] = ptr_2d_i64(c, Wt, oc, K_tot, c.b.build_shl(ch, c.i32(3)));
    }

    // Precompute ldmatrix shared addresses (kk = 0; kk = 1 is addr ^ 32).
    Value* a_lane_row = c.b.build_and(lane, c.i32(15));
    Value* a_lane_ch = c.b.build_lshr(lane, c.i32(4));
    Value* wm_base = c.b.build_mul(w_m, c.i32(wm));
    std::vector<Value*> a_ld_off(static_cast<std::size_t>(mt));
    for (int i = 0; i < mt; ++i) {
        Value* row = c.b.build_add(c.b.build_add(wm_base, c.i32(i * 16)), a_lane_row);
        a_ld_off[static_cast<std::size_t>(i)] = c.b.build_add(sA_base, swz_bytes(c, row, a_lane_ch));
    }
    Value* mi = c.b.build_lshr(lane, c.i32(3));
    Value* b_lane_row = c.b.build_add(c.b.build_shl(c.b.build_lshr(mi, c.i32(1)), c.i32(3)),
                                      c.b.build_and(lane, c.i32(7)));
    Value* b_lane_ch = c.b.build_and(mi, c.i32(1));
    Value* wn_base = c.b.build_mul(w_n, c.i32(wn));
    std::vector<Value*> b_ld_off(static_cast<std::size_t>(nt / 2));
    for (int j = 0; j < nt / 2; ++j) {
        Value* row = c.b.build_add(c.b.build_add(wn_base, c.i32(j * 16)), b_lane_row);
        b_ld_off[static_cast<std::size_t>(j)] = c.b.build_add(sB_base, swz_bytes(c, row, b_lane_ch));
    }

    auto emit_load_B_cp = [&](Value* st_idx, Value* kt_idx) {
        Value* b_st_off = c.b.build_mul(st_idx, c.i32(b_stage_bytes));
        Value* kt_bytes = c.b.build_shl(c.b.build_zext_i64(kt_idx), kb.const_i64(6));
        Value* k_ok = c.b.build_slt(c.b.build_add(c.b.build_shl(kt_idx, c.i32(5)), lane_ch_k), K_tot);
        for (int i = 0; i < b_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = c.b.build_add(b_s_off[iu], b_st_off);
            Value* src = c.b.build_add(b_g_base[iu], kt_bytes);
            Value* ok = c.b.build_and(b_row_ok[iu], k_ok);
            Value* safe_src = c.b.build_select(ok, src, Wt);
            Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
            c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, safe_src, sz});
        }
        c.b.build_call("ptx_cp_async_commit", Type::void_type());
    };

    auto emit_fetch_B_u16 = [&](Value* kt_idx) -> std::vector<Value*> {
        std::vector<Value*> packed(static_cast<std::size_t>(b_chunks * 4));
        Value* kt_bytes = c.b.build_shl(c.b.build_zext_i64(kt_idx), kb.const_i64(6));
        Value* gk0 = c.b.build_add(c.b.build_shl(kt_idx, c.i32(5)), lane_ch_k);
        for (int i = 0; i < b_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* base = c.b.build_add(b_g_base[iu], kt_bytes);
            for (int pr = 0; pr < 4; ++pr) {
                Value* u0 = nullptr;
                Value* u1 = nullptr;
                for (int q = 0; q < 2; ++q) {
                    const int e = 2 * pr + q;
                    Value* ok = c.b.build_and(b_row_ok[iu], c.b.build_slt(c.b.build_add(gk0, c.i32(e)), K_tot));
                    Value* src = c.b.build_add(base, kb.const_i64(e * 2));
                    Value* v = c.b.build_select(
                        ok, kb.load_u16(c.b.build_select(ok, src, Wt)), c.i32(0));
                    if (q == 0) u0 = v;
                    else u1 = v;
                }
                packed[iu * 4u + static_cast<std::size_t>(pr)] = pack_u16_pair(c, u0, u1);
            }
        }
        return packed;
    };

    auto emit_fetch_A_u16 = [&](Value* kt_idx) -> std::vector<Value*> {
        // Precompute warp-uniform k offsets across q = 0..7 once for both rows.
        Value* gk0 = c.b.build_add(c.b.build_shl(kt_idx, c.i32(5)), c.b.build_shl(a_ch, c.i32(3)));
        std::array<Value*, 8> ic{}, kh{}, kw{}, k_off{}, k_ok{};
        for (int q = 0; q < 8; ++q) {
            Value* gk = c.b.build_add(gk0, c.i32(q));
            k_ok[q] = c.b.build_slt(gk, K_tot);
            Value* safe_k = c.b.build_select(k_ok[q], gk, c.i32(0));
            if (spec.shape_id == 1u) {
                ic[q] = safe_k;
                k_off[q] = c.b.build_mul(ic[q], HW_in);
            } else {
                ic[q] = c.b.build_udiv(safe_k, c.i32(khw_sz));
                Value* rem = c.b.build_umod(safe_k, c.i32(khw_sz));
                kh[q] = c.b.build_udiv(rem, c.i32(sh.kw));
                kw[q] = c.b.build_umod(rem, c.i32(sh.kw));
                k_off[q] = c.b.build_add(
                    c.b.build_mul(ic[q], HW_in),
                    c.b.build_add(c.b.build_mul(kh[q], W), kw[q]));
            }
        }
        Value* all_k_ok = k_ok[7];
        std::vector<Value*> out(8);
        for (int r = 0; r < a_rows_thr; ++r) {
            const ARowInfo& a = ar[static_cast<std::size_t>(r)];
            Value* fast_ok = c.b.build_and(a.is_int, all_k_ok);
            auto vals = emit_if_else_values(
                c, fast_ok, std::vector<Type>(4, Type::i32()),
                [&]() -> std::vector<Value*> {
                    std::vector<Value*> res(4);
                    for (int pr = 0; pr < 4; ++pr) {
                        Value* p0 = c.b.build_add(a.x_fast, c.b.build_shl(c.b.build_zext_i64(k_off[2 * pr]), kb.const_i64(1)));
                        Value* p1 = c.b.build_add(a.x_fast, c.b.build_shl(c.b.build_zext_i64(k_off[2 * pr + 1]), kb.const_i64(1)));
                        res[static_cast<std::size_t>(pr)] = pack_u16_pair(c, kb.load_u16(p0), kb.load_u16(p1));
                    }
                    return res;
                },
                [&]() -> std::vector<Value*> {
                    return emit_if_else_values(
                        c, a.m_ok, std::vector<Type>(4, Type::i32()),
                        [&]() -> std::vector<Value*> {
                            std::vector<Value*> res(4);
                            for (int pr = 0; pr < 4; ++pr) {
                                Value* u[2];
                                for (int q0 = 0; q0 < 2; ++q0) {
                                    const int q = 2 * pr + q0;
                                    if (spec.shape_id == 1u) {
                                        Value* p0 = c.b.build_add(
                                            a.x_fast, c.b.build_shl(c.b.build_zext_i64(k_off[q]), kb.const_i64(1)));
                                        u[q0] = c.b.build_select(
                                            k_ok[q], kb.load_u16(c.b.build_select(k_ok[q], p0, X)), c.i32(0));
                                    } else {
                                        Value* ih = c.b.build_add(a.ih0, kh[q]);
                                        Value* iw = c.b.build_add(a.iw0, kw[q]);
                                        Value* ok = c.b.build_and(
                                            k_ok[q],
                                            c.b.build_and(
                                                c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, H)),
                                                c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, W))));
                                        Value* xoff = c.b.build_add(
                                            c.b.build_mul(ic[q], HW_in),
                                            c.b.build_add(c.b.build_mul(ih, W), iw));
                                        Value* safe_off = c.b.build_select(ok, xoff, c.i32(0));
                                        u[q0] = c.b.build_select(
                                            ok, kb.load_u16(ptr_at(c, a.x_n, safe_off, 2)), c.i32(0));
                                    }
                                }
                                res[static_cast<std::size_t>(pr)] = pack_u16_pair(c, u[0], u[1]);
                            }
                            return res;
                        },
                        [&]() -> std::vector<Value*> {
                            return {c.i32(0), c.i32(0), c.i32(0), c.i32(0)};
                        });
                });
            for (int pr = 0; pr < 4; ++pr) {
                out[static_cast<std::size_t>(r * 4 + pr)] = vals[static_cast<std::size_t>(pr)];
            }
        }
        return out;
    };

    auto emit_store_A = [&](Value* st_idx, const std::vector<Value*>& a_words) {
        Value* a_ch_off = c.b.build_mul(st_idx, c.i32(a_stage_chunks));
        for (int r = 0; r < a_rows_thr; ++r) {
            const std::size_t o = static_cast<std::size_t>(r * 4);
            Value* v4 = make_i32x4(c, a_words[o], a_words[o + 1], a_words[o + 2], a_words[o + 3]);
            kb.shared_store_i32x4_indexed(
                smem, c.b.build_add(ar[static_cast<std::size_t>(r)].chunk_off, a_ch_off), v4);
        }
    };

    auto emit_store_B_u16 = [&](Value* st_idx, const std::vector<Value*>& b_words) {
        Value* b_ch_off = c.b.build_mul(st_idx, c.i32(b_stage_chunks));
        for (int i = 0; i < b_chunks; ++i) {
            const std::size_t o = static_cast<std::size_t>(i * 4);
            Value* v4 = make_i32x4(c, b_words[o], b_words[o + 1], b_words[o + 2], b_words[o + 3]);
            kb.shared_store_i32x4_indexed(
                smem, c.b.build_add(b_chunk_off[static_cast<std::size_t>(i)], b_ch_off), v4);
        }
    };

    const char* mma_sym = (spec.dt == Dtype::FP16) ? "ptx_mma_m16n8k16_f16_f32"
                                                 : "ptx_mma_m16n8k16_bf16_f32";
    auto emit_mma = [&](Value* st_idx, std::vector<Value*>& accs) {
        Value* a_st_off = c.b.build_mul(st_idx, c.i32(a_stage_bytes));
        Value* b_st_off = c.b.build_mul(st_idx, c.i32(b_stage_bytes));
        for (int kk = 0; kk < 2; ++kk) {
            auto a_addr = [&](int i) -> Value* {
                Value* p0 = c.b.build_add(a_ld_off[static_cast<std::size_t>(i)], a_st_off);
                return (kk == 0) ? p0 : c.b.build_xor(p0, c.i32(32));
            };
            auto b_addr = [&](int j) -> Value* {
                Value* p0 = c.b.build_add(b_ld_off[static_cast<std::size_t>(j)], b_st_off);
                return (kk == 0) ? p0 : c.b.build_xor(p0, c.i32(32));
            };
            std::vector<Value*> b0(static_cast<std::size_t>(nt));
            std::vector<Value*> b1(static_cast<std::size_t>(nt));
            for (int j = 0; j < nt / 2; ++j) {
                Value* r = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {b_addr(j)});
                const auto j2 = static_cast<std::size_t>(2 * j);
                b0[j2] = c.b.build_vextract_lane(r, 0);
                b1[j2] = c.b.build_vextract_lane(r, 1);
                b0[j2 + 1u] = c.b.build_vextract_lane(r, 2);
                b1[j2 + 1u] = c.b.build_vextract_lane(r, 3);
            }
            Value* af_cur = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {a_addr(0)});
            for (int i = 0; i < mt; ++i) {
                Value* af_next = (i + 1 < mt)
                    ? c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {a_addr(i + 1)})
                    : nullptr;
                for (int j = 0; j < nt; ++j) {
                    const auto idx = static_cast<std::size_t>(i * nt + j);
                    const auto ju = static_cast<std::size_t>(j);
                    accs[idx] = c.b.build_call(mma_sym, Type::f32x4(), {af_cur, b0[ju], b1[ju], accs[idx]});
                }
                af_cur = af_next;
            }
        }
    };

    kb.if_then(c.b.build_slt(c.i32(0), KT), [&] {
        if (spec.aligned_k8) emit_load_B_cp(c.i32(0), c.i32(0));
        auto a0 = emit_fetch_A_u16(c.i32(0));
        emit_store_A(c.i32(0), a0);
        if (!spec.aligned_k8) {
            auto b0 = emit_fetch_B_u16(c.i32(0));
            emit_store_B_u16(c.i32(0), b0);
        } else {
            c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(0)});
        }
        kb.sync();
    });

    std::vector<Value*> init;
    init.reserve(static_cast<std::size_t>(num_accs + 1));
    for (int i = 0; i < num_accs; ++i) init.push_back(c.b.build_vzero(Type::f32x4()));
    init.push_back(c.i32(0));

    Value* kt_lim = c.b.build_sub(KT, c.i32(1));
    std::vector<Value*> state = kb.for_range_reduce_n(
        c.i32(0), kt_lim, c.i32(1), init,
        [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
            std::vector<Value*> cur_accs(cur.begin(), cur.begin() + num_accs);
            Value* cur_st = cur.back();
            Value* nxt_st = c.b.build_xor(cur_st, c.i32(1));
            Value* knext = c.b.build_add(kt, c.i32(1));

            if (spec.aligned_k8) emit_load_B_cp(nxt_st, knext);
            auto a_next = emit_fetch_A_u16(knext);
            std::vector<Value*> b_next;
            if (!spec.aligned_k8) b_next = emit_fetch_B_u16(knext);
            emit_mma(cur_st, cur_accs);
            emit_store_A(nxt_st, a_next);
            if (!spec.aligned_k8) {
                emit_store_B_u16(nxt_st, b_next);
            } else {
                c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(0)});
            }
            kb.sync();

            cur_accs.push_back(nxt_st);
            return cur_accs;
        });

    std::vector<Type> tail_types(static_cast<std::size_t>(num_accs), Type::f32x4());
    std::vector<Value*> init_accs(state.begin(), state.begin() + num_accs);
    Value* tail_st = state.back();
    std::vector<Value*> accs = emit_if_else_values(
        c, c.b.build_sgt(KT, c.i32(0)), tail_types,
        [&]() -> std::vector<Value*> {
            std::vector<Value*> a = init_accs;
            emit_mma(tail_st, a);
            return a;
        },
        [&]() -> std::vector<Value*> { return init_accs; });

    // Direct register epilogue: Y[(n * C_out + oc) * HW_out + sp].
    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t2 = c.b.build_shl(c.b.build_and(lane, c.i32(3)), c.i32(1));
    Value* CHW_out = c.b.build_mul(C_out, HW_out);

    for (int i = 0; i < mt; ++i) {
        for (int h = 0; h < 2; ++h) {
            Value* m_g = c.b.build_add(c.b.build_add(m0, wm_base), c.b.build_add(c.i32(i * 16 + h * 8), g));
            kb.if_then(c.b.build_slt(m_g, M_tot), [&] {
                Value* n_b = c.b.build_udiv(m_g, HW_out);
                Value* sp = c.b.build_umod(m_g, HW_out);
                Value* y_base = c.b.build_add(
                    Y, c.b.build_shl(
                        c.b.build_add(c.b.build_mul(c.b.build_zext_i64(n_b), c.b.build_zext_i64(CHW_out)),
                                      c.b.build_zext_i64(sp)),
                        kb.const_i64(1)));
                for (int j = 0; j < nt; ++j) {
                    Value* frag = accs[static_cast<std::size_t>(i * nt + j)];
                    for (int c_off = 0; c_off < 2; ++c_off) {
                        Value* oc = c.b.build_add(
                            c.b.build_add(n0, wn_base), c.b.build_add(c.i32(j * 8 + c_off), t2));
                        kb.if_then(c.b.build_slt(oc, C_out), [&] {
                            Value* v = c.b.build_vextract_lane(frag, 2 * h + c_off);
                            if (spec.has_bias) {
                                v = c.b.build_add(v, load_elem_dt(c, spec.dt, bias, oc));
                            }
                            Value* dst = c.b.build_add(
                                y_base,
                                c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(oc), c.b.build_zext_i64(HW_out)),
                                              kb.const_i64(1)));
                            store_lanes(c, spec.dt, 1, dst, {v});
                        });
                    }
                }
            });
        }
    }

    c.b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

}  // namespace brotensor::kernels
