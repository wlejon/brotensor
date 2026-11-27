#include "mir_conv_int8w.h"

#include "mir_common.h"
#include "mir_quant_helpers.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kConvBm = 64;
constexpr int kConvBn = 64;
constexpr int kConvBk = 32;
constexpr int kConvThreads = 128;

Value* swz_bytes(Ctx& c, Value* row, Value* ch) {
    Value* row_off = c.b.build_shl(row, c.i32(6));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    return c.b.build_or(row_off, c.b.build_shl(c.b.build_xor(ch, swz), c.i32(4)));
}

Value* swz_chunk(Ctx& c, Value* row, Value* ch) {
    Value* row_off = c.b.build_shl(row, c.i32(2));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    return c.b.build_or(row_off, c.b.build_xor(ch, swz));
}

Value* make_i32x4(Ctx& c, Value* w0, Value* w1, Value* w2, Value* w3) {
    Value* v = c.b.build_vzero(Type::i32x4());
    v = c.b.build_vinsert_lane(v, w0, 0);
    v = c.b.build_vinsert_lane(v, w1, 1);
    v = c.b.build_vinsert_lane(v, w2, 2);
    return c.b.build_vinsert_lane(v, w3, 3);
}

struct ConvCfg { int kh, kw, ph, pw, sh, sw; };
constexpr ConvCfg kConvCfgs[3] = {
    {3, 3, 1, 1, 1, 1},
    {1, 1, 0, 0, 1, 1},
    {3, 3, 1, 1, 2, 2},
};

}  // namespace

Function* build_conv2d_int8w_mma_mir(Module& mod, std::uint8_t cfg_id,
                                     bool has_bias, bool aligned_k8) {
    if (cfg_id > 2u) {
        throw std::runtime_error("build_conv2d_int8w_mma_mir: bad cfg_id");
    }
    const ConvCfg& cc = kConvCfgs[cfg_id];
    const int khw_sz = cc.kh * cc.kw;
    static constexpr int mt = 2;  // WM = 32 -> 2 * 16
    static constexpr int nt = 4;  // WN = 32 -> 4 * 8
    static constexpr int num_accs = mt * nt;
    static constexpr int a_chunks = (kConvBm * 4) / kConvThreads;  // 2
    static constexpr int b_chunks = (kConvBn * 4) / kConvThreads;  // 2
    static constexpr int a_stage_bytes = kConvBm * kConvBk * 2;    // 4096
    static constexpr int b_stage_bytes = kConvBn * kConvBk * 2;    // 4096
    static constexpr int a_stage_chunks = kConvBm * 4;             // 256
    static constexpr int sB_chunk_base = 2 * a_stage_chunks;       // 512
    static constexpr int b_stage_chunks = kConvBn * 4;             // 256

    std::vector<Type> param_types = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function("mir_conv2d_int8w_mma", Type::void_type(), param_types);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* X = p[0];
    Value* W_i8 = p[1];
    Value* scales = p[2];
    Value* bias = p[3];
    Value* Y = p[4];
    Value* C_in = p[6];
    Value* H = p[7];
    Value* W = p[8];
    Value* C_out = p[9];
    Value* W_out = p[11];
    Value* M_tot = p[12];
    Value* HW_out = p[13];
    Value* K_tot = p[14];

    Value* smem = kb.shared_alloc_u16(2u * (kConvBm + kConvBn) * kConvBk);
    Value* sA_base = c.b.build_trunc_i32(smem);
    Value* sB_base = c.b.build_add(sA_base, c.i32(2 * a_stage_bytes));

    Value* tid = kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* w_m = c.b.build_lshr(warp, c.i32(1));
    Value* w_n = c.b.build_and(warp, c.i32(1));
    Value* m0 = c.b.build_mul(kb.ctaid_y(), c.i32(kConvBm));
    Value* n0 = c.b.build_mul(kb.ctaid_x(), c.i32(kConvBn));

    Value* HW_in = c.b.build_mul(H, W);
    Value* CHW_in = c.b.build_mul(C_in, HW_in);
    Value* KT = c.b.build_lshr(c.b.build_add(K_tot, c.i32(31)), c.i32(5));

    // Precompute A spatial coordinates and B row pointers/scales outside the K-loop.
    struct AGather {
        Value* chunk_off;
        Value* ch;
        Value* m_ok;
        Value* x_base;
        Value* in_h0;
        Value* in_w0;
    };
    std::array<AGather, a_chunks> ag{};
    for (int i = 0; i < a_chunks; ++i) {
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * kConvThreads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        Value* m_g = c.b.build_add(m0, row);
        Value* m_ok = c.b.build_slt(m_g, M_tot);
        Value* safe_m = c.b.build_select(m_ok, m_g, c.i32(0));
        Value* n_b = c.b.build_udiv(safe_m, HW_out);
        Value* sp = c.b.build_umod(safe_m, HW_out);
        Value* oh = c.b.build_udiv(sp, W_out);
        Value* ow = c.b.build_umod(sp, W_out);
        Value* in_h0 = (cc.kh == 1 && cc.sh == 1 && cc.ph == 0)
            ? oh
            : c.b.build_sub(c.b.build_mul(oh, c.i32(cc.sh)), c.i32(cc.ph));
        Value* in_w0 = (cc.kw == 1 && cc.sw == 1 && cc.pw == 0)
            ? ow
            : c.b.build_sub(c.b.build_mul(ow, c.i32(cc.sw)), c.i32(cc.pw));
        Value* x_n = c.b.build_add(X, c.b.build_shl(
            c.b.build_mul(c.b.build_zext_i64(n_b), c.b.build_zext_i64(CHW_in)), kb.const_i64(1)));
        if (cc.kh == 1 && cc.kw == 1 && cc.ph == 0 && cc.pw == 0) {
            Value* hw_off = c.b.build_add(c.b.build_mul(oh, W), ow);
            x_n = c.b.build_add(x_n, c.b.build_shl(c.b.build_zext_i64(hw_off), kb.const_i64(1)));
        }
        ag[static_cast<std::size_t>(i)] = {swz_chunk(c, row, ch), ch, m_ok, x_n, in_h0, in_w0};
    }

    struct BLoad {
        Value* chunk_off;
        Value* ch;
        Value* oc_ok;
        Value* w_row_ptr;
        Value* scale;
    };
    std::array<BLoad, b_chunks> bl{};
    for (int i = 0; i < b_chunks; ++i) {
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * kConvThreads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        Value* oc = c.b.build_add(n0, row);
        Value* oc_ok = c.b.build_slt(oc, C_out);
        Value* safe_oc = c.b.build_select(oc_ok, oc, c.i32(0));
        Value* w_ptr = c.b.build_add(
            W_i8, c.b.build_mul(c.b.build_zext_i64(safe_oc), c.b.build_zext_i64(K_tot)));
        Value* sc = c.b.build_select(oc_ok, kb.load_f32(ptr_at(c, scales, safe_oc, 4)), c.f32(0.0f));
        bl[static_cast<std::size_t>(i)] = {
            c.b.build_add(c.i32(sB_chunk_base), swz_chunk(c, row, ch)), ch, oc_ok, w_ptr, sc};
    }

    // Precompute ldmatrix addresses (kk = 0; kk = 1 is addr ^ 32).
    Value* a_lane_row = c.b.build_and(lane, c.i32(15));
    Value* a_lane_ch = c.b.build_lshr(lane, c.i32(4));
    Value* wm_base = c.b.build_mul(w_m, c.i32(32));
    std::array<Value*, mt> a_ld_off{};
    for (int i = 0; i < mt; ++i) {
        Value* row = c.b.build_add(c.b.build_add(wm_base, c.i32(i * 16)), a_lane_row);
        a_ld_off[static_cast<std::size_t>(i)] = c.b.build_add(sA_base, swz_bytes(c, row, a_lane_ch));
    }
    Value* mi = c.b.build_lshr(lane, c.i32(3));
    Value* b_lane_row = c.b.build_add(c.b.build_shl(c.b.build_lshr(mi, c.i32(1)), c.i32(3)),
                                      c.b.build_and(lane, c.i32(7)));
    Value* b_lane_ch = c.b.build_and(mi, c.i32(1));
    Value* wn_base = c.b.build_mul(w_n, c.i32(32));
    std::array<Value*, nt / 2> b_ld_off{};
    for (int j = 0; j < nt / 2; ++j) {
        Value* row = c.b.build_add(c.b.build_add(wn_base, c.i32(j * 16)), b_lane_row);
        b_ld_off[static_cast<std::size_t>(j)] = c.b.build_add(sB_base, swz_bytes(c, row, b_lane_ch));
    }

    auto emit_load_tiles = [&](Value* st_idx, Value* kt_idx) {
        Value* k0 = c.b.build_shl(kt_idx, c.i32(5));
        Value* a_ch_off = c.b.build_mul(st_idx, c.i32(a_stage_chunks));
        Value* b_ch_off = c.b.build_mul(st_idx, c.i32(b_stage_chunks));

        // Gather A tile directly as packed u16x2 pairs (no f16->f32 round-trip).
        for (int i = 0; i < a_chunks; ++i) {
            const AGather& a = ag[static_cast<std::size_t>(i)];
            Value* gk0 = c.b.build_add(k0, c.b.build_shl(a.ch, c.i32(3)));
            std::array<Value*, 4> pairs{};
            for (int pr = 0; pr < 4; ++pr) {
                Value* p_u16[2];
                for (int q = 0; q < 2; ++q) {
                    const int col = 2 * pr + q;
                    Value* gk = c.b.build_add(gk0, c.i32(col));
                    if (cfg_id == 1u) {
                        Value* ok = c.b.build_and(a.m_ok, c.b.build_slt(gk, C_in));
                        Value* xp = c.b.build_add(
                            a.x_base,
                            c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(gk), c.b.build_zext_i64(HW_in)),
                                          kb.const_i64(1)));
                        p_u16[q] = emit_if_else_values(
                            c, ok, {Type::i32()},
                            [&]() -> std::vector<Value*> { return {kb.load_u16(xp)}; },
                            [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                    } else {
                        Value* k_ok = c.b.build_and(a.m_ok, c.b.build_slt(gk, K_tot));
                        Value* safe_k = c.b.build_select(k_ok, gk, c.i32(0));
                        Value* ic = c.b.build_udiv(safe_k, c.i32(khw_sz));
                        Value* rem = c.b.build_umod(safe_k, c.i32(khw_sz));
                        Value* kh = c.b.build_udiv(rem, c.i32(cc.kw));
                        Value* kw = c.b.build_umod(rem, c.i32(cc.kw));
                        Value* ih = c.b.build_add(a.in_h0, kh);
                        Value* iw = c.b.build_add(a.in_w0, kw);
                        Value* in_ok = c.b.build_and(
                            k_ok,
                            c.b.build_and(
                                c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, H)),
                                c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, W))));
                        Value* xoff = c.b.build_add(
                            c.b.build_mul(ic, HW_in),
                            c.b.build_add(c.b.build_mul(ih, W), iw));
                        Value* xp = ptr_at(c, a.x_base, xoff, 2);
                        p_u16[q] = emit_if_else_values(
                            c, in_ok, {Type::i32()},
                            [&]() -> std::vector<Value*> { return {kb.load_u16(xp)}; },
                            [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                    }
                }
                pairs[static_cast<std::size_t>(pr)] = c.b.build_or(
                    c.b.build_and(p_u16[0], c.i32(0xffff)),
                    c.b.build_shl(p_u16[1], c.i32(16)));
            }
            kb.shared_store_i32x4_indexed(
                smem, c.b.build_add(a.chunk_off, a_ch_off),
                make_i32x4(c, pairs[0], pairs[1], pairs[2], pairs[3]));
        }

        // Dequant B tile (INT8 -> FP16).
        for (int i = 0; i < b_chunks; ++i) {
            const BLoad& b = bl[static_cast<std::size_t>(i)];
            Value* gk0 = c.b.build_add(k0, c.b.build_shl(b.ch, c.i32(3)));
            std::array<Value*, 8> fv{};
            if (aligned_k8) {
                Value* ok = c.b.build_and(b.oc_ok, c.b.build_slt(gk0, K_tot));
                Value* q64 = emit_if_else_values(
                    c, ok, {Type::i64()},
                    [&]() -> std::vector<Value*> {
                        return {kb.load_i64(c.b.build_add(b.w_row_ptr, c.b.build_zext_i64(gk0)))};
                    },
                    [&]() -> std::vector<Value*> { return {kb.const_i64(0)}; })[0];
                Value* qlo = c.b.build_trunc_i32(q64);
                Value* qhi = c.b.build_trunc_i32(c.b.build_lshr(q64, kb.const_i64(32)));
                for (int q = 0; q < 8; ++q) {
                    Value* bw = (q < 4) ? qlo : qhi;
                    Value* sb = c.b.build_and(c.b.build_lshr(bw, c.i32((q & 3) * 8)), c.i32(0xff));
                    fv[static_cast<std::size_t>(q)] = c.b.build_mul(emit_s8_byte_to_f32(c, sb), b.scale);
                }
            } else {
                for (int q = 0; q < 8; ++q) {
                    Value* gk = c.b.build_add(gk0, c.i32(q));
                    Value* ok = c.b.build_and(b.oc_ok, c.b.build_slt(gk, K_tot));
                    fv[static_cast<std::size_t>(q)] = emit_if_else_values(
                        c, ok, {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* ub = kb.load_u8(c.b.build_add(b.w_row_ptr, c.b.build_zext_i64(gk)));
                            return {c.b.build_mul(emit_s8_byte_to_f32(c, ub), b.scale)};
                        },
                        [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                }
            }
            Value* v4 = make_i32x4(
                c,
                kb.pack_f16x2(fv[0], fv[1]),
                kb.pack_f16x2(fv[2], fv[3]),
                kb.pack_f16x2(fv[4], fv[5]),
                kb.pack_f16x2(fv[6], fv[7]));
            kb.shared_store_i32x4_indexed(smem, c.b.build_add(b.chunk_off, b_ch_off), v4);
        }
    };

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
            std::array<Value*, nt> b0{};
            std::array<Value*, nt> b1{};
            for (int j = 0; j < nt / 2; ++j) {
                Value* r = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {b_addr(j)});
                const auto j2 = static_cast<std::size_t>(2 * j);
                b0[j2] = c.b.build_vextract_lane(r, 0);
                b1[j2] = c.b.build_vextract_lane(r, 1);
                b0[j2 + 1u] = c.b.build_vextract_lane(r, 2);
                b1[j2 + 1u] = c.b.build_vextract_lane(r, 3);
            }
            for (int i = 0; i < mt; ++i) {
                Value* af = c.b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {a_addr(i)});
                for (int j = 0; j < nt; ++j) {
                    const auto idx = static_cast<std::size_t>(i * nt + j);
                    const auto ju = static_cast<std::size_t>(j);
                    accs[idx] = c.b.build_call(
                        "ptx_mma_m16n8k16_f16_f32", Type::f32x4(), {af, b0[ju], b1[ju], accs[idx]});
                }
            }
        }
    };

    kb.if_then(c.b.build_slt(c.i32(0), KT), [&] {
        emit_load_tiles(c.i32(0), c.i32(0));
        kb.sync();
    });

    std::vector<Value*> init;
    init.reserve(static_cast<std::size_t>(num_accs + 1));
    for (int i = 0; i < num_accs; ++i) init.push_back(c.b.build_vzero(Type::f32x4()));
    init.push_back(c.i32(0));

    std::vector<Value*> state = kb.for_range_reduce_n(
        c.i32(0), KT, c.i32(1), init,
        [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
            std::vector<Value*> cur_accs(cur.begin(), cur.begin() + num_accs);
            Value* cur_st = cur.back();
            Value* nxt_st = c.b.build_xor(cur_st, c.i32(1));
            Value* knext = c.b.build_add(kt, c.i32(1));
            emit_mma(cur_st, cur_accs);
            kb.if_then(c.b.build_slt(knext, KT), [&] {
                emit_load_tiles(nxt_st, knext);
                kb.sync();
            });
            cur_accs.push_back(nxt_st);
            return cur_accs;
        });

    std::vector<Value*> accs(state.begin(), state.begin() + num_accs);

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
                Value* y_base_idx = c.b.build_add(c.b.build_mul(n_b, CHW_out), sp);
                for (int j = 0; j < nt; ++j) {
                    Value* frag = accs[static_cast<std::size_t>(i * nt + j)];
                    for (int c_off = 0; c_off < 2; ++c_off) {
                        Value* oc = c.b.build_add(
                            c.b.build_add(n0, wn_base), c.b.build_add(c.i32(j * 8 + c_off), t2));
                        kb.if_then(c.b.build_slt(oc, C_out), [&] {
                            Value* v = c.b.build_vextract_lane(frag, 2 * h + c_off);
                            if (has_bias) {
                                v = c.b.build_add(v, load_elem_dt(c, Dtype::FP16, bias, oc));
                            }
                            Value* dst_idx = c.b.build_add(y_base_idx, c.b.build_mul(oc, HW_out));
                            store_elem_dt(c, Dtype::FP16, Y, dst_idx, v);
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

Function* build_matmul_int8w_fp16_mir(Module& mod) {
    constexpr int kTile = 16;
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()
    };
    Function* fn = mod.create_function("mir_matmul_int8w_fp16", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // W(0) scales(1) X(2) Y(3) | M(4) N(5) K(6)
    Value* sW = kb.shared_alloc_f32(kTile * kTile);
    Value* sX = kb.shared_alloc_f32(kTile * kTile);

    Value* tid = kb.tid_x();
    Value* tx = c.b.build_and(tid, c.i32(15));
    Value* ty = c.b.build_lshr(tid, c.i32(4));
    Value* row = c.b.build_add(c.b.build_mul(kb.ctaid_y(), c.i32(kTile)), ty);
    Value* col = c.b.build_add(c.b.build_mul(kb.ctaid_x(), c.i32(kTile)), tx);
    Value* row_ok = c.b.build_slt(row, p[4]);
    Value* col_ok = c.b.build_slt(col, p[5]);
    Value* row_scale = emit_if_else_values(
        c, row_ok, {Type::f32()},
        [&]() -> std::vector<Value*> { return {kb.load_f32(ptr_at(c, p[1], row, 4))}; },
        [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];

    Value* n_tiles = c.b.build_lshr(c.b.build_add(p[6], c.i32(15)), c.i32(4));
    Value* ty16 = c.b.build_shl(ty, c.i32(4));
    Value* s_tid = c.b.build_add(ty16, tx);

    Value* acc = kb.for_range_reduce(
        c.i32(0), n_tiles, c.i32(1), c.f32(0.0f),
        [&](Value* t, Value* cur) -> Value* {
            Value* t0 = c.b.build_shl(t, c.i32(4));
            Value* w_col = c.b.build_add(t0, tx);
            Value* x_row = c.b.build_add(t0, ty);
            Value* w_val = emit_if_else_values(
                c, c.b.build_and(row_ok, c.b.build_slt(w_col, p[6])), {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* wi = c.b.build_add(c.b.build_mul(row, p[6]), w_col);
                    Value* ub = kb.load_u8(ptr_at(c, p[0], wi, 1));
                    return {c.b.build_mul(emit_s8_byte_to_f32(c, ub), row_scale)};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            Value* x_val = emit_if_else_values(
                c, c.b.build_and(c.b.build_slt(x_row, p[6]), col_ok), {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* xi = c.b.build_add(c.b.build_mul(x_row, p[5]), col);
                    return {load_elem_dt(c, Dtype::FP16, p[2], xi)};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            kb.shared_store_f32_indexed(sW, s_tid, w_val);
            kb.shared_store_f32_indexed(sX, s_tid, x_val);
            kb.sync();
            Value* next_acc = kb.for_range_reduce(
                c.i32(0), c.i32(kTile), c.i32(1), cur,
                [&](Value* k, Value* a) -> Value* {
                    Value* wv = kb.shared_load_f32_indexed(sW, c.b.build_add(ty16, k));
                    Value* xv = kb.shared_load_f32_indexed(sX, c.b.build_add(c.b.build_shl(k, c.i32(4)), tx));
                    return c.b.build_fma_f32(wv, xv, a);
                });
            kb.sync();
            return next_acc;
        });

    kb.if_then(c.b.build_and(row_ok, col_ok), [&] {
        store_elem_dt(c, Dtype::FP16, p[3], c.b.build_add(c.b.build_mul(row, p[5]), col), acc);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_conv2d_int8w_fallback_mir(Module& mod, bool has_bias) {
    std::vector<Type> params(5, Type::ptr());
    params.insert(params.end(), 18, Type::i32());
    Function* fn = mod.create_function("mir_conv2d_int8w_fallback", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // 0:X 1:W 2:scales 3:bias 4:Y
    // 5:N 6:Ci 7:H 8:Wi 9:Co 10:kH 11:kW 12:Ho 13:Wo 14:sh 15:sw 16:ph 17:pw 18:dh 19:dw 20:Cgi 21:Cgo 22:total
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_slt(idx, p[22]), [&] {
        Value* ow = c.b.build_umod(idx, p[13]);
        Value* t = c.b.build_udiv(idx, p[13]);
        Value* oh = c.b.build_umod(t, p[12]);
        Value* t2 = c.b.build_udiv(t, p[12]);
        Value* oc = c.b.build_umod(t2, p[9]);
        Value* n = c.b.build_udiv(t2, p[9]);

        Value* ih0 = c.b.build_sub(c.b.build_mul(oh, p[14]), p[16]);
        Value* iw0 = c.b.build_sub(c.b.build_mul(ow, p[15]), p[17]);
        Value* scale = kb.load_f32(ptr_at(c, p[2], oc, 4));
        Value* ic_abs_base = c.b.build_mul(c.b.build_udiv(oc, p[21]), p[20]);
        Value* ksz = c.b.build_mul(p[10], p[11]);
        Value* w_oc_base = c.b.build_mul(oc, c.b.build_mul(p[20], ksz));
        Value* hw_in = c.b.build_mul(p[7], p[8]);
        Value* x_n_base = c.b.build_mul(n, c.b.build_mul(p[6], hw_in));

        Value* acc = kb.for_range_reduce(
            c.i32(0), p[20], c.i32(1), c.f32(0.0f),
            [&](Value* ic_loc, Value* a0) -> Value* {
                Value* ic = c.b.build_add(ic_abs_base, ic_loc);
                Value* w_ic = c.b.build_add(w_oc_base, c.b.build_mul(ic_loc, ksz));
                Value* x_ic = c.b.build_add(x_n_base, c.b.build_mul(ic, hw_in));
                return kb.for_range_reduce(
                    c.i32(0), p[10], c.i32(1), a0,
                    [&](Value* kh, Value* a1) -> Value* {
                        Value* ih = c.b.build_add(ih0, c.b.build_mul(kh, p[18]));
                        Value* h_ok = c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, p[7]));
                        return emit_if_else_values(
                            c, h_ok, {Type::f32()},
                            [&]() -> std::vector<Value*> {
                                Value* w_kh = c.b.build_add(w_ic, c.b.build_mul(kh, p[11]));
                                Value* x_row = c.b.build_add(x_ic, c.b.build_mul(ih, p[8]));
                                return {kb.for_range_reduce(
                                    c.i32(0), p[11], c.i32(1), a1,
                                    [&](Value* kw, Value* a2) -> Value* {
                                        Value* iw = c.b.build_add(iw0, c.b.build_mul(kw, p[19]));
                                        Value* w_ok = c.b.build_and(
                                            c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, p[8]));
                                        return emit_if_else_values(
                                            c, w_ok, {Type::f32()},
                                            [&]() -> std::vector<Value*> {
                                                Value* xv = load_elem_dt(
                                                    c, Dtype::FP16, p[0], c.b.build_add(x_row, iw));
                                                Value* ub = kb.load_u8(
                                                    ptr_at(c, p[1], c.b.build_add(w_kh, kw), 1));
                                                Value* wv = c.b.build_mul(emit_s8_byte_to_f32(c, ub), scale);
                                                return {c.b.build_fma_f32(xv, wv, a2)};
                                            },
                                            [&]() -> std::vector<Value*> { return {a2}; })[0];
                                    })};
                            },
                            [&]() -> std::vector<Value*> { return {a1}; })[0];
                    });
            });

        if (has_bias) {
            acc = c.b.build_add(acc, load_elem_dt(c, Dtype::FP16, p[3], oc));
        }
        store_elem_dt(c, Dtype::FP16, p[4], idx, acc);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
