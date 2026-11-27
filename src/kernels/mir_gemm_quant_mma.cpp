#include "mir_gemm_quant_mma.h"

#include "mir_common.h"
#include "mir_quant_helpers.h"
#include "../vulkan/shaders/op_codes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kQuantBk = 32u;
constexpr std::uint32_t kQuantStages = 2u;
constexpr std::uint32_t kL2PanelN = 16u;

Value* swz_bytes(Ctx& c, Value* row, Value* chunk) {
    Value* row_off = c.b.build_shl(row, c.i32(6));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    Value* ch_off = c.b.build_shl(c.b.build_xor(chunk, swz), c.i32(4));
    return c.b.build_or(row_off, ch_off);
}

Value* swz_chunk(Ctx& c, Value* row, Value* chunk) {
    Value* row_off = c.b.build_shl(row, c.i32(2));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    return c.b.build_or(row_off, c.b.build_xor(chunk, swz));
}

Value* ptr_2d_i64(Ctx& c, Value* base, Value* row, Value* ld, Value* col, int esz_shift) {
    Value* elem = c.b.build_add(
        c.b.build_mul(c.b.build_zext_i64(row), c.b.build_zext_i64(ld)),
        c.b.build_zext_i64(col));
    return c.b.build_add(base, c.b.build_shl(elem, c.kb.const_i64(esz_shift)));
}

Value* pack_pair_i32(Ctx& c, Dtype dt, Value* v0, Value* v1) {
    if (dt == Dtype::FP16) return c.kb.pack_f16x2(v0, v1);
    Value* b0 = c.b.build_and(c.kb.f32_to_bf16(v0), c.i32(0xffff));
    Value* b1 = c.b.build_shl(c.kb.f32_to_bf16(v1), c.i32(16));
    return c.b.build_or(b0, b1);
}

std::pair<Value*, Value*> unpack_pair_f32(Ctx& c, Dtype dt, Value* packed) {
    Value* lo = c.b.build_and(packed, c.i32(0xffff));
    Value* hi = c.b.build_lshr(packed, c.i32(16));
    if (dt == Dtype::FP16) return {c.kb.f16_to_f32(lo), c.kb.f16_to_f32(hi)};
    return {c.kb.bf16_to_f32(lo), c.kb.bf16_to_f32(hi)};
}

// Assemble 32-bit word from two 2-byte aligned u16 loads (safe for Q8_0 34B & Q6_K 210B blocks).
Value* load_u32_align2(Ctx& c, Value* ptr) {
    Value* lo = c.kb.load_u16(ptr);
    Value* hi = c.kb.load_u16(c.b.build_add(ptr, c.kb.const_i64(2)));
    return c.b.build_or(lo, c.b.build_shl(hi, c.i32(16)));
}

Value* make_i32x4(Ctx& c, Value* w0, Value* w1, Value* w2, Value* w3) {
    Value* v = c.b.build_vzero(Type::i32x4());
    v = c.b.build_vinsert_lane(v, w0, 0);
    v = c.b.build_vinsert_lane(v, w1, 1);
    v = c.b.build_vinsert_lane(v, w2, 2);
    return c.b.build_vinsert_lane(v, w3, 3);
}

// Split B load and dequantize so global loads can overlap with Tensor Core MMA.
struct RawBChunk {
    Value* w0 = nullptr;  // INT8: i64 q64; Q8_0: d_u16; Q4_K: hdr i32x4; Q6_K: qlo i32
    Value* w1 = nullptr;  // Q8_0: qlo i32; Q4_K: nib64 i64; Q6_K: qhi i32
    Value* w2 = nullptr;  // Q8_0: qhi i32; Q6_K: hlo i32
    Value* w3 = nullptr;  // Q6_K: hhi i32
    Value* w4 = nullptr;  // Q6_K: bx_u16
    Value* w5 = nullptr;  // Q6_K: d_u16
    Value* in_range = nullptr;
};

RawBChunk fetch_b_raw(Ctx& c, const GemmQuantMmaSpec& spec, Value* row_ptr,
                      Value* kt_idx, Value* ch, Value* in_range) {
    RawBChunk r;
    r.in_range = in_range;
    auto do_fetch = [&](std::vector<Type> types, auto&& fn) -> std::vector<Value*> {
        if (!in_range) return fn();
        std::vector<Value*> zeros;
        zeros.reserve(types.size());
        for (Type t : types) {
            if (t == Type::i64()) zeros.push_back(c.kb.const_i64(0));
            else if (t == Type::i32x4()) zeros.push_back(c.b.build_vzero(Type::i32x4()));
            else zeros.push_back(c.i32(0));
        }
        return emit_if_else_values(c, in_range, types, fn, [&] { return zeros; });
    };

    if (spec.fmt == QF_INT8) {
        auto v = do_fetch({Type::i64()}, [&]() -> std::vector<Value*> {
            Value* gk = c.b.build_add(c.b.build_shl(kt_idx, c.i32(5)), c.b.build_shl(ch, c.i32(3)));
            return {c.kb.load_i64(c.b.build_add(row_ptr, c.b.build_zext_i64(gk)))};
        });
        r.w0 = v[0];
    } else if (spec.fmt == QF_Q8_0) {
        auto v = do_fetch({Type::i32(), Type::i32(), Type::i32()}, [&]() -> std::vector<Value*> {
            Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(kt_idx), c.kb.const_i64(34)));
            Value* d_u16 = c.kb.load_u16(blk);
            Value* qp = c.b.build_add(blk, c.b.build_zext_i64(c.b.build_add(c.i32(2), c.b.build_shl(ch, c.i32(3)))));
            return {d_u16, load_u32_align2(c, qp), load_u32_align2(c, c.b.build_add(qp, c.kb.const_i64(4)))};
        });
        r.w0 = v[0];
        r.w1 = v[1];
        r.w2 = v[2];
    } else if (spec.fmt == QF_Q4K) {
        auto v = do_fetch({Type::i32x4(), Type::i64()}, [&]() -> std::vector<Value*> {
            Value* blk = c.b.build_add(
                row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(kt_idx, c.i32(3))), c.kb.const_i64(144)));
            Value* hdr = c.kb.vload(Type::i32x4(), blk);
            Value* j = c.b.build_and(kt_idx, c.i32(7));
            Value* b_off = c.b.build_add(
                c.i32(16), c.b.build_add(c.b.build_shl(c.b.build_lshr(j, c.i32(1)), c.i32(5)),
                                         c.b.build_shl(ch, c.i32(3))));
            Value* nib64 = c.kb.load_i64(c.b.build_add(blk, c.b.build_zext_i64(b_off)));
            return {hdr, nib64};
        });
        r.w0 = v[0];
        r.w1 = v[1];
    } else if (spec.fmt == QF_Q6K) {
        auto v = do_fetch(std::vector<Type>(6, Type::i32()), [&]() -> std::vector<Value*> {
            Value* blk = c.b.build_add(
                row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(kt_idx, c.i32(3))), c.kb.const_i64(210)));
            Value* e0 = c.b.build_add(c.b.build_shl(c.b.build_and(kt_idx, c.i32(7)), c.i32(5)),
                                      c.b.build_shl(ch, c.i32(3)));
            Value* g = c.b.build_lshr(e0, c.i32(7));
            Value* quad = c.b.build_lshr(c.b.build_and(e0, c.i32(127)), c.i32(5));
            Value* l0 = c.b.build_and(e0, c.i32(31));
            Value* ql_off = c.b.build_add(
                c.b.build_add(c.b.build_shl(g, c.i32(6)), c.b.build_shl(c.b.build_and(quad, c.i32(1)), c.i32(5))),
                l0);
            Value* qh_off = c.b.build_add(c.i32(128), c.b.build_add(c.b.build_shl(g, c.i32(5)), l0));
            Value* sc_off = c.b.build_add(c.i32(192), c.b.build_add(c.b.build_shl(g, c.i32(3)),
                                                                    c.b.build_shl(quad, c.i32(1))));
            Value* qlp = c.b.build_add(blk, c.b.build_zext_i64(ql_off));
            Value* qhp = c.b.build_add(blk, c.b.build_zext_i64(qh_off));
            return {
                load_u32_align2(c, qlp),
                load_u32_align2(c, c.b.build_add(qlp, c.kb.const_i64(4))),
                load_u32_align2(c, qhp),
                load_u32_align2(c, c.b.build_add(qhp, c.kb.const_i64(4))),
                c.kb.load_u16(c.b.build_add(blk, c.b.build_zext_i64(sc_off))),
                c.kb.load_u16(c.b.build_add(blk, c.kb.const_i64(208))),
            };
        });
        r.w0 = v[0]; r.w1 = v[1]; r.w2 = v[2]; r.w3 = v[3]; r.w4 = v[4]; r.w5 = v[5];
    } else {
        throw std::runtime_error("build_gemm_quant_mma_mir: unsupported quant fmt");
    }
    return r;
}

Value* decode_raw_b(Ctx& c, const GemmQuantMmaSpec& spec, const RawBChunk& r,
                    Value* row_scale, Value* kt_idx, Value* ch) {
    std::array<Value*, 8> w{};
    if (spec.fmt == QF_INT8) {
        Value* qlo = c.b.build_trunc_i32(r.w0);
        Value* qhi = c.b.build_trunc_i32(c.b.build_lshr(r.w0, c.kb.const_i64(32)));
        for (int e = 0; e < 8; ++e) {
            Value* bw = (e < 4) ? qlo : qhi;
            Value* b = c.b.build_and(c.b.build_lshr(bw, c.i32((e & 3) * 8)), c.i32(0xff));
            w[static_cast<std::size_t>(e)] = c.b.build_mul(emit_s8_byte_to_f32(c, b), row_scale);
        }
    } else if (spec.fmt == QF_Q8_0) {
        Value* d = c.kb.f16_to_f32(r.w0);
        for (int e = 0; e < 8; ++e) {
            Value* bw = (e < 4) ? r.w1 : r.w2;
            Value* b = c.b.build_and(c.b.build_lshr(bw, c.i32((e & 3) * 8)), c.i32(0xff));
            w[static_cast<std::size_t>(e)] = c.b.build_mul(emit_s8_byte_to_f32(c, b), d);
        }
    } else if (spec.fmt == QF_Q4K) {
        Value* a0 = c.b.build_vextract_lane(r.w0, 0);
        Value* s0 = c.b.build_vextract_lane(r.w0, 1);
        Value* s1 = c.b.build_vextract_lane(r.w0, 2);
        Value* s2 = c.b.build_vextract_lane(r.w0, 3);
        Value* j = c.b.build_and(kt_idx, c.i32(7));
        Value* blo = c.b.build_trunc_i32(r.w1);
        Value* bhi = c.b.build_trunc_i32(c.b.build_lshr(r.w1, c.kb.const_i64(32)));
        Value* low_j = c.b.build_ult(j, c.i32(4));
        Value* j4 = c.b.build_add(j, c.i32(4));
        Value* jm4 = c.b.build_sub(j, c.i32(4));
        Value* bs_j = emit_extract_byte(c, emit_sel_word3(c, c.b.build_lshr(j, c.i32(2)), s0, s1, s2), j);
        Value* bs_j4 = emit_extract_byte(c, emit_sel_word3(c, c.b.build_lshr(j4, c.i32(2)), s0, s1, s2), j4);
        Value* bs_jm4 = emit_extract_byte(c, emit_sel_word3(c, c.b.build_lshr(jm4, c.i32(2)), s0, s1, s2), jm4);
        Value* sc = c.b.build_select(
            low_j, c.b.build_and(bs_j, c.i32(63)),
            c.b.build_or(c.b.build_and(bs_j4, c.i32(15)),
                         c.b.build_shl(c.b.build_lshr(bs_jm4, c.i32(6)), c.i32(4))));
        Value* mn = c.b.build_select(
            low_j, c.b.build_and(bs_j4, c.i32(63)),
            c.b.build_or(c.b.build_lshr(bs_j4, c.i32(4)),
                         c.b.build_shl(c.b.build_lshr(bs_j, c.i32(6)), c.i32(4))));
        Value* d = c.kb.f16_to_f32(c.b.build_and(a0, c.i32(0xffff)));
        Value* dmin = c.kb.f16_to_f32(c.b.build_lshr(a0, c.i32(16)));
        Value* ws = c.b.build_mul(c.kb.u32_to_f32(sc), d);
        Value* wm = c.b.build_mul(c.kb.u32_to_f32(mn), dmin);
        Value* sh = c.b.build_shl(c.b.build_and(j, c.i32(1)), c.i32(2));
        for (int e = 0; e < 8; ++e) {
            Value* bw = (e < 4) ? blo : bhi;
            Value* q4 = c.b.build_and(c.b.build_lshr(bw, c.b.build_add(sh, c.i32((e & 3) * 8))), c.i32(0x0f));
            w[static_cast<std::size_t>(e)] =
                c.b.build_fma_f32(ws, c.kb.u32_to_f32(q4), c.b.build_neg(wm));
        }
    } else {
        Value* e0 = c.b.build_add(c.b.build_shl(c.b.build_and(kt_idx, c.i32(7)), c.i32(5)),
                                  c.b.build_shl(ch, c.i32(3)));
        Value* quad = c.b.build_lshr(c.b.build_and(e0, c.i32(127)), c.i32(5));
        Value* l0 = c.b.build_and(e0, c.i32(31));
        Value* d = c.kb.f16_to_f32(r.w5);
        Value* sbyte = c.b.build_and(
            c.b.build_lshr(r.w4, c.b.build_shl(c.b.build_lshr(l0, c.i32(4)), c.i32(3))),
            c.i32(0xff));
        Value* s = c.b.build_mul(d, emit_s8_byte_to_f32(c, sbyte));
        Value* ql_sh = c.b.build_select(c.b.build_uge(quad, c.i32(2)), c.i32(4), c.i32(0));
        Value* qh_sh = c.b.build_shl(quad, c.i32(1));
        for (int e = 0; e < 8; ++e) {
            Value* bw = (e < 4) ? r.w0 : r.w1;
            Value* hw = (e < 4) ? r.w2 : r.w3;
            Value* bsh = c.i32((e & 3) * 8);
            Value* nib = c.b.build_and(c.b.build_lshr(c.b.build_lshr(bw, bsh), ql_sh), c.i32(0x0f));
            Value* hb = c.b.build_shl(
                c.b.build_and(c.b.build_lshr(c.b.build_lshr(hw, bsh), qh_sh), c.i32(0x03)), c.i32(4));
            Value* q6 = c.b.build_sub(c.kb.u32_to_f32(c.b.build_or(nib, hb)), c.f32(32.0f));
            w[static_cast<std::size_t>(e)] = c.b.build_mul(s, q6);
        }
    }
    Value* p0 = pack_pair_i32(c, spec.dt, w[0], w[1]);
    Value* p1 = pack_pair_i32(c, spec.dt, w[2], w[3]);
    Value* p2 = pack_pair_i32(c, spec.dt, w[4], w[5]);
    Value* p3 = pack_pair_i32(c, spec.dt, w[6], w[7]);
    if (!r.in_range) return make_i32x4(c, p0, p1, p2, p3);
    Value* zero = c.i32(0);
    return make_i32x4(
        c,
        c.b.build_select(r.in_range, p0, zero),
        c.b.build_select(r.in_range, p1, zero),
        c.b.build_select(r.in_range, p2, zero),
        c.b.build_select(r.in_range, p3, zero));
}

}  // namespace

Function* build_gemm_quant_mma_mir(Module& mod, const GemmQuantMmaSpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_gemm_quant_mma_mir: only FP16/BF16 activations supported");
    }
    if (spec.dt == Dtype::BF16 && spec.fmt != QF_INT8) {
        throw std::runtime_error("build_gemm_quant_mma_mir: BF16 only supported for QF_INT8");
    }
    const GemmQuantMmaCfg& cfg = kGemmQuantMmaCfgs[std::min<std::uint8_t>(spec.cfg, 1u)];
    const int bm = static_cast<int>(cfg.bm);
    const int bn = static_cast<int>(cfg.bn);
    const int warps_n = static_cast<int>(cfg.warps_n);
    const int threads = static_cast<int>(cfg.threads);
    const int wm = bm / static_cast<int>(cfg.warps_m);
    const int wn = bn / warps_n;
    const int mt = wm / 16;
    const int nt = wn / 8;
    const int a_chunks = (bm * 4) / threads;
    const int b_chunks = (bn * 4) / threads;
    const int num_accs = mt * nt;
    const int a_stage_bytes = bm * static_cast<int>(kQuantBk) * 2;
    const int sB_chunk_base = static_cast<int>(kQuantStages) * bm * 4;
    const int b_stage_chunks = bn * 4;
    const int b_stage_bytes = bn * static_cast<int>(kQuantBk) * 2;

    std::vector<Type> param_types = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()
    };
    Function* fn = mod.create_function("mir_gemm_quant_mma", Type::void_type(), param_types);

    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* X = p[0];
    Value* W = p[1];
    Value* scales = p[2];
    Value* bias = p[3];
    Value* Y = p[4];
    Value* B = p[5];
    Value* M = p[6];
    Value* K = p[7];
    Value* row_bytes = p[8];
    Value* grid_m = p[9];

    Value* smem = kb.shared_alloc_u16(kQuantStages * (cfg.bm + cfg.bn) * kQuantBk);
    Value* sA_base = c.b.build_trunc_i32(smem);
    Value* sB_base = c.b.build_add(sA_base, c.i32(static_cast<int>(kQuantStages) * a_stage_bytes));

    Value* tid = kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* w_m = c.b.build_udiv(warp, c.i32(warps_n));
    Value* w_n = c.b.build_umod(warp, c.i32(warps_n));

    Value* bx = kb.ctaid_x();
    Value* by = kb.ctaid_y();
    Value* tile_m = by;
    Value* tile_n = bx;
    if (spec.swizzle_l2) {
        Value* gn = kb.nctaid_x();
        Value* lid = c.b.build_add(c.b.build_mul(by, gn), bx);
        Value* panel_n = c.i32(static_cast<int>(kL2PanelN));
        Value* per_panel = c.b.build_mul(panel_n, grid_m);
        Value* pan = c.b.build_udiv(lid, per_panel);
        Value* rem = c.b.build_umod(lid, per_panel);
        Value* n_start = c.b.build_mul(pan, panel_n);
        Value* pw = smin_i32(c, panel_n, c.b.build_sub(gn, n_start));
        tile_m = c.b.build_udiv(rem, pw);
        tile_n = c.b.build_add(n_start, c.b.build_umod(rem, pw));
    }
    Value* m0 = c.b.build_mul(tile_m, c.i32(bm));
    Value* n0 = c.b.build_mul(tile_n, c.i32(bn));

    Value* KT = c.b.build_lshr(c.b.build_add(K, c.i32(31)), c.i32(5));
    Value* lane_ch_k = c.b.build_shl(c.b.build_and(tid, c.i32(3)), c.i32(3));

    Value* a_row_mask = spec.aligned_mnk ? nullptr : c.i32(0);
    std::vector<Value*> a_s_off(static_cast<std::size_t>(a_chunks));
    std::vector<Value*> a_g_base(static_cast<std::size_t>(a_chunks));
    for (int i = 0; i < a_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        a_s_off[iu] = c.b.build_add(sA_base, swz_bytes(c, row, ch));
        Value* gr = c.b.build_add(m0, row);
        a_g_base[iu] = ptr_2d_i64(c, X, gr, K, c.b.build_shl(ch, c.i32(3)), 1);
        if (!spec.aligned_mnk) {
            Value* bit = c.b.build_select(c.b.build_slt(gr, B), c.i32(1 << i), c.i32(0));
            a_row_mask = c.b.build_or(a_row_mask, bit);
        }
    }

    std::vector<Value*> b_chunk_off(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_row_ptr(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_row_ok(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_ch(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_scale(static_cast<std::size_t>(b_chunks), nullptr);
    for (int i = 0; i < b_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        b_ch[iu] = ch;
        b_chunk_off[iu] = c.b.build_add(c.i32(sB_chunk_base), swz_chunk(c, row, ch));
        Value* grow = c.b.build_add(n0, row);
        Value* ok = spec.aligned_mnk ? nullptr : c.b.build_slt(grow, M);
        b_row_ok[iu] = ok;
        Value* safe_row = spec.aligned_mnk ? grow : c.b.build_select(ok, grow, c.i32(0));
        b_row_ptr[iu] = c.b.build_add(
            W, c.b.build_mul(c.b.build_zext_i64(safe_row), c.b.build_zext_i64(row_bytes)));
        if (spec.fmt == QF_INT8) {
            b_scale[iu] = kb.load_f32(ptr_at(c, scales, safe_row, 4));
        }
    }

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

    auto emit_load_A = [&](Value* st_idx, Value* kt_idx) {
        Value* a_st_off = c.b.build_mul(st_idx, c.i32(a_stage_bytes));
        Value* kt_bytes = c.b.build_shl(c.b.build_zext_i64(kt_idx), kb.const_i64(6));
        Value* k0 = c.b.build_shl(kt_idx, c.i32(5));
        Value* gk = c.b.build_add(k0, lane_ch_k);
        Value* k_ok_mask = spec.aligned_mnk
            ? nullptr
            : c.b.build_select(c.b.build_slt(gk, K), a_row_mask, c.i32(0));
        for (int i = 0; i < a_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = c.b.build_add(a_s_off[iu], a_st_off);
            Value* src = c.b.build_add(a_g_base[iu], kt_bytes);
            if (spec.aligned_mnk) {
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src});
            } else {
                Value* ok = c.b.build_ne(c.b.build_and(k_ok_mask, c.i32(1 << i)), c.i32(0));
                Value* safe_src = c.b.build_select(ok, src, X);
                Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, safe_src, sz});
            }
        }
    };

    auto emit_fetch_B = [&](Value* kt_idx) -> std::vector<RawBChunk> {
        Value* k0 = c.b.build_shl(kt_idx, c.i32(5));
        Value* gk_ok = (spec.aligned_mnk || spec.fmt != QF_INT8)
            ? nullptr
            : c.b.build_slt(c.b.build_add(k0, lane_ch_k), K);
        std::vector<RawBChunk> raw(static_cast<std::size_t>(b_chunks));
        for (int i = 0; i < b_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* in_range = spec.aligned_mnk ? nullptr : b_row_ok[iu];
            if (in_range && gk_ok) in_range = c.b.build_and(in_range, gk_ok);
            raw[iu] = fetch_b_raw(c, spec, b_row_ptr[iu], kt_idx, b_ch[iu], in_range);
        }
        return raw;
    };

    auto emit_store_B = [&](Value* st_idx, Value* kt_idx, const std::vector<RawBChunk>& raw) {
        Value* b_ch_off = c.b.build_mul(st_idx, c.i32(b_stage_chunks));
        for (int i = 0; i < b_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* v4 = decode_raw_b(c, spec, raw[iu], b_scale[iu], kt_idx, b_ch[iu]);
            kb.shared_store_i32x4_indexed(smem, c.b.build_add(b_chunk_off[iu], b_ch_off), v4);
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

    // Prefill stage 0 before entering the software-pipelined loop.
    kb.if_then(c.b.build_slt(c.i32(0), KT), [&] {
        emit_load_A(c.i32(0), c.i32(0));
        c.b.build_call("ptx_cp_async_commit", Type::void_type());
        auto raw0 = emit_fetch_B(c.i32(0));
        emit_store_B(c.i32(0), c.i32(0), raw0);
        c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(0)});
        kb.sync();
    });

    std::vector<Value*> init;
    init.reserve(static_cast<std::size_t>(num_accs + 1));
    for (int i = 0; i < num_accs; ++i) init.push_back(c.b.build_vzero(Type::f32x4()));
    init.push_back(c.i32(0));  // cur_stage

    Value* kt_lim = c.b.build_sub(KT, c.i32(1));
    std::vector<Value*> state = kb.for_range_reduce_n(
        c.i32(0), kt_lim, c.i32(1), init,
        [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
            std::vector<Value*> cur_accs(cur.begin(), cur.begin() + num_accs);
            Value* cur_st = cur.back();
            Value* nxt_st = c.b.build_xor(cur_st, c.i32(1));
            Value* knext = c.b.build_add(kt, c.i32(1));

            // Overlap next tile's global reads (A cp.async + B raw loads) with cur tile's MMA.
            emit_load_A(nxt_st, knext);
            c.b.build_call("ptx_cp_async_commit", Type::void_type());
            auto raw_next = emit_fetch_B(knext);
            emit_mma(cur_st, cur_accs);
            emit_store_B(nxt_st, knext, raw_next);
            c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(0)});
            kb.sync();

            cur_accs.push_back(nxt_st);
            return cur_accs;
        });

    // Epilogue MMA for the last K tile (KT - 1).
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

    // Direct register epilogue straight from m16n8k16 accumulators.
    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t2 = c.b.build_shl(c.b.build_and(lane, c.i32(3)), c.i32(1));
    for (int j = 0; j < nt; ++j) {
        Value* col0 = c.b.build_add(c.b.build_add(n0, wn_base), c.b.build_add(c.i32(j * 8), t2));
        Value* col1 = c.b.build_add(col0, c.i32(1));
        if (spec.aligned_mnk) {
            Value* b0 = nullptr;
            Value* b1 = nullptr;
            if (spec.has_bias) {
                std::tie(b0, b1) = unpack_pair_f32(c, spec.dt, kb.load_i32(ptr_at(c, bias, col0, 2)));
            }
            for (int i = 0; i < mt; ++i) {
                Value* frag = accs[static_cast<std::size_t>(i * nt + j)];
                for (int h = 0; h < 2; ++h) {
                    Value* row = c.b.build_add(c.b.build_add(m0, wm_base), c.b.build_add(c.i32(i * 16 + h * 8), g));
                    Value* v0 = c.b.build_vextract_lane(frag, 2 * h);
                    Value* v1 = c.b.build_vextract_lane(frag, 2 * h + 1);
                    if (spec.has_bias) {
                        v0 = c.b.build_add(v0, b0);
                        v1 = c.b.build_add(v1, b1);
                    }
                    kb.store_i32(ptr_2d_i64(c, Y, row, M, col0, 1), pack_pair_i32(c, spec.dt, v0, v1));
                }
            }
        } else {
            Value* c0_ok = c.b.build_slt(col0, M);
            Value* c1_ok = c.b.build_slt(col1, M);
            kb.if_then(c0_ok, [&] {
                Value* b0 = spec.has_bias ? load_elem_dt(c, spec.dt, bias, col0) : nullptr;
                Value* b1 = nullptr;
                if (spec.has_bias) {
                    b1 = emit_if_else_values(
                        c, c1_ok, {Type::f32()},
                        [&]() -> std::vector<Value*> { return {load_elem_dt(c, spec.dt, bias, col1)}; },
                        [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                }
                for (int i = 0; i < mt; ++i) {
                    Value* frag = accs[static_cast<std::size_t>(i * nt + j)];
                    for (int h = 0; h < 2; ++h) {
                        Value* row = c.b.build_add(c.b.build_add(m0, wm_base),
                                                   c.b.build_add(c.i32(i * 16 + h * 8), g));
                        kb.if_then(c.b.build_slt(row, B), [&] {
                            Value* v0 = c.b.build_vextract_lane(frag, 2 * h);
                            Value* v1 = c.b.build_vextract_lane(frag, 2 * h + 1);
                            if (spec.has_bias) {
                                v0 = c.b.build_add(v0, b0);
                                v1 = c.b.build_add(v1, b1);
                            }
                            Value* dst0 = ptr_2d_i64(c, Y, row, M, col0, 1);
                            store_lanes(c, spec.dt, 1, dst0, {v0});
                            kb.if_then(c1_ok, [&] {
                                store_lanes(c, spec.dt, 1, c.b.build_add(dst0, kb.const_i64(2)), {v1});
                            });
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
