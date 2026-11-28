#include "mir_gemm_mma.h"
#include "mir_common.h"
#include "mir_math.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kMmaBk = 32u;
constexpr std::uint32_t kMmaStages = 3u;
constexpr std::uint32_t kL2GroupM = 8u;

Value* swz_bytes(Ctx& c, Value* row, Value* chunk) {
    Value* row_off = c.b.build_shl(row, c.i32(6));
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    Value* ch_off = c.b.build_shl(c.b.build_xor(chunk, swz), c.i32(4));
    return c.b.build_or(row_off, ch_off);
}

Value* ptr_2d_i64(Ctx& c, Value* base, Value* row, Value* ld, Value* col, int esz_shift) {
    Value* row64 = c.b.build_zext_i64(row);
    Value* ld64 = c.b.build_zext_i64(ld);
    Value* col64 = c.b.build_zext_i64(col);
    Value* elem = c.b.build_add(c.b.build_mul(row64, ld64), col64);
    Value* bytes = c.b.build_shl(elem, c.kb.const_i64(esz_shift));
    return c.b.build_add(base, bytes);
}

Value* pack_pair_i32(Ctx& c, Dtype dt, Value* v0, Value* v1) {
    if (dt == Dtype::FP16) return c.kb.pack_f16x2(v0, v1);
    Value* b0 = c.b.build_and(c.kb.f32_to_bf16(v0), c.i32(0xffff));
    Value* b1 = c.b.build_shl(c.kb.f32_to_bf16(v1), c.i32(16));
    return c.b.build_or(b0, b1);
}

std::pair<Value*, Value*> unpack_pair_f32(Ctx& c, Dtype dt, Value* packed) {
    Value* lo_bits = c.b.build_and(packed, c.i32(0xffff));
    Value* hi_bits = c.b.build_lshr(packed, c.i32(16));
    if (dt == Dtype::FP16) {
        return {c.kb.f16_to_f32(lo_bits), c.kb.f16_to_f32(hi_bits)};
    }
    return {c.kb.bf16_to_f32(lo_bits), c.kb.bf16_to_f32(hi_bits)};
}

}  // namespace

Function* build_gemm_mma_mir(Module& mod, const GemmMmaSpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_gemm_mma_mir: only FP16/BF16 supported");
    }
    const GemmMmaCfg& cfg = kGemmMmaCfgs[std::min<std::uint8_t>(spec.cfg, 3u)];
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
    const int a_stage_bytes = bm * static_cast<int>(kMmaBk) * 2;
    const int b_stage_bytes = bn * static_cast<int>(kMmaBk) * 2;
    const bool swiglu_inline = (spec.epi == 3u && !spec.has_partial);

    std::vector<Type> param_types = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()
    };
    if (spec.batched && !spec.has_partial) {
        param_types.push_back(Type::i32());
        param_types.push_back(Type::i32());
        param_types.push_back(Type::i32());
    }
    Function* fn = mod.create_function("mir_gemm_mma", Type::void_type(), param_types);

    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: A(0) W(1) C(2) bias(3) partial(4) M(5) N(6) K(7) kps(8) grid_m(9) [sA(10) sB(11) sC(12)]
    Value* A_in = p[0];
    Value* W_in = p[1];
    Value* C_in = p[2];
    if (spec.batched && !spec.has_partial) {
        Value* bz64 = c.b.build_zext_i64(kb.ctaid_z());
        auto shift_batch = [&](Value* base, Value* stride32) {
            Value* elems = c.b.build_mul(bz64, c.b.build_zext_i64(stride32));
            return c.b.build_add(base, c.b.build_shl(elems, kb.const_i64(1)));
        };
        A_in = shift_batch(p[0], p[10]);
        W_in = shift_batch(p[1], p[11]);
        C_in = shift_batch(p[2], p[12]);
    }
    Value* M = p[5];
    Value* N = p[6];
    Value* K = p[7];
    Value* kps = p[8];
    Value* grid_m = p[9];

    Value* smem = kb.shared_alloc_u16(kMmaStages * (cfg.bm + cfg.bn) * kMmaBk);
    // `ptx_shared_alloc_*` produces a .shared state-space address (`mov.u64 %rd, smem_0`),
    // already relative to .shared (not generic); truncating directly avoids `cvta.to.shared` underflow.
    Value* sA_base = c.b.build_trunc_i32(smem);
    Value* sB_base = c.b.build_add(sA_base, c.i32(static_cast<int>(kMmaStages) * a_stage_bytes));

    Value* tid = kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* w_m = c.b.build_udiv(warp, c.i32(warps_n));
    Value* w_n = c.b.build_umod(warp, c.i32(warps_n));

    Value* bx = kb.ctaid_x();
    Value* by = kb.ctaid_y();
    Value* tile_m = by;
    Value* tile_n = bx;
    if (spec.swizzle_l2 && !spec.has_partial) {
        Value* gn = kb.nctaid_x();
        Value* linear = c.b.build_add(c.b.build_mul(by, gn), bx);
        Value* per_slab = c.b.build_mul(c.i32(static_cast<int>(kL2GroupM)), gn);
        Value* slab = c.b.build_udiv(linear, per_slab);
        Value* in_slab = c.b.build_umod(linear, per_slab);
        Value* m_start = c.b.build_mul(slab, c.i32(static_cast<int>(kL2GroupM)));
        Value* slab_h = smin_i32(c, c.i32(static_cast<int>(kL2GroupM)), c.b.build_sub(grid_m, m_start));
        tile_n = c.b.build_udiv(in_slab, slab_h);
        tile_m = c.b.build_add(m_start, c.b.build_umod(in_slab, slab_h));
    }
    Value* m0 = c.b.build_mul(tile_m, c.i32(bm));
    Value* n0 = c.b.build_mul(tile_n, c.i32(bn));
    Value* N_half = (swiglu_inline || (spec.epi == 2 && !spec.has_partial))
                        ? c.b.build_lshr(N, c.i32(1))
                        : nullptr;
    const int bn_half = bn / 2;
    Value* n0_half = swiglu_inline ? c.b.build_mul(tile_n, c.i32(bn_half)) : nullptr;

    Value* kt_all = c.b.build_lshr(c.b.build_add(K, c.i32(31)), c.i32(5));
    Value* KT = kt_all;
    Value* A_split = A_in;
    Value* W_split = W_in;
    Value* klim = K;
    if (spec.has_partial) {
        Value* kt0 = c.b.build_mul(kb.ctaid_z(), kps);
        KT = smin_i32(c, kps, c.b.build_sub(kt_all, kt0));
        Value* k0_bytes = c.b.build_shl(c.b.build_zext_i64(kt0), kb.const_i64(6));
        A_split = c.b.build_add(A_in, k0_bytes);
        W_split = c.b.build_add(W_in, k0_bytes);
        klim = c.b.build_sub(K, c.b.build_shl(kt0, c.i32(5)));
    }

    // Precompute cp.async source/dest addresses (loop-invariant).
    // Pack out-of-bounds row guards into one i32 bitmask (a_chunks + b_chunks <= 6)
    // to keep live loop-carried predicates within SM89's 128-reg/thread budget.
    Value* lane_ch_k = c.b.build_shl(c.b.build_and(tid, c.i32(3)), c.i32(3));
    Value* row_ok_mask = spec.aligned_mnk ? nullptr : c.i32(0);
    std::vector<Value*> a_s_off(static_cast<std::size_t>(a_chunks));
    std::vector<Value*> a_g_base(static_cast<std::size_t>(a_chunks));
    for (int i = 0; i < a_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        a_s_off[iu] = c.b.build_add(sA_base, swz_bytes(c, row, ch));
        Value* gr = c.b.build_add(m0, row);
        Value* gk0 = c.b.build_shl(ch, c.i32(3));
        a_g_base[iu] = ptr_2d_i64(c, A_split, gr, K, gk0, 1);
        if (!spec.aligned_mnk) {
            Value* bit = c.b.build_select(c.b.build_slt(gr, M), c.i32(1 << i), c.i32(0));
            row_ok_mask = c.b.build_or(row_ok_mask, bit);
        }
    }

    std::vector<Value*> b_s_off(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_g_base(static_cast<std::size_t>(b_chunks));
    for (int i = 0; i < b_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        b_s_off[iu] = c.b.build_add(sB_base, swz_bytes(c, row, ch));
        Value* gk0 = c.b.build_shl(ch, c.i32(3));
        Value* gr = nullptr;
        Value* valid_lim_cond = nullptr;
        if (swiglu_inline) {
            Value* w_col = c.b.build_lshr(row, c.i32(5));
            Value* in_w = c.b.build_and(row, c.i32(31));
            Value* is_up = c.b.build_ne(c.b.build_and(in_w, c.i32(16)), c.i32(0));
            Value* loc_out = c.b.build_or(c.b.build_shl(w_col, c.i32(4)), c.b.build_and(in_w, c.i32(15)));
            Value* out_col = c.b.build_add(n0_half, loc_out);
            gr = c.b.build_select(is_up, c.b.build_add(N_half, out_col), out_col);
            if (!spec.aligned_mnk) valid_lim_cond = c.b.build_slt(out_col, N_half);
        } else {
            gr = c.b.build_add(n0, row);
            if (!spec.aligned_mnk) valid_lim_cond = c.b.build_slt(gr, N);
        }
        b_g_base[iu] = ptr_2d_i64(c, W_split, gr, K, gk0, 1);
        if (!spec.aligned_mnk) {
            Value* bit = c.b.build_select(valid_lim_cond, c.i32(1 << (a_chunks + i)), c.i32(0));
            row_ok_mask = c.b.build_or(row_ok_mask, bit);
        }
    }

    // Precompute ldmatrix shared addresses for kk = 0; kk = 1 is addr ^ 32.
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

    auto emit_load_stage_off = [&](Value* a_st_off, Value* b_st_off, Value* kt_idx) {
        Value* kt_bytes = c.b.build_shl(c.b.build_zext_i64(kt_idx), kb.const_i64(6));
        Value* k0 = c.b.build_shl(kt_idx, c.i32(5));
        Value* gk = c.b.build_add(k0, lane_ch_k);
        Value* k_ok_mask = spec.aligned_mnk
            ? nullptr
            : c.b.build_select(c.b.build_slt(gk, klim), row_ok_mask, c.i32(0));

        for (int i = 0; i < a_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = a_st_off ? c.b.build_add(a_s_off[iu], a_st_off) : a_s_off[iu];
            Value* src = c.b.build_add(a_g_base[iu], kt_bytes);
            if (spec.aligned_mnk) {
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src});
            } else {
                Value* ok = c.b.build_ne(c.b.build_and(k_ok_mask, c.i32(1 << i)), c.i32(0));
                Value* safe_src = c.b.build_select(ok, src, A_split);
                Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, safe_src, sz});
            }
        }
        for (int i = 0; i < b_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = b_st_off ? c.b.build_add(b_s_off[iu], b_st_off) : b_s_off[iu];
            Value* src = c.b.build_add(b_g_base[iu], kt_bytes);
            if (spec.aligned_mnk) {
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src});
            } else {
                Value* ok = c.b.build_ne(
                    c.b.build_and(k_ok_mask, c.i32(1 << (a_chunks + i))), c.i32(0));
                Value* safe_src = c.b.build_select(ok, src, W_split);
                Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, safe_src, sz});
            }
        }
    };
    auto emit_load_stage = [&](int stage, Value* kt_idx) {
        emit_load_stage_off(
            (stage == 0) ? nullptr : c.i32(stage * a_stage_bytes),
            (stage == 0) ? nullptr : c.i32(stage * b_stage_bytes),
            kt_idx);
    };

    const bool use_fast_acc = (spec.fast_accum && spec.dt == Dtype::FP16);
    const char* mma_sym = (spec.dt == Dtype::FP16) ? "ptx_mma_m16n8k16_f16_f32"
                                                 : "ptx_mma_m16n8k16_bf16_f32";

    auto emit_compute_stage_off = [&](Value* a_st_off, Value* b_st_off, std::vector<Value*>& accs) {
        for (int kk = 0; kk < 2; ++kk) {
            auto a_addr = [&](int i) -> Value* {
                Value* base = a_ld_off[static_cast<std::size_t>(i)];
                Value* p0 = a_st_off ? c.b.build_add(base, a_st_off) : base;
                return (kk == 0) ? p0 : c.b.build_xor(p0, c.i32(32));
            };
            auto b_addr = [&](int j) -> Value* {
                Value* base = b_ld_off[static_cast<std::size_t>(j)];
                Value* p0 = b_st_off ? c.b.build_add(base, b_st_off) : base;
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
                    if (use_fast_acc) {
                        Value* zero32 = c.i32(0);
                        Value* packed = c.b.build_call(
                            "ptx_mma_m16n8k16_f16_f16", Type::i64(),
                            {af_cur, b0[ju], b1[ju], zero32, zero32});
                        Value* q0 = c.b.build_trunc_i32(packed);
                        Value* q1 = c.b.build_trunc_i32(c.b.build_lshr(packed, c.kb.const_i64(32)));
                        auto [x0, x1] = unpack_pair_f32(c, Dtype::FP16, q0);
                        auto [x2, x3] = unpack_pair_f32(c, Dtype::FP16, q1);
                        Value* cur = accs[idx];
                        cur = c.b.build_vinsert_lane(cur, c.b.build_add(c.b.build_vextract_lane(cur, 0), x0), 0);
                        cur = c.b.build_vinsert_lane(cur, c.b.build_add(c.b.build_vextract_lane(cur, 1), x1), 1);
                        cur = c.b.build_vinsert_lane(cur, c.b.build_add(c.b.build_vextract_lane(cur, 2), x2), 2);
                        cur = c.b.build_vinsert_lane(cur, c.b.build_add(c.b.build_vextract_lane(cur, 3), x3), 3);
                        accs[idx] = cur;
                    } else {
                        accs[idx] = c.b.build_call(
                            mma_sym, Type::f32x4(),
                            {af_cur, b0[ju], b1[ju], accs[idx]});
                    }
                }
                af_cur = af_next;
            }
        }
    };
    auto emit_compute_stage = [&](int stage, std::vector<Value*>& accs) {
        emit_compute_stage_off(
            (stage == 0) ? nullptr : c.i32(stage * a_stage_bytes),
            (stage == 0) ? nullptr : c.i32(stage * b_stage_bytes),
            accs);
    };

    // Prologue: prefill stages 0 and 1.
    kb.if_then(c.b.build_slt(c.i32(0), KT), [&] { emit_load_stage(0, c.i32(0)); });
    c.b.build_call("ptx_cp_async_commit", Type::void_type());
    kb.if_then(c.b.build_slt(c.i32(1), KT), [&] { emit_load_stage(1, c.i32(1)); });
    c.b.build_call("ptx_cp_async_commit", Type::void_type());

    std::vector<Value*> accs;
    if (use_fast_acc) {
        // Dynamic 3-stage ring avoids LICM hoisting of 3x-unrolled stage offsets,
        // keeping live regs under sm_89's 128-reg/thread 2-CTA budget with 0 spills.
        std::vector<Value*> init;
        init.reserve(static_cast<std::size_t>(num_accs + 2));
        for (int i = 0; i < num_accs; ++i) init.push_back(c.b.build_vzero(Type::f32x4()));
        init.push_back(c.i32(0));  // sr
        init.push_back(c.i32(2));  // sw
        std::vector<Value*> st = kb.for_range_reduce_n(
            c.i32(0), KT, c.i32(1), init,
            [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
                std::vector<Value*> cur_accs(cur.begin(), cur.begin() + num_accs);
                Value* sr = cur[static_cast<std::size_t>(num_accs)];
                Value* sw = cur[static_cast<std::size_t>(num_accs + 1)];
                c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(1)});
                kb.sync();
                Value* knext = c.b.build_add(kt, c.i32(2));
                kb.if_then(c.b.build_slt(knext, KT), [&] {
                    emit_load_stage_off(
                        c.b.build_mul(sw, c.i32(a_stage_bytes)),
                        c.b.build_mul(sw, c.i32(b_stage_bytes)),
                        knext);
                });
                c.b.build_call("ptx_cp_async_commit", Type::void_type());
                emit_compute_stage_off(
                    c.b.build_mul(sr, c.i32(a_stage_bytes)),
                    c.b.build_mul(sr, c.i32(b_stage_bytes)),
                    cur_accs);
                auto inc_mod3 = [&](Value* v) {
                    return c.b.build_select(c.b.build_eq(v, c.i32(2)), c.i32(0), c.b.build_add(v, c.i32(1)));
                };
                cur_accs.push_back(inc_mod3(sr));
                cur_accs.push_back(inc_mod3(sw));
                return cur_accs;
            });
        accs.assign(st.begin(), st.begin() + num_accs);
    } else {
        // Initial accumulator fragments + loop-carried kt_rem counter.
        std::vector<Value*> state_init;
        state_init.reserve(static_cast<std::size_t>(num_accs + 1));
        for (int i = 0; i < num_accs; ++i) state_init.push_back(c.b.build_vzero(Type::f32x4()));
        state_init.push_back(c.i32(0));

        // Hot unrolled 3-stage pipeline loop (step = 3, zero modulo or interior bounds branches).
        Value* kt_limit = c.b.build_sub(KT, c.i32(4));
        std::vector<Value*> state = kb.for_range_reduce_n(
            c.i32(0), kt_limit, c.i32(3), state_init,
            [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
                std::vector<Value*> cur_accs(cur.begin(), cur.begin() + num_accs);
                for (int s = 0; s < 3; ++s) {
                    c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(1)});
                    kb.sync();
                    emit_load_stage((s + 2) % 3, c.b.build_add(kt, c.i32(s + 2)));
                    c.b.build_call("ptx_cp_async_commit", Type::void_type());
                    emit_compute_stage(s, cur_accs);
                }
                cur_accs.push_back(c.b.build_add(kt, c.i32(3)));
                return cur_accs;
            });

        Value* kt_rem = state.back();
        accs.assign(state.begin(), state.begin() + num_accs);
        const std::vector<Type> acc_types(static_cast<std::size_t>(num_accs), Type::f32x4());

        // 4-step tail (kt_rem % 3 == 0 by construction).
        for (int d = 0; d < 4; ++d) {
            const int st_read = d % 3;
            const int st_write = (d + 2) % 3;
            Value* kcur = (d == 0) ? kt_rem : c.b.build_add(kt_rem, c.i32(d));
            accs = emit_if_else_values(
                c, c.b.build_slt(kcur, KT), acc_types,
                [&]() -> std::vector<Value*> {
                    std::vector<Value*> cur_accs = accs;
                    c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(1)});
                    kb.sync();
                    if (d == 0 || d == 1) {
                        Value* knext = c.b.build_add(kcur, c.i32(2));
                        kb.if_then(c.b.build_slt(knext, KT), [&] { emit_load_stage(st_write, knext); });
                    }
                    c.b.build_call("ptx_cp_async_commit", Type::void_type());
                    emit_compute_stage(st_read, cur_accs);
                    return cur_accs;
                },
                [&]() -> std::vector<Value*> { return accs; });
        }
    }
    c.b.build_call("ptx_cp_async_wait_all", Type::void_type());

    // Epilogue straight from m16n8k16 accumulators.
    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t2 = c.b.build_shl(c.b.build_and(lane, c.i32(3)), c.i32(1));
    Value* p_base = nullptr;
    if (spec.has_partial) {
        Value* z64 = c.b.build_zext_i64(kb.ctaid_z());
        Value* mn64 = c.b.build_mul(c.b.build_zext_i64(M), c.b.build_zext_i64(N));
        p_base = c.b.build_add(p[4], c.b.build_shl(c.b.build_mul(z64, mn64), kb.const_i64(2)));
    }

    if (swiglu_inline) {
        const int wn_half = wn / 2;
        Value* wn_half_base = c.b.build_mul(w_n, c.i32(wn_half));
        for (int j = 0; j < nt / 2; ++j) {
            Value* col = c.b.build_add(c.b.build_add(n0_half, wn_half_base), c.b.build_add(c.i32(j * 8), t2));
            auto emit_col = [&] {
                Value* bg0 = nullptr;
                Value* bg1 = nullptr;
                Value* bu0 = nullptr;
                Value* bu1 = nullptr;
                if (spec.has_bias) {
                    Value* bgp = ptr_at(c, p[3], col, 2);
                    Value* bup = ptr_at(c, p[3], c.b.build_add(N_half, col), 2);
                    std::tie(bg0, bg1) = unpack_pair_f32(c, spec.dt, kb.load_i32(bgp));
                    std::tie(bu0, bu1) = unpack_pair_f32(c, spec.dt, kb.load_i32(bup));
                }
                for (int i = 0; i < mt; ++i) {
                    Value* frag_g = accs[static_cast<std::size_t>(i * nt + j)];
                    Value* frag_u = accs[static_cast<std::size_t>(i * nt + j + nt / 2)];
                    for (int h = 0; h < 2; ++h) {
                        Value* row = c.b.build_add(c.b.build_add(m0, wm_base),
                                                   c.b.build_add(c.i32(i * 16 + h * 8), g));
                        Value* g0 = c.b.build_vextract_lane(frag_g, 2 * h);
                        Value* g1 = c.b.build_vextract_lane(frag_g, 2 * h + 1);
                        Value* u0 = c.b.build_vextract_lane(frag_u, 2 * h);
                        Value* u1 = c.b.build_vextract_lane(frag_u, 2 * h + 1);
                        auto emit_store = [&] {
                            if (spec.has_bias) {
                                g0 = c.b.build_add(g0, bg0);
                                g1 = c.b.build_add(g1, bg1);
                                u0 = c.b.build_add(u0, bu0);
                                u1 = c.b.build_add(u1, bu1);
                            }
                            Value* y0 = c.b.build_mul(emit_gemm_act(c, 4, g0), u0);
                            Value* y1 = c.b.build_mul(emit_gemm_act(c, 4, g1), u1);
                            Value* dst_c = ptr_2d_i64(c, C_in, row, N_half, col, 1);
                            kb.store_i32(dst_c, pack_pair_i32(c, spec.dt, y0, y1));
                        };
                        if (spec.aligned_mnk) emit_store();
                        else kb.if_then(c.b.build_slt(row, M), emit_store);
                    }
                }
            };
            if (spec.aligned_mnk) emit_col();
            else kb.if_then(c.b.build_slt(col, N_half), emit_col);
        }
    } else {
    for (int j = 0; j < nt; ++j) {
        Value* col = c.b.build_add(c.b.build_add(n0, wn_base), c.b.build_add(c.i32(j * 8), t2));
        auto emit_col = [&] {
            Value* b0 = nullptr;
            Value* b1 = nullptr;
            if (!spec.has_partial && spec.has_bias) {
                Value* bp = ptr_at(c, p[3], col, 2);
                std::tie(b0, b1) = unpack_pair_f32(c, spec.dt, kb.load_i32(bp));
            }
            for (int i = 0; i < mt; ++i) {
                Value* frag = accs[static_cast<std::size_t>(i * nt + j)];
                for (int h = 0; h < 2; ++h) {
                    Value* row = c.b.build_add(c.b.build_add(m0, wm_base),
                                               c.b.build_add(c.i32(i * 16 + h * 8), g));
                    Value* v0 = c.b.build_vextract_lane(frag, 2 * h);
                    Value* v1 = c.b.build_vextract_lane(frag, 2 * h + 1);
                    auto emit_store = [&] {
                        if (spec.has_partial) {
                            Value* dst_p = ptr_2d_i64(c, p_base, row, N, col, 2);
                            kb.store_f32(dst_p, v0, 0);
                            kb.store_f32(dst_p, v1, 4);
                        } else {
                            if (spec.has_bias) {
                                v0 = c.b.build_add(v0, b0);
                                v1 = c.b.build_add(v1, b1);
                            }
                            if (spec.epi == 2) {
                                Value* gv = c.b.build_mul(v0, emit_gelu_exact(c, v1));
                                Value* col_h = c.b.build_lshr(col, c.i32(1));
                                Value* dst_g = ptr_2d_i64(c, C_in, row, N_half, col_h, 1);
                                store_lanes(c, spec.dt, 1, dst_g, {gv});
                            } else {
                                if (spec.act != 0) {
                                    v0 = emit_gemm_act(c, spec.act, v0);
                                    v1 = emit_gemm_act(c, spec.act, v1);
                                }
                                Value* dst_c = ptr_2d_i64(c, C_in, row, N, col, 1);
                                if (spec.epi == 1) {
                                    auto [old0, old1] = unpack_pair_f32(c, spec.dt, kb.load_i32(dst_c));
                                    v0 = c.b.build_add(v0, old0);
                                    v1 = c.b.build_add(v1, old1);
                                }
                                kb.store_i32(dst_c, pack_pair_i32(c, spec.dt, v0, v1));
                            }
                        }
                    };
                    if (spec.aligned_mnk) emit_store();
                    else kb.if_then(c.b.build_slt(row, M), emit_store);
                }
            }
        };
        if (spec.aligned_mnk) emit_col();
        else kb.if_then(c.b.build_slt(col, N), emit_col);
    }
    }

    c.b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

}  // namespace brotensor::kernels
