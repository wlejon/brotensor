#include "mir_gemm_mma_nn.h"
#include "mir_gemm_mma.h"
#include "mir_common.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kNnBk = 32u;
constexpr std::uint32_t kNnStages = 3u;
constexpr std::uint32_t kNnL2GroupM = 8u;

// A tile has 32 FP16/BF16 cols -> 64 bytes/row -> 4 x 16B chunks per row.
Value* swz_a_bytes(Ctx& c, Value* row, Value* ch) {
    Value* swz = c.b.build_and(c.b.build_ashr(row, c.i32(1)), c.i32(3));
    return c.b.build_or(c.b.build_shl(row, c.i32(6)),
                        c.b.build_shl(c.b.build_xor(ch, swz), c.i32(4)));
}

// B tile has BN cols: bn==128 -> 256 bytes/row (16 chunks), bn==64 -> 128 bytes/row (8 chunks).
Value* swz_b_bytes(Ctx& c, Value* k_row, Value* n_ch, int bn_shift) {
    Value* swz = c.b.build_and(k_row, c.i32(7));
    return c.b.build_or(c.b.build_shl(k_row, c.i32(bn_shift)),
                        c.b.build_shl(c.b.build_xor(n_ch, swz), c.i32(4)));
}

Value* ptr_2d_i64(Ctx& c, Value* base, Value* row, Value* ld, Value* col) {
    Value* elem = c.b.build_add(
        c.b.build_mul(c.b.build_zext_i64(row), c.b.build_zext_i64(ld)),
        c.b.build_zext_i64(col));
    return c.b.build_add(base, c.b.build_shl(elem, c.kb.const_i64(1)));
}

Value* pack_pair_i32(Ctx& c, Dtype dt, Value* v0, Value* v1) {
    if (dt == Dtype::FP16) return c.kb.pack_f16x2(v0, v1);
    Value* b0 = c.b.build_and(c.kb.f32_to_bf16(v0), c.i32(0xffff));
    Value* b1 = c.b.build_shl(c.kb.f32_to_bf16(v1), c.i32(16));
    return c.b.build_or(b0, b1);
}

}  // namespace

Function* build_gemm_mma_nn_mir(Module& mod, const GemmMmaNnSpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_gemm_mma_nn_mir: only FP16/BF16 supported");
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
    const int bn_ch = bn / 8;
    const int bn_ch_shift = (bn == 128) ? 4 : 3;
    const int bn_row_shift = (bn == 128) ? 8 : 7;  // log2(bn * 2 bytes)
    const int b_chunks = (static_cast<int>(kNnBk) * bn_ch) / threads;
    const int num_accs = mt * nt;
    const int a_stage_bytes = bm * static_cast<int>(kNnBk) * 2;
    const int b_stage_bytes = static_cast<int>(kNnBk) * bn * 2;

    Function* fn = mod.create_function(
        "mir_gemm_mma_nn", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32()});

    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: A(0) B(1) C(2) M(3) N(4) K(5) grid_m(6)
    Value* A = p[0];
    Value* B = p[1];
    Value* C = p[2];
    Value* M = p[3];
    Value* N = p[4];
    Value* K = p[5];
    Value* grid_m = p[6];

    Value* smem = kb.shared_alloc_u16(kNnStages * (cfg.bm + cfg.bn) * kNnBk);
    Value* sA_base = c.b.build_trunc_i32(smem);
    Value* sB_base = c.b.build_add(sA_base, c.i32(static_cast<int>(kNnStages) * a_stage_bytes));

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
        Value* linear = c.b.build_add(c.b.build_mul(by, gn), bx);
        Value* per_slab = c.b.build_mul(c.i32(static_cast<int>(kNnL2GroupM)), gn);
        Value* slab = c.b.build_udiv(linear, per_slab);
        Value* in_slab = c.b.build_umod(linear, per_slab);
        Value* m_start = c.b.build_mul(slab, c.i32(static_cast<int>(kNnL2GroupM)));
        Value* slab_h = smin_i32(c, c.i32(static_cast<int>(kNnL2GroupM)), c.b.build_sub(grid_m, m_start));
        tile_n = c.b.build_udiv(in_slab, slab_h);
        tile_m = c.b.build_add(m_start, c.b.build_umod(in_slab, slab_h));
    }
    Value* m0 = c.b.build_mul(tile_m, c.i32(bm));
    Value* n0 = c.b.build_mul(tile_n, c.i32(bn));
    Value* KT = c.b.build_lshr(c.b.build_add(K, c.i32(31)), c.i32(5));

    // Precompute cp.async source/dest addresses + single-i32 guard bitmask.
    Value* a_lane_k = c.b.build_shl(c.b.build_and(tid, c.i32(3)), c.i32(3));
    Value* guard_mask = spec.aligned_mnk ? nullptr : c.i32(0);
    std::vector<Value*> a_s_off(static_cast<std::size_t>(a_chunks));
    std::vector<Value*> a_g_base(static_cast<std::size_t>(a_chunks));
    for (int i = 0; i < a_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* row = c.b.build_lshr(lin, c.i32(2));
        Value* ch = c.b.build_and(lin, c.i32(3));
        a_s_off[iu] = c.b.build_add(sA_base, swz_a_bytes(c, row, ch));
        Value* gr = c.b.build_add(m0, row);
        a_g_base[iu] = ptr_2d_i64(c, A, gr, K, c.b.build_shl(ch, c.i32(3)));
        if (!spec.aligned_mnk) {
            Value* bit = c.b.build_select(c.b.build_slt(gr, M), c.i32(1 << i), c.i32(0));
            guard_mask = c.b.build_or(guard_mask, bit);
        }
    }

    std::vector<Value*> b_s_off(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_g_base(static_cast<std::size_t>(b_chunks));
    std::vector<Value*> b_k_row(static_cast<std::size_t>(b_chunks));
    for (int i = 0; i < b_chunks; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* lin = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * threads));
        Value* k_row = c.b.build_lshr(lin, c.i32(bn_ch_shift));
        Value* n_ch = c.b.build_and(lin, c.i32(bn_ch - 1));
        b_s_off[iu] = c.b.build_add(sB_base, swz_b_bytes(c, k_row, n_ch, bn_row_shift));
        b_k_row[iu] = k_row;
        Value* gcol = c.b.build_add(n0, c.b.build_shl(n_ch, c.i32(3)));
        b_g_base[iu] = ptr_2d_i64(c, B, k_row, N, gcol);
        if (!spec.aligned_mnk) {
            Value* bit = c.b.build_select(c.b.build_slt(gcol, N), c.i32(1 << (a_chunks + i)), c.i32(0));
            guard_mask = c.b.build_or(guard_mask, bit);
        }
    }

    // Precompute ldmatrix addresses for kk=0; kk=1 is xor 32 for A and +(16*bn*2) = xor (16<<bn_row_shift) for B.
    Value* a_lane_row = c.b.build_and(lane, c.i32(15));
    Value* a_lane_ch = c.b.build_lshr(lane, c.i32(4));
    Value* wm_base = c.b.build_mul(w_m, c.i32(wm));
    std::vector<Value*> a_ld_off(static_cast<std::size_t>(mt));
    for (int i = 0; i < mt; ++i) {
        Value* row = c.b.build_add(c.b.build_add(wm_base, c.i32(i * 16)), a_lane_row);
        a_ld_off[static_cast<std::size_t>(i)] = c.b.build_add(sA_base, swz_a_bytes(c, row, a_lane_ch));
    }

    Value* mi = c.b.build_lshr(lane, c.i32(3));
    Value* bk_row0 = c.b.build_add(c.b.build_shl(c.b.build_and(mi, c.i32(1)), c.i32(3)),
                                   c.b.build_and(lane, c.i32(7)));
    Value* b_lane_nch = c.b.build_lshr(mi, c.i32(1));
    Value* wn_base_ch = c.b.build_lshr(c.b.build_mul(w_n, c.i32(wn)), c.i32(3));
    std::vector<Value*> b_ld_off(static_cast<std::size_t>(nt / 2));
    for (int j = 0; j < nt / 2; ++j) {
        Value* n_ch = c.b.build_add(c.b.build_add(wn_base_ch, c.i32(j * 2)), b_lane_nch);
        b_ld_off[static_cast<std::size_t>(j)] = c.b.build_add(
            sB_base, swz_b_bytes(c, bk_row0, n_ch, bn_row_shift));
    }
    const int b_kk1_off = 16 << bn_row_shift;  // +16 K-rows in byte offset (swz unchanged)

    Value* N64 = c.b.build_zext_i64(N);
    auto emit_load_stage = [&](int stage, Value* kt_idx) {
        Value* k0 = c.b.build_shl(kt_idx, c.i32(5));
        Value* a_k_bytes = c.b.build_shl(c.b.build_zext_i64(kt_idx), kb.const_i64(6));  // 32 elems * 2B
        Value* b_k_bytes = c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(k0), N64), kb.const_i64(1));
        Value* a_k_ok_mask = spec.aligned_mnk
            ? nullptr
            : c.b.build_select(c.b.build_slt(c.b.build_add(k0, a_lane_k), K), guard_mask, c.i32(0));
        Value* a_st_off = c.i32(stage * a_stage_bytes);
        Value* b_st_off = c.i32(stage * b_stage_bytes);

        for (int i = 0; i < a_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = (stage == 0) ? a_s_off[iu] : c.b.build_add(a_s_off[iu], a_st_off);
            Value* src = c.b.build_add(a_g_base[iu], a_k_bytes);
            if (spec.aligned_mnk) {
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src});
            } else {
                Value* ok = c.b.build_ne(c.b.build_and(a_k_ok_mask, c.i32(1 << i)), c.i32(0));
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(),
                               {dst, c.b.build_select(ok, src, A), c.b.build_select(ok, c.i32(16), c.i32(0))});
            }
        }
        for (int i = 0; i < b_chunks; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = (stage == 0) ? b_s_off[iu] : c.b.build_add(b_s_off[iu], b_st_off);
            Value* src = c.b.build_add(b_g_base[iu], b_k_bytes);
            if (spec.aligned_mnk) {
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src});
            } else {
                Value* k_ok = c.b.build_slt(c.b.build_add(k0, b_k_row[iu]), K);
                Value* n_ok = c.b.build_ne(c.b.build_and(guard_mask, c.i32(1 << (a_chunks + i))), c.i32(0));
                Value* ok = c.b.build_and(k_ok, n_ok);
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(),
                               {dst, c.b.build_select(ok, src, B), c.b.build_select(ok, c.i32(16), c.i32(0))});
            }
        }
    };

    const char* mma_sym = (spec.dt == Dtype::FP16) ? "ptx_mma_m16n8k16_f16_f32"
                                                 : "ptx_mma_m16n8k16_bf16_f32";

    auto emit_compute_stage = [&](int stage, std::vector<Value*>& accs) {
        Value* a_st_off = c.i32(stage * a_stage_bytes);
        Value* b_st_off = c.i32(stage * b_stage_bytes);
        for (int kk = 0; kk < 2; ++kk) {
            auto a_addr = [&](int i) -> Value* {
                Value* base = a_ld_off[static_cast<std::size_t>(i)];
                Value* p0 = (stage == 0) ? base : c.b.build_add(base, a_st_off);
                return (kk == 0) ? p0 : c.b.build_xor(p0, c.i32(32));
            };
            auto b_addr = [&](int j) -> Value* {
                Value* base = b_ld_off[static_cast<std::size_t>(j)];
                Value* p0 = (stage == 0) ? base : c.b.build_add(base, b_st_off);
                return (kk == 0) ? p0 : c.b.build_add(p0, c.i32(b_kk1_off));
            };

            std::vector<Value*> b0(static_cast<std::size_t>(nt));
            std::vector<Value*> b1(static_cast<std::size_t>(nt));
            for (int j = 0; j < nt / 2; ++j) {
                Value* r = c.b.build_call("ptx_ldmatrix_x4_trans", Type::i32x4(), {b_addr(j)});
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
                    accs[idx] = c.b.build_call(
                        mma_sym, Type::f32x4(),
                        {af_cur, b0[static_cast<std::size_t>(j)], b1[static_cast<std::size_t>(j)], accs[idx]});
                }
                af_cur = af_next;
            }
        }
    };

    kb.if_then(c.b.build_slt(c.i32(0), KT), [&] { emit_load_stage(0, c.i32(0)); });
    c.b.build_call("ptx_cp_async_commit", Type::void_type());
    kb.if_then(c.b.build_slt(c.i32(1), KT), [&] { emit_load_stage(1, c.i32(1)); });
    c.b.build_call("ptx_cp_async_commit", Type::void_type());

    std::vector<Value*> state_init;
    state_init.reserve(static_cast<std::size_t>(num_accs + 1));
    for (int i = 0; i < num_accs; ++i) state_init.push_back(c.b.build_vzero(Type::f32x4()));
    state_init.push_back(c.i32(0));

    Value* kt_limit = c.b.build_sub(KT, c.i32(4));
    std::vector<Value*> state = kb.for_range_reduce_n(
        c.i32(0), kt_limit, c.i32(3), state_init,
        [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
            std::vector<Value*> accs(cur.begin(), cur.begin() + num_accs);
            for (int s = 0; s < 3; ++s) {
                c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(1)});
                kb.sync();
                emit_load_stage((s + 2) % 3, c.b.build_add(kt, c.i32(s + 2)));
                c.b.build_call("ptx_cp_async_commit", Type::void_type());
                emit_compute_stage(s, accs);
            }
            accs.push_back(c.b.build_add(kt, c.i32(3)));
            return accs;
        });

    Value* kt_rem = state.back();
    std::vector<Value*> accs(state.begin(), state.begin() + num_accs);
    const std::vector<Type> acc_types(static_cast<std::size_t>(num_accs), Type::f32x4());
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
    c.b.build_call("ptx_cp_async_wait_all", Type::void_type());

    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t2 = c.b.build_shl(c.b.build_and(lane, c.i32(3)), c.i32(1));
    Value* wn_base = c.b.build_mul(w_n, c.i32(wn));
    for (int j = 0; j < nt; ++j) {
        Value* col = c.b.build_add(c.b.build_add(n0, wn_base), c.b.build_add(c.i32(j * 8), t2));
        auto emit_col = [&] {
            for (int i = 0; i < mt; ++i) {
                Value* frag = accs[static_cast<std::size_t>(i * nt + j)];
                for (int h = 0; h < 2; ++h) {
                    Value* row = c.b.build_add(c.b.build_add(m0, wm_base),
                                               c.b.build_add(c.i32(i * 16 + h * 8), g));
                    Value* v0 = c.b.build_vextract_lane(frag, 2 * h);
                    Value* v1 = c.b.build_vextract_lane(frag, 2 * h + 1);
                    auto emit_st = [&] {
                        kb.store_i32(ptr_2d_i64(c, C, row, N, col), pack_pair_i32(c, spec.dt, v0, v1));
                    };
                    if (spec.aligned_mnk) emit_st();
                    else kb.if_then(c.b.build_slt(row, M), emit_st);
                }
            }
        };
        if (spec.aligned_mnk) emit_col();
        else kb.if_then(c.b.build_slt(col, N), emit_col);
    }

    c.b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

}  // namespace brotensor::kernels
