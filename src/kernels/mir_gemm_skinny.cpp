#include "mir_gemm_skinny.h"
#include "mir_common.h"
#include "mir_math.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;      // 256
constexpr int kCtaCols = kWarps * 16;      // 128
constexpr int kChunk = 64;                 // K elements per stage (128B)
constexpr int kStages = 4;
constexpr int kTileStride = kCtaCols + 4;  // 132 floats per tile row

Value* ptr_2d_i64(Ctx& c, Value* base, Value* row, Value* ld, Value* col, int esz_shift) {
    Value* elem = c.b.build_add(
        c.b.build_mul(c.b.build_zext_i64(row), c.b.build_zext_i64(ld)),
        c.b.build_zext_i64(col));
    return c.b.build_add(base, c.b.build_shl(elem, c.kb.const_i64(esz_shift)));
}

Value* pack_i32x4(Ctx& c, Value* x0, Value* x1, Value* x2, Value* x3) {
    Value* v = c.b.build_vzero(Type::i32x4());
    v = c.b.build_vinsert_lane(v, x0, 0);
    v = c.b.build_vinsert_lane(v, x1, 1);
    v = c.b.build_vinsert_lane(v, x2, 2);
    v = c.b.build_vinsert_lane(v, x3, 3);
    return v;
}

Value* load_bias_f32(Ctx& c, Dtype dt, Value* bias_ptr, Value* n) {
    Value* raw = c.kb.load_u16(ptr_at(c, bias_ptr, n, 2));
    return (dt == Dtype::FP16) ? c.kb.f16_to_f32(raw) : c.kb.bf16_to_f32(raw);
}

Value* build_w_row(Ctx& c, Value* cb, Value* lr, Value* N, int epi) {
    if (epi == 3) {
        Value* half = c.b.build_lshr(N, c.i32(1));
        Value* j = c.b.build_add(c.b.build_mul(cb, c.i32(64)), c.b.build_and(lr, c.i32(63)));
        Value* cand = c.b.build_select(c.b.build_slt(lr, c.i32(64)), j, c.b.build_add(half, j));
        return c.b.build_select(c.b.build_slt(j, half), cand, c.i32(-1));
    }
    Value* n = c.b.build_add(c.b.build_mul(cb, c.i32(kCtaCols)), lr);
    return c.b.build_select(c.b.build_slt(n, N), n, c.i32(-1));
}

void emit_out_elem(Ctx& c, Dtype dt, Value* Y, Value* m, Value* col, Value* out_cols,
                   Value* a, Value* b, int act, int epi) {
    Value* v = nullptr;
    if (epi == 2) {
        v = c.b.build_mul(a, emit_gelu_exact(c, b));
    } else if (epi == 3) {
        v = c.b.build_mul(emit_gemm_act(c, 4, a), b);
    } else {
        v = (act != 0) ? emit_gemm_act(c, act, a) : a;
    }
    Value* dst = ptr_2d_i64(c, Y, m, out_cols, col, 1);
    if (epi == 1) {
        Value* old_raw = c.kb.load_u16(dst);
        Value* old_v = (dt == Dtype::FP16) ? c.kb.f16_to_f32(old_raw) : c.kb.bf16_to_f32(old_raw);
        v = c.b.build_add(v, old_v);
    }
    store_lanes(c, dt, 1, dst, {v});
}

}  // namespace

Function* build_gemm_skinny_mir(Module& mod, const GemmSkinnySpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_gemm_skinny_mir: only FP16/BF16 supported");
    }
    const int nt = std::clamp(static_cast<int>(spec.nt), 2, 8);
    const int rows = nt * 8;
    const int stage_bytes = (kCtaCols + rows) * 128;
    const int w_tile_bytes = kCtaCols * 128;

    Function* fn = mod.create_function(
        "mir_gemm_skinny", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32()});

    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: X(0) W(1) Y(2) bias(3) partial(4) M(5) N(6) K(7) cps(8)
    Value* X = p[0];
    Value* W = p[1];
    Value* Y = p[2];
    Value* bias = p[3];
    Value* partial = p[4];
    Value* M = p[5];
    Value* N = p[6];
    Value* K = p[7];
    Value* cps = p[8];

    Value* smem_dyn = c.b.build_call("ptx_shared_dynamic", Type::ptr());
    Value* s_base_u32 = c.b.build_trunc_i32(smem_dyn);
    Value* evict_first = c.b.build_call("ptx_l2_evict_first_policy", Type::i64());

    Value* tid = kb.tid_x();
    Value* lane = c.b.build_and(tid, c.i32(31));
    Value* warp = c.b.build_lshr(tid, c.i32(5));
    Value* g = c.b.build_lshr(lane, c.i32(2));
    Value* t = c.b.build_and(lane, c.i32(3));
    Value* cb = kb.ctaid_x();

    Value* n_chunks = c.b.build_lshr(c.b.build_add(K, c.i32(kChunk - 1)), c.i32(6));
    Value* c0 = c.b.build_mul(kb.ctaid_y(), cps);
    Value* KT = c.b.build_sub(smin_i32(c, n_chunks, c.b.build_add(c0, cps)), c0);

    auto piece_off = [&](Value* row, Value* u) -> Value* {
        Value* swz = c.b.build_shl(c.b.build_and(row, c.i32(1)), c.i32(2));
        Value* col_bytes = c.b.build_shl(c.b.build_xor(u, swz), c.i32(4));
        return c.b.build_add(c.b.build_shl(row, c.i32(7)), col_bytes);
    };

    // Precompute W cp.async addresses (4 pieces/thread) and X cp.async addresses.
    constexpr int kWIters = (kCtaCols * 8) / kThreads;  // 4
    std::vector<Value*> w_s0(kWIters), w_g0(kWIters), w_uk(kWIters), w_ok(kWIters);
    for (int i = 0; i < kWIters; ++i) {
        Value* q = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * kThreads));
        Value* row = c.b.build_lshr(q, c.i32(3));
        Value* u = c.b.build_and(q, c.i32(7));
        Value* n = build_w_row(c, cb, row, N, spec.epi);
        Value* uk = c.b.build_shl(u, c.i32(3));
        w_s0[static_cast<std::size_t>(i)] = c.b.build_add(s_base_u32, piece_off(row, u));
        w_uk[static_cast<std::size_t>(i)] = uk;
        w_ok[static_cast<std::size_t>(i)] = c.b.build_sge(n, c.i32(0));
        Value* safe_n = c.b.build_select(w_ok[static_cast<std::size_t>(i)], n, c.i32(0));
        w_g0[static_cast<std::size_t>(i)] = ptr_2d_i64(c, W, safe_n, K, uk, 1);
    }

    const int x_iters = (rows * 8 + kThreads - 1) / kThreads;  // 1 or 2
    std::vector<Value*> x_s0(static_cast<std::size_t>(x_iters));
    std::vector<Value*> x_g0(static_cast<std::size_t>(x_iters));
    std::vector<Value*> x_uk(static_cast<std::size_t>(x_iters));
    std::vector<Value*> x_ok(static_cast<std::size_t>(x_iters));
    for (int i = 0; i < x_iters; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        Value* q = (i == 0) ? tid : c.b.build_add(tid, c.i32(i * kThreads));
        Value* row = c.b.build_lshr(q, c.i32(3));
        Value* u = c.b.build_and(q, c.i32(7));
        Value* uk = c.b.build_shl(u, c.i32(3));
        x_s0[iu] = c.b.build_add(c.b.build_add(s_base_u32, c.i32(w_tile_bytes)), piece_off(row, u));
        x_uk[iu] = uk;
        x_ok[iu] = c.b.build_slt(row, M);
        Value* safe_r = c.b.build_select(x_ok[iu], row, c.i32(0));
        x_g0[iu] = ptr_2d_i64(c, X, safe_r, K, uk, 1);
    }

    auto emit_load_stage = [&](Value* stage_u32, Value* chunk_idx) {
        Value* st_off = c.b.build_mul(stage_u32, c.i32(stage_bytes));
        Value* kc = c.b.build_shl(chunk_idx, c.i32(6));
        Value* kc_bytes = c.b.build_shl(c.b.build_zext_i64(chunk_idx), kb.const_i64(7));
        for (int i = 0; i < kWIters; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            Value* dst = c.b.build_add(w_s0[iu], st_off);
            Value* ok = c.b.build_and(w_ok[iu], c.b.build_slt(c.b.build_add(kc, w_uk[iu]), K));
            Value* src = c.b.build_select(ok, c.b.build_add(w_g0[iu], kc_bytes), W);
            Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
            c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src, sz, evict_first});
        }
        for (int i = 0; i < x_iters; ++i) {
            const auto iu = static_cast<std::size_t>(i);
            const int rem = rows * 8 - i * kThreads;
            auto emit_x = [&] {
                Value* dst = c.b.build_add(x_s0[iu], st_off);
                Value* ok = c.b.build_and(x_ok[iu], c.b.build_slt(c.b.build_add(kc, x_uk[iu]), K));
                Value* src = c.b.build_select(ok, c.b.build_add(x_g0[iu], kc_bytes), X);
                Value* sz = c.b.build_select(ok, c.i32(16), c.i32(0));
                c.b.build_call("ptx_cp_async_cg_16", Type::void_type(), {dst, src, sz});
            };
            if (rem < kThreads) {
                kb.if_then(c.b.build_slt(tid, c.i32(rem)), emit_x);
            } else {
                emit_x();
            }
        }
    };

    for (int s = 0; s < kStages - 1; ++s) {
        kb.if_then(c.b.build_slt(c.i32(s), KT), [&] {
            emit_load_stage(c.i32(s), c.b.build_add(c0, c.i32(s)));
        });
        c.b.build_call("ptx_cp_async_commit", Type::void_type());
    }

    Value* ra = c.b.build_add(c.b.build_shl(warp, c.i32(4)), g);
    Value* rb = c.b.build_add(ra, c.i32(8));
    Value* ra8 = c.b.build_shl(ra, c.i32(3));
    Value* rb8 = c.b.build_shl(rb, c.i32(3));
    Value* g_swz = c.b.build_shl(c.b.build_and(g, c.i32(1)), c.i32(2));
    Value* t_q0 = c.b.build_xor(t, g_swz);
    Value* t_q1 = c.b.build_xor(c.b.build_add(t, c.i32(4)), g_swz);

    std::vector<Value*> acc_init(static_cast<std::size_t>(nt));
    for (int j = 0; j < nt; ++j) acc_init[static_cast<std::size_t>(j)] = c.b.build_vzero(Type::f32x4());

    const char* mma_sym = (spec.dt == Dtype::FP16) ? "ptx_mma_m16n8k16_f16_f32"
                                                 : "ptx_mma_m16n8k16_bf16_f32";

    std::vector<Value*> acc = kb.for_range_reduce_n(
        c.i32(0), KT, c.i32(1), acc_init,
        [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
            std::vector<Value*> next = cur;
            c.b.build_call("ptx_cp_async_wait_group", Type::void_type(), {c.i32(kStages - 2)});
            kb.sync();
            Value* nk = c.b.build_add(kt, c.i32(kStages - 1));
            kb.if_then(c.b.build_slt(nk, KT), [&] {
                emit_load_stage(c.b.build_and(nk, c.i32(3)), c.b.build_add(c0, nk));
            });
            c.b.build_call("ptx_cp_async_commit", Type::void_type());

            Value* st = c.b.build_and(kt, c.i32(3));
            Value* s_ptr = c.b.build_add(smem_dyn, c.b.build_zext_i64(c.b.build_mul(st, c.i32(stage_bytes))));
            Value* xs_ptr = c.b.build_add(s_ptr, kb.const_i64(w_tile_bytes));

            for (int q = 0; q < 2; ++q) {
                Value* u_swz = (q == 0) ? t_q0 : t_q1;
                Value* wa = c.b.build_call("ptx_shared_load_i32x4_indexed", Type::i32x4(),
                                           {s_ptr, c.b.build_or(ra8, u_swz)});
                Value* wb = c.b.build_call("ptx_shared_load_i32x4_indexed", Type::i32x4(),
                                           {s_ptr, c.b.build_or(rb8, u_swz)});
                Value* a0 = pack_i32x4(c, c.b.build_vextract_lane(wa, 0), c.b.build_vextract_lane(wb, 0),
                                       c.b.build_vextract_lane(wa, 1), c.b.build_vextract_lane(wb, 1));
                Value* a1 = pack_i32x4(c, c.b.build_vextract_lane(wa, 2), c.b.build_vextract_lane(wb, 2),
                                       c.b.build_vextract_lane(wa, 3), c.b.build_vextract_lane(wb, 3));
                for (int j = 0; j < nt; ++j) {
                    const auto ju = static_cast<std::size_t>(j);
                    Value* x_idx = c.b.build_or(
                        c.b.build_shl(c.b.build_add(g, c.i32(j * 8)), c.i32(3)), u_swz);
                    Value* xv = c.b.build_call("ptx_shared_load_i32x4_indexed", Type::i32x4(),
                                               {xs_ptr, x_idx});
                    next[ju] = c.b.build_call(
                        mma_sym, Type::f32x4(),
                        {a0, c.b.build_vextract_lane(xv, 0), c.b.build_vextract_lane(xv, 1), next[ju]});
                    next[ju] = c.b.build_call(
                        mma_sym, Type::f32x4(),
                        {a1, c.b.build_vextract_lane(xv, 2), c.b.build_vextract_lane(xv, 3), next[ju]});
                }
            }
            return next;
        });

    c.b.build_call("ptx_cp_async_wait_all", Type::void_type());
    kb.sync();

    // Spill-free accumulator -> shared [m][lr] FP32 tile (stride 132).
    Value* t2 = c.b.build_shl(t, c.i32(1));
    for (int j = 0; j < nt; ++j) {
        Value* frag = acc[static_cast<std::size_t>(j)];
        Value* m0_r = c.b.build_add(c.i32(j * 8), t2);
        Value* r0_off = c.b.build_mul(m0_r, c.i32(kTileStride));
        Value* r1_off = c.b.build_add(r0_off, c.i32(kTileStride));
        kb.shared_store_f32_indexed(smem_dyn, c.b.build_add(r0_off, ra), c.b.build_vextract_lane(frag, 0));
        kb.shared_store_f32_indexed(smem_dyn, c.b.build_add(r1_off, ra), c.b.build_vextract_lane(frag, 1));
        kb.shared_store_f32_indexed(smem_dyn, c.b.build_add(r0_off, rb), c.b.build_vextract_lane(frag, 2));
        kb.shared_store_f32_indexed(smem_dyn, c.b.build_add(r1_off, rb), c.b.build_vextract_lane(frag, 3));
    }
    kb.sync();

    if (spec.has_partial) {
        Value* mn64 = c.b.build_mul(c.b.build_zext_i64(M), c.b.build_zext_i64(N));
        Value* p_base = c.b.build_add(
            partial, c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(kb.ctaid_y()), mn64), kb.const_i64(2)));
        Value* total = c.b.build_mul(M, c.i32(kCtaCols));
        kb.for_range(tid, total, c.i32(kThreads), [&](Value* idx) {
            Value* m = c.b.build_lshr(idx, c.i32(7));
            Value* lr = c.b.build_and(idx, c.i32(127));
            Value* n = build_w_row(c, cb, lr, N, spec.epi);
            kb.if_then(c.b.build_sge(n, c.i32(0)), [&] {
                Value* v = kb.shared_load_f32_indexed(
                    smem_dyn, c.b.build_add(c.b.build_mul(m, c.i32(kTileStride)), lr));
                kb.store_f32(ptr_2d_i64(c, p_base, m, N, n, 2), v, 0);
            });
        });
    } else if (spec.epi <= 1u) {
        Value* total = c.b.build_mul(M, c.i32(kCtaCols));
        kb.for_range(tid, total, c.i32(kThreads), [&](Value* idx) {
            Value* m = c.b.build_lshr(idx, c.i32(7));
            Value* lr = c.b.build_and(idx, c.i32(127));
            Value* n = build_w_row(c, cb, lr, N, spec.epi);
            kb.if_then(c.b.build_sge(n, c.i32(0)), [&] {
                Value* v = kb.shared_load_f32_indexed(
                    smem_dyn, c.b.build_add(c.b.build_mul(m, c.i32(kTileStride)), lr));
                if (spec.has_bias) v = c.b.build_add(v, load_bias_f32(c, spec.dt, bias, n));
                emit_out_elem(c, spec.dt, Y, m, n, N, v, nullptr, spec.act, spec.epi);
            });
        });
    } else {
        constexpr int H = kCtaCols / 2;  // 64
        Value* total = c.b.build_mul(M, c.i32(H));
        Value* out_cols = c.b.build_lshr(N, c.i32(1));
        kb.for_range(tid, total, c.i32(kThreads), [&](Value* idx) {
            Value* m = c.b.build_lshr(idx, c.i32(6));
            Value* pp = c.b.build_and(idx, c.i32(63));
            Value* row_off = c.b.build_mul(m, c.i32(kTileStride));
            if (spec.epi == 2u) {
                Value* n = c.b.build_add(c.b.build_mul(cb, c.i32(kCtaCols)), c.b.build_shl(pp, c.i32(1)));
                kb.if_then(c.b.build_slt(c.b.build_add(n, c.i32(1)), N), [&] {
                    Value* p2 = c.b.build_add(row_off, c.b.build_shl(pp, c.i32(1)));
                    Value* va = kb.shared_load_f32_indexed(smem_dyn, p2);
                    Value* vb = kb.shared_load_f32_indexed(smem_dyn, c.b.build_add(p2, c.i32(1)));
                    if (spec.has_bias) {
                        va = c.b.build_add(va, load_bias_f32(c, spec.dt, bias, n));
                        vb = c.b.build_add(vb, load_bias_f32(c, spec.dt, bias, c.b.build_add(n, c.i32(1))));
                    }
                    emit_out_elem(c, spec.dt, Y, m, c.b.build_lshr(n, c.i32(1)), out_cols, va, vb, 0, 2);
                });
            } else {
                Value* j = build_w_row(c, cb, pp, N, 3);
                kb.if_then(c.b.build_sge(j, c.i32(0)), [&] {
                    Value* rp = c.b.build_add(row_off, pp);
                    Value* va = kb.shared_load_f32_indexed(smem_dyn, rp);
                    Value* vb = kb.shared_load_f32_indexed(smem_dyn, c.b.build_add(rp, c.i32(H)));
                    if (spec.has_bias) {
                        va = c.b.build_add(va, load_bias_f32(c, spec.dt, bias, j));
                        vb = c.b.build_add(vb, load_bias_f32(c, spec.dt, bias, c.b.build_add(out_cols, j)));
                    }
                    emit_out_elem(c, spec.dt, Y, m, j, out_cols, va, vb, 0, 3);
                });
            }
        });
    }

    c.b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

}  // namespace brotensor::kernels
