#include "mir_gnorm.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <vector>

namespace brotensor::kernels {

namespace {

std::array<Value*, 3> chan_combine(Ctx& c, Value* na, Value* ma, Value* qa,
                                   Value* nb, Value* mb, Value* qb) {
    Value* n_sum = c.b.build_add(na, nb);
    Value* has_b = c.b.build_sgt(nb, c.f32(0.0f));
    Value* has_a = c.b.build_sgt(na, c.f32(0.0f));
    Value* f = c.b.build_sdiv(nb, c.kb.fmax(n_sum, c.f32(1e-30f)));
    Value* d = c.b.build_sub(mb, ma);
    Value* ma_b = c.b.build_fma_f32(d, f, ma);
    Value* qa_b = c.b.build_fma_f32(c.b.build_mul(d, d), c.b.build_mul(na, f), c.b.build_add(qb, qa));
    Value* mn = c.b.build_select(has_b, c.b.build_select(has_a, n_sum, nb), na);
    Value* mm = c.b.build_select(has_b, c.b.build_select(has_a, ma_b, mb), ma);
    Value* mq = c.b.build_select(has_b, c.b.build_select(has_a, qa_b, qb), qa);
    return {mn, mm, mq};
}

std::array<Value*, 2> emit_tile_stats(Ctx& c, Value* part, Value* tile_idx, Value* splits, Value* eps) {
    Value* s_n = c.kb.shared_alloc_f32(kGnormBlock);
    Value* s_m = c.kb.shared_alloc_f32(kGnormBlock);
    Value* s_q = c.kb.shared_alloc_f32(kGnormBlock);
    Value* tid = c.kb.tid_x();
    Value* ts = c.b.build_mul(tile_idx, splits);

    auto red = c.kb.for_range_reduce_n(
        tid, splits, c.i32(static_cast<int>(kGnormBlock)),
        {c.f32(0.0f), c.f32(0.0f), c.f32(0.0f)},
        [&](Value* s, const std::vector<Value*>& cur) {
            Value* o = c.b.build_mul(c.b.build_add(ts, s), c.i32(3));
            Value* nb = c.kb.load_f32(ptr_at(c, part, o, 4));
            Value* mb = c.kb.load_f32(ptr_at(c, part, c.b.build_add(o, c.i32(1)), 4));
            Value* qb = c.kb.load_f32(ptr_at(c, part, c.b.build_add(o, c.i32(2)), 4));
            auto nxt = chan_combine(c, cur[0], cur[1], cur[2], nb, mb, qb);
            return std::vector<Value*>{nxt[0], nxt[1], nxt[2]};
        });
    c.kb.shared_store_f32_indexed(s_n, tid, red[0]);
    c.kb.shared_store_f32_indexed(s_m, tid, red[1]);
    c.kb.shared_store_f32_indexed(s_q, tid, red[2]);
    c.kb.sync();

    for (std::uint32_t s = kGnormBlock >> 1; s > 0; s >>= 1) {
        c.kb.if_then(c.b.build_ult(tid, c.i32(static_cast<int>(s))), [&] {
            Value* other = c.b.build_add(tid, c.i32(static_cast<int>(s)));
            auto cmb = chan_combine(
                c,
                c.kb.shared_load_f32_indexed(s_n, tid),
                c.kb.shared_load_f32_indexed(s_m, tid),
                c.kb.shared_load_f32_indexed(s_q, tid),
                c.kb.shared_load_f32_indexed(s_n, other),
                c.kb.shared_load_f32_indexed(s_m, other),
                c.kb.shared_load_f32_indexed(s_q, other));
            c.kb.shared_store_f32_indexed(s_n, tid, cmb[0]);
            c.kb.shared_store_f32_indexed(s_m, tid, cmb[1]);
            c.kb.shared_store_f32_indexed(s_q, tid, cmb[2]);
        });
        c.kb.sync();
    }
    Value* mean = c.kb.shared_load_f32_indexed(s_m, c.i32(0));
    Value* n0 = c.kb.fmax(c.kb.shared_load_f32_indexed(s_n, c.i32(0)), c.f32(1.0f));
    Value* q0 = c.kb.fmax(c.kb.shared_load_f32_indexed(s_q, c.i32(0)), c.f32(0.0f));
    Value* var = c.b.build_add(c.b.build_sdiv(q0, n0), eps);
    Value* rstd = emit_rsqrt(c, c.kb.fmax(var, c.f32(1e-12f)));
    c.kb.sync();
    return {mean, rstd};
}

void build_gn_stats(Ctx& c, const std::vector<Value*>& p, Dtype dt, int lanes) {
    const int esz = elem_bytes(dt);
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kGnormBlock));
    Value* tile = c.kb.ctaid_y();
    Value* s = c.kb.ctaid_x();
    Value* base = c.b.build_mul(tile, p[7]);
    Value* lo = c.b.build_mul(s, p[12]);
    Value* hi = c.b.build_select(c.b.build_ult(c.b.build_add(lo, p[12]), p[7]),
                                 c.b.build_add(lo, p[12]), p[7]);
    Value* K = load_lanes(c, dt, 1, ptr_at(c, p[0], c.b.build_add(base, lo), esz))[0];
    Value* s_red = c.kb.shared_alloc_f32(8);

    Value* start = c.b.build_add(vec_count(c, lo, lanes), tid);
    Value* end = vec_count(c, hi, lanes);
    auto sums = c.kb.for_range_reduce_n(start, end, block, {c.f32(0.0f), c.f32(0.0f)},
                                        [&](Value* iv, const std::vector<Value*>& cur) {
        Value* idx = c.b.build_add(base, col_elem_idx(c, iv, lanes));
        auto xs = load_lanes(c, dt, lanes, ptr_at(c, p[0], idx, esz));
        Value* a1 = cur[0];
        Value* a2 = cur[1];
        for (int j = 0; j < lanes; ++j) {
            Value* d = c.b.build_sub(xs[static_cast<std::size_t>(j)], K);
            a1 = c.b.build_add(a1, d);
            a2 = c.b.build_fma_f32(d, d, a2);
        }
        return std::vector<Value*>{a1, a2};
    });
    Value* s1 = emit_block_sum_f32(c, s_red, sums[0]);
    Value* s2 = emit_block_sum_f32(c, s_red, sums[1]);
    Value* cnt = c.kb.fmax(c.kb.i32_to_f32(c.b.build_sub(hi, lo)), c.f32(1.0f));
    Value* s1_over_cnt = c.b.build_sdiv(s1, cnt);
    Value* mean = c.b.build_add(K, s1_over_cnt);
    Value* q = c.kb.fmax(c.b.build_sub(s2, c.b.build_mul(s1, s1_over_cnt)), c.f32(0.0f));
    c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
        Value* o = c.b.build_mul(c.b.build_add(c.b.build_mul(tile, p[11]), s), c.i32(3));
        c.kb.store_f32(ptr_at(c, p[4], o, 4), cnt);
        c.kb.store_f32(ptr_at(c, p[4], c.b.build_add(o, c.i32(1)), 4), mean);
        c.kb.store_f32(ptr_at(c, p[4], c.b.build_add(o, c.i32(2)), 4), q);
    });
}

void build_gn_apply(Ctx& c, const std::vector<Value*>& p, Dtype dt, bool silu, int lanes) {
    const int esz = elem_bytes(dt);
    Value* tile = c.kb.ctaid_y();
    auto st = emit_tile_stats(c, p[4], tile, p[11], p[15]);
    Value* mean = st[0];
    Value* rstd = st[1];
    Value* base = c.b.build_mul(tile, p[7]);
    Value* c0 = c.b.build_mul(c.b.build_umod(tile, p[10]), p[9]);
    Value* hw_v = vec_count(c, p[8], lanes);
    emit_vec_grid_stride(c, p[7], lanes, kGnormBlock, [&](Value* iv) {
        Value* ch = c.b.build_add(c0, c.b.build_udiv(iv, hw_v));
        Value* ga = c.b.build_mul(load_lanes(c, dt, 1, ptr_at(c, p[2], ch, esz))[0], rstd);
        Value* be = load_lanes(c, dt, 1, ptr_at(c, p[3], ch, esz))[0];
        Value* idx = c.b.build_add(base, col_elem_idx(c, iv, lanes));
        auto xs = load_lanes(c, dt, lanes, ptr_at(c, p[0], idx, esz));
        std::vector<Value*> ys(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            Value* v = c.b.build_fma_f32(c.b.build_sub(xs[static_cast<std::size_t>(j)], mean), ga, be);
            if (silu) v = c.b.build_mul(v, emit_sigmoid(c, v));
            ys[static_cast<std::size_t>(j)] = v;
        }
        store_lanes(c, dt, lanes, ptr_at(c, p[1], idx, esz), ys);
    });
}

void build_gn_bwd_ch(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    const int esz = elem_bytes(dt);
    Value* ch = c.kb.ctaid_x();
    Value* img = c.kb.ctaid_y();
    Value* tid = c.kb.tid_x();
    Value* tile = c.b.build_add(c.b.build_mul(img, p[10]), c.b.build_udiv(ch, p[9]));
    auto st = emit_tile_stats(c, p[4], tile, p[11], p[15]);
    Value* s_red = c.kb.shared_alloc_f32(8);
    Value* base = c.b.build_mul(c.b.build_add(c.b.build_mul(img, p[14]), ch), p[8]);
    auto sums = c.kb.for_range_reduce_n(
        tid, p[8], c.i32(static_cast<int>(kGnormBlock)), {c.f32(0.0f), c.f32(0.0f)},
        [&](Value* i, const std::vector<Value*>& cur) {
            Value* idx = c.b.build_add(base, i);
            Value* dy = load_lanes(c, dt, 1, ptr_at(c, p[3], idx, esz))[0];
            Value* xv = load_lanes(c, dt, 1, ptr_at(c, p[0], idx, esz))[0];
            Value* xh = c.b.build_mul(c.b.build_sub(xv, st[0]), st[1]);
            return std::vector<Value*>{c.b.build_add(cur[0], dy), c.b.build_fma_f32(dy, xh, cur[1])};
        });
    Value* s1 = emit_block_sum_f32(c, s_red, sums[0]);
    Value* s2 = emit_block_sum_f32(c, s_red, sums[1]);
    c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
        Value* o = c.b.build_shl(c.b.build_add(c.b.build_mul(img, p[14]), ch), c.i32(1));
        c.kb.store_f32(ptr_at(c, p[5], o, 4), s1);
        c.kb.store_f32(ptr_at(c, p[5], c.b.build_add(o, c.i32(1)), 4), s2);
        c.kb.if_then(c.b.build_eq(c.b.build_umod(ch, p[9]), c.i32(0)), [&] {
            Value* t2 = c.b.build_shl(tile, c.i32(1));
            c.kb.store_f32(ptr_at(c, p[6], t2, 4), st[0]);
            c.kb.store_f32(ptr_at(c, p[6], c.b.build_add(t2, c.i32(1)), 4), st[1]);
        });
    });
}

void build_gn_bwd_dx(Ctx& c, const std::vector<Value*>& p, Dtype dt) {
    const int esz = elem_bytes(dt);
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kGnormBlock));
    Value* tile = c.kb.ctaid_y();
    Value* img = c.b.build_udiv(tile, p[10]);
    Value* c0 = c.b.build_mul(c.b.build_umod(tile, p[10]), p[9]);
    Value* s_red = c.kb.shared_alloc_f32(8);
    auto sums = c.kb.for_range_reduce_n(tid, p[9], block, {c.f32(0.0f), c.f32(0.0f)},
                                        [&](Value* k, const std::vector<Value*>& cur) {
        Value* ck = c.b.build_add(c0, k);
        Value* o = c.b.build_shl(c.b.build_add(c.b.build_mul(img, p[14]), ck), c.i32(1));
        Value* gv = load_lanes(c, dt, 1, ptr_at(c, p[2], ck, esz))[0];
        Value* r0 = c.kb.load_f32(ptr_at(c, p[5], o, 4));
        Value* r1 = c.kb.load_f32(ptr_at(c, p[5], c.b.build_add(o, c.i32(1)), 4));
        return std::vector<Value*>{c.b.build_fma_f32(gv, r0, cur[0]), c.b.build_fma_f32(gv, r1, cur[1])};
    });
    Value* inv_m = c.b.build_sdiv(c.f32(1.0f), c.kb.fmax(c.kb.i32_to_f32(p[7]), c.f32(1.0f)));
    Value* sum1 = c.b.build_mul(emit_block_sum_f32(c, s_red, sums[0]), inv_m);
    Value* sum2 = c.b.build_mul(emit_block_sum_f32(c, s_red, sums[1]), inv_m);
    Value* t2 = c.b.build_shl(tile, c.i32(1));
    Value* mean = c.kb.load_f32(ptr_at(c, p[6], t2, 4));
    Value* rstd = c.kb.load_f32(ptr_at(c, p[6], c.b.build_add(t2, c.i32(1)), 4));
    Value* base = c.b.build_mul(tile, p[7]);
    Value* gid_x = c.kb.global_tid_x();
    Value* stride = c.kb.mul(c.kb.nctaid_x(), block);
    c.kb.for_range(gid_x, p[7], stride, [&](Value* i) {
        Value* ch = c.b.build_add(c0, c.b.build_udiv(i, p[8]));
        Value* idx = c.b.build_add(base, i);
        Value* xv = load_lanes(c, dt, 1, ptr_at(c, p[0], idx, esz))[0];
        Value* dy = load_lanes(c, dt, 1, ptr_at(c, p[3], idx, esz))[0];
        Value* gv = load_lanes(c, dt, 1, ptr_at(c, p[2], ch, esz))[0];
        Value* xh = c.b.build_mul(c.b.build_sub(xv, mean), rstd);
        Value* dxh = c.b.build_mul(dy, gv);
        Value* corr = c.b.build_fma_f32(xh, sum2, sum1);
        store_lanes(c, dt, 1, ptr_at(c, p[1], idx, esz),
                    {c.b.build_mul(rstd, c.b.build_sub(dxh, corr))});
    });
}

void build_gn_misc(Ctx& c, const std::vector<Value*>& p, std::uint32_t mode, Dtype dt) {
    const int esz = elem_bytes(dt);
    Value* block = c.i32(static_cast<int>(kGnormBlock));
    if (mode == GN_BWD_PARAM) {
        emit_vec_grid_stride(c, p[14], 1, kGnormBlock, [&](Value* ch) {
            auto sums = c.kb.for_range_reduce_n(c.i32(0), p[13], c.i32(1), {c.f32(0.0f), c.f32(0.0f)},
                                                [&](Value* img, const std::vector<Value*>& cur) {
                Value* o = c.b.build_shl(c.b.build_add(c.b.build_mul(img, p[14]), ch), c.i32(1));
                Value* v1 = c.kb.load_f32(ptr_at(c, p[5], o, 4));
                Value* v2 = c.kb.load_f32(ptr_at(c, p[5], c.b.build_add(o, c.i32(1)), 4));
                return std::vector<Value*>{c.b.build_add(cur[0], v1), c.b.build_add(cur[1], v2)};
            });
            Value* yp = ptr_at(c, p[1], ch, esz);
            Value* gp = ptr_at(c, p[2], ch, esz);
            store_lanes(c, dt, 1, yp, {c.b.build_add(load_lanes(c, dt, 1, yp)[0], sums[1])});
            store_lanes(c, dt, 1, gp, {c.b.build_add(load_lanes(c, dt, 1, gp)[0], sums[0])});
        });
    } else if (mode == GN_BN_INFER) {
        emit_vec_grid_stride(c, p[13], 1, kGnormBlock, [&](Value* i) {
            Value* ch = c.b.build_umod(c.b.build_udiv(i, p[8]), p[14]);
            Value* gv = load_lanes(c, dt, 1, ptr_at(c, p[2], ch, esz))[0];
            Value* bv = load_lanes(c, dt, 1, ptr_at(c, p[3], ch, esz))[0];
            Value* rmv = load_lanes(c, dt, 1, ptr_at(c, p[5], ch, esz))[0];
            Value* rvv = load_lanes(c, dt, 1, ptr_at(c, p[6], ch, esz))[0];
            Value* a = c.b.build_mul(gv, emit_rsqrt(c, c.kb.fmax(c.b.build_add(rvv, p[15]), c.f32(1e-12f))));
            Value* bb = c.b.build_sub(bv, c.b.build_mul(rmv, a));
            Value* xv = load_lanes(c, dt, 1, ptr_at(c, p[0], i, esz))[0];
            store_lanes(c, dt, 1, ptr_at(c, p[1], i, esz), {c.b.build_fma_f32(xv, a, bb)});
        });
    } else if (mode == GN_BIAS_GRAD) {
        Value* ch = c.b.build_add(c.b.build_mul(c.kb.ctaid_y(), c.i32(65535)), c.kb.ctaid_x());
        c.kb.if_then(c.b.build_ult(ch, p[14]), [&] {
            Value* tid = c.kb.tid_x();
            Value* s_red = c.kb.shared_alloc_f32(8);
            Value* s_acc = c.kb.for_range_reduce(c.i32(0), p[13], c.i32(1), c.f32(0.0f),
                                                 [&](Value* img, Value* cur) {
                Value* base = c.b.build_mul(c.b.build_add(c.b.build_mul(img, p[14]), ch), p[8]);
                return c.kb.for_range_reduce(tid, p[8], block, cur, [&](Value* i, Value* inner) {
                    Value* xv = load_lanes(c, dt, 1, ptr_at(c, p[0], c.b.build_add(base, i), esz))[0];
                    return c.b.build_add(inner, xv);
                });
            });
            Value* tot = emit_block_sum_f32(c, s_red, s_acc);
            c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                Value* yp = ptr_at(c, p[1], ch, esz);
                store_lanes(c, dt, 1, yp, {c.b.build_add(load_lanes(c, dt, 1, yp)[0], tot)});
            });
        });
    } else {  // GN_L2_NCHW
        Value* eps_safe = c.kb.fmax(p[15], c.f32(1e-30f));
        emit_vec_grid_stride(c, p[13], 1, kGnormBlock, [&](Value* i) {
            Value* img = c.b.build_udiv(i, p[8]);
            Value* px = c.b.build_umod(i, p[8]);
            Value* base = c.b.build_add(c.b.build_mul(c.b.build_mul(img, p[14]), p[8]), px);
            Value* ss = c.kb.for_range_reduce(c.i32(0), p[14], c.i32(1), c.f32(0.0f), [&](Value* ch, Value* cur) {
                Value* v = load_lanes(c, dt, 1, ptr_at(c, p[0], c.b.build_add(base, c.b.build_mul(ch, p[8])), esz))[0];
                return c.b.build_fma_f32(v, v, cur);
            });
            Value* inv = c.b.build_sdiv(c.f32(1.0f), c.kb.fmax(emit_sqrt(c, c.kb.fmax(ss, c.f32(0.0f))), eps_safe));
            c.kb.for_range(c.i32(0), p[14], c.i32(1), [&](Value* ch) {
                Value* o = c.b.build_add(base, c.b.build_mul(ch, p[8]));
                Value* v = load_lanes(c, dt, 1, ptr_at(c, p[0], o, esz))[0];
                store_lanes(c, dt, 1, ptr_at(c, p[1], o, esz), {c.b.build_mul(v, inv)});
            });
        });
    }
}

}  // namespace

Function* build_gnorm_mir(Module& mod, std::uint32_t mode, Dtype dt, bool silu, int lanes) {
    Function* fn = mod.create_function(
        "mir_gnorm", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    if (mode == GN_STATS)       build_gn_stats(c, p, dt, lanes);
    else if (mode == GN_APPLY)  build_gn_apply(c, p, dt, silu, lanes);
    else if (mode == GN_BWD_CH) build_gn_bwd_ch(c, p, dt);
    else if (mode == GN_BWD_DX) build_gn_bwd_dx(c, p, dt);
    else                        build_gn_misc(c, p, mode, dt);
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_bnorm_mir(Module& mod, std::uint32_t mode) {
    Function* fn = mod.create_function(
        "mir_bnorm", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* tid = kb.tid_x();
    Value* block = c.i32(static_cast<int>(kGnormBlock));
    Value* s_red = kb.shared_alloc_f32(8);
    Value* N = p[11];
    Value* C = p[12];
    Value* hw = p[13];
    Value* M = c.b.build_mul(N, hw);
    Value* M_safe = kb.fmax(kb.i32_to_f32(M), c.f32(1.0f));
    Value* invM = c.b.build_sdiv(c.f32(1.0f), M_safe);

    auto idx_at = [&](Value* ch, Value* j) -> Value* {
        Value* n = c.b.build_udiv(j, hw);
        Value* px = c.b.build_sub(j, c.b.build_mul(n, hw));
        return c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(n, C), ch), hw), px);
    };

    kb.for_range(kb.ctaid_x(), C, kb.nctaid_x(), [&](Value* ch) {
        if (mode == BN_TRAIN_FWD) {
            Value* sum = kb.for_range_reduce(tid, M, block, c.f32(0.0f), [&](Value* j, Value* a) {
                return c.b.build_add(a, kb.load_f32(ptr_at(c, p[0], idx_at(ch, j), 4)));
            });
            Value* mean = c.b.build_mul(emit_block_sum_f32(c, s_red, sum), invM);
            Value* sq = kb.for_range_reduce(tid, M, block, c.f32(0.0f), [&](Value* j, Value* a) {
                Value* d = c.b.build_sub(kb.load_f32(ptr_at(c, p[0], idx_at(ch, j), 4)), mean);
                return c.b.build_fma_f32(d, d, a);
            });
            Value* var_b = c.b.build_mul(emit_block_sum_f32(c, s_red, sq), invM);
            Value* rstd = emit_rsqrt(c, kb.fmax(c.b.build_add(var_b, p[14]), c.f32(1e-12f)));
            Value* mom = p[15];
            Value* one_m = c.b.build_sub(c.f32(1.0f), mom);
            Value* m_sub1 = c.b.build_select(c.b.build_ugt(M, c.i32(1)), c.b.build_sub(M, c.i32(1)), c.i32(1));
            Value* bessel = c.b.build_select(
                c.b.build_ugt(M, c.i32(1)),
                c.b.build_sdiv(M_safe, kb.i32_to_f32(m_sub1)),
                c.f32(1.0f));
            kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                kb.store_f32(ptr_at(c, p[6], ch, 4), mean);
                kb.store_f32(ptr_at(c, p[7], ch, 4), rstd);
                Value* rm_p = ptr_at(c, p[4], ch, 4);
                Value* rv_p = ptr_at(c, p[5], ch, 4);
                kb.store_f32(rm_p, c.b.build_fma_f32(one_m, kb.load_f32(rm_p), c.b.build_mul(mom, mean)));
                kb.store_f32(rv_p, c.b.build_fma_f32(one_m, kb.load_f32(rv_p),
                                                     c.b.build_mul(mom, c.b.build_mul(var_b, bessel))));
            });
            Value* gv = kb.load_f32(ptr_at(c, p[2], ch, 4));
            Value* bv = kb.load_f32(ptr_at(c, p[3], ch, 4));
            kb.for_range(tid, M, block, [&](Value* j) {
                Value* i = idx_at(ch, j);
                Value* xh = c.b.build_mul(c.b.build_sub(kb.load_f32(ptr_at(c, p[0], i, 4)), mean), rstd);
                kb.store_f32(ptr_at(c, p[1], i, 4), c.b.build_fma_f32(xh, gv, bv));
            });
        } else {
            Value* mean = kb.load_f32(ptr_at(c, p[6], ch, 4));
            Value* rstd = kb.load_f32(ptr_at(c, p[7], ch, 4));
            Value* gv = kb.load_f32(ptr_at(c, p[2], ch, 4));
            auto sums = kb.for_range_reduce_n(tid, M, block, {c.f32(0.0f), c.f32(0.0f)},
                                              [&](Value* j, const std::vector<Value*>& cur) {
                Value* i = idx_at(ch, j);
                Value* dyv = kb.load_f32(ptr_at(c, p[8], i, 4));
                Value* xh = c.b.build_mul(c.b.build_sub(kb.load_f32(ptr_at(c, p[0], i, 4)), mean), rstd);
                return std::vector<Value*>{c.b.build_add(cur[0], dyv), c.b.build_fma_f32(dyv, xh, cur[1])};
            });
            Value* t0 = emit_block_sum_f32(c, s_red, sums[0]);
            Value* t1 = emit_block_sum_f32(c, s_red, sums[1]);
            kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                Value* dgp = ptr_at(c, p[9], ch, 4);
                Value* dbp = ptr_at(c, p[10], ch, 4);
                kb.store_f32(dgp, c.b.build_add(kb.load_f32(dgp), t1));
                kb.store_f32(dbp, c.b.build_add(kb.load_f32(dbp), t0));
            });
            Value* s1 = c.b.build_mul(gv, t0);
            Value* s2 = c.b.build_mul(gv, t1);
            kb.for_range(tid, M, block, [&](Value* j) {
                Value* i = idx_at(ch, j);
                Value* xh = c.b.build_mul(c.b.build_sub(kb.load_f32(ptr_at(c, p[0], i, 4)), mean), rstd);
                Value* dxh = c.b.build_mul(kb.load_f32(ptr_at(c, p[8], i, 4)), gv);
                Value* corr = c.b.build_mul(c.b.build_fma_f32(xh, s2, s1), invM);
                kb.store_f32(ptr_at(c, p[1], i, 4), c.b.build_mul(rstd, c.b.build_sub(dxh, corr)));
            });
        }
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
