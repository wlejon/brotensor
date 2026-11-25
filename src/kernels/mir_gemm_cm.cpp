#include "mir_gemm_cm.h"

#include "mir_common.h"
#include "mir_coop_common.h"
#include "mir_math.h"
#include "mir_quant_helpers.h"
#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

int dt_eb(Dtype dt) { return dt == Dtype::FP32 ? 4 : 2; }

Value* batch_base(Ctx& c, Value* base, Value* stride, Value* z, int eb) {
    return ptr_at(c, base, c.b.build_mul(z, stride), eb);
}

Value* load8_scalar(Ctx& c, Dtype dt, Value* base, Value* ld, Value* row, Value* row_ok,
                    Value* col, Value* ncols, bool apply_scale, const std::array<Value*, 8>& amul) {
    Value* any_ok = c.b.build_and(row_ok, c.b.build_ult(col, ncols));
    return emit_if_else_values(
        c, any_ok, {Type::i32x4()},
        [&]() -> std::vector<Value*> {
            Value* idx = c.b.build_add(c.b.build_mul(row, ld), col);
            std::array<Value*, 8> h{};
            for (int e = 0; e < 8; ++e) {
                Value* ok = c.b.build_ult(c.b.build_add(col, c.i32(e)), ncols);
                Value* v = emit_if_else_values(
                    c, ok, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        return {load_elem_dt(c, dt, base, c.b.build_add(idx, c.i32(e)))};
                    },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
                if (apply_scale) v = c.b.build_mul(v, amul[static_cast<std::size_t>(e)]);
                h[static_cast<std::size_t>(e)] = v;
            }
            return {emit_pack_f32x8_to_f16_v4(c, h)};
        },
        [&]() -> std::vector<Value*> { return {emit_v4_zero(c)}; })[0];
}

}  // namespace

brass::Function* build_gemm_cm_mir(brass::Module& mod, const GemmCmSpec& s) {
    if (s.cfg_idx >= 4u) throw std::runtime_error("brotensor: mir_gemm_cm: invalid cfg_idx");
    const GemmCmTile& cfg = kGemmCmTiles[s.cfg_idx];
    const std::uint32_t BM = cfg.bm, BN = cfg.bn, BK = cfg.bk, WM = cfg.wm, WN = cfg.wn, WG = cfg.wg;
    const std::uint32_t TM = WM / 16u, TN = WN / 16u;
    const std::uint32_t NSG = (BM / WM) * (BN / WN);
    const std::uint32_t LA = (BM * BK / 8u) / WG;
    const std::uint32_t LB = (BN * BK / 8u) / WG;
    const std::uint32_t CA = s.ta ? (BM / 8u) : (BK / 8u);
    const std::uint32_t CB = s.nb ? (BN / 8u) : (BK / 8u);
    const std::uint32_t SKA = CA + 1u;
    const std::uint32_t SKB = CB + 1u;
    const std::uint32_t ASZ = (s.ta ? BK : BM) * SKA;
    const std::uint32_t BSZ = (s.nb ? BK : BN) * SKB;
    const Dtype dt_bias = (s.dt_a != Dtype::FP32 && s.dt_c == Dtype::FP32) ? s.dt_a : s.dt_c;

    const std::string name = std::string("mir_gemm_cm_c") + std::to_string(s.cfg_idx) +
                             "_a" + std::to_string(static_cast<int>(s.dt_a)) +
                             "_b" + std::to_string(static_cast<int>(s.dt_b)) +
                             "_o" + std::to_string(static_cast<int>(s.dt_c)) +
                             "_e" + std::to_string(s.epi) + "_x" + std::to_string(s.act) +
                             "_q" + std::to_string(s.qb) + (s.ta ? "_ta" : "") + (s.nb ? "_nb" : "") +
                             (s.vec ? "_v" : "_s") + (s.direct ? "_d" : "") +
                             (s.has_bias ? "_bi" : "") + (s.scale_a ? "_sc" : "");

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: 0:a 1:b 2:c 3:bias 4:scale 5:ascale
    // ints: 6:m 7:n 8:k 9:lda 10:ldb 11:ldc 12:sa 13:sb 14:sc 15:half_n 16:tm0 17:tn0

    Value* As = kb.shared_alloc_i32x4(2u * ASZ);
    Value* Bs = kb.shared_alloc_i32x4(2u * BSZ);
    Value* s_fp8 = (s.qb == QF_FP8) ? kb.shared_alloc_f32(256u) : nullptr;

    Value* tid = c.b.build_and(kb.tid_x(), c.i32(static_cast<int>(WG - 1u)));
    Value* sg = c.b.build_and(kb.subgroup_id(), c.i32(static_cast<int>(NSG - 1u)));
    Value* lane = c.b.build_and(kb.lane_id(), c.i32(31));

    if (s_fp8) {
        for (std::uint32_t off = 0; off < 256u; off += WG) {
            Value* idx = (off == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(off)));
            kb.shared_store_f32_indexed(s_fp8, idx, emit_fp8_e4m3_to_f32(c, idx));
        }
        kb.sync();
    }

    Value* z = kb.ctaid_z();
    Value* A = batch_base(c, p[0], p[12], z, dt_eb(s.dt_a));
    Value* B = (s.qb != 0u) ? p[1] : batch_base(c, p[1], p[13], z, 2);
    Value* C = batch_base(c, p[2], p[14], z, dt_eb(s.dt_c));
    Value* m0 = c.b.build_mul(c.b.build_add(kb.ctaid_y(), p[16]), c.i32(static_cast<int>(BM)));
    const std::uint32_t bn_step = (s.epi == EPI_SWIGLU) ? (BN / 2u) : BN;
    Value* n0 = c.b.build_mul(c.b.build_add(kb.ctaid_x(), p[17]), c.i32(static_cast<int>(bn_step)));
    Value* wm0 = c.b.build_mul(c.b.build_umod(sg, c.i32(static_cast<int>(BM / WM))), c.i32(static_cast<int>(WM)));
    Value* wn0 = c.b.build_mul(c.b.build_udiv(sg, c.i32(static_cast<int>(BM / WM))), c.i32(static_cast<int>(WN)));

    Value* AS = nullptr;
    std::vector<std::array<Value*, 8>> amul(LA);
    if (s.scale_a) {
        Value* as_off = c.b.build_select(c.b.build_ne(p[12], c.i32(0)), c.b.build_mul(z, p[6]), c.i32(0));
        AS = ptr_at(c, p[5], as_off, 4);
        for (std::uint32_t j = 0; j < LA; ++j) {
            Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
            if (!s.ta) {
                Value* r = c.b.build_udiv(i, c.i32(static_cast<int>(CA)));
                Value* row = c.b.build_add(m0, r);
                Value* ex = emit_if_else_values(
                    c, c.b.build_ult(row, p[6]), {Type::i32()},
                    [&]() -> std::vector<Value*> { return {kb.load_i32(ptr_at(c, AS, row, 4))}; },
                    [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                Value* mv = kb.ldexp_f32(c.f32(1.0f), c.b.build_neg(ex));
                for (int e = 0; e < 8; ++e) amul[j][static_cast<std::size_t>(e)] = mv;
            } else {
                Value* mb = c.b.build_add(m0, c.b.build_shl(c.b.build_umod(i, c.i32(static_cast<int>(CA))), c.i32(3)));
                for (int e = 0; e < 8; ++e) {
                    Value* row = c.b.build_add(mb, c.i32(e));
                    Value* ex = emit_if_else_values(
                        c, c.b.build_ult(row, p[6]), {Type::i32()},
                        [&]() -> std::vector<Value*> { return {kb.load_i32(ptr_at(c, AS, row, 4))}; },
                        [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                    amul[j][static_cast<std::size_t>(e)] = kb.ldexp_f32(c.f32(1.0f), c.b.build_neg(ex));
                }
            }
        }
    }

    std::vector<Value*> fa(TM), fb(TN), acc(TM * TN);
    for (std::uint32_t i = 0; i < TM; ++i) fa[i] = kb.coop_alloc_a();
    for (std::uint32_t j = 0; j < TN; ++j) fb[j] = kb.coop_alloc_b();
    for (std::uint32_t i = 0; i < TM * TN; ++i) {
        acc[i] = kb.coop_alloc_acc();
        kb.coop_fill_acc(acc[i], c.f32(0.0f));
    }

    // Precompute per-chunk shared tile offsets & row/col coordinates fixed across K.
    std::vector<Value*> a_r(LA), a_q(LA), a_smem(LA);
    for (std::uint32_t j = 0; j < LA; ++j) {
        Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
        a_r[j] = c.b.build_udiv(i, c.i32(static_cast<int>(CA)));
        a_q[j] = c.b.build_umod(i, c.i32(static_cast<int>(CA)));
        a_smem[j] = c.b.build_add(c.b.build_mul(a_r[j], c.i32(static_cast<int>(SKA))), a_q[j]);
    }
    std::vector<Value*> b_r(LB), b_q(LB), b_smem(LB), b_row(LB), b_row_ok(LB);
    for (std::uint32_t j = 0; j < LB; ++j) {
        Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
        b_r[j] = c.b.build_udiv(i, c.i32(static_cast<int>(CB)));
        b_q[j] = c.b.build_umod(i, c.i32(static_cast<int>(CB)));
        b_smem[j] = c.b.build_add(c.b.build_mul(b_r[j], c.i32(static_cast<int>(SKB))), b_q[j]);
        if (!s.nb) {
            if (s.epi == EPI_SWIGLU) {
                Value* tb = c.b.build_lshr(b_r[j], c.i32(4));
                Value* o = c.b.build_add(
                    n0, c.b.build_add(c.b.build_shl(c.b.build_lshr(tb, c.i32(1)), c.i32(4)),
                                      c.b.build_and(b_r[j], c.i32(15))));
                b_row_ok[j] = c.b.build_ult(o, p[15]);
                Value* half_off = c.b.build_select(c.b.build_ne(c.b.build_and(tb, c.i32(1)), c.i32(0)),
                                                   p[15], c.i32(0));
                b_row[j] = c.b.build_add(half_off, o);
            } else {
                b_row[j] = c.b.build_add(n0, b_r[j]);
                b_row_ok[j] = c.b.build_ult(b_row[j], p[7]);
            }
        }
    }

    struct Loaded {
        std::vector<Value*> ra, ra2, va;
        std::vector<Value*> rb, vb, qk;
        std::vector<QRawMir> rq;
    };

    auto load_tiles = [&](Value* k0) -> Loaded {
        Loaded t;
        t.ra.resize(LA);
        t.va.resize(LA);
        if (s.dt_a == Dtype::FP32 && s.vec) t.ra2.resize(LA);
        t.rb.resize(LB);
        t.vb.resize(LB);
        if (s.qb != 0u) {
            t.rq.resize(LB);
            t.qk.resize(LB);
        }
        for (std::uint32_t j = 0; j < LA; ++j) {
            Value* row = s.ta ? c.b.build_add(k0, a_r[j]) : c.b.build_add(m0, a_r[j]);
            Value* row_ok = s.ta ? c.b.build_ult(row, p[8]) : c.b.build_ult(row, p[6]);
            Value* col = c.b.build_add(s.ta ? m0 : k0, c.b.build_shl(a_q[j], c.i32(3)));
            Value* ncols = s.ta ? p[6] : p[8];
            if (s.vec) {
                Value* ok = c.b.build_and(row_ok, c.b.build_ult(col, ncols));
                Value* idx = c.b.build_select(ok, c.b.build_add(c.b.build_mul(row, p[9]), col), c.i32(0));
                if (s.dt_a == Dtype::FP32) {
                    Value* base_ptr = ptr_at(c, A, idx, 4);
                    t.ra[j] = kb.vload(Type::i32x4(), base_ptr);
                    t.ra2[j] = kb.vload(Type::i32x4(), c.b.build_add(base_ptr, kb.const_i64(16)));
                } else {
                    t.ra[j] = kb.vload(Type::i32x4(), ptr_at(c, A, idx, 2));
                }
                t.va[j] = ok;
            } else {
                t.ra[j] = load8_scalar(c, s.dt_a, A, p[9], row, row_ok, col, ncols, s.scale_a, amul[j]);
                t.va[j] = c.i32(1);
            }
        }
        if (s.qb != 0u) {
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* col = c.b.build_add(k0, c.b.build_shl(b_q[j], c.i32(3)));
                Value* ok = c.b.build_and(b_row_ok[j], c.b.build_ult(col, p[8]));
                Value* srow = c.b.build_select(ok, b_row[j], c.i32(0));
                Value* scol = c.b.build_select(ok, col, c.i32(0));
                Value* row_ptr = c.b.build_add(B, c.b.build_mul(c.b.build_zext_i64(srow), c.b.build_zext_i64(p[10])));
                t.vb[j] = ok;
                t.qk[j] = scol;
                t.rq[j] = emit_qload8_mir(c, s.qb, s.vec, row_ptr, p[4], srow, scol, p[8]);
            }
        } else {
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* row = s.nb ? c.b.build_add(k0, b_r[j]) : b_row[j];
                Value* row_ok = s.nb ? c.b.build_ult(row, p[8]) : b_row_ok[j];
                Value* col = c.b.build_add(s.nb ? n0 : k0, c.b.build_shl(b_q[j], c.i32(3)));
                Value* ncols = s.nb ? p[7] : p[8];
                if (s.vec) {
                    Value* ok = c.b.build_and(row_ok, c.b.build_ult(col, ncols));
                    Value* idx = c.b.build_select(ok, c.b.build_add(c.b.build_mul(row, p[10]), col), c.i32(0));
                    t.rb[j] = kb.vload(Type::i32x4(), ptr_at(c, B, idx, 2));
                    t.vb[j] = ok;
                } else {
                    std::array<Value*, 8> dummy{};
                    t.rb[j] = load8_scalar(c, s.dt_b, B, p[10], row, row_ok, col, ncols, false, dummy);
                    t.vb[j] = c.i32(1);
                }
            }
        }
        return t;
    };

    auto store_tiles = [&](Value* buf, const Loaded& t) {
        Value* b01 = c.b.build_and(buf, c.i32(1));
        Value* abase = c.b.build_mul(b01, c.i32(static_cast<int>(ASZ)));
        Value* bbase = c.b.build_mul(b01, c.i32(static_cast<int>(BSZ)));
        Value* z4 = emit_v4_zero(c);
        for (std::uint32_t j = 0; j < LA; ++j) {
            Value* h = t.ra[j];
            if (s.vec) {
                if (s.dt_a == Dtype::FP32) {
                    h = emit_f32x8_to_f16x8_scaled(c, t.ra[j], t.ra2[j], amul[j]);
                } else if (s.dt_a == Dtype::BF16) {
                    h = s.scale_a ? emit_bf16x8_to_f16x8_scaled(c, h, amul[j]) : emit_bf16x8_to_f16x8(c, h);
                }
                h = emit_v4_select(c, t.va[j], h, z4);
            }
            kb.shared_store_i32x4_indexed(As, c.b.build_add(abase, a_smem[j]), h);
        }
        for (std::uint32_t j = 0; j < LB; ++j) {
            Value* h = nullptr;
            if (s.qb != 0u) {
                h = emit_v4_select(c, t.vb[j], emit_qdecode8_f16_mir(c, s.qb, t.rq[j], t.qk[j], s_fp8), z4);
            } else {
                h = t.rb[j];
                if (s.vec) {
                    if (s.dt_b == Dtype::BF16) h = emit_bf16x8_to_f16x8(c, h);
                    h = emit_v4_select(c, t.vb[j], h, z4);
                }
            }
            kb.shared_store_i32x4_indexed(Bs, c.b.build_add(bbase, b_smem[j]), h);
        }
    };

    Value* stride_a = c.i32(static_cast<int>(SKA * 8u));
    Value* stride_b = c.i32(static_cast<int>(SKB * 8u));
    auto mma_tile = [&](Value* buf) {
        Value* b01 = c.b.build_and(buf, c.i32(1));
        Value* abase = c.b.build_mul(b01, c.i32(static_cast<int>(ASZ)));
        Value* bbase = c.b.build_mul(b01, c.i32(static_cast<int>(BSZ)));
        for (std::uint32_t kk = 0; kk < BK / 16u; ++kk) {
            for (std::uint32_t i = 0; i < TM; ++i) {
                Value* row_m = c.b.build_add(wm0, c.i32(static_cast<int>(i * 16u)));
                Value* off = s.ta
                    ? c.b.build_add(c.i32(static_cast<int>(kk * 16u * SKA)), c.b.build_lshr(row_m, c.i32(3)))
                    : c.b.build_add(c.b.build_mul(row_m, c.i32(static_cast<int>(SKA))), c.i32(static_cast<int>(kk * 2u)));
                Value* boff = c.b.build_shl(c.b.build_add(abase, off), c.i32(4));
                kb.coop_load_a(fa[i], c.b.build_add(As, boff), stride_a, s.ta);
            }
            for (std::uint32_t j = 0; j < TN; ++j) {
                Value* col_n = c.b.build_add(wn0, c.i32(static_cast<int>(j * 16u)));
                Value* off = s.nb
                    ? c.b.build_add(c.i32(static_cast<int>(kk * 16u * SKB)), c.b.build_lshr(col_n, c.i32(3)))
                    : c.b.build_add(c.b.build_mul(col_n, c.i32(static_cast<int>(SKB))), c.i32(static_cast<int>(kk * 2u)));
                Value* boff = c.b.build_shl(c.b.build_add(bbase, off), c.i32(4));
                kb.coop_load_b(fb[j], c.b.build_add(Bs, boff), stride_b, !s.nb);
            }
            for (std::uint32_t i = 0; i < TM; ++i) {
                for (std::uint32_t j = 0; j < TN; ++j) {
                    kb.coop_mma(acc[i * TN + j], fa[i], fb[j]);
                }
            }
        }
    };

    Value* nk = c.b.build_lshr(c.b.build_add(p[8], c.i32(static_cast<int>(BK - 1u))), c.i32(5));
    kb.if_then(c.b.build_ugt(nk, c.i32(0)), [&] {
        Value* krot_raw = c.b.build_mul(
            c.b.build_add(kb.ctaid_x(), c.b.build_mul(kb.ctaid_y(), c.i32(5))), c.i32(37));
        Value* krot = c.b.build_umod(krot_raw, nk);
        store_tiles(c.i32(0), load_tiles(c.b.build_shl(krot, c.i32(5))));
        kb.sync();

        Value* nk_m1 = c.b.build_sub(nk, c.i32(1));
        kb.for_range(c.i32(0), nk_m1, c.i32(1), [&](Value* kt) {
            Value* cur = c.b.build_and(kt, c.i32(1));
            Value* nxt = c.b.build_xor(cur, c.i32(1));
            Value* knxt = c.b.build_shl(c.b.build_umod(c.b.build_add(c.b.build_add(kt, c.i32(1)), krot), nk),
                                        c.i32(5));
            Loaded next_t = load_tiles(knxt);
            mma_tile(cur);
            store_tiles(nxt, next_t);
            kb.sync();
        });
        mma_tile(c.b.build_and(nk_m1, c.i32(1)));
    });

    const std::uint32_t step = (s.epi == EPI_SWIGLU) ? 2u : 1u;

    if (s.direct) {
        Value* tmp_acc = (s.has_bias || s.epi == EPI_ACCUM) ? kb.coop_alloc_acc() : nullptr;
        for (std::uint32_t i = 0; i < TM; ++i) {
            Value* gm = c.b.build_add(c.b.build_add(m0, wm0), c.i32(static_cast<int>(i * 16u)));
            Value* row_off = c.b.build_mul(gm, p[11]);
            for (std::uint32_t j = 0; j < TN; j += step) {
                Value* r = acc[i * TN + j];
                Value* col = (s.epi == EPI_SWIGLU)
                    ? c.b.build_add(n0, c.b.build_shl(
                        c.b.build_lshr(c.b.build_add(c.b.build_lshr(wn0, c.i32(4)), c.i32(static_cast<int>(j))), c.i32(1)),
                        c.i32(4)))
                    : c.b.build_add(c.b.build_add(n0, wn0), c.i32(static_cast<int>(j * 16u)));
                Value* c_ptr = ptr_at(c, C, c.b.build_add(row_off, col), 2);
                if (s.has_bias) {
                    kb.coop_load_acc(tmp_acc, ptr_at(c, p[3], col, 2), c.i32(0), false, true);
                    kb.coop_add_acc(r, tmp_acc);
                }
                if (s.epi == EPI_SWIGLU) {
                    Value* u = acc[i * TN + j + 1u];
                    if (s.has_bias) {
                        kb.coop_load_acc(tmp_acc, ptr_at(c, p[3], c.b.build_add(p[15], col), 2), c.i32(0), false, true);
                        kb.coop_add_acc(u, tmp_acc);
                    }
                    emit_coop_acc_map(c, r, [&](Value* idx, Value* rv) {
                        Value* uv = kb.coop_get_acc(u, idx);
                        return c.b.build_mul(c.b.build_mul(rv, emit_sigmoid(c, rv)), uv);
                    });
                } else if (s.act != LACT_NONE) {
                    emit_coop_acc_map(c, r, [&](Value*, Value* rv) { return emit_gemm_act(c, s.act, rv); });
                }
                if (s.epi == EPI_ACCUM) {
                    kb.coop_load_acc(tmp_acc, c_ptr, p[11], false, true);
                    kb.coop_add_acc(r, tmp_acc);
                }
                kb.coop_store_acc(r, c_ptr, p[11], false, true);
            }
        }
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    // Shared-scratch epilogue aliasing As (2 x 256 floats = 128 i32x4 per subgroup).
    kb.sync();
    Value* sbase_v4 = c.b.build_shl(sg, c.i32(7));       // sg * 128 (in i32x4)
    Value* sbase_f = c.b.build_shl(sbase_v4, c.i32(2));  // sg * 512 (in f32)
    Value* sbase_bytes = c.b.build_shl(sbase_v4, c.i32(4));
    Value* sbase2_bytes = c.b.build_shl(c.b.build_add(sbase_v4, c.i32(64)), c.i32(4));
    Value* sbase2_f = c.b.build_add(sbase_f, c.i32(256));

    auto row_mul = [&](Value* gm) -> Value* {
        if (!s.scale_a) return c.f32(1.0f);
        return kb.ldexp_f32(c.f32(1.0f), kb.load_i32(ptr_at(c, AS, gm, 4)));
    };
    auto bias_at = [&](Value* col) -> Value* {
        if (!s.has_bias) return c.f32(0.0f);
        return load_elem_dt(c, dt_bias, p[3], col);
    };

    for (std::uint32_t i = 0; i < TM; ++i) {
        Value* gm0 = c.b.build_add(c.b.build_add(m0, wm0), c.i32(static_cast<int>(i * 16u)));
        for (std::uint32_t j = 0; j < TN; j += step) {
            kb.coop_store_acc(acc[i * TN + j], c.b.build_add(As, sbase_bytes), c.i32(16), false, false);
            if (s.epi == EPI_SWIGLU) {
                kb.coop_store_acc(acc[i * TN + j + 1u], c.b.build_add(As, sbase2_bytes), c.i32(16), false, false);
            }
            kb.subgroup_barrier();
            if (s.epi == EPI_GEGLU) {
                for (int e0 = 0; e0 < 128; e0 += 32) {
                    Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
                    Value* rr = c.b.build_lshr(e, c.i32(3));
                    Value* pr = c.b.build_and(e, c.i32(7));
                    Value* gm = c.b.build_add(gm0, rr);
                    Value* nr = c.b.build_add(
                        c.b.build_add(n0, wn0),
                        c.b.build_add(c.i32(static_cast<int>(j * 16u)), c.b.build_shl(pr, c.i32(1))));
                    Value* ok = c.b.build_and(c.b.build_ult(gm, p[6]), c.b.build_ult(c.b.build_add(nr, c.i32(1)), p[7]));
                    kb.if_then(ok, [&] {
                        Value* sc = row_mul(gm);
                        Value* fidx = c.b.build_add(sbase_f, c.b.build_shl(e, c.i32(1)));
                        Value* av = c.b.build_add(c.b.build_mul(kb.shared_load_f32_from_v4(As, fidx), sc), bias_at(nr));
                        Value* gv = c.b.build_add(
                            c.b.build_mul(kb.shared_load_f32_from_v4(As, c.b.build_add(fidx, c.i32(1))), sc),
                            bias_at(c.b.build_add(nr, c.i32(1))));
                        Value* out_idx = c.b.build_add(c.b.build_mul(gm, p[11]), c.b.build_lshr(nr, c.i32(1)));
                        store_elem_dt(c, s.dt_c, C, out_idx, c.b.build_mul(av, emit_gelu_exact(c, gv)));
                    });
                }
            } else if (s.epi == EPI_SWIGLU) {
                Value* ob = c.b.build_lshr(c.b.build_add(c.b.build_lshr(wn0, c.i32(4)), c.i32(static_cast<int>(j))), c.i32(1));
                Value* oc0 = c.b.build_add(n0, c.b.build_shl(ob, c.i32(4)));
                for (int e0 = 0; e0 < 256; e0 += 32) {
                    Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
                    Value* gm = c.b.build_add(gm0, c.b.build_lshr(e, c.i32(4)));
                    Value* oc = c.b.build_add(oc0, c.b.build_and(e, c.i32(15)));
                    Value* ok = c.b.build_and(c.b.build_ult(gm, p[6]), c.b.build_ult(oc, p[15]));
                    kb.if_then(ok, [&] {
                        Value* sc = row_mul(gm);
                        Value* gv = c.b.build_add(
                            c.b.build_mul(kb.shared_load_f32_from_v4(As, c.b.build_add(sbase_f, e)), sc),
                            bias_at(oc));
                        Value* uv = c.b.build_add(
                            c.b.build_mul(kb.shared_load_f32_from_v4(As, c.b.build_add(sbase2_f, e)), sc),
                            bias_at(c.b.build_add(p[15], oc)));
                        Value* out_idx = c.b.build_add(c.b.build_mul(gm, p[11]), oc);
                        store_elem_dt(c, s.dt_c, C, out_idx, c.b.build_mul(c.b.build_mul(gv, emit_sigmoid(c, gv)), uv));
                    });
                }
            } else {
                Value* nc0 = c.b.build_add(c.b.build_add(n0, wn0), c.i32(static_cast<int>(j * 16u)));
                for (int e0 = 0; e0 < 256; e0 += 32) {
                    Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
                    Value* gm = c.b.build_add(gm0, c.b.build_lshr(e, c.i32(4)));
                    Value* nc = c.b.build_add(nc0, c.b.build_and(e, c.i32(15)));
                    Value* ok = c.b.build_and(c.b.build_ult(gm, p[6]), c.b.build_ult(nc, p[7]));
                    kb.if_then(ok, [&] {
                        Value* v = kb.shared_load_f32_from_v4(As, c.b.build_add(sbase_f, e));
                        if (s.scale_a) v = c.b.build_mul(v, row_mul(gm));
                        if (s.has_bias) v = c.b.build_add(v, bias_at(nc));
                        if (s.act != LACT_NONE) v = emit_gemm_act(c, s.act, v);
                        Value* o = c.b.build_add(c.b.build_mul(gm, p[11]), nc);
                        if (s.epi == EPI_ACCUM) v = c.b.build_add(v, load_elem_dt(c, s.dt_c, C, o));
                        store_elem_dt(c, s.dt_c, C, o, v);
                    });
                }
            }
            kb.subgroup_barrier();
        }
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
