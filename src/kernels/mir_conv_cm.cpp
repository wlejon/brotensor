#include "mir_conv_cm.h"

#include "mir_common.h"
#include "mir_coop_common.h"
#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

using jit::gpu_mir::bits_to_f32;

Value* conv_act(Ctx& c, std::uint32_t act, Value* v) {
    return emit_gemm_act(c, act, v);
}

Value* bf16_bits_to_f16_bits(Ctx& c, Value* b16) {
    return c.kb.f32_to_f16(bits_to_f32(c, c.b.build_shl(b16, c.i32(16))));
}

Value* load8_a_scalar(Ctx& c, Dtype dt, Value* base, Value* k_tot, Value* row, Value* row_ok, Value* col) {
    Value* any_ok = c.b.build_and(row_ok, c.b.build_ult(col, k_tot));
    return emit_if_else_values(
        c, any_ok, {Type::i32x4()},
        [&]() -> std::vector<Value*> {
            Value* idx = c.b.build_add(c.b.build_mul(row, k_tot), col);
            std::array<Value*, 8> v{};
            for (int e = 0; e < 8; ++e) {
                Value* ok = c.b.build_ult(c.b.build_add(col, c.i32(e)), k_tot);
                v[static_cast<std::size_t>(e)] = emit_if_else_values(
                    c, ok, {Type::f32()},
                    [&]() -> std::vector<Value*> {
                        return {load_elem_dt(c, dt, base, c.b.build_add(idx, c.i32(e)))};
                    },
                    [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
            }
            return {emit_pack_f32x8_to_f16_v4(c, v)};
        },
        [&]() -> std::vector<Value*> { return {emit_v4_zero(c)}; })[0];
}

}  // namespace

brass::Function* build_conv_cm_mir(brass::Module& mod, const ConvCmSpec& s) {
    if (s.cfg_idx >= 4u) throw std::runtime_error("brotensor: mir_conv_cm: invalid cfg_idx");
    const GemmCmTile& cfg = kGemmCmTiles[s.cfg_idx];
    const std::uint32_t BM = cfg.bm, BN = cfg.bn, BK = cfg.bk, WM = cfg.wm, WN = cfg.wn, WG = cfg.wg;
    const std::uint32_t TM = WM / 16u, TN = WN / 16u;
    const std::uint32_t NSG = (BM / WM) * (BN / WN);
    const std::uint32_t LA = (BM * BK / 8u) / WG;
    const std::uint32_t LB = (BN * BK / 8u) / WG;
    const std::uint32_t CA = BK / 8u;
    const std::uint32_t CB = BN / 8u;
    const std::uint32_t SKA = CA + 1u;
    const std::uint32_t SKB = CB + 1u;
    const std::uint32_t ASZ = BM * SKA;
    const std::uint32_t BSZ = BK * SKB;
    const int KHW = s.kh * s.kw;

    const std::string name = std::string("mir_conv_cm_c") + std::to_string(s.cfg_idx) +
                             (s.dt == Dtype::FP16 ? "_f16" : "_bf16") +
                             "_bv" + std::to_string(s.bvec) + "_qa" + std::to_string(s.qa) +
                             "_a" + std::to_string(s.act) + (s.avec ? "_av" : "") +
                             (s.direct ? "_d" : "") + (s.has_bias ? "_bi" : "") +
                             (s.rowch ? "_rc" : "") + (s.accum ? "_ac" : "") +
                             "_k" + std::to_string(s.kh) + "x" + std::to_string(s.kw) +
                             "_s" + std::to_string(s.sh) + "x" + std::to_string(s.sw) +
                             "_p" + std::to_string(s.ph) + "x" + std::to_string(s.pw) +
                             "_d" + std::to_string(s.dh) + "x" + std::to_string(s.dw);

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: 0:x 1:w 2:y 3:bias 4:scale
    // ints: 5:m 6:n 7:k 8:h 9:wd 10:wo 11:cin 12:cout 13:groups 14:tm0 15:tn0

    Value* As = kb.shared_alloc_i32x4(2u * ASZ);
    Value* Bs = kb.shared_alloc_i32x4(2u * BSZ);

    Value* tid = c.b.build_and(kb.tid_x(), c.i32(static_cast<int>(WG - 1u)));
    Value* sg = c.b.build_and(kb.subgroup_id(), c.i32(static_cast<int>(NSG - 1u)));
    Value* lane = c.b.build_and(kb.lane_id(), c.i32(31));

    Value* z = kb.ctaid_z();
    Value* img = c.b.build_udiv(z, p[13]);
    Value* grp = c.b.build_umod(z, p[13]);
    Value* P = p[6];
    Value* cgi = c.b.build_udiv(p[11], p[13]);
    Value* x_ch_off = c.b.build_add(c.b.build_mul(img, p[11]), c.b.build_mul(grp, cgi));
    Value* hw_in = c.b.build_mul(c.b.build_zext_i64(p[8]), c.b.build_zext_i64(p[9]));
    Value* X = c.b.build_add(p[0], c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(x_ch_off), hw_in), kb.const_i64(1)));
    Value* w_off_elems = c.b.build_mul(c.b.build_mul(c.b.build_zext_i64(grp), c.b.build_zext_i64(p[5])),
                                       c.b.build_zext_i64(p[7]));
    Value* A = c.b.build_add(p[1], (s.qa != 0u) ? w_off_elems : c.b.build_shl(w_off_elems, kb.const_i64(1)));
    Value* y_ch_off = c.b.build_add(c.b.build_mul(img, p[12]), c.b.build_mul(grp, p[5]));
    Value* C = c.b.build_add(p[2], c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(y_ch_off), c.b.build_zext_i64(P)),
                                                 kb.const_i64(1)));
    Value* BIASP = ptr_at(c, p[3], c.b.build_mul(grp, p[5]), 2);

    Value* m0 = c.b.build_mul(c.b.build_add(kb.ctaid_y(), p[14]), c.i32(static_cast<int>(BM)));
    Value* n0 = c.b.build_mul(c.b.build_add(kb.ctaid_x(), p[15]), c.i32(static_cast<int>(BN)));
    Value* wm0 = c.b.build_mul(c.b.build_umod(sg, c.i32(static_cast<int>(BM / WM))), c.i32(static_cast<int>(WM)));
    Value* wn0 = c.b.build_mul(c.b.build_udiv(sg, c.i32(static_cast<int>(BM / WM))), c.i32(static_cast<int>(WN)));

    std::vector<Value*> fa(TM), fb(TN), acc(TM * TN);
    for (std::uint32_t i = 0; i < TM; ++i) fa[i] = kb.coop_alloc_a();
    for (std::uint32_t j = 0; j < TN; ++j) fb[j] = kb.coop_alloc_b();
    for (std::uint32_t i = 0; i < TM * TN; ++i) {
        acc[i] = kb.coop_alloc_acc();
        kb.coop_fill_acc(acc[i], c.f32(0.0f));
    }

    // Precompute A and B shared offsets + fixed output pixel geometry per B chunk.
    std::vector<Value*> a_r(LA), a_q(LA), a_smem(LA);
    for (std::uint32_t j = 0; j < LA; ++j) {
        Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
        a_r[j] = c.b.build_udiv(i, c.i32(static_cast<int>(CA)));
        a_q[j] = c.b.build_umod(i, c.i32(static_cast<int>(CA)));
        a_smem[j] = c.b.build_add(c.b.build_mul(a_r[j], c.i32(static_cast<int>(SKA))), a_q[j]);
    }
    std::vector<Value*> b_r(LB), b_smem(LB), cb_p0(LB), cb_ih(LB), cb_iw(LB), cb_any_ok(LB);
    std::vector<std::array<Value*, 8>> cb_ih_e(LB), cb_iw_e(LB), cb_ok_e(LB);
    for (std::uint32_t j = 0; j < LB; ++j) {
        Value* i = (j == 0) ? tid : c.b.build_add(tid, c.i32(static_cast<int>(j * WG)));
        b_r[j] = c.b.build_udiv(i, c.i32(static_cast<int>(CB)));
        Value* q = c.b.build_umod(i, c.i32(static_cast<int>(CB)));
        b_smem[j] = c.b.build_add(c.b.build_mul(b_r[j], c.i32(static_cast<int>(SKB))), q);
        cb_p0[j] = c.b.build_add(n0, c.b.build_shl(q, c.i32(3)));
        cb_any_ok[j] = c.b.build_ult(cb_p0[j], P);
        Value* oh = c.b.build_udiv(cb_p0[j], p[10]);
        Value* ow = c.b.build_umod(cb_p0[j], p[10]);
        cb_ih[j] = c.b.build_sub(c.b.build_mul(oh, c.i32(s.sh)), c.i32(s.ph));
        cb_iw[j] = c.b.build_sub(c.b.build_mul(ow, c.i32(s.sw)), c.i32(s.pw));
        if (s.bvec == 0u && !s.rowch) {
            for (int e = 0; e < 8; ++e) {
                Value* pe = c.b.build_add(cb_p0[j], c.i32(e));
                cb_ok_e[j][static_cast<std::size_t>(e)] = c.b.build_ult(pe, P);
                Value* ohe = c.b.build_udiv(pe, p[10]);
                Value* owe = c.b.build_umod(pe, p[10]);
                cb_ih_e[j][static_cast<std::size_t>(e)] = c.b.build_sub(c.b.build_mul(ohe, c.i32(s.sh)), c.i32(s.ph));
                cb_iw_e[j][static_cast<std::size_t>(e)] = c.b.build_sub(c.b.build_mul(owe, c.i32(s.sw)), c.i32(s.pw));
            }
        }
    }

    Value* H = p[8];
    Value* W = p[9];
    Value* W8 = c.b.build_lshr(W, c.i32(3));
    Value* scale_grp = c.b.build_mul(grp, p[5]);

    struct Loaded {
        std::vector<Value*> ra, va, qka;
        std::vector<QRawMir> rqa;
        std::vector<Value*> rbv, rbv2, vb_mask, vb_sft;
        std::vector<std::array<Value*, 8>> rb_u16, rb_ok;
    };

    auto load_tiles = [&](Value* k0) -> Loaded {
        Loaded t;
        t.ra.resize(LA);
        t.va.resize(LA);
        if (s.qa != 0u) {
            t.rqa.resize(LA);
            t.qka.resize(LA);
        }
        for (std::uint32_t j = 0; j < LA; ++j) {
            Value* row = c.b.build_add(m0, a_r[j]);
            Value* col = c.b.build_add(k0, c.b.build_shl(a_q[j], c.i32(3)));
            Value* row_ok = c.b.build_ult(row, p[5]);
            if (s.qa != 0u) {
                Value* ok = c.b.build_and(row_ok, c.b.build_ult(col, p[7]));
                Value* srow = c.b.build_select(ok, row, c.i32(0));
                Value* scol = c.b.build_select(ok, col, c.i32(0));
                Value* rptr = c.b.build_add(A, c.b.build_mul(c.b.build_zext_i64(srow), c.b.build_zext_i64(p[7])));
                t.qka[j] = scol;
                t.rqa[j] = emit_qload8_mir(c, QF_INT8, s.avec, rptr, p[4], c.b.build_add(scale_grp, srow), scol, p[7]);
                t.va[j] = ok;
            } else if (s.avec) {
                Value* ok = c.b.build_and(row_ok, c.b.build_ult(col, p[7]));
                Value* idx = c.b.build_select(ok, c.b.build_add(c.b.build_mul(row, p[7]), col), c.i32(0));
                t.ra[j] = kb.vload(Type::i32x4(), ptr_at(c, A, idx, 2));
                t.va[j] = ok;
            } else {
                t.ra[j] = load8_a_scalar(c, s.dt, A, p[7], row, row_ok, col);
                t.va[j] = c.i32(1);
            }
        }

        if (s.bvec == 1u) {
            t.rbv.resize(LB);
            t.vb_mask.resize(LB);
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* kk = c.b.build_add(k0, b_r[j]);
                Value* v = c.b.build_and(c.b.build_ult(kk, p[7]), cb_any_ok[j]);
                Value* idx = c.b.build_select(v, c.b.build_add(c.b.build_mul(kk, P), cb_p0[j]), c.i32(0));
                t.rbv[j] = kb.vload(Type::i32x4(), ptr_at(c, X, idx, 2));
                t.vb_mask[j] = v;
            }
        } else if (s.bvec == 2u) {
            t.rbv.resize(LB);
            t.rbv2.resize(LB);
            t.vb_mask.resize(LB);
            t.vb_sft.resize(LB);
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* kk = c.b.build_add(k0, b_r[j]);
                Value* ic = c.b.build_udiv(kk, c.i32(KHW));
                Value* rem = c.b.build_umod(kk, c.i32(KHW));
                Value* ih = c.b.build_add(cb_ih[j], c.b.build_mul(c.b.build_udiv(rem, c.i32(s.kw)), c.i32(s.dh)));
                Value* iw0 = c.b.build_add(cb_iw[j], c.b.build_mul(c.b.build_umod(rem, c.i32(s.kw)), c.i32(s.dw)));
                Value* ih_ok = c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, H));
                Value* rok = c.b.build_and(c.b.build_ult(kk, p[7]), c.b.build_and(ih_ok, cb_any_ok[j]));
                Value* sft = c.b.build_and(iw0, c.i32(7));
                Value* q0 = c.b.build_ashr(c.b.build_sub(iw0, sft), c.i32(3));
                Value* ih_cl = smax_i32(c, ih, c.i32(0));
                Value* rowq = c.b.build_mul(c.b.build_add(c.b.build_mul(ic, H), ih_cl), W8);
                Value* v0 = c.b.build_and(rok, c.b.build_and(c.b.build_sge(q0, c.i32(0)), c.b.build_slt(q0, W8)));
                Value* q1 = c.b.build_add(q0, c.i32(1));
                Value* v1 = c.b.build_and(rok, c.b.build_and(c.b.build_sge(q1, c.i32(0)), c.b.build_slt(q1, W8)));
                t.rbv[j] = kb.vload(Type::i32x4(), ptr_at(c, X, c.b.build_select(v0, c.b.build_add(rowq, q0), c.i32(0)), 16));
                t.rbv2[j] = kb.vload(Type::i32x4(), ptr_at(c, X, c.b.build_select(v1, c.b.build_add(rowq, q1), c.i32(0)), 16));
                Value* lo = clamp_i32(c, c.b.build_neg(iw0), c.i32(0), c.i32(8));
                Value* hi = clamp_i32(c, c.b.build_sub(W, iw0), c.i32(0), c.i32(8));
                Value* m_hi = c.b.build_sub(c.b.build_shl(c.i32(1), hi), c.i32(1));
                Value* m_lo = c.b.build_sub(c.b.build_shl(c.i32(1), lo), c.i32(1));
                Value* m_bits = c.b.build_and(m_hi, c.b.build_xor(m_lo, c.i32(-1)));
                t.vb_mask[j] = c.b.build_select(c.b.build_and(rok, c.b.build_sgt(hi, lo)), m_bits, c.i32(0));
                t.vb_sft[j] = sft;
            }
        } else {
            t.rb_u16.resize(LB);
            t.rb_ok.resize(LB);
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* kk = c.b.build_add(k0, b_r[j]);
                Value* kok = c.b.build_ult(kk, p[7]);
                Value* ic = c.b.build_udiv(kk, c.i32(KHW));
                Value* rem = c.b.build_umod(kk, c.i32(KHW));
                Value* kh_off = c.b.build_mul(c.b.build_udiv(rem, c.i32(s.kw)), c.i32(s.dh));
                Value* kw_off = c.b.build_mul(c.b.build_umod(rem, c.i32(s.kw)), c.i32(s.dw));
                Value* cbase = c.b.build_mul(ic, H);
                if (s.rowch) {
                    Value* ihb = c.b.build_add(cb_ih[j], kh_off);
                    Value* iwb = c.b.build_add(cb_iw[j], kw_off);
                    Value* rok = c.b.build_and(
                        kok, c.b.build_and(cb_any_ok[j],
                                           c.b.build_and(c.b.build_sge(ihb, c.i32(0)), c.b.build_slt(ihb, H))));
                    Value* rowbase = c.b.build_mul(c.b.build_add(cbase, smax_i32(c, ihb, c.i32(0))), W);
                    for (int e = 0; e < 8; ++e) {
                        Value* iw = c.b.build_add(iwb, c.i32(e * s.sw));
                        Value* v = c.b.build_and(rok, c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, W)));
                        Value* idx = c.b.build_select(v, c.b.build_add(rowbase, iw), c.i32(0));
                        t.rb_u16[j][static_cast<std::size_t>(e)] = kb.load_u16(ptr_at(c, X, idx, 2));
                        t.rb_ok[j][static_cast<std::size_t>(e)] = v;
                    }
                } else {
                    for (int e = 0; e < 8; ++e) {
                        Value* ih = c.b.build_add(cb_ih_e[j][static_cast<std::size_t>(e)], kh_off);
                        Value* iw = c.b.build_add(cb_iw_e[j][static_cast<std::size_t>(e)], kw_off);
                        Value* v = c.b.build_and(
                            kok, c.b.build_and(
                                cb_ok_e[j][static_cast<std::size_t>(e)],
                                c.b.build_and(c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, H)),
                                              c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, W)))));
                        Value* idx = c.b.build_select(
                            v, c.b.build_add(c.b.build_mul(c.b.build_add(cbase, ih), W), iw), c.i32(0));
                        t.rb_u16[j][static_cast<std::size_t>(e)] = kb.load_u16(ptr_at(c, X, idx, 2));
                        t.rb_ok[j][static_cast<std::size_t>(e)] = v;
                    }
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
            Value* h = nullptr;
            if (s.qa != 0u) {
                h = emit_v4_select(c, t.va[j], emit_qdecode8_f16_mir(c, QF_INT8, t.rqa[j], t.qka[j]), z4);
            } else {
                h = t.ra[j];
                if (s.avec) {
                    if (s.dt == Dtype::BF16) h = emit_bf16x8_to_f16x8(c, h);
                    h = emit_v4_select(c, t.va[j], h, z4);
                }
            }
            kb.shared_store_i32x4_indexed(As, c.b.build_add(abase, a_smem[j]), h);
        }

        if (s.bvec == 1u) {
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* h = (s.dt == Dtype::BF16) ? emit_bf16x8_to_f16x8(c, t.rbv[j]) : t.rbv[j];
                h = emit_v4_select(c, t.vb_mask[j], h, z4);
                kb.shared_store_i32x4_indexed(Bs, c.b.build_add(bbase, b_smem[j]), h);
            }
        } else if (s.bvec == 2u) {
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* u[8];
                for (std::uint32_t q = 0; q < 4; ++q) {
                    u[q] = c.b.build_vextract_lane(t.rbv[j], q);
                    u[4u + q] = c.b.build_vextract_lane(t.rbv2[j], q);
                }
                Value* sft = t.vb_sft[j];
                Value* a = c.b.build_lshr(sft, c.i32(1));
                Value* odd = c.b.build_ne(c.b.build_and(sft, c.i32(1)), c.i32(0));
                Value* eq0 = c.b.build_eq(a, c.i32(0));
                Value* eq1 = c.b.build_eq(a, c.i32(1));
                Value* eq2 = c.b.build_eq(a, c.i32(2));
                Value* w[4];
                for (std::uint32_t q = 0; q < 4; ++q) {
                    Value* w0 = c.b.build_select(eq0, u[q], c.b.build_select(eq1, u[q + 1],
                                c.b.build_select(eq2, u[q + 2], u[q + 3])));
                    Value* w1 = c.b.build_select(eq0, u[q + 1], c.b.build_select(eq1, u[q + 2],
                                c.b.build_select(eq2, u[q + 3], u[q + 4])));
                    w[q] = c.b.build_select(odd, c.b.build_or(c.b.build_lshr(w0, c.i32(16)), c.b.build_shl(w1, c.i32(16))), w0);
                }
                Value* h = emit_v4_pack(c, w[0], w[1], w[2], w[3]);
                if (s.dt == Dtype::BF16) h = emit_bf16x8_to_f16x8(c, h);
                Value* m = t.vb_mask[j];
                Value* mw[4];
                for (std::uint32_t q = 0; q < 4; ++q) {
                    Value* lo_ok = c.b.build_ne(c.b.build_and(c.b.build_lshr(m, c.i32(static_cast<int>(2u * q))), c.i32(1)), c.i32(0));
                    Value* hi_ok = c.b.build_ne(c.b.build_and(c.b.build_lshr(m, c.i32(static_cast<int>(2u * q + 1u))), c.i32(1)), c.i32(0));
                    Value* mm = c.b.build_or(c.b.build_select(lo_ok, c.i32(0xffff), c.i32(0)),
                                             c.b.build_select(hi_ok, c.i32(static_cast<int>(0xffff0000u)), c.i32(0)));
                    mw[q] = c.b.build_and(c.b.build_vextract_lane(h, q), mm);
                }
                kb.shared_store_i32x4_indexed(Bs, c.b.build_add(bbase, b_smem[j]), emit_v4_pack(c, mw[0], mw[1], mw[2], mw[3]));
            }
        } else {
            for (std::uint32_t j = 0; j < LB; ++j) {
                Value* w[4];
                for (int q = 0; q < 4; ++q) {
                    Value* lo = t.rb_u16[j][static_cast<std::size_t>(2 * q)];
                    Value* hi = t.rb_u16[j][static_cast<std::size_t>(2 * q + 1)];
                    if (s.dt == Dtype::BF16) {
                        lo = bf16_bits_to_f16_bits(c, lo);
                        hi = bf16_bits_to_f16_bits(c, hi);
                    }
                    lo = c.b.build_select(t.rb_ok[j][static_cast<std::size_t>(2 * q)], lo, c.i32(0));
                    hi = c.b.build_select(t.rb_ok[j][static_cast<std::size_t>(2 * q + 1)], hi, c.i32(0));
                    w[q] = c.b.build_or(lo, c.b.build_shl(hi, c.i32(16)));
                }
                kb.shared_store_i32x4_indexed(Bs, c.b.build_add(bbase, b_smem[j]), emit_v4_pack(c, w[0], w[1], w[2], w[3]));
            }
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
                Value* off = c.b.build_add(c.b.build_mul(row_m, c.i32(static_cast<int>(SKA))), c.i32(static_cast<int>(kk * 2u)));
                kb.coop_load_a(fa[i], c.b.build_add(As, c.b.build_shl(c.b.build_add(abase, off), c.i32(4))), stride_a, false);
            }
            for (std::uint32_t j = 0; j < TN; ++j) {
                Value* col_n = c.b.build_add(wn0, c.i32(static_cast<int>(j * 16u)));
                Value* off = c.b.build_add(c.i32(static_cast<int>(kk * 16u * SKB)), c.b.build_lshr(col_n, c.i32(3)));
                kb.coop_load_b(fb[j], c.b.build_add(Bs, c.b.build_shl(c.b.build_add(bbase, off), c.i32(4))), stride_b, false);
            }
            for (std::uint32_t i = 0; i < TM; ++i) {
                for (std::uint32_t j = 0; j < TN; ++j) kb.coop_mma(acc[i * TN + j], fa[i], fb[j]);
            }
        }
    };

    Value* nk = c.b.build_lshr(c.b.build_add(p[7], c.i32(static_cast<int>(BK - 1u))), c.i32(5));
    kb.if_then(c.b.build_ugt(nk, c.i32(0)), [&] {
        store_tiles(c.i32(0), load_tiles(c.i32(0)));
        kb.sync();
        Value* nk_m1 = c.b.build_sub(nk, c.i32(1));
        kb.for_range(c.i32(0), nk_m1, c.i32(1), [&](Value* kt) {
            Value* cur = c.b.build_and(kt, c.i32(1));
            Value* nxt = c.b.build_xor(cur, c.i32(1));
            Loaded next_t = load_tiles(c.b.build_shl(c.b.build_add(kt, c.i32(1)), c.i32(5)));
            mma_tile(cur);
            store_tiles(nxt, next_t);
            kb.sync();
        });
        mma_tile(c.b.build_and(nk_m1, c.i32(1)));
    });

    if (s.direct) {
        Value* tmp_acc = (s.has_bias || s.accum) ? kb.coop_alloc_acc() : nullptr;
        for (std::uint32_t i = 0; i < TM; ++i) {
            Value* row = c.b.build_add(c.b.build_add(m0, wm0), c.i32(static_cast<int>(i * 16u)));
            Value* row_off = c.b.build_mul(row, P);
            for (std::uint32_t j = 0; j < TN; ++j) {
                Value* r = acc[i * TN + j];
                Value* col = c.b.build_add(c.b.build_add(n0, wn0), c.i32(static_cast<int>(j * 16u)));
                Value* c_ptr = ptr_at(c, C, c.b.build_add(row_off, col), 2);
                if (s.has_bias) {
                    kb.coop_load_acc(tmp_acc, ptr_at(c, BIASP, row, 2), c.i32(0), true, true);
                    kb.coop_add_acc(r, tmp_acc);
                }
                if (s.act != LACT_NONE) {
                    emit_coop_acc_map(c, r, [&](Value*, Value* rv) { return conv_act(c, s.act, rv); });
                }
                if (s.accum) {
                    kb.coop_load_acc(tmp_acc, c_ptr, P, false, true);
                    kb.coop_add_acc(r, tmp_acc);
                }
                kb.coop_store_acc(r, c_ptr, P, false, true);
            }
        }
        c.b.build_ret_void();
        return finalize_fn(fn);
    }

    kb.sync();
    Value* sbase_v4 = c.b.build_shl(sg, c.i32(7));
    Value* sbase_f = c.b.build_shl(sbase_v4, c.i32(2));
    Value* sbase_bytes = c.b.build_shl(sbase_v4, c.i32(4));
    for (std::uint32_t i = 0; i < TM; ++i) {
        Value* gm0 = c.b.build_add(c.b.build_add(m0, wm0), c.i32(static_cast<int>(i * 16u)));
        for (std::uint32_t j = 0; j < TN; ++j) {
            kb.coop_store_acc(acc[i * TN + j], c.b.build_add(As, sbase_bytes), c.i32(16), false, false);
            kb.subgroup_barrier();
            Value* gn0 = c.b.build_add(c.b.build_add(n0, wn0), c.i32(static_cast<int>(j * 16u)));
            for (int e0 = 0; e0 < 256; e0 += 32) {
                Value* e = (e0 == 0) ? lane : c.b.build_add(lane, c.i32(e0));
                Value* gm = c.b.build_add(gm0, c.b.build_lshr(e, c.i32(4)));
                Value* gn = c.b.build_add(gn0, c.b.build_and(e, c.i32(15)));
                kb.if_then(c.b.build_and(c.b.build_ult(gm, p[5]), c.b.build_ult(gn, P)), [&] {
                    Value* v = kb.shared_load_f32_from_v4(As, c.b.build_add(sbase_f, e));
                    if (s.has_bias) v = c.b.build_add(v, load_elem_dt(c, s.dt, BIASP, gm));
                    if (s.act != LACT_NONE) v = conv_act(c, s.act, v);
                    Value* o = c.b.build_add(c.b.build_mul(gm, P), gn);
                    if (s.accum) v = c.b.build_add(v, load_elem_dt(c, s.dt, C, o));
                    store_elem_dt(c, s.dt, C, o, v);
                });
            }
            kb.subgroup_barrier();
        }
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
