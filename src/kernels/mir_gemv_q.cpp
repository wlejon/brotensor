#include "mir_gemv_q.h"

#include "mir_common.h"
#include "mir_math.h"
#include "mir_quant_helpers.h"
#include "../vulkan/shaders/op_codes.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

std::vector<Value*> load_x_slice8(Ctx& c, Dtype dt, Value* base, Value* elem_idx, int count, bool aligned_vec) {
    if (aligned_vec && count == 8) {
        return load_lanes(c, dt, 8, ptr_at(c, base, elem_idx, 2));
    }
    std::vector<Value*> out(static_cast<std::size_t>(count));
    for (int e = 0; e < count; ++e) {
        out[static_cast<std::size_t>(e)] = load_elem_dt(c, dt, base, c.b.build_add(elem_idx, c.i32(e)));
    }
    return out;
}

std::vector<Value*> load_u16_bytes(Ctx& c, Value* ptr, int nu16) {
    std::vector<Value*> bytes(static_cast<std::size_t>(nu16 * 2));
    for (int i = 0; i < nu16; ++i) {
        Value* p_i = (i == 0) ? ptr : c.b.build_add(ptr, c.kb.const_i64(i * 2));
        Value* u = c.kb.load_u16(p_i);
        bytes[static_cast<std::size_t>(2 * i)] = c.b.build_and(u, c.i32(0xff));
        bytes[static_cast<std::size_t>(2 * i + 1)] = c.b.build_lshr(u, c.i32(8));
    }
    return bytes;
}

}  // namespace

brass::Function* build_gemv_q_mir(brass::Module& mod, Dtype dt, std::uint32_t fmt,
                                  std::uint32_t mb, std::uint32_t lpr, bool vec, bool has_bias,
                                  std::uint32_t unr, std::uint32_t nr,
                                  std::uint32_t act, std::uint32_t epi) {
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("brotensor: mir_gemv_q: only FP16 and BF16 activations are supported");
    }
    if (mb < 1u || mb > 8u || (lpr != 16u && lpr != 32u && lpr != 64u) || nr < 1u || nr > 8u) {
        throw std::runtime_error("brotensor: mir_gemv_q: invalid geometry");
    }
    const bool glu = (epi == EPI_GEGLU || epi == EPI_SWIGLU);
    if (glu && (nr < 2u || (nr & 1u) != 0u)) {
        throw std::runtime_error("brotensor: mir_gemv_q: GLU epilogue requires even nr >= 2");
    }
    const bool byte_fmt = (fmt == QF_INT8 || fmt == QF_FP8);
    if (!byte_fmt && fmt != QF_Q8_0 && fmt != QF_Q4K && fmt != QF_Q6K) {
        throw std::runtime_error("brotensor: mir_gemv_q: unknown quant format");
    }
    const std::uint32_t eff_unr = (byte_fmt && !vec) ? 1u : std::max(1u, unr);
    const std::string name = std::string("mir_gemv_q_") + (dt == Dtype::FP16 ? "f16" : "bf16") +
                             "_fmt" + std::to_string(fmt) + "_m" + std::to_string(mb) +
                             "_l" + std::to_string(lpr) + (vec ? "_v" : "_s") +
                             (has_bias ? "_b" : "") + "_u" + std::to_string(eff_unr) +
                             "_r" + std::to_string(nr) + "_a" + std::to_string(act) +
                             "_e" + std::to_string(epi);

    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p[0..4] = x, w, y, bias, scale; p[5..9] = n, k, ldx, ldy, rowbytes

    const std::uint32_t ng = 256u / lpr;
    Value* s_part = (lpr == 64u) ? kb.shared_alloc_f32(ng * 2u * nr * mb) : nullptr;
    Value* tid = kb.tid_x();
    Value* unit = c.b.build_udiv(tid, c.i32(static_cast<int>(lpr)));
    Value* lane = c.b.build_umod(tid, c.i32(static_cast<int>(lpr)));
    Value* wg = c.b.build_add(kb.ctaid_x(), c.b.build_mul(kb.ctaid_y(), kb.nctaid_x()));
    Value* rowgroup = c.b.build_add(c.b.build_mul(wg, c.i32(static_cast<int>(ng))), unit);

    std::vector<Value*> wrow(nr);
    std::vector<Value*> row_ok(nr);
    if (epi == EPI_SWIGLU) {
        Value* half_n = c.b.build_lshr(p[5], c.i32(1));
        for (std::uint32_t q = 0; q < nr / 2u; ++q) {
            Value* oc = c.b.build_add(c.b.build_mul(rowgroup, c.i32(static_cast<int>(nr / 2u))),
                                      c.i32(static_cast<int>(q)));
            Value* ok = c.b.build_ult(oc, half_n);
            wrow[2u * q] = oc;
            wrow[2u * q + 1u] = c.b.build_add(oc, half_n);
            row_ok[2u * q] = ok;
            row_ok[2u * q + 1u] = ok;
        }
    } else {
        for (std::uint32_t r = 0; r < nr; ++r) {
            wrow[r] = c.b.build_add(c.b.build_mul(rowgroup, c.i32(static_cast<int>(nr))),
                                    c.i32(static_cast<int>(r)));
            row_ok[r] = c.b.build_ult(wrow[r], p[5]);
        }
    }

    std::vector<Value*> acc_init(nr * mb, c.f32(0.0f));
    std::vector<Value*> acc = emit_if_else_values(
        c, row_ok[0], std::vector<Type>(nr * mb, Type::f32()),
        [&]() -> std::vector<Value*> {
            std::vector<Value*> row_ptrs(nr);
            for (std::uint32_t r = 0; r < nr; ++r) {
                Value* safe_r = (r == 0) ? wrow[0] : c.b.build_select(row_ok[r], wrow[r], wrow[0]);
                row_ptrs[r] = c.b.build_add(
                    p[1], c.b.build_mul(c.b.build_zext_i64(safe_r), c.b.build_zext_i64(p[9])));
            }

            if (byte_fmt && !vec) {
                std::vector<Value*> res = kb.for_range_reduce_n(
                    lane, p[6], c.i32(static_cast<int>(lpr)), acc_init,
                    [&](Value* col, const std::vector<Value*>& cur) -> std::vector<Value*> {
                        std::vector<Value*> next = cur;
                        std::vector<Value*> xv(mb);
                        for (std::uint32_t b = 0; b < mb; ++b) {
                            xv[b] = load_elem_dt(c, dt, p[0], c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[7]), col));
                        }
                        for (std::uint32_t r = 0; r < nr; ++r) {
                            Value* qb = kb.load_u8(c.b.build_add(row_ptrs[r], c.b.build_zext_i64(col)));
                            Value* wv = (fmt == QF_FP8) ? emit_fp8_e4m3_to_f32(c, qb) : emit_s8_byte_to_f32(c, qb);
                            for (std::uint32_t b = 0; b < mb; ++b) {
                                std::size_t idx = r * mb + b;
                                next[idx] = c.b.build_fma_f32(xv[b], wv, next[idx]);
                            }
                        }
                        return next;
                    });
                for (std::uint32_t r = 0; r < nr; ++r) {
                    Value* safe_r = (r == 0) ? wrow[0] : c.b.build_select(row_ok[r], wrow[r], wrow[0]);
                    Value* sc = kb.load_f32(ptr_at(c, p[4], safe_r, 4));
                    for (std::uint32_t b = 0; b < mb; ++b) {
                        std::size_t idx = r * mb + b;
                        res[idx] = c.b.build_mul(res[idx], sc);
                    }
                }
                return res;
            }

            const int ie_shift = (byte_fmt || fmt == QF_Q8_0) ? 4 : 5;
            Value* nitems = c.b.build_lshr(p[6], c.i32(ie_shift));
            std::vector<Value*> res = kb.for_range_reduce_n(
                lane, nitems, c.i32(static_cast<int>(lpr * eff_unr)), acc_init,
                [&](Value* c0, const std::vector<Value*>& cur) -> std::vector<Value*> {
                    std::vector<Value*> next = cur;
                    for (std::uint32_t u = 0; u < eff_unr; ++u) {
                        Value* col_u = (u == 0) ? c0 : c.b.build_add(c0, c.i32(static_cast<int>(u * lpr)));
                        Value* in_k = (u == 0) ? c.i32(1) : c.b.build_ult(col_u, nitems);
                        Value* cc = (u == 0) ? c0 : c.b.build_select(in_k, col_u, c0);
                        Value* ok_f = (u == 0) ? c.f32(1.0f) : c.b.build_select(in_k, c.f32(1.0f), c.f32(0.0f));
                        Value* col_elem = c.b.build_shl(cc, c.i32(ie_shift));

                        if (byte_fmt) {
                            std::vector<std::vector<Value*>> xs(mb);
                            for (std::uint32_t b = 0; b < mb; ++b) {
                                Value* xi = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[7]), col_elem);
                                auto lo = load_x_slice8(c, dt, p[0], xi, 8, true);
                                auto hi = load_x_slice8(c, dt, p[0], c.b.build_add(xi, c.i32(8)), 8, true);
                                lo.insert(lo.end(), hi.begin(), hi.end());
                                xs[b] = std::move(lo);
                            }
                            for (std::uint32_t r = 0; r < nr; ++r) {
                                Value* wv128 = kb.vload(Type::i32x4(), c.b.build_add(row_ptrs[r], c.b.build_zext_i64(col_elem)));
                                Value* w4[4] = {
                                    c.b.build_vextract_lane(wv128, 0), c.b.build_vextract_lane(wv128, 1),
                                    c.b.build_vextract_lane(wv128, 2), c.b.build_vextract_lane(wv128, 3),
                                };
                                Value* wvals[16];
                                for (int e = 0; e < 16; ++e) {
                                    Value* qb = c.b.build_and(c.b.build_lshr(w4[e >> 2], c.i32((e & 3) * 8)), c.i32(0xff));
                                    wvals[e] = (fmt == QF_FP8) ? emit_fp8_e4m3_to_f32(c, qb) : emit_s8_byte_to_f32(c, qb);
                                }
                                for (std::uint32_t b = 0; b < mb; ++b) {
                                    Value* dot = c.f32(0.0f);
                                    for (int e = 0; e < 16; ++e) dot = c.b.build_fma_f32(xs[b][e], wvals[e], dot);
                                    std::size_t idx = r * mb + b;
                                    next[idx] = c.b.build_fma_f32(dot, ok_f, next[idx]);
                                }
                            }
                        } else if (fmt == QF_Q8_0) {
                            std::vector<std::vector<Value*>> xs(mb);
                            for (std::uint32_t b = 0; b < mb; ++b) {
                                Value* xi = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[7]), col_elem);
                                auto lo = load_x_slice8(c, dt, p[0], xi, 8, true);
                                auto hi = load_x_slice8(c, dt, p[0], c.b.build_add(xi, c.i32(8)), 8, true);
                                lo.insert(lo.end(), hi.begin(), hi.end());
                                xs[b] = std::move(lo);
                            }
                            Value* blk_idx = c.b.build_lshr(cc, c.i32(1));
                            Value* half_off = c.b.build_shl(c.b.build_and(cc, c.i32(1)), c.i32(4));
                            for (std::uint32_t r = 0; r < nr; ++r) {
                                Value* blk = c.b.build_add(
                                    row_ptrs[r], c.b.build_mul(c.b.build_zext_i64(blk_idx), kb.const_i64(34)));
                                Value* d = kb.f16_to_f32(kb.load_u16(blk));
                                Value* qp = c.b.build_add(blk, c.b.build_zext_i64(c.b.build_add(c.i32(2), half_off)));
                                auto bytes = load_u16_bytes(c, qp, 8);
                                Value* wvals[16];
                                for (int e = 0; e < 16; ++e) wvals[e] = emit_s8_byte_to_f32(c, bytes[e]);
                                Value* d_ok = c.b.build_mul(d, ok_f);
                                for (std::uint32_t b = 0; b < mb; ++b) {
                                    Value* dot = c.f32(0.0f);
                                    for (int e = 0; e < 16; ++e) dot = c.b.build_fma_f32(xs[b][e], wvals[e], dot);
                                    std::size_t idx = r * mb + b;
                                    next[idx] = c.b.build_fma_f32(dot, d_ok, next[idx]);
                                }
                            }
                        } else if (fmt == QF_Q4K) {
                            std::vector<std::vector<Value*>> xs(mb);
                            for (std::uint32_t b = 0; b < mb; ++b) {
                                xs[b].reserve(32);
                                Value* xi = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[7]), col_elem);
                                for (int ch = 0; ch < 32; ch += 8) {
                                    auto sl = load_x_slice8(c, dt, p[0], c.b.build_add(xi, c.i32(ch)), 8, true);
                                    xs[b].insert(xs[b].end(), sl.begin(), sl.end());
                                }
                            }
                            std::vector<Value*> sum_x(mb);
                            for (std::uint32_t b = 0; b < mb; ++b) {
                                Value* s = c.f32(0.0f);
                                for (int e = 0; e < 32; ++e) s = c.b.build_add(s, xs[b][e]);
                                sum_x[b] = s;
                            }
                            Value* sb = c.b.build_lshr(cc, c.i32(3));
                            Value* j = c.b.build_and(cc, c.i32(7));
                            Value* low_j = c.b.build_ult(j, c.i32(4));
                            Value* j4 = c.b.build_add(j, c.i32(4));
                            Value* jm4 = c.b.build_sub(j, c.i32(4));
                            Value* q_off = c.b.build_add(c.i32(16), c.b.build_shl(c.b.build_lshr(j, c.i32(1)), c.i32(5)));
                            Value* sh = c.b.build_shl(c.b.build_and(j, c.i32(1)), c.i32(2));

                            for (std::uint32_t r = 0; r < nr; ++r) {
                                Value* blk = c.b.build_add(
                                    row_ptrs[r], c.b.build_mul(c.b.build_zext_i64(sb), kb.const_i64(144)));
                                Value* hdr = kb.vload(Type::i32x4(), blk);
                                Value* a0 = c.b.build_vextract_lane(hdr, 0);
                                Value* s0 = c.b.build_vextract_lane(hdr, 1);
                                Value* s1 = c.b.build_vextract_lane(hdr, 2);
                                Value* s2 = c.b.build_vextract_lane(hdr, 3);
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
                                Value* d = kb.f16_to_f32(c.b.build_and(a0, c.i32(0xffff)));
                                Value* dmin = kb.f16_to_f32(c.b.build_lshr(a0, c.i32(16)));
                                Value* ws = c.b.build_mul(c.b.build_mul(kb.u32_to_f32(sc), d), ok_f);
                                Value* wm = c.b.build_mul(c.b.build_mul(kb.u32_to_f32(mn), dmin), ok_f);

                                Value* q_base = c.b.build_add(blk, c.b.build_zext_i64(q_off));
                                Value* qv0 = kb.vload(Type::i32x4(), q_base);
                                Value* qv1 = kb.vload(Type::i32x4(), c.b.build_add(q_base, kb.const_i64(16)));
                                Value* qw[8] = {
                                    c.b.build_vextract_lane(qv0, 0), c.b.build_vextract_lane(qv0, 1),
                                    c.b.build_vextract_lane(qv0, 2), c.b.build_vextract_lane(qv0, 3),
                                    c.b.build_vextract_lane(qv1, 0), c.b.build_vextract_lane(qv1, 1),
                                    c.b.build_vextract_lane(qv1, 2), c.b.build_vextract_lane(qv1, 3),
                                };
                                Value* qvals[32];
                                for (int e = 0; e < 32; ++e) {
                                    Value* q4 = c.b.build_and(
                                        c.b.build_lshr(qw[e >> 2], c.b.build_add(sh, c.i32((e & 3) * 8))),
                                        c.i32(0x0f));
                                    qvals[e] = kb.u32_to_f32(q4);
                                }
                                for (std::uint32_t b = 0; b < mb; ++b) {
                                    Value* dot = c.f32(0.0f);
                                    for (int e = 0; e < 32; ++e) dot = c.b.build_fma_f32(xs[b][e], qvals[e], dot);
                                    std::size_t idx = r * mb + b;
                                    Value* t = c.b.build_fma_f32(ws, dot, next[idx]);
                                    next[idx] = c.b.build_fma_f32(c.b.build_neg(wm), sum_x[b], t);
                                }
                            }
                        } else {
                            // QF_Q6K: 32 elements = (g, quad) within 210-byte superblock
                            std::vector<std::vector<Value*>> xs(mb);
                            for (std::uint32_t b = 0; b < mb; ++b) {
                                xs[b].reserve(32);
                                Value* xi = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[7]), col_elem);
                                for (int ch = 0; ch < 32; ch += 8) {
                                    auto sl = load_x_slice8(c, dt, p[0], c.b.build_add(xi, c.i32(ch)), 8, true);
                                    xs[b].insert(xs[b].end(), sl.begin(), sl.end());
                                }
                            }
                            Value* sb = c.b.build_lshr(cc, c.i32(3));
                            Value* g = c.b.build_and(c.b.build_lshr(cc, c.i32(2)), c.i32(1));
                            Value* quad = c.b.build_and(cc, c.i32(3));
                            Value* ql_off = c.b.build_add(c.b.build_shl(g, c.i32(6)),
                                                          c.b.build_shl(c.b.build_and(quad, c.i32(1)), c.i32(5)));
                            Value* qh_off = c.b.build_add(c.i32(128), c.b.build_shl(g, c.i32(5)));
                            Value* sc_off = c.b.build_add(c.i32(192), c.b.build_add(c.b.build_shl(g, c.i32(3)),
                                                                                    c.b.build_shl(quad, c.i32(1))));
                            Value* ql_sh = c.b.build_select(c.b.build_uge(quad, c.i32(2)), c.i32(4), c.i32(0));
                            Value* qh_sh = c.b.build_shl(quad, c.i32(1));

                            for (std::uint32_t r = 0; r < nr; ++r) {
                                Value* blk = c.b.build_add(
                                    row_ptrs[r], c.b.build_mul(c.b.build_zext_i64(sb), kb.const_i64(210)));
                                Value* sc_pair = kb.load_u16(c.b.build_add(blk, c.b.build_zext_i64(sc_off)));
                                Value* d = c.b.build_mul(kb.f16_to_f32(kb.load_u16(c.b.build_add(blk, kb.const_i64(208)))), ok_f);
                                Value* s0 = c.b.build_mul(d, emit_s8_byte_to_f32(c, c.b.build_and(sc_pair, c.i32(0xff))));
                                Value* s1 = c.b.build_mul(d, emit_s8_byte_to_f32(c, c.b.build_lshr(sc_pair, c.i32(8))));
                                auto ql_b = load_u16_bytes(c, c.b.build_add(blk, c.b.build_zext_i64(ql_off)), 16);
                                auto qh_b = load_u16_bytes(c, c.b.build_add(blk, c.b.build_zext_i64(qh_off)), 16);
                                Value* qvals[32];
                                for (int e = 0; e < 32; ++e) {
                                    Value* nib = c.b.build_and(c.b.build_lshr(ql_b[e], ql_sh), c.i32(0x0f));
                                    Value* hb = c.b.build_shl(c.b.build_and(c.b.build_lshr(qh_b[e], qh_sh), c.i32(0x03)), c.i32(4));
                                    qvals[e] = c.b.build_sub(kb.u32_to_f32(c.b.build_or(nib, hb)), c.f32(32.0f));
                                }
                                for (std::uint32_t b = 0; b < mb; ++b) {
                                    Value* dot0 = c.f32(0.0f);
                                    Value* dot1 = c.f32(0.0f);
                                    for (int e = 0; e < 16; ++e) dot0 = c.b.build_fma_f32(xs[b][e], qvals[e], dot0);
                                    for (int e = 16; e < 32; ++e) dot1 = c.b.build_fma_f32(xs[b][e], qvals[e], dot1);
                                    std::size_t idx = r * mb + b;
                                    next[idx] = c.b.build_fma_f32(s1, dot1, c.b.build_fma_f32(s0, dot0, next[idx]));
                                }
                            }
                        }
                    }
                    return next;
                });

            if (byte_fmt) {
                for (std::uint32_t r = 0; r < nr; ++r) {
                    Value* safe_r = (r == 0) ? wrow[0] : c.b.build_select(row_ok[r], wrow[r], wrow[0]);
                    Value* sc = kb.load_f32(ptr_at(c, p[4], safe_r, 4));
                    for (std::uint32_t b = 0; b < mb; ++b) {
                        std::size_t idx = r * mb + b;
                        res[idx] = c.b.build_mul(res[idx], sc);
                    }
                }
            }
            return res;
        },
        [&]() -> std::vector<Value*> { return acc_init; });

    // Reduce across `lpr` lanes cleanly in 32-lane slices; when lpr == 64, combine the two halves via shared memory.
    const std::uint32_t shfl_start = (lpr == 16u) ? 8u : 16u;
    for (std::uint32_t q = 0; q < nr * mb; ++q) {
        Value* s = acc[q];
        for (std::uint32_t m = shfl_start; m >= 1u; m >>= 1) {
            s = c.b.build_add(s, kb.shfl_bfly_f32(s, m));
        }
        acc[q] = s;
    }

    if (lpr == 64u) {
        Value* lane32 = c.b.build_and(lane, c.i32(31));
        Value* half = c.b.build_lshr(lane, c.i32(5));
        Value* slot_base = c.b.build_mul(
            c.b.build_add(c.b.build_shl(unit, c.i32(1)), half),
            c.i32(static_cast<int>(nr * mb)));
        kb.if_then(c.b.build_eq(lane32, c.i32(0)), [&] {
            for (std::uint32_t q = 0; q < nr * mb; ++q) {
                kb.shared_store_f32_indexed(s_part, c.b.build_add(slot_base, c.i32(static_cast<int>(q))), acc[q]);
            }
        });
        kb.sync();
    }

    kb.if_then(c.b.build_and(c.b.build_eq(lane, c.i32(0)), row_ok[0]), [&] {
        std::vector<Value*> sum(nr * mb);
        if (lpr == 64u) {
            Value* b0 = c.b.build_mul(c.b.build_shl(unit, c.i32(1)), c.i32(static_cast<int>(nr * mb)));
            Value* b1 = c.b.build_add(b0, c.i32(static_cast<int>(nr * mb)));
            for (std::uint32_t q = 0; q < nr * mb; ++q) {
                sum[q] = c.b.build_add(
                    kb.shared_load_f32_indexed(s_part, c.b.build_add(b0, c.i32(static_cast<int>(q)))),
                    kb.shared_load_f32_indexed(s_part, c.b.build_add(b1, c.i32(static_cast<int>(q)))));
            }
        } else {
            sum = acc;
        }

        if (glu) {
            for (std::uint32_t q = 0; q < nr / 2u; ++q) {
                const std::uint32_t r0 = 2u * q, r1 = 2u * q + 1u;
                auto emit_glu_pair = [&] {
                    Value* b0 = has_bias ? load_elem_dt(c, dt, p[3], wrow[r0]) : nullptr;
                    Value* b1 = has_bias ? load_elem_dt(c, dt, p[3], wrow[r1]) : nullptr;
                    Value* col_out = (epi == EPI_GEGLU) ? c.b.build_lshr(wrow[r0], c.i32(1)) : wrow[r0];
                    for (std::uint32_t b = 0; b < mb; ++b) {
                        Value* v0 = sum[r0 * mb + b];
                        Value* v1 = sum[r1 * mb + b];
                        if (has_bias) {
                            v0 = c.b.build_add(v0, b0);
                            v1 = c.b.build_add(v1, b1);
                        }
                        v0 = emit_gemm_act(c, act, v0);
                        v1 = emit_gemm_act(c, act, v1);
                        Value* out = (epi == EPI_GEGLU)
                                         ? c.b.build_mul(v0, emit_gelu_exact(c, v1))
                                         : c.b.build_mul(c.b.build_mul(v0, emit_sigmoid(c, v0)), v1);
                        Value* oidx = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[8]), col_out);
                        store_elem_dt(c, dt, p[2], oidx, out);
                    }
                };
                if (q == 0) emit_glu_pair();
                else kb.if_then(row_ok[r0], emit_glu_pair);
            }
        } else {
            for (std::uint32_t r = 0; r < nr; ++r) {
                auto emit_row = [&] {
                    Value* bv = has_bias ? load_elem_dt(c, dt, p[3], wrow[r]) : nullptr;
                    for (std::uint32_t b = 0; b < mb; ++b) {
                        Value* v = sum[r * mb + b];
                        if (has_bias) v = c.b.build_add(v, bv);
                        v = emit_gemm_act(c, act, v);
                        Value* oidx = c.b.build_add(c.b.build_mul(c.i32(static_cast<int>(b)), p[8]), wrow[r]);
                        if (epi == EPI_ACCUM) v = c.b.build_add(v, load_elem_dt(c, dt, p[2], oidx));
                        store_elem_dt(c, dt, p[2], oidx, v);
                    }
                };
                if (r == 0) emit_row();
                else kb.if_then(row_ok[r], emit_row);
            }
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
