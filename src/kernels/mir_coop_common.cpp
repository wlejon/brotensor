#include "mir_coop_common.h"

#include "mir_quant_helpers.h"
#include "../vulkan/shaders/op_codes.h"

namespace brotensor::kernels {

using jit::gpu_mir::bits_to_f32;

Value* emit_v4_zero(Ctx& c) {
    return c.b.build_vzero(Type::i32x4());
}

Value* emit_v4_pack(Ctx& c, Value* w0, Value* w1, Value* w2, Value* w3) {
    Value* v = c.b.build_vzero(Type::i32x4());
    v = c.b.build_vinsert_lane(v, w0, 0);
    v = c.b.build_vinsert_lane(v, w1, 1);
    v = c.b.build_vinsert_lane(v, w2, 2);
    v = c.b.build_vinsert_lane(v, w3, 3);
    return v;
}

Value* emit_v4_select(Ctx& c, Value* cond, Value* a, Value* b) {
    Value* w[4];
    for (std::uint32_t i = 0; i < 4; ++i) {
        w[i] = c.b.build_select(cond, c.b.build_vextract_lane(a, i), c.b.build_vextract_lane(b, i));
    }
    return emit_v4_pack(c, w[0], w[1], w[2], w[3]);
}

Value* emit_pack_f32x8_to_f16_v4(Ctx& c, const std::array<Value*, 8>& v) {
    Value* w0 = c.kb.pack_f16x2(v[0], v[1]);
    Value* w1 = c.kb.pack_f16x2(v[2], v[3]);
    Value* w2 = c.kb.pack_f16x2(v[4], v[5]);
    Value* w3 = c.kb.pack_f16x2(v[6], v[7]);
    return emit_v4_pack(c, w0, w1, w2, w3);
}

Value* emit_bf16x8_to_f16x8(Ctx& c, Value* v4) {
    Value* out[4];
    for (std::uint32_t i = 0; i < 4; ++i) {
        Value* w = c.b.build_vextract_lane(v4, i);
        Value* lo = bits_to_f32(c, c.b.build_shl(w, c.i32(16)));
        Value* hi = bits_to_f32(c, c.b.build_and(w, c.i32(static_cast<int>(0xffff0000u))));
        out[i] = c.kb.pack_f16x2(lo, hi);
    }
    return emit_v4_pack(c, out[0], out[1], out[2], out[3]);
}

Value* emit_bf16x8_to_f16x8_scaled(Ctx& c, Value* v4, const std::array<Value*, 8>& mul) {
    Value* out[4];
    for (std::uint32_t i = 0; i < 4; ++i) {
        Value* w = c.b.build_vextract_lane(v4, i);
        Value* lo = c.b.build_mul(bits_to_f32(c, c.b.build_shl(w, c.i32(16))), mul[2u * i]);
        Value* hi = c.b.build_mul(bits_to_f32(c, c.b.build_and(w, c.i32(static_cast<int>(0xffff0000u)))), mul[2u * i + 1u]);
        out[i] = c.kb.pack_f16x2(lo, hi);
    }
    return emit_v4_pack(c, out[0], out[1], out[2], out[3]);
}

Value* emit_f32x8_to_f16x8_scaled(Ctx& c, Value* v0, Value* v1, const std::array<Value*, 8>& mul) {
    std::array<Value*, 8> f{};
    for (std::uint32_t i = 0; i < 4; ++i) {
        f[i] = c.b.build_mul(bits_to_f32(c, c.b.build_vextract_lane(v0, i)), mul[i]);
        f[4u + i] = c.b.build_mul(bits_to_f32(c, c.b.build_vextract_lane(v1, i)), mul[4u + i]);
    }
    return emit_pack_f32x8_to_f16_v4(c, f);
}

QRawMir emit_qload8_mir(Ctx& c, std::uint32_t fmt, bool vec, Value* row_ptr,
                        Value* scale_ptr, Value* ridx, Value* k, Value* kmax) {
    QRawMir o;
    if (fmt == QF_INT8 || fmt == QF_FP8) {
        if (vec) {
            Value* q64 = c.kb.load_i64(c.b.build_add(row_ptr, c.b.build_zext_i64(k)));
            o.w0 = c.b.build_trunc_i32(q64);
            o.w1 = c.b.build_trunc_i32(c.b.build_lshr(q64, c.kb.const_i64(32)));
        } else {
            Value* w[2] = {c.i32(0), c.i32(0)};
            for (int e = 0; e < 8; ++e) {
                Value* ke = c.b.build_add(k, c.i32(e));
                Value* ok = c.b.build_ult(ke, kmax);
                Value* b = emit_if_else_values(
                    c, ok, {Type::i32()},
                    [&]() -> std::vector<Value*> {
                        return {c.kb.load_u8(c.b.build_add(row_ptr, c.b.build_zext_i64(ke)))};
                    },
                    [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
                w[e >> 2] = c.b.build_or(w[e >> 2], c.b.build_shl(b, c.i32((e & 3) * 8)));
            }
            o.w0 = w[0];
            o.w1 = w[1];
        }
        o.c = c.kb.load_i32(ptr_at(c, scale_ptr, ridx, 4));
    } else if (fmt == QF_Q8_0) {
        Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(k, c.i32(5))),
                                                          c.kb.const_i64(34)));
        Value* qp = c.b.build_add(blk, c.b.build_zext_i64(c.b.build_add(c.i32(2), c.b.build_and(k, c.i32(31)))));
        for (int i = 0; i < 4; ++i) {
            o.s0[static_cast<std::size_t>(i)] = c.kb.load_u16(c.b.build_add(qp, c.kb.const_i64(i * 2)));
        }
        o.c = c.kb.load_u16(blk);
    } else if (fmt == QF_Q4K) {
        Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(k, c.i32(8))),
                                                          c.kb.const_i64(144)));
        Value* j = c.b.build_lshr(c.b.build_and(k, c.i32(255)), c.i32(5));
        Value* boff = c.b.build_add(c.i32(16), c.b.build_add(c.b.build_shl(c.b.build_lshr(j, c.i32(1)), c.i32(5)),
                                                             c.b.build_and(k, c.i32(31))));
        o.a = c.kb.vload(Type::i32x4(), blk);
        Value* n64 = c.kb.load_i64(c.b.build_add(blk, c.b.build_zext_i64(boff)));
        o.w0 = c.b.build_trunc_i32(n64);
        o.w1 = c.b.build_trunc_i32(c.b.build_lshr(n64, c.kb.const_i64(32)));
    } else {
        // QF_Q6K: 210-byte super-blocks (2-byte aligned)
        Value* blk = c.b.build_add(row_ptr, c.b.build_mul(c.b.build_zext_i64(c.b.build_lshr(k, c.i32(8))),
                                                          c.kb.const_i64(210)));
        Value* e0 = c.b.build_and(k, c.i32(255));
        Value* g = c.b.build_lshr(e0, c.i32(7));
        Value* quad = c.b.build_lshr(c.b.build_and(e0, c.i32(127)), c.i32(5));
        Value* l0 = c.b.build_and(e0, c.i32(31));
        Value* ql_off = c.b.build_add(c.b.build_add(c.b.build_shl(g, c.i32(6)),
                                                    c.b.build_shl(c.b.build_and(quad, c.i32(1)), c.i32(5))),
                                      l0);
        Value* qh_off = c.b.build_add(c.i32(128), c.b.build_add(c.b.build_shl(g, c.i32(5)), l0));
        Value* sc_off = c.b.build_add(c.i32(192), c.b.build_add(c.b.build_shl(g, c.i32(3)),
                                                                c.b.build_shl(quad, c.i32(1))));
        Value* qlp = c.b.build_add(blk, c.b.build_zext_i64(ql_off));
        Value* qhp = c.b.build_add(blk, c.b.build_zext_i64(qh_off));
        for (int i = 0; i < 4; ++i) {
            o.s0[static_cast<std::size_t>(i)] = c.kb.load_u16(c.b.build_add(qlp, c.kb.const_i64(i * 2)));
            o.s1[static_cast<std::size_t>(i)] = c.kb.load_u16(c.b.build_add(qhp, c.kb.const_i64(i * 2)));
        }
        o.c = c.kb.load_u16(c.b.build_add(blk, c.b.build_zext_i64(sc_off)));
        o.d = c.kb.load_u16(c.b.build_add(blk, c.kb.const_i64(208)));
    }
    return o;
}

Value* emit_qdecode8_f16_mir(Ctx& c, std::uint32_t fmt, const QRawMir& q, Value* k, Value* s_fp8) {
    std::array<Value*, 8> w{};
    if (fmt == QF_FP8) {
        Value* sc = bits_to_f32(c, q.c);
        for (int e = 0; e < 8; ++e) {
            Value* word = (e < 4) ? q.w0 : q.w1;
            Value* b = c.b.build_and(c.b.build_lshr(word, c.i32((e & 3) * 8)), c.i32(0xff));
            Value* dv = s_fp8 ? c.kb.shared_load_f32_indexed(s_fp8, b) : emit_fp8_e4m3_to_f32(c, b);
            w[static_cast<std::size_t>(e)] = c.b.build_mul(dv, sc);
        }
    } else if (fmt == QF_INT8 || fmt == QF_Q8_0) {
        Value* sc = (fmt == QF_INT8) ? bits_to_f32(c, q.c) : c.kb.f16_to_f32(q.c);
        Value* qlo = (fmt == QF_INT8) ? q.w0 : c.b.build_or(q.s0[0], c.b.build_shl(q.s0[1], c.i32(16)));
        Value* qhi = (fmt == QF_INT8) ? q.w1 : c.b.build_or(q.s0[2], c.b.build_shl(q.s0[3], c.i32(16)));
        for (int e = 0; e < 8; ++e) {
            Value* word = (e < 4) ? qlo : qhi;
            Value* b = c.b.build_and(c.b.build_lshr(word, c.i32((e & 3) * 8)), c.i32(0xff));
            w[static_cast<std::size_t>(e)] = c.b.build_mul(emit_s8_byte_to_f32(c, b), sc);
        }
    } else if (fmt == QF_Q4K) {
        Value* a0 = c.b.build_vextract_lane(q.a, 0);
        Value* s0 = c.b.build_vextract_lane(q.a, 1);
        Value* s1 = c.b.build_vextract_lane(q.a, 2);
        Value* s2 = c.b.build_vextract_lane(q.a, 3);
        Value* j = c.b.build_lshr(c.b.build_and(k, c.i32(255)), c.i32(5));
        Value* low_j = c.b.build_ult(j, c.i32(4));
        Value* j4 = c.b.build_add(j, c.i32(4));
        Value* jm4 = c.b.build_sub(j, c.i32(4));
        Value* bs_j = emit_extract_byte(c, emit_sel_word3(c, c.b.build_lshr(j, c.i32(2)), s0, s1, s2), j);
        Value* bs_j4 = emit_extract_byte(c, emit_sel_word3(c, c.b.build_lshr(j4, c.i32(2)), s0, s1, s2), j4);
        Value* bs_jm4 = emit_extract_byte(c, emit_sel_word3(c, c.b.build_lshr(jm4, c.i32(2)), s0, s1, s2), jm4);
        Value* sc = c.b.build_select(
            low_j, c.b.build_and(bs_j, c.i32(63)),
            c.b.build_or(c.b.build_and(bs_j4, c.i32(15)), c.b.build_shl(c.b.build_lshr(bs_jm4, c.i32(6)), c.i32(4))));
        Value* mn = c.b.build_select(
            low_j, c.b.build_and(bs_j4, c.i32(63)),
            c.b.build_or(c.b.build_lshr(bs_j4, c.i32(4)), c.b.build_shl(c.b.build_lshr(bs_j, c.i32(6)), c.i32(4))));
        Value* d = c.kb.f16_to_f32(c.b.build_and(a0, c.i32(0xffff)));
        Value* dmin = c.kb.f16_to_f32(c.b.build_lshr(a0, c.i32(16)));
        Value* ws = c.b.build_mul(c.kb.u32_to_f32(sc), d);
        Value* wm = c.b.build_mul(c.kb.u32_to_f32(mn), dmin);
        Value* sh = c.b.build_shl(c.b.build_and(j, c.i32(1)), c.i32(2));
        for (int e = 0; e < 8; ++e) {
            Value* word = (e < 4) ? q.w0 : q.w1;
            Value* q4 = c.b.build_and(c.b.build_lshr(word, c.b.build_add(sh, c.i32((e & 3) * 8))), c.i32(0x0f));
            w[static_cast<std::size_t>(e)] = c.b.build_fma_f32(ws, c.kb.u32_to_f32(q4), c.b.build_neg(wm));
        }
    } else {
        Value* e0 = c.b.build_and(k, c.i32(255));
        Value* quad = c.b.build_lshr(c.b.build_and(e0, c.i32(127)), c.i32(5));
        Value* l0 = c.b.build_and(e0, c.i32(31));
        Value* sbyte = c.b.build_and(c.b.build_lshr(q.c, c.b.build_shl(c.b.build_lshr(l0, c.i32(4)), c.i32(3))),
                                     c.i32(0xff));
        Value* sc = c.b.build_mul(c.kb.f16_to_f32(q.d), emit_s8_byte_to_f32(c, sbyte));
        Value* qlo = c.b.build_or(q.s0[0], c.b.build_shl(q.s0[1], c.i32(16)));
        Value* qhi = c.b.build_or(q.s0[2], c.b.build_shl(q.s0[3], c.i32(16)));
        Value* hlo = c.b.build_or(q.s1[0], c.b.build_shl(q.s1[1], c.i32(16)));
        Value* hhi = c.b.build_or(q.s1[2], c.b.build_shl(q.s1[3], c.i32(16)));
        Value* ql_sh = c.b.build_select(c.b.build_uge(quad, c.i32(2)), c.i32(4), c.i32(0));
        Value* qh_sh = c.b.build_shl(quad, c.i32(1));
        for (int e = 0; e < 8; ++e) {
            Value* bw = (e < 4) ? qlo : qhi;
            Value* hw = (e < 4) ? hlo : hhi;
            Value* bsh = c.i32((e & 3) * 8);
            Value* nib = c.b.build_and(c.b.build_lshr(c.b.build_lshr(bw, bsh), ql_sh), c.i32(0x0f));
            Value* hb = c.b.build_shl(c.b.build_and(c.b.build_lshr(c.b.build_lshr(hw, bsh), qh_sh), c.i32(0x03)),
                                      c.i32(4));
            Value* q6 = c.b.build_sub(c.kb.u32_to_f32(c.b.build_or(nib, hb)), c.f32(32.0f));
            w[static_cast<std::size_t>(e)] = c.b.build_mul(sc, q6);
        }
    }
    return emit_pack_f32x8_to_f16_v4(c, w);
}

void emit_coop_acc_map(Ctx& c, Value* acc, const std::function<Value*(Value* idx, Value* val)>& fn) {
    Value* len = c.kb.coop_acc_len(acc);
    c.kb.for_range(c.i32(0), len, c.i32(1), [&](Value* e) {
        Value* cur = c.kb.coop_get_acc(acc, e);
        c.kb.coop_set_acc(acc, e, fn(e, cur));
    });
}

Value* emit_cluster16_max_f32(Ctx& c, Value* v) {
    for (std::uint32_t m = 8; m >= 1; m >>= 1) v = c.kb.fmax(v, c.kb.shfl_bfly_f32(v, m));
    return v;
}

Value* emit_cluster16_add_f32(Ctx& c, Value* v) {
    for (std::uint32_t m = 8; m >= 1; m >>= 1) v = c.b.build_add(v, c.kb.shfl_bfly_f32(v, m));
    return v;
}

Value* emit_cluster16_min_i32(Ctx& c, Value* v) {
    for (std::uint32_t m = 8; m >= 1; m >>= 1) {
        Value* o = c.kb.shfl_bfly_i32(v, m);
        v = c.b.build_select(c.b.build_ult(o, v), o, v);
    }
    return v;
}

Value* emit_cluster16_max_i32(Ctx& c, Value* v) {
    for (std::uint32_t m = 8; m >= 1; m >>= 1) {
        Value* o = c.kb.shfl_bfly_i32(v, m);
        v = c.b.build_select(c.b.build_ugt(o, v), o, v);
    }
    return v;
}

Value* emit_subgroup32_any_i32(Ctx& c, Value* cond) {
    Value* v = c.b.build_select(cond, c.i32(1), c.i32(0));
    for (std::uint32_t m = 16; m >= 1; m >>= 1) v = c.b.build_or(v, c.kb.shfl_bfly_i32(v, m));
    return c.b.build_ne(v, c.i32(0));
}

Value* emit_subgroup32_all_i32(Ctx& c, Value* cond) {
    Value* v = c.b.build_select(cond, c.i32(1), c.i32(0));
    for (std::uint32_t m = 16; m >= 1; m >>= 1) v = c.b.build_and(v, c.kb.shfl_bfly_i32(v, m));
    return c.b.build_ne(v, c.i32(0));
}

}  // namespace brotensor::kernels
