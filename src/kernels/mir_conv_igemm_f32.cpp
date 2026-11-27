#include "mir_conv_igemm_f32.h"

#include "mir_common.h"

#include <array>
#include <cstdint>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kBm = 128;
constexpr int kBn = 64;
constexpr int kBk = 16;
constexpr int kBnp = kBn + 4;  // 68 floats per W row

using jit::gpu_mir::bits_to_f32;

Value* make_f32x4(Ctx& c, Value* v0, Value* v1, Value* v2, Value* v3) {
    Value* v = c.b.build_vzero(Type::f32x4());
    v = c.b.build_vinsert_lane(v, v0, 0);
    v = c.b.build_vinsert_lane(v, v1, 1);
    v = c.b.build_vinsert_lane(v, v2, 2);
    return c.b.build_vinsert_lane(v, v3, 3);
}

}  // namespace

Function* build_conv2d_igemm_f32_mir(Module& mod, bool has_bias, bool vec_store) {
    std::vector<Type> params(4, Type::ptr());
    params.insert(params.end(), 18, Type::i32());
    Function* fn = mod.create_function("mir_conv2d_igemm_f32", Type::void_type(), params);
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // 0:X 1:Wt 2:bias 3:Y
    // 4:Cin 5:H 6:W 7:Cout 8:kH 9:kW 10:sh 11:sw 12:ph 13:pw 14:dh 15:dw 16:Ho 17:Wo 18:M 19:K 20:KHW 21:HWo
    Value* X = p[0];
    Value* Wt = p[1];
    Value* bias = p[2];
    Value* Y = p[3];
    Value* Cin = p[4];
    Value* H = p[5];
    Value* W = p[6];
    Value* Cout = p[7];
    Value* kW = p[9];
    Value* sh = p[10];
    Value* sw = p[11];
    Value* ph = p[12];
    Value* pw = p[13];
    Value* dh = p[14];
    Value* dw = p[15];
    Value* Wo = p[17];
    Value* M = p[18];
    Value* K = p[19];
    Value* KHW = p[20];
    Value* HWo = p[21];

    Value* As = kb.shared_alloc_f32(2u * kBk * kBm);   // [2][16][128]
    Value* Bs = kb.shared_alloc_f32(2u * kBk * kBnp);  // [2][16][68]

    Value* tid = kb.tid_x();
    Value* m0 = c.b.build_mul(kb.ctaid_x(), c.i32(kBm));
    Value* n0 = c.b.build_mul(kb.ctaid_y(), c.i32(kBn));

    // A loader coordinates: each thread owns 1 pixel column (a_m), 8 K rows (a_k + 2*i).
    Value* a_m = c.b.build_and(tid, c.i32(kBm - 1));
    Value* a_k = c.b.build_lshr(tid, c.i32(7));
    Value* m_glob = c.b.build_add(m0, a_m);
    Value* m_ok = c.b.build_slt(m_glob, M);
    Value* safe_m = c.b.build_select(m_ok, m_glob, c.i32(0));
    Value* n_img = c.b.build_udiv(safe_m, HWo);
    Value* rem_p = c.b.build_umod(safe_m, HWo);
    Value* oh = c.b.build_udiv(rem_p, Wo);
    Value* ow = c.b.build_umod(rem_p, Wo);
    Value* ih0 = c.b.build_sub(c.b.build_mul(oh, sh), ph);
    Value* iw0 = c.b.build_sub(c.b.build_mul(ow, sw), pw);
    Value* hw_in = c.b.build_mul(H, W);
    Value* chw_in = c.b.build_mul(Cin, hw_in);
    Value* xbase = c.b.build_add(
        X, c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(n_img), c.b.build_zext_i64(chw_in)), kb.const_i64(2)));

    // W loader coordinates: each thread owns 1 output channel (b_n), 4 consecutive K (b_k + 0..3).
    Value* b_n = c.b.build_lshr(tid, c.i32(2));
    Value* b_k = c.b.build_shl(c.b.build_and(tid, c.i32(3)), c.i32(2));
    Value* bn_glob = c.b.build_add(n0, b_n);
    Value* n_ok = c.b.build_slt(bn_glob, Cout);
    Value* safe_bn = c.b.build_select(n_ok, bn_glob, c.i32(0));
    Value* wrow = c.b.build_add(
        Wt, c.b.build_shl(c.b.build_mul(c.b.build_zext_i64(safe_bn), c.b.build_zext_i64(K)), kb.const_i64(2)));

    auto load_tiles = [&](Value* k0) -> std::vector<Value*> {
        std::vector<Value*> regs(12);
        for (int i = 0; i < 8; ++i) {
            Value* k = c.b.build_add(k0, c.b.build_add(a_k, c.i32(2 * i)));
            Value* mk_ok = c.b.build_and(m_ok, c.b.build_slt(k, K));
            Value* sk = c.b.build_select(mk_ok, k, c.i32(0));
            Value* ic = c.b.build_udiv(sk, KHW);
            Value* r = c.b.build_umod(sk, KHW);
            Value* kh = c.b.build_udiv(r, kW);
            Value* kw = c.b.build_umod(r, kW);
            Value* ih = c.b.build_add(ih0, c.b.build_mul(kh, dh));
            Value* iw = c.b.build_add(iw0, c.b.build_mul(kw, dw));
            Value* in_ok = c.b.build_and(
                mk_ok,
                c.b.build_and(
                    c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, H)),
                    c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, W))));
            Value* raw_idx = c.b.build_add(
                c.b.build_mul(c.b.build_add(c.b.build_mul(c.b.build_zext_i64(ic),
                                                          c.b.build_zext_i64(H)),
                                            c.b.build_zext_i64(ih)),
                              c.b.build_zext_i64(W)),
                c.b.build_zext_i64(iw));
            Value* safe_idx = c.b.build_select(in_ok, raw_idx, kb.const_i64(0));
            Value* loaded = kb.load_f32(c.b.build_add(xbase, c.b.build_shl(safe_idx, kb.const_i64(2))));
            regs[static_cast<std::size_t>(i)] = c.b.build_select(in_ok, loaded, c.f32(0.0f));
        }
        for (int e = 0; e < 4; ++e) {
            Value* k = c.b.build_add(k0, c.b.build_add(b_k, c.i32(e)));
            Value* ok = c.b.build_and(n_ok, c.b.build_slt(k, K));
            Value* sk = c.b.build_select(ok, k, c.i32(0));
            regs[static_cast<std::size_t>(8 + e)] = c.b.build_select(
                ok, kb.load_f32(ptr_at(c, wrow, sk, 4)), c.f32(0.0f));
        }
        return regs;
    };

    auto store_tiles = [&](Value* buf, const std::vector<Value*>& regs) {
        Value* a_buf = c.b.build_mul(buf, c.i32(kBk * kBm));
        Value* b_buf = c.b.build_mul(buf, c.i32(kBk * kBnp));
        for (int i = 0; i < 8; ++i) {
            Value* k = c.b.build_add(a_k, c.i32(2 * i));
            Value* sidx = c.b.build_add(a_buf, c.b.build_add(c.b.build_mul(k, c.i32(kBm)), a_m));
            kb.shared_store_f32_indexed(As, sidx, regs[static_cast<std::size_t>(i)]);
        }
        for (int e = 0; e < 4; ++e) {
            Value* k = c.b.build_add(b_k, c.i32(e));
            Value* sidx = c.b.build_add(b_buf, c.b.build_add(c.b.build_mul(k, c.i32(kBnp)), b_n));
            kb.shared_store_f32_indexed(Bs, sidx, regs[static_cast<std::size_t>(8 + e)]);
        }
    };

    // Prefill stage 0.
    store_tiles(c.i32(0), load_tiles(c.i32(0)));
    kb.sync();

    Value* tm = c.b.build_and(tid, c.i32(15));
    Value* tn = c.b.build_lshr(tid, c.i32(4));
    Value* a0_base = c.b.build_shl(tm, c.i32(2));
    Value* a1_base = c.b.build_add(c.i32(64), a0_base);
    Value* b_base = c.b.build_shl(tn, c.i32(2));

    auto emit_fma_tile = [&](Value* buf, std::vector<Value*>& accs) {
        Value* a_buf_v4 = c.b.build_mul(buf, c.i32(kBk * (kBm / 4)));
        Value* b_buf_v4 = c.b.build_mul(buf, c.i32(kBk * (kBnp / 4)));
        for (int k = 0; k < kBk; ++k) {
            Value* a0_v4 = kb.shared_load_i32x4_indexed(
                As, c.b.build_add(a_buf_v4, c.b.build_add(c.i32(k * (kBm / 4)), tm)));
            Value* a1_v4 = kb.shared_load_i32x4_indexed(
                As, c.b.build_add(a_buf_v4, c.b.build_add(c.i32(k * (kBm / 4) + 16), tm)));
            Value* b_v4 = kb.shared_load_i32x4_indexed(
                Bs, c.b.build_add(b_buf_v4, c.b.build_add(c.i32(k * (kBnp / 4)), tn)));
            std::array<Value*, 8> av{};
            std::array<Value*, 4> bv{};
            for (std::uint32_t u = 0; u < 4; ++u) {
                av[u] = bits_to_f32(c, c.b.build_vextract_lane(a0_v4, u));
                av[4u + u] = bits_to_f32(c, c.b.build_vextract_lane(a1_v4, u));
                bv[u] = bits_to_f32(c, c.b.build_vextract_lane(b_v4, u));
            }
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 4; ++j) {
                    const auto idx = static_cast<std::size_t>(i * 4 + j);
                    accs[idx] = c.b.build_fma_f32(av[static_cast<std::size_t>(i)], bv[static_cast<std::size_t>(j)], accs[idx]);
                }
            }
        }
    };
    (void)a0_base; (void)a1_base; (void)b_base;

    std::vector<Value*> init(33, c.f32(0.0f));
    init[32] = c.i32(0);  // buf

    Value* k_limit = c.b.build_sub(K, c.i32(kBk));
    std::vector<Value*> state = kb.for_range_reduce_n(
        c.i32(0), k_limit, c.i32(kBk), init,
        [&](Value* k0, const std::vector<Value*>& cur) -> std::vector<Value*> {
            std::vector<Value*> cur_accs(cur.begin(), cur.begin() + 32);
            Value* buf = cur[32];
            Value* nxt_buf = c.b.build_xor(buf, c.i32(1));
            auto nxt_regs = load_tiles(c.b.build_add(k0, c.i32(kBk)));
            emit_fma_tile(buf, cur_accs);
            store_tiles(nxt_buf, nxt_regs);
            kb.sync();
            cur_accs.push_back(nxt_buf);
            return cur_accs;
        });

    std::vector<Value*> accs(state.begin(), state.begin() + 32);
    emit_fma_tile(state[32], accs);

    // Epilogue scatter: Y[n, oc, sp] (+ bias[oc]).
    Value* cout_hwo = c.b.build_mul(c.b.build_zext_i64(Cout), c.b.build_zext_i64(HWo));
    Value* hwo64 = c.b.build_zext_i64(HWo);
    for (int j = 0; j < 4; ++j) {
        Value* oc = c.b.build_add(n0, c.b.build_add(c.b.build_shl(tn, c.i32(2)), c.i32(j)));
        kb.if_then(c.b.build_slt(oc, Cout), [&] {
            Value* bv = has_bias ? kb.load_f32(ptr_at(c, bias, oc, 4)) : c.f32(0.0f);
            Value* oc_off = c.b.build_mul(c.b.build_zext_i64(oc), hwo64);
            for (int h = 0; h < 2; ++h) {
                Value* mb = c.b.build_add(m0, c.b.build_add(c.i32(h * 64), c.b.build_shl(tm, c.i32(2))));
                kb.if_then(c.b.build_slt(mb, M), [&] {
                    auto emit_scalar = [&] {
                        for (int i = 0; i < 4; ++i) {
                            Value* m_i = c.b.build_add(mb, c.i32(i));
                            kb.if_then(c.b.build_slt(m_i, M), [&] {
                                Value* ni = c.b.build_udiv(m_i, HWo);
                                Value* pi = c.b.build_umod(m_i, HWo);
                                Value* y_idx = c.b.build_add(
                                    c.b.build_add(c.b.build_mul(c.b.build_zext_i64(ni), cout_hwo), oc_off),
                                    c.b.build_zext_i64(pi));
                                Value* v = c.b.build_add(accs[static_cast<std::size_t>((h * 4 + i) * 4 + j)], bv);
                                kb.store_f32(c.b.build_add(Y, c.b.build_shl(y_idx, kb.const_i64(2))), v);
                            });
                        }
                    };
                    if (vec_store) {
                        emit_if_else_values(
                            c, c.b.build_slt(c.b.build_add(mb, c.i32(3)), M), {},
                            [&]() -> std::vector<Value*> {
                                Value* n_b = c.b.build_udiv(mb, HWo);
                                Value* p_b = c.b.build_umod(mb, HWo);
                                Value* y_idx = c.b.build_add(
                                    c.b.build_add(c.b.build_mul(c.b.build_zext_i64(n_b), cout_hwo), oc_off),
                                    c.b.build_zext_i64(p_b));
                                Value* v4 = make_f32x4(
                                    c,
                                    c.b.build_add(accs[static_cast<std::size_t>((h * 4 + 0) * 4 + j)], bv),
                                    c.b.build_add(accs[static_cast<std::size_t>((h * 4 + 1) * 4 + j)], bv),
                                    c.b.build_add(accs[static_cast<std::size_t>((h * 4 + 2) * 4 + j)], bv),
                                    c.b.build_add(accs[static_cast<std::size_t>((h * 4 + 3) * 4 + j)], bv));
                                kb.vstore_f32x4(c.b.build_add(Y, c.b.build_shl(y_idx, kb.const_i64(2))), v4);
                                return {};
                            },
                            [&]() -> std::vector<Value*> {
                                emit_scalar();
                                return {};
                            });
                    } else {
                        emit_scalar();
                    }
                });
            }
        });
    }

    c.b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

}  // namespace brotensor::kernels
