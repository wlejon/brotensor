#include "mir_gemm_simt.h"

#include "mir_common.h"
#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <algorithm>
#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr std::uint32_t kSimtWg = 256;
constexpr std::uint32_t kSimtBk = 16;

struct TileDims {
    std::uint32_t bm, bn, tm, tn, la, lb, pa, pb, ntx;
    int bn_shift;
};

TileDims make_tile_dims(bool big128) {
    TileDims d{};
    d.bm = big128 ? 128u : 64u;
    d.bn = big128 ? 128u : 64u;
    d.tm = big128 ? 8u : 4u;
    d.tn = big128 ? 8u : 4u;
    d.la = (d.bm * kSimtBk) / kSimtWg;
    d.lb = (d.bn * kSimtBk) / kSimtWg;
    d.pa = d.bm + 4u;
    d.pb = d.bn + 4u;
    d.ntx = d.bn / d.tn;
    d.bn_shift = big128 ? 7 : 6;
    return d;
}

Value* ptr_offset_i64(Ctx& c, Value* base, Value* elem_off_i64, int esz) {
    return c.b.build_add(base, c.b.build_mul(elem_off_i64, c.kb.const_i64(esz)));
}

Value* batch_base_ptr(Ctx& c, Value* base, Value* z, Value* stride, int esz) {
    Value* off = c.b.build_mul(c.b.build_zext_i64(z), c.b.build_zext_i64(stride));
    return ptr_offset_i64(c, base, off, esz);
}

template <typename LoadFn, typename StoreFn>
std::vector<Value*> run_simt_k_loop(Ctx& c, const TileDims& d, Value* k_dim,
                                    Value* As, Value* Bs,
                                    LoadFn&& load_tiles, StoreFn&& store_tiles) {
    const std::size_t n_acc = static_cast<std::size_t>(d.tm) * d.tn;
    const std::size_t n_reg = static_cast<std::size_t>(d.la) + d.lb;
    Value* nk = c.b.build_lshr(c.b.build_add(k_dim, c.i32(static_cast<int>(kSimtBk - 1u))), c.i32(4));
    Value* tid = c.kb.tid_x();
    Value* tx = c.b.build_umod(tid, c.i32(static_cast<int>(d.ntx)));
    Value* ty = c.b.build_udiv(tid, c.i32(static_cast<int>(d.ntx)));
    Value* tx_tn = c.b.build_mul(tx, c.i32(static_cast<int>(d.tn)));
    Value* ty_tm = c.b.build_mul(ty, c.i32(static_cast<int>(d.tm)));

    std::vector<Type> reg_types(n_reg, Type::f32());
    std::vector<Value*> reg0 = emit_if_else_values(
        c, c.b.build_sgt(nk, c.i32(0)), reg_types,
        [&]() -> std::vector<Value*> { return load_tiles(c.i32(0)); },
        [&]() -> std::vector<Value*> { return std::vector<Value*>(n_reg, c.f32(0.0f)); });

    std::vector<Value*> init = std::move(reg0);
    init.resize(n_reg + n_acc, c.f32(0.0f));

    std::vector<Value*> fin = c.kb.for_range_reduce_n(
        c.i32(0), nk, c.i32(1), init,
        [&](Value* kt, const std::vector<Value*>& cur) -> std::vector<Value*> {
            store_tiles(cur);
            c.kb.sync();

            Value* kt1 = c.b.build_add(kt, c.i32(1));
            std::vector<Value*> next_reg = emit_if_else_values(
                c, c.b.build_ult(kt1, nk), reg_types,
                [&]() -> std::vector<Value*> { return load_tiles(c.b.build_shl(kt1, c.i32(4))); },
                [&]() -> std::vector<Value*> { return std::vector<Value*>(n_reg, c.f32(0.0f)); });

            std::vector<Value*> acc(cur.begin() + static_cast<std::ptrdiff_t>(n_reg), cur.end());
            for (std::uint32_t kk = 0; kk < kSimtBk; ++kk) {
                Value* a_base = c.b.build_add(c.i32(static_cast<int>(kk * d.pa)), ty_tm);
                Value* b_base = c.b.build_add(c.i32(static_cast<int>(kk * d.pb)), tx_tn);
                std::vector<Value*> av(d.tm), bv(d.tn);
                for (std::uint32_t i = 0; i < d.tm; ++i) {
                    av[i] = c.kb.shared_load_f32_indexed(As, c.b.build_add(a_base, c.i32(static_cast<int>(i))));
                }
                for (std::uint32_t j = 0; j < d.tn; ++j) {
                    bv[j] = c.kb.shared_load_f32_indexed(Bs, c.b.build_add(b_base, c.i32(static_cast<int>(j))));
                }
                for (std::uint32_t i = 0; i < d.tm; ++i) {
                    for (std::uint32_t j = 0; j < d.tn; ++j) {
                        std::size_t idx = static_cast<std::size_t>(i) * d.tn + j;
                        acc[idx] = c.b.build_fma_f32(av[i], bv[j], acc[idx]);
                    }
                }
            }
            c.kb.sync();

            std::vector<Value*> out = std::move(next_reg);
            out.insert(out.end(), acc.begin(), acc.end());
            return out;
        });

    return std::vector<Value*>(fin.begin() + static_cast<std::ptrdiff_t>(n_reg), fin.end());
}

}  // namespace

brass::Function* build_gemm_simt_mir(brass::Module& mod, Dtype dt_a, Dtype dt_b, Dtype dt_c, Dtype dt_bias,
                                     bool big128, std::uint32_t epi, std::uint32_t act,
                                     bool ta, bool nb, bool has_bias) {
    const TileDims d = make_tile_dims(big128);
    const std::string name = std::string("mir_gemm_simt_") + (big128 ? "128_" : "64_") +
                             (ta ? "t" : "n") + (nb ? "t" : "n") + "_e" + std::to_string(epi) +
                             "_a" + std::to_string(act) + (has_bias ? "_b" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: a(0) b(1) c(2) bias(3) m(4) n(5) k(6) lda(7) ldb(8) ldc(9) sa(10) sb(11) sc(12)

    Value* As = kb.shared_alloc_f32(kSimtBk * d.pa);
    Value* Bs = kb.shared_alloc_f32(kSimtBk * d.pb);
    Value* z = kb.ctaid_z();
    Value* A = batch_base_ptr(c, p[0], z, p[10], elem_bytes(dt_a));
    Value* B = batch_base_ptr(c, p[1], z, p[11], elem_bytes(dt_b));
    Value* C = batch_base_ptr(c, p[2], z, p[12], elem_bytes(dt_c));
    Value* m0 = c.b.build_mul(kb.ctaid_y(), c.i32(static_cast<int>(d.bm)));
    Value* n0 = c.b.build_mul(kb.ctaid_x(), c.i32(static_cast<int>(d.bn)));
    Value* tid = kb.tid_x();

    auto load_tiles = [&](Value* k0) -> std::vector<Value*> {
        std::vector<Value*> regs(d.la + d.lb);
        for (std::uint32_t j = 0; j < d.la; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* m = ta ? c.b.build_umod(e, c.i32(static_cast<int>(d.bm)))
                          : c.b.build_lshr(e, c.i32(4));
            Value* k = ta ? c.b.build_udiv(e, c.i32(static_cast<int>(d.bm)))
                          : c.b.build_and(e, c.i32(15));
            Value* gm = c.b.build_add(m0, m);
            Value* gk = c.b.build_add(k0, k);
            Value* ok = c.b.build_and(c.b.build_ult(gm, p[4]), c.b.build_ult(gk, p[6]));
            regs[j] = emit_if_else_values(
                c, ok, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* idx = ta ? c.b.build_add(c.b.build_mul(gk, p[7]), gm)
                                    : c.b.build_add(c.b.build_mul(gm, p[7]), gk);
                    return {load_elem_dt(c, dt_a, A, idx)};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
        }
        for (std::uint32_t j = 0; j < d.lb; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* n = nb ? c.b.build_umod(e, c.i32(static_cast<int>(d.bn)))
                          : c.b.build_lshr(e, c.i32(4));
            Value* k = nb ? c.b.build_udiv(e, c.i32(static_cast<int>(d.bn)))
                          : c.b.build_and(e, c.i32(15));
            Value* gn = c.b.build_add(n0, n);
            Value* gk = c.b.build_add(k0, k);
            Value* ok = c.b.build_and(c.b.build_ult(gn, p[5]), c.b.build_ult(gk, p[6]));
            regs[d.la + j] = emit_if_else_values(
                c, ok, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* idx = nb ? c.b.build_add(c.b.build_mul(gk, p[8]), gn)
                                    : c.b.build_add(c.b.build_mul(gn, p[8]), gk);
                    return {load_elem_dt(c, dt_b, B, idx)};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
        }
        return regs;
    };

    auto store_tiles = [&](const std::vector<Value*>& regs) {
        for (std::uint32_t j = 0; j < d.la; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* idx = ta ? c.b.build_add(c.b.build_mul(c.b.build_udiv(e, c.i32(static_cast<int>(d.bm))),
                                                          c.i32(static_cast<int>(d.pa))),
                                            c.b.build_umod(e, c.i32(static_cast<int>(d.bm))))
                            : c.b.build_add(c.b.build_mul(c.b.build_and(e, c.i32(15)), c.i32(static_cast<int>(d.pa))),
                                            c.b.build_lshr(e, c.i32(4)));
            kb.shared_store_f32_indexed(As, idx, regs[j]);
        }
        for (std::uint32_t j = 0; j < d.lb; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* idx = nb ? c.b.build_add(c.b.build_mul(c.b.build_udiv(e, c.i32(static_cast<int>(d.bn))),
                                                          c.i32(static_cast<int>(d.pb))),
                                            c.b.build_umod(e, c.i32(static_cast<int>(d.bn))))
                            : c.b.build_add(c.b.build_mul(c.b.build_and(e, c.i32(15)), c.i32(static_cast<int>(d.pb))),
                                            c.b.build_lshr(e, c.i32(4)));
            kb.shared_store_f32_indexed(Bs, idx, regs[d.la + j]);
        }
    };

    std::vector<Value*> acc = run_simt_k_loop(c, d, p[6], As, Bs, load_tiles, store_tiles);
    Value* tx = c.b.build_umod(tid, c.i32(static_cast<int>(d.ntx)));
    Value* ty = c.b.build_udiv(tid, c.i32(static_cast<int>(d.ntx)));

    for (std::uint32_t i = 0; i < d.tm; ++i) {
        Value* gm = c.b.build_add(m0, c.b.build_add(c.b.build_mul(ty, c.i32(static_cast<int>(d.tm))),
                                                    c.i32(static_cast<int>(i))));
        kb.if_then(c.b.build_ult(gm, p[4]), [&] {
            for (std::uint32_t j = 0; j < d.tn; ++j) {
                Value* gn = c.b.build_add(n0, c.b.build_add(c.b.build_mul(tx, c.i32(static_cast<int>(d.tn))),
                                                            c.i32(static_cast<int>(j))));
                kb.if_then(c.b.build_ult(gn, p[5]), [&] {
                    Value* v = acc[static_cast<std::size_t>(i) * d.tn + j];
                    if (has_bias) v = c.b.build_add(v, load_elem_dt(c, dt_bias, p[3], gn));
                    v = emit_gemm_act(c, act, v);
                    Value* o = c.b.build_add(c.b.build_mul(gm, p[9]), gn);
                    if (epi == EPI_ACCUM) v = c.b.build_add(v, load_elem_dt(c, dt_c, C, o));
                    store_elem_dt(c, dt_c, C, o, v);
                });
            }
        });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_conv_simt_mir(brass::Module& mod, Dtype dt, bool big128, bool accum, bool has_bias,
                                     int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    const TileDims d = make_tile_dims(big128);
    const int khw = kh * kw;
    const std::string name = std::string("mir_conv_simt_") + (big128 ? "128_" : "64_") +
                             std::to_string(kh) + "x" + std::to_string(kw) +
                             "_s" + std::to_string(sh) + "x" + std::to_string(sw) +
                             "_p" + std::to_string(ph) + "x" + std::to_string(pw) +
                             "_d" + std::to_string(dh) + "x" + std::to_string(dw) +
                             (accum ? "_ac" : "") + (has_bias ? "_b" : "");
    std::vector<Type> params = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
        Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: x(0) w(1) y(2) bias(3) m(4) n(5) k(6) h(7) wd(8) wo(9) cin(10) cout(11) groups(12)

    const int esz = elem_bytes(dt);
    Value* As = kb.shared_alloc_f32(kSimtBk * d.pa);
    Value* Bs = kb.shared_alloc_f32(kSimtBk * d.pb);
    Value* z = kb.ctaid_z();
    Value* img = c.b.build_udiv(z, p[12]);
    Value* g = c.b.build_umod(z, p[12]);
    Value* cgi = c.b.build_udiv(p[10], p[12]);

    Value* x_chan = c.b.build_add(c.b.build_mul(c.b.build_zext_i64(img), c.b.build_zext_i64(p[10])),
                                  c.b.build_mul(c.b.build_zext_i64(g), c.b.build_zext_i64(cgi)));
    Value* X = ptr_offset_i64(c, p[0], c.b.build_mul(x_chan, c.b.build_mul(c.b.build_zext_i64(p[7]),
                                                                            c.b.build_zext_i64(p[8]))), esz);
    Value* A = ptr_offset_i64(c, p[1], c.b.build_mul(c.b.build_zext_i64(g),
                                                     c.b.build_mul(c.b.build_zext_i64(p[4]),
                                                                   c.b.build_zext_i64(p[6]))), esz);
    Value* c_chan = c.b.build_add(c.b.build_mul(c.b.build_zext_i64(img), c.b.build_zext_i64(p[11])),
                                  c.b.build_mul(c.b.build_zext_i64(g), c.b.build_zext_i64(p[4])));
    Value* C = ptr_offset_i64(c, p[2], c.b.build_mul(c_chan, c.b.build_zext_i64(p[5])), esz);

    Value* m0 = c.b.build_mul(kb.ctaid_y(), c.i32(static_cast<int>(d.bm)));
    Value* n0 = c.b.build_mul(kb.ctaid_x(), c.i32(static_cast<int>(d.bn)));
    Value* tid = kb.tid_x();

    Value* px = c.b.build_add(n0, c.b.build_and(tid, c.i32(static_cast<int>(d.bn - 1u))));
    Value* px_ok = c.b.build_ult(px, p[5]);
    Value* px_ih = c.b.build_sub(c.b.build_mul(c.b.build_udiv(px, p[9]), c.i32(sh)), c.i32(ph));
    Value* px_iw = c.b.build_sub(c.b.build_mul(c.b.build_umod(px, p[9]), c.i32(sw)), c.i32(pw));

    auto load_tiles = [&](Value* k0) -> std::vector<Value*> {
        std::vector<Value*> regs(d.la + d.lb);
        for (std::uint32_t j = 0; j < d.la; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* gm = c.b.build_add(m0, c.b.build_lshr(e, c.i32(4)));
            Value* gk = c.b.build_add(k0, c.b.build_and(e, c.i32(15)));
            Value* ok = c.b.build_and(c.b.build_ult(gm, p[4]), c.b.build_ult(gk, p[6]));
            regs[j] = emit_if_else_values(
                c, ok, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    return {load_elem_dt(c, dt, A, c.b.build_add(c.b.build_mul(gm, p[6]), gk))};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
        }
        for (std::uint32_t j = 0; j < d.lb; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* gk = c.b.build_add(k0, c.b.build_lshr(e, c.i32(d.bn_shift)));
            Value* ic = c.b.build_udiv(gk, c.i32(khw));
            Value* t = c.b.build_umod(gk, c.i32(khw));
            Value* ih = c.b.build_add(px_ih, c.b.build_mul(c.b.build_udiv(t, c.i32(kw)), c.i32(dh)));
            Value* iw = c.b.build_add(px_iw, c.b.build_mul(c.b.build_umod(t, c.i32(kw)), c.i32(dw)));
            Value* in_h = c.b.build_and(c.b.build_sge(ih, c.i32(0)), c.b.build_slt(ih, p[7]));
            Value* in_w = c.b.build_and(c.b.build_sge(iw, c.i32(0)), c.b.build_slt(iw, p[8]));
            Value* ok = c.b.build_and(c.b.build_and(px_ok, c.b.build_ult(gk, p[6])), c.b.build_and(in_h, in_w));
            regs[d.la + j] = emit_if_else_values(
                c, ok, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* xidx = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(ic, p[7]), ih), p[8]), iw);
                    return {load_elem_dt(c, dt, X, xidx)};
                },
                [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
        }
        return regs;
    };

    auto store_tiles = [&](const std::vector<Value*>& regs) {
        for (std::uint32_t j = 0; j < d.la; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* idx = c.b.build_add(c.b.build_mul(c.b.build_and(e, c.i32(15)), c.i32(static_cast<int>(d.pa))),
                                       c.b.build_lshr(e, c.i32(4)));
            kb.shared_store_f32_indexed(As, idx, regs[j]);
        }
        for (std::uint32_t j = 0; j < d.lb; ++j) {
            Value* e = c.b.build_add(tid, c.i32(static_cast<int>(j * kSimtWg)));
            Value* idx = c.b.build_add(c.b.build_mul(c.b.build_lshr(e, c.i32(d.bn_shift)),
                                                     c.i32(static_cast<int>(d.pb))),
                                       c.b.build_and(e, c.i32(static_cast<int>(d.bn - 1u))));
            kb.shared_store_f32_indexed(Bs, idx, regs[d.la + j]);
        }
    };

    std::vector<Value*> acc = run_simt_k_loop(c, d, p[6], As, Bs, load_tiles, store_tiles);
    Value* tx = c.b.build_umod(tid, c.i32(static_cast<int>(d.ntx)));
    Value* ty = c.b.build_udiv(tid, c.i32(static_cast<int>(d.ntx)));
    Value* gm_bias = c.b.build_mul(g, p[4]);

    for (std::uint32_t i = 0; i < d.tm; ++i) {
        Value* gm = c.b.build_add(m0, c.b.build_add(c.b.build_mul(ty, c.i32(static_cast<int>(d.tm))),
                                                    c.i32(static_cast<int>(i))));
        kb.if_then(c.b.build_ult(gm, p[4]), [&] {
            Value* bv = has_bias ? load_elem_dt(c, dt, p[3], c.b.build_add(gm_bias, gm)) : c.f32(0.0f);
            for (std::uint32_t j = 0; j < d.tn; ++j) {
                Value* gn = c.b.build_add(n0, c.b.build_add(c.b.build_mul(tx, c.i32(static_cast<int>(d.tn))),
                                                            c.i32(static_cast<int>(j))));
                kb.if_then(c.b.build_ult(gn, p[5]), [&] {
                    Value* v = c.b.build_add(acc[static_cast<std::size_t>(i) * d.tn + j], bv);
                    Value* o = c.b.build_add(c.b.build_mul(gm, p[5]), gn);
                    if (accum) v = c.b.build_add(v, load_elem_dt(c, dt, C, o));
                    store_elem_dt(c, dt, C, o, v);
                });
            }
        });
    }

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
