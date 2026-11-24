#include "mir_select.h"
#include "mir_common.h"
#include "mir_diffusion.h"
#include "mir_math.h"

#include <functional>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr float kFixScale = 72057594037927936.0f;    // 2^56
constexpr float kExpScale = 1099511627776.0f;        // 2^40
constexpr float kInvExpScale = 1.0f / 1099511627776.0f;

Value* pack_u64(Ctx& c, Value* lo32, Value* hi32) {
    Value* hi64 = c.b.build_shl(c.b.build_zext_i64(hi32), c.kb.const_i64(32));
    return c.b.build_or(hi64, c.b.build_zext_i64(lo32));
}

Value* neg_inf(Ctx& c) {
    return jit::gpu_mir::bits_to_f32(c, c.i32(static_cast<int>(0xff800000u)));
}

Value* min_u32(Ctx& c, Value* a, Value* b) {
    return c.b.build_select(c.b.build_ult(a, b), a, b);
}

// Structured while loop carrying arbitrary loop values in block parameters.
std::vector<Value*> emit_while_n(Ctx& c, const std::vector<Value*>& inits,
                                 const std::function<Value*(const std::vector<Value*>&)>& cond_fn,
                                 const std::function<std::vector<Value*>(const std::vector<Value*>&)>& body_fn) {
    BasicBlock* from = c.b.current_block();
    BasicBlock* head = c.b.append_block("while_head");
    BasicBlock* body = c.b.append_block("while_body");
    BasicBlock* exit_bb = c.b.append_block("while_exit");

    std::vector<Value*> accs;
    accs.reserve(inits.size());
    for (Value* init : inits) accs.push_back(c.b.add_block_param(head, init->type()));

    c.b.position_at_end(from);
    c.b.build_br(head, inits);

    c.b.position_at_end(head);
    c.b.build_br_if(cond_fn(accs), body, exit_bb);

    c.b.position_at_end(body);
    std::vector<Value*> next = body_fn(accs);
    c.b.build_br(head, next);

    c.b.position_at_end(exit_bb);
    return accs;
}

struct SelectSmem {
    Value* sf;
    Value* su;
    Value* sq_hi;
    Value* sq_lo;
};

SelectSmem alloc_select_smem(Ctx& c, bool need_sq, bool need_su) {
    SelectSmem s{};
    s.sf = c.kb.shared_alloc_f32(kSelectBlock);
    s.su = need_su ? c.kb.shared_alloc_i32(kSelectBlock) : nullptr;
    s.sq_hi = need_sq ? c.kb.shared_alloc_i32(kSelectBlock) : nullptr;
    s.sq_lo = need_sq ? c.kb.shared_alloc_i32(kSelectBlock) : nullptr;
    return s;
}

Value* wg_max(Ctx& c, Value* s_f, Value* v) {
    Value* tid = c.kb.tid_x();
    c.kb.shared_store_f32_indexed(s_f, tid, v);
    c.kb.sync();
    for (int s = 128; s >= 1; s >>= 1) {
        c.kb.if_then(c.b.build_ult(tid, c.i32(s)), [&] {
            Value* a = c.kb.shared_load_f32_indexed(s_f, tid);
            Value* b = c.kb.shared_load_f32_indexed(s_f, c.b.build_add(tid, c.i32(s)));
            c.kb.shared_store_f32_indexed(s_f, tid, c.kb.fmax(a, b));
        });
        c.kb.sync();
    }
    Value* res = c.kb.shared_load_f32_indexed(s_f, c.i32(0));
    c.kb.sync();
    return res;
}

Value* wg_sum64(Ctx& c, Value* s_hi, Value* s_lo, Value* v64) {
    Value* tid = c.kb.tid_x();
    Value* hi = c.b.build_trunc_i32(c.b.build_lshr(v64, c.kb.const_i64(32)));
    Value* lo = c.b.build_trunc_i32(v64);
    c.kb.shared_store_i32_indexed(s_hi, tid, hi);
    c.kb.shared_store_i32_indexed(s_lo, tid, lo);
    c.kb.sync();
    for (int s = 128; s >= 1; s >>= 1) {
        c.kb.if_then(c.b.build_ult(tid, c.i32(s)), [&] {
            Value* p = c.b.build_add(tid, c.i32(s));
            Value* a = pack_u64(c, c.kb.shared_load_i32_indexed(s_lo, tid), c.kb.shared_load_i32_indexed(s_hi, tid));
            Value* b = pack_u64(c, c.kb.shared_load_i32_indexed(s_lo, p), c.kb.shared_load_i32_indexed(s_hi, p));
            Value* sum = c.b.build_add(a, b);
            c.kb.shared_store_i32_indexed(s_hi, tid, c.b.build_trunc_i32(c.b.build_lshr(sum, c.kb.const_i64(32))));
            c.kb.shared_store_i32_indexed(s_lo, tid, c.b.build_trunc_i32(sum));
        });
        c.kb.sync();
    }
    Value* res = pack_u64(c, c.kb.shared_load_i32_indexed(s_lo, c.i32(0)), c.kb.shared_load_i32_indexed(s_hi, c.i32(0)));
    c.kb.sync();
    return res;
}

Value* wg_argmax(Ctx& c, Value* s_f, Value* s_u, Value* val, Value* idx) {
    Value* tid = c.kb.tid_x();
    c.kb.shared_store_f32_indexed(s_f, tid, val);
    c.kb.shared_store_i32_indexed(s_u, tid, idx);
    c.kb.sync();
    for (int s = 128; s >= 1; s >>= 1) {
        c.kb.if_then(c.b.build_ult(tid, c.i32(s)), [&] {
            Value* p = c.b.build_add(tid, c.i32(s));
            Value* va = c.kb.shared_load_f32_indexed(s_f, tid);
            Value* ia = c.kb.shared_load_i32_indexed(s_u, tid);
            Value* vb = c.kb.shared_load_f32_indexed(s_f, p);
            Value* ib = c.kb.shared_load_i32_indexed(s_u, p);
            Value* better = c.b.build_or(c.b.build_sgt(vb, va),
                                         c.b.build_and(c.b.build_eq(vb, va), c.b.build_ult(ib, ia)));
            c.kb.shared_store_f32_indexed(s_f, tid, c.b.build_select(better, vb, va));
            c.kb.shared_store_i32_indexed(s_u, tid, c.b.build_select(better, ib, ia));
        });
        c.kb.sync();
    }
    Value* res = c.kb.shared_load_i32_indexed(s_u, c.i32(0));
    c.kb.sync();
    return res;
}

Value* fixp_prob(Ctx& c, Value* p) {
    return c.b.build_fptosi_i64_f32(c.b.build_mul(p, c.f32(kFixScale)));
}

Value* sort_key(Ctx& c, Value* p, Value* v) {
    Value* pbits = jit::gpu_mir::f32_to_bits(c, p);
    Value* hi = c.b.build_sub(c.i32(0x7fffffff), pbits);
    return pack_u64(c, v, hi);
}

Value* emit_below(Ctx& c, const SelectSmem& sm, Value* prow, Value* V, Value* key64, bool mass_mode) {
    Value* tid = c.kb.tid_x();
    Value* acc = c.kb.for_range_reduce(tid, V, c.i32(kSelectBlock), c.kb.const_i64(0), [&](Value* v, Value* cur) {
        Value* p = c.kb.load_f32(ptr_at(c, prow, v, 4));
        Value* keep = c.b.build_ule(sort_key(c, p, v), key64);
        Value* add_v = mass_mode ? fixp_prob(c, p) : c.kb.const_i64(1);
        return c.b.build_select(keep, c.b.build_add(cur, add_v), cur);
    });
    return wg_sum64(c, sm.sq_hi, sm.sq_lo, acc);
}

Value* emit_search(Ctx& c, const SelectSmem& sm, Value* prow, Value* V, Value* hi, bool mass_mode, Value* target,
                   bool strict) {
    auto res = emit_while_n(
        c, {c.kb.const_i64(0), hi},
        [&](const std::vector<Value*>& st) { return c.b.build_ult(st[0], st[1]); },
        [&](const std::vector<Value*>& st) -> std::vector<Value*> {
            Value* lo = st[0];
            Value* top = st[1];
            Value* mid = c.b.build_add(lo, c.b.build_lshr(c.b.build_sub(top, lo), c.kb.const_i64(1)));
            Value* b = emit_below(c, sm, prow, V, mid, mass_mode);
            Value* ok = strict ? c.b.build_ugt(b, target) : c.b.build_uge(b, target);
            Value* n_lo = c.b.build_select(ok, lo, c.b.build_add(mid, c.kb.const_i64(1)));
            Value* n_top = c.b.build_select(ok, mid, top);
            return {n_lo, n_top};
        });
    return res[0];
}

Value* emit_row_lse(Ctx& c, const SelectSmem& sm, Value* row_ptr, Value* V) {
    Value* tid = c.kb.tid_x();
    Value* block = c.i32(static_cast<int>(kSelectBlock));
    Value* m0 = c.kb.for_range_reduce(tid, V, block, neg_inf(c), [&](Value* v, Value* cur) {
        return c.kb.fmax(cur, c.kb.load_f32(ptr_at(c, row_ptr, v, 4)));
    });
    Value* m = wg_max(c, sm.sf, m0);
    Value* s0 = c.kb.for_range_reduce(tid, V, block, c.kb.const_i64(0), [&](Value* v, Value* cur) {
        Value* ex = emit_exp_acc(c, c.b.build_sub(c.kb.load_f32(ptr_at(c, row_ptr, v, 4)), m));
        return c.b.build_add(cur, c.b.build_fptosi_i64_f32(c.b.build_mul(ex, c.f32(kExpScale))));
    });
    Value* s = wg_sum64(c, sm.sq_hi, sm.sq_lo, s0);
    return c.b.build_add(m, emit_log_acc(c, c.b.build_mul(c.b.build_sitofp_f32_i64(s), c.f32(kInvExpScale))));
}

Value* okey_f32(Ctx& c, Value* x) {
    Value* bits = jit::gpu_mir::f32_to_bits(c, x);
    Value* neg = c.b.build_ne(c.b.build_lshr(bits, c.i32(31)), c.i32(0));
    return c.b.build_select(neg, c.b.build_not(bits), c.b.build_or(bits, c.i32(static_cast<int>(0x80000000u))));
}

Value* hash_uniform_u01(Ctx& c, Value* seed64, std::uint64_t domain, Value* index64) {
    Value* dom = c.kb.const_i64(static_cast<std::int64_t>(domain));
    Value* x = c.b.build_xor(seed64, c.b.build_add(dom, index64));
    Value* z = c.b.build_add(x, c.kb.const_i64(static_cast<std::int64_t>(0x9E3779B97F4A7C15ULL)));
    return emit_splitmix64_mix_u01(c, z);
}

Value* emit_gumbel(Ctx& c, Value* u) {
    Value* eps = c.f32(1e-10f);
    Value* a = c.b.build_add(u, eps);
    Value* b = c.b.build_add(c.b.build_neg(emit_log_acc(c, a)), eps);
    return c.b.build_neg(emit_log_acc(c, b));
}

}  // namespace

brass::Function* build_select_sample_mir(brass::Module& mod, bool has_counter) {
    Function* fn = mod.create_function(
        has_counter ? "mir_select_sample_ctr" : "mir_select_sample", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: a(0), b(1), d(2), e(3), key_lo(4), key_hi(5), V(6), u1(7), u2(8), top_k(9), rows(10), T(11), top_p(12)
    SelectSmem sm = alloc_select_smem(c, true, true);
    Value* tid = kb.tid_x();
    Value* block = c.i32(static_cast<int>(kSelectBlock));
    Value* key64 = pack_u64(c, p[4], p[5]);
    Value* V = p[6];
    Value* top_k = p[9];
    Value* T = p[11];
    Value* top_p = p[12];
    Value* kmax = c.kb.const_i64(0x7fffffffffffffffLL);

    kb.for_range(kb.ctaid_x(), p[10], kb.nctaid_x(), [&](Value* row) {
        Value* row_off = c.b.build_mul(c.b.build_zext_i64(row), c.b.build_zext_i64(V));
        Value* lrow = c.b.build_add(p[0], c.b.build_shl(row_off, c.kb.const_i64(2)));
        Value* is_greedy = c.b.build_eq(T, c.f32(0.0f));

        emit_if_else_values(
            c, is_greedy, {},
            [&]() -> std::vector<Value*> {
                Value* inv_bi = c.i32(-1);
                auto best = kb.for_range_reduce_n(
                    tid, V, block, {c.f32(-3.4028235e38f), inv_bi}, [&](Value* v, const std::vector<Value*>& cur) {
                        Value* x = kb.load_f32(ptr_at(c, lrow, v, 4));
                        Value* gt = c.b.build_sgt(x, cur[0]);
                        return std::vector<Value*>{c.b.build_select(gt, x, cur[0]), c.b.build_select(gt, v, cur[1])};
                    });
                Value* bi_safe = c.b.build_select(c.b.build_eq(best[1], inv_bi), c.i32(-2), best[1]);
                Value* r = wg_argmax(c, sm.sf, sm.su, best[0], bi_safe);
                kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                    Value* out = c.b.build_select(c.b.build_uge(r, V), c.i32(0), r);
                    kb.store_i32(ptr_at(c, p[2], row, 4), out);
                });
                return {};
            },
            [&]() -> std::vector<Value*> {
                Value* prow = c.b.build_add(p[1], c.b.build_shl(row_off, c.kb.const_i64(2)));
                Value* m0 = kb.for_range_reduce(tid, V, block, c.f32(-3.4028235e38f), [&](Value* v, Value* cur) {
                    return kb.fmax(cur, c.b.build_sdiv(kb.load_f32(ptr_at(c, lrow, v, 4)), T));
                });
                Value* m = wg_max(c, sm.sf, m0);
                Value* es0 = kb.for_range_reduce(tid, V, block, c.kb.const_i64(0), [&](Value* v, Value* cur) {
                    Value* x = c.b.build_sdiv(kb.load_f32(ptr_at(c, lrow, v, 4)), T);
                    Value* ex = emit_exp_acc(c, c.b.build_sub(x, m));
                    kb.store_f32(ptr_at(c, prow, v, 4), ex);
                    return c.b.build_add(cur, c.b.build_fptosi_i64_f32(c.b.build_mul(ex, c.f32(kExpScale))));
                });
                Value* es = wg_sum64(c, sm.sq_hi, sm.sq_lo, es0);
                Value* sum = c.b.build_mul(c.b.build_sitofp_f32_i64(es), c.f32(kInvExpScale));
                Value* inv = c.b.build_select(c.b.build_sgt(sum, c.f32(0.0f)),
                                              c.b.build_sdiv(c.f32(1.0f), sum), c.f32(0.0f));
                kb.for_range(tid, V, block, [&](Value* v) {
                    Value* vp = ptr_at(c, prow, v, 4);
                    kb.store_f32(vp, c.b.build_mul(kb.load_f32(vp), inv));
                });
                kb.sync();

                Value* has_k = c.b.build_and(c.b.build_ugt(top_k, c.i32(0)), c.b.build_ult(top_k, V));
                Value* bound = c.b.build_select(has_k, top_k, V);
                Value* kk0 = emit_if_else_values(
                    c, c.b.build_ult(bound, V), {Type::i64()},
                    [&]() -> std::vector<Value*> {
                        return {emit_search(c, sm, prow, V, kmax, false, c.b.build_zext_i64(bound), false)};
                    },
                    [&]() -> std::vector<Value*> { return {kmax}; })[0];

                Value* kk = emit_if_else_values(
                    c, c.b.build_slt(top_p, c.f32(1.0f)), {Type::i64()},
                    [&]() -> std::vector<Value*> {
                        Value* tp = fixp_prob(c, top_p);
                        Value* b0 = emit_below(c, sm, prow, V, kk0, true);
                        return emit_if_else_values(
                            c, c.b.build_uge(b0, tp), {Type::i64()},
                            [&]() -> std::vector<Value*> { return {emit_search(c, sm, prow, V, kk0, true, tp, false)}; },
                            [&]() -> std::vector<Value*> { return {kk0}; });
                    },
                    [&]() -> std::vector<Value*> { return {kk0}; })[0];

                Value* kept = emit_below(c, sm, prow, V, kk, true);
                Value* base = has_counter ? c.b.build_zext_i64(kb.load_i32(p[3])) : pack_u64(c, p[7], p[8]);
                Value* k0 = c.b.build_trunc_i32(key64);
                Value* k1 = c.b.build_trunc_i32(c.b.build_lshr(key64, c.kb.const_i64(32)));
                Value* u24 = c.b.build_lshr(philox_words(c, k0, k1, c.b.build_add(base, c.b.build_zext_i64(row))).first,
                                            c.i32(8));
                Value* u24_64 = c.b.build_zext_i64(u24);
                Value* chosen = emit_if_else_values(
                    c, c.b.build_ugt(kept, c.kb.const_i64(0)), {Type::i64()},
                    [&]() -> std::vector<Value*> {
                        Value* hi_part = c.b.build_mul(c.b.build_lshr(kept, c.kb.const_i64(24)), u24_64);
                        Value* lo_part = c.b.build_lshr(
                            c.b.build_mul(c.b.build_and(kept, c.kb.const_i64(0xffffff)), u24_64), c.kb.const_i64(24));
                        Value* target = c.b.build_add(hi_part, lo_part);
                        Value* b_kk = emit_below(c, sm, prow, V, kk, true);
                        return emit_if_else_values(
                            c, c.b.build_ugt(b_kk, target), {Type::i64()},
                            [&]() -> std::vector<Value*> {
                                return {emit_search(c, sm, prow, V, kk, true, target, true)};
                            },
                            [&]() -> std::vector<Value*> { return {kk}; });
                    },
                    [&]() -> std::vector<Value*> {
                        return {emit_search(c, sm, prow, V, kmax, false, c.kb.const_i64(1), false)};
                    })[0];

                kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
                    kb.store_i32(ptr_at(c, p[2], row, 4), c.b.build_trunc_i32(chosen));
                });
                return {};
            });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_select_counter_mir(brass::Module& mod) {
    Function* fn = mod.create_function("mir_select_counter", Type::void_type(), {Type::ptr(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    kb.if_then(c.b.build_eq(kb.global_tid_x(), c.i32(0)), [&] {
        kb.store_i32(p[0], c.b.build_add(kb.load_i32(p[0]), p[1]));
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_select_commit_mir(brass::Module& mod) {
    Function* fn = mod.create_function(
        "mir_select_commit", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    emit_vec_grid_stride(c, p[4], 1, kSelectBlock, [&](Value* i) {
        Value* idx = kb.load_i32(ptr_at(c, p[1], i, 4));
        Value* inb = c.b.build_and(c.b.build_sge(idx, c.i32(0)), c.b.build_slt(idx, p[5]));
        kb.if_then(inb, [&] {
            kb.store_i32(ptr_at(c, p[2], idx, 4), kb.load_i32(ptr_at(c, p[0], idx, 4)));
            kb.store_i32(ptr_at(c, p[3], idx, 4), p[6]);
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

brass::Function* build_select_md_mir(brass::Module& mod, bool guided, bool has_class_temp, bool has_pos_temp) {
    std::string name = "mir_select_md_" + std::to_string(guided ? 1 : 0) + "_" +
                       std::to_string(has_class_temp ? 1 : 0) + "_" + std::to_string(has_pos_temp ? 1 : 0);
    Function* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::f32(), Type::f32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: a(0) b(1) c(2) d(3) e(4) f(5) key_lo(6) key_hi(7) T(8) C(9) V(10) mask_id(11) k_keep(12)
    //    guidance(13) layer_penalty(14) pos_t(15) class_t(16)
    SelectSmem sm = alloc_select_smem(c, true, true);
    Value* tid = kb.tid_x();
    Value* block = c.i32(static_cast<int>(kSelectBlock));
    Value* key64 = pack_u64(c, p[6], p[7]);
    Value* T = p[8];
    Value* C = p[9];
    Value* V = p[10];
    Value* mask_id = p[11];
    Value* k_keep = p[12];
    Value* ncells = c.b.build_mul(C, T);
    Value* stride = c.b.build_mul(c.b.build_zext_i64(C), c.b.build_zext_i64(V));
    Value* ninf = neg_inf(c);

    kb.for_range(kb.ctaid_x(), ncells, kb.nctaid_x(), [&](Value* cell) {
        Value* ch = c.b.build_udiv(cell, T);
        Value* fr = c.b.build_sub(cell, c.b.build_mul(ch, T));
        Value* ch_v = c.b.build_mul(c.b.build_zext_i64(ch), c.b.build_zext_i64(V));
        Value* cond_off = c.b.build_add(c.b.build_mul(c.b.build_zext_i64(fr), stride), ch_v);
        Value* cond = c.b.build_add(p[0], c.b.build_shl(cond_off, c.kb.const_i64(2)));
        Value* lp_off = c.b.build_mul(c.b.build_zext_i64(cell), c.b.build_zext_i64(V));
        Value* lp = c.b.build_add(p[2], c.b.build_shl(lp_off, c.kb.const_i64(2)));

        if (guided) {
            Value* unc_off = c.b.build_add(c.b.build_mul(c.b.build_zext_i64(c.b.build_add(T, fr)), stride), ch_v);
            Value* unc = c.b.build_add(p[0], c.b.build_shl(unc_off, c.kb.const_i64(2)));
            Value* lc = emit_row_lse(c, sm, cond, V);
            Value* lu = emit_row_lse(c, sm, unc, V);
            kb.for_range(tid, V, block, [&](Value* v) {
                Value* av = c.b.build_sub(kb.load_f32(ptr_at(c, cond, v, 4)), lc);
                Value* bv = c.b.build_sub(kb.load_f32(ptr_at(c, unc, v, 4)), lu);
                Value* g = c.b.build_add(av, c.b.build_mul(p[13], c.b.build_sub(av, bv)));
                kb.store_f32(ptr_at(c, lp, v, 4), g);
            });
            kb.sync();
            Value* lg = emit_row_lse(c, sm, lp, V);
            kb.for_range(tid, V, block, [&](Value* v) {
                Value* ptr = ptr_at(c, lp, v, 4);
                kb.store_f32(ptr, c.b.build_sub(kb.load_f32(ptr), lg));
            });
        } else {
            Value* l = emit_row_lse(c, sm, cond, V);
            kb.for_range(tid, V, block, [&](Value* v) {
                kb.store_f32(ptr_at(c, lp, v, 4), c.b.build_sub(kb.load_f32(ptr_at(c, cond, v, 4)), l));
            });
        }
        kb.sync();
        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] { kb.store_f32(ptr_at(c, lp, mask_id, 4), ninf); });
        kb.sync();

        Value* cmax = kb.for_range_reduce(tid, V, block, ninf, [&](Value* v, Value* cur) {
            return kb.fmax(cur, kb.load_f32(ptr_at(c, lp, v, 4)));
        });
        Value* conf = wg_max(c, sm.sf, cmax);

        Value* best = nullptr;
        if (has_class_temp) {
            auto rng = emit_while_n(
                c, {c.kb.const_i64(0), c.kb.const_i64(0xffffffffLL)},
                [&](const std::vector<Value*>& st) { return c.b.build_ult(st[0], st[1]); },
                [&](const std::vector<Value*>& st) -> std::vector<Value*> {
                    Value* lo = st[0];
                    Value* hi = st[1];
                    Value* mid = c.b.build_add(
                        lo, c.b.build_lshr(c.b.build_add(c.b.build_sub(hi, lo), c.kb.const_i64(1)), c.kb.const_i64(1)));
                    Value* n0 = kb.for_range_reduce(tid, V, block, c.kb.const_i64(0), [&](Value* v, Value* cur) {
                        Value* kx = c.b.build_zext_i64(okey_f32(c, kb.load_f32(ptr_at(c, lp, v, 4))));
                        return c.b.build_add(cur, c.b.build_select(c.b.build_uge(kx, mid),
                                                                   c.kb.const_i64(1), c.kb.const_i64(0)));
                    });
                    Value* n = wg_sum64(c, sm.sq_hi, sm.sq_lo, n0);
                    Value* ok = c.b.build_uge(n, c.b.build_zext_i64(k_keep));
                    return {c.b.build_select(ok, mid, lo), c.b.build_select(ok, hi, c.b.build_sub(mid, c.kb.const_i64(1)))};
                });
            Value* thr = c.b.build_trunc_i32(rng[0]);
            Value* ng0 = kb.for_range_reduce(tid, V, block, c.kb.const_i64(0), [&](Value* v, Value* cur) {
                Value* kx = okey_f32(c, kb.load_f32(ptr_at(c, lp, v, 4)));
                return c.b.build_add(cur, c.b.build_select(c.b.build_ugt(kx, thr),
                                                           c.kb.const_i64(1), c.kb.const_i64(0)));
            });
            Value* ng = wg_sum64(c, sm.sq_hi, sm.sq_lo, ng0);
            Value* m_rem = c.b.build_sub(k_keep, c.b.build_trunc_i32(ng));
            Value* chunk = c.b.build_udiv(c.b.build_add(V, c.i32(255)), block);
            Value* v0 = min_u32(c, c.b.build_mul(tid, chunk), V);
            Value* v1 = min_u32(c, c.b.build_add(v0, chunk), V);
            Value* eq = kb.for_range_reduce(v0, v1, c.i32(1), c.i32(0), [&](Value* v, Value* cur) {
                Value* kx = okey_f32(c, kb.load_f32(ptr_at(c, lp, v, 4)));
                return c.b.build_add(cur, c.b.build_select(c.b.build_eq(kx, thr), c.i32(1), c.i32(0)));
            });
            kb.shared_store_i32_indexed(sm.su, tid, eq);
            kb.sync();
            Value* before0 = kb.for_range_reduce(c.i32(0), tid, c.i32(1), c.i32(0), [&](Value* j, Value* cur) {
                return c.b.build_add(cur, kb.shared_load_i32_indexed(sm.su, j));
            });
            kb.sync();
            Value* base_idx = c.b.build_mul(c.b.build_zext_i64(cell), c.b.build_zext_i64(V));
            Value* inv_bi = c.i32(-1);
            auto arg = kb.for_range_reduce_n(
                v0, v1, c.i32(1), {ninf, inv_bi, before0}, [&](Value* v, const std::vector<Value*>& st) {
                    Value* bs = st[0];
                    Value* bi = st[1];
                    Value* bef = st[2];
                    Value* x = kb.load_f32(ptr_at(c, lp, v, 4));
                    Value* kx = okey_f32(c, x);
                    Value* gt_thr = c.b.build_ugt(kx, thr);
                    Value* eq_thr = c.b.build_eq(kx, thr);
                    Value* kept_eq = c.b.build_and(eq_thr, c.b.build_ult(bef, m_rem));
                    Value* kept = c.b.build_or(gt_thr, kept_eq);
                    Value* next_bef = c.b.build_select(eq_thr, c.b.build_add(bef, c.i32(1)), bef);
                    Value* score = emit_if_else_values(
                        c, kept, {Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* u = hash_uniform_u01(c, key64, 0x14057B7EF767814FULL,
                                                        c.b.build_add(base_idx, c.b.build_zext_i64(v)));
                            Value* q = c.b.build_sdiv(x, p[16]);
                            return {c.b.build_add(q, emit_gumbel(c, u))};
                        },
                        [&]() -> std::vector<Value*> { return {ninf}; })[0];
                    Value* win = c.b.build_or(c.b.build_eq(bi, inv_bi), c.b.build_sgt(score, bs));
                    return std::vector<Value*>{c.b.build_select(win, score, bs), c.b.build_select(win, v, bi), next_bef};
                });
            Value* bi_safe = c.b.build_select(c.b.build_eq(arg[1], inv_bi), c.i32(-2), arg[1]);
            best = wg_argmax(c, sm.sf, sm.su, arg[0], bi_safe);
        } else {
            Value* inv_bi = c.i32(-1);
            auto arg = kb.for_range_reduce_n(
                tid, V, block, {ninf, inv_bi}, [&](Value* v, const std::vector<Value*>& st) {
                    Value* x = kb.load_f32(ptr_at(c, lp, v, 4));
                    Value* win = c.b.build_or(c.b.build_eq(st[1], inv_bi), c.b.build_sgt(x, st[0]));
                    return std::vector<Value*>{c.b.build_select(win, x, st[0]), c.b.build_select(win, v, st[1])};
                });
            Value* bi_safe = c.b.build_select(c.b.build_eq(arg[1], inv_bi), c.i32(-2), arg[1]);
            best = wg_argmax(c, sm.sf, sm.su, arg[0], bi_safe);
        }

        kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            kb.store_i32(ptr_at(c, p[3], cell, 4), c.b.build_select(c.b.build_uge(best, V), c.i32(0), best));
            kb.store_f32(ptr_at(c, p[5], cell, 4), conf);
            Value* is_mask = c.b.build_eq(kb.load_i32(ptr_at(c, p[1], cell, 4)), mask_id);
            Value* s = emit_if_else_values(
                c, is_mask, {Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* sv = c.b.build_sub(conf, c.b.build_mul(kb.u32_to_f32(ch), p[14]));
                    if (has_pos_temp) {
                        Value* u = hash_uniform_u01(c, key64, 0x5851F42D4C957F2DULL, c.b.build_zext_i64(cell));
                        sv = c.b.build_add(c.b.build_sdiv(sv, p[15]), emit_gumbel(c, u));
                    }
                    return {sv};
                },
                [&]() -> std::vector<Value*> { return {ninf}; })[0];
            kb.store_f32(ptr_at(c, p[4], cell, 4), s);
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
