#include "mir_xent.h"

#include "mir_math.h"

#include <string>

namespace brotensor::kernels {

namespace {

Value* clamp_i(Ctx& c, Value* v, Value* lo, Value* hi) {
    return c.b.build_select(c.b.build_slt(v, lo), lo, c.b.build_select(c.b.build_sgt(v, hi), hi, v));
}

Value* is_valid(Ctx& c, Value* mask, Value* at, bool has_mask) {
    if (!has_mask) return nullptr;
    return c.b.build_sge(c.kb.load_f32(ptr_at(c, mask, at, 4)), c.f32(0.5f));
}

Value* emit_log1p_acc(Ctx& c, Value* x) {
    Value* u = c.b.build_add(c.f32(1.0f), x);
    Value* exact1 = c.b.build_eq(u, c.f32(1.0f));
    Value* log_u = emit_log_acc(c, c.b.build_select(exact1, c.f32(1.0f), u));
    Value* ratio = c.b.build_select(exact1, c.f32(1.0f), emit_div_rn(c, x, c.b.build_sub(u, c.f32(1.0f))));
    return c.b.build_select(exact1, x, c.b.build_mul(log_u, ratio));
}

}  // namespace

brass::Function* build_xent_mir(brass::Module& mod, std::uint32_t mode, bool has_mask, bool has_off) {
    const std::string name = "mir_xent_" + std::to_string(mode) + "_m" + std::to_string(has_mask ? 1 : 0) +
                             "_o" + std::to_string(has_off ? 1 : 0);
    auto* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *logits = p[0], *target = p[1], *mask = p[2], *off = p[3];
    Value *probs = p[4], *dlog = p[5], *loss_buf = p[6];
    Value *rows = p[7], *n = p[8], *heads = p[9], *pos_w = p[10];

    Value* tid = c.kb.tid_x();
    Value* scratch8 = c.kb.shared_alloc_f32(8);
    Value* bx = c.kb.ctaid_x();
    Value* nbx = c.kb.nctaid_x();
    Value* block_sz = c.i32(static_cast<int>(jit::gpu_mir::kEwBlock));

    c.kb.for_range(bx, rows, nbx, [&](Value* b) {
        Value* ro = c.b.build_mul(b, n);
        Value* row_loss = c.f32(0.0f);
        if (mode == 1u) {
            row_loss = c.kb.for_range_reduce(tid, n, block_sz, c.f32(0.0f), [&](Value* i, Value* acc) {
                Value* at = c.b.build_add(ro, i);
                Value* p_at = ptr_at(c, probs, at, 4);
                Value* d_at = ptr_at(c, dlog, at, 4);
                auto eval = [&]() -> Value* {
                    Value* z = c.kb.load_f32(ptr_at(c, logits, at, 4));
                    Value* y = c.kb.load_f32(ptr_at(c, target, at, 4));
                    Value* e = emit_exp_acc(c, c.b.build_neg(c.kb.fabs(z)));
                    Value* l1 = emit_log1p_acc(c, e);
                    Value* sp_neg = c.b.build_add(
                        c.b.build_select(c.b.build_sgt(z, c.f32(0.0f)), c.f32(0.0f), c.b.build_neg(z)), l1);
                    Value* sp_pos = c.b.build_add(
                        c.b.build_select(c.b.build_sgt(z, c.f32(0.0f)), z, c.f32(0.0f)), l1);
                    Value* one_my = c.b.build_sub(c.f32(1.0f), y);
                    Value* wy = c.b.build_mul(pos_w, y);
                    Value* loss_i = c.b.build_add(c.b.build_mul(wy, sp_neg), c.b.build_mul(one_my, sp_pos));
                    Value* denom = c.b.build_add(c.f32(1.0f), e);
                    Value* sig = c.b.build_select(c.b.build_sge(z, c.f32(0.0f)),
                                                  emit_div_rn(c, c.f32(1.0f), denom), emit_div_rn(c, e, denom));
                    c.kb.store_f32(p_at, sig);
                    c.kb.store_f32(d_at, c.b.build_sub(c.b.build_mul(sig, c.b.build_add(wy, one_my)), wy));
                    return c.b.build_add(acc, loss_i);
                };
                if (!has_mask) return eval();
                return emit_if_else_values(
                    c, is_valid(c, mask, at, true), {Type::f32()},
                    [&]() -> std::vector<Value*> { return {eval()}; },
                    [&]() -> std::vector<Value*> {
                        c.kb.store_f32(p_at, c.f32(0.0f));
                        c.kb.store_f32(d_at, c.f32(0.0f));
                        return {acc};
                    })[0];
            });
        } else {
            row_loss = c.kb.for_range_reduce(c.i32(0), heads, c.i32(1), c.f32(0.0f), [&](Value* h, Value* h_loss) {
                Value* s = c.i32(0);
                Value* e = n;
                if (has_off) {
                    s = clamp_i(c, c.kb.load_i32(ptr_at(c, off, h, 4)), c.i32(0), n);
                    e = clamp_i(c, c.kb.load_i32(ptr_at(c, off, c.b.build_add(h, c.i32(1)), 4)), s, n);
                }
                Value* start_i = c.b.build_add(s, tid);
                Value* m_loc = c.kb.for_range_reduce(start_i, e, block_sz, c.f32(-1e30f), [&](Value* i, Value* mx) {
                    Value* at = c.b.build_add(ro, i);
                    Value* lv = c.kb.load_f32(ptr_at(c, logits, at, 4));
                    if (!has_mask) return c.kb.fmax(mx, lv);
                    return c.b.build_select(is_valid(c, mask, at, true), c.kb.fmax(mx, lv), mx);
                });
                Value* m = emit_block_max_f32(c, scratch8, m_loc);

                Value* sum_loc = c.kb.for_range_reduce(start_i, e, block_sz, c.f32(0.0f), [&](Value* i, Value* sm) {
                    Value* at = c.b.build_add(ro, i);
                    Value* ev = emit_exp(c, c.b.build_sub(c.kb.load_f32(ptr_at(c, logits, at, 4)), m));
                    if (!has_mask) return c.b.build_add(sm, ev);
                    return c.b.build_select(is_valid(c, mask, at, true), c.b.build_add(sm, ev), sm);
                });
                Value* sum = emit_block_sum_f32(c, scratch8, sum_loc);
                Value* inv = c.b.build_select(c.b.build_sgt(sum, c.f32(0.0f)),
                                              c.b.build_sdiv(c.f32(1.0f), sum), c.f32(0.0f));

                return c.kb.for_range_reduce(start_i, e, block_sz, h_loss, [&](Value* i, Value* l_acc) {
                    Value* at = c.b.build_add(ro, i);
                    Value* p_at = ptr_at(c, probs, at, 4);
                    Value* d_at = ptr_at(c, dlog, at, 4);
                    auto write_elem = [&]() -> Value* {
                        Value* pv = c.b.build_mul(
                            emit_exp(c, c.b.build_sub(c.kb.load_f32(ptr_at(c, logits, at, 4)), m)), inv);
                        Value* tv = c.kb.load_f32(ptr_at(c, target, at, 4));
                        c.kb.store_f32(p_at, pv);
                        c.kb.store_f32(d_at, c.b.build_sub(pv, tv));
                        Value* term = c.b.build_mul(tv, emit_log(c, c.kb.fmax(pv, c.f32(1e-12f))));
                        return c.b.build_select(c.b.build_sgt(tv, c.f32(0.0f)),
                                                c.b.build_sub(l_acc, term), l_acc);
                    };
                    if (!has_mask) return write_elem();
                    return emit_if_else_values(
                        c, is_valid(c, mask, at, true), {Type::f32()},
                        [&]() -> std::vector<Value*> { return {write_elem()}; },
                        [&]() -> std::vector<Value*> {
                            c.kb.store_f32(p_at, c.f32(0.0f));
                            c.kb.store_f32(d_at, c.f32(0.0f));
                            return {l_acc};
                        })[0];
                });
            });
        }
        Value* total_loss = emit_block_sum_f32(c, scratch8, row_loss);
        c.kb.if_then(c.b.build_eq(tid, c.i32(0)), [&] {
            c.kb.store_f32(ptr_at(c, loss_buf, b, 4), total_loss);
        });
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
