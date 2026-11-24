#include "mir_remap.h"

#include "../vulkan/shaders/op_codes.h"

#include <functional>
#include <string>

namespace brotensor::kernels {

namespace {

constexpr int kRemapCpt = 16;

Value* load_raw(Ctx& c, Value* base, Value* elem_idx, int es) {
    Value* ptr = ptr_at(c, base, elem_idx, es);
    return es == 2 ? c.kb.load_u16(ptr) : c.kb.load_i32(ptr);
}

void store_raw(Ctx& c, Value* base, Value* elem_idx, Value* val, int es) {
    Value* ptr = ptr_at(c, base, elem_idx, es);
    if (es == 2) {
        c.kb.store_u16(ptr, val);
    } else {
        c.kb.store_i32(ptr, val);
    }
}

Value* cond_load_raw(Ctx& c, Value* cond, Value* ptr, const std::function<Value*()>& make_idx, int es) {
    return emit_if_else_values(
        c, cond, {Type::i32()},
        [&]() -> std::vector<Value*> { return {load_raw(c, ptr, make_idx(), es)}; },
        [&]() -> std::vector<Value*> { return {c.i32(0)}; })[0];
}

Value* emit_pad_src(Ctx& c, Value* rel, Value* len, Value* pad_mode) {
    Value* in_bounds = c.b.build_and(c.b.build_sge(rel, c.i32(0)), c.b.build_slt(rel, len));
    Value* len_m1 = c.b.build_sub(len, c.i32(1));
    Value* rep = c.b.build_select(c.b.build_slt(rel, c.i32(0)), c.i32(0), len_m1);

    Value* per_raw = c.b.build_mul(len_m1, c.i32(2));
    Value* period = c.b.build_select(c.b.build_sgt(per_raw, c.i32(0)), per_raw, c.i32(1));
    Value* q_raw = c.b.build_smod(rel, period);
    Value* q = c.b.build_select(c.b.build_slt(q_raw, c.i32(0)), c.b.build_add(q_raw, period), q_raw);
    Value* ref_raw = c.b.build_select(c.b.build_slt(q, len), q, c.b.build_sub(period, q));
    Value* refl = c.b.build_select(c.b.build_eq(len, c.i32(1)), c.i32(0), ref_raw);

    Value* oob = c.b.build_select(
        c.b.build_eq(pad_mode, c.i32(0)), c.i32(-1),
        c.b.build_select(c.b.build_eq(pad_mode, c.i32(2)), rep, refl));
    return c.b.build_select(in_bounds, rel, oob);
}

}  // namespace

brass::Function* build_remap_mir(brass::Module& mod, std::uint32_t mode, int elem_bytes) {
    auto* fn = mod.create_function(
        "mir_remap_" + std::to_string(mode) + "_b" + std::to_string(elem_bytes), Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *x = p[0], *y = p[1], *idx = p[2];
    Value *c_out = p[3], *cin = p[4], *hi = p[5], *wi = p[6], *ho = p[7], *wo = p[8];
    Value *a0 = p[9], *a1 = p[10], *a2 = p[11], *a3 = p[12], *a4 = p[13], *a5 = p[14], *a6 = p[15];

    Value* plane_o = c.b.build_mul(ho, wo);
    Value* plane_i = c.b.build_mul(hi, wi);
    Value* img = c.kb.ctaid_z();
    Value* c0 = c.b.build_mul(c.kb.ctaid_y(), c.i32(kRemapCpt));
    Value* c1 = c.b.build_select(c.b.build_ult(c.b.build_add(c0, c.i32(kRemapCpt)), c_out),
                                 c.b.build_add(c0, c.i32(kRemapCpt)), c_out);

    c.kb.for_range(c.kb.global_tid_x(), plane_o, grid_stride_ew(c), [&](Value* pos) {
        Value* oh = c.b.build_udiv(pos, wo);
        Value* ow = c.b.build_umod(pos, wo);
        Value* obase = c.b.build_add(c.b.build_mul(c.b.build_mul(img, c_out), plane_o), pos);

        if (mode == RM_PAD || mode == RM_SLICE || mode == RM_UP2 || mode == RM_WIN_PART || mode == RM_WIN_REV) {
            Value *sh = nullptr, *sw = nullptr, *ibase = nullptr;
            if (mode == RM_PAD) {
                sh = emit_pad_src(c, c.b.build_sub(oh, a0), hi, a2);
                sw = emit_pad_src(c, c.b.build_sub(ow, a1), wi, a2);
                ibase = c.b.build_mul(c.b.build_mul(img, cin), plane_i);
            } else if (mode == RM_SLICE) {
                sh = c.b.build_add(oh, a0);
                sw = c.b.build_add(ow, a1);
                ibase = c.b.build_mul(c.b.build_mul(img, cin), plane_i);
            } else if (mode == RM_UP2) {
                sh = c.b.build_lshr(oh, c.i32(1));
                sw = c.b.build_lshr(ow, c.i32(1));
                ibase = c.b.build_mul(c.b.build_mul(img, cin), plane_i);
            } else if (mode == RM_WIN_PART) {
                Value* per = c.b.build_mul(a1, a2);
                Value* n = c.b.build_udiv(img, per);
                Value* r = c.b.build_umod(img, per);
                sh = c.b.build_add(c.b.build_mul(c.b.build_udiv(r, a2), a0), oh);
                sw = c.b.build_add(c.b.build_mul(c.b.build_umod(r, a2), a0), ow);
                ibase = c.b.build_mul(c.b.build_mul(n, cin), plane_i);
            } else {
                Value* win = a0;
                Value* b_win = c.b.build_add(
                    c.b.build_mul(c.b.build_add(c.b.build_mul(img, a1), c.b.build_udiv(oh, win)), a2),
                    c.b.build_udiv(ow, win));
                sh = c.b.build_umod(oh, win);
                sw = c.b.build_umod(ow, win);
                ibase = c.b.build_mul(c.b.build_mul(b_win, cin), plane_i);
            }
            Value* ok = c.b.build_and(c.b.build_sge(sh, c.i32(0)), c.b.build_sge(sw, c.i32(0)));
            Value* ip = c.b.build_add(c.b.build_mul(sh, wi), sw);
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* v = cond_load_raw(c, ok, x, [&] {
                    return c.b.build_add(c.b.build_add(ibase, c.b.build_mul(ch, plane_i)), ip);
                }, elem_bytes);
                store_raw(c, y, c.b.build_add(obase, c.b.build_mul(ch, plane_o)), v, elem_bytes);
            });
        } else if (mode == RM_UNFOLD) {
            Value* kK = c.b.build_mul(a0, a1);
            Value* yb = c.b.build_sub(c.b.build_mul(oh, a2), a4);
            Value* xb = c.b.build_sub(c.b.build_mul(ow, a3), a5);
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* ci = c.b.build_udiv(ch, kK);
                Value* k = c.b.build_umod(ch, kK);
                Value* sy = emit_pad_src(c, c.b.build_add(yb, c.b.build_udiv(k, a1)), hi, a6);
                Value* sx = emit_pad_src(c, c.b.build_add(xb, c.b.build_umod(k, a1)), wi, a6);
                Value* ok = c.b.build_and(c.b.build_sge(sy, c.i32(0)), c.b.build_sge(sx, c.i32(0)));
                Value* v = cond_load_raw(c, ok, x, [&] {
                    Value* row_b = c.b.build_mul(c.b.build_add(c.b.build_mul(img, cin), ci), plane_i);
                    return c.b.build_add(row_b, c.b.build_add(c.b.build_mul(sy, wi), sx));
                }, elem_bytes);
                store_raw(c, y, c.b.build_add(obase, c.b.build_mul(ch, plane_o)), v, elem_bytes);
            });
        } else if (mode == RM_MERGE2) {
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* ch_maj = c.b.build_ne(a0, c.i32(0));
                Value* blk = c.b.build_select(ch_maj, c.b.build_and(ch, c.i32(3)), c.b.build_udiv(ch, cin));
                Value* ci = c.b.build_select(ch_maj, c.b.build_lshr(ch, c.i32(2)), c.b.build_umod(ch, cin));
                Value* ih = c.b.build_add(c.b.build_mul(oh, c.i32(2)), c.b.build_lshr(blk, c.i32(1)));
                Value* iw = c.b.build_add(c.b.build_mul(ow, c.i32(2)), c.b.build_and(blk, c.i32(1)));
                Value* src_i = c.b.build_add(
                    c.b.build_mul(c.b.build_add(c.b.build_mul(img, cin), ci), plane_i),
                    c.b.build_add(c.b.build_mul(ih, wi), iw));
                store_raw(c, y, c.b.build_add(obase, c.b.build_mul(ch, plane_o)), load_raw(c, x, src_i, elem_bytes),
                          elem_bytes);
            });
        } else if (mode == RM_PSHUF) {
            Value* i = c.b.build_and(oh, c.i32(1));
            Value* j = c.b.build_and(ow, c.i32(1));
            Value* ij = c.b.build_add(c.b.build_mul(i, c.i32(2)), j);
            Value* ip = c.b.build_add(c.b.build_mul(c.b.build_lshr(oh, c.i32(1)), wi), c.b.build_lshr(ow, c.i32(1)));
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* sc = c.b.build_udiv(c.b.build_add(c.b.build_mul(ch, c.i32(4)), ij), a0);
                Value* src_i = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(img, cin), sc), plane_i), ip);
                store_raw(c, y, c.b.build_add(obase, c.b.build_mul(ch, plane_o)), load_raw(c, x, src_i, elem_bytes),
                          elem_bytes);
            });
        } else if (mode == RM_GATHER_ROWS) {
            Value* r = clamp_i32(c, c.kb.load_i32(ptr_at(c, idx, img, 4)), c.i32(0), c.b.build_sub(a0, c.i32(1)));
            Value* v = load_raw(c, x, c.b.build_add(c.b.build_mul(r, wo), pos), elem_bytes);
            store_raw(c, y, c.b.build_add(c.b.build_mul(img, wo), pos), v, elem_bytes);
        } else if (mode == RM_SCATTER_ROWS) {
            Value* r = c.kb.load_i32(ptr_at(c, idx, img, 4));
            c.kb.if_then(c.b.build_and(c.b.build_sge(r, c.i32(0)), c.b.build_slt(r, a0)), [&] {
                Value* v = load_raw(c, x, c.b.build_add(c.b.build_mul(img, wo), pos), elem_bytes);
                store_raw(c, y, c.b.build_add(c.b.build_mul(r, wo), pos), v, elem_bytes);
            });
        } else {
            Value* P = a0;
            Value* PP = c.b.build_mul(P, P);
            Value* tok = c.b.build_add(c.b.build_mul(c.b.build_udiv(oh, P), a3), c.b.build_udiv(ow, P));
            Value* blk = c.b.build_add(c.b.build_mul(c.b.build_umod(oh, P), P), c.b.build_umod(ow, P));
            Value* row = c.b.build_mul(c.b.build_mul(tok, PP), a1);
            Value* ch_maj = c.b.build_ne(a2, c.i32(0));
            c.kb.for_range(c0, c1, c.i32(1), [&](Value* ch) {
                Value* col = c.b.build_select(
                    ch_maj, c.b.build_add(c.b.build_mul(ch, PP), blk),
                    c.b.build_add(c.b.build_mul(blk, a1), ch));
                store_raw(c, y, c.b.build_add(obase, c.b.build_mul(ch, plane_o)),
                          load_raw(c, x, c.b.build_add(row, col), elem_bytes), elem_bytes);
            });
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
