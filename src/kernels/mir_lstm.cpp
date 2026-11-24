#include "mir_lstm.h"

#include "mir_math.h"
#include "../vulkan/shaders/op_codes.h"

#include <string>

namespace brotensor::kernels {

brass::Function* build_lstm_mir(brass::Module& mod, std::uint32_t mode, bool has_bhh, bool has_c0) {
    const std::string name = "mir_lstm_" + std::to_string(mode) + "_b" + std::to_string(has_bhh ? 1 : 0) +
                             "_c" + std::to_string(has_c0 ? 1 : 0);
    auto* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value *gates = p[0], *cell = p[1], *y = p[2], *bhh = p[3], *c0 = p[4];
    Value *dy = p[5], *dz = p[6], *dh_buf = p[7], *dc_buf = p[8];
    Value *B = p[9], *H = p[10], *row0 = p[11];

    Value* G = c.b.build_mul(H, c.i32(4));
    Value* total = c.b.build_mul(B, H);
    Value* row0_gt0 = c.b.build_ugt(row0, c.i32(0));

    c.kb.for_range(c.kb.global_tid_x(), total, grid_stride_ew(c), [&](Value* e) {
        Value* b = c.b.build_udiv(e, H);
        Value* n = c.b.build_umod(e, H);
        Value* row = c.b.build_add(row0, b);
        Value* gr = c.b.build_mul(row, G);
        Value* rn_idx = c.b.build_add(c.b.build_mul(row, H), n);
        Value* bn_idx = c.b.build_add(c.b.build_mul(b, H), n);

        Value* cprev = emit_if_else_values(
            c, row0_gt0, {Type::f32()},
            [&]() -> std::vector<Value*> {
                Value* prev_idx = c.b.build_add(c.b.build_mul(c.b.build_sub(row, B), H), n);
                return {c.kb.load_f32(ptr_at(c, cell, prev_idx, 4))};
            },
            [&]() -> std::vector<Value*> {
                return {has_c0 ? c.kb.load_f32(ptr_at(c, c0, bn_idx, 4)) : c.f32(0.0f)};
            })[0];

        Value* h2 = c.b.build_mul(H, c.i32(2));
        Value* h3 = c.b.build_mul(H, c.i32(3));
        Value* gi0 = c.b.build_add(gr, n);
        Value* gi1 = c.b.build_add(gi0, H);
        Value* gi2 = c.b.build_add(gi0, h2);
        Value* gi3 = c.b.build_add(gi0, h3);

        if (mode == LS_FWD) {
            Value* zi = c.kb.load_f32(ptr_at(c, gates, gi0, 4));
            Value* zf = c.kb.load_f32(ptr_at(c, gates, gi1, 4));
            Value* zg = c.kb.load_f32(ptr_at(c, gates, gi2, 4));
            Value* zo = c.kb.load_f32(ptr_at(c, gates, gi3, 4));
            if (has_bhh) {
                zi = c.b.build_add(zi, c.kb.load_f32(ptr_at(c, bhh, n, 4)));
                zf = c.b.build_add(zf, c.kb.load_f32(ptr_at(c, bhh, c.b.build_add(H, n), 4)));
                zg = c.b.build_add(zg, c.kb.load_f32(ptr_at(c, bhh, c.b.build_add(h2, n), 4)));
                zo = c.b.build_add(zo, c.kb.load_f32(ptr_at(c, bhh, c.b.build_add(h3, n), 4)));
            }
            Value* ig = emit_sigmoid(c, zi);
            Value* fg = emit_sigmoid(c, zf);
            Value* gg = emit_tanh(c, zg);
            Value* og = emit_sigmoid(c, zo);
            Value* cn = c.b.build_add(c.b.build_mul(fg, cprev), c.b.build_mul(ig, gg));
            c.kb.store_f32(ptr_at(c, gates, gi0, 4), ig);
            c.kb.store_f32(ptr_at(c, gates, gi1, 4), fg);
            c.kb.store_f32(ptr_at(c, gates, gi2, 4), gg);
            c.kb.store_f32(ptr_at(c, gates, gi3, 4), og);
            c.kb.store_f32(ptr_at(c, cell, rn_idx, 4), cn);
            c.kb.store_f32(ptr_at(c, y, rn_idx, 4), c.b.build_mul(og, emit_tanh(c, cn)));
        } else {
            Value* ig = c.kb.load_f32(ptr_at(c, gates, gi0, 4));
            Value* fg = c.kb.load_f32(ptr_at(c, gates, gi1, 4));
            Value* gg = c.kb.load_f32(ptr_at(c, gates, gi2, 4));
            Value* og = c.kb.load_f32(ptr_at(c, gates, gi3, 4));
            Value* tc = emit_tanh(c, c.kb.load_f32(ptr_at(c, cell, rn_idx, 4)));
            Value* dh = c.b.build_add(c.kb.load_f32(ptr_at(c, dy, rn_idx, 4)),
                                      c.kb.load_f32(ptr_at(c, dh_buf, bn_idx, 4)));
            Value* one_m_tc2 = c.b.build_sub(c.f32(1.0f), c.b.build_mul(tc, tc));
            Value* dc = c.b.build_add(c.b.build_mul(c.b.build_mul(dh, og), one_m_tc2),
                                      c.kb.load_f32(ptr_at(c, dc_buf, bn_idx, 4)));
            c.kb.store_f32(ptr_at(c, dz, gi0, 4),
                           c.b.build_mul(c.b.build_mul(dc, gg), c.b.build_mul(ig, c.b.build_sub(c.f32(1.0f), ig))));
            c.kb.store_f32(ptr_at(c, dz, gi1, 4),
                           c.b.build_mul(c.b.build_mul(dc, cprev), c.b.build_mul(fg, c.b.build_sub(c.f32(1.0f), fg))));
            c.kb.store_f32(ptr_at(c, dz, gi2, 4),
                           c.b.build_mul(c.b.build_mul(dc, ig), c.b.build_sub(c.f32(1.0f), c.b.build_mul(gg, gg))));
            c.kb.store_f32(ptr_at(c, dz, gi3, 4),
                           c.b.build_mul(c.b.build_mul(dh, tc), c.b.build_mul(og, c.b.build_sub(c.f32(1.0f), og))));
            c.kb.store_f32(ptr_at(c, dc_buf, bn_idx, 4), c.b.build_mul(dc, fg));
        }
    });

    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
