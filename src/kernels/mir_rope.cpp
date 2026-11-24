#include "mir_rope.h"
#include "mir_common.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <utility>

namespace brotensor::kernels {

namespace {

void emit_rotate_pair(Ctx& c, Dtype dt, bool bwd, Value* src, Value* dst, Value* o,
                      Value* cos_v, Value* sin_v) {
    const int esz = elem_bytes(dt);
    Value* s = bwd ? c.b.build_neg(sin_v) : sin_v;
    Value* p0_in = ptr_at(c, src, o, esz);
    Value* p1_in = ptr_at(c, src, c.b.build_add(o, c.i32(1)), esz);
    Value* x0 = load_lanes(c, dt, 1, p0_in)[0];
    Value* x1 = load_lanes(c, dt, 1, p1_in)[0];
    Value* y0 = c.b.build_sub(c.b.build_mul(x0, cos_v), c.b.build_mul(x1, s));
    Value* y1 = c.b.build_add(c.b.build_mul(x0, s), c.b.build_mul(x1, cos_v));
    Value* p0_out = ptr_at(c, dst, o, esz);
    Value* p1_out = ptr_at(c, dst, c.b.build_add(o, c.i32(1)), esz);
    store_lanes(c, dt, 1, p0_out, {y0});
    store_lanes(c, dt, 1, p1_out, {y1});
}

std::pair<Value*, Value*> load_cs(Ctx& c, Value* ct, Value* st, Value* idx) {
    Value* off = byte_offset(c, idx, 4);
    Value* cv = c.kb.load_f32(c.b.build_add(ct, off));
    Value* sv = c.kb.load_f32(c.b.build_add(st, off));
    return {cv, sv};
}

}  // namespace

Function* build_rope_mir(Module& mod, std::uint32_t mode, bool bwd, Dtype dt) {
    Function* fn = mod.create_function(
        "mir_rope", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::i32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);

    Value* rows = p[11];
    Value* heads = p[12];
    Value* hd = p[13];
    Value* hp = c.b.build_lshr(hd, c.i32(1));
    Value* total = c.b.build_mul(c.b.build_mul(rows, heads), hp);
    Value* stride_d = c.b.build_mul(heads, hd);

    emit_vec_grid_stride(c, total, 1, kRopeBlock, [&](Value* idx) {
        Value* rest = c.b.build_udiv(idx, hp);
        Value* i = c.b.build_sub(idx, c.b.build_mul(rest, hp));
        Value* row = c.b.build_udiv(rest, heads);
        Value* h = c.b.build_sub(rest, c.b.build_mul(row, heads));
        Value* pair_off = c.b.build_add(c.b.build_mul(h, hd), c.b.build_shl(i, c.i32(1)));

        if (mode == ROPE_PACKED) {
            Value* pos = kb.load_i32(ptr_at(c, p[8], row, 4));
            kb.if_then(c.b.build_ult(pos, p[15]), [&] {
                Value* tbl_idx = c.b.build_add(c.b.build_mul(pos, hp), i);
                auto [cv, sv] = load_cs(c, p[2], p[3], tbl_idx);
                Value* o = c.b.build_add(c.b.build_mul(row, c.b.build_mul(c.i32(3), stride_d)), pair_off);
                emit_rotate_pair(c, dt, bwd, p[0], p[0], o, cv, sv);
                emit_rotate_pair(c, dt, bwd, p[0], p[0], c.b.build_add(o, stride_d), cv, sv);
            });
            return;
        }

        Value* cv = nullptr;
        Value* sv = nullptr;
        if (mode == ROPE_THETA) {
            Value* hd_f = kb.fmax(kb.i32_to_f32(hd), c.f32(2.0f));
            Value* neg_2i = c.b.build_neg(kb.i32_to_f32(c.b.build_shl(i, c.i32(1))));
            Value* a = emit_div_rn(c, neg_2i, hd_f);
            Value* pos_f = kb.i32_to_f32(c.b.build_add(row, p[17]));
            Value* theta = c.b.build_mul(pos_f, emit_exp_acc(c, c.b.build_mul(a, p[18])));
            auto [s_v, c_v] = emit_sincos_acc(c, theta);
            cv = c_v;
            sv = s_v;
        } else if (mode == ROPE_TABLE) {
            auto [c_v, s_v] = load_cs(c, p[2], p[3], c.b.build_add(c.b.build_mul(row, hp), i));
            cv = c_v;
            sv = s_v;
        } else if (mode == ROPE_PERHEAD) {
            Value* tidx = c.b.build_add(c.b.build_mul(c.b.build_add(c.b.build_mul(row, heads), h), hp), i);
            auto [c_v, s_v] = load_cs(c, p[2], p[3], tidx);
            cv = c_v;
            sv = s_v;
        } else {
            Value* dt_n = p[14];
            Value* dh_n = p[15];
            Value* dw_n = p[16];
            auto cs = emit_if_else_values(
                c, c.b.build_ult(i, dt_n), {Type::f32(), Type::f32()},
                [&]() -> std::vector<Value*> {
                    Value* pos = kb.load_i32(ptr_at(c, p[8], row, 4));
                    auto [c_v, s_v] = load_cs(c, p[2], p[3], c.b.build_add(c.b.build_mul(pos, dt_n), i));
                    return {c_v, s_v};
                },
                [&]() -> std::vector<Value*> {
                    Value* dth = c.b.build_add(dt_n, dh_n);
                    return emit_if_else_values(
                        c, c.b.build_ult(i, dth), {Type::f32(), Type::f32()},
                        [&]() -> std::vector<Value*> {
                            Value* pos = kb.load_i32(ptr_at(c, p[9], row, 4));
                            Value* l = c.b.build_sub(i, dt_n);
                            auto [c_v, s_v] = load_cs(c, p[4], p[5], c.b.build_add(c.b.build_mul(pos, dh_n), l));
                            return {c_v, s_v};
                        },
                        [&]() -> std::vector<Value*> {
                            Value* pos = kb.load_i32(ptr_at(c, p[10], row, 4));
                            Value* l = c.b.build_sub(i, dth);
                            auto [c_v, s_v] = load_cs(c, p[6], p[7], c.b.build_add(c.b.build_mul(pos, dw_n), l));
                            return {c_v, s_v};
                        });
                });
            cv = cs[0];
            sv = cs[1];
        }
        Value* o = c.b.build_add(c.b.build_mul(row, stride_d), pair_off);
        emit_rotate_pair(c, dt, bwd, p[0], p[1], o, cv, sv);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
