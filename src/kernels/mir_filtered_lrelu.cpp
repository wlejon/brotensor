#include "mir_filtered_lrelu.h"
#include "mir_common.h"

#include <cstdint>
#include <vector>

namespace brotensor::kernels {

namespace {

constexpr int kFlTh = 16;
constexpr int kFlTw = 16;
constexpr int kFlThreads = kFlTh * kFlTw;

Value* alloc_fl_stage(Ctx& c, Dtype dt, std::uint32_t count) {
    return (dt == Dtype::FP32) ? c.kb.shared_alloc_f32(count) : c.kb.shared_alloc_u16(count);
}

Value* load_fl_stage(Ctx& c, Dtype dt, Value* buf, Value* idx) {
    if (dt == Dtype::FP32) return c.kb.shared_load_f32_indexed(buf, idx);
    Value* raw = c.kb.shared_load_u16_indexed(buf, idx);
    return (dt == Dtype::FP16) ? c.kb.f16_to_f32(raw) : c.kb.bf16_to_f32(raw);
}

void store_fl_stage(Ctx& c, Dtype dt, Value* buf, Value* idx, Value* val) {
    if (dt == Dtype::FP32) {
        c.kb.shared_store_f32_indexed(buf, idx, val);
        return;
    }
    Value* raw = (dt == Dtype::FP16) ? c.kb.f32_to_f16(val) : c.kb.f32_to_bf16(val);
    c.kb.shared_store_u16_indexed(buf, idx, raw);
}

Function* build_polyu_fn(Module& mod, Dtype dt) {
    Function* fn = mod.create_function(
        "mir_fl_polyu", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: fu(0) poly(1) fuH(2) fuW(3) up(4) JyMax(5) JxMax(6)
    Value* total = c.b.build_mul(c.b.build_mul(p[4], p[4]), c.b.build_mul(p[5], p[6]));
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_ult(idx, total), [&] {
        Value* i = c.b.build_umod(idx, p[6]);
        Value* t0 = c.b.build_udiv(idx, p[6]);
        Value* j = c.b.build_umod(t0, p[5]);
        Value* t1 = c.b.build_udiv(t0, p[5]);
        Value* rx = c.b.build_umod(t1, p[4]);
        Value* ry = c.b.build_udiv(t1, p[4]);
        Value* fuy = c.b.build_add(ry, c.b.build_mul(j, p[4]));
        Value* fux = c.b.build_add(rx, c.b.build_mul(i, p[4]));
        Value* ok = c.b.build_and(c.b.build_slt(fuy, p[2]), c.b.build_slt(fux, p[3]));
        Value* val = emit_if_else_values(
            c, ok, {Type::f32()},
            [&]() -> std::vector<Value*> {
                Value* r = c.b.build_sub(c.b.build_sub(p[2], c.i32(1)), fuy);
                Value* col = c.b.build_sub(c.b.build_sub(p[3], c.i32(1)), fux);
                Value* off = c.b.build_add(c.b.build_mul(r, p[3]), col);
                return {load_elem_dt(c, dt, p[0], off)};
            },
            [&]() -> std::vector<Value*> { return {c.f32(0.0f)}; })[0];
        store_elem_dt(c, dt, p[1], idx, val);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_polyd_fn(Module& mod, Dtype dt) {
    Function* fn = mod.create_function(
        "mir_fl_polyd", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: fd(0) poly(1) fdH(2) fdW(3)
    Value* total = c.b.build_mul(p[2], p[3]);
    Value* idx = kb.global_tid_x();
    kb.if_then(c.b.build_ult(idx, total), [&] {
        Value* fdx = c.b.build_umod(idx, p[3]);
        Value* fdy = c.b.build_udiv(idx, p[3]);
        Value* r = c.b.build_sub(c.b.build_sub(p[2], c.i32(1)), fdy);
        Value* col = c.b.build_sub(c.b.build_sub(p[3], c.i32(1)), fdx);
        Value* val = load_elem_dt(c, dt, p[0], c.b.build_add(c.b.build_mul(r, p[3]), col));
        store_elem_dt(c, dt, p[1], idx, val);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_main_fn(Module& mod, Dtype dt, bool has_bias, bool has_clamp, std::uint64_t aux) {
    const int up = static_cast<int>(aux & 0xffu);
    const int down = static_cast<int>((aux >> 8) & 0xffu);
    const int JyMax = static_cast<int>((aux >> 16) & 0xffu);
    const int JxMax = static_cast<int>((aux >> 24) & 0xffu);
    const int fdH = static_cast<int>((aux >> 32) & 0xffu);
    const int fdW = static_cast<int>((aux >> 40) & 0xffu);
    const int AH = (kFlTh - 1) * down + fdH;
    const int AW = (kFlTw - 1) * down + fdW;
    const int acells = AH * AW;
    const int bank = JyMax * JxMax;
    const int esz = elem_bytes(dt);

    Function* fn = mod.create_function(
        "mir_filtered_lrelu", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
         Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(), Type::i32(),
         Type::f32(), Type::f32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // ptrs: X(0) poly_u(1) poly_d(2) b(3) Y(4)
    // ints: C(5) H(6) W(7) Hout(8) Wout(9) px0(10) py0(11)
    // flts: up_gain(12) gain(13) slope(14) clamp_v(15)

    Value* act = alloc_fl_stage(c, dt, static_cast<std::uint32_t>(acells));

    Value* nc = kb.ctaid_z();
    Value* oh0 = c.b.build_mul(kb.ctaid_y(), c.i32(kFlTh));
    Value* ow0 = c.b.build_mul(kb.ctaid_x(), c.i32(kFlTw));
    Value* ay0 = c.b.build_mul(oh0, c.i32(down));
    Value* ax0 = c.b.build_mul(ow0, c.i32(down));
    Value* tid = kb.tid_x();

    Value* H = p[6];
    Value* W = p[7];
    Value* Hout = p[8];
    Value* Wout = p[9];
    Value* px0 = p[10];
    Value* py0 = p[11];

    Value* nc64 = c.b.build_zext_i64(nc);
    Value* hw64 = c.b.build_mul(c.b.build_zext_i64(H), c.b.build_zext_i64(W));
    Value* x_base = c.b.build_add(p[0], c.b.build_mul(nc64, c.b.build_mul(hw64, kb.const_i64(esz))));
    Value* hwo64 = c.b.build_mul(c.b.build_zext_i64(Hout), c.b.build_zext_i64(Wout));
    Value* y_base = c.b.build_add(p[4], c.b.build_mul(nc64, c.b.build_mul(hwo64, kb.const_i64(esz))));

    Value* bias_v = c.f32(0.0f);
    if (has_bias) {
        Value* ch = c.b.build_umod(nc, p[5]);
        bias_v = load_elem_dt(c, dt, p[3], ch);
    }

    const bool up_pow2 = (up & (up - 1)) == 0;
    const int up_shift = up_pow2 ? log2_exact(up) : 0;

    // Phase A: compute shared activation tile.
    kb.for_range(tid, c.i32(acells), c.i32(kFlThreads), [&](Value* a) {
        Value* ly = c.b.build_udiv(a, c.i32(AW));
        Value* lx = c.b.build_umod(a, c.i32(AW));
        Value* ay = c.b.build_add(ay0, ly);
        Value* ax = c.b.build_add(ax0, lx);

        Value* ry = nullptr;
        Value* rx = nullptr;
        Value* iyb = nullptr;
        Value* ixb = nullptr;
        if (up == 1) {
            ry = c.i32(0);
            rx = c.i32(0);
            iyb = c.b.build_sub(ay, py0);
            ixb = c.b.build_sub(ax, px0);
        } else if (up_pow2) {
            ry = c.b.build_and(c.b.build_sub(py0, ay), c.i32(up - 1));
            rx = c.b.build_and(c.b.build_sub(px0, ax), c.i32(up - 1));
            iyb = c.b.build_ashr(c.b.build_sub(c.b.build_add(ay, ry), py0), c.i32(up_shift));
            ixb = c.b.build_ashr(c.b.build_sub(c.b.build_add(ax, rx), px0), c.i32(up_shift));
        } else {
            Value* dy = c.b.build_smod(c.b.build_sub(py0, ay), c.i32(up));
            Value* dx = c.b.build_smod(c.b.build_sub(px0, ax), c.i32(up));
            ry = c.b.build_select(c.b.build_slt(dy, c.i32(0)), c.b.build_add(dy, c.i32(up)), dy);
            rx = c.b.build_select(c.b.build_slt(dx, c.i32(0)), c.b.build_add(dx, c.i32(up)), dx);
            iyb = c.b.build_sdiv(c.b.build_sub(c.b.build_add(ay, ry), py0), c.i32(up));
            ixb = c.b.build_sdiv(c.b.build_sub(c.b.build_add(ax, rx), px0), c.i32(up));
        }

        Value* jlo = smax_i32(c, c.i32(0), c.b.build_neg(iyb));
        Value* jhi = smin_i32(c, c.i32(JyMax), c.b.build_sub(H, iyb));
        Value* ilo = smax_i32(c, c.i32(0), c.b.build_neg(ixb));
        Value* ihi = smin_i32(c, c.i32(JxMax), c.b.build_sub(W, ixb));

        Value* bank_off = c.b.build_mul(c.b.build_add(c.b.build_mul(ry, c.i32(up)), rx), c.i32(bank));
        Value* u = kb.for_range_reduce(jlo, jhi, c.i32(1), c.f32(0.0f), [&](Value* j, Value* uj) {
            Value* x_row_base = c.b.build_add(c.b.build_mul(c.b.build_add(iyb, j), W), ixb);
            Value* p_row_base = c.b.build_add(bank_off, c.b.build_mul(j, c.i32(JxMax)));
            return kb.for_range_reduce(ilo, ihi, c.i32(1), uj, [&](Value* i, Value* ui) {
                Value* xv = load_elem_dt(c, dt, x_base, c.b.build_add(x_row_base, i));
                Value* pv = load_elem_dt(c, dt, p[1], c.b.build_add(p_row_base, i));
                Value* xvb = has_bias ? c.b.build_add(xv, bias_v) : xv;
                return c.b.build_fma_f32(pv, xvb, ui);
            });
        });

        Value* ug = c.b.build_mul(u, p[12]);
        Value* act0 = c.b.build_select(c.b.build_sgt(ug, c.f32(0.0f)), ug, c.b.build_mul(ug, p[14]));
        Value* av = c.b.build_mul(act0, p[13]);
        if (has_clamp) {
            av = kb.clamp_f32(av, c.b.build_neg(p[15]), p[15]);
        }
        store_fl_stage(c, dt, act, a, av);
    });
    kb.sync();

    // Phase B: one thread per output pixel in the 16x16 tile; downsample from shared.
    Value* ly_out = c.b.build_lshr(tid, c.i32(4));
    Value* lx_out = c.b.build_and(tid, c.i32(15));
    Value* oh = c.b.build_add(oh0, ly_out);
    Value* ow = c.b.build_add(ow0, lx_out);
    kb.if_then(c.b.build_and(c.b.build_slt(oh, Hout), c.b.build_slt(ow, Wout)), [&] {
        Value* ly0 = c.b.build_mul(ly_out, c.i32(down));
        Value* lx0 = c.b.build_mul(lx_out, c.i32(down));
        Value* act_base = c.b.build_add(c.b.build_mul(ly0, c.i32(AW)), lx0);

        Value* acc = c.f32(0.0f);
        if (fdH * fdW <= 64) {
            for (int fdy = 0; fdy < fdH; ++fdy) {
                for (int fdx = 0; fdx < fdW; ++fdx) {
                    Value* a_val = load_fl_stage(c, dt, act, c.b.build_add(act_base, c.i32(fdy * AW + fdx)));
                    Value* p_val = load_elem_dt(c, dt, p[2], c.i32(fdy * fdW + fdx));
                    acc = c.b.build_fma_f32(p_val, a_val, acc);
                }
            }
        } else {
            acc = kb.for_range_reduce(c.i32(0), c.i32(fdH), c.i32(1), c.f32(0.0f), [&](Value* fdy, Value* ay_acc) {
                Value* ar = c.b.build_add(act_base, c.b.build_mul(fdy, c.i32(AW)));
                Value* pr = c.b.build_mul(fdy, c.i32(fdW));
                return kb.for_range_reduce(c.i32(0), c.i32(fdW), c.i32(1), ay_acc, [&](Value* fdx, Value* ax_acc) {
                    Value* a_val = load_fl_stage(c, dt, act, c.b.build_add(ar, fdx));
                    Value* p_val = load_elem_dt(c, dt, p[2], c.b.build_add(pr, fdx));
                    return c.b.build_fma_f32(p_val, a_val, ax_acc);
                });
            });
        }
        Value* out_idx = c.b.build_add(c.b.build_mul(oh, Wout), ow);
        store_elem_dt(c, dt, y_base, out_idx, acc);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace

Function* build_filtered_lrelu_mir(Module& mod, std::uint32_t mode, Dtype dt,
                                   bool has_bias, bool has_clamp, std::uint64_t aux) {
    if (mode == 0u) return build_polyu_fn(mod, dt);
    if (mode == 1u) return build_polyd_fn(mod, dt);
    return build_main_fn(mod, dt, has_bias, has_clamp, aux);
}

}  // namespace brotensor::kernels
