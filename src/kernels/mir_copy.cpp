#include "mir_copy.h"
#include "mir_common.h"

namespace brotensor::kernels {

Function* build_fill_bytes_mir(Module& mod) {
    Function* fn = mod.create_function(
        "mir_fill_bytes", Type::void_type(), {Type::ptr(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* v = c.b.build_and(p[2], c.i32(0xff));
    emit_vec_grid_stride(c, p[1], 1, kFillBlock, [&](Value* i) {
        kb.store_u8(c.b.build_add(p[0], byte_offset(c, i, 1)), v);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_copy2d_mir(Module& mod, int esz, int lanes) {
    Function* fn = mod.create_function(
        "mir_copy2d", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* gid_x = kb.global_tid_x();
    Value* stride_x = kb.mul(kb.nctaid_x(), c.i32(static_cast<int>(kCopyBlock)));
    Value* n_vecs = vec_count(c, p[4], lanes);

    kb.for_range(kb.ctaid_y(), p[5], kb.nctaid_y(), [&](Value* r) {
        Value* so = c.b.build_mul(r, p[2]);
        Value* dof = c.b.build_mul(r, p[3]);
        kb.for_range(gid_x, n_vecs, stride_x, [&](Value* kv) {
            Value* col = col_elem_idx(c, kv, lanes);
            Value* sp = ptr_at(c, p[0], c.b.build_add(so, col), esz);
            Value* dp = ptr_at(c, p[1], c.b.build_add(dof, col), esz);
            if (lanes > 1) {
                Value* vec = kb.vload(Type::i32x4(), sp);
                kb.vstore(Type::i32x4(), dp, vec);
            } else if (esz == 4) {
                kb.store_i32(dp, kb.load_i32(sp));
            } else if (esz == 2) {
                kb.store_u16(dp, kb.load_u16(sp));
            } else {
                kb.store_u8(dp, kb.load_u8(sp));
            }
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_transpose_mir(Module& mod, int esz) {
    Function* fn = mod.create_function(
        "mir_transpose", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    Value* tile = kb.shared_alloc_i32(32u * 33u);
    Value* tid = kb.tid_x();
    Value* tx = c.b.build_and(tid, c.i32(31));
    Value* ty = c.b.build_and(c.b.build_lshr(tid, c.i32(5)), c.i32(7));
    Value* rows = p[3];
    Value* cols = p[4];
    Value* plane = c.b.build_mul(rows, cols);
    Value* tiles_c = c.b.build_lshr(c.b.build_add(cols, c.i32(31)), c.i32(5));
    Value* tiles_r = c.b.build_lshr(c.b.build_add(rows, c.i32(31)), c.i32(5));

    kb.for_range(kb.ctaid_z(), p[2], kb.nctaid_z(), [&](Value* b) {
        Value* base = c.b.build_mul(b, plane);
        kb.for_range(kb.ctaid_y(), tiles_r, kb.nctaid_y(), [&](Value* trow) {
            kb.for_range(kb.ctaid_x(), tiles_c, kb.nctaid_x(), [&](Value* tcol) {
                Value* r0 = c.b.build_shl(trow, c.i32(5));
                Value* c0 = c.b.build_shl(tcol, c.i32(5));
                Value* gc = c.b.build_add(c0, tx);
                for (int k = 0; k < 32; k += 8) {
                    Value* tyk = (k == 0) ? ty : c.b.build_add(ty, c.i32(k));
                    Value* gr = c.b.build_add(r0, tyk);
                    kb.if_then(c.b.build_and(c.b.build_ult(gr, rows), c.b.build_ult(gc, cols)), [&] {
                        Value* idx = c.b.build_add(base, c.b.build_add(c.b.build_mul(gr, cols), gc));
                        Value* sp = ptr_at(c, p[0], idx, esz);
                        Value* val = (esz == 4) ? kb.load_i32(sp) : kb.load_u16(sp);
                        Value* slot = c.b.build_add(c.b.build_mul(tyk, c.i32(33)), tx);
                        kb.shared_store_i32_indexed(tile, slot, val);
                    });
                }
                kb.sync();
                Value* wr = c.b.build_add(r0, tx);
                Value* tx33 = c.b.build_mul(tx, c.i32(33));
                for (int k = 0; k < 32; k += 8) {
                    Value* tyk = (k == 0) ? ty : c.b.build_add(ty, c.i32(k));
                    Value* wc = c.b.build_add(c0, tyk);
                    kb.if_then(c.b.build_and(c.b.build_ult(wr, rows), c.b.build_ult(wc, cols)), [&] {
                        Value* val = kb.shared_load_i32_indexed(tile, c.b.build_add(tx33, tyk));
                        Value* idx = c.b.build_add(base, c.b.build_add(c.b.build_mul(wc, rows), wr));
                        Value* dp = ptr_at(c, p[1], idx, esz);
                        if (esz == 4) kb.store_i32(dp, val);
                        else          kb.store_u16(dp, val);
                    });
                }
                kb.sync();
            });
        });
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
