#include "mir_gemm_splitk.h"
#include "mir_common.h"
#include "mir_math.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace brotensor::kernels {

namespace {

Value* load_elem_f32(Ctx& c, Dtype dt, Value* ptr, Value* idx) {
    Value* ep = ptr_at(c, ptr, idx, 2);
    Value* raw = c.kb.load_u16(ep);
    return (dt == Dtype::FP16) ? c.kb.f16_to_f32(raw) : c.kb.bf16_to_f32(raw);
}

void store_elem_f32(Ctx& c, Dtype dt, Value* ptr, Value* idx, Value* v) {
    Value* ep = ptr_at(c, ptr, idx, 2);
    store_lanes(c, dt, 1, ep, {v});
}

}  // namespace

Function* build_gemm_splitk_reduce_mir(Module& mod, const GemmSplitkSpec& spec) {
    if (spec.dt != Dtype::FP16 && spec.dt != Dtype::BF16) {
        throw std::runtime_error("build_gemm_splitk_reduce_mir: only FP16/BF16 supported");
    }
    Function* fn = mod.create_function(
        "mir_gemm_splitk_reduce", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});

    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    // p: partial(0), C(1), bias(2), M(3), N(4), splits(5)
    Value* M = p[3];
    Value* N = p[4];
    Value* splits = p[5];

    const bool gated = (spec.epi >= 2u);
    Value* out_cols = gated ? c.b.build_lshr(N, c.i32(1)) : N;
    Value* total = c.b.build_mul(M, out_cols);
    Value* idx = kb.global_tid_x();

    kb.if_then(c.b.build_ult(idx, total), [&] {
        Value* m = c.b.build_udiv(idx, out_cols);
        Value* col = c.b.build_umod(idx, out_cols);
        Value* stride = c.b.build_mul(M, N);

        auto sum_at = [&](Value* n_col) -> Value* {
            Value* base = c.b.build_add(c.b.build_mul(m, N), n_col);
            Value* p_col = ptr_at(c, p[0], base, 4);
            Value* s_bytes = c.b.build_shl(c.b.build_zext_i64(stride), kb.const_i64(2));
            std::vector<Value*> red = kb.for_range_reduce_n(
                c.i32(0), splits, c.i32(1), {c.f32(0.0f), p_col},
                [&](Value*, const std::vector<Value*>& cur) -> std::vector<Value*> {
                    Value* v = kb.load_f32(cur[1], 0);
                    return {c.b.build_add(cur[0], v), c.b.build_add(cur[1], s_bytes)};
                });
            Value* val = red[0];
            if (spec.has_bias) {
                val = c.b.build_add(val, load_elem_f32(c, spec.dt, p[2], n_col));
            }
            return val;
        };

        Value* out_v = nullptr;
        if (!gated) {
            Value* v = sum_at(col);
            if (spec.act != 0u) v = emit_gemm_act(c, spec.act, v);
            if (spec.epi == 1u) {
                v = c.b.build_add(v, load_elem_f32(c, spec.dt, p[1], idx));
            }
            out_v = v;
        } else if (spec.epi == 2u) {
            Value* n0 = c.b.build_shl(col, c.i32(1));
            Value* n1 = c.b.build_add(n0, c.i32(1));
            Value* a = sum_at(n0);
            Value* b = sum_at(n1);
            out_v = c.b.build_mul(a, emit_gelu_exact(c, b));
        } else {
            Value* n0 = col;
            Value* n1 = c.b.build_add(out_cols, col);
            Value* a = sum_at(n0);
            Value* b = sum_at(n1);
            out_v = c.b.build_mul(emit_gemm_act(c, 4, a), b);  // SiLU(a) * b
        }
        store_elem_f32(c, spec.dt, p[1], idx, out_v);
    });

    c.b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

}  // namespace brotensor::kernels
