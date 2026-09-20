#include "cpu_jit.h"
#include <brotensor/detail/cpu/thread_pool.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace brotensor::detail::cpu::jit {
namespace ref {

inline void rms_norm(const float* X, const float* gamma, float eps, float* Y, int B, int D) {
    if (B <= 0 || D <= 0) return;
    const float inv_D = 1.0f / static_cast<float>(D);
    auto row_fn = [&](int b) {
        const float* xr = X + static_cast<size_t>(b) * D;
        float* yr       = Y + static_cast<size_t>(b) * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += xr[i] * xr[i];
        float rrms = 1.0f / std::sqrt(sum * inv_D + eps);
        for (int i = 0; i < D; ++i) yr[i] = xr[i] * gamma[i] * rrms;
    };
    if (B > 1 && static_cast<int64_t>(B) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<size_t>(B), [&](size_t bi) {
            row_fn(static_cast<int>(bi));
        });
    } else {
        for (int b = 0; b < B; ++b) row_fn(b);
    }
}

inline void swiglu(const float* X, float* Y, int B, int D) {
    if (B <= 0 || D <= 0) return;
    auto row_fn = [&](int b) {
        const float* gate = X + static_cast<size_t>(b) * 2 * D;
        const float* up   = gate + D;
        float* out        = Y + static_cast<size_t>(b) * D;
        for (int i = 0; i < D; ++i) {
            float g = gate[i];
            float silu = g / (1.0f + std::exp(-g));
            out[i] = silu * up[i];
        }
    };
    if (B > 1 && static_cast<int64_t>(B) * D >= 16384) {
        detail::cpu::parallel_for(static_cast<size_t>(B), [&](size_t bi) {
            row_fn(static_cast<int>(bi));
        });
    } else {
        for (int b = 0; b < B; ++b) row_fn(b);
    }
}

inline void modulate(const float* X, const float* scale, const float* shift, float* Y, int L, int D) {
    if (L <= 0 || D <= 0) return;
    auto row_fn = [&](int l) {
        const float* xr = X + static_cast<size_t>(l) * D;
        float* yr       = Y + static_cast<size_t>(l) * D;
        for (int d = 0; d < D; ++d) {
            yr[d] = xr[d] * (1.0f + scale[d]) + shift[d];
        }
    };
    if (L > 1 && static_cast<int64_t>(L) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<size_t>(L), [&](size_t li) {
            row_fn(static_cast<int>(li));
        });
    } else {
        for (int l = 0; l < L; ++l) row_fn(l);
    }
}

inline void broadcast_mul(const float* X, const float* v, float* Y, int L, int D) {
    if (L <= 0 || D <= 0) return;
    auto row_fn = [&](int l) {
        const float* xr = X + static_cast<size_t>(l) * D;
        float* yr       = Y + static_cast<size_t>(l) * D;
        for (int d = 0; d < D; ++d) {
            yr[d] = xr[d] * v[d];
        }
    };
    if (L > 1 && static_cast<int64_t>(L) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<size_t>(L), [&](size_t li) {
            row_fn(static_cast<int>(li));
        });
    } else {
        for (int l = 0; l < L; ++l) row_fn(l);
    }
}

inline void layernorm_forward_inference_batched(const float* X, const float* gamma, const float* beta,
                                               float eps, float* Y, int R, int D) {
    if (R <= 0 || D <= 0) return;
    const float inv_D = 1.0f / static_cast<float>(D);
    auto row_fn = [&](int r) {
        const float* xr = X + static_cast<size_t>(r) * D;
        float* yr       = Y + static_cast<size_t>(r) * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += xr[i];
        float mean = sum * inv_D;
        float sumsq = 0.0f;
        for (int i = 0; i < D; ++i) {
            float diff = xr[i] - mean;
            sumsq += diff * diff;
        }
        float var = sumsq * inv_D;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i) {
            float g = gamma ? gamma[i] : 1.0f;
            float b = beta ? beta[i] : 0.0f;
            yr[i] = g * ((xr[i] - mean) * rstd) + b;
        }
    };
    if (R > 1 && static_cast<int64_t>(R) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<size_t>(R), [&](size_t ri) {
            row_fn(static_cast<int>(ri));
        });
    } else {
        for (int r = 0; r < R; ++r) row_fn(r);
    }
}

} // namespace ref
} // namespace brotensor::detail::cpu::jit

#if BROTENSOR_HAS_BRASS_JIT

#include <brass/codegen/kernel_jit.hpp>
#include <brass/codegen/ml_fusion.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <cmath>
#include <mutex>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

namespace brotensor::detail::cpu::jit {

namespace {

using namespace brass;
using namespace brass::codegen;

struct JitKernels {
    bool available = false;

    FusedSwiGLUFn swiglu_fn = nullptr;

    using AdaLNModulateFn = void (*)(const float* x, const float* scale, const float* shift, float* out, uint64_t n);
    AdaLNModulateFn adaln_fn = nullptr;

    using BroadcastMulFn = void (*)(const float* x, const float* v, float* out, uint64_t n);
    BroadcastMulFn broadcast_mul_fn = nullptr;

    using RmsNormRowFn = void (*)(const float* x, const float* gamma, float* y, uint64_t d, float eps, float inv_d);
    RmsNormRowFn rms_norm_row_fn = nullptr;

    using LayerNormRowFn = void (*)(const float* x, const float* gamma, const float* beta, float* y, uint64_t d, float eps, float inv_d);
    LayerNormRowFn layernorm_row_fn = nullptr;

    std::vector<std::shared_ptr<JitExecutionEngine>> engines;
};

// Horizontal sum of an 8-wide float vector using destination scratch buffer
static Value* reduce_vsum8(KernelBuilder& kb, Value* scratch, Value* vsum) {
    kb.vstore_f32x8(scratch, vsum);
    Value* s0 = kb.load_f32(scratch, 0);  Value* s1 = kb.load_f32(scratch, 4);
    Value* s2 = kb.load_f32(scratch, 8);  Value* s3 = kb.load_f32(scratch, 12);
    Value* s4 = kb.load_f32(scratch, 16); Value* s5 = kb.load_f32(scratch, 20);
    Value* s6 = kb.load_f32(scratch, 24); Value* s7 = kb.load_f32(scratch, 28);
    Value* sum0123 = kb.add(kb.add(s0, s1), kb.add(s2, s3));
    Value* sum4567 = kb.add(kb.add(s4, s5), kb.add(s6, s7));
    return kb.add(sum0123, sum4567);
}

// ── 1. Vectorized Broadcast Mul (AVX2 8-wide + scalar remainder) ────────────
static KernelFunction build_broadcast_mul(KernelJit& jit) {
    Module mod("mod_broadcast_mul");
    Function* fn = mod.create_function("broadcast_mul", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* v = kb.builder().add_block_param(entry, Type::ptr());
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* v_head = kb.builder().create_block("v_head");
    BasicBlock* v_body = kb.builder().create_block("v_body");
    BasicBlock* s_head = kb.builder().create_block("s_head");
    BasicBlock* s_body = kb.builder().create_block("s_body");
    BasicBlock* exit = kb.builder().create_block("exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    Value* two_i64 = kb.builder().build_iconst_i64(2);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* mask_eight = kb.builder().build_iconst_i64(~int64_t(7));
    Value* vec_n = kb.builder().build_and(n, mask_eight);
    kb.builder().build_br(v_head, {zero_i64});

    // Vector loop: 8 elements per iteration
    fn->append_block(v_head);
    Value* iv_v = kb.builder().add_block_param(v_head, Type::i64());
    kb.position_at_end(v_head);
    Value* v_cond = kb.builder().build_slt(iv_v, vec_n);
    kb.builder().build_br_if(v_cond, v_body, {}, s_head, {iv_v});

    fn->append_block(v_body);
    kb.position_at_end(v_body);
    Value* byte_off = kb.builder().build_shl(iv_v, two_i64);
    Value* px = kb.builder().build_add(x, byte_off);
    Value* pv = kb.builder().build_add(v, byte_off);
    Value* py = kb.builder().build_add(out, byte_off);
    Value* vx = kb.vload_f32x8(px);
    Value* vv = kb.vload_f32x8(pv);
    Value* vy = kb.vmul(vx, vv);
    kb.vstore_f32x8(py, vy);
    Value* next_iv_v = kb.builder().build_add(iv_v, eight_i64);
    kb.builder().build_br(v_head, {next_iv_v});

    // Scalar remainder loop
    fn->append_block(s_head);
    Value* iv_s = kb.builder().add_block_param(s_head, Type::i64());
    kb.position_at_end(s_head);
    Value* s_cond = kb.builder().build_slt(iv_s, n);
    kb.builder().build_br_if(s_cond, s_body, exit);

    fn->append_block(s_body);
    kb.position_at_end(s_body);
    Value* xi = kb.load_f32_indexed(x, iv_s, 4, 0);
    Value* vi = kb.load_f32_indexed(v, iv_s, 4, 0);
    Value* yi = kb.mul(xi, vi);
    kb.store_f32_indexed(out, iv_s, yi, 4, 0);
    Value* next_iv_s = kb.builder().build_add(iv_s, one_i64);
    kb.builder().build_br(s_head, {next_iv_s});

    fn->append_block(exit);
    kb.position_at_end(exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "broadcast_mul");
}

// ── 2. Vectorized AdaLN Modulate (AVX2 8-wide + FMA + scalar remainder) ─────
static KernelFunction build_adaln_modulate(KernelJit& jit) {
    Module mod("mod_adaln_modulate");
    Function* fn = mod.create_function("fused_adaln_modulate", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* scale = kb.builder().add_block_param(entry, Type::ptr());
    Value* shift = kb.builder().add_block_param(entry, Type::ptr());
    Value* out = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* v_head = kb.builder().create_block("v_head");
    BasicBlock* v_body = kb.builder().create_block("v_body");
    BasicBlock* s_head = kb.builder().create_block("s_head");
    BasicBlock* s_body = kb.builder().create_block("s_body");
    BasicBlock* exit = kb.builder().create_block("exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    Value* two_i64 = kb.builder().build_iconst_i64(2);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* one_f32 = kb.builder().build_fconst_f32(1.0f);
    Value* v_one = kb.vbroadcast(Type::f32x8(), one_f32);
    Value* mask_eight = kb.builder().build_iconst_i64(~int64_t(7));
    Value* vec_n = kb.builder().build_and(n, mask_eight);
    kb.builder().build_br(v_head, {zero_i64});

    // Vector loop: modulated = x * (1.0f + scale) + shift via AVX2 FMA
    fn->append_block(v_head);
    Value* iv_v = kb.builder().add_block_param(v_head, Type::i64());
    kb.position_at_end(v_head);
    Value* v_cond = kb.builder().build_slt(iv_v, vec_n);
    kb.builder().build_br_if(v_cond, v_body, {}, s_head, {iv_v});

    fn->append_block(v_body);
    kb.position_at_end(v_body);
    Value* byte_off = kb.builder().build_shl(iv_v, two_i64);
    Value* px = kb.builder().build_add(x, byte_off);
    Value* pscale = kb.builder().build_add(scale, byte_off);
    Value* pshift = kb.builder().build_add(shift, byte_off);
    Value* py = kb.builder().build_add(out, byte_off);
    Value* vx = kb.vload_f32x8(px);
    Value* vscale = kb.vload_f32x8(pscale);
    Value* vshift = kb.vload_f32x8(pshift);
    Value* v_1_plus_scale = kb.vadd(v_one, vscale);
    Value* vy = kb.vfma(vx, v_1_plus_scale, vshift);
    kb.vstore_f32x8(py, vy);
    Value* next_iv_v = kb.builder().build_add(iv_v, eight_i64);
    kb.builder().build_br(v_head, {next_iv_v});

    // Scalar remainder loop
    fn->append_block(s_head);
    Value* iv_s = kb.builder().add_block_param(s_head, Type::i64());
    kb.position_at_end(s_head);
    Value* s_cond = kb.builder().build_slt(iv_s, n);
    kb.builder().build_br_if(s_cond, s_body, exit);

    fn->append_block(s_body);
    kb.position_at_end(s_body);
    Value* xi = kb.load_f32_indexed(x, iv_s, 4, 0);
    Value* scale_i = kb.load_f32_indexed(scale, iv_s, 4, 0);
    Value* shift_i = kb.load_f32_indexed(shift, iv_s, 4, 0);
    Value* one_plus_scale = kb.add(one_f32, scale_i);
    Value* yi = kb.builder().build_fma_f32(xi, one_plus_scale, shift_i);
    kb.store_f32_indexed(out, iv_s, yi, 4, 0);
    Value* next_iv_s = kb.builder().build_add(iv_s, one_i64);
    kb.builder().build_br(s_head, {next_iv_s});

    fn->append_block(exit);
    kb.position_at_end(exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "fused_adaln_modulate");
}

// ── 3. Vectorized RMSNorm Row (AVX2 FMA reduction + AVX2 affine scale) ─────
static KernelFunction build_rms_norm_row(KernelJit& jit) {
    Module mod("mod_rms_norm_row");
    mod.add_external_symbol("rsqrtf");
    Function* fn = mod.create_function("rms_norm_row", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32(), Type::f32()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* gamma = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* d = kb.builder().add_block_param(entry, Type::i64());
    Value* eps = kb.builder().add_block_param(entry, Type::f32());
    Value* inv_d = kb.builder().add_block_param(entry, Type::f32());

    BasicBlock* l1_vhead = kb.builder().create_block("l1_vhead");
    BasicBlock* l1_vbody = kb.builder().create_block("l1_vbody");
    BasicBlock* l1_vexit = kb.builder().create_block("l1_vexit");
    BasicBlock* l1_shead = kb.builder().create_block("l1_shead");
    BasicBlock* l1_sbody = kb.builder().create_block("l1_sbody");
    BasicBlock* l1_exit = kb.builder().create_block("l1_exit");

    BasicBlock* l2_vhead = kb.builder().create_block("l2_vhead");
    BasicBlock* l2_vbody = kb.builder().create_block("l2_vbody");
    BasicBlock* l2_shead = kb.builder().create_block("l2_shead");
    BasicBlock* l2_sbody = kb.builder().create_block("l2_sbody");
    BasicBlock* l2_exit = kb.builder().create_block("l2_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    Value* two_i64 = kb.builder().build_iconst_i64(2);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* vzero = kb.vzero(Type::f32x8());
    Value* mask_eight = kb.builder().build_iconst_i64(~int64_t(7));
    Value* vec_d = kb.builder().build_and(d, mask_eight);

    Value* has_vec = kb.builder().build_sge(d, eight_i64);
    kb.builder().build_br_if(has_vec, l1_vhead, {zero_i64, vzero}, l1_shead, {zero_i64, zero_f32});

    // Pass 1: Vector loop accumulating sum_sq via vfma
    fn->append_block(l1_vhead);
    Value* iv1_v = kb.builder().add_block_param(l1_vhead, Type::i64());
    Value* vsum = kb.builder().add_block_param(l1_vhead, Type::f32x8());
    kb.position_at_end(l1_vhead);
    Value* cond1_v = kb.builder().build_slt(iv1_v, vec_d);
    kb.builder().build_br_if(cond1_v, l1_vbody, {}, l1_vexit, {iv1_v, vsum});

    fn->append_block(l1_vbody);
    kb.position_at_end(l1_vbody);
    Value* byte_off1 = kb.builder().build_shl(iv1_v, two_i64);
    Value* px1 = kb.builder().build_add(x, byte_off1);
    Value* vx1 = kb.vload_f32x8(px1);
    Value* next_vsum = kb.vfma(vx1, vx1, vsum);
    Value* next_iv1_v = kb.builder().build_add(iv1_v, eight_i64);
    kb.builder().build_br(l1_vhead, {next_iv1_v, next_vsum});

    // Pass 1: Vector exit -> horizontal vector reduction via y scratch buffer
    fn->append_block(l1_vexit);
    Value* exit_iv1 = kb.builder().add_block_param(l1_vexit, Type::i64());
    Value* exit_vsum = kb.builder().add_block_param(l1_vexit, Type::f32x8());
    kb.position_at_end(l1_vexit);
    Value* init_scalar_sum = reduce_vsum8(kb, y, exit_vsum);
    kb.builder().build_br(l1_shead, {exit_iv1, init_scalar_sum});

    // Pass 1: Scalar remainder loop
    fn->append_block(l1_shead);
    Value* iv1_s = kb.builder().add_block_param(l1_shead, Type::i64());
    Value* sum_sq = kb.builder().add_block_param(l1_shead, Type::f32());
    kb.position_at_end(l1_shead);
    Value* cond1_s = kb.builder().build_slt(iv1_s, d);
    kb.builder().build_br_if(cond1_s, l1_sbody, {}, l1_exit, {sum_sq});

    fn->append_block(l1_sbody);
    kb.position_at_end(l1_sbody);
    Value* xi = kb.load_f32_indexed(x, iv1_s, 4, 0);
    Value* xi_sq = kb.mul(xi, xi);
    Value* next_sum = kb.add(sum_sq, xi_sq);
    Value* next_iv1_s = kb.builder().build_add(iv1_s, one_i64);
    kb.builder().build_br(l1_shead, {next_iv1_s, next_sum});

    // Pass 1 exit -> compute rrms = rsqrtf(mean_sq + eps)
    fn->append_block(l1_exit);
    Value* final_sum = kb.builder().add_block_param(l1_exit, Type::f32());
    kb.position_at_end(l1_exit);
    Value* mean_sq = kb.mul(final_sum, inv_d);
    Value* denom = kb.add(mean_sq, eps);
    Value* rrms = kb.builder().build_call("rsqrtf", Type::f32(), {denom});
    Value* vrrms = kb.vbroadcast(Type::f32x8(), rrms);
    kb.builder().build_br(l2_vhead, {zero_i64, rrms, vrrms});

    // Pass 2: Vector loop y[i] = x[i] * gamma[i] * rrms
    fn->append_block(l2_vhead);
    Value* iv2_v = kb.builder().add_block_param(l2_vhead, Type::i64());
    Value* rrms_val = kb.builder().add_block_param(l2_vhead, Type::f32());
    Value* vrrms_val = kb.builder().add_block_param(l2_vhead, Type::f32x8());
    kb.position_at_end(l2_vhead);
    Value* cond2_v = kb.builder().build_slt(iv2_v, vec_d);
    kb.builder().build_br_if(cond2_v, l2_vbody, {}, l2_shead, {iv2_v, rrms_val});

    fn->append_block(l2_vbody);
    kb.position_at_end(l2_vbody);
    Value* byte_off2 = kb.builder().build_shl(iv2_v, two_i64);
    Value* px2 = kb.builder().build_add(x, byte_off2);
    Value* pg2 = kb.builder().build_add(gamma, byte_off2);
    Value* py2 = kb.builder().build_add(y, byte_off2);
    Value* vx2 = kb.vload_f32x8(px2);
    Value* vg2 = kb.vload_f32x8(pg2);
    Value* vxg = kb.vmul(vx2, vg2);
    Value* vy = kb.vmul(vxg, vrrms_val);
    kb.vstore_f32x8(py2, vy);
    Value* next_iv2_v = kb.builder().build_add(iv2_v, eight_i64);
    kb.builder().build_br(l2_vhead, {next_iv2_v, rrms_val, vrrms_val});

    // Pass 2: Scalar remainder loop
    fn->append_block(l2_shead);
    Value* iv2_s = kb.builder().add_block_param(l2_shead, Type::i64());
    Value* rrms_scalar = kb.builder().add_block_param(l2_shead, Type::f32());
    kb.position_at_end(l2_shead);
    Value* cond2_s = kb.builder().build_slt(iv2_s, d);
    kb.builder().build_br_if(cond2_s, l2_sbody, l2_exit);

    fn->append_block(l2_sbody);
    kb.position_at_end(l2_sbody);
    Value* x2 = kb.load_f32_indexed(x, iv2_s, 4, 0);
    Value* g2 = kb.load_f32_indexed(gamma, iv2_s, 4, 0);
    Value* xg2 = kb.mul(x2, g2);
    Value* out_val = kb.mul(xg2, rrms_scalar);
    kb.store_f32_indexed(y, iv2_s, out_val, 4, 0);
    Value* next_iv2_s = kb.builder().build_add(iv2_s, one_i64);
    kb.builder().build_br(l2_shead, {next_iv2_s, rrms_scalar});

    fn->append_block(l2_exit);
    kb.position_at_end(l2_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "rms_norm_row");
}

// ── 4. Vectorized LayerNorm Row (AVX2 mean + var + affine) ──────────────────
static KernelFunction build_layernorm_row(KernelJit& jit) {
    Module mod("mod_layernorm_row");
    mod.add_external_symbol("rsqrtf");
    Function* fn = mod.create_function("layernorm_row", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32(), Type::f32()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* gamma = kb.builder().add_block_param(entry, Type::ptr());
    Value* beta = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* d = kb.builder().add_block_param(entry, Type::i64());
    Value* eps = kb.builder().add_block_param(entry, Type::f32());
    Value* inv_d = kb.builder().add_block_param(entry, Type::f32());

    // Pass 1 blocks (mean)
    BasicBlock* l1_vhead = kb.builder().create_block("ln_l1_vhead");
    BasicBlock* l1_vbody = kb.builder().create_block("ln_l1_vbody");
    BasicBlock* l1_vexit = kb.builder().create_block("ln_l1_vexit");
    BasicBlock* l1_shead = kb.builder().create_block("ln_l1_shead");
    BasicBlock* l1_sbody = kb.builder().create_block("ln_l1_sbody");
    BasicBlock* l1_exit = kb.builder().create_block("ln_l1_exit");

    // Pass 2 blocks (variance)
    BasicBlock* l2_vhead = kb.builder().create_block("ln_l2_vhead");
    BasicBlock* l2_vbody = kb.builder().create_block("ln_l2_vbody");
    BasicBlock* l2_vexit = kb.builder().create_block("ln_l2_vexit");
    BasicBlock* l2_shead = kb.builder().create_block("ln_l2_shead");
    BasicBlock* l2_sbody = kb.builder().create_block("ln_l2_sbody");
    BasicBlock* l2_exit = kb.builder().create_block("ln_l2_exit");

    // Pass 3 blocks (affine)
    BasicBlock* l3_vhead = kb.builder().create_block("ln_l3_vhead");
    BasicBlock* l3_vbody = kb.builder().create_block("ln_l3_vbody");
    BasicBlock* l3_shead = kb.builder().create_block("ln_l3_shead");
    BasicBlock* l3_sbody = kb.builder().create_block("ln_l3_sbody");
    BasicBlock* l3_exit = kb.builder().create_block("ln_l3_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    Value* two_i64 = kb.builder().build_iconst_i64(2);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* vzero = kb.vzero(Type::f32x8());
    Value* mask_eight = kb.builder().build_iconst_i64(~int64_t(7));
    Value* vec_d = kb.builder().build_and(d, mask_eight);

    Value* has_vec = kb.builder().build_sge(d, eight_i64);
    kb.builder().build_br_if(has_vec, l1_vhead, {zero_i64, vzero}, l1_shead, {zero_i64, zero_f32});

    // Pass 1: Vector sum
    fn->append_block(l1_vhead);
    Value* iv1_v = kb.builder().add_block_param(l1_vhead, Type::i64());
    Value* vsum1 = kb.builder().add_block_param(l1_vhead, Type::f32x8());
    kb.position_at_end(l1_vhead);
    Value* cond1_v = kb.builder().build_slt(iv1_v, vec_d);
    kb.builder().build_br_if(cond1_v, l1_vbody, {}, l1_vexit, {iv1_v, vsum1});

    fn->append_block(l1_vbody);
    kb.position_at_end(l1_vbody);
    Value* byte_off1 = kb.builder().build_shl(iv1_v, two_i64);
    Value* px1 = kb.builder().build_add(x, byte_off1);
    Value* vx1 = kb.vload_f32x8(px1);
    Value* next_vsum1 = kb.vadd(vsum1, vx1);
    Value* next_iv1_v = kb.builder().build_add(iv1_v, eight_i64);
    kb.builder().build_br(l1_vhead, {next_iv1_v, next_vsum1});

    // Pass 1: Vector exit -> horizontal sum
    fn->append_block(l1_vexit);
    Value* exit_iv1 = kb.builder().add_block_param(l1_vexit, Type::i64());
    Value* exit_vsum1 = kb.builder().add_block_param(l1_vexit, Type::f32x8());
    kb.position_at_end(l1_vexit);
    Value* init_sum1 = reduce_vsum8(kb, y, exit_vsum1);
    kb.builder().build_br(l1_shead, {exit_iv1, init_sum1});

    // Pass 1: Scalar remainder
    fn->append_block(l1_shead);
    Value* iv1_s = kb.builder().add_block_param(l1_shead, Type::i64());
    Value* sum1 = kb.builder().add_block_param(l1_shead, Type::f32());
    kb.position_at_end(l1_shead);
    Value* cond1_s = kb.builder().build_slt(iv1_s, d);
    kb.builder().build_br_if(cond1_s, l1_sbody, {}, l1_exit, {sum1});

    fn->append_block(l1_sbody);
    kb.position_at_end(l1_sbody);
    Value* xi1 = kb.load_f32_indexed(x, iv1_s, 4, 0);
    Value* next_sum1 = kb.add(sum1, xi1);
    Value* next_iv1_s = kb.builder().build_add(iv1_s, one_i64);
    kb.builder().build_br(l1_shead, {next_iv1_s, next_sum1});

    // Pass 1: Mean computed -> enter Pass 2
    fn->append_block(l1_exit);
    Value* final_sum1 = kb.builder().add_block_param(l1_exit, Type::f32());
    kb.position_at_end(l1_exit);
    Value* mean = kb.mul(final_sum1, inv_d);
    Value* vmean = kb.vbroadcast(Type::f32x8(), mean);
    kb.builder().build_br_if(has_vec, l2_vhead, {zero_i64, vzero, mean, vmean}, l2_shead, {zero_i64, zero_f32, mean});

    // Pass 2: Vector sum of squared deviations
    fn->append_block(l2_vhead);
    Value* iv2_v = kb.builder().add_block_param(l2_vhead, Type::i64());
    Value* vsum2 = kb.builder().add_block_param(l2_vhead, Type::f32x8());
    Value* m2_scalar = kb.builder().add_block_param(l2_vhead, Type::f32());
    Value* vm2 = kb.builder().add_block_param(l2_vhead, Type::f32x8());
    kb.position_at_end(l2_vhead);
    Value* cond2_v = kb.builder().build_slt(iv2_v, vec_d);
    kb.builder().build_br_if(cond2_v, l2_vbody, {}, l2_vexit, {iv2_v, vsum2, m2_scalar});

    fn->append_block(l2_vbody);
    kb.position_at_end(l2_vbody);
    Value* byte_off2 = kb.builder().build_shl(iv2_v, two_i64);
    Value* px2 = kb.builder().build_add(x, byte_off2);
    Value* vx2 = kb.vload_f32x8(px2);
    Value* vdev = kb.vsub(vx2, vm2);
    Value* next_vsum2 = kb.vfma(vdev, vdev, vsum2);
    Value* next_iv2_v = kb.builder().build_add(iv2_v, eight_i64);
    kb.builder().build_br(l2_vhead, {next_iv2_v, next_vsum2, m2_scalar, vm2});

    // Pass 2: Vector exit -> horizontal sum
    fn->append_block(l2_vexit);
    Value* exit_iv2 = kb.builder().add_block_param(l2_vexit, Type::i64());
    Value* exit_vsum2 = kb.builder().add_block_param(l2_vexit, Type::f32x8());
    Value* m2_out = kb.builder().add_block_param(l2_vexit, Type::f32());
    kb.position_at_end(l2_vexit);
    Value* init_sumsq = reduce_vsum8(kb, y, exit_vsum2);
    kb.builder().build_br(l2_shead, {exit_iv2, init_sumsq, m2_out});

    // Pass 2: Scalar remainder
    fn->append_block(l2_shead);
    Value* iv2_s = kb.builder().add_block_param(l2_shead, Type::i64());
    Value* sumsq = kb.builder().add_block_param(l2_shead, Type::f32());
    Value* m2_s = kb.builder().add_block_param(l2_shead, Type::f32());
    kb.position_at_end(l2_shead);
    Value* cond2_s = kb.builder().build_slt(iv2_s, d);
    kb.builder().build_br_if(cond2_s, l2_sbody, {}, l2_exit, {sumsq, m2_s});

    fn->append_block(l2_sbody);
    kb.position_at_end(l2_sbody);
    Value* xi2 = kb.load_f32_indexed(x, iv2_s, 4, 0);
    Value* diff2 = kb.sub(xi2, m2_s);
    Value* diff_sq2 = kb.mul(diff2, diff2);
    Value* next_sumsq = kb.add(sumsq, diff_sq2);
    Value* next_iv2_s = kb.builder().build_add(iv2_s, one_i64);
    kb.builder().build_br(l2_shead, {next_iv2_s, next_sumsq, m2_s});

    // Pass 2 exit -> compute rstd
    fn->append_block(l2_exit);
    Value* final_sumsq = kb.builder().add_block_param(l2_exit, Type::f32());
    Value* final_mean = kb.builder().add_block_param(l2_exit, Type::f32());
    kb.position_at_end(l2_exit);
    Value* var = kb.mul(final_sumsq, inv_d);
    Value* var_eps = kb.add(var, eps);
    Value* rstd = kb.builder().build_call("rsqrtf", Type::f32(), {var_eps});
    Value* vmean3 = kb.vbroadcast(Type::f32x8(), final_mean);
    Value* vrstd3 = kb.vbroadcast(Type::f32x8(), rstd);
    kb.builder().build_br(l3_vhead, {zero_i64, final_mean, rstd, vmean3, vrstd3});

    // Pass 3: Vector affine y[i] = gamma[i] * ((x[i] - mean) * rstd) + beta[i]
    fn->append_block(l3_vhead);
    Value* iv3_v = kb.builder().add_block_param(l3_vhead, Type::i64());
    Value* m3_s = kb.builder().add_block_param(l3_vhead, Type::f32());
    Value* r3_s = kb.builder().add_block_param(l3_vhead, Type::f32());
    Value* vm3 = kb.builder().add_block_param(l3_vhead, Type::f32x8());
    Value* vr3 = kb.builder().add_block_param(l3_vhead, Type::f32x8());
    kb.position_at_end(l3_vhead);
    Value* cond3_v = kb.builder().build_slt(iv3_v, vec_d);
    kb.builder().build_br_if(cond3_v, l3_vbody, {}, l3_shead, {iv3_v, m3_s, r3_s});

    fn->append_block(l3_vbody);
    kb.position_at_end(l3_vbody);
    Value* byte_off3 = kb.builder().build_shl(iv3_v, two_i64);
    Value* px3 = kb.builder().build_add(x, byte_off3);
    Value* pg3 = kb.builder().build_add(gamma, byte_off3);
    Value* pb3 = kb.builder().build_add(beta, byte_off3);
    Value* py3 = kb.builder().build_add(y, byte_off3);
    Value* vx3 = kb.vload_f32x8(px3);
    Value* vg3 = kb.vload_f32x8(pg3);
    Value* vb3 = kb.vload_f32x8(pb3);
    Value* vdiff3 = kb.vsub(vx3, vm3);
    Value* vxhat3 = kb.vmul(vdiff3, vr3);
    Value* vy3 = kb.vfma(vg3, vxhat3, vb3);
    kb.vstore_f32x8(py3, vy3);
    Value* next_iv3_v = kb.builder().build_add(iv3_v, eight_i64);
    kb.builder().build_br(l3_vhead, {next_iv3_v, m3_s, r3_s, vm3, vr3});

    // Pass 3: Scalar remainder
    fn->append_block(l3_shead);
    Value* iv3_s = kb.builder().add_block_param(l3_shead, Type::i64());
    Value* m3_sc = kb.builder().add_block_param(l3_shead, Type::f32());
    Value* r3_sc = kb.builder().add_block_param(l3_shead, Type::f32());
    kb.position_at_end(l3_shead);
    Value* cond3_s = kb.builder().build_slt(iv3_s, d);
    kb.builder().build_br_if(cond3_s, l3_sbody, l3_exit);

    fn->append_block(l3_sbody);
    kb.position_at_end(l3_sbody);
    Value* x3 = kb.load_f32_indexed(x, iv3_s, 4, 0);
    Value* g3 = kb.load_f32_indexed(gamma, iv3_s, 4, 0);
    Value* b3 = kb.load_f32_indexed(beta, iv3_s, 4, 0);
    Value* diff3 = kb.sub(x3, m3_sc);
    Value* xhat3 = kb.mul(diff3, r3_sc);
    Value* y3 = kb.builder().build_fma_f32(g3, xhat3, b3);
    kb.store_f32_indexed(y, iv3_s, y3, 4, 0);
    Value* next_iv3_s = kb.builder().build_add(iv3_s, one_i64);
    kb.builder().build_br(l3_shead, {next_iv3_s, m3_sc, r3_sc});

    fn->append_block(l3_exit);
    kb.position_at_end(l3_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "layernorm_row");
}

// ── Kernel Initialization & JIT Engine Management ───────────────────────────
static const JitKernels& get_kernels() {
    static JitKernels kernels;
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        try {
            KernelOptions opts;
            opts.enable_optimizations = true;
            opts.enable_avx2 = true;
            opts.enable_fma = true;
            opts.enable_vectorize = true;
            opts.enable_unroll = true;
            opts.unroll_factor = 4;
            opts.enable_fp_reassociation = true;

            KernelJit ml_jit(opts);
            MlFusionCompiler ml_compiler(std::move(ml_jit));
            KernelJit jit(opts);

            // 1. SwiGLU
            KernelFunction k_swiglu = ml_compiler.compile_swiglu();
            kernels.swiglu_fn = k_swiglu.as<FusedSwiGLUFn>();
            kernels.engines.push_back(k_swiglu.engine());

            // 2. Vectorized AdaLN Modulate
            KernelFunction k_adaln = build_adaln_modulate(jit);
            kernels.adaln_fn = k_adaln.as<JitKernels::AdaLNModulateFn>();
            kernels.engines.push_back(k_adaln.engine());

            // 3. Vectorized Broadcast Mul
            KernelFunction k_bmul = build_broadcast_mul(jit);
            kernels.broadcast_mul_fn = k_bmul.as<JitKernels::BroadcastMulFn>();
            kernels.engines.push_back(k_bmul.engine());

            // 4. Vectorized RMSNorm row
            KernelFunction k_rmsnorm = build_rms_norm_row(jit);
            kernels.rms_norm_row_fn = k_rmsnorm.as<JitKernels::RmsNormRowFn>();
            kernels.engines.push_back(k_rmsnorm.engine());

            // 5. Vectorized LayerNorm row
            KernelFunction k_ln = build_layernorm_row(jit);
            kernels.layernorm_row_fn = k_ln.as<JitKernels::LayerNormRowFn>();
            kernels.engines.push_back(k_ln.engine());

            kernels.available = (kernels.swiglu_fn != nullptr &&
                                 kernels.adaln_fn != nullptr &&
                                 kernels.broadcast_mul_fn != nullptr &&
                                 kernels.rms_norm_row_fn != nullptr &&
                                 kernels.layernorm_row_fn != nullptr);
        } catch (const std::exception& e) {
            std::cerr << "brotensor: Brass JIT initialization failed: " << e.what() << "\n";
            kernels.available = false;
        }
    });
    return kernels;
}

} // namespace

bool is_jit_available() {
    return get_kernels().available;
}

void rms_norm_forward(const float* X, const float* gamma, float eps, float* Y, int B, int D) {
    const auto& k = get_kernels();
    if (!k.available || k.rms_norm_row_fn == nullptr) {
        ref::rms_norm(X, gamma, eps, Y, B, D);
        return;
    }

    const float inv_D = 1.0f / static_cast<float>(D);

    // Single-thread vectorized loop executes in sub-microsecond latency.
    // Multi-threaded pool dispatch only when workload amortizes thread wake-up overhead.
    if (B > 1 && static_cast<int64_t>(B) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<std::size_t>(B), [&](std::size_t bi) {
            const int b = static_cast<int>(bi);
            const float* xr = X + static_cast<std::size_t>(b) * D;
            float* yr = Y + static_cast<std::size_t>(b) * D;
            k.rms_norm_row_fn(xr, gamma, yr, static_cast<uint64_t>(D), eps, inv_D);
        });
    } else {
        for (int b = 0; b < B; ++b) {
            const float* xr = X + static_cast<std::size_t>(b) * D;
            float* yr = Y + static_cast<std::size_t>(b) * D;
            k.rms_norm_row_fn(xr, gamma, yr, static_cast<uint64_t>(D), eps, inv_D);
        }
    }
}

void swiglu_forward(const float* X, float* Y, int B, int D) {
    const auto& k = get_kernels();
    if (!k.available || k.swiglu_fn == nullptr) {
        ref::swiglu(X, Y, B, D);
        return;
    }

    if (B > 1 && static_cast<int64_t>(B) * D >= 16384) {
        detail::cpu::parallel_for(static_cast<std::size_t>(B), [&](std::size_t bi) {
            const int b = static_cast<int>(bi);
            const float* gate = X + static_cast<std::size_t>(b) * 2 * D;
            const float* up   = gate + D;
            float* out        = Y + static_cast<std::size_t>(b) * D;
            k.swiglu_fn(gate, up, out, static_cast<uint64_t>(D));
        });
    } else {
        for (int b = 0; b < B; ++b) {
            const float* gate = X + static_cast<std::size_t>(b) * 2 * D;
            const float* up   = gate + D;
            float* out        = Y + static_cast<std::size_t>(b) * D;
            k.swiglu_fn(gate, up, out, static_cast<uint64_t>(D));
        }
    }
}

void modulate(const float* X, const float* scale, const float* shift, float* Y, int L, int D) {
    const auto& k = get_kernels();
    if (!k.available || k.adaln_fn == nullptr) {
        ref::modulate(X, scale, shift, Y, L, D);
        return;
    }

    if (L > 1 && static_cast<int64_t>(L) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<std::size_t>(L), [&](std::size_t li) {
            const int l = static_cast<int>(li);
            const float* xr = X + static_cast<std::size_t>(l) * D;
            float* yr = Y + static_cast<std::size_t>(l) * D;
            k.adaln_fn(xr, scale, shift, yr, static_cast<uint64_t>(D));
        });
    } else {
        for (int l = 0; l < L; ++l) {
            const float* xr = X + static_cast<std::size_t>(l) * D;
            float* yr = Y + static_cast<std::size_t>(l) * D;
            k.adaln_fn(xr, scale, shift, yr, static_cast<uint64_t>(D));
        }
    }
}

void broadcast_mul(const float* X, const float* v, float* Y, int L, int D) {
    const auto& k = get_kernels();
    if (!k.available || k.broadcast_mul_fn == nullptr) {
        ref::broadcast_mul(X, v, Y, L, D);
        return;
    }

    if (L > 1 && static_cast<int64_t>(L) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<std::size_t>(L), [&](std::size_t li) {
            const int l = static_cast<int>(li);
            const float* xr = X + static_cast<std::size_t>(l) * D;
            float* yr = Y + static_cast<std::size_t>(l) * D;
            k.broadcast_mul_fn(xr, v, yr, static_cast<uint64_t>(D));
        });
    } else {
        for (int l = 0; l < L; ++l) {
            const float* xr = X + static_cast<std::size_t>(l) * D;
            float* yr = Y + static_cast<std::size_t>(l) * D;
            k.broadcast_mul_fn(xr, v, yr, static_cast<uint64_t>(D));
        }
    }
}

void layernorm_forward_inference_batched(const float* X, const float* gamma, const float* beta, float eps, float* Y, int R, int D) {
    const auto& k = get_kernels();
    if (!k.available || k.layernorm_row_fn == nullptr) {
        ref::layernorm_forward_inference_batched(X, gamma, beta, eps, Y, R, D);
        return;
    }

    const float inv_D = 1.0f / static_cast<float>(D);

    if (R > 1 && static_cast<int64_t>(R) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<std::size_t>(R), [&](std::size_t ri) {
            const int r = static_cast<int>(ri);
            const float* xr = X + static_cast<std::size_t>(r) * D;
            float* yr = Y + static_cast<std::size_t>(r) * D;
            k.layernorm_row_fn(xr, gamma, beta, yr, static_cast<uint64_t>(D), eps, inv_D);
        });
    } else {
        for (int r = 0; r < R; ++r) {
            const float* xr = X + static_cast<std::size_t>(r) * D;
            float* yr = Y + static_cast<std::size_t>(r) * D;
            k.layernorm_row_fn(xr, gamma, beta, yr, static_cast<uint64_t>(D), eps, inv_D);
        }
    }
}

} // namespace brotensor::detail::cpu::jit

#else // !BROTENSOR_HAS_BRASS_JIT

namespace brotensor::detail::cpu::jit {

bool is_jit_available() { return false; }
void rms_norm_forward(const float* X, const float* gamma, float eps, float* Y, int B, int D) {
    ref::rms_norm(X, gamma, eps, Y, B, D);
}
void swiglu_forward(const float* X, float* Y, int B, int D) {
    ref::swiglu(X, Y, B, D);
}
void modulate(const float* X, const float* scale, const float* shift, float* Y, int L, int D) {
    ref::modulate(X, scale, shift, Y, L, D);
}
void broadcast_mul(const float* X, const float* v, float* Y, int L, int D) {
    ref::broadcast_mul(X, v, Y, L, D);
}
void layernorm_forward_inference_batched(const float* X, const float* gamma, const float* beta, float eps, float* Y, int R, int D) {
    ref::layernorm_forward_inference_batched(X, gamma, beta, eps, Y, R, D);
}

} // namespace brotensor::detail::cpu::jit

#endif // BROTENSOR_HAS_BRASS_JIT
