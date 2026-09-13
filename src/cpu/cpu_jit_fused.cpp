#include "cpu_jit.h"
#include <brotensor/detail/cpu/thread_pool.h>

#if BROTENSOR_HAS_BRASS_JIT

#include <brass/codegen/kernel_jit.hpp>
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

struct JitFusedKernels {
    bool available = false;

    using FusedResidualRmsNormRowFn = void (*)(float* x, const float* res, const float* gamma, float* y, uint64_t d, float eps, float inv_d);
    FusedResidualRmsNormRowFn fused_residual_rmsnorm_row_fn = nullptr;

    using FusedLayerNormModulateRowFn = void (*)(const float* x, const float* gamma, const float* beta, const float* scale, const float* shift, float* y, uint64_t d, float eps, float inv_d);
    FusedLayerNormModulateRowFn fused_layernorm_modulate_row_fn = nullptr;

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

// ── 1. Fused Residual RMSNorm Row (Option B: X += res in-place + RMSNorm) ───
//
// Computes in-place residual addition X += res, writes X, and accumulates sum_sq
// in the EXACT same vector register pass, eliminating a full memory read/write pass.
static KernelFunction build_fused_residual_rmsnorm_row(KernelJit& jit) {
    Module mod("mod_fused_residual_rmsnorm_row");
    mod.add_external_symbol("rsqrtf");
    Function* fn = mod.create_function("fused_residual_rmsnorm_row", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32(), Type::f32()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* res = kb.builder().add_block_param(entry, Type::ptr());
    Value* gamma = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* d = kb.builder().add_block_param(entry, Type::i64());
    Value* eps = kb.builder().add_block_param(entry, Type::f32());
    Value* inv_d = kb.builder().add_block_param(entry, Type::f32());

    BasicBlock* l1_vhead = kb.builder().create_block("res_rms_l1_vhead");
    BasicBlock* l1_vbody = kb.builder().create_block("res_rms_l1_vbody");
    BasicBlock* l1_vexit = kb.builder().create_block("res_rms_l1_vexit");
    BasicBlock* l1_shead = kb.builder().create_block("res_rms_l1_shead");
    BasicBlock* l1_sbody = kb.builder().create_block("res_rms_l1_sbody");
    BasicBlock* l1_exit = kb.builder().create_block("res_rms_l1_exit");

    BasicBlock* l2_vhead = kb.builder().create_block("res_rms_l2_vhead");
    BasicBlock* l2_vbody = kb.builder().create_block("res_rms_l2_vbody");
    BasicBlock* l2_shead = kb.builder().create_block("res_rms_l2_shead");
    BasicBlock* l2_sbody = kb.builder().create_block("res_rms_l2_sbody");
    BasicBlock* l2_exit = kb.builder().create_block("res_rms_l2_exit");

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

    // Pass 1: Vector loop X += res in-place AND accumulates sum_sq in same register pass
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
    Value* pres1 = kb.builder().build_add(res, byte_off1);
    Value* vx1 = kb.vload_f32x8(px1);
    Value* vr1 = kb.vload_f32x8(pres1);
    Value* vx_new = kb.vadd(vx1, vr1);
    kb.vstore_f32x8(px1, vx_new); // X += res in-place
    Value* next_vsum = kb.vfma(vx_new, vx_new, vsum); // sum_sq accumulation in exact same pass
    Value* next_iv1_v = kb.builder().build_add(iv1_v, eight_i64);
    kb.builder().build_br(l1_vhead, {next_iv1_v, next_vsum});

    // Pass 1: Vector exit -> horizontal reduction
    fn->append_block(l1_vexit);
    Value* exit_iv1 = kb.builder().add_block_param(l1_vexit, Type::i64());
    Value* exit_vsum = kb.builder().add_block_param(l1_vexit, Type::f32x8());
    kb.position_at_end(l1_vexit);
    Value* init_scalar_sum = reduce_vsum8(kb, y, exit_vsum);
    kb.builder().build_br(l1_shead, {exit_iv1, init_scalar_sum});

    // Pass 1: Scalar remainder
    fn->append_block(l1_shead);
    Value* iv1_s = kb.builder().add_block_param(l1_shead, Type::i64());
    Value* sum_sq = kb.builder().add_block_param(l1_shead, Type::f32());
    kb.position_at_end(l1_shead);
    Value* cond1_s = kb.builder().build_slt(iv1_s, d);
    kb.builder().build_br_if(cond1_s, l1_sbody, {}, l1_exit, {sum_sq});

    fn->append_block(l1_sbody);
    kb.position_at_end(l1_sbody);
    Value* xi = kb.load_f32_indexed(x, iv1_s, 4, 0);
    Value* ri = kb.load_f32_indexed(res, iv1_s, 4, 0);
    Value* xi_new = kb.add(xi, ri);
    kb.store_f32_indexed(x, iv1_s, xi_new, 4, 0);
    Value* xi_sq = kb.mul(xi_new, xi_new);
    Value* next_sum = kb.add(sum_sq, xi_sq);
    Value* next_iv1_s = kb.builder().build_add(iv1_s, one_i64);
    kb.builder().build_br(l1_shead, {next_iv1_s, next_sum});

    // Pass 1 exit -> compute rrms
    fn->append_block(l1_exit);
    Value* final_sum = kb.builder().add_block_param(l1_exit, Type::f32());
    kb.position_at_end(l1_exit);
    Value* mean_sq = kb.mul(final_sum, inv_d);
    Value* denom = kb.add(mean_sq, eps);
    Value* rrms = kb.builder().build_call("rsqrtf", Type::f32(), {denom});
    Value* vrrms = kb.vbroadcast(Type::f32x8(), rrms);
    kb.builder().build_br(l2_vhead, {zero_i64, rrms, vrrms});

    // Pass 2: Vector loop y = x * gamma * rrms
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

    // Pass 2: Scalar remainder
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

    return jit.compile(mod, "fused_residual_rmsnorm_row");
}

// ── 2. Fused LayerNorm + Modulate Row (Option B: Zero Intermediate Memory) ──
//
// Computes LayerNorm directly into registers and immediately applies AdaLN
// modulation by (1 + scale) + shift into output Y, writing zero intermediate
// tensors to memory.
static KernelFunction build_fused_layernorm_modulate_row(KernelJit& jit) {
    Module mod("mod_fused_layernorm_modulate_row");
    mod.add_external_symbol("rsqrtf");
    Function* fn = mod.create_function("fused_layernorm_modulate_row", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::f32(), Type::f32()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* gamma = kb.builder().add_block_param(entry, Type::ptr());
    Value* beta = kb.builder().add_block_param(entry, Type::ptr());
    Value* scale = kb.builder().add_block_param(entry, Type::ptr());
    Value* shift = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* d = kb.builder().add_block_param(entry, Type::i64());
    Value* eps = kb.builder().add_block_param(entry, Type::f32());
    Value* inv_d = kb.builder().add_block_param(entry, Type::f32());

    // Pass 1 blocks (mean)
    BasicBlock* l1_vhead = kb.builder().create_block("flm_l1_vhead");
    BasicBlock* l1_vbody = kb.builder().create_block("flm_l1_vbody");
    BasicBlock* l1_vexit = kb.builder().create_block("flm_l1_vexit");
    BasicBlock* l1_shead = kb.builder().create_block("flm_l1_shead");
    BasicBlock* l1_sbody = kb.builder().create_block("flm_l1_sbody");
    BasicBlock* l1_exit = kb.builder().create_block("flm_l1_exit");

    // Pass 2 blocks (variance)
    BasicBlock* l2_vhead = kb.builder().create_block("flm_l2_vhead");
    BasicBlock* l2_vbody = kb.builder().create_block("flm_l2_vbody");
    BasicBlock* l2_vexit = kb.builder().create_block("flm_l2_vexit");
    BasicBlock* l2_shead = kb.builder().create_block("flm_l2_shead");
    BasicBlock* l2_sbody = kb.builder().create_block("flm_l2_sbody");
    BasicBlock* l2_exit = kb.builder().create_block("flm_l2_exit");

    // Pass 3 blocks (fused LayerNorm + Modulate directly to Y)
    BasicBlock* l3_vhead = kb.builder().create_block("flm_l3_vhead");
    BasicBlock* l3_vbody = kb.builder().create_block("flm_l3_vbody");
    BasicBlock* l3_shead = kb.builder().create_block("flm_l3_shead");
    BasicBlock* l3_sbody = kb.builder().create_block("flm_l3_sbody");
    BasicBlock* l3_exit = kb.builder().create_block("flm_l3_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    Value* two_i64 = kb.builder().build_iconst_i64(2);
    Value* eight_i64 = kb.builder().build_iconst_i64(8);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_f32 = kb.builder().build_fconst_f32(1.0f);
    Value* vzero = kb.vzero(Type::f32x8());
    Value* vone = kb.vbroadcast(Type::f32x8(), one_f32);
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

    // Pass 3: Vector fused LayerNorm + AdaLN Modulate directly into registers, zero intermediate RAM
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
    Value* pscale3 = kb.builder().build_add(scale, byte_off3);
    Value* pshift3 = kb.builder().build_add(shift, byte_off3);
    Value* py3 = kb.builder().build_add(y, byte_off3);
    Value* vx3 = kb.vload_f32x8(px3);
    Value* vg3 = kb.vload_f32x8(pg3);
    Value* vb3 = kb.vload_f32x8(pb3);
    Value* vscale3 = kb.vload_f32x8(pscale3);
    Value* vshift3 = kb.vload_f32x8(pshift3);
    Value* vdiff3 = kb.vsub(vx3, vm3);
    Value* vxhat3 = kb.vmul(vdiff3, vr3);
    Value* v_ln3 = kb.vfma(vg3, vxhat3, vb3); // LayerNorm in registers
    Value* v_1_plus_scale = kb.vadd(vone, vscale3);
    Value* vy3 = kb.vfma(v_ln3, v_1_plus_scale, vshift3); // Modulated in registers
    kb.vstore_f32x8(py3, vy3); // Stored directly to destination!
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
    Value* s3 = kb.load_f32_indexed(scale, iv3_s, 4, 0);
    Value* sh3 = kb.load_f32_indexed(shift, iv3_s, 4, 0);
    Value* diff3 = kb.sub(x3, m3_sc);
    Value* xhat3 = kb.mul(diff3, r3_sc);
    Value* ln_val3 = kb.builder().build_fma_f32(g3, xhat3, b3);
    Value* one_plus_s3 = kb.add(one_f32, s3);
    Value* y3 = kb.builder().build_fma_f32(ln_val3, one_plus_s3, sh3);
    kb.store_f32_indexed(y, iv3_s, y3, 4, 0);
    Value* next_iv3_s = kb.builder().build_add(iv3_s, one_i64);
    kb.builder().build_br(l3_shead, {next_iv3_s, m3_sc, r3_sc});

    fn->append_block(l3_exit);
    kb.position_at_end(l3_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "fused_layernorm_modulate_row");
}

// ── Kernel Initialization & JIT Engine Management ───────────────────────────
static const JitFusedKernels& get_fused_kernels() {
    static JitFusedKernels kernels;
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

            KernelJit jit(opts);

            // 1. Fused Residual RMSNorm row (Option B)
            KernelFunction k_fused_res_rms = build_fused_residual_rmsnorm_row(jit);
            kernels.fused_residual_rmsnorm_row_fn = k_fused_res_rms.as<JitFusedKernels::FusedResidualRmsNormRowFn>();
            kernels.engines.push_back(k_fused_res_rms.engine());

            // 2. Fused LayerNorm Modulate row (Option B)
            KernelFunction k_fused_ln_mod = build_fused_layernorm_modulate_row(jit);
            kernels.fused_layernorm_modulate_row_fn = k_fused_ln_mod.as<JitFusedKernels::FusedLayerNormModulateRowFn>();
            kernels.engines.push_back(k_fused_ln_mod.engine());

            kernels.available = (kernels.fused_residual_rmsnorm_row_fn != nullptr &&
                                 kernels.fused_layernorm_modulate_row_fn != nullptr);
        } catch (const std::exception& e) {
            std::cerr << "brotensor: Brass JIT fused kernels initialization failed: " << e.what() << "\n";
            kernels.available = false;
        }
    });
    return kernels;
}

} // namespace

void fused_residual_rmsnorm(float* X, const float* res, const float* gamma, float eps, float* Y, int B, int D) {
    const auto& k = get_fused_kernels();
    if (!k.available || k.fused_residual_rmsnorm_row_fn == nullptr) return;

    const float inv_D = 1.0f / static_cast<float>(D);

    if (B > 1 && static_cast<int64_t>(B) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<std::size_t>(B), [&](std::size_t bi) {
            const int b = static_cast<int>(bi);
            float* xr = X + static_cast<std::size_t>(b) * D;
            const float* rr = res + static_cast<std::size_t>(b) * D;
            float* yr = Y + static_cast<std::size_t>(b) * D;
            k.fused_residual_rmsnorm_row_fn(xr, rr, gamma, yr, static_cast<uint64_t>(D), eps, inv_D);
        });
    } else {
        for (int b = 0; b < B; ++b) {
            float* xr = X + static_cast<std::size_t>(b) * D;
            const float* rr = res + static_cast<std::size_t>(b) * D;
            float* yr = Y + static_cast<std::size_t>(b) * D;
            k.fused_residual_rmsnorm_row_fn(xr, rr, gamma, yr, static_cast<uint64_t>(D), eps, inv_D);
        }
    }
}

void fused_layernorm_modulate(const float* X, const float* gamma, const float* beta,
                              const float* scale, const float* shift, float eps, float* Y, int R, int D) {
    const auto& k = get_fused_kernels();
    if (!k.available || k.fused_layernorm_modulate_row_fn == nullptr) return;

    const float inv_D = 1.0f / static_cast<float>(D);

    if (R > 1 && static_cast<int64_t>(R) * D >= 262144) {
        detail::cpu::parallel_for(static_cast<std::size_t>(R), [&](std::size_t ri) {
            const int r = static_cast<int>(ri);
            const float* xr = X + static_cast<std::size_t>(r) * D;
            float* yr = Y + static_cast<std::size_t>(r) * D;
            k.fused_layernorm_modulate_row_fn(xr, gamma, beta, scale, shift, yr, static_cast<uint64_t>(D), eps, inv_D);
        });
    } else {
        for (int r = 0; r < R; ++r) {
            const float* xr = X + static_cast<std::size_t>(r) * D;
            float* yr = Y + static_cast<std::size_t>(r) * D;
            k.fused_layernorm_modulate_row_fn(xr, gamma, beta, scale, shift, yr, static_cast<uint64_t>(D), eps, inv_D);
        }
    }
}

} // namespace brotensor::detail::cpu::jit

#else // !BROTENSOR_HAS_BRASS_JIT

namespace brotensor::detail::cpu::jit {

void fused_residual_rmsnorm(float*, const float*, const float*, float, float*, int, int) {}
void fused_layernorm_modulate(const float*, const float*, const float*, const float*, const float*, float, float*, int, int) {}

} // namespace brotensor::detail::cpu::jit

#endif // BROTENSOR_HAS_BRASS_JIT
