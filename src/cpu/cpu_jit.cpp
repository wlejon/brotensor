#include "cpu_jit.h"
#include <brotensor/detail/cpu/thread_pool.h>

#if BROTENSOR_HAS_BRASS_JIT

#include <brass/codegen/kernel_jit.hpp>
#include <brass/codegen/ml_fusion.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
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
    FusedAdaLNModulateFn adaln_fn = nullptr;

    using BroadcastMulFn = void (*)(const float* x, const float* v, float* out, uint64_t n);
    BroadcastMulFn broadcast_mul_fn = nullptr;

    using RmsNormRowFn = void (*)(const float* x, const float* gamma, float* y, uint64_t d, float eps, float inv_d);
    RmsNormRowFn rms_norm_row_fn = nullptr;

    using LayerNormRowFn = void (*)(const float* x, const float* gamma, const float* beta, float* y, uint64_t d, float eps, float inv_d);
    LayerNormRowFn layernorm_row_fn = nullptr;

    // Chunk helpers for N=1 multi-threaded parallelization
    using SumSqChunkFn = float (*)(const float* x, uint64_t n);
    SumSqChunkFn sum_sq_chunk_fn = nullptr;

    using ScaleGammaChunkFn = void (*)(const float* x, const float* gamma, float* y, float rrms, uint64_t n);
    ScaleGammaChunkFn scale_gamma_chunk_fn = nullptr;

    using SumChunkFn = float (*)(const float* x, uint64_t n);
    SumChunkFn sum_chunk_fn = nullptr;

    using SumSqDevChunkFn = float (*)(const float* x, float mean, uint64_t n);
    SumSqDevChunkFn sum_sq_dev_chunk_fn = nullptr;

    using AffineNormChunkFn = void (*)(const float* x, const float* gamma, const float* beta, float* y, float mean, float rstd, uint64_t n);
    AffineNormChunkFn affine_norm_chunk_fn = nullptr;

    std::vector<std::shared_ptr<JitExecutionEngine>> engines;
};

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

    BasicBlock* loop_head = kb.builder().create_block("b_head");
    BasicBlock* loop_body = kb.builder().create_block("b_body");
    BasicBlock* loop_exit = kb.builder().create_block("b_exit");

    kb.position_at_end(entry);
    Value* zero = kb.builder().build_iconst_i64(0);
    Value* one = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero});

    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, loop_exit);

    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* vi = kb.load_f32_indexed(v, i, 4, 0);
    Value* yi = kb.mul(xi, vi);
    kb.store_f32_indexed(out, i, yi, 4, 0);
    Value* next_i = kb.add(i, one);
    kb.builder().build_br(loop_head, {next_i});

    fn->append_block(loop_exit);
    kb.position_at_end(loop_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "broadcast_mul");
}

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

    BasicBlock* l1_head = kb.builder().create_block("l1_head");
    BasicBlock* l1_body = kb.builder().create_block("l1_body");
    BasicBlock* l1_exit = kb.builder().create_block("l1_exit");
    BasicBlock* l2_head = kb.builder().create_block("l2_head");
    BasicBlock* l2_body = kb.builder().create_block("l2_body");
    BasicBlock* l2_exit = kb.builder().create_block("l2_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(l1_head, {zero_i64, zero_f32});

    // Pass 1: compute sum of squares
    fn->append_block(l1_head);
    Value* i1 = kb.builder().add_block_param(l1_head, Type::i64());
    Value* sum_sq = kb.builder().add_block_param(l1_head, Type::f32());
    kb.position_at_end(l1_head);
    Value* cond1 = kb.builder().build_slt(i1, d);
    kb.builder().build_br_if(cond1, l1_body, {}, l1_exit, {sum_sq});

    fn->append_block(l1_body);
    kb.position_at_end(l1_body);
    Value* xi = kb.load_f32_indexed(x, i1, 4, 0);
    Value* xi_sq = kb.mul(xi, xi);
    Value* next_sum = kb.add(sum_sq, xi_sq);
    Value* next_i1 = kb.add(i1, one_i64);
    kb.builder().build_br(l1_head, {next_i1, next_sum});

    fn->append_block(l1_exit);
    Value* final_sum = kb.builder().add_block_param(l1_exit, Type::f32());
    kb.position_at_end(l1_exit);
    Value* mean_sq = kb.mul(final_sum, inv_d);
    Value* denom = kb.add(mean_sq, eps);
    Value* rrms = kb.builder().build_call("rsqrtf", Type::f32(), {denom});
    kb.builder().build_br(l2_head, {zero_i64, rrms});

    // Pass 2: y[i] = x[i] * gamma[i] * rrms
    fn->append_block(l2_head);
    Value* i2 = kb.builder().add_block_param(l2_head, Type::i64());
    Value* rrms_val = kb.builder().add_block_param(l2_head, Type::f32());
    kb.position_at_end(l2_head);
    Value* cond2 = kb.builder().build_slt(i2, d);
    kb.builder().build_br_if(cond2, l2_body, l2_exit);

    fn->append_block(l2_body);
    kb.position_at_end(l2_body);
    Value* x2 = kb.load_f32_indexed(x, i2, 4, 0);
    Value* g2 = kb.load_f32_indexed(gamma, i2, 4, 0);
    Value* xg = kb.mul(x2, g2);
    Value* out_val = kb.mul(xg, rrms_val);
    kb.store_f32_indexed(y, i2, out_val, 4, 0);
    Value* next_i2 = kb.add(i2, one_i64);
    kb.builder().build_br(l2_head, {next_i2, rrms_val});

    fn->append_block(l2_exit);
    kb.position_at_end(l2_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "rms_norm_row");
}

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

    BasicBlock* l1_head = kb.builder().create_block("ln_l1_head");
    BasicBlock* l1_body = kb.builder().create_block("ln_l1_body");
    BasicBlock* l1_exit = kb.builder().create_block("ln_l1_exit");
    BasicBlock* l2_head = kb.builder().create_block("ln_l2_head");
    BasicBlock* l2_body = kb.builder().create_block("ln_l2_body");
    BasicBlock* l2_exit = kb.builder().create_block("ln_l2_exit");
    BasicBlock* l3_head = kb.builder().create_block("ln_l3_head");
    BasicBlock* l3_body = kb.builder().create_block("ln_l3_body");
    BasicBlock* l3_exit = kb.builder().create_block("ln_l3_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(l1_head, {zero_i64, zero_f32});

    // Pass 1: compute mean
    fn->append_block(l1_head);
    Value* i1 = kb.builder().add_block_param(l1_head, Type::i64());
    Value* sum1 = kb.builder().add_block_param(l1_head, Type::f32());
    kb.position_at_end(l1_head);
    Value* cond1 = kb.builder().build_slt(i1, d);
    kb.builder().build_br_if(cond1, l1_body, {}, l1_exit, {sum1});

    fn->append_block(l1_body);
    kb.position_at_end(l1_body);
    Value* xi = kb.load_f32_indexed(x, i1, 4, 0);
    Value* next_sum1 = kb.add(sum1, xi);
    Value* next_i1 = kb.add(i1, one_i64);
    kb.builder().build_br(l1_head, {next_i1, next_sum1});

    fn->append_block(l1_exit);
    Value* final_sum1 = kb.builder().add_block_param(l1_exit, Type::f32());
    kb.position_at_end(l1_exit);
    Value* mean = kb.mul(final_sum1, inv_d);
    kb.builder().build_br(l2_head, {zero_i64, zero_f32, mean});

    // Pass 2: compute sum of squared deviations from mean
    fn->append_block(l2_head);
    Value* i2 = kb.builder().add_block_param(l2_head, Type::i64());
    Value* sumsq = kb.builder().add_block_param(l2_head, Type::f32());
    Value* mean_param2 = kb.builder().add_block_param(l2_head, Type::f32());
    kb.position_at_end(l2_head);
    Value* cond2 = kb.builder().build_slt(i2, d);
    kb.builder().build_br_if(cond2, l2_body, {}, l2_exit, {sumsq, mean_param2});

    fn->append_block(l2_body);
    kb.position_at_end(l2_body);
    Value* x2 = kb.load_f32_indexed(x, i2, 4, 0);
    Value* dev = kb.sub(x2, mean_param2);
    Value* dev_sq = kb.mul(dev, dev);
    Value* next_sumsq = kb.add(sumsq, dev_sq);
    Value* next_i2 = kb.add(i2, one_i64);
    kb.builder().build_br(l2_head, {next_i2, next_sumsq, mean_param2});

    fn->append_block(l2_exit);
    Value* final_sumsq = kb.builder().add_block_param(l2_exit, Type::f32());
    Value* mean_exit2 = kb.builder().add_block_param(l2_exit, Type::f32());
    kb.position_at_end(l2_exit);
    Value* var = kb.mul(final_sumsq, inv_d);
    Value* var_eps = kb.add(var, eps);
    Value* rstd = kb.builder().build_call("rsqrtf", Type::f32(), {var_eps});
    kb.builder().build_br(l3_head, {zero_i64, mean_exit2, rstd});

    // Pass 3: y[i] = gamma[i] * (x[i] - mean) * rstd + beta[i]
    fn->append_block(l3_head);
    Value* i3 = kb.builder().add_block_param(l3_head, Type::i64());
    Value* mean_param3 = kb.builder().add_block_param(l3_head, Type::f32());
    Value* rstd_param3 = kb.builder().add_block_param(l3_head, Type::f32());
    kb.position_at_end(l3_head);
    Value* cond3 = kb.builder().build_slt(i3, d);
    kb.builder().build_br_if(cond3, l3_body, l3_exit);

    fn->append_block(l3_body);
    kb.position_at_end(l3_body);
    Value* x3 = kb.load_f32_indexed(x, i3, 4, 0);
    Value* g3 = kb.load_f32_indexed(gamma, i3, 4, 0);
    Value* b3 = kb.load_f32_indexed(beta, i3, 4, 0);
    Value* diff3 = kb.sub(x3, mean_param3);
    Value* xhat3 = kb.mul(diff3, rstd_param3);
    Value* scaled3 = kb.mul(g3, xhat3);
    Value* out3 = kb.add(scaled3, b3);
    kb.store_f32_indexed(y, i3, out3, 4, 0);
    Value* next_i3 = kb.add(i3, one_i64);
    kb.builder().build_br(l3_head, {next_i3, mean_param3, rstd_param3});

    fn->append_block(l3_exit);
    kb.position_at_end(l3_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "layernorm_row");
}

static KernelFunction build_sum_sq_chunk(KernelJit& jit) {
    Module mod("mod_sum_sq_chunk");
    Function* fn = mod.create_function("sum_sq_chunk", Type::f32(), {
        Type::ptr(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("l_head");
    BasicBlock* loop_body = kb.builder().create_block("l_body");
    BasicBlock* loop_exit = kb.builder().create_block("l_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero_i64, zero_f32});

    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    Value* sum = kb.builder().add_block_param(loop_head, Type::f32());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, {}, loop_exit, {sum});

    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* xi_sq = kb.mul(xi, xi);
    Value* next_sum = kb.add(sum, xi_sq);
    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i, next_sum});

    fn->append_block(loop_exit);
    Value* final_sum = kb.builder().add_block_param(loop_exit, Type::f32());
    kb.position_at_end(loop_exit);
    kb.builder().build_ret(final_sum);

    return jit.compile(mod, "sum_sq_chunk");
}

static KernelFunction build_scale_gamma_chunk(KernelJit& jit) {
    Module mod("mod_scale_gamma_chunk");
    Function* fn = mod.create_function("scale_gamma_chunk", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::f32(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* gamma = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* rrms = kb.builder().add_block_param(entry, Type::f32());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("l_head");
    BasicBlock* loop_body = kb.builder().create_block("l_body");
    BasicBlock* loop_exit = kb.builder().create_block("l_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero_i64});

    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, loop_exit);

    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* gi = kb.load_f32_indexed(gamma, i, 4, 0);
    Value* xg = kb.mul(xi, gi);
    Value* yi = kb.mul(xg, rrms);
    kb.store_f32_indexed(y, i, yi, 4, 0);
    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i});

    fn->append_block(loop_exit);
    kb.position_at_end(loop_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "scale_gamma_chunk");
}

static KernelFunction build_sum_chunk(KernelJit& jit) {
    Module mod("mod_sum_chunk");
    Function* fn = mod.create_function("sum_chunk", Type::f32(), {
        Type::ptr(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("l_head");
    BasicBlock* loop_body = kb.builder().create_block("l_body");
    BasicBlock* loop_exit = kb.builder().create_block("l_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero_i64, zero_f32});

    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    Value* sum = kb.builder().add_block_param(loop_head, Type::f32());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, {}, loop_exit, {sum});

    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* next_sum = kb.add(sum, xi);
    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i, next_sum});

    fn->append_block(loop_exit);
    Value* final_sum = kb.builder().add_block_param(loop_exit, Type::f32());
    kb.position_at_end(loop_exit);
    kb.builder().build_ret(final_sum);

    return jit.compile(mod, "sum_chunk");
}

static KernelFunction build_sum_sq_dev_chunk(KernelJit& jit) {
    Module mod("mod_sum_sq_dev_chunk");
    Function* fn = mod.create_function("sum_sq_dev_chunk", Type::f32(), {
        Type::ptr(), Type::f32(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* mean = kb.builder().add_block_param(entry, Type::f32());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("l_head");
    BasicBlock* loop_body = kb.builder().create_block("l_body");
    BasicBlock* loop_exit = kb.builder().create_block("l_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* zero_f32 = kb.builder().build_fconst_f32(0.0f);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero_i64, zero_f32});

    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    Value* sumsq = kb.builder().add_block_param(loop_head, Type::f32());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, {}, loop_exit, {sumsq});

    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* dev = kb.sub(xi, mean);
    Value* dev_sq = kb.mul(dev, dev);
    Value* next_sumsq = kb.add(sumsq, dev_sq);
    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i, next_sumsq});

    fn->append_block(loop_exit);
    Value* final_sumsq = kb.builder().add_block_param(loop_exit, Type::f32());
    kb.position_at_end(loop_exit);
    kb.builder().build_ret(final_sumsq);

    return jit.compile(mod, "sum_sq_dev_chunk");
}

static KernelFunction build_affine_norm_chunk(KernelJit& jit) {
    Module mod("mod_affine_norm_chunk");
    Function* fn = mod.create_function("affine_norm_chunk", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::f32(), Type::f32(), Type::i64()
    });
    KernelBuilder kb(mod, fn);

    BasicBlock* entry = kb.builder().append_block("entry");
    Value* x = kb.builder().add_block_param(entry, Type::ptr());
    Value* gamma = kb.builder().add_block_param(entry, Type::ptr());
    Value* beta = kb.builder().add_block_param(entry, Type::ptr());
    Value* y = kb.builder().add_block_param(entry, Type::ptr());
    Value* mean = kb.builder().add_block_param(entry, Type::f32());
    Value* rstd = kb.builder().add_block_param(entry, Type::f32());
    Value* n = kb.builder().add_block_param(entry, Type::i64());

    BasicBlock* loop_head = kb.builder().create_block("l_head");
    BasicBlock* loop_body = kb.builder().create_block("l_body");
    BasicBlock* loop_exit = kb.builder().create_block("l_exit");

    kb.position_at_end(entry);
    Value* zero_i64 = kb.builder().build_iconst_i64(0);
    Value* one_i64 = kb.builder().build_iconst_i64(1);
    kb.builder().build_br(loop_head, {zero_i64});

    fn->append_block(loop_head);
    Value* i = kb.builder().add_block_param(loop_head, Type::i64());
    kb.position_at_end(loop_head);
    Value* cond = kb.builder().build_slt(i, n);
    kb.builder().build_br_if(cond, loop_body, loop_exit);

    fn->append_block(loop_body);
    kb.position_at_end(loop_body);
    Value* xi = kb.load_f32_indexed(x, i, 4, 0);
    Value* gi = kb.load_f32_indexed(gamma, i, 4, 0);
    Value* bi = kb.load_f32_indexed(beta, i, 4, 0);
    Value* diff = kb.sub(xi, mean);
    Value* xhat = kb.mul(diff, rstd);
    Value* scaled = kb.mul(gi, xhat);
    Value* out_val = kb.add(scaled, bi);
    kb.store_f32_indexed(y, i, out_val, 4, 0);
    Value* next_i = kb.add(i, one_i64);
    kb.builder().build_br(loop_head, {next_i});

    fn->append_block(loop_exit);
    kb.position_at_end(loop_exit);
    kb.builder().build_ret_void();

    return jit.compile(mod, "affine_norm_chunk");
}

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

            // 2. AdaLN Modulate
            KernelFunction k_adaln = ml_compiler.compile_adaln_modulate(false);
            kernels.adaln_fn = k_adaln.as<FusedAdaLNModulateFn>();
            kernels.engines.push_back(k_adaln.engine());

            // 3. Broadcast Mul
            KernelFunction k_bmul = build_broadcast_mul(jit);
            kernels.broadcast_mul_fn = k_bmul.as<JitKernels::BroadcastMulFn>();
            kernels.engines.push_back(k_bmul.engine());

            // 4. RMSNorm row
            KernelFunction k_rmsnorm = build_rms_norm_row(jit);
            kernels.rms_norm_row_fn = k_rmsnorm.as<JitKernels::RmsNormRowFn>();
            kernels.engines.push_back(k_rmsnorm.engine());

            // 5. LayerNorm row
            KernelFunction k_ln = build_layernorm_row(jit);
            kernels.layernorm_row_fn = k_ln.as<JitKernels::LayerNormRowFn>();
            kernels.engines.push_back(k_ln.engine());

            // 6-10. Chunk helpers for N=1 multi-threading
            KernelFunction k_sq = build_sum_sq_chunk(jit);
            kernels.sum_sq_chunk_fn = k_sq.as<JitKernels::SumSqChunkFn>();
            kernels.engines.push_back(k_sq.engine());

            KernelFunction k_sc = build_scale_gamma_chunk(jit);
            kernels.scale_gamma_chunk_fn = k_sc.as<JitKernels::ScaleGammaChunkFn>();
            kernels.engines.push_back(k_sc.engine());

            KernelFunction k_sum = build_sum_chunk(jit);
            kernels.sum_chunk_fn = k_sum.as<JitKernels::SumChunkFn>();
            kernels.engines.push_back(k_sum.engine());

            KernelFunction k_dev = build_sum_sq_dev_chunk(jit);
            kernels.sum_sq_dev_chunk_fn = k_dev.as<JitKernels::SumSqDevChunkFn>();
            kernels.engines.push_back(k_dev.engine());

            KernelFunction k_aff = build_affine_norm_chunk(jit);
            kernels.affine_norm_chunk_fn = k_aff.as<JitKernels::AffineNormChunkFn>();
            kernels.engines.push_back(k_aff.engine());

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
    if (!k.available) return;

    const float inv_D = 1.0f / static_cast<float>(D);

    // Only invoke thread pool if total work justifies the ~100-150us futex wakeup overhead.
    if (B > 1 && static_cast<int64_t>(B) * D >= 32768) {
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
    if (!k.available) return;

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
    if (!k.available) return;

    if (L > 1 && static_cast<int64_t>(L) * D >= 32768) {
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
    if (!k.available) return;

    if (L > 1 && static_cast<int64_t>(L) * D >= 32768) {
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
    if (!k.available) return;

    const float inv_D = 1.0f / static_cast<float>(D);

    if (R > 1 && static_cast<int64_t>(R) * D >= 32768) {
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
void rms_norm_forward(const float*, const float*, float, float*, int, int) {}
void swiglu_forward(const float*, float*, int, int) {}
void modulate(const float*, const float*, const float*, float*, int, int) {}
void broadcast_mul(const float*, const float*, float*, int, int) {}
void layernorm_forward_inference_batched(const float*, const float*, const float*, float, float*, int, int) {}

} // namespace brotensor::detail::cpu::jit

#endif // BROTENSOR_HAS_BRASS_JIT
