// ─── Issue #139: Lifter Audit of Vulkan .comp & PTX Backward Shaders ────────
//
// Lifts compiled Vulkan SPIR-V binaries (`shader_table.h` and runtime-emitted
// `jit::spirv` kernels) via `brass::spirv::parse_binary` into `spirv::Module`
// and verifies:
//   1. `brass::spirv::verify_spirv_structural` (SSA dominance & CFG invariants)
//   2. `brass::verify::verify_spirv_memory_safety` (workgroup `shared` memory
//      bounds `0 <= idx < array_length` via `BVSolver`)
//   3. `brass::verify::verify_spirv_barrier_uniformity` (uniform control-flow
//      workgroup barriers `OpControlBarrier` & subgroup collectives)
//   4. `brass::verify::prove_spirv_shader_safety` (division/reciprocal/sqrt
//      hazard analysis + full static compute shader audit)
// Plus PTX kernel verification via `verify_ptx_memory_safety` and
// `verify_ptx_barrier_uniformity`.

#include <brotensor/backward_verify.h>

#if BROTENSOR_HAS_BRASS_JIT
#include "jit/spirv_emit.h"
#include "jit/gpu_mir_emit_detail.h"
#include "jit/trace_dag.h"
#include "jit/trace_plan.h"
#include "kernels/mir_copy.h"
#include "kernels/mir_delta.h"
#include "kernels/mir_diffusion.h"
#include "kernels/mir_elementwise.h"
#include "kernels/mir_gnorm.h"
#include "kernels/mir_lstm.h"
#include "kernels/mir_norm.h"
#include "kernels/mir_quant_helpers.h"
#include "kernels/mir_reduce.h"
#include "kernels/mir_remap.h"
#include "kernels/mir_resample.h"
#include "kernels/mir_resample_bwd.h"
#include "kernels/mir_attn_small.h"
#include "kernels/mir_audio.h"
#include "kernels/mir_conv_direct.h"
#include "kernels/mir_fa_bwd.h"
#include "kernels/mir_fa_rows.h"
#include "kernels/mir_gemm_simt.h"
#include "kernels/mir_gemv.h"
#include "kernels/mir_gemv_q.h"
#include "kernels/mir_rope.h"
#include "kernels/mir_scatter.h"
#include "kernels/mir_select.h"
#include "kernels/mir_misc.h"
#include "kernels/mir_spectral.h"
#include "kernels/mir_vision.h"
#include "kernels/mir_topk.h"
#include "kernels/mir_xent.h"
#include "vulkan/shaders/op_codes.h"
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#endif

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
#include "shader_table.h"
#endif

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace brotensor::verify {

#if BROTENSOR_HAS_BRASS_JIT

namespace {

std::vector<brass::verify::GpuBufferSpec> buffer_specs_for_shader(const std::string& name) {
    if (name == "fill_bytes") {
        return {{0, 0, 1, 1, 1}};
    }
    if (name == "act_bwd_f32") {
        return {{0, 0, 3, 4, 16}, {1, 0, 3, 4, 16}, {2, 0, 3, 4, 16}};
    }
    if (name == "act_bwd_f16" || name == "act_bwd_bf16") {
        return {{0, 0, 3, 2, 16}, {1, 0, 3, 2, 16}, {2, 0, 3, 2, 16}};
    }
    if (name == "jit_spirv_elementwise_vec") {
        return {{0, 0, 3, 16, 16}, {1, 0, 3, 16, 16}, {2, 0, 3, 16, 16}};
    }
    if (name == "jit_spirv_elementwise_scalar") {
        return {{0, 0, 3, 4, 4}, {1, 0, 3, 4, 4}, {2, 0, 3, 4, 4}};
    }
    if (name == "rowvec_f32" || name == "rowvec_f16" || name == "rowvec_bf16") {
        const std::uint32_t eb = (name == "rowvec_f32") ? 4u : 2u;
        return {{0, 0, 4, eb, 16}, {1, 0, 5, eb, 16}, {2, 0, 5, eb, 16}, {3, 0, 4, eb, 16}};
    }
    if (name == "sum_cols_f32" || name == "sum_cols_f16" || name == "sum_cols_bf16") {
        const std::uint32_t eb = (name == "sum_cols_f32") ? 4u : 2u;
        return {{0, 0, 2, eb, 16, 3, -1, 1}, {1, 0, 3, eb, 16}};
    }
    if (name == "sum_rows_f32" || name == "sum_rows_f16" || name == "sum_rows_bf16") {
        const std::uint32_t eb = (name == "sum_rows_f32") ? 4u : 2u;
        return {{0, 0, 2, eb, 16, 3, -1, 1}, {1, 0, 2, eb, 16}};
    }
    if (name == "argmax_rows_f32" || name == "argmax_rows_f16" || name == "argmax_rows_bf16") {
        const std::uint32_t eb = (name == "argmax_rows_f32") ? 4u : 2u;
        return {{0, 0, 2, eb, 16, 3, -1, 1}, {1, 0, 2, 4, 16}};
    }
    if (name == "copy2d_b1" || name == "copy2d_b2" || name == "copy2d_b4") {
        const std::uint32_t eb = (name == "copy2d_b4") ? 4u : (name == "copy2d_b2") ? 2u : 1u;
        return {{0, 0, 5, eb, eb, 2, -1, 1}, {1, 0, 5, eb, eb, 3, -1, 1}};
    }
    if (name == "transpose_b2" || name == "transpose_b4") {
        const std::uint32_t eb = (name == "transpose_b4") ? 4u : 2u;
        return {{0, 0, 2, eb, eb, 3, 4, 1}, {1, 0, 2, eb, eb, 3, 4, 1}};
    }
    if (name == "jit_spirv_rmsnorm_64_vec" || name == "jit_spirv_rmsnorm_64_scalar") {
        return {{0, 4096, -1, 4, 16}, {1, 4096, -1, 4, 16}, {2, 256, -1, 4, 16}};
    }
    if (name == "jit_spirv_rmsnorm_4096_vec" || name == "jit_spirv_rmsnorm_4096_scalar") {
        return {{0, 262144, -1, 4, 16}, {1, 262144, -1, 4, 16}, {2, 16384, -1, 4, 16}};
    }
    if (name == "jit_spirv_layernorm_64_vec" || name == "jit_spirv_layernorm_64_scalar") {
        return {{0, 4096, -1, 4, 16}, {1, 4096, -1, 4, 16}, {2, 256, -1, 4, 16}, {3, 256, -1, 4, 16}};
    }
    if (name == "jit_spirv_layernorm_4096_vec" || name == "jit_spirv_layernorm_4096_scalar") {
        return {{0, 262144, -1, 4, 16}, {1, 262144, -1, 4, 16}, {2, 16384, -1, 4, 16}, {3, 16384, -1, 4, 16}};
    }
    if (name == "glu_f32" || name == "glu_f16" || name == "glu_bf16" ||
        name == "glu_f32_to_f16" || name == "glu_f32_to_bf16") {
        const std::uint32_t eb_x = (name == "glu_f16" || name == "glu_bf16") ? 2u : 4u;
        const std::uint32_t eb_y = (name == "glu_f32") ? 4u : 2u;
        return {
            {0, 0, 3, eb_x, 16, -1, -1, 2}, // x: 2 * n elements
            {1, 0, 3, eb_x, 16, -1, -1, 1}, // dy: n elements
            {2, 0, 3, eb_y, 16, -1, -1, 1}, // y: n elements
        };
    }
    if (name == "glu_bwd_f32" || name == "glu_bwd_f16" || name == "glu_bwd_bf16") {
        const std::uint32_t eb = (name == "glu_bwd_f32") ? 4u : 2u;
        return {
            {0, 0, 3, eb, 16, -1, -1, 2}, // x: 2 * n elements
            {1, 0, 3, eb, 16, -1, -1, 1}, // dy: n elements
            {2, 0, 3, eb, 16, -1, -1, 2}, // dx: 2 * n elements
        };
    }
    if (name == "attn_aux_f32" || name == "attn_aux_f16" || name == "attn_aux_bf16") {
        const std::uint32_t eb_y = (name == "attn_aux_f32") ? 4u : 2u;
        return {
            {0, 0, 3, 4, 16, 4, -1, 1},    // x: rows * n floats
            {1, 0, 3, eb_y, 16, 4, -1, 1}, // y: rows * n elements
            {2, 0, 3, 4, 16, -1, -1, 1},   // mask: rows floats (when non-null)
        };
    }
    if (name == "norm_f32" || name == "norm_f16" || name == "norm_bf16" ||
        name == "norm_f16_g32" || name == "norm_bf16_g32" || name == "norm_f32_y16" ||
        name == "norm_bwd_f32" || name == "norm_bwd_f16" || name == "norm_bwd_bf16" ||
        name == "norm_col_f32" || name == "norm_col_f16" || name == "norm_col_bf16") {
        const std::uint32_t eb_x =
            (name == "norm_f32" || name == "norm_f32_y16" || name == "norm_bwd_f32" || name == "norm_col_f32") ? 4u : 2u;
        const std::uint32_t eb_y =
            (name == "norm_f32" || name == "norm_bwd_f32" || name == "norm_col_f32") ? 4u : 2u;
        const std::uint32_t eb_g =
            (name == "norm_f16" || name == "norm_bf16" || name == "norm_bwd_f16" || name == "norm_bwd_bf16" ||
             name == "norm_col_f16" || name == "norm_col_bf16") ? 2u : 4u;
        return {
            {0, 0, 7, eb_x, 16, 8, -1, 1}, // x: rows * d
            {1, 0, 7, eb_y, 16, 8, -1, 1}, // y: rows * d
            {2, 0, 8, eb_g, 16, -1, -1, 1}, // g: d
            {3, 0, 8, eb_g, 16, -1, -1, 1}, // b: d
            {4, 0, 7, eb_x, 16, 8, -1, 1}, // xh: rows * d
            {5, 0, 7, 4, 16, -1, -1, 1},   // s0: rows floats
            {6, 0, 7, 4, 16, -1, -1, 1},   // s1: rows floats
        };
    }
    if (name == "bnorm" || name == "bnorm_bwd") {
        std::vector<brass::verify::GpuBufferSpec> specs = {
            {0, 0, 11, 4, 16, 12, 13, 1},  // x: n * c * hw floats
            {1, 0, 11, 4, 16, 12, 13, 1},  // y: n * c * hw floats
            {2, 0, 12, 4, 16, -1, -1, 1},  // g: c
            {3, 0, 12, 4, 16, -1, -1, 1},  // b: c
            {4, 0, 12, 4, 16, -1, -1, 1},  // rm: c
            {5, 0, 12, 4, 16, -1, -1, 1},  // rv: c
            {6, 0, 12, 4, 16, -1, -1, 1},  // sm: c
            {7, 0, 12, 4, 16, -1, -1, 1},  // sr: c
        };
        if (name == "bnorm_bwd") {
            specs.push_back({8, 0, 11, 4, 16, 12, 13, 1});  // dy
            specs.push_back({9, 0, 12, 4, 16, -1, -1, 1});  // dg
            specs.push_back({10, 0, 12, 4, 16, -1, -1, 1}); // db
        }
        return specs;
    }
    return {};
}

std::vector<brass::verify::GpuParamConstraint> param_constraints_for_shader(const std::string& name) {
    if (name == "rowvec_f32" || name == "rowvec_f16" || name == "rowvec_bf16") {
        return {{5, UINT32_MAX, 1, 0}};
    }
    if (name == "copy2d_b1" || name == "copy2d_b2" || name == "copy2d_b4") {
        return {{4, 2, 0, 0}, {4, 3, 0, 0}};
    }
    if (name == "glu_f32" || name == "glu_f16" || name == "glu_bf16" ||
        name == "glu_f32_to_f16" || name == "glu_f32_to_bf16" ||
        name == "glu_bwd_f32" || name == "glu_bwd_f16" || name == "glu_bwd_bf16") {
        return {
            {4, UINT32_MAX, 1, 0, UINT32_MAX, 0}, // d >= 1
            {3, UINT32_MAX, 0, 0, 4, 0},          // n % d == 0
        };
    }
    if (name == "attn_aux_f32" || name == "attn_aux_f16" || name == "attn_aux_bf16") {
        return {{4, UINT32_MAX, 1, 0}}; // n >= 1
    }
    if (name == "norm_f32" || name == "norm_f16" || name == "norm_bf16" ||
        name == "norm_f16_g32" || name == "norm_bf16_g32" || name == "norm_f32_y16" ||
        name == "norm_bwd_f32" || name == "norm_bwd_f16" || name == "norm_bwd_bf16" ||
        name == "norm_col_f32" || name == "norm_col_f16" || name == "norm_col_bf16") {
        return {{7, UINT32_MAX, 1, 0}, {8, UINT32_MAX, 1, 0}};
    }
    if (name == "bnorm" || name == "bnorm_bwd") {
        return {{13, UINT32_MAX, 1, 0}}; // hw >= 1
    }
    return {};
}

} // namespace

ShaderAuditSummary audit_spirv_binary_words(
    const std::string& name,
    const std::uint32_t* words,
    std::size_t nwords) {
    ShaderAuditSummary sum;
    sum.shader_name = name;
    if (!words || nwords == 0) {
        sum.diagnostic = "Empty SPIR-V word stream";
        return sum;
    }

    std::span<const std::uint32_t> sp(words, nwords);
    brass::spirv::Module mod;
    std::string parse_err;
    sum.parse_ok = brass::spirv::parse_binary(sp, mod, &parse_err);
    if (!sum.parse_ok) {
        sum.diagnostic = "parse_binary failed: " + parse_err;
        return sum;
    }

    std::string struct_err;
    sum.structural_ok = brass::spirv::verify_spirv_structural(mod, &struct_err);

    brass::spirv::ComputeShaderAuditReport rep = brass::verify::prove_spirv_shader_safety(mod);
    sum.num_functions = rep.num_functions;
    sum.num_blocks = rep.num_blocks;
    sum.num_instructions = rep.num_instructions;
    sum.num_workgroup_arrays = rep.num_workgroup_arrays;

    sum.memory_safe = rep.shared_bounds_ok;
    sum.barriers_uniform = rep.barriers_uniform;
    sum.div_sqrt_safe = rep.div_sqrt_ok;
    sum.shared_accesses_proven = static_cast<std::uint32_t>(rep.shared_accesses.size());
    for (const auto& b : rep.barriers) {
        if (b.uniform_control_flow) {
            if (b.op == spv::OpControlBarrier) {
                ++sum.barriers_proven;
            } else {
                ++sum.collectives_proven;
            }
        }
    }

    // For small single-function kernels (Brass JIT kernels and synthetic negative
    // test kernels) and known elementwise/cast/fill ABI kernels, also run symbolic
    // lockstep execution prover (verify_spirv_memory_safety + verify_spirv_barrier_uniformity).
    const auto specs = buffer_specs_for_shader(name);
    const bool small_kernel = (rep.num_functions == 1 && rep.num_blocks <= 6 && rep.num_instructions <= 180);
    if (small_kernel || !specs.empty()) {
        brass::verify::GpuProverOptions opts;
        opts.local_size_x = std::max(1u, rep.local_size[0]);
        opts.local_size_y = std::max(1u, rep.local_size[1]);
        opts.local_size_z = std::max(1u, rep.local_size[2]);
        opts.loop_unroll_bound = 2;
        opts.max_sat_conflicts = 4000;
        opts.check_global_bounds = !specs.empty();
        opts.buffers = specs;
        opts.param_constraints = param_constraints_for_shader(name);

        brass::verify::GpuMemorySafetyResult mem_res =
            brass::verify::verify_spirv_memory_safety(mod, opts);

        sum.shared_accesses_proven = std::max(sum.shared_accesses_proven, mem_res.shared_accesses_proven);
        sum.global_accesses_proven = std::max(sum.global_accesses_proven, mem_res.global_accesses_proven);

        if (specs.empty()) {
            sum.memory_safe = sum.memory_safe && mem_res.is_safe();
            if (!mem_res.is_safe() && sum.diagnostic.empty()) {
                sum.diagnostic = mem_res.diagnostic;
            }
        } else {
            const bool bounded_loop_ok =
                (mem_res.status == brass::verify::EquivStatus::Unknown &&
                 mem_res.violations.empty() &&
                 mem_res.global_accesses_proven > 0);
            const bool unsupported_ok =
                (mem_res.status == brass::verify::EquivStatus::Unsupported &&
                 mem_res.violations.empty());
            const bool mem_ok =
                !mem_res.is_counterexample() &&
                mem_res.violations.empty() &&
                (mem_res.is_safe() || bounded_loop_ok || unsupported_ok);
            sum.memory_safe = sum.memory_safe && mem_ok;
            if (!mem_ok && sum.diagnostic.empty()) {
                sum.diagnostic = mem_res.diagnostic.empty()
                    ? "Global buffer bounds verification failed"
                    : mem_res.diagnostic;
            }
        }

        if (small_kernel) {
            brass::verify::BarrierUniformityResult bar_res =
                brass::verify::verify_spirv_barrier_uniformity(mod, opts);
            sum.barriers_uniform = sum.barriers_uniform && bar_res.is_uniform();
            sum.barriers_proven = std::max(sum.barriers_proven, bar_res.barriers_proven);
            sum.collectives_proven = std::max(sum.collectives_proven, bar_res.collectives_proven);
            if (!bar_res.is_uniform() && sum.diagnostic.empty()) {
                sum.diagnostic = bar_res.diagnostic;
            }
        }
    }

    if (!sum.structural_ok) {
        sum.diagnostic = struct_err;
    } else if (!rep.hazards.empty()) {
        sum.diagnostic = rep.hazards.front().message;
    } else if (sum.diagnostic.empty()) {
        sum.diagnostic = "OK";
    }
    return sum;
}

namespace {

void audit_jit_spirv_and_ptx_kernels(BackwardShaderAuditReport& report) {
    static float dummy[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // 1. Elementwise JIT kernel with SiLU, GELU, Tanh, Sigmoid, Clamp
    {
        jit::TraceDAG dag;
        const int x = dag.add_node(jit::TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, Dtype::FP32, 64, 256);
        const int dy = dag.add_node(jit::TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, Dtype::FP32, 64, 256);
        const int s = dag.add_node(jit::TraceOpKind::SiLU, {x}, 0.0f, nullptr, Device::CPU, Dtype::FP32, 64, 256);
        const int g = dag.add_node(jit::TraceOpKind::GELU, {s}, 0.0f, nullptr, Device::CPU, Dtype::FP32, 64, 256);
        const int cl = dag.add_node(jit::TraceOpKind::Clamp, {g}, -20.0f, nullptr, Device::CPU, Dtype::FP32, 64, 256, 20.0f);
        const int out = dag.add_node(jit::TraceOpKind::Mul, {cl, dy}, 0.0f, nullptr, Device::CPU, Dtype::FP32, 64, 256);
        dag.nodes()[static_cast<std::size_t>(out)].is_live_at_end = true;

        jit::plan::ElementwisePlan ew;
        if (jit::plan::build_plan(dag, ew)) {
            jit::spirv::Kernels k = jit::spirv::emit_elementwise(dag, ew);
            if (k.vec.has_value() && !k.vec->words.empty()) {
                report.shaders.push_back(audit_spirv_binary_words(
                    "jit_spirv_elementwise_vec", k.vec->words.data(), k.vec->words.size()));
            }
            if (!k.scalar.words.empty()) {
                report.shaders.push_back(audit_spirv_binary_words(
                    "jit_spirv_elementwise_scalar", k.scalar.words.data(), k.scalar.words.size()));
            }
        }
    }

    // 2. RowNorm JIT kernels (RMSNorm & LayerNorm, multi-warp D=4096 and single-warp D=64)
    const int cols_list[] = {64, 4096};
    for (int cols : cols_list) {
        for (int is_ln = 0; is_ln <= 1; ++is_ln) {
            jit::TraceDAG dag;
            const int x = dag.add_node(jit::TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, Dtype::FP32, 16, cols);
            const int g = dag.add_node(jit::TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, Dtype::FP32, 1, cols);
            std::vector<int> ins = {x, g};
            if (is_ln) {
                ins.push_back(dag.add_node(jit::TraceOpKind::Input, {}, 0.0f, dummy, Device::CPU, Dtype::FP32, 1, cols));
            }
            const int n = dag.add_node(is_ln ? jit::TraceOpKind::LayerNorm : jit::TraceOpKind::RMSNorm,
                                       ins, 1e-5f, nullptr, Device::CPU, Dtype::FP32, 16, cols);
            dag.nodes()[static_cast<std::size_t>(n)].is_live_at_end = true;

            jit::plan::RowNormPlan rn;
            if (jit::plan::build_row_norm_plan(dag, rn)) {
                jit::spirv::Kernels k = jit::spirv::emit_row_norm(dag, rn);
                const std::string prefix = std::string(is_ln ? "jit_spirv_layernorm_" : "jit_spirv_rmsnorm_") +
                                           std::to_string(cols);
                if (k.vec.has_value() && !k.vec->words.empty()) {
                    report.shaders.push_back(audit_spirv_binary_words(
                        prefix + "_vec", k.vec->words.data(), k.vec->words.size()));
                }
                if (!k.scalar.words.empty()) {
                    report.shaders.push_back(audit_spirv_binary_words(
                        prefix + "_scalar", k.scalar.words.data(), k.scalar.words.size()));
                }
            }
        }
    }

    // 3. PTX lowered backward reduction kernel audited via verify_ptx_memory_safety & verify_ptx_barrier_uniformity
    {
        brass::Module mod("ptx_bwd_audit");
        std::vector<brass::Type> params = {brass::Type::ptr(), brass::Type::ptr()};
        brass::Function* fn = mod.create_function("ptx_bwd_reduce", brass::Type::void_type(),
                                                  brass::Span<const brass::Type>(params));
        brass::codegen::KernelBuilder kb(mod, fn);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        brass::Value* out_ptr = b.add_block_param(entry, brass::Type::ptr());
        brass::Value* in_ptr  = b.add_block_param(entry, brass::Type::ptr());

        brass::Value* smem = kb.shared_alloc_f32(32);
        brass::Value* tid  = kb.tid_x();
        brass::Value* lane = b.build_and(tid, b.build_iconst_i32(31));
        brass::Value* warp = b.build_lshr(tid, b.build_iconst_i32(5));
        brass::Value* off  = b.build_shl(b.build_zext_i64(tid), b.build_iconst_i64(2));
        brass::Value* v    = b.build_load(brass::Type::f32(), b.build_add(in_ptr, off), 0);
        for (uint32_t m = 16; m >= 1; m >>= 1) {
            v = b.build_add(v, kb.shfl_bfly_f32(v, m));
        }
        kb.if_then(b.build_eq(lane, b.build_iconst_i32(0)), [&] {
            kb.shared_store_f32_indexed(smem, warp, v);
        });
        kb.sync();
        brass::Value* red = kb.shared_load_f32_indexed(smem, b.build_and(warp, b.build_iconst_i32(7)));
        b.build_store(brass::Type::f32(), b.build_add(out_ptr, off), 0, red);
        b.build_ret_void();

        brass::ptx::PtxISel isel;
        brass::ptx::Function ptx_fn = isel.lower(*fn);

        brass::verify::GpuProverOptions opts;
        opts.local_size_x = 256;
        opts.buffers.push_back({0, 1024, -1, 4, 16});
        opts.buffers.push_back({1, 1024, -1, 4, 16});
        auto mem_res = brass::verify::verify_ptx_memory_safety(ptx_fn, opts);
        auto bar_res = brass::verify::verify_ptx_barrier_uniformity(ptx_fn, opts);

        ShaderAuditSummary ptx_sum;
        ptx_sum.shader_name = "ptx_bwd_reduce_sm80";
        ptx_sum.parse_ok = true;
        ptx_sum.structural_ok = true;
        ptx_sum.memory_safe = mem_res.is_safe();
        ptx_sum.barriers_uniform = bar_res.is_uniform();
        ptx_sum.div_sqrt_safe = true;
        ptx_sum.num_functions = 1;
        ptx_sum.num_blocks = static_cast<std::uint32_t>(ptx_fn.blocks.size());
        ptx_sum.num_workgroup_arrays = static_cast<std::uint32_t>(ptx_fn.shared.size());
        ptx_sum.shared_accesses_proven = mem_res.shared_accesses_proven;
        ptx_sum.global_accesses_proven = mem_res.global_accesses_proven;
        ptx_sum.barriers_proven = bar_res.barriers_proven;
        ptx_sum.collectives_proven = bar_res.collectives_proven;
        ptx_sum.diagnostic = (ptx_sum.memory_safe && ptx_sum.barriers_uniform) ? "OK" : mem_res.diagnostic;
        report.shaders.push_back(std::move(ptx_sum));
    }

    // 3b. Single-source brass-MIR Wave 1 & Wave 2 kernels
    {
        brass::target::SpirvOptions spv_opts;
        spv_opts.stage = brass::target::SpirvStage::Compute;
        spv_opts.local_size_x = kernels::kEwBlock;
        spv_opts.local_size_spec_constants = true;
        auto audit_mir = [&](const char* name, const std::function<brass::Function*(brass::Module&)>& make) {
            brass::Module mir_mod(name);
            brass::Function* fn = make(mir_mod);
            brass::target::SpirvKernel sk = brass::target::SpirvTarget::compile(*fn, spv_opts);
            report.shaders.push_back(audit_spirv_binary_words(name, sk.words.data(), sk.words.size()));
        };
        for (const auto& [name, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"act_bwd_f32", Dtype::FP32}, {"act_bwd_f16", Dtype::FP16}, {"act_bwd_bf16", Dtype::BF16}}) {
            audit_mir(name, [&](brass::Module& m) { return kernels::build_act_bwd_mir(m, GOP_RELU, dt, 1); });
            (void)dt;
        }
        for (const auto& [name, op, dt] : std::vector<std::tuple<const char*, std::uint32_t, Dtype>>{
                 {"glu_f32",      GLU_SWIGLU,     Dtype::FP32},
                 {"glu_f16",      GLU_SWIGLU,     Dtype::FP16},
                 {"glu_bf16",     GLU_SWIGLU,     Dtype::BF16},
                 {"glu_bwd_f32",  GLU_SWIGLU_BWD, Dtype::FP32},
                 {"glu_bwd_f16",  GLU_SWIGLU_BWD, Dtype::FP16},
                 {"glu_bwd_bf16", GLU_SWIGLU_BWD, Dtype::BF16}}) {
            audit_mir(name, [op, dt](brass::Module& m) { return kernels::build_glu_mir(m, op, dt, dt, 1); });
        }
        for (const auto& [prefix, kind] : std::vector<std::pair<const char*, int>>{
                 {"rowvec", 0}, {"sum_rows", 1}, {"argmax_rows", 2}, {"sum_cols", 3}, {"rope", 4}}) {
            for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                     {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
                const std::string name = std::string(prefix) + sfx;
                audit_mir(name.c_str(), [kind, dt](brass::Module& m) -> brass::Function* {
                    switch (kind) {
                        case 0: return kernels::build_rowvec_mir(m, ROWVEC_MODULATE, dt, 1);
                        case 1: return kernels::build_sum_rows_mir(m, dt, 1);
                        case 2: return kernels::build_argmax_rows_mir(m, dt, false, 1);
                        case 3: return kernels::build_sum_cols_mir(m, dt, 1);
                        default: return kernels::build_rope_mir(m, ROPE_THETA, true, dt);
                    }
                });
            }
        }
        const kernels::NormMirFlags all_flags{true, true, true, true, true};
        for (const auto& [name, mode, dtx, dty, dtg] :
             std::vector<std::tuple<const char*, std::uint32_t, Dtype, Dtype, Dtype>>{
                 {"norm_f32",      NORM_LN,      Dtype::FP32, Dtype::FP32, Dtype::FP32},
                 {"norm_f16",      NORM_LN,      Dtype::FP16, Dtype::FP16, Dtype::FP16},
                 {"norm_bf16",     NORM_LN,      Dtype::BF16, Dtype::BF16, Dtype::BF16},
                 {"norm_f16_g32",  NORM_LN,      Dtype::FP16, Dtype::FP16, Dtype::FP32},
                 {"norm_bf16_g32", NORM_LN,      Dtype::BF16, Dtype::BF16, Dtype::FP32},
                 {"norm_f32_y16",  NORM_LN,      Dtype::FP32, Dtype::FP16, Dtype::FP32},
                 {"norm_bwd_f32",  NORM_LN_BWD,  Dtype::FP32, Dtype::FP32, Dtype::FP32},
                 {"norm_bwd_f16",  NORM_LN_BWD,  Dtype::FP16, Dtype::FP16, Dtype::FP16},
                 {"norm_bwd_bf16", NORM_LN_BWD,  Dtype::BF16, Dtype::BF16, Dtype::BF16},
                 {"norm_col_f32",  NORM_COL_LN,  Dtype::FP32, Dtype::FP32, Dtype::FP32},
                 {"norm_col_f16",  NORM_COL_LN,  Dtype::FP16, Dtype::FP16, Dtype::FP16},
                 {"norm_col_bf16", NORM_COL_LN,  Dtype::BF16, Dtype::BF16, Dtype::BF16}}) {
            audit_mir(name, [=](brass::Module& m) {
                return kernels::build_norm_mir(m, mode, dtx, dty, dtg, all_flags, 1);
            });
        }
        // Wave 3 kernels
        {
            brass::target::SpirvOptions fill_opts = spv_opts;
            fill_opts.local_size_x = kernels::kFillBlock;
            brass::Module mir_mod("fill_bytes");
            brass::Function* fn = kernels::build_fill_bytes_mir(mir_mod);
            brass::target::SpirvKernel sk = brass::target::SpirvTarget::compile(*fn, fill_opts);
            report.shaders.push_back(audit_spirv_binary_words("fill_bytes", sk.words.data(), sk.words.size()));
        }
        for (const auto& [name, eb] : std::vector<std::pair<const char*, int>>{
                 {"copy2d_b1", 1}, {"copy2d_b2", 2}, {"copy2d_b4", 4}}) {
            audit_mir(name, [eb](brass::Module& m) { return kernels::build_copy2d_mir(m, eb, 1); });
        }
        for (const auto& [name, eb] : std::vector<std::pair<const char*, int>>{
                 {"transpose_b2", 2}, {"transpose_b4", 4}}) {
            audit_mir(name, [eb](brass::Module& m) { return kernels::build_transpose_mir(m, eb); });
        }
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string sa_name = std::string("scatter_add") + sfx;
            audit_mir(sa_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_scatter_add_mir(m, SA_SUM, dt);
            });
            const std::string gn_name = std::string("gnorm") + sfx;
            audit_mir(gn_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_gnorm_mir(m, GN_BWD_DX, dt, false, 1);
            });
            const std::string gc_name = std::string("gnorm_ch") + sfx;
            audit_mir(gc_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_gnorm_mir(m, GN_BWD_CH, dt, false, 1);
            });
            const std::string gs_name = std::string("gnorm_stats") + sfx;
            audit_mir(gs_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_gnorm_mir(m, GN_STATS, dt, false, 1);
            });
        }
        audit_mir("scatter_local", [](brass::Module& m) {
            return kernels::build_scatter_add_mir(m, SA_LOCAL, Dtype::FP32);
        });
        audit_mir("bnorm", [](brass::Module& m) {
            return kernels::build_bnorm_mir(m, BN_TRAIN_FWD);
        });
        audit_mir("bnorm_bwd", [](brass::Module& m) {
            return kernels::build_bnorm_mir(m, BN_TRAIN_BWD);
        });
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string rb_name = std::string("resample_bwd") + sfx;
            audit_mir(rb_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_resample_bwd_mir(m, RB_INTERP, dt, 1);
            });
            const std::string rs_name = std::string("resample") + sfx;
            audit_mir(rs_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_resample_mir(m, RS_INTERP, dt, false, 1);
            });
            const std::string smp_name = std::string("sampler") + sfx;
            audit_mir(smp_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_sampler_mir(m, SMP_DDIM, dt, dt, 1);
            });
        }
        audit_mir("remap_b2", [](brass::Module& m) { return kernels::build_remap_mir(m, RM_PAD, 2); });
        audit_mir("remap_b4", [](brass::Module& m) { return kernels::build_remap_mir(m, RM_PAD, 4); });
        audit_mir("philox", [](brass::Module& m) { return kernels::build_philox_mir(m, RNG_NORMAL); });
        audit_mir("lstm", [](brass::Module& m) { return kernels::build_lstm_mir(m, LS_BWD, false, true); });
        audit_mir("lstm_fwd", [](brass::Module& m) { return kernels::build_lstm_mir(m, LS_FWD, true, true); });
        audit_mir("delta_rule", [](brass::Module& m) { return kernels::build_delta_rule_mir(m, 32u, 4u); });
        audit_mir("xent", [](brass::Module& m) { return kernels::build_xent_mir(m, 0u, true, true); });
        audit_mir("xent_bce", [](brass::Module& m) { return kernels::build_xent_mir(m, 1u, true, false); });
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string tk_name = std::string("topk_seg") + sfx;
            audit_mir(tk_name.c_str(), [dt](brass::Module& m) { return kernels::build_topk_mir(m, 0u, dt); });
            const std::string mi_name = std::string("misc") + sfx;
            audit_mir(mi_name.c_str(), [dt](brass::Module& m) { return kernels::build_misc_mir(m, MI_BIAS_ACT_DB, dt, true); });
            const std::string up_name = std::string("vision_upfirdn") + sfx;
            audit_mir(up_name.c_str(), [dt](brass::Module& m) { return kernels::build_vision_mir(m, MI_UPFIRDN, dt, false, false); });
        }
        audit_mir("misc_moments_f32", [](brass::Module& m) { return kernels::build_misc_mir(m, MI_MOMENTS, Dtype::FP32, false); });
        audit_mir("misc_count_above_f32", [](brass::Module& m) { return kernels::build_misc_mir(m, MI_COUNT_ABOVE, Dtype::FP32, false); });
        audit_mir("vision_modw_bwd_f32", [](brass::Module& m) { return kernels::build_vision_mir(m, MI_MODW_BWD, Dtype::FP32, false, false); });
        audit_mir("vision_col2im2d_f32", [](brass::Module& m) { return kernels::build_vision_mir(m, MI_COL2IM2D, Dtype::FP32, true, false); });
        audit_mir("vision_deform_col_f32", [](brass::Module& m) { return kernels::build_vision_mir(m, MI_DEFORM_COL, Dtype::FP32, true, true); });
        audit_mir("select_sample", [](brass::Module& m) { return kernels::build_select_sample_mir(m, false); });
        audit_mir("select_md", [](brass::Module& m) { return kernels::build_select_md_mir(m, true, true, true); });
        audit_mir("select_commit", [](brass::Module& m) { return kernels::build_select_commit_mir(m); });
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string sn_name = std::string("audio_snake_bwd") + sfx;
            audit_mir(sn_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_audio_mir(m, AU_SNAKE_BWD, dt, 1u, false);
            });
        }
        audit_mir("audio_convt_bwd_w_f32", [](brass::Module& m) {
            return kernels::build_audio_mir(m, AU_CONVT_BWD_W, Dtype::FP32, 0u, false);
        });
        audit_mir("audio_vq", [](brass::Module& m) {
            return kernels::build_audio_mir(m, AU_VQ, Dtype::FP32, 0u, false);
        });
        audit_mir("audio_causal_f32", [](brass::Module& m) {
            return kernels::build_audio_mir(m, AU_CAUSAL, Dtype::FP32, 1u, false);
        });
        audit_mir("spectral_basis", [](brass::Module& m) {
            return kernels::build_spectral_mir(m, SP_BASIS, BASIS_R2C, true, true);
        });
        audit_mir("spectral_ola", [](brass::Module& m) {
            return kernels::build_spectral_mir(m, SP_OLA, 0u, false, false);
        });
        audit_mir("spectral_stft_adj", [](brass::Module& m) {
            return kernels::build_spectral_mir(m, SP_STFT_ADJ, 1u, false, false);
        });
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string ax_name = std::string("attn_aux") + sfx;
            audit_mir(ax_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_attn_aux_mir(m, 0u, dt, true);
            });
            const std::string cd_name = std::string("conv_direct") + sfx;
            audit_mir(cd_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_conv_direct_mir(m, 0u, dt, true);
            });
            const std::string fc_name = std::string("fa_combine") + sfx;
            audit_mir(fc_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_fa_combine_mir(m, dt);
            });
            const std::string sm_name = std::string("attn_softmax") + sfx;
            audit_mir(sm_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_attn_softmax_mir(m, dt, 1u, true, true, false);
            });
        }
        audit_mir("attn_bwd", [](brass::Module& m) {
            return kernels::build_attn_bwd_mir(m, true, true);
        });
        audit_mir("gemm_rowscale_f32", [](brass::Module& m) {
            return kernels::build_gemm_rowscale_mir(m, Dtype::FP32, false, true);
        });
        audit_mir("gemm_rowscale_bf16", [](brass::Module& m) {
            return kernels::build_gemm_rowscale_mir(m, Dtype::BF16, false, true);
        });
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string wq_name = std::string("wq_rows") + sfx;
            audit_mir(wq_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_wq_rows_mir(m, 0u, dt, true);
            });
        }
        audit_mir("dequant", [](brass::Module& m) {
            return kernels::build_dequant_mir(m, QF_Q4K, true);
        });
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string cs_name = std::string("conv_simt") + sfx;
            audit_mir(cs_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_conv_simt_mir(m, dt, false, true, true, 3, 3, 1, 1, 1, 1, 1, 1);
            });
            const std::string gs_name = std::string("gemm_simt") + sfx;
            audit_mir(gs_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_gemm_simt_mir(m, dt, dt, dt, dt, false, 1u, 1u, false, false, true);
            });
            const std::string gv_name = std::string("gemv") + sfx;
            audit_mir(gv_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_gemv_mir(m, dt, dt, dt, 1u, 1u, 1u, true, true);
            });
        }
        for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                 {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
            const std::string gq_name = std::string("gemv_q") + sfx;
            audit_mir(gq_name.c_str(), [dt](brass::Module& m) {
                return kernels::build_gemv_q_mir(m, dt, QF_Q4K, 1u, 64u, true, true, 1u, 2u, 0u, 0u);
            });
        }
        {
            brass::target::SpirvOptions fa_opts = spv_opts;
            fa_opts.local_size_x = kernels::kFaRowsBlock;
            for (const auto& [sfx, dt] : std::vector<std::pair<const char*, Dtype>>{
                     {"_f32", Dtype::FP32}, {"_f16", Dtype::FP16}, {"_bf16", Dtype::BF16}}) {
                const std::string fa_name = std::string("fa_rows") + sfx;
                brass::Module mir_mod(fa_name.c_str());
                brass::Function* fn = kernels::build_fa_rows_mir(
                    mir_mod, dt, 64u, 2u, 0u, true, true, true, true, true);
                brass::target::SpirvKernel sk = brass::target::SpirvTarget::compile(*fn, fa_opts);
                report.shaders.push_back(audit_spirv_binary_words(fa_name.c_str(), sk.words.data(), sk.words.size()));
                const std::string fb_name = std::string("fa_bwd") + sfx;
                brass::Module fb_mod(fb_name.c_str());
                brass::Function* fbfn = kernels::build_fa_bwd_mir(
                    fb_mod, dt, 128u, 64u, 64u, 32u, 16u, 0u, true, 1u);
                brass::target::SpirvKernel fbsk = brass::target::SpirvTarget::compile(*fbfn, fa_opts);
                report.shaders.push_back(audit_spirv_binary_words(fb_name.c_str(), fbsk.words.data(), fbsk.words.size()));
            }
        }
    }

    // 4. Synthetic negative test if not already covered by ShaderId::test_shared_overflow
    if (!report.negative_overflow_shader_caught) {
        brass::Module mod("neg_shared_oob");
        std::vector<brass::Type> params = {brass::Type::ptr()};
        brass::Function* fn = mod.create_function("neg_shared_oob", brass::Type::void_type(),
                                                  brass::Span<const brass::Type>(params));
        brass::codegen::KernelBuilder kb(mod, fn);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        (void)b.add_block_param(entry, brass::Type::ptr());
        brass::Value* smem = kb.shared_alloc_f32(16); // length 16, but tid_x in [0, 256)
        kb.shared_store_f32_indexed(smem, kb.tid_x(), b.build_fconst_f32(1.0f));
        b.build_ret_void();

        brass::target::SpirvOptions spv_opts;
        spv_opts.local_size_x = 256;
        auto sk = brass::target::SpirvTarget::compile(*fn, spv_opts);
        ShaderAuditSummary neg = audit_spirv_binary_words(
            "synthetic_test_shared_overflow", sk.words.data(), sk.words.size());
        if (!neg.memory_safe) {
            report.negative_overflow_shader_caught = true;
            report.negative_test_detail = neg.diagnostic;
        }
    }
}

} // namespace

BackwardShaderAuditReport run_backward_shader_audit() {
    BackwardShaderAuditReport report;

#if defined(BROTENSOR_HAS_VULKAN) || defined(BROTENSOR_HAS_SHADER_TABLE)
    const auto count = static_cast<std::uint32_t>(detail::vulkan::ShaderId::kCount);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto id = static_cast<detail::vulkan::ShaderId>(i);
        const auto& blob = detail::vulkan::shader_blob(id);
        ShaderAuditSummary s = audit_spirv_binary_words(blob.name, blob.words, blob.nwords);
        if (std::string(blob.name) == "test_shared_overflow") {
            // Intentional negative test shader: must be flagged as a shared-memory OOB hazard!
            report.negative_overflow_shader_caught = !s.memory_safe;
            report.negative_test_detail = s.diagnostic;
        } else {
            report.shaders.push_back(std::move(s));
        }
    }
#endif

    audit_jit_spirv_and_ptx_kernels(report);
    return report;
}

#endif // BROTENSOR_HAS_BRASS_JIT

} // namespace brotensor::verify
