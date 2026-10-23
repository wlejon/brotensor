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
#include "jit/spirv_emit_detail.h"
#include "jit/trace_dag.h"
#include "jit/trace_plan.h"
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
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace brotensor::verify {

#if BROTENSOR_HAS_BRASS_JIT

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
    // test kernels), also run symbolic lockstep execution prover
    // (verify_spirv_memory_safety + verify_spirv_barrier_uniformity).
    if (rep.num_functions == 1 && rep.num_blocks <= 6 && rep.num_instructions <= 180) {
        brass::verify::GpuProverOptions opts;
        opts.local_size_x = std::max(1u, rep.local_size[0]);
        opts.local_size_y = std::max(1u, rep.local_size[1]);
        opts.local_size_z = std::max(1u, rep.local_size[2]);
        opts.loop_unroll_bound = 2;
        opts.max_sat_conflicts = 4000;

        brass::verify::GpuMemorySafetyResult mem_res =
            brass::verify::verify_spirv_memory_safety(mod, opts);
        brass::verify::BarrierUniformityResult bar_res =
            brass::verify::verify_spirv_barrier_uniformity(mod, opts);

        sum.memory_safe = sum.memory_safe && mem_res.is_safe();
        sum.barriers_uniform = sum.barriers_uniform && bar_res.is_uniform();
        sum.shared_accesses_proven = std::max(sum.shared_accesses_proven, mem_res.shared_accesses_proven);
        sum.barriers_proven = std::max(sum.barriers_proven, bar_res.barriers_proven);
        sum.collectives_proven = std::max(sum.collectives_proven, bar_res.collectives_proven);

        if (!mem_res.is_safe() && sum.diagnostic.empty()) {
            sum.diagnostic = mem_res.diagnostic;
        } else if (!bar_res.is_uniform() && sum.diagnostic.empty()) {
            sum.diagnostic = bar_res.diagnostic;
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
        ptx_sum.barriers_proven = bar_res.barriers_proven;
        ptx_sum.collectives_proven = bar_res.collectives_proven;
        ptx_sum.diagnostic = (ptx_sum.memory_safe && ptx_sum.barriers_uniform) ? "OK" : mem_res.diagnostic;
        report.shaders.push_back(std::move(ptx_sum));
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
