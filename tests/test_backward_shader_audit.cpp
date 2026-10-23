// Issue #139: Lifter Audit of Vulkan .comp & PTX Backward Shaders
// (Shared-Memory Bounds & Barrier Uniformity).
//
// Exercises:
//   1. Full static lifter audit of compiled Vulkan SPIR-V shaders (all 137
//      production shaders when built with Vulkan + all runtime-emitted Brass
//      SPIR-V & PTX kernels on every build):
//        - brass::spirv::verify_spirv_structural
//        - brass::verify::verify_spirv_memory_safety (BVSolver workgroup bounds)
//        - brass::verify::verify_spirv_barrier_uniformity (uniform barriers & subgroup ops)
//        - brass::verify::prove_spirv_shader_safety (div/sqrt hazard audit)
//   2. Negative tests:
//        - Intentional out-of-bounds workgroup shared-memory shader
//          (`test_shared_overflow`) is caught (`memory_safe == false`).
//        - Intentional thread-divergent barrier kernel (`if (tid == 0) barrier()`)
//          is caught (`barriers_uniform == false`).

#include <brotensor/backward_verify.h>

#if BROTENSOR_HAS_BRASS_JIT
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/spirv_target.hpp>
#endif

#include <cstdio>
#include <string>
#include <unordered_set>

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static void test_backward_shader_audit_suite() {
    auto report = brotensor::verify::run_backward_shader_audit();
    CHECK(!report.shaders.empty());
    CHECK(report.negative_overflow_shader_caught);

    std::uint32_t total_insts = 0;
    std::uint32_t total_shared_arrays = 0;
    std::uint32_t total_shared_accesses = 0;
    std::uint32_t total_barriers = 0;
    std::uint32_t total_collectives = 0;
    std::unordered_set<std::string> seen;

    for (const auto& s : report.shaders) {
        seen.insert(s.shader_name);
        total_insts += s.num_instructions;
        total_shared_arrays += s.num_workgroup_arrays;
        total_shared_accesses += s.shared_accesses_proven;
        total_barriers += s.barriers_proven;
        total_collectives += s.collectives_proven;

        if (!s.ok()) {
            std::printf("  [SHADER FAIL] %-36s : parse=%d struct=%d mem=%d bar=%d div=%d (%s)\n",
                        s.shader_name.c_str(),
                        s.parse_ok ? 1 : 0,
                        s.structural_ok ? 1 : 0,
                        s.memory_safe ? 1 : 0,
                        s.barriers_uniform ? 1 : 0,
                        s.div_sqrt_safe ? 1 : 0,
                        s.diagnostic.c_str());
        }
        CHECK(s.ok());
    }

    std::printf("  Audited %zu shaders (%u SPIR-V/PTX instructions, %u workgroup arrays, "
                "%u shared accesses proven, %u uniform barriers, %u uniform collectives)\n",
                report.shaders.size(),
                total_insts,
                total_shared_arrays,
                total_shared_accesses,
                total_barriers,
                total_collectives);
    std::printf("  Negative shared-overflow shader caught: %s\n",
                report.negative_test_detail.c_str());

    // Always-present JIT SPIR-V & PTX kernels
    CHECK(seen.count("jit_spirv_elementwise_vec") == 1);
    CHECK(seen.count("jit_spirv_rmsnorm_64_vec") == 1);
    CHECK(seen.count("jit_spirv_layernorm_4096_vec") == 1);
    CHECK(seen.count("ptx_bwd_reduce_sm80") == 1);

#if defined(BROTENSOR_HAS_VULKAN)
    // Production Vulkan backward shaders from shader_table.h
    const char* required_vk_shaders[] = {
        "act_bwd_f32",
        "act_bwd_f16",
        "act_bwd_bf16",
        "glu_f32",
        "glu_f16",
        "glu_bf16",
        "norm_f32",
        "norm_f16",
        "norm_bf16",
        "gnorm_f32",
        "bnorm",
        "attn_bwd",
        "fa_bwd_f32",
        "fa_bwd_f16",
        "fa_bwd_bf16",
        "fa_bwd_cm_f16",
        "conv_simt_f32",
        "conv_direct_f32",
        "resample_bwd_f32",
        "rope_f32",
        "misc_f32",
        "scatter_add_f32",
        "lstm",
        "delta_rule",
        "xent"
    };
    for (const char* name : required_vk_shaders) {
        CHECK(seen.count(name) == 1);
    }
#endif

    CHECK(report.all_passed());
}

static void test_negative_divergent_barrier_caught() {
#if BROTENSOR_HAS_BRASS_JIT
    // Construct a SPIR-V compute kernel with a barrier inside a thread-dependent
    // branch `if (tid_x == 0) { barrier(); }` and verify the lifter rejects it.
    brass::Module mod("neg_divergent_barrier");
    std::vector<brass::Type> params = {brass::Type::ptr()};
    brass::Function* fn = mod.create_function(
        "neg_divergent_barrier", brass::Type::void_type(),
        brass::Span<const brass::Type>(params));
    brass::codegen::KernelBuilder kb(mod, fn);
    brass::Builder& b = kb.builder();
    brass::BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    (void)b.add_block_param(entry, brass::Type::ptr());

    brass::Value* tid = kb.tid_x();
    kb.if_then(b.build_eq(tid, b.build_iconst_i32(0)), [&] {
        kb.sync();
    });
    b.build_ret_void();

    brass::target::SpirvOptions spv_opts;
    spv_opts.local_size_x = 64;
    auto sk = brass::target::SpirvTarget::compile(*fn, spv_opts);

    auto summary = brotensor::verify::audit_spirv_binary_words(
        "neg_divergent_barrier", sk.words.data(), sk.words.size());
    std::printf("  Negative divergent-barrier shader caught: barriers_uniform=%d (%s)\n",
                summary.barriers_uniform ? 1 : 0,
                summary.diagnostic.c_str());
    CHECK(summary.parse_ok);
    CHECK(summary.structural_ok);
    CHECK(!summary.barriers_uniform);
    CHECK(!summary.ok());
#endif
}

int main() {
    std::printf("=== Issue #139: Vulkan SPIR-V & PTX Backward Shader Lifter Audit ===\n");
    test_backward_shader_audit_suite();
    test_negative_divergent_barrier_caught();

    if (g_failures == 0) {
        std::printf("ALL PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
