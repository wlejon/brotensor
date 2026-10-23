#pragma once

// Formal verification & static audit entrypoints for brotensor backward passes
// (Issues #138, #139, #140):
//   1. Adjoint index transpose duality & [0, numel) bounds proofs via BVSolver
//   2. SPIR-V & PTX compute shader lifter audit (structural, shared-memory bounds,
//      barrier/subgroup uniformity, division/sqrt safety)
//   3. IEEE-754 gradient NaN-freedom proofs & boundary clamping verification

#include <brotensor/tensor.h>

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::verify {

struct ProofEntry {
    std::string name;
    bool proven = false;
    std::string detail;
};

struct BackwardProofSuiteReport {
    std::vector<ProofEntry> entries;

    bool all_passed() const noexcept {
        if (entries.empty()) return false;
        for (const auto& e : entries) {
            if (!e.proven) return false;
        }
        return true;
    }

    std::size_t passed_count() const noexcept {
        std::size_t n = 0;
        for (const auto& e : entries) {
            if (e.proven) ++n;
        }
        return n;
    }
};

// ─── Issue #138: Adjoint Index Transpose Duality & OOB Bounds Proofs ─────────

BackwardProofSuiteReport prove_conv2d_adjoint_duality();
BackwardProofSuiteReport prove_upfirdn2d_adjoint_duality();
BackwardProofSuiteReport refute_upfirdn2d_pre_995ac3c_bug();
BackwardProofSuiteReport prove_resample2d_adjoint_duality();
BackwardProofSuiteReport prove_rope_adjoint_duality();
BackwardProofSuiteReport prove_flash_attention_mask_adjoint_duality();
BackwardProofSuiteReport run_all_backward_index_proofs();

// ─── Issue #139: Vulkan SPIR-V & PTX Backward Shader Lifter Audit ───────────

struct ShaderAuditSummary {
    std::string shader_name;
    bool parse_ok = false;
    bool structural_ok = false;
    bool memory_safe = false;
    bool barriers_uniform = false;
    bool div_sqrt_safe = false;
    std::uint32_t num_functions = 0;
    std::uint32_t num_blocks = 0;
    std::uint32_t num_instructions = 0;
    std::uint32_t num_workgroup_arrays = 0;
    std::uint32_t shared_accesses_proven = 0;
    std::uint32_t barriers_proven = 0;
    std::uint32_t collectives_proven = 0;
    std::string diagnostic;

    bool ok() const noexcept {
        return parse_ok && structural_ok && memory_safe &&
               barriers_uniform && div_sqrt_safe;
    }
};

struct BackwardShaderAuditReport {
    std::vector<ShaderAuditSummary> shaders;
    bool negative_overflow_shader_caught = false;
    std::string negative_test_detail;

    bool all_passed() const noexcept {
        if (shaders.empty() || !negative_overflow_shader_caught) return false;
        for (const auto& s : shaders) {
            if (!s.ok()) return false;
        }
        return true;
    }
};

ShaderAuditSummary audit_spirv_binary_words(
    const std::string& name,
    const std::uint32_t* words,
    std::size_t nwords);

BackwardShaderAuditReport run_backward_shader_audit();

// ─── Issue #140: Gradient NaN-Freedom Proofs & Clamped Sqrt Backward ────────

void sqrt_backward_clamped(
    const Tensor& x,
    const Tensor& dY,
    Tensor& dX,
    float eps = 1e-12f);

BackwardProofSuiteReport prove_activation_backward_nan_freedom();
BackwardProofSuiteReport prove_norm_backward_nan_freedom();
BackwardProofSuiteReport prove_attention_softmax_backward_nan_freedom();
BackwardProofSuiteReport run_all_backward_nan_freedom_proofs();

} // namespace brotensor::verify
