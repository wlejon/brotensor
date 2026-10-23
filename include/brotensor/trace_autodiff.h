#pragma once

// ─── brotensor::jit — Verified Fused Backward Trace JIT ─────────────────────
//
// Reverse-mode automatic differentiation (`TraceDAG::backward()`) for traced
// tensor expressions, with compile-time formal verification via Brass:
//   1. `mirEquivProven`: Formal equivalence proof (`brass::verify::verify_mir_equivalence`)
//      between the unoptimized and optimized backward MIR functions.
//   2. `spirvPtxEquivProven`: Cross-ISA equivalence proof (`brass::verify::verify_spirv_ptx_equivalence`)
//      proving semantic equivalence between the backward MIR, Vulkan SPIR-V, and
//      NVIDIA PTX kernels.
//   3. `nanSafeProven`: Static safety proof (`brass::verify::prove_spirv_shader_safety`)
//      certifying absence of unguarded division-by-zero, domain violations, and
//      NaN/Inf propagation across all division/logarithm/square-root operations.

#include <brotensor/jit/trace.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace brotensor::jit {

class TraceDAG;

struct BackwardTraceVerificationReport {
    bool mirEquivProven = false;
    bool spirvPtxEquivProven = false;
    bool nanSafeProven = false;
    size_t forwardNodeCount = 0;
    size_t rawBackwardNodeCount = 0;
    size_t loweredBackwardNodeCount = 0;
    size_t unoptimizedMirInsts = 0;
    size_t optimizedMirInsts = 0;
    size_t spirvWordCount = 0;
    size_t ptxByteCount = 0;
    std::string diagnostic;

    bool all_proven() const noexcept {
        return mirEquivProven && spirvPtxEquivProven && nanSafeProven;
    }
};

struct VerifiedBackwardTrace {
    BackwardTraceVerificationReport verification;
    size_t num_inputs = 0;
    int rows = 0;
    int cols = 0;
    Device device = Device::CPU;
    Dtype dtype = Dtype::FP32;
    std::string ptx_source;
    std::vector<uint32_t> spirv_words;

    // Internal lowered backward DAG and per-input lowered DAGs (stored via
    // shared_ptr to keep VerifiedBackwardTrace header-independent of internal
    // src/jit headers).
    struct Impl;
    std::shared_ptr<Impl> impl;

    // Evaluates the fused backward trace for all primal inputs given `primal_inputs`
    // and upstream gradient `grad_output` (`dL/dy`).
    // Compiles and executes a single fused backward kernel on the tensors' target
    // backend (CPU AVX2/FMA, Vulkan SPIR-V compute shader, or CUDA PTX kernel).
    std::vector<Tensor> execute(const std::vector<Tensor>& primal_inputs,
                                const Tensor& grad_output) const;

    // Evaluates the single-input backward trace for `input_idx` on the target backend.
    Tensor execute_for_input(int input_idx,
                             const std::vector<Tensor>& primal_inputs,
                             const Tensor& grad_output) const;
};

// Lowering pass that rewrites high-level TraceOpKind nodes (Const, Neg, Abs,
// Exp, Log, Sqrt, Rsqrt) into combinations of core fused elementwise ops
// with provably positive denominators and domain clamping.
TraceDAG canonicalize_trace_dag_for_codegen(const TraceDAG& dag);

// Runs the three compile-time formal verification checks on a backward TraceDAG:
//   - verify_mir_equivalence(raw_mir, optimized_mir)
//   - verify_spirv_ptx_equivalence(gpu_mir)
//   - prove_spirv_shader_safety(spirv_words)
BackwardTraceVerificationReport verify_backward_trace_dag(const TraceDAG& bwd_dag);

// Builds a VerifiedBackwardTrace from an existing forward TraceDAG, running
// symbolic reverse-mode autodiff (`forward_dag.backward()`), lowering, PTX +
// SPIR-V code generation, and compile-time formal verification.
VerifiedBackwardTrace build_verified_backward_trace(const TraceDAG& forward_dag);

// Captures a forward TraceDAG by invoking `fn` with symbolic traced inputs
// shaped like `example_inputs`, differentiates it via `TraceDAG::backward()`,
// and verifies the resulting fused backward kernel across MIR, SPIR-V, and PTX.
VerifiedBackwardTrace trace_and_verify_backward(
    const std::vector<Tensor>& example_inputs,
    const std::function<Tensor(const std::vector<Tensor>&)>& fn);

} // namespace brotensor::jit

namespace brotensor {
using jit::BackwardTraceVerificationReport;
using jit::VerifiedBackwardTrace;
using jit::canonicalize_trace_dag_for_codegen;
using jit::verify_backward_trace_dag;
using jit::build_verified_backward_trace;
using jit::trace_and_verify_backward;
} // namespace brotensor
