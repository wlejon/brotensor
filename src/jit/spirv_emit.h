#pragma once

// MIR emitters for the trace JIT's Vulkan backend.
//
// The same two kernel shapes the CUDA compiler emits as PTX text
// (ptx_emit.h), built here as brass MIR through KernelBuilder and lowered by
// brass's SPIR-V target (target::SpirvTarget::compile):
//
//   elementwise — a grid-stride map over the trace's dominant shape, `vec`
//     elements per thread so every full access moves 16 bytes (f32x4, or
//     eight halves as an i32x4); (1, cols) rows and (1, 1) scalars are
//     addressed by column, never materialised. BF16/FP16 are unpacked to FP32
//     on load and rounded back on store.
//
//   row-norm — RMSNorm / LayerNorm plus the elementwise chain around it, as
//     row groups of `tpr` threads (plan::row_norm_geometry), subgroup
//     shuffles plus one shared round for the reduction.
//
// Each emitter returns one kernel per entry: a vector entry (absent when the
// plan's `vec` is 1) and a scalar entry, with the same parameter list, so the
// launcher can pick at bind time from the pointers' alignment. Parameters are
// the plan's outputs then its inputs (all `ptr`), then for the elementwise
// kernel the slot count `n` (i32). Shapes, eps and the row geometry are baked
// in as constants: the trace signature already carries them.

#include "trace_dag.h"
#include "trace_plan.h"

#include <brass/target/spirv_target.hpp>

#include <cstdint>
#include <optional>

namespace brotensor::jit::spirv {

struct Kernels {
    std::optional<brass::target::SpirvKernel> vec;   // `lanes` = plan vec
    brass::target::SpirvKernel scalar;
};

// Workgroup size of the elementwise kernel.
inline constexpr std::uint32_t kEwBlock = 256;

// Throws std::runtime_error for an op the emitters have no form for (callers
// check ops_supported() first).
Kernels emit_elementwise(const TraceDAG& dag, const plan::ElementwisePlan& plan);
Kernels emit_row_norm(const TraceDAG& dag, const plan::RowNormPlan& plan);

// True when every non-input node of `dag` is an op the elementwise chain can
// evaluate (the norms excepted: the row-norm emitter handles exactly one).
bool ops_supported(const TraceDAG& dag);

inline constexpr const char* kEntryVec = "trace_ew_vec";
inline constexpr const char* kEntryScalar = "trace_ew_scalar";
inline constexpr const char* kEntryRowNorm = "trace_row_norm_vec";
inline constexpr const char* kEntryRowNormScalar = "trace_row_norm_scalar";

}  // namespace brotensor::jit::spirv
