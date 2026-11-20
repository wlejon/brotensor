#pragma once

// Metal Shading Language (MSL) trace kernel emitter backed by the shared
// Brass GPU MIR builders and MslTarget.

#include "trace_dag.h"
#include "trace_plan.h"

#include <brass/target/msl_target.hpp>

#include <cstdint>
#include <optional>

namespace brotensor::jit::msl {

using plan::ElementwisePlan;
using plan::RowNormPlan;

struct Kernels {
    std::optional<brass::target::MslKernel> vec;
    brass::target::MslKernel scalar;
};

inline constexpr std::uint32_t kEwBlock = 256;

Kernels emit_elementwise(const TraceDAG& dag, const plan::ElementwisePlan& plan);
Kernels emit_row_norm(const TraceDAG& dag, const plan::RowNormPlan& plan);
bool ops_supported(const TraceDAG& dag);

inline constexpr const char* kEntryVec = "trace_ew_vec";
inline constexpr const char* kEntryScalar = "trace_ew_scalar";
inline constexpr const char* kEntryRowNorm = "trace_row_norm_vec";
inline constexpr const char* kEntryRowNormScalar = "trace_row_norm_scalar";

}  // namespace brotensor::jit::msl
