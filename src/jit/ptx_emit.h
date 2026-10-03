#pragma once

// PTX emitters for the trace JIT's CUDA backend.
//
// Two shapes of kernel cover every pattern the DiT/VAE/scheduler surfaces
// hand the tracer:
//
//   elementwise — a pure map over the trace's dominant (rows, cols) shape.
//     One thread handles `vec` consecutive elements so each access moves 16
//     bytes; operands that are (1, cols) rows or (1, 1) scalars are addressed
//     by column rather than materialised. Every buffer carries its own dtype:
//     BF16/FP16 are unpacked to FP32 on load and rounded back on store, so
//     the arithmetic is FP32 while the traffic is 2 bytes per element.
//
//   row-norm — a row reduction (RMSNorm or LayerNorm) followed by an
//     elementwise chain, as one block per row. This is the LN-modulate and
//     the VAE's norm+silu seam: the eager form is two kernels with a full
//     (rows, cols) round trip between them, the fused form reads the row
//     twice and writes once.
//
// Both emit a single module carrying a vector entry and a scalar entry; the
// launcher picks between them from the bound pointers' alignment and the
// element count, so a misaligned row view still runs (slower) rather than
// faulting.

#include "trace_dag.h"
#include "trace_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::jit::ptx {

// The plans and their helpers are device-neutral (trace_plan.h); the Vulkan
// compiler builds the same ones.
using plan::BufferSpec;
using plan::ElementwisePlan;
using plan::RowReduce;
using plan::RowNormPlan;
using plan::vec_width_for;
using plan::elem_bytes;
using plan::dtype_supported;
using plan::kRowThreads;
using plan::row_norm_geometry;

// One module, two entries (kEntryVec and kEntryScalar).
std::string emit_elementwise(const TraceDAG& dag, const ElementwisePlan& plan,
                             const std::string& arch);

// One module, two entries (kEntryRowNorm and kEntryRowNormScalar).
std::string emit_row_norm(const TraceDAG& dag, const RowNormPlan& plan,
                          const std::string& arch);

inline constexpr const char* kEntryVec = "trace_ew_vec";
inline constexpr const char* kEntryScalar = "trace_ew_scalar";
inline constexpr const char* kEntryRowNorm = "trace_row_norm_vec";
inline constexpr const char* kEntryRowNormScalar = "trace_row_norm_scalar";

}  // namespace brotensor::jit::ptx
