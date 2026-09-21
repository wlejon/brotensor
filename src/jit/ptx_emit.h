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

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::jit::ptx {

// One kernel parameter: a device pointer plus how to read through it.
struct BufferSpec {
    int node_id = -1;
    Dtype dtype = Dtype::FP32;
    BroadcastKind bcast = BroadcastKind::Full;
};

struct ElementwisePlan {
    std::vector<BufferSpec> inputs;
    std::vector<BufferSpec> outputs;
    int rows = 0;
    int cols = 0;
    std::int64_t numel = 0;
    int vec = 1;              // elements per thread in the vector entry
    bool any_row_bcast = false;
};

// Reduction at the head of a row-norm kernel.
enum class RowReduce { RMS, Mean };

struct RowNormPlan {
    ElementwisePlan ew;       // the chain applied after the norm
    RowReduce reduce = RowReduce::RMS;
    int x_input = -1;         // index into ew.inputs: the tensor being normed
    int gamma_input = -1;     // index into ew.inputs, or -1
    int beta_input = -1;      // index into ew.inputs, or -1
    int norm_node = -1;       // DAG node id the normalised value feeds
    float eps = 1e-5f;
};

// Elements per thread for a 16-byte access at this dtype.
int vec_width_for(Dtype d);

// Bytes per element.
int elem_bytes(Dtype d);

// True when the emitters can handle this dtype at all.
bool dtype_supported(Dtype d);

// One module, two entries (kEntryVec and kEntryScalar).
std::string emit_elementwise(const TraceDAG& dag, const ElementwisePlan& plan,
                             const std::string& arch);

// One module, one entry (kEntryRowNorm). The reduction makes a scalar variant
// pointless: the block already walks the row with a stride loop.
std::string emit_row_norm(const TraceDAG& dag, const RowNormPlan& plan,
                          const std::string& arch);

inline constexpr const char* kEntryVec = "trace_ew_vec";
inline constexpr const char* kEntryScalar = "trace_ew_scalar";
inline constexpr const char* kEntryRowNorm = "trace_row_norm";

// Threads per block for the row-norm kernel; also the reduction width.
inline constexpr int kRowThreads = 256;

}  // namespace brotensor::jit::ptx
