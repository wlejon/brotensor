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
    ElementwisePlan ew;       // buffers and shape; `vec` is always 1 here
    RowReduce reduce = RowReduce::RMS;

    int norm_node = -1;       // the RMSNorm / LayerNorm node
    int x_node = -1;          // what feeds it: an input, or the end of pre_ids
    int gamma_input = -1;     // index into ew.inputs, or -1
    int beta_input = -1;      // index into ew.inputs, or -1
    float eps = 1e-5f;

    // Nodes to evaluate before the reduction and after it. The pre chain runs
    // in both passes — that is what lets `x += p; rms_norm(x)` fuse, since the
    // reduction needs the updated x that the same kernel is about to write.
    std::vector<int> pre_ids;
    std::vector<int> post_ids;

    // Per ew.inputs index: whether the pre chain reads it, and whether pass
    // two needs it. Inputs only the post chain wants (a modulation row, say)
    // stay out of pass one, and vice versa.
    std::vector<char> pre_input;
    std::vector<char> post_input;

    // When the pre chain's result is itself one of the trace's outputs, pass
    // one can store it and pass two can read it back instead of replaying the
    // chain — one fewer read of every pre-chain input, and the value comes
    // back out of L2. `x_output` indexes ew.outputs; -1 disables the reload.
    int x_output = -1;

    // Per ew.outputs index: produced by the pre chain, so written in pass one.
    std::vector<char> pre_output;
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

// One module, two entries (kEntryRowNorm and kEntryRowNormScalar).
std::string emit_row_norm(const TraceDAG& dag, const RowNormPlan& plan,
                          const std::string& arch);

inline constexpr const char* kEntryVec = "trace_ew_vec";
inline constexpr const char* kEntryScalar = "trace_ew_scalar";
inline constexpr const char* kEntryRowNorm = "trace_row_norm_vec";
inline constexpr const char* kEntryRowNormScalar = "trace_row_norm_scalar";

// Upper bound on a row-norm block. The actual block is rows_per_block rows of
// threads_per_row threads each, never more than this.
inline constexpr int kRowThreads = 256;

// How the row-norm kernel tiles a (rows, cols) trace at `lanes` elements per
// access. `tpr` threads cooperate on one row and `rpb` rows share a block.
//
// A row needs ceil(cols/lanes) threads to cover it in one stride. Giving it
// 256 regardless — which is what a one-row-per-block kernel does — is right
// for a 4096-wide transformer activation and catastrophic for a VAE feature
// map, where cols is 96 to 384 and rows runs into the millions: seven eighths
// of every block idles, and the launch is a million blocks deep.
//
// `rpb` is also held to a divisor of `rows`, so the "my row is past the end"
// exit is uniform across a block and the reduction's barrier is safe.
void row_norm_geometry(int cols, int rows, int lanes, int& tpr, int& rpb);

}  // namespace brotensor::jit::ptx
