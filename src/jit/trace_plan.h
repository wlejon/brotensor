#pragma once

// Device-neutral fusion plans for the trace JIT's GPU compilers.
//
// A plan is what a GPU emitter needs to know about a traced DAG before it
// writes any code: which nodes are kernel parameters (inputs read, outputs
// written), how each parameter's buffer maps onto the trace's dominant
// (rows, cols) shape (full, a (1, cols) row, or a (1, 1) scalar), its dtype,
// and how many elements one thread moves per access. The CUDA compiler
// (trace_compiler_cuda.cpp, PTX text) and the Vulkan compiler
// (trace_compiler_vulkan.cpp, brass MIR lowered to SPIR-V) build the same two
// plans from the same DAG and differ only in how they emit them.
//
//   ElementwisePlan  a pure map over the dominant shape.
//   RowNormPlan      one row reduction (RMSNorm or LayerNorm) with the
//                    elementwise nodes it depends on (the "pre" chain,
//                    evaluated before the reduction) and the ones that depend
//                    on it (the "post" chain), as one block per row group.

#include "trace_dag.h"

#include <cstdint>
#include <vector>

namespace brotensor::jit::plan {

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
    ElementwisePlan ew;       // buffers and shape
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
    // back out of cache. `x_output` indexes ew.outputs; -1 disables the reload.
    int x_output = -1;

    // Per ew.outputs index: produced by the pre chain, so written in pass one.
    std::vector<char> pre_output;
};

// Bytes per element (0 for a dtype the emitters cannot address).
int elem_bytes(Dtype d);

// True when the emitters can handle this dtype at all.
bool dtype_supported(Dtype d);

// Elements per thread for a 16-byte access at this dtype.
int vec_width_for(Dtype d);

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

// Collects the trace's parameter buffers and rejects anything the emitters
// cannot address. Returns false rather than throwing so the caller can try a
// different fusion.
bool build_plan(const TraceDAG& dag, ElementwisePlan& plan);

// A trace is a row-norm when exactly one node is a reduction and every other
// node is elementwise. The nodes the reduction depends on become the `pre`
// chain, which the kernel evaluates in both passes; everything downstream
// becomes `post`.
bool build_row_norm_plan(const TraceDAG& dag, RowNormPlan& plan);

bool all_fp32(const TraceDAG& dag);

}  // namespace brotensor::jit::plan
