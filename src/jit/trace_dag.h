#pragma once

#include <brotensor/tensor.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

namespace brotensor::jit {

enum class TraceOpKind : uint8_t {
    Input = 0,
    Add,
    Sub,
    Mul,
    Div,
    AddScalar,
    SubScalar,
    MulScalar,
    DivScalar,
    ScalarSub,
    ScalarDiv,
    FMA,
    SiLU,
    GELU,
    ReLU,
    RMSNorm,
    LayerNorm,
    Modulate,
    Tanh,
    Sigmoid,
    Copy
};

const char* op_kind_name(TraceOpKind op);

// How a node's buffer maps onto the trace's dominant (rows, cols) shape.
//
// Row and Scalar are the broadcast forms the DiT modulation surfaces need: a
// (1, D) gate or scale row multiplied against an (N, D) activation, and a
// (1, 1) scalar. The emitters turn these into a different address computation
// for that operand, not into a materialised expansion — a broadcast operand
// contributes D (or 1) elements of traffic, not N*D.
enum class BroadcastKind : uint8_t { Full = 0, Row = 1, Scalar = 2 };

BroadcastKind broadcast_of(int node_rows, int node_cols, int rows, int cols);

struct TraceNode {
    int id = -1;
    TraceOpKind op = TraceOpKind::Input;
    std::vector<int> inputs;
    std::vector<int> outputs; // consumer node indices
    float scalar = 0.0f;
    // Destination or input data pointer — and null for every node the fused
    // kernel evaluates in registers and never spills.
    //
    // A buffer is present on exactly three kinds of node: an Input and an
    // in-place target, whose storage the caller owns and whose address is
    // therefore stable for the whole trace; a store() destination, likewise
    // caller-owned; and a live-at-end intermediate, which end_trace()
    // allocates and installs into the Tensor the caller is still holding.
    // Every other node — every dead intermediate — carries a null buffer and
    // is addressed by `id` alone, which is the reason the tracer no longer
    // allocates a full-size tensor per traced op. Nothing hashes the buffer,
    // so a null one cannot collide two different traces.
    void* buffer = nullptr;
    Device device = Device::CPU;
    Dtype dtype = Dtype::FP32;
    int rows = 0;
    int cols = 0;
    bool is_inplace = false;
    int inplace_target_slot = -1;
    bool is_live_at_end = false;
};

class TraceDAG {
public:
    TraceDAG() = default;

    int register_input(const Tensor& t);
    int add_node(TraceOpKind op, const std::vector<int>& inputs, float scalar = 0.0f,
                 void* buffer = nullptr, const Device& dev = Device::CPU,
                 Dtype dt = Dtype::FP32, int rows = 0, int cols = 0);
    int add_inplace_node(TraceOpKind op, int target_slot, int arg_slot,
                         void* buffer, float scalar = 0.0f);

    void analyze_liveness();
    uint64_t compute_hash() const;

    // The trace's dominant shape: the largest (rows * cols) any node carries.
    // Every non-broadcast node must match it; (1, cols) and (1, 1) nodes are
    // broadcast operands.
    void dominant_shape(int& rows, int& cols) const;

    const std::vector<TraceNode>& nodes() const noexcept { return nodes_; }
    std::vector<TraceNode>& nodes() noexcept { return nodes_; }
    const TraceNode& node(int id) const { return nodes_.at(id); }
    size_t node_count() const noexcept { return nodes_.size(); }

    // True when `slot` still describes `t` (same buffer, shape, dtype, device).
    bool slot_matches(int slot, const Tensor& t) const;
    void reset();

private:
    std::vector<TraceNode> nodes_;
    std::unordered_map<const void*, int> input_ptr_to_slot_;
};

// ─── TraceContext ───────────────────────────────────────────────────────────
//
// The thread's trace in progress, and the symbolic tensors it has handed out.
//
// A traced op does not produce a tensor; it produces a *name* for a node of
// the DAG. record_op() returns that name as a Tensor with a null `data`, a
// real shape/dtype/device and `jit_slot` set — so the expression the caller
// wrote goes on type-checking and composing exactly as before while allocating
// nothing. Node identity follows suit: a symbolic operand is identified by its
// slot, and only a caller-owned buffer (an input, an in-place target, a store()
// destination) is still identified by its pointer, which is safe precisely
// because the caller owns it and it cannot be recycled mid-trace.
//
// The catch a tracked-pointer table solves: a value the caller still holds
// when the trace closes has to end up owning a real buffer it can read from.
// Every symbolic Tensor handed out is recorded here as slot -> Tensor*, and
// Tensor's move ctor / move assignment / destructor keep that table pointing
// at wherever the caller moved the value (see detail::jit_slot_retarget in
// tensor.h). end() then allocates for exactly those slots.
class TraceContext {
public:
    static TraceContext& current();

    void begin();
    // Runs liveness, gives every still-held live-at-end intermediate a real
    // buffer, and returns the DAG. Any symbolic Tensor the caller still holds
    // whose value the fused kernel only consumes internally is neutralised to
    // an empty tensor — its contents were never computed to memory.
    TraceDAG end();
    // Throws away a trace in progress and neutralises every symbolic Tensor
    // the caller still holds to an empty tensor. Leaves the context idle.
    void discard();
    bool is_active() const noexcept { return active_; }

    int get_or_register_slot(const Tensor& t);

    // Records `op` and returns the symbolic Tensor standing for its result.
    Tensor record_op(TraceOpKind op, const std::vector<int>& in_slots, float scalar,
                     const Device& dev, Dtype dt, int rows, int cols);

    // Records a Copy into `dst`, a buffer the caller already owns.
    void record_store(int src_slot, const Tensor& dst);

    int record_inplace_op(TraceOpKind op, int target_slot, int arg_slot, float scalar,
                          Tensor& target_tensor);

    // Tensor lifetime callback: the symbolic Tensor for `slot` moved from
    // `from` to `to`, or was destroyed (`to` == nullptr). Never throws and
    // never allocates — Tensor's move and destructor are noexcept.
    void retarget(int slot, const Tensor* from, Tensor* to) noexcept;

private:
    TraceContext() = default;

    // Allocates and installs the buffers for live-at-end nodes that do not
    // already have one; demotes a live-at-end node nobody can observe.
    void materialize_();
    // Turns every still-tracked symbolic Tensor into an ordinary empty one.
    void release_tracked_() noexcept;

    bool active_ = false;
    TraceDAG dag_;
    std::unordered_map<const void*, int> active_ptr_to_slot_;
    std::unordered_map<int, Tensor*> slot_to_tensor_;
};

} // namespace brotensor::jit
