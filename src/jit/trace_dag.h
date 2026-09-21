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
    Sigmoid
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
    void* buffer = nullptr; // destination or input data pointer
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

class TraceContext {
public:
    static TraceContext& current();

    void begin();
    TraceDAG end();
    bool is_active() const noexcept { return active_; }

    int get_or_register_slot(const Tensor& t);
    int record_op(TraceOpKind op, const std::vector<int>& in_slots, float scalar, const Tensor& out_tensor);
    int record_inplace_op(TraceOpKind op, int target_slot, int arg_slot, float scalar, const Tensor& target_tensor);

private:
    TraceContext() = default;
    bool active_ = false;
    TraceDAG dag_;
    std::unordered_map<const void*, int> active_ptr_to_slot_;
};

} // namespace brotensor::jit
