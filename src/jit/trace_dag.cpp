#include "trace_dag.h"
#include <cstring>
#include <stdexcept>

namespace brotensor::jit {

const char* op_kind_name(TraceOpKind op) {
    switch (op) {
        case TraceOpKind::Input: return "Input";
        case TraceOpKind::Add: return "Add";
        case TraceOpKind::Sub: return "Sub";
        case TraceOpKind::Mul: return "Mul";
        case TraceOpKind::Div: return "Div";
        case TraceOpKind::AddScalar: return "AddScalar";
        case TraceOpKind::SubScalar: return "SubScalar";
        case TraceOpKind::MulScalar: return "MulScalar";
        case TraceOpKind::DivScalar: return "DivScalar";
        case TraceOpKind::ScalarSub: return "ScalarSub";
        case TraceOpKind::ScalarDiv: return "ScalarDiv";
        case TraceOpKind::FMA: return "FMA";
        case TraceOpKind::SiLU: return "SiLU";
        case TraceOpKind::GELU: return "GELU";
        case TraceOpKind::ReLU: return "ReLU";
        case TraceOpKind::RMSNorm: return "RMSNorm";
        case TraceOpKind::LayerNorm: return "LayerNorm";
        case TraceOpKind::Modulate: return "Modulate";
        case TraceOpKind::Tanh: return "Tanh";
        case TraceOpKind::Sigmoid: return "Sigmoid";
    }
    return "Unknown";
}

BroadcastKind broadcast_of(int node_rows, int node_cols, int rows, int cols) {
    if (node_rows == 1 && node_cols == 1 && rows * cols != 1) {
        return BroadcastKind::Scalar;
    }
    if (node_rows == 1 && node_cols == cols && rows != 1) {
        return BroadcastKind::Row;
    }
    // Anything else is addressed linearly. A node whose element count does not
    // match the dominant shape is rejected by the compiler before it gets
    // here (see trace_compiler_cuda.cpp's plan builder).
    return BroadcastKind::Full;
}

bool TraceDAG::slot_matches(int slot, const Tensor& t) const {
    if (slot < 0 || slot >= static_cast<int>(nodes_.size())) return false;
    const TraceNode& n = nodes_[static_cast<std::size_t>(slot)];
    return n.buffer == t.data && n.rows == t.rows && n.cols == t.cols &&
           n.dtype == t.dtype && n.device == t.device;
}

int TraceDAG::register_input(const Tensor& t) {
    auto it = input_ptr_to_slot_.find(t.data);
    if (it != input_ptr_to_slot_.end() && slot_matches(it->second, t)) {
        return it->second;
    }

    int id = static_cast<int>(nodes_.size());
    TraceNode node;
    node.id = id;
    node.op = TraceOpKind::Input;
    node.buffer = t.data;
    node.device = t.device;
    node.dtype = t.dtype;
    node.rows = t.rows;
    node.cols = t.cols;
    node.is_inplace = false;
    node.is_live_at_end = false;

    nodes_.push_back(std::move(node));
    input_ptr_to_slot_[t.data] = id;
    return id;
}

int TraceDAG::add_node(TraceOpKind op, const std::vector<int>& inputs, float scalar,
                       void* buffer, const Device& dev, Dtype dt, int rows, int cols) {
    int id = static_cast<int>(nodes_.size());
    TraceNode node;
    node.id = id;
    node.op = op;
    node.inputs = inputs;
    node.scalar = scalar;
    node.buffer = buffer;
    node.device = dev;
    node.dtype = dt;
    node.rows = rows;
    node.cols = cols;
    node.is_inplace = false;
    node.is_live_at_end = false;

    nodes_.push_back(std::move(node));
    return id;
}

int TraceDAG::add_inplace_node(TraceOpKind op, int target_slot, int arg_slot,
                               void* buffer, float scalar) {
    int id = static_cast<int>(nodes_.size());
    const auto& target_node = nodes_.at(target_slot);

    TraceNode node;
    node.id = id;
    node.op = op;
    node.inputs = {target_slot, arg_slot};
    node.scalar = scalar;
    node.buffer = buffer;
    node.device = target_node.device;
    node.dtype = target_node.dtype;
    node.rows = target_node.rows;
    node.cols = target_node.cols;
    node.is_inplace = true;
    node.inplace_target_slot = target_slot;
    node.is_live_at_end = true;

    nodes_.push_back(std::move(node));
    return id;
}

void TraceDAG::analyze_liveness() {
    for (auto& n : nodes_) {
        n.outputs.clear();
    }
    for (const auto& n : nodes_) {
        for (int in_id : n.inputs) {
            if (in_id >= 0 && in_id < static_cast<int>(nodes_.size())) {
                nodes_[in_id].outputs.push_back(n.id);
            }
        }
    }
    for (auto& n : nodes_) {
        if (n.op == TraceOpKind::Input) {
            n.is_live_at_end = false;
        } else if (n.is_inplace) {
            n.is_live_at_end = true;
        } else if (n.outputs.empty()) {
            n.is_live_at_end = true;
        } else {
            n.is_live_at_end = false;
        }
    }
}

void TraceDAG::dominant_shape(int& rows, int& cols) const {
    rows = 0;
    cols = 0;
    int64_t best = 0;
    for (const auto& n : nodes_) {
        const int64_t numel = static_cast<int64_t>(n.rows) * static_cast<int64_t>(n.cols);
        if (numel > best) {
            best = numel;
            rows = n.rows;
            cols = n.cols;
        }
    }
}

uint64_t TraceDAG::compute_hash() const {
    // 64-bit FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    auto hash_val = [&hash](uint64_t val) {
        hash ^= val;
        hash *= 1099511628211ULL;
    };

    hash_val(nodes_.size());
    for (const auto& n : nodes_) {
        hash_val(static_cast<uint64_t>(n.op));
        hash_val(static_cast<uint64_t>(n.device.type));
        hash_val(static_cast<uint64_t>(n.device.index));
        hash_val(static_cast<uint64_t>(n.dtype));
        hash_val(static_cast<uint64_t>(n.rows));
        hash_val(static_cast<uint64_t>(n.cols));
        hash_val(n.is_inplace ? 1ULL : 0ULL);
        hash_val(n.is_live_at_end ? 1ULL : 0ULL);

        uint32_t scalar_bits = 0;
        std::memcpy(&scalar_bits, &n.scalar, sizeof(float));
        hash_val(scalar_bits);

        hash_val(n.inputs.size());
        for (int in_id : n.inputs) {
            hash_val(static_cast<uint64_t>(in_id));
        }
    }
    return hash;
}

void TraceDAG::reset() {
    nodes_.clear();
    input_ptr_to_slot_.clear();
}

TraceContext& TraceContext::current() {
    static thread_local TraceContext instance;
    return instance;
}

void TraceContext::begin() {
    if (active_) {
        throw std::runtime_error("brotensor::jit: Nested begin_trace() calls are not permitted");
    }
    dag_.reset();
    active_ptr_to_slot_.clear();
    active_ = true;
}

TraceDAG TraceContext::end() {
    if (!active_) {
        throw std::runtime_error("brotensor::jit: end_trace() called without matching begin_trace()");
    }
    dag_.analyze_liveness();
    active_ = false;
    active_ptr_to_slot_.clear();
    return std::move(dag_);
}

int TraceContext::get_or_register_slot(const Tensor& t) {
    // A traced op's output Tensor owns its buffer and frees it when the
    // enclosing expression ends, so an address seen earlier in this trace can
    // be handed back by the allocator to a different tensor. Confirm the
    // recorded node still describes the tensor in hand before reusing its
    // slot; a mismatch means the entry is stale and the buffer is a new input.
    auto it = active_ptr_to_slot_.find(t.data);
    if (it != active_ptr_to_slot_.end() && dag_.slot_matches(it->second, t)) {
        return it->second;
    }
    int slot = dag_.register_input(t);
    active_ptr_to_slot_[t.data] = slot;
    return slot;
}

int TraceContext::record_op(TraceOpKind op, const std::vector<int>& in_slots, float scalar, const Tensor& out_tensor) {
    int slot = dag_.add_node(op, in_slots, scalar, out_tensor.data,
                             out_tensor.device, out_tensor.dtype, out_tensor.rows, out_tensor.cols);
    active_ptr_to_slot_[out_tensor.data] = slot;
    return slot;
}

int TraceContext::record_inplace_op(TraceOpKind op, int target_slot, int arg_slot, float scalar, const Tensor& target_tensor) {
    int slot = dag_.add_inplace_node(op, target_slot, arg_slot, target_tensor.data, scalar);
    active_ptr_to_slot_[target_tensor.data] = slot;
    return slot;
}

} // namespace brotensor::jit
