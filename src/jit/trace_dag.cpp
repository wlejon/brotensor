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
        case TraceOpKind::Copy: return "Copy";
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

namespace {

// The hook Tensor calls from its move ctor / move assignment / destructor.
// Installed by begin(), which is the earliest point at which a symbolic
// tensor can exist, so there is no static-initialisation order to get wrong.
void slot_retarget(int slot, const Tensor* from, Tensor* to) {
    TraceContext::current().retarget(slot, from, to);
}

}  // namespace

void TraceContext::retarget(int slot, const Tensor* from, Tensor* to) noexcept {
    auto it = slot_to_tensor_.find(slot);
    // A stale entry — the slot was already materialised or re-tracked — must
    // not be clobbered by a tensor that no longer stands for it.
    if (it == slot_to_tensor_.end() || it->second != from) return;
    if (to) {
        it->second = to;
    } else {
        slot_to_tensor_.erase(it);
    }
}

void TraceContext::release_tracked_() noexcept {
    for (auto& kv : slot_to_tensor_) {
        Tensor* t = kv.second;
        if (!t) continue;
        // It never owned anything, and its value was never written anywhere,
        // so the honest result is an empty tensor rather than a shape with no
        // storage behind it.
        t->jit_slot = -1;
        t->rows = 0;
        t->cols = 0;
    }
    slot_to_tensor_.clear();
}

void TraceContext::begin() {
    if (active_) {
        throw std::runtime_error("brotensor::jit: Nested begin_trace() calls are not permitted");
    }
    detail::jit_slot_retarget = &slot_retarget;
    dag_.reset();
    active_ptr_to_slot_.clear();
    release_tracked_();
    active_ = true;
}

void TraceContext::discard() {
    dag_.reset();
    active_ptr_to_slot_.clear();
    release_tracked_();
    active_ = false;
}

void TraceContext::materialize_() {
    for (TraceNode& n : dag_.nodes()) {
        if (!n.is_live_at_end) continue;
        if (n.buffer != nullptr) continue;  // caller already owns the storage

        auto it = slot_to_tensor_.find(n.id);
        if (it == slot_to_tensor_.end() || it->second == nullptr) {
            // The caller dropped the Tensor, so nothing can read this value.
            // Leave it an interior value of the fused kernel rather than
            // allocating a buffer no one will look at.
            n.is_live_at_end = false;
            continue;
        }

        Tensor* held = it->second;
        Tensor buf = Tensor::empty_on(n.device, n.rows, n.cols, n.dtype);
        if (buf.data == nullptr) {
            n.is_live_at_end = false;
            continue;
        }
        // Drop the tracking first: the move assignment below must see an
        // ordinary tensor on both sides, or it would retarget the slot it is
        // in the middle of retiring.
        slot_to_tensor_.erase(it);
        held->jit_slot = -1;
        *held = std::move(buf);
        n.buffer = held->data;
    }
}

TraceDAG TraceContext::end() {
    if (!active_) {
        throw std::runtime_error("brotensor::jit: end_trace() called without matching begin_trace()");
    }
    dag_.analyze_liveness();
    materialize_();
    active_ = false;
    active_ptr_to_slot_.clear();
    // Whatever is still tracked stands for a value the fused kernel only ever
    // holds in registers; there is nothing to hand back.
    release_tracked_();

    bool any_output = false;
    for (const TraceNode& n : dag_.nodes()) {
        if (n.is_live_at_end) { any_output = true; break; }
    }
    if (!any_output) {
        dag_.reset();
        throw std::runtime_error(
            "brotensor::jit: the trace has no observable result — every value it "
            "produced was dropped before end_trace(), so there is nothing for the "
            "fused kernel to write. Keep the result Tensor alive across "
            "end_trace(), store() it into a buffer you own, or write in place "
            "into one of the inputs");
    }
    return std::move(dag_);
}

int TraceContext::get_or_register_slot(const Tensor& t) {
    // A symbolic intermediate already *is* a node: it names its own slot, and
    // it has no buffer to key on.
    if (t.jit_slot >= 0) return t.jit_slot;

    // Everything else is caller-owned storage, keyed on its address. The
    // address is stable for the trace's lifetime because the caller owns it —
    // but a slot recorded earlier can still be stale if that tensor was freed
    // and its address reused, so confirm the node still describes the tensor
    // in hand before reusing its slot.
    auto it = active_ptr_to_slot_.find(t.data);
    if (it != active_ptr_to_slot_.end() && dag_.slot_matches(it->second, t)) {
        return it->second;
    }
    int slot = dag_.register_input(t);
    active_ptr_to_slot_[t.data] = slot;
    return slot;
}

Tensor TraceContext::record_op(TraceOpKind op, const std::vector<int>& in_slots,
                               float scalar, const Device& dev, Dtype dt,
                               int rows, int cols) {
    const int slot = dag_.add_node(op, in_slots, scalar, /*buffer=*/nullptr,
                                   dev, dt, rows, cols);
    Tensor out;
    out.device = dev;
    out.dtype = dt;
    out.rows = rows;
    out.cols = cols;
    out.jit_slot = slot;
    // With copy elision `out` is already the caller's object; without it, the
    // move ctor retargets this entry onto wherever it lands.
    slot_to_tensor_[slot] = &out;
    return out;
}

void TraceContext::record_store(int src_slot, const Tensor& dst) {
    if (dst.jit_slot >= 0) {
        throw std::runtime_error(
            "brotensor::jit: store() needs a destination the caller owns; its "
            "target is a traced intermediate, which has no buffer to write to");
    }
    const int slot = dag_.add_node(TraceOpKind::Copy, {src_slot}, 0.0f, dst.data,
                                   dst.device, dst.dtype, dst.rows, dst.cols);
    active_ptr_to_slot_[dst.data] = slot;
}

int TraceContext::record_inplace_op(TraceOpKind op, int target_slot, int arg_slot,
                                    float scalar, Tensor& target_tensor) {
    int slot = dag_.add_inplace_node(op, target_slot, arg_slot, target_tensor.data, scalar);
    if (target_tensor.jit_slot >= 0) {
        // In-place on a traced intermediate: it has no buffer, and from here
        // on it stands for the node this op just produced.
        slot_to_tensor_.erase(target_tensor.jit_slot);
        target_tensor.jit_slot = slot;
        slot_to_tensor_[slot] = &target_tensor;
    } else {
        active_ptr_to_slot_[target_tensor.data] = slot;
    }
    return slot;
}

} // namespace brotensor::jit
