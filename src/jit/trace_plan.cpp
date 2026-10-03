// Device-neutral fusion plans (trace_plan.h). Moved out of the CUDA trace
// compiler so the Vulkan compiler builds exactly the same plans.

#include "trace_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace brotensor::jit::plan {

int elem_bytes(Dtype d) {
    switch (d) {
        case Dtype::FP32: return 4;
        case Dtype::FP16:
        case Dtype::BF16: return 2;
        default: return 0;
    }
}

bool dtype_supported(Dtype d) { return elem_bytes(d) != 0; }

int vec_width_for(Dtype d) {
    const int e = elem_bytes(d);
    return e ? 16 / e : 1;
}

void row_norm_geometry(int cols, int rows, int lanes, int& tpr, int& rpb) {
    if (lanes < 1) lanes = 1;
    const int per_row = (cols + lanes - 1) / lanes;   // threads to cover a row
    int t = 32;
    while (t < per_row && t < kRowThreads) t <<= 1;
    int r = kRowThreads / t;
    // The out-of-range exit has to be block-uniform, or the threads that stay
    // hit a barrier the ones that left never will.
    while (r > 1 && rows % r != 0) r >>= 1;
    tpr = t;
    rpb = r;
}

bool build_plan(const TraceDAG& dag, ElementwisePlan& plan) {
    int rows = 0, cols = 0;
    dag.dominant_shape(rows, cols);
    if (rows <= 0 || cols <= 0) return false;

    plan.rows = rows;
    plan.cols = cols;
    plan.numel = static_cast<std::int64_t>(rows) * cols;

    int lanes = 8;
    for (const auto& n : dag.nodes()) {
        const bool is_in = (n.op == TraceOpKind::Input);
        const bool is_out = n.is_live_at_end;
        if (!is_in && !is_out) continue;
        if (!dtype_supported(n.dtype)) return false;

        const BroadcastKind bk = broadcast_of(n.rows, n.cols, rows, cols);
        if (bk == BroadcastKind::Full &&
            static_cast<std::int64_t>(n.rows) * n.cols != plan.numel) {
            return false;
        }
        if (is_out && bk != BroadcastKind::Full) return false;

        lanes = std::min(lanes, vec_width_for(n.dtype));
        if (bk == BroadcastKind::Row) plan.any_row_bcast = true;

        const BufferSpec spec{n.id, n.dtype, bk};
        if (is_in) plan.inputs.push_back(spec);
        if (is_out) plan.outputs.push_back(spec);
    }
    if (plan.outputs.empty()) return false;

    if (plan.numel % lanes != 0) lanes = 1;
    if (plan.any_row_bcast && cols % lanes != 0) lanes = 1;
    plan.vec = lanes;
    return true;
}

bool build_row_norm_plan(const TraceDAG& dag, RowNormPlan& plan) {
    const auto& nodes = dag.nodes();
    const TraceNode* norm = nullptr;
    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm) {
            if (norm) return false;  // more than one reduction
            norm = &n;
        }
    }
    if (!norm || norm->inputs.empty()) return false;

    if (!build_plan(dag, plan.ew)) return false;
    if (plan.ew.cols != norm->cols) return false;
    if (plan.ew.rows != norm->rows) return false;

    auto input_index = [&](int node_id) {
        for (std::size_t i = 0; i < plan.ew.inputs.size(); ++i) {
            if (plan.ew.inputs[i].node_id == node_id) return static_cast<int>(i);
        }
        return -1;
    };
    if (norm->inputs.size() > 1) {
        plan.gamma_input = input_index(norm->inputs[1]);
        if (plan.gamma_input < 0) return false;
    }
    if (norm->inputs.size() > 2) {
        plan.beta_input = input_index(norm->inputs[2]);
        if (plan.beta_input < 0) return false;
    }

    plan.x_node = norm->inputs[0];
    plan.norm_node = norm->id;
    plan.reduce = (norm->op == TraceOpKind::LayerNorm) ? RowReduce::Mean : RowReduce::RMS;
    plan.eps = norm->scalar > 0.0f ? norm->scalar : 1e-5f;

    // Reverse reachability from the reduction's data input. Node ids are
    // assigned in creation order, which is topological, so one backward sweep
    // is enough.
    std::vector<char> in_pre(nodes.size(), 0);
    if (plan.x_node >= 0) in_pre[static_cast<std::size_t>(plan.x_node)] = 1;
    for (std::size_t i = nodes.size(); i-- > 0;) {
        if (!in_pre[i]) continue;
        for (int src : nodes[i].inputs) {
            if (src >= 0) in_pre[static_cast<std::size_t>(src)] = 1;
        }
    }
    in_pre[static_cast<std::size_t>(norm->id)] = 0;

    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::Input) continue;
        if (n.id == norm->id) continue;
        if (in_pre[static_cast<std::size_t>(n.id)]) {
            plan.pre_ids.push_back(n.id);
        } else {
            plan.post_ids.push_back(n.id);
        }
    }

    plan.pre_input.assign(plan.ew.inputs.size(), 0);
    for (std::size_t i = 0; i < plan.ew.inputs.size(); ++i) {
        const int id = plan.ew.inputs[i].node_id;
        plan.pre_input[i] = in_pre[static_cast<std::size_t>(id)] ? 1 : 0;
    }
    plan.pre_output.assign(plan.ew.outputs.size(), 0);
    for (std::size_t i = 0; i < plan.ew.outputs.size(); ++i) {
        const int id = plan.ew.outputs[i].node_id;
        plan.pre_output[i] = in_pre[static_cast<std::size_t>(id)] ? 1 : 0;
    }

    // Pass two can skip the pre chain entirely when its result is one of the
    // trace's own outputs — pass one writes it, pass two reads it back — and
    // when nothing downstream of the reduction needs any *other* pre value.
    plan.x_output = -1;
    if (!plan.pre_ids.empty()) {
        int x_out = -1;
        for (std::size_t i = 0; i < plan.ew.outputs.size(); ++i) {
            if (plan.ew.outputs[i].node_id == plan.x_node) x_out = static_cast<int>(i);
        }
        bool post_only_needs_x = true;
        for (int pid : plan.post_ids) {
            for (int src : nodes[static_cast<std::size_t>(pid)].inputs) {
                if (src < 0) continue;
                if (src == plan.x_node) continue;
                if (nodes[static_cast<std::size_t>(src)].op == TraceOpKind::Input) continue;
                if (in_pre[static_cast<std::size_t>(src)]) post_only_needs_x = false;
            }
        }
        if (x_out >= 0 && post_only_needs_x) plan.x_output = x_out;
    }

    // Without the reload, pass two replays the pre chain from its inputs —
    // and pass one has already overwritten whatever it stored. An in-place
    // pre chain (`x += p` with x a pass-one output) would replay as
    // x + p + p. Only the reload form is correct once pass one stores.
    if (plan.x_output < 0) {
        for (char w : plan.pre_output) {
            if (w) return false;
        }
    }

    // What pass two loads: with the reload it is the reduction's gamma/beta
    // plus whatever the post chain reads; without it, everything.
    plan.post_input.assign(plan.ew.inputs.size(), plan.x_output >= 0 ? 0 : 1);
    if (plan.x_output >= 0) {
        if (plan.gamma_input >= 0) plan.post_input[static_cast<std::size_t>(plan.gamma_input)] = 1;
        if (plan.beta_input >= 0) plan.post_input[static_cast<std::size_t>(plan.beta_input)] = 1;
        for (int pid : plan.post_ids) {
            for (int src : nodes[static_cast<std::size_t>(pid)].inputs) {
                if (src < 0) continue;
                for (std::size_t i = 0; i < plan.ew.inputs.size(); ++i) {
                    if (plan.ew.inputs[i].node_id == src) plan.post_input[i] = 1;
                }
            }
        }
    }

    // The row kernel's stride loop takes `vec` columns per thread per step.
    // build_plan already reduced `vec` to the narrowest dtype in play; the row
    // form additionally needs D to divide evenly so a vector never straddles a
    // row boundary.
    if (plan.ew.cols % plan.ew.vec != 0) plan.ew.vec = 1;
    return true;
}

bool all_fp32(const TraceDAG& dag) {
    for (const auto& n : dag.nodes()) {
        if (n.dtype != Dtype::FP32) return false;
    }
    return true;
}

}  // namespace brotensor::jit::plan
