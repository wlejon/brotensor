#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"
#include "ptx_emit.h"

#if BROTENSOR_HAS_CUDA

#include "../cuda/cuda_jit.h"
#include "../cuda/cuda_jit_engine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::jit::cuda {

using detail::cuda::jit::CudaJitEngine;

namespace {

std::string query_cuda_arch() {
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess) {
            const int sm = major * 10 + minor;
            if (sm >= 75) return "sm_" + std::to_string(sm);
        }
    }
    return "sm_75";
}

// Stable argument storage for one bound launch. The pointer array and the
// element count are filled once at bind time and the `params` vector points
// into them, so replaying a trace does no allocation — which matters when the
// whole point of the fusion is that the kernel costs a few microseconds.
struct LaunchArgs {
    std::vector<CUdeviceptr> ptrs;
    std::uint32_t n = 0;
    std::vector<void*> params;

    void build(const std::vector<void*>& outs, const std::vector<void*>& ins,
               std::uint32_t count, bool with_count) {
        ptrs.clear();
        ptrs.reserve(outs.size() + ins.size());
        for (void* p : outs) ptrs.push_back(reinterpret_cast<CUdeviceptr>(p));
        for (void* p : ins) ptrs.push_back(reinterpret_cast<CUdeviceptr>(p));
        n = count;
        params.clear();
        params.reserve(ptrs.size() + 1);
        for (auto& d : ptrs) params.push_back(&d);
        if (with_count) params.push_back(&n);
    }
};

bool aligned(void* p, int bytes) {
    return (reinterpret_cast<std::uintptr_t>(p) % static_cast<std::uintptr_t>(bytes)) == 0;
}

void launch(CUfunction fn, unsigned grid, unsigned block, LaunchArgs& args,
            const char* what) {
    const CUresult res = detail::cuda::drv::cuLaunchKernel(
        fn, grid, 1, 1, block, 1, 1, 0,
        reinterpret_cast<CUstream>(cuda_current_stream()),
        args.params.data(), nullptr);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("brotensor::jit: cuLaunchKernel failed for ") + what);
    }
}

// ── plan construction ───────────────────────────────────────────────────────

// Collects the trace's parameter buffers and rejects anything the emitters
// cannot address. Returns false rather than throwing so the caller can try a
// different fusion.
bool build_plan(const TraceDAG& dag, ptx::ElementwisePlan& plan) {
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
        if (!ptx::dtype_supported(n.dtype)) return false;

        const BroadcastKind bk = broadcast_of(n.rows, n.cols, rows, cols);
        if (bk == BroadcastKind::Full &&
            static_cast<std::int64_t>(n.rows) * n.cols != plan.numel) {
            return false;
        }
        if (is_out && bk != BroadcastKind::Full) return false;

        lanes = std::min(lanes, ptx::vec_width_for(n.dtype));
        if (bk == BroadcastKind::Row) plan.any_row_bcast = true;

        const ptx::BufferSpec spec{n.id, n.dtype, bk};
        if (is_in) plan.inputs.push_back(spec);
        if (is_out) plan.outputs.push_back(spec);
    }
    if (plan.outputs.empty()) return false;

    if (plan.numel % lanes != 0) lanes = 1;
    if (plan.any_row_bcast && cols % lanes != 0) lanes = 1;
    plan.vec = lanes;
    return true;
}

// A trace is a row-norm when exactly one node is a reduction and every other
// node is elementwise. The nodes the reduction depends on become the `pre`
// chain, which the kernel evaluates in both passes; everything downstream
// becomes `post`. That partition is what lets `x += p; rms_norm(x)` fuse at
// any dtype — the reduction consumes an x this same kernel produces.
bool build_row_norm_plan(const TraceDAG& dag, ptx::RowNormPlan& plan) {
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
    plan.reduce = (norm->op == TraceOpKind::LayerNorm) ? ptx::RowReduce::Mean
                                                       : ptx::RowReduce::RMS;
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

// ── generic elementwise ─────────────────────────────────────────────────────

std::shared_ptr<TraceHandleImpl> compile_elementwise(const TraceDAG& dag,
                                                     const ptx::ElementwisePlan& plan) {
    const std::string arch = query_cuda_arch();
    const std::string ptx_src = ptx::emit_elementwise(dag, plan, arch);
    const std::string key = "trace_ew_" + arch + "_" + std::to_string(dag.compute_hash());

    // One key per entry: the engine caches CUfunctions, not modules.
    CUfunction fn_vec =
        CudaJitEngine::instance().get_function(key + ":v", ptx_src, ptx::kEntryVec);
    CUfunction fn_sca =
        CudaJitEngine::instance().get_function(key + ":s", ptx_src, ptx::kEntryScalar);

    const int lanes = plan.vec;
    const std::int64_t numel = plan.numel;

    // Access width in bytes per buffer, for the alignment test that decides
    // between the two entries.
    std::vector<int> in_align(plan.inputs.size()), out_align(plan.outputs.size());
    for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
        in_align[i] = (plan.inputs[i].bcast == BroadcastKind::Scalar)
                          ? ptx::elem_bytes(plan.inputs[i].dtype)
                          : lanes * ptx::elem_bytes(plan.inputs[i].dtype);
    }
    for (std::size_t i = 0; i < plan.outputs.size(); ++i) {
        out_align[i] = lanes * ptx::elem_bytes(plan.outputs[i].dtype);
    }

    auto bind = [fn_vec, fn_sca, lanes, numel, in_align, out_align](
                    const std::vector<void*>& ins, const std::vector<void*>& outs) {
        bool use_vec = lanes > 1;
        if (use_vec) {
            for (std::size_t i = 0; i < ins.size() && use_vec; ++i) {
                if (!aligned(ins[i], in_align[i])) use_vec = false;
            }
            for (std::size_t i = 0; i < outs.size() && use_vec; ++i) {
                if (!aligned(outs[i], out_align[i])) use_vec = false;
            }
        }

        const std::int64_t slots = use_vec ? numel / lanes : numel;
        auto args = std::make_shared<LaunchArgs>();
        args->build(outs, ins, static_cast<std::uint32_t>(slots), true);

        unsigned grid = static_cast<unsigned>((slots + 255) / 256);
        if (grid == 0) grid = 1;
        if (grid > 65535) grid = 65535;

        CUfunction fn = use_vec ? fn_vec : fn_sca;
        auto h = std::make_shared<TraceHandleImpl>();
        h->is_cuda = true;
        h->launch_count = 1;
        h->fusion_name = use_vec ? "elementwise-vec" : "elementwise-scalar";
        h->execute_fn = [fn, grid, args]() {
            launch(fn, grid, 256, *args, "trace elementwise");
        };
        return h;
    };

    std::vector<void*> ins, outs;
    for (const auto& s : plan.inputs) ins.push_back(dag.node(s.node_id).buffer);
    for (const auto& s : plan.outputs) outs.push_back(dag.node(s.node_id).buffer);

    auto h = bind(ins, outs);
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return bind(in_b, out_b);
    };
    return h;
}

// ── generic row-norm ────────────────────────────────────────────────────────

std::shared_ptr<TraceHandleImpl> compile_row_norm(const TraceDAG& dag,
                                                  const ptx::RowNormPlan& plan) {
    const std::string arch = query_cuda_arch();
    const std::string ptx_src = ptx::emit_row_norm(dag, plan, arch);
    const std::string key = "trace_rn_" + arch + "_" + std::to_string(dag.compute_hash());

    CUfunction fn_vec =
        CudaJitEngine::instance().get_function(key + ":v", ptx_src, ptx::kEntryRowNorm);
    CUfunction fn_sca =
        CudaJitEngine::instance().get_function(key + ":s", ptx_src, ptx::kEntryRowNormScalar);

    const int rows = plan.ew.rows;
    const int cols = plan.ew.cols;
    const int lanes = plan.ew.vec;
    const bool is_ln = (plan.reduce == ptx::RowReduce::Mean);

    std::vector<int> in_align(plan.ew.inputs.size()), out_align(plan.ew.outputs.size());
    for (std::size_t i = 0; i < plan.ew.inputs.size(); ++i) {
        in_align[i] = (plan.ew.inputs[i].bcast == BroadcastKind::Scalar)
                          ? ptx::elem_bytes(plan.ew.inputs[i].dtype)
                          : lanes * ptx::elem_bytes(plan.ew.inputs[i].dtype);
    }
    for (std::size_t i = 0; i < plan.ew.outputs.size(); ++i) {
        out_align[i] = lanes * ptx::elem_bytes(plan.ew.outputs[i].dtype);
    }

    auto bind = [fn_vec, fn_sca, rows, cols, lanes, is_ln, in_align, out_align](
                    const std::vector<void*>& ins, const std::vector<void*>& outs) {
        bool use_vec = lanes > 1;
        for (std::size_t i = 0; i < ins.size() && use_vec; ++i) {
            if (!aligned(ins[i], in_align[i])) use_vec = false;
        }
        for (std::size_t i = 0; i < outs.size() && use_vec; ++i) {
            if (!aligned(outs[i], out_align[i])) use_vec = false;
        }

        auto args = std::make_shared<LaunchArgs>();
        args->build(outs, ins, 0, /*with_count=*/false);

        // The entry chosen here decides the tiling: the scalar fallback covers
        // a row with more threads than the vector one does, so its geometry
        // has to be recomputed, not inherited.
        int tpr = ptx::kRowThreads, rpb = 1;
        ptx::row_norm_geometry(cols, rows, use_vec ? lanes : 1, tpr, rpb);
        unsigned grid = static_cast<unsigned>((rows + rpb - 1) / rpb);
        if (grid == 0) grid = 1;
        const unsigned block = static_cast<unsigned>(tpr * rpb);

        CUfunction fn = use_vec ? fn_vec : fn_sca;
        auto h = std::make_shared<TraceHandleImpl>();
        h->is_cuda = true;
        h->launch_count = 1;
        h->fusion_name = is_ln ? (use_vec ? "row-layernorm-chain" : "row-layernorm-chain-scalar")
                               : (use_vec ? "row-rmsnorm-chain" : "row-rmsnorm-chain-scalar");
        h->execute_fn = [fn, grid, block, args]() {
            launch(fn, grid, block, *args, "trace row-norm");
        };
        return h;
    };

    std::vector<void*> ins, outs;
    for (const auto& s : plan.ew.inputs) ins.push_back(dag.node(s.node_id).buffer);
    for (const auto& s : plan.ew.outputs) outs.push_back(dag.node(s.node_id).buffer);

    auto h = bind(ins, outs);
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return bind(in_b, out_b);
    };
    return h;
}

// ── hand-written residual + RMSNorm ─────────────────────────────────────────

std::shared_ptr<TraceHandleImpl> compile_residual_rmsnorm(const TraceDAG& dag) {
    const auto& nodes = dag.nodes();
    const TraceNode* node_rms = nullptr;
    const TraceNode* node_add = nullptr;
    const TraceNode* node_res = nullptr;
    const TraceNode* node_gamma = nullptr;

    for (const auto& n : nodes) {
        if (n.op != TraceOpKind::RMSNorm || n.inputs.empty()) continue;
        node_rms = &n;
        node_add = &nodes[static_cast<std::size_t>(n.inputs[0])];
        if (node_add->inputs.size() < 2) return nullptr;
        node_res = &nodes[static_cast<std::size_t>(node_add->inputs[1])];
        if (n.inputs.size() > 1) node_gamma = &nodes[static_cast<std::size_t>(n.inputs[1])];
        break;
    }
    if (!node_rms || !node_add || !node_res) return nullptr;

    const int B = node_add->rows;
    const int D = node_add->cols;
    const float eps = node_rms->scalar > 0.0f ? node_rms->scalar : 1e-5f;

    auto bind = [B, D, eps](void* x, const void* res, const void* gamma, void* y) {
        auto h = std::make_shared<TraceHandleImpl>();
        h->is_cuda = true;
        h->launch_count = 1;
        h->fusion_name = "residual-rmsnorm";
        h->execute_fn = [x, res, gamma, y, B, D, eps]() {
            detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
                static_cast<float*>(x), static_cast<const float*>(res),
                static_cast<const float*>(gamma), static_cast<float*>(y), B, D, eps,
                cuda_current_stream());
        };
        return h;
    };

    auto h = bind(node_add->buffer, node_res->buffer,
                  node_gamma ? node_gamma->buffer : nullptr, node_rms->buffer);
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        const void* res = in_b.size() > 1 ? in_b[1] : nullptr;
        const void* gamma = in_b.size() > 2 ? in_b[2] : nullptr;
        void* x = !out_b.empty() ? out_b[0] : in_b[0];
        void* y = out_b.size() > 1 ? out_b[1] : (!out_b.empty() ? out_b[0] : nullptr);
        return bind(x, res, gamma, y);
    };
    return h;
}

}  // namespace

std::shared_ptr<TraceHandleImpl> compile_cuda(const TraceDAG& dag, FusionPattern pattern) {
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<TraceHandleImpl> h;

    // `x += p; rms_norm(x)` at FP32 keeps brass's hand-written kernel, which
    // is a single-pass form the generic emitter does not match. BROTENSOR_JIT_
    // PREFER=generic forces the emitted kernel instead, which is how the
    // benchmark puts the two side by side.
    static const bool prefer_generic = []() {
        const char* v = std::getenv("BROTENSOR_JIT_PREFER");
        return v && std::string(v) == "generic";
    }();
    if (!prefer_generic && pattern == FusionPattern::ResidualRMSNorm && all_fp32(dag)) {
        h = compile_residual_rmsnorm(dag);
    }

    if (!h) {
        ptx::RowNormPlan rn;
        if (build_row_norm_plan(dag, rn)) {
            h = compile_row_norm(dag, rn);
        }
    }

    if (!h) {
        ptx::ElementwisePlan ew;
        if (!build_plan(dag, ew)) {
            throw std::runtime_error(
                "brotensor::jit: the traced expression has no CUDA fusion — check that "
                "every operand is FP32/FP16/BF16 and is either the full (rows, cols) "
                "shape, a (1, cols) row, or a (1, 1) scalar");
        }
        h = compile_elementwise(dag, ew);
    }

    h->compile_us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
    return h;
}

namespace {
struct CudaTraceRegistrar {
    CudaTraceRegistrar() { register_cuda_trace_compiler(&compile_cuda); }
};
CudaTraceRegistrar s_cuda_trace_registrar;
}  // namespace

}  // namespace brotensor::jit::cuda

#endif  // BROTENSOR_HAS_CUDA
