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

// A trace is a row-norm when exactly one node is a norm, that node reads a
// traced input directly, and every other node is elementwise. The residual
// forms (`x += p; rms_norm(x)`) keep the hand-written kernel — the generic
// emitter would have to replay the pre-norm chain in both passes.
bool build_row_norm_plan(const TraceDAG& dag, ptx::RowNormPlan& plan) {
    const auto& nodes = dag.nodes();
    const TraceNode* norm = nullptr;
    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm) {
            if (norm) return false;  // more than one reduction
            norm = &n;
        }
    }
    if (!norm) return false;
    if (norm->inputs.empty()) return false;
    if (nodes[static_cast<std::size_t>(norm->inputs[0])].op != TraceOpKind::Input) return false;

    if (!build_plan(dag, plan.ew)) return false;
    if (plan.ew.cols != norm->cols) return false;

    auto input_index = [&](int node_id) {
        for (std::size_t i = 0; i < plan.ew.inputs.size(); ++i) {
            if (plan.ew.inputs[i].node_id == node_id) return static_cast<int>(i);
        }
        return -1;
    };
    plan.x_input = input_index(norm->inputs[0]);
    if (plan.x_input < 0) return false;
    if (norm->inputs.size() > 1) {
        plan.gamma_input = input_index(norm->inputs[1]);
        if (plan.gamma_input < 0) return false;
    }
    if (norm->inputs.size() > 2) {
        plan.beta_input = input_index(norm->inputs[2]);
        if (plan.beta_input < 0) return false;
    }
    plan.reduce = (norm->op == TraceOpKind::LayerNorm) ? ptx::RowReduce::Mean
                                                       : ptx::RowReduce::RMS;
    plan.norm_node = norm->id;
    plan.eps = norm->scalar > 0.0f ? norm->scalar : 1e-5f;
    // The row kernel walks one element per iteration; the vector entry is not
    // emitted for it.
    plan.ew.vec = 1;
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

    CUfunction fn = CudaJitEngine::instance().get_function(key, ptx_src, ptx::kEntryRowNorm);

    const int rows = plan.ew.rows;
    const bool is_ln = (plan.reduce == ptx::RowReduce::Mean);

    auto bind = [fn, rows, is_ln](const std::vector<void*>& ins,
                                  const std::vector<void*>& outs) {
        auto args = std::make_shared<LaunchArgs>();
        args->build(outs, ins, 0, /*with_count=*/false);

        unsigned grid = static_cast<unsigned>(rows);
        if (grid == 0) grid = 1;

        auto h = std::make_shared<TraceHandleImpl>();
        h->is_cuda = true;
        h->launch_count = 1;
        h->fusion_name = is_ln ? "row-layernorm-chain" : "row-rmsnorm-chain";
        h->execute_fn = [fn, grid, args]() {
            launch(fn, grid, static_cast<unsigned>(ptx::kRowThreads), *args, "trace row-norm");
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

    // `x += p; rms_norm(x)` keeps brass's hand-written kernel: it fuses the
    // residual write into the reduction's first pass, which the generic row
    // emitter cannot express. FP32 only — the kernel takes float*.
    if (pattern == FusionPattern::ResidualRMSNorm && all_fp32(dag)) {
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
