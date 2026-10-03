// ─── Trace JIT compiler for Vulkan ──────────────────────────────────────────
//
// The CUDA compiler's two fusion plans (trace_plan.h), emitted as brass MIR
// (spirv_emit*.cpp), lowered by brass's SPIR-V target and dispatched on the
// device's stream through vulkan_jit.h. A trace becomes one dispatch:
//
//   1. a row-norm plan (one RMSNorm / LayerNorm plus its elementwise chain),
//   2. else an elementwise plan,
//   3. else nullptr — the caller (TraceCompiler::compile_and_cache) replays
//      the DAG op by op (trace_eager.cpp), which is the Vulkan analogue of
//      the CUDA compiler's "no fusion" throw: the CUDA caller has no eager
//      replay and takes its own eager path, Vulkan has one and uses it.
//
// Each plan compiles to a vector entry (16-byte accesses) and a scalar
// entry; bind time picks the vector one when every bound pointer is aligned
// for it, so a misaligned row view still runs. Bind also packs the push-
// constant block once, so a replay is one vkCmdBindPipeline +
// vkCmdPushConstants + vkCmdDispatch with nothing allocated.
//
// A kernel the device cannot run (a SPIR-V capability it lacks) or any other
// compile failure falls back to the eager replay too, with the reason on
// stderr once per process; BROTENSOR_JIT_STRICT=1 rethrows instead (tests).
// BROTENSOR_JIT_VULKAN=eager disables the compiler (benchmarks compare the
// two in one process; it is read on every compile, which is rare).

#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"
#include "trace_plan.h"
#include "spirv_emit.h"

#include "../vulkan/vulkan_jit.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::jit::vulkan_trace {

namespace dv = ::brotensor::detail::vulkan;

namespace {

// Push block for one bound launch; the layout comes from the kernel's
// SpirvKernel::params (outputs, inputs, then the slot count for elementwise).
struct Push {
    std::array<std::uint8_t, 256> bytes{};
};

bool aligned(void* p, int bytes) {
    return (reinterpret_cast<std::uintptr_t>(p) % static_cast<std::uintptr_t>(bytes)) == 0;
}

void put(Push& pb, const brass::target::SpirvParam& prm, std::uint64_t v) {
    std::memcpy(pb.bytes.data() + prm.offset, &v, prm.size);   // little-endian
}

std::shared_ptr<Push> pack(const brass::target::SpirvKernel& k, const std::vector<void*>& outs,
                           const std::vector<void*>& ins, const std::uint32_t* count) {
    auto pb = std::make_shared<Push>();
    std::size_t i = 0;
    for (void* p : outs) put(*pb, k.params.at(i++), dv::addr(p));
    for (void* p : ins) put(*pb, k.params.at(i++), dv::addr(p));
    if (count) put(*pb, k.params.at(i++), *count);
    if (i != k.params.size()) throw std::logic_error("brotensor::jit: vulkan push block does not match the kernel");
    return pb;
}

// Per-buffer access width of the vector entry, for the alignment test.
void access_widths(const plan::ElementwisePlan& ew, int lanes, std::vector<int>& in_w,
                   std::vector<int>& out_w) {
    in_w.resize(ew.inputs.size());
    out_w.resize(ew.outputs.size());
    for (std::size_t i = 0; i < ew.inputs.size(); ++i) {
        in_w[i] = plan::elem_bytes(ew.inputs[i].dtype) *
                  (ew.inputs[i].bcast == BroadcastKind::Scalar ? 1 : lanes);
    }
    for (std::size_t i = 0; i < ew.outputs.size(); ++i) {
        out_w[i] = plan::elem_bytes(ew.outputs[i].dtype) * lanes;
    }
}

bool all_aligned(const std::vector<void*>& ins, const std::vector<void*>& outs,
                 const std::vector<int>& in_w, const std::vector<int>& out_w) {
    for (std::size_t i = 0; i < ins.size(); ++i) {
        if (!aligned(ins[i], in_w[i])) return false;
    }
    for (std::size_t i = 0; i < outs.size(); ++i) {
        if (!aligned(outs[i], out_w[i])) return false;
    }
    return true;
}

// The compiled pair for one plan, shared by every rebound handle.
struct Compiled {
    dv::DeviceCtx* ctx = nullptr;
    std::optional<brass::target::SpirvKernel> vec;
    brass::target::SpirvKernel scalar;
    dv::JitPipeline vec_pipe, scalar_pipe;
};

std::shared_ptr<TraceHandleImpl> make_handle(std::shared_ptr<const Compiled> cc, bool use_vec,
                                             std::shared_ptr<Push> push, std::uint32_t gx,
                                             std::uint32_t gy, const char* name) {
    auto h = std::make_shared<TraceHandleImpl>();
    h->launch_count = 1;
    h->fusion_name = name;
    const dv::JitPipeline pipe = use_vec ? cc->vec_pipe : cc->scalar_pipe;
    dv::DeviceCtx* ctx = cc->ctx;
    h->execute_fn = [cc, ctx, pipe, push, gx, gy]() {
        dv::jit_dispatch(*ctx, pipe, push->bytes.data(), gx, gy);
    };
    return h;
}

std::vector<void*> buffers(const TraceDAG& dag, const std::vector<plan::BufferSpec>& specs) {
    std::vector<void*> out;
    for (const auto& s : specs) out.push_back(dag.node(s.node_id).buffer);
    return out;
}

// ── elementwise ─────────────────────────────────────────────────────────────

std::shared_ptr<TraceHandleImpl> compile_elementwise(const TraceDAG& dag, const plan::ElementwisePlan& ew,
                                                     std::shared_ptr<Compiled> cc) {
    spirv::Kernels k = spirv::emit_elementwise(dag, ew);
    cc->vec = std::move(k.vec);
    cc->scalar = std::move(k.scalar);
    if (cc->vec) cc->vec_pipe = dv::jit_pipeline(*cc->ctx, *cc->vec, spirv::kEwBlock);
    cc->scalar_pipe = dv::jit_pipeline(*cc->ctx, cc->scalar, spirv::kEwBlock);

    const int lanes = ew.vec;
    const std::int64_t numel = ew.numel;
    std::vector<int> in_w, out_w;
    access_widths(ew, lanes, in_w, out_w);
    std::shared_ptr<const Compiled> ccc = cc;

    auto bind = [ccc, lanes, numel, in_w, out_w](const std::vector<void*>& ins,
                                                 const std::vector<void*>& outs) {
        const bool use_vec = ccc->vec.has_value() && all_aligned(ins, outs, in_w, out_w);
        const std::uint32_t slots = static_cast<std::uint32_t>(use_vec ? numel / lanes : numel);
        std::uint64_t grid = (slots + spirv::kEwBlock - 1) / spirv::kEwBlock;
        if (grid == 0) grid = 1;
        if (grid > 65535) grid = 65535;   // grid-stride loop covers the rest
        auto push = pack(use_vec ? *ccc->vec : ccc->scalar, outs, ins, &slots);
        return make_handle(ccc, use_vec, push, static_cast<std::uint32_t>(grid), 1,
                           use_vec ? "elementwise-vec" : "elementwise-scalar");
    };

    auto h = bind(buffers(dag, ew.inputs), buffers(dag, ew.outputs));
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return bind(in_b, out_b);
    };
    return h;
}

// ── row-norm ────────────────────────────────────────────────────────────────

std::shared_ptr<TraceHandleImpl> compile_row_norm(const TraceDAG& dag, const plan::RowNormPlan& rn,
                                                  std::shared_ptr<Compiled> cc) {
    const plan::ElementwisePlan& ew = rn.ew;
    spirv::Kernels k = spirv::emit_row_norm(dag, rn);
    cc->vec = std::move(k.vec);
    cc->scalar = std::move(k.scalar);

    const int rows = ew.rows, cols = ew.cols, lanes = ew.vec;
    int tpr = 0, rpb = 0;
    if (cc->vec) {
        plan::row_norm_geometry(cols, rows, lanes, tpr, rpb);
        cc->vec_pipe = dv::jit_pipeline(*cc->ctx, *cc->vec, static_cast<std::uint32_t>(tpr * rpb));
    }
    plan::row_norm_geometry(cols, rows, 1, tpr, rpb);
    cc->scalar_pipe = dv::jit_pipeline(*cc->ctx, cc->scalar, static_cast<std::uint32_t>(tpr * rpb));

    const bool is_ln = (rn.reduce == plan::RowReduce::Mean);
    std::vector<int> in_w, out_w;
    access_widths(ew, lanes, in_w, out_w);
    std::shared_ptr<const Compiled> ccc = cc;

    auto bind = [ccc, rows, cols, lanes, is_ln, in_w, out_w](const std::vector<void*>& ins,
                                                             const std::vector<void*>& outs) {
        const bool use_vec = ccc->vec.has_value() && all_aligned(ins, outs, in_w, out_w);
        // The entry decides the tiling: the scalar entry covers a row with
        // more threads, so its geometry is its own.
        int t = 0, r = 0;
        plan::row_norm_geometry(cols, rows, use_vec ? lanes : 1, t, r);
        const std::uint32_t groups = static_cast<std::uint32_t>((rows + r - 1) / r);
        const std::uint32_t gx = groups < 65535u ? groups : 65535u;
        const std::uint32_t gy = (groups + gx - 1) / gx;
        auto push = pack(use_vec ? *ccc->vec : ccc->scalar, outs, ins, nullptr);
        const char* name = is_ln ? (use_vec ? "row-layernorm-chain" : "row-layernorm-chain-scalar")
                                 : (use_vec ? "row-rmsnorm-chain" : "row-rmsnorm-chain-scalar");
        return make_handle(ccc, use_vec, push, gx, gy, name);
    };

    auto h = bind(buffers(dag, ew.inputs), buffers(dag, ew.outputs));
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return bind(in_b, out_b);
    };
    return h;
}

// What the kernels can address: one Vulkan device, 32-bit element indices,
// and a push block within the pipeline layout.
dv::DeviceCtx* device_for(const TraceDAG& dag) {
    if (dag.nodes().empty()) return nullptr;
    const Device d = dag.nodes().front().device;
    if (!d.is_vulkan()) return nullptr;
    for (const auto& n : dag.nodes()) {
        if (n.device != d) return nullptr;
    }
    return &dv::device(d.index);
}

bool addressable(const plan::ElementwisePlan& ew, dv::DeviceCtx& ctx) {
    if (ew.numel <= 0 || ew.numel >= (std::int64_t(1) << 31)) return false;
    const std::size_t push = 8 * (ew.inputs.size() + ew.outputs.size()) + 4;
    return push <= ctx.pipelines().push_bytes();
}

std::shared_ptr<TraceHandleImpl> compile_plans(const TraceDAG& dag, dv::DeviceCtx& ctx) {
    if (!spirv::ops_supported(dag)) return nullptr;
    auto cc = std::make_shared<Compiled>();
    cc->ctx = &ctx;
    plan::RowNormPlan rn;
    if (plan::build_row_norm_plan(dag, rn)) {
        return addressable(rn.ew, ctx) ? compile_row_norm(dag, rn, cc) : nullptr;
    }
    for (const auto& n : dag.nodes()) {
        if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm) return nullptr;
    }
    plan::ElementwisePlan ew;
    if (!plan::build_plan(dag, ew) || !addressable(ew, ctx)) return nullptr;
    return compile_elementwise(dag, ew, cc);
}

bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

}  // namespace

std::shared_ptr<TraceHandleImpl> compile_vulkan(const TraceDAG& dag, FusionPattern /*pattern*/) {
    if (env_is("BROTENSOR_JIT_VULKAN", "eager")) return nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<TraceHandleImpl> h;
    try {
        dv::DeviceCtx* ctx = device_for(dag);
        if (!ctx) return nullptr;
        h = compile_plans(dag, *ctx);
    } catch (const std::exception& e) {
        if (env_is("BROTENSOR_JIT_STRICT", "1")) throw;
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::fprintf(stderr,
                         "brotensor: the Vulkan trace compiler failed (%s); replaying traces op by op\n",
                         e.what());
        }
        return nullptr;
    }
    if (h) {
        h->compile_us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - t0).count();
    }
    return h;
}

namespace {
struct VulkanTraceRegistrar {
    VulkanTraceRegistrar() { register_vulkan_trace_compiler(&compile_vulkan); }
};
VulkanTraceRegistrar s_vulkan_trace_registrar;
}  // namespace

}  // namespace brotensor::jit::vulkan_trace
