// Metal compute backend for the trace JIT (BROTENSOR_WITH_METAL &&
// BROTENSOR_HAS_BRASS_JIT). Lowers the shared Brass GPU MIR elementwise and
// row-norm kernels via MslTarget, caches the MSL artifacts on disk, and
// compiles/dispatches them via Metal compute pipelines.

#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"
#include "trace_plan.h"
#include "msl_emit.h"
#include "kernel_disk_cache.h"

#if BROTENSOR_HAS_METAL

#import "../metal/internal.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace brotensor::jit::metal_trace {

namespace {

bool aligned(void* p, int bytes) {
    return (reinterpret_cast<std::uintptr_t>(p) % static_cast<std::uintptr_t>(bytes)) == 0;
}

void access_widths(const plan::ElementwisePlan& ew, int lanes,
                   std::vector<int>& in_w, std::vector<int>& out_w) {
    in_w.resize(ew.inputs.size());
    for (std::size_t i = 0; i < ew.inputs.size(); ++i) {
        in_w[i] = (ew.inputs[i].bcast == BroadcastKind::Scalar)
                      ? plan::elem_bytes(ew.inputs[i].dtype)
                      : lanes * plan::elem_bytes(ew.inputs[i].dtype);
    }
    out_w.resize(ew.outputs.size());
    for (std::size_t i = 0; i < ew.outputs.size(); ++i) {
        out_w[i] = lanes * plan::elem_bytes(ew.outputs[i].dtype);
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

struct Compiled {
    std::optional<brass::target::MslKernel> vec;
    brass::target::MslKernel scalar;
    id<MTLComputePipelineState> vec_pso = nil;
    id<MTLComputePipelineState> scalar_pso = nil;
};

id<MTLComputePipelineState> build_pso(const brass::target::MslKernel& k) {
    NSString* src = [[NSString alloc] initWithBytes:k.source.data()
                                              length:k.source.size()
                                            encoding:NSUTF8StringEncoding];
    NSString* entry = [[NSString alloc] initWithBytes:k.entry_name.data()
                                                length:k.entry_name.size()
                                              encoding:NSUTF8StringEncoding];
    return metal_impl::compile_pipeline(src, entry);
}

std::vector<uint8_t> pack_scalar_count(const brass::target::MslKernel& k,
                                       const std::uint32_t* count) {
    if (k.scalar_params_bytes == 0) return {};
    std::vector<uint8_t> bytes(k.scalar_params_bytes, 0);
    if (count && !k.scalar_param_offsets.empty()) {
        std::memcpy(bytes.data() + k.scalar_param_offsets[0], count, sizeof(std::uint32_t));
    }
    return bytes;
}

std::shared_ptr<TraceHandleImpl> make_handle(std::shared_ptr<const Compiled> cc,
                                             bool use_vec,
                                             std::vector<void*> outs,
                                             std::vector<void*> ins,
                                             std::vector<uint8_t> scalar_bytes,
                                             std::uint32_t gx,
                                             std::uint32_t gy,
                                             std::uint32_t block_threads,
                                             const char* fusion_name) {
    auto h = std::make_shared<TraceHandleImpl>();
    h->is_cuda = false;
    h->launch_count = 1;
    h->fusion_name = fusion_name;

    id<MTLComputePipelineState> pso = use_vec ? cc->vec_pso : cc->scalar_pso;
    const NSUInteger scalar_buf_idx = static_cast<NSUInteger>(outs.size() + ins.size());
    std::vector<void*> all_ptrs;
    all_ptrs.reserve(outs.size() + ins.size());
    all_ptrs.insert(all_ptrs.end(), outs.begin(), outs.end());
    all_ptrs.insert(all_ptrs.end(), ins.begin(), ins.end());

    h->execute_fn = [cc, pso, all_ptrs = std::move(all_ptrs),
                     scalar_bytes = std::move(scalar_bytes),
                     scalar_buf_idx, gx, gy, block_threads]() {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = metal_impl::new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            for (NSUInteger i = 0; i < all_ptrs.size(); ++i) {
                void* ptr = all_ptrs[i];
                id<MTLBuffer> buf = metal_impl::pool_lookup(ptr);
                const NSUInteger off = metal_impl::pool_lookup_offset(ptr);
                [enc setBuffer:buf offset:off atIndex:i];
            }
            if (!scalar_bytes.empty()) {
                [enc setBytes:scalar_bytes.data()
                       length:scalar_bytes.size()
                      atIndex:scalar_buf_idx];
            }
            const NSUInteger tg = std::min<NSUInteger>(
                std::max<NSUInteger>(1, block_threads), pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            metal_impl::submit(cmd);
        }
    };
    return h;
}

std::vector<void*> buffers(const TraceDAG& dag, const std::vector<plan::BufferSpec>& specs) {
    std::vector<void*> out;
    out.reserve(specs.size());
    for (const auto& s : specs) out.push_back(dag.node(s.node_id).buffer);
    return out;
}

template <typename EmitFn>
msl::Kernels emit_or_load_msl(const char* tag, const TraceDAG& dag, int lanes, EmitFn&& emit_fn) {
    auto& cache = KernelDiskCache::instance();
    const std::string base = std::string(tag) + "/v" + std::to_string(lanes) + "/" + dag.key_material();
    msl::Kernels k;
    if (lanes > 1) {
        brass::target::MslKernel vk;
        if (cache.load_msl_kernel(base + "/s", k.scalar) && cache.load_msl_kernel(base + "/v", vk)) {
            k.vec = std::move(vk);
            return k;
        }
    } else if (cache.load_msl_kernel(base + "/s", k.scalar)) {
        return k;
    }
    k = emit_fn();
    cache.store_msl_kernel(base + "/s", k.scalar);
    if (k.vec) cache.store_msl_kernel(base + "/v", *k.vec);
    return k;
}

std::shared_ptr<TraceHandleImpl> compile_elementwise(const TraceDAG& dag,
                                                     const plan::ElementwisePlan& ew,
                                                     std::shared_ptr<Compiled> cc) {
    msl::Kernels k = emit_or_load_msl("mtl_trace_ew", dag, ew.vec, [&] {
        return msl::emit_elementwise(dag, ew);
    });
    cc->vec = std::move(k.vec);
    cc->scalar = std::move(k.scalar);
    if (cc->vec) cc->vec_pso = build_pso(*cc->vec);
    cc->scalar_pso = build_pso(cc->scalar);

    const int lanes = ew.vec;
    const std::int64_t numel = ew.numel;
    std::vector<int> in_w, out_w;
    access_widths(ew, lanes, in_w, out_w);
    std::shared_ptr<const Compiled> ccc = cc;

    auto bind = [ccc, lanes, numel, in_w, out_w](const std::vector<void*>& ins,
                                                 const std::vector<void*>& outs) {
        const bool use_vec = ccc->vec.has_value() && all_aligned(ins, outs, in_w, out_w);
        const std::uint32_t slots = static_cast<std::uint32_t>(use_vec ? numel / lanes : numel);
        std::uint64_t grid = (slots + msl::kEwBlock - 1) / msl::kEwBlock;
        if (grid == 0) grid = 1;
        if (grid > 65535) grid = 65535;
        const auto& kern = use_vec ? *ccc->vec : ccc->scalar;
        auto scalar_bytes = pack_scalar_count(kern, &slots);
        return make_handle(ccc, use_vec, outs, ins, std::move(scalar_bytes),
                           static_cast<std::uint32_t>(grid), 1, msl::kEwBlock,
                           use_vec ? "elementwise-vec" : "elementwise-scalar");
    };

    auto h = bind(buffers(dag, ew.inputs), buffers(dag, ew.outputs));
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return bind(in_b, out_b);
    };
    return h;
}

std::shared_ptr<TraceHandleImpl> compile_row_norm(const TraceDAG& dag,
                                                  const plan::RowNormPlan& rn,
                                                  std::shared_ptr<Compiled> cc) {
    const plan::ElementwisePlan& ew = rn.ew;
    msl::Kernels k = emit_or_load_msl("mtl_trace_rn", dag, ew.vec, [&] {
        return msl::emit_row_norm(dag, rn);
    });
    cc->vec = std::move(k.vec);
    cc->scalar = std::move(k.scalar);
    if (cc->vec) cc->vec_pso = build_pso(*cc->vec);
    cc->scalar_pso = build_pso(cc->scalar);

    const int rows = ew.rows, cols = ew.cols, lanes = ew.vec;
    const bool is_ln = (rn.reduce == plan::RowReduce::Mean);
    std::vector<int> in_w, out_w;
    access_widths(ew, lanes, in_w, out_w);
    std::shared_ptr<const Compiled> ccc = cc;

    auto bind = [ccc, rows, cols, lanes, is_ln, in_w, out_w](const std::vector<void*>& ins,
                                                             const std::vector<void*>& outs) {
        const bool use_vec = ccc->vec.has_value() && all_aligned(ins, outs, in_w, out_w);
        int t = 0, r = 0;
        plan::row_norm_geometry(cols, rows, use_vec ? lanes : 1, t, r);
        const std::uint32_t groups = static_cast<std::uint32_t>((rows + r - 1) / r);
        const std::uint32_t gx = groups < 65535u ? groups : 65535u;
        const std::uint32_t gy = (groups + gx - 1) / gx;
        const char* name = is_ln ? (use_vec ? "row-layernorm-chain" : "row-layernorm-chain-scalar")
                                 : (use_vec ? "row-rmsnorm-chain" : "row-rmsnorm-chain-scalar");
        return make_handle(ccc, use_vec, outs, ins, {}, gx, gy,
                           static_cast<std::uint32_t>(t * r), name);
    };

    auto h = bind(buffers(dag, ew.inputs), buffers(dag, ew.outputs));
    h->rebind_fn = [bind](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return bind(in_b, out_b);
    };
    return h;
}

bool addressable(const plan::ElementwisePlan& ew) {
    if (ew.numel <= 0 || ew.numel >= (std::int64_t(1) << 31)) return false;
    // Metal supports up to 31 buffer arguments; leave room for the scalar _Params buffer.
    return (ew.inputs.size() + ew.outputs.size()) < 30;
}

std::shared_ptr<TraceHandleImpl> compile_plans(const TraceDAG& dag) {
    if (dag.nodes().empty()) return nullptr;
    const Device d = dag.nodes().front().device;
    if (!d.is_metal()) return nullptr;
    for (const auto& n : dag.nodes()) {
        if (n.device != d) return nullptr;
    }
    if (!msl::ops_supported(dag)) return nullptr;

    auto cc = std::make_shared<Compiled>();
    plan::RowNormPlan rn;
    if (plan::build_row_norm_plan(dag, rn)) {
        return addressable(rn.ew) ? compile_row_norm(dag, rn, cc) : nullptr;
    }
    for (const auto& n : dag.nodes()) {
        if (n.op == TraceOpKind::RMSNorm || n.op == TraceOpKind::LayerNorm) return nullptr;
    }
    plan::ElementwisePlan ew;
    if (!plan::build_plan(dag, ew) || !addressable(ew)) return nullptr;
    return compile_elementwise(dag, ew, cc);
}

bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

}  // namespace

std::shared_ptr<TraceHandleImpl> compile_metal(const TraceDAG& dag, FusionPattern /*pattern*/) {
    if (env_is("BROTENSOR_JIT_METAL", "eager")) return nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<TraceHandleImpl> h;
    try {
        h = compile_plans(dag);
    } catch (const std::exception& e) {
        if (env_is("BROTENSOR_JIT_STRICT", "1")) throw;
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::fprintf(stderr,
                         "brotensor: the Metal trace compiler failed (%s); replaying traces op by op\n",
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

void register_metal_trace_jit() {
    register_metal_trace_compiler(&compile_metal);
}

namespace {
struct MetalTraceRegistrar {
    MetalTraceRegistrar() { register_metal_trace_jit(); }
};
MetalTraceRegistrar s_metal_trace_registrar;
}  // namespace

}  // namespace brotensor::jit::metal_trace

#endif // BROTENSOR_HAS_METAL
