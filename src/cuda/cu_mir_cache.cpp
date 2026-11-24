#include "cu_mir_cache.h"

#include "cuda_driver.h"
#include "cuda_jit_engine.h"
#include "../kernels/mir_elementwise.h"

#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/mir/module.hpp>
#include <brass/target/ptx_target.hpp>
#endif

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brotensor::detail::cuda {

namespace {

struct MirKeyHash {
    std::size_t operator()(const MirKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.kind);
        h = (h * 1315423911u) ^ k.op;
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.dt0) << 8) ^ static_cast<std::size_t>(k.dt1);
        h = (h * 1315423911u) ^ k.lanes;
        return h;
    }
};

struct MirCache {
    std::mutex mu;
    std::unordered_map<MirKey, CUfunction, MirKeyHash> map;
};

MirCache& mir_cache() {
    static auto* c = new MirCache();
    return *c;
}

const char* entry_name_for(MirKind kind) {
    switch (kind) {
        case MirKind::Unary:  return "mir_unary";
        case MirKind::Binary: return "mir_binary";
        case MirKind::ActBwd: return "mir_act_bwd";
        case MirKind::Bias:   return "mir_bias";
        case MirKind::Cast:   return "mir_cast";
        case MirKind::Glu:    return "mir_glu";
    }
    throw std::runtime_error("cu_mir_cache: unknown MirKind");
}

std::string key_string_for(const MirKey& k) {
    return "cu_mir_" + std::to_string(static_cast<unsigned>(k.kind)) + "_" +
           std::to_string(k.op) + "_" +
           std::to_string(static_cast<unsigned>(k.dt0)) + "_" +
           std::to_string(static_cast<unsigned>(k.dt1)) + "_v" +
           std::to_string(static_cast<unsigned>(k.lanes));
}

CUfunction compile_mir(const MirKey& k) {
#if !BROTENSOR_HAS_BRASS_CUDA_JIT
    (void)k;
    throw std::runtime_error("brotensor: Brass CUDA JIT is required for CUDA elementwise MIR kernels");
#else
    brass::Module mod("cu_mir");
    brass::Function* fn = nullptr;
    switch (k.kind) {
        case MirKind::Unary:  fn = kernels::build_unary_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Binary: fn = kernels::build_binary_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::ActBwd: fn = kernels::build_act_bwd_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Bias:   fn = kernels::build_bias_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Cast:   fn = kernels::build_cast_mir(mod, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Glu:    fn = kernels::build_glu_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
    }
    const std::string ptx = brass::target::PtxTarget::emit_function(*fn, jit::make_ptx_opts());
    CUfunction cu_fn = jit::CudaJitEngine::instance().get_function(
        key_string_for(k), ptx, entry_name_for(k.kind));

    MirCache& cache = mir_cache();
    std::lock_guard<std::mutex> lk(cache.mu);
    auto [it, inserted] = cache.map.emplace(k, cu_fn);
    (void)inserted;
    return it->second;
#endif
}

void launch_kernel(CUfunction fn, std::uint32_t gx, void** params, const char* what) {
    const CUstream stream = jit::resolve_stream(nullptr);
    const CUresult res = drv::cuLaunchKernel(
        fn, gx, 1, 1, kernels::kEwBlock, 1, 1, 0, stream, params, nullptr);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("cuLaunchKernel failed for ") + what);
    }
}

}  // namespace

CUfunction mir_function(const MirKey& key) {
    {
        MirCache& cache = mir_cache();
        std::lock_guard<std::mutex> lk(cache.mu);
        if (auto it = cache.map.find(key); it != cache.map.end()) return it->second;
    }
    return compile_mir(key);
}

int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uintptr_t a0, std::uintptr_t a1, std::uintptr_t a2) {
    const std::uintptr_t m = a0 | a1 | a2;
    if (dt == Dtype::FP32) {
        return (n % 4u == 0u && (m % 16u == 0u)) ? 4 : 1;
    }
    if (n % 8u == 0u && (m % 16u == 0u)) return 8;
    if (n % 4u == 0u && (m % 8u == 0u)) return 4;
    return 1;
}

int mir_vec_lanes_mixed(Dtype dt0, Dtype dt1, std::uint32_t n, std::uintptr_t a0, std::uintptr_t a1) {
    if (dt0 == dt1) return mir_vec_lanes(dt0, n, a0, a1);
    const std::uintptr_t m = a0 | a1;
    if (dt0 != Dtype::FP32 && dt1 != Dtype::FP32) {
        if (n % 8u == 0u && (m % 16u == 0u)) return 8;
        if (n % 4u == 0u && (m % 8u == 0u)) return 4;
        return 1;
    }
    const std::uint32_t r0 = (dt0 == Dtype::FP32) ? 16u : 8u;
    const std::uint32_t r1 = (dt1 == Dtype::FP32) ? 16u : 8u;
    if (n % 4u == 0u && (a0 % r0 == 0u) && (a1 % r1 == 0u) && (m % 8u == 0u)) return 4;
    return 1;
}

void mir_dispatch_5(CUfunction fn, std::uint32_t gx,
                    const void* p0, const void* p1, std::uint32_t p2, float p3, float p4) {
    CUdeviceptr a0 = reinterpret_cast<CUdeviceptr>(p0);
    CUdeviceptr a1 = reinterpret_cast<CUdeviceptr>(p1);
    std::int32_t a2 = static_cast<std::int32_t>(p2);
    float a3 = p3;
    float a4 = p4;
    void* params[] = {&a0, &a1, &a2, &a3, &a4};
    launch_kernel(fn, gx, params, "mir_unary/mir_binary");
}

void mir_dispatch_act_bwd(CUfunction fn, std::uint32_t gx,
                          const void* p, const void* dy, const void* dx, std::uint32_t n, float a) {
    CUdeviceptr ap = reinterpret_cast<CUdeviceptr>(p);
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr adx = reinterpret_cast<CUdeviceptr>(dx);
    std::int32_t an = static_cast<std::int32_t>(n);
    float aa = a;
    void* params[] = {&ap, &ady, &adx, &an, &aa};
    launch_kernel(fn, gx, params, "mir_act_bwd");
}

void mir_dispatch_bias(CUfunction fn, std::uint32_t gx,
                       const void* y, const void* bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr ab = reinterpret_cast<CUdeviceptr>(bias);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    std::int32_t al = static_cast<std::int32_t>(l);
    void* params[] = {&ay, &ab, &an, &adim, &al};
    launch_kernel(fn, gx, params, "mir_bias");
}

void mir_dispatch_cast(CUfunction fn, std::uint32_t gx,
                       const void* src, const void* dst, std::uint32_t n) {
    CUdeviceptr asrc = reinterpret_cast<CUdeviceptr>(src);
    CUdeviceptr adst = reinterpret_cast<CUdeviceptr>(dst);
    std::int32_t an = static_cast<std::int32_t>(n);
    void* params[] = {&asrc, &adst, &an};
    launch_kernel(fn, gx, params, "mir_cast");
}

void mir_dispatch_glu(CUfunction fn, std::uint32_t gx,
                      const void* x, const void* dy, const void* y, std::uint32_t n, std::uint32_t dim) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    void* params[] = {&ax, &ady, &ay, &an, &adim};
    launch_kernel(fn, gx, params, "mir_glu");
}

}  // namespace brotensor::detail::cuda
