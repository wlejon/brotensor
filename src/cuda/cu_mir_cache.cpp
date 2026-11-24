#include "cu_mir_cache.h"

#include "cuda_driver.h"
#include "cuda_jit_engine.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_norm.h"
#include "../kernels/mir_reduce.h"
#include "../kernels/mir_rope.h"

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
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.dt0) << 16) ^
            (static_cast<std::size_t>(k.dt1) << 8) ^ static_cast<std::size_t>(k.dt2);
        h = (h * 1315423911u) ^ (static_cast<std::size_t>(k.lanes) << 8) ^ k.flags;
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
        case MirKind::Unary:      return "mir_unary";
        case MirKind::Binary:     return "mir_binary";
        case MirKind::ActBwd:     return "mir_act_bwd";
        case MirKind::Bias:       return "mir_bias";
        case MirKind::Cast:       return "mir_cast";
        case MirKind::Glu:        return "mir_glu";
        case MirKind::Rowvec:     return "mir_rowvec";
        case MirKind::SumRows:    return "mir_sum_rows";
        case MirKind::ArgmaxRows: return "mir_argmax_rows";
        case MirKind::SumCols:    return "mir_sum_cols";
        case MirKind::Norm:       return "mir_norm";
        case MirKind::Rope:       return "mir_rope";
    }
    throw std::runtime_error("cu_mir_cache: unknown MirKind");
}

std::string key_string_for(const MirKey& k) {
    return "cu_mir_" + std::to_string(static_cast<unsigned>(k.kind)) + "_" +
           std::to_string(k.op) + "_" +
           std::to_string(static_cast<unsigned>(k.dt0)) + "_" +
           std::to_string(static_cast<unsigned>(k.dt1)) + "_" +
           std::to_string(static_cast<unsigned>(k.dt2)) + "_v" +
           std::to_string(static_cast<unsigned>(k.lanes)) + "_f" +
           std::to_string(static_cast<unsigned>(k.flags));
}

bool ptr_ok(std::uintptr_t p, Dtype dt, int lanes) {
    if (p == 0) return true;
    const std::uintptr_t req =
        static_cast<std::uintptr_t>(lanes) * (dt == Dtype::FP32 ? 4u : 2u);
    return (p % req) == 0u;
}

CUfunction compile_mir(const MirKey& k) {
#if !BROTENSOR_HAS_BRASS_CUDA_JIT
    (void)k;
    throw std::runtime_error("brotensor: Brass CUDA JIT is required for CUDA MIR kernels");
#else
    brass::Module mod("cu_mir");
    brass::Function* fn = nullptr;
    switch (k.kind) {
        case MirKind::Unary:      fn = kernels::build_unary_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Binary:     fn = kernels::build_binary_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::ActBwd:     fn = kernels::build_act_bwd_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Bias:       fn = kernels::build_bias_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::Cast:       fn = kernels::build_cast_mir(mod, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Glu:        fn = kernels::build_glu_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Rowvec:     fn = kernels::build_rowvec_mir(mod, k.op, k.dt0, k.lanes); break;
        case MirKind::SumRows:    fn = kernels::build_sum_rows_mir(mod, k.dt0, k.lanes); break;
        case MirKind::ArgmaxRows: fn = kernels::build_argmax_rows_mir(mod, k.dt0, k.flags != 0, k.lanes); break;
        case MirKind::SumCols:    fn = kernels::build_sum_cols_mir(mod, k.dt0, k.lanes); break;
        case MirKind::Norm:
            fn = kernels::build_norm_mir(
                mod, k.op, k.dt0, k.dt1, k.dt2,
                kernels::NormMirFlags::unpack(k.flags), k.lanes);
            break;
        case MirKind::Rope:       fn = kernels::build_rope_mir(mod, k.op, k.flags != 0, k.dt0); break;
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

void launch_kernel(CUfunction fn, std::uint32_t gx, std::uint32_t block_x, void** params, const char* what) {
    const CUstream stream = jit::resolve_stream(nullptr);
    const CUresult res = drv::cuLaunchKernel(
        fn, gx, 1, 1, block_x, 1, 1, 0, stream, params, nullptr);
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

int mir_vec_lanes(Dtype dt, std::uint32_t n, std::uintptr_t a0,
                  std::uintptr_t a1, std::uintptr_t a2, std::uintptr_t a3) {
    const std::uintptr_t m = a0 | a1 | a2 | a3;
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
    if (n % 4u == 0u && ptr_ok(a0, dt0, 4) && ptr_ok(a1, dt1, 4)) return 4;
    return 1;
}

int mir_vec_lanes_norm(Dtype dt_x, Dtype dt_y, Dtype dt_g, std::uint32_t d,
                       std::uintptr_t x, std::uintptr_t y, std::uintptr_t g,
                       std::uintptr_t b, std::uintptr_t xh, std::uintptr_t s0_mask) {
    const bool any_f32 = (dt_x == Dtype::FP32) || (dt_y == Dtype::FP32) ||
                         (g != 0 && dt_g == Dtype::FP32) ||
                         (b != 0 && dt_g == Dtype::FP32) || (s0_mask != 0);
    const int candidates[2] = {any_f32 ? 4 : 8, 4};
    for (int l : candidates) {
        if (d % static_cast<std::uint32_t>(l) != 0u) continue;
        if (ptr_ok(x, dt_x, l) && ptr_ok(y, dt_y, l) && ptr_ok(g, dt_g, l) &&
            ptr_ok(b, dt_g, l) && ptr_ok(xh, dt_x, l) &&
            ptr_ok(s0_mask, Dtype::FP32, l)) {
            return l;
        }
    }
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
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_unary/mir_binary");
}

void mir_dispatch_act_bwd(CUfunction fn, std::uint32_t gx,
                          const void* p, const void* dy, const void* dx, std::uint32_t n, float a) {
    CUdeviceptr ap = reinterpret_cast<CUdeviceptr>(p);
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr adx = reinterpret_cast<CUdeviceptr>(dx);
    std::int32_t an = static_cast<std::int32_t>(n);
    float aa = a;
    void* params[] = {&ap, &ady, &adx, &an, &aa};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_act_bwd");
}

void mir_dispatch_bias(CUfunction fn, std::uint32_t gx,
                       const void* y, const void* bias, std::uint32_t n, std::uint32_t dim, std::uint32_t l) {
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr ab = reinterpret_cast<CUdeviceptr>(bias);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    std::int32_t al = static_cast<std::int32_t>(l);
    void* params[] = {&ay, &ab, &an, &adim, &al};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_bias");
}

void mir_dispatch_cast(CUfunction fn, std::uint32_t gx,
                       const void* src, const void* dst, std::uint32_t n) {
    CUdeviceptr asrc = reinterpret_cast<CUdeviceptr>(src);
    CUdeviceptr adst = reinterpret_cast<CUdeviceptr>(dst);
    std::int32_t an = static_cast<std::int32_t>(n);
    void* params[] = {&asrc, &adst, &an};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_cast");
}

void mir_dispatch_glu(CUfunction fn, std::uint32_t gx,
                      const void* x, const void* dy, const void* y, std::uint32_t n, std::uint32_t dim) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    void* params[] = {&ax, &ady, &ay, &an, &adim};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_glu");
}

void mir_dispatch_rowvec(CUfunction fn, std::uint32_t gx,
                         const void* x, const void* v, const void* w, const void* y,
                         std::uint32_t n, std::uint32_t dim) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr av = reinterpret_cast<CUdeviceptr>(v);
    CUdeviceptr aw = reinterpret_cast<CUdeviceptr>(w);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    void* params[] = {&ax, &av, &aw, &ay, &an, &adim};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_rowvec");
}

void mir_dispatch_reduce(CUfunction fn, std::uint32_t gx,
                         const void* x, const void* y, std::uint32_t m, std::uint32_t n) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    std::int32_t am = static_cast<std::int32_t>(m);
    std::int32_t an = static_cast<std::int32_t>(n);
    void* params[] = {&ax, &ay, &am, &an};
    launch_kernel(fn, gx, kernels::kReduceBlock, params, "mir_reduce");
}

void mir_dispatch_norm(CUfunction fn, std::uint32_t gx,
                       const void* x, const void* y, const void* g, const void* b,
                       const void* xh, const void* s0, const void* s1,
                       std::uint32_t rows, std::uint32_t dim, float eps, float rstd) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr ag = reinterpret_cast<CUdeviceptr>(g);
    CUdeviceptr ab = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr axh = reinterpret_cast<CUdeviceptr>(xh);
    CUdeviceptr as0 = reinterpret_cast<CUdeviceptr>(s0);
    CUdeviceptr as1 = reinterpret_cast<CUdeviceptr>(s1);
    std::int32_t arows = static_cast<std::int32_t>(rows);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    float aeps = eps;
    float arstd = rstd;
    void* params[] = {&ax, &ay, &ag, &ab, &axh, &as0, &as1, &arows, &adim, &aeps, &arstd};
    launch_kernel(fn, gx, kernels::kNormBlock, params, "mir_norm");
}

void mir_dispatch_rope(CUfunction fn, std::uint32_t gx,
                       const void* x, const void* y, const void* ct, const void* st,
                       const void* ch, const void* sh, const void* cw, const void* sw,
                       const void* pt, const void* ph, const void* pw,
                       std::uint32_t rows, std::uint32_t heads, std::uint32_t hd,
                       std::uint32_t dt, std::uint32_t dh, std::uint32_t dw,
                       std::int32_t offset, float ln_base) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr act = reinterpret_cast<CUdeviceptr>(ct);
    CUdeviceptr ast = reinterpret_cast<CUdeviceptr>(st);
    CUdeviceptr ach = reinterpret_cast<CUdeviceptr>(ch);
    CUdeviceptr ash = reinterpret_cast<CUdeviceptr>(sh);
    CUdeviceptr acw = reinterpret_cast<CUdeviceptr>(cw);
    CUdeviceptr asw = reinterpret_cast<CUdeviceptr>(sw);
    CUdeviceptr apt = reinterpret_cast<CUdeviceptr>(pt);
    CUdeviceptr aph = reinterpret_cast<CUdeviceptr>(ph);
    CUdeviceptr apw = reinterpret_cast<CUdeviceptr>(pw);
    std::int32_t arows = static_cast<std::int32_t>(rows);
    std::int32_t aheads = static_cast<std::int32_t>(heads);
    std::int32_t ahd = static_cast<std::int32_t>(hd);
    std::int32_t adt = static_cast<std::int32_t>(dt);
    std::int32_t adh = static_cast<std::int32_t>(dh);
    std::int32_t adw = static_cast<std::int32_t>(dw);
    std::int32_t aoffset = offset;
    float aln_base = ln_base;
    void* params[] = {
        &ax, &ay, &act, &ast, &ach, &ash, &acw, &asw, &apt, &aph, &apw,
        &arows, &aheads, &ahd, &adt, &adh, &adw, &aoffset, &aln_base};
    launch_kernel(fn, gx, kernels::kRopeBlock, params, "mir_rope");
}

}  // namespace brotensor::detail::cuda
