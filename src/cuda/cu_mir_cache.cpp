#include "cu_mir_cache.h"

#include "cuda_driver.h"
#include "cuda_jit_engine.h"
#include "../kernels/mir_copy.h"
#include "../kernels/mir_delta.h"
#include "../kernels/mir_diffusion.h"
#include "../kernels/mir_elementwise.h"
#include "../kernels/mir_gnorm.h"
#include "../kernels/mir_lstm.h"
#include "../kernels/mir_norm.h"
#include "../kernels/mir_reduce.h"
#include "../kernels/mir_remap.h"
#include "../kernels/mir_resample.h"
#include "../kernels/mir_resample_bwd.h"
#include "../kernels/mir_rope.h"
#include "../kernels/mir_scatter.h"
#include "../kernels/mir_topk.h"
#include "../kernels/mir_xent.h"

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
        case MirKind::FillBytes:  fn = kernels::build_fill_bytes_mir(mod); break;
        case MirKind::Copy2d:     fn = kernels::build_copy2d_mir(mod, static_cast<int>(k.op), k.lanes); break;
        case MirKind::Transpose:  fn = kernels::build_transpose_mir(mod, static_cast<int>(k.op)); break;
        case MirKind::ScatterAdd: fn = kernels::build_scatter_add_mir(mod, k.op, k.dt0); break;
        case MirKind::Gnorm:      fn = kernels::build_gnorm_mir(mod, k.op, k.dt0, k.flags != 0, k.lanes); break;
        case MirKind::Bnorm:      fn = kernels::build_bnorm_mir(mod, k.op); break;
        case MirKind::Sampler:    fn = kernels::build_sampler_mir(mod, k.op, k.dt0, k.dt1, k.lanes); break;
        case MirKind::Philox:     fn = kernels::build_philox_mir(mod, k.op); break;
        case MirKind::Remap:      fn = kernels::build_remap_mir(mod, k.op, k.lanes); break;
        case MirKind::Resample: {
            const bool align = (k.flags & 1u) != 0;
            const int imode = static_cast<int>(k.flags >> 1);
            fn = kernels::build_resample_mir(mod, k.op, k.dt0, align, imode);
            break;
        }
        case MirKind::ResampleBwd:
            fn = kernels::build_resample_bwd_mir(mod, k.op, k.dt0, static_cast<std::uint32_t>(k.flags));
            break;
        case MirKind::Lstm:
            fn = kernels::build_lstm_mir(mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::DeltaRule:
            fn = kernels::build_delta_rule_mir(mod, k.op, static_cast<std::uint32_t>(k.flags));
            break;
        case MirKind::Xent:
            fn = kernels::build_xent_mir(mod, k.op, (k.flags & 1u) != 0, (k.flags & 2u) != 0);
            break;
        case MirKind::Topk:
            fn = kernels::build_topk_mir(mod, k.op, k.dt0);
            break;
    }
    const std::string ptx = brass::target::PtxTarget::emit_function(*fn, jit::make_ptx_opts());
    CUfunction cu_fn = jit::CudaJitEngine::instance().get_function(
        key_string_for(k), ptx, std::string(fn->name()));

    MirCache& cache = mir_cache();
    std::lock_guard<std::mutex> lk(cache.mu);
    auto [it, inserted] = cache.map.emplace(k, cu_fn);
    (void)inserted;
    return it->second;
#endif
}

void launch_kernel_3d(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                      std::uint32_t block_x, void** params, const char* what) {
    const CUstream stream = jit::resolve_stream(nullptr);
    const CUresult res = drv::cuLaunchKernel(
        fn, gx, gy, gz, block_x, 1, 1, 0, stream, params, nullptr);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("cuLaunchKernel failed for ") + what);
    }
}

void launch_kernel(CUfunction fn, std::uint32_t gx, std::uint32_t block_x, void** params, const char* what) {
    launch_kernel_3d(fn, gx, 1, 1, block_x, params, what);
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

int mir_copy2d_lanes(std::uint32_t esz, std::uint32_t sp, std::uint32_t dp,
                     std::uint32_t w, std::uintptr_t sa, std::uintptr_t da) {
    if (esz != 2u && esz != 4u) return 1;
    const int lanes = static_cast<int>(16u / esz);
    if ((w % static_cast<std::uint32_t>(lanes)) != 0u) return 1;
    if ((sp % static_cast<std::uint32_t>(lanes)) != 0u ||
        (dp % static_cast<std::uint32_t>(lanes)) != 0u) return 1;
    if ((sa % 16u) != 0u || (da % 16u) != 0u) return 1;
    return lanes;
}

void mir_dispatch_fill(CUfunction fn, std::uint32_t gx, void* dst, std::uint32_t n, std::uint32_t val) {
    CUdeviceptr ad = reinterpret_cast<CUdeviceptr>(dst);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t av = static_cast<std::int32_t>(val);
    void* params[] = {&ad, &an, &av};
    launch_kernel(fn, gx, kernels::kFillBlock, params, "mir_fill_bytes");
}

void mir_dispatch_copy2d(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                         const void* src, void* dst, std::uint32_t src_pitch, std::uint32_t dst_pitch,
                         std::uint32_t width, std::uint32_t height) {
    CUdeviceptr asrc = reinterpret_cast<CUdeviceptr>(src);
    CUdeviceptr adst = reinterpret_cast<CUdeviceptr>(dst);
    std::int32_t asp = static_cast<std::int32_t>(src_pitch);
    std::int32_t adp = static_cast<std::int32_t>(dst_pitch);
    std::int32_t aw = static_cast<std::int32_t>(width);
    std::int32_t ah = static_cast<std::int32_t>(height);
    void* params[] = {&asrc, &adst, &asp, &adp, &aw, &ah};
    launch_kernel_3d(fn, gx, gy, 1, kernels::kCopyBlock, params, "mir_copy2d");
}

void mir_dispatch_transpose(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                            const void* src, void* dst,
                            std::uint32_t batch, std::uint32_t rows, std::uint32_t cols) {
    CUdeviceptr asrc = reinterpret_cast<CUdeviceptr>(src);
    CUdeviceptr adst = reinterpret_cast<CUdeviceptr>(dst);
    std::int32_t ab = static_cast<std::int32_t>(batch);
    std::int32_t ar = static_cast<std::int32_t>(rows);
    std::int32_t ac = static_cast<std::int32_t>(cols);
    void* params[] = {&asrc, &adst, &ab, &ar, &ac};
    launch_kernel_3d(fn, gx, gy, gz, kernels::kTransposeBlock, params, "mir_transpose");
}

void mir_dispatch_scatter(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                          const void* idx, const void* keys, const void* keys_out,
                          const void* src, void* dst,
                          std::uint32_t m, std::uint32_t r, std::uint32_t c,
                          std::uint32_t p, std::uint32_t k, std::uint32_t j) {
    CUdeviceptr aidx = reinterpret_cast<CUdeviceptr>(idx);
    CUdeviceptr ak = reinterpret_cast<CUdeviceptr>(keys);
    CUdeviceptr ako = reinterpret_cast<CUdeviceptr>(keys_out);
    CUdeviceptr asrc = reinterpret_cast<CUdeviceptr>(src);
    CUdeviceptr adst = reinterpret_cast<CUdeviceptr>(dst);
    std::int32_t am = static_cast<std::int32_t>(m);
    std::int32_t ar = static_cast<std::int32_t>(r);
    std::int32_t ac = static_cast<std::int32_t>(c);
    std::int32_t ap = static_cast<std::int32_t>(p);
    std::int32_t a_k = static_cast<std::int32_t>(k);
    std::int32_t aj = static_cast<std::int32_t>(j);
    void* params[] = {&aidx, &ak, &ako, &asrc, &adst, &am, &ar, &ac, &ap, &a_k, &aj};
    launch_kernel_3d(fn, gx, gy, 1, kernels::kScatterBlock, params, "mir_scatter_add");
}

void mir_dispatch_gnorm(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                        const void* x, void* y, const void* g, const void* b,
                        const void* part, const void* rm, const void* rv,
                        std::uint32_t tile, std::uint32_t hw, std::uint32_t cpg,
                        std::uint32_t groups, std::uint32_t splits, std::uint32_t chunk,
                        std::uint32_t total, std::uint32_t c, float eps) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr ag = reinterpret_cast<CUdeviceptr>(g);
    CUdeviceptr ab = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr apart = reinterpret_cast<CUdeviceptr>(part);
    CUdeviceptr arm = reinterpret_cast<CUdeviceptr>(rm);
    CUdeviceptr arv = reinterpret_cast<CUdeviceptr>(rv);
    std::int32_t atile = static_cast<std::int32_t>(tile);
    std::int32_t ahw = static_cast<std::int32_t>(hw);
    std::int32_t acpg = static_cast<std::int32_t>(cpg);
    std::int32_t agroups = static_cast<std::int32_t>(groups);
    std::int32_t asplits = static_cast<std::int32_t>(splits);
    std::int32_t achunk = static_cast<std::int32_t>(chunk);
    std::int32_t atotal = static_cast<std::int32_t>(total);
    std::int32_t ac = static_cast<std::int32_t>(c);
    float aeps = eps;
    void* params[] = {
        &ax, &ay, &ag, &ab, &apart, &arm, &arv,
        &atile, &ahw, &acpg, &agroups, &asplits, &achunk, &atotal, &ac, &aeps};
    launch_kernel_3d(fn, gx, gy, 1, kernels::kGnormBlock, params, "mir_gnorm");
}

void mir_dispatch_bnorm(CUfunction fn, std::uint32_t gx,
                        const void* x, void* y, const void* g, const void* b,
                        void* rm, void* rv, void* sm, void* sr,
                        const void* dy, void* dg, void* db,
                        std::uint32_t n, std::uint32_t c, std::uint32_t hw,
                        float eps, float momentum) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr ag = reinterpret_cast<CUdeviceptr>(g);
    CUdeviceptr ab = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr arm = reinterpret_cast<CUdeviceptr>(rm);
    CUdeviceptr arv = reinterpret_cast<CUdeviceptr>(rv);
    CUdeviceptr asm_p = reinterpret_cast<CUdeviceptr>(sm);
    CUdeviceptr asr_p = reinterpret_cast<CUdeviceptr>(sr);
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr adg = reinterpret_cast<CUdeviceptr>(dg);
    CUdeviceptr adb = reinterpret_cast<CUdeviceptr>(db);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t ac = static_cast<std::int32_t>(c);
    std::int32_t ahw = static_cast<std::int32_t>(hw);
    float aeps = eps;
    float amom = momentum;
    void* params[] = {
        &ax, &ay, &ag, &ab, &arm, &arv, &asm_p, &asr_p, &ady, &adg, &adb,
        &an, &ac, &ahw, &aeps, &amom};
    launch_kernel(fn, gx, kernels::kGnormBlock, params, "mir_bnorm");
}

void mir_dispatch_sampler(CUfunction fn, std::uint32_t gx,
                          const void* x, const void* e, const void* p, void* y, void* z,
                          std::uint32_t n, std::uint32_t dim, std::uint32_t half,
                          std::uint32_t hw, std::uint32_t c,
                          float k0, float k1, float k2, float k3) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ae = reinterpret_cast<CUdeviceptr>(e);
    CUdeviceptr ap = reinterpret_cast<CUdeviceptr>(p);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr az = reinterpret_cast<CUdeviceptr>(z);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t adim = static_cast<std::int32_t>(dim);
    std::int32_t ahalf = static_cast<std::int32_t>(half);
    std::int32_t ahw = static_cast<std::int32_t>(hw);
    std::int32_t ac = static_cast<std::int32_t>(c);
    float ak0 = k0, ak1 = k1, ak2 = k2, ak3 = k3;
    void* params[] = {
        &ax, &ae, &ap, &ay, &az, &an, &adim, &ahalf, &ahw, &ac, &ak0, &ak1, &ak2, &ak3};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_sampler");
}

void mir_dispatch_philox(CUfunction fn, std::uint32_t gx,
                         void* y, std::uint64_t key, std::uint64_t counter, std::uint64_t n,
                         float p, float lo, float hi) {
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    std::int32_t klo = static_cast<std::int32_t>(key);
    std::int32_t khi = static_cast<std::int32_t>(key >> 32);
    std::int32_t clo = static_cast<std::int32_t>(counter);
    std::int32_t chi = static_cast<std::int32_t>(counter >> 32);
    std::int32_t nlo = static_cast<std::int32_t>(n);
    std::int32_t nhi = static_cast<std::int32_t>(n >> 32);
    float ap = p, alo = lo, ahi = hi;
    void* params[] = {&ay, &klo, &khi, &clo, &chi, &nlo, &nhi, &ap, &alo, &ahi};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_philox");
}

void mir_dispatch_remap(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                        const void* x, void* y, const void* idx,
                        std::uint32_t c, std::uint32_t cin, std::uint32_t hi, std::uint32_t wi,
                        std::uint32_t ho, std::uint32_t wo,
                        std::int32_t a0, std::int32_t a1, std::int32_t a2, std::int32_t a3,
                        std::int32_t a4, std::int32_t a5, std::int32_t a6) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr aidx = reinterpret_cast<CUdeviceptr>(idx);
    std::int32_t ac = static_cast<std::int32_t>(c);
    std::int32_t acin = static_cast<std::int32_t>(cin);
    std::int32_t ahi = static_cast<std::int32_t>(hi);
    std::int32_t awi = static_cast<std::int32_t>(wi);
    std::int32_t aho = static_cast<std::int32_t>(ho);
    std::int32_t awo = static_cast<std::int32_t>(wo);
    std::int32_t p0 = a0, p1 = a1, p2 = a2, p3 = a3, p4 = a4, p5 = a5, p6 = a6;
    void* params[] = {
        &ax, &ay, &aidx, &ac, &acin, &ahi, &awi, &aho, &awo,
        &p0, &p1, &p2, &p3, &p4, &p5, &p6};
    launch_kernel_3d(fn, gx, gy, gz, kernels::kEwBlock, params, "mir_remap");
}

void mir_dispatch_resample(CUfunction fn, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                           const void* x, void* y, const void* aux,
                           std::uint32_t c, std::uint32_t hi, std::uint32_t wi,
                           std::uint32_t ho, std::uint32_t wo,
                           std::uint32_t kh, std::uint32_t kw,
                           std::int32_t sh, std::int32_t sw, std::int32_t ph, std::int32_t pw) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr aaux = reinterpret_cast<CUdeviceptr>(aux);
    std::int32_t ac = static_cast<std::int32_t>(c);
    std::int32_t ahi = static_cast<std::int32_t>(hi);
    std::int32_t awi = static_cast<std::int32_t>(wi);
    std::int32_t aho = static_cast<std::int32_t>(ho);
    std::int32_t awo = static_cast<std::int32_t>(wo);
    std::int32_t akh = static_cast<std::int32_t>(kh);
    std::int32_t akw = static_cast<std::int32_t>(kw);
    std::int32_t ash = sh, asw = sw, aph = ph, apw = pw;
    void* params[] = {
        &ax, &ay, &aaux, &ac, &ahi, &awi, &aho, &awo,
        &akh, &akw, &ash, &asw, &aph, &apw};
    launch_kernel_3d(fn, gx, gy, gz, kernels::kEwBlock, params, "mir_resample");
}

void mir_dispatch_resample_bwd(CUfunction fn, std::uint32_t gx,
                               const void* dy, void* dx, const void* aux,
                               std::uint32_t planes, std::uint32_t hi, std::uint32_t wi,
                               std::uint32_t ho, std::uint32_t wo,
                               std::int32_t pt, std::int32_t pl) {
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr adx = reinterpret_cast<CUdeviceptr>(dx);
    CUdeviceptr aaux = reinterpret_cast<CUdeviceptr>(aux);
    std::int32_t aplanes = static_cast<std::int32_t>(planes);
    std::int32_t ahi = static_cast<std::int32_t>(hi);
    std::int32_t awi = static_cast<std::int32_t>(wi);
    std::int32_t aho = static_cast<std::int32_t>(ho);
    std::int32_t awo = static_cast<std::int32_t>(wo);
    std::int32_t apt = pt, apl = pl;
    void* params[] = {
        &ady, &adx, &aaux, &aplanes, &ahi, &awi, &aho, &awo, &apt, &apl};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_resample_bwd");
}

void mir_dispatch_lstm(CUfunction fn, std::uint32_t gx,
                       const void* gates, const void* c, const void* y,
                       const void* bhh, const void* c0,
                       const void* dy, void* dz, void* dh, void* dc,
                       std::uint32_t b, std::uint32_t h, std::uint32_t row0) {
    CUdeviceptr ag = reinterpret_cast<CUdeviceptr>(gates);
    CUdeviceptr ac = reinterpret_cast<CUdeviceptr>(c);
    CUdeviceptr ay = reinterpret_cast<CUdeviceptr>(y);
    CUdeviceptr abhh = reinterpret_cast<CUdeviceptr>(bhh);
    CUdeviceptr ac0 = reinterpret_cast<CUdeviceptr>(c0);
    CUdeviceptr ady = reinterpret_cast<CUdeviceptr>(dy);
    CUdeviceptr adz = reinterpret_cast<CUdeviceptr>(dz);
    CUdeviceptr adh = reinterpret_cast<CUdeviceptr>(dh);
    CUdeviceptr adc = reinterpret_cast<CUdeviceptr>(dc);
    std::int32_t ab = static_cast<std::int32_t>(b);
    std::int32_t ah = static_cast<std::int32_t>(h);
    std::int32_t ar0 = static_cast<std::int32_t>(row0);
    void* params[] = {&ag, &ac, &ay, &abhh, &ac0, &ady, &adz, &adh, &adc, &ab, &ah, &ar0};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_lstm");
}

void mir_dispatch_delta(CUfunction fn, std::uint32_t gx,
                        const void* q, const void* k, const void* v,
                        const void* a, const void* b, const void* loga,
                        void* s, void* o,
                        std::uint32_t L, std::uint32_t H, std::uint32_t dk, std::uint32_t dv,
                        std::uint32_t ldq, std::uint32_t ldv) {
    CUdeviceptr aq = reinterpret_cast<CUdeviceptr>(q);
    CUdeviceptr ak = reinterpret_cast<CUdeviceptr>(k);
    CUdeviceptr av = reinterpret_cast<CUdeviceptr>(v);
    CUdeviceptr aa = reinterpret_cast<CUdeviceptr>(a);
    CUdeviceptr ab = reinterpret_cast<CUdeviceptr>(b);
    CUdeviceptr aloga = reinterpret_cast<CUdeviceptr>(loga);
    CUdeviceptr as = reinterpret_cast<CUdeviceptr>(s);
    CUdeviceptr ao = reinterpret_cast<CUdeviceptr>(o);
    std::int32_t aL = static_cast<std::int32_t>(L);
    std::int32_t aH = static_cast<std::int32_t>(H);
    std::int32_t adk = static_cast<std::int32_t>(dk);
    std::int32_t adv = static_cast<std::int32_t>(dv);
    std::int32_t aldq = static_cast<std::int32_t>(ldq);
    std::int32_t aldv = static_cast<std::int32_t>(ldv);
    void* params[] = {
        &aq, &ak, &av, &aa, &ab, &aloga, &as, &ao,
        &aL, &aH, &adk, &adv, &aldq, &aldv};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_delta");
}

void mir_dispatch_xent(CUfunction fn, std::uint32_t gx,
                       const void* logits, const void* target, const void* mask, const void* off,
                       void* probs, void* dlog, void* loss,
                       std::uint32_t rows, std::uint32_t n, std::uint32_t heads, float pos_weight) {
    CUdeviceptr al = reinterpret_cast<CUdeviceptr>(logits);
    CUdeviceptr at = reinterpret_cast<CUdeviceptr>(target);
    CUdeviceptr am = reinterpret_cast<CUdeviceptr>(mask);
    CUdeviceptr ao = reinterpret_cast<CUdeviceptr>(off);
    CUdeviceptr ap = reinterpret_cast<CUdeviceptr>(probs);
    CUdeviceptr ad = reinterpret_cast<CUdeviceptr>(dlog);
    CUdeviceptr aloss = reinterpret_cast<CUdeviceptr>(loss);
    std::int32_t arows = static_cast<std::int32_t>(rows);
    std::int32_t an = static_cast<std::int32_t>(n);
    std::int32_t aheads = static_cast<std::int32_t>(heads);
    float apw = pos_weight;
    void* params[] = {&al, &at, &am, &ao, &ap, &ad, &aloss, &arows, &an, &aheads, &apw};
    launch_kernel(fn, gx, kernels::kEwBlock, params, "mir_xent");
}

void mir_dispatch_topk(CUfunction fn, std::uint32_t gx, std::uint32_t gy,
                       const void* x, void* vals, void* idx, const void* off,
                       std::uint32_t rows, std::uint32_t cols, std::uint32_t k, std::uint32_t nseg) {
    CUdeviceptr ax = reinterpret_cast<CUdeviceptr>(x);
    CUdeviceptr av = reinterpret_cast<CUdeviceptr>(vals);
    CUdeviceptr aidx = reinterpret_cast<CUdeviceptr>(idx);
    CUdeviceptr aoff = reinterpret_cast<CUdeviceptr>(off);
    std::int32_t arows = static_cast<std::int32_t>(rows);
    std::int32_t acols = static_cast<std::int32_t>(cols);
    std::int32_t ak = static_cast<std::int32_t>(k);
    std::int32_t anseg = static_cast<std::int32_t>(nseg);
    void* params[] = {&ax, &av, &aidx, &aoff, &arows, &acols, &ak, &anseg};
    launch_kernel_3d(fn, gx, gy, 1, kernels::kEwBlock, params, "mir_topk");
}

}  // namespace brotensor::detail::cuda
