// The GEMM dispatcher behind every Vulkan matrix op: kernel choice, tile
// configuration, alignment checks, batch chunking. Design: detail/gemm.h and
// docs/vulkan.md "Matrix multiply".

#include "detail/gemm.h"
#include "detail/kernels.h"
#include "vk_autotune.h"
#include "vk_mir_cache.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;

namespace {

// Cooperative-matrix tiles: BM x BN workgroup tile, K step BK, WM x WN per
// subgroup of 32.
struct CmCfg { std::uint32_t bm, bn, bk, wm, wn; };
constexpr CmCfg kCmCfgs[] = {
    {256, 128, 32, 64, 64},
    {128, 128, 32, 64, 64},
    {128, 64, 32, 64, 32},
    {64, 64, 32, 32, 32},
};

struct SimtCfg { std::uint32_t bm, bn, bk, tm, tn; };
constexpr SimtCfg kSimtCfgs[] = {
    {128, 128, 16, 8, 8},
    {64, 64, 16, 4, 4},
};

std::atomic<int> g_override{0};
std::atomic<int> g_cm_cfg{-1};
std::atomic<int> g_noscale{0};    // 1: set_gemm_scaling(0) turned the scaling off

bool is16(Dtype t) { return t == Dtype::FP16 || t == Dtype::BF16; }

std::uint64_t cdiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

[[noreturn]] void fail(const GemmArgs& g, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + g.op + ": " + why);
}

std::uint32_t esize(Dtype t) { return t == Dtype::FP32 ? 4u : 2u; }

// 16-byte loads of 8-element chunks: aligned bases, leading dimensions and
// batch strides, and (for the cooperative-matrix kernel, whose VEC loads are
// unconditional) contiguous extents in whole chunks.
bool vec_ok(const GemmArgs& g) {
    return g.a % 16 == 0 && g.b % 16 == 0 && g.lda % 8 == 0 && g.ldb % 8 == 0 &&
           g.sa % 8 == 0 && g.sb % 8 == 0;
}

bool cm_vec_ok(const GemmArgs& g) {
    const int ea = g.ta ? g.m : g.k, eb = g.nb ? g.n : g.k;
    if (g.qb != 0) {   // A as usual; a byte-format (INT8 / FP8) B row in 8-byte chunks (the GGUF formats have no scalar form)
        return g.a % 16 == 0 && g.lda % 8 == 0 && g.k > 0 && g.k % 8 == 0 &&
               ((g.qb != QF_INT8 && g.qb != QF_FP8) || (g.b % 8 == 0 && g.ldb % 8 == 0));
    }
    return vec_ok(g) && g.k > 0 && ea % 8 == 0 && eb % 8 == 0;
}

bool gemv_eligible(DeviceCtx& d, const GemmArgs& g) {
    if (g_override.load(std::memory_order_relaxed) != 0 || g.qb != 0 || g.epi == EPI_RESIDUAL_ADD) return false;
    const std::uint32_t sgs = d.info().subgroup_size;
    if (sgs < 32 || sgs > 64 || 64 % sgs != 0) return false;
    if (g.ta || g.nb || g.batch != 1 || g.m > 8) return false;
    if (g.da != g.dc) return false;
    return g.db == g.da || (g.da == Dtype::FP32 && is16(g.db));
}

bool coopmat_eligible(DeviceCtx& d, const GemmArgs& g) {
    if (!d.info().coopmat_f16) return false;
    if (g.qb != 0) return is16(g.da) && g.dc == g.da;
    if (g_override.load(std::memory_order_relaxed) == 2) return false;
    if (is16(g.da)) return g.db == g.da && (g.dc == g.da || g.dc == Dtype::FP32);
    return g.round_a && g.da == Dtype::FP32 && is16(g.db) && g.dc == Dtype::FP32;
}

std::uint32_t pick_cm_idx(const GemmArgs& g, std::uint32_t nout) {
    const int ci = g_cm_cfg.load(std::memory_order_relaxed);
    if (ci >= 0 && ci < static_cast<int>(std::size(kCmCfgs))) return static_cast<std::uint32_t>(ci);
    auto tiles = [&](const CmCfg& c) {
        const std::uint32_t wout = g.epi == EPI_SWIGLU ? c.bn / 2 : c.bn;
        return cdiv(static_cast<std::uint64_t>(g.m), c.bm) * cdiv(nout, wout) * static_cast<std::uint64_t>(g.batch);
    };
    if (g.m >= 512 && (tiles(kCmCfgs[0]) >= 150 || (g.k >= 8192 && tiles(kCmCfgs[0]) >= 32))) return 0u;
    if (g.m > 64 && tiles(kCmCfgs[1]) >= 24) return 1u;
    if (g.m > 64 && tiles(kCmCfgs[2]) >= 24) return 2u;
    return 3u;
}

// Launches `mk` over the batch in chunks of at most 65535 workgroups in z.
void launch_cm_batched(DeviceCtx& d, const MirKernel& mk, const GemmArgs& g, std::uint64_t ascale,
                       std::uint32_t tm0, std::uint32_t tn0, std::uint32_t gx, std::uint32_t gy) {
    if (gx > 65535 || gy > 65535) fail(g, "matrix too large for one dispatch");
    const std::int32_t ints[] = {
        g.m, g.n, g.k, g.lda, g.ldb, g.ldc,
        static_cast<std::int32_t>(g.sa), static_cast<std::int32_t>(g.sb), static_cast<std::int32_t>(g.sc),
        g.n / 2, static_cast<std::int32_t>(tm0), static_cast<std::int32_t>(tn0),
    };
    for (int z0 = 0; z0 < g.batch; z0 += 65535) {
        const int nz = std::min(65535, g.batch - z0);
        const std::uint64_t as = (ascale && g.sa != 0)
            ? ascale + static_cast<std::uint64_t>(z0) * static_cast<std::uint64_t>(g.m) * 4u
            : ascale;
        const std::uint64_t ptrs[] = {
            g.a + static_cast<std::uint64_t>(z0) * g.sa * esize(g.da),
            g.b + static_cast<std::uint64_t>(z0) * g.sb * esize(g.db),
            g.c + static_cast<std::uint64_t>(z0) * g.sc * esize(g.dc),
            g.bias,
            g.scale,
            as,
            g.epi == EPI_RESIDUAL_ADD
                ? (g.residual + static_cast<std::uint64_t>(z0) * g.sc * esize(g.dc))
                : 0u,
        };
        const std::span<const std::uint64_t> ptr_span(
            ptrs, g.epi == EPI_RESIDUAL_ADD ? 7u : 6u);
        mir_dispatch_args_3d(d, mk, gx, gy, static_cast<std::uint32_t>(nz), ptr_span, ints, {});
    }
}

void check_args(const GemmArgs& g) {
    if (g.m < 0 || g.n < 0 || g.k < 0 || g.batch < 1) fail(g, "negative dimension");
    if (g.sa < 0 || g.sb < 0 || g.sc < 0 || g.sa > 0x7fffffffLL || g.sb > 0x7fffffffLL || g.sc > 0x7fffffffLL) {
        fail(g, "batch strides must be in [0, 2^31)");
    }
    if ((g.epi == EPI_GEGLU || g.epi == EPI_SWIGLU) && (g.ta || g.nb || g.n % 2 != 0 || g.act != 0)) {
        fail(g, "the GeGLU / SwiGLU epilogues need the linear (NT) layout, an even width and act 0");
    }
    if (g.epi == EPI_RESIDUAL_ADD && g.residual == 0) {
        fail(g, "EPI_RESIDUAL_ADD requires a residual device address");
    }
    if (g.qb != 0 && (g.ta || g.nb || g.batch != 1)) fail(g, "a quantised B needs the NT layout and one batch");
    // Element indices are 32-bit in the kernels (no buffer exceeds 4 GiB).
    auto fits = [](long long rows, long long ld) { return rows * ld < (1LL << 32); };
    if (!fits(g.ta ? g.k : g.m, g.lda) || (g.qb == 0 && !fits(g.nb ? g.k : g.n, g.ldb)) || !fits(g.m, g.ldc)) {
        fail(g, "operand larger than the 4 GiB buffer limit");
    }
}

// Gates an FP32 r (rows, 2 half) into y of dtype `dt`.
void glu_pass(DeviceCtx& d, std::uint32_t op, std::uint64_t x, std::uint64_t y, Dtype dt,
              std::uint32_t rows, std::uint32_t half) {
    const std::uint32_t n = rows * half;
    const int lanes = (op != GLU_GEGLU_PAIRS) ? mir_vec_lanes_mixed(Dtype::FP32, dt, half, x, y) : 1;
    const MirKernel mk = mir_kernel(d, {MirKind::Glu, op, Dtype::FP32, dt, static_cast<std::uint8_t>(lanes)});
    mir_dispatch_glu(d, mk, mir_groups_1d(n, lanes), x, 0, y, n, half);
}

// A BF16 A on the cooperative-matrix kernel is staged with per-row
// power-of-two scales (docs/vulkan-bf16.md) unless the test hook turned it
// off; an FP32 A always is.
bool scale_a(const GemmArgs& g) {
    if (g.da == Dtype::FP32) return true;
    if (g.da != Dtype::BF16) return false;
    return g_noscale.load(std::memory_order_relaxed) != 1;
}

// Writes the exponent of every row of A (per batch slice, or once when A is
// broadcast) into `out` (INT32): gemm_rowscale.comp.
void row_scales(DeviceCtx& d, const GemmArgs& g, std::uint64_t out) {
    const bool vec = !g.ta && g.a % 16 == 0 && g.lda % 8 == 0 && g.k % 8 == 0 && g.sa % 8 == 0;
    const std::uint8_t flags = static_cast<std::uint8_t>((g.ta ? 1u : 0u) | (vec ? 2u : 0u));
    const MirKernel mk = mir_kernel(d, {MirKind::GemmRowscale, 0u, g.da, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint64_t groups = g.ta ? cdiv(g.m, 256) : cdiv(g.m, 4);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(groups, 65535)));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(groups, gx));
    if (gy > 65535) fail(g, "matrix too large for one dispatch");
    const int nb = g.sa != 0 ? g.batch : 1;
    for (int z0 = 0; z0 < nb; z0 += 65535) {
        const int nz = std::min(65535, nb - z0);
        const std::uint64_t ptrs[] = {
            g.a + static_cast<std::uint64_t>(z0) * g.sa * esize(g.da),
            out + static_cast<std::uint64_t>(z0) * static_cast<std::uint64_t>(g.m) * 4u,
        };
        const std::int32_t ints[] = {g.m, g.k, g.lda, static_cast<std::int32_t>(g.sa)};
        mir_dispatch_args_3d(d, mk, gx, gy, static_cast<std::uint32_t>(nz), ptrs, ints, {});
    }
}

}  // namespace

void set_gemm_scaling(int mode) { g_noscale.store(mode == 0 ? 1 : 0, std::memory_order_relaxed); }

void set_gemm_override(int mode) { g_override.store(mode, std::memory_order_relaxed); }

int set_gemm_cm_config(int index) {
    g_cm_cfg.store(index, std::memory_order_relaxed);
    return static_cast<int>(std::size(kCmCfgs));
}

const char* gemm_cm_config_name(int index) {
    static std::string names[std::size(kCmCfgs)];
    if (index < 0 || index >= static_cast<int>(std::size(kCmCfgs))) return "auto";
    const CmCfg& c = kCmCfgs[index];
    if (names[index].empty()) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%ux%ux%u/w%ux%u", c.bm, c.bn, c.bk, c.wm, c.wn);
        names[index] = buf;
    }
    return names[index].c_str();
}

const char* gemm_path(DeviceCtx& d, const GemmArgs& g) {
    if (gemv_eligible(d, g)) return "gemv";
    if (coopmat_eligible(d, g)) return "coopmat";
    return "simt";
}

void gemm(DeviceCtx& d, const GemmArgs& g) {
    check_args(g);
    const std::uint32_t nout = g.epi == EPI_SWIGLU || g.epi == EPI_GEGLU ? g.n / 2 : g.n;
    if (g.m == 0 || nout == 0) return;

    if (gemv_eligible(d, g)) {
        const bool vec = vec_ok(g) && g.k % 8 == 0 && g.k > 0;
        const std::uint32_t op = static_cast<std::uint32_t>(g.m) |
                                 (static_cast<std::uint32_t>(g.epi) << 4) |
                                 (static_cast<std::uint32_t>(g.act) << 8);
        const std::uint8_t flags = static_cast<std::uint8_t>((vec ? 1u : 0u) | (g.bias ? 2u : 0u));
        const MirKernel mk = mir_kernel(d, {MirKind::Gemv, op, g.da, g.db, 1, g.dc, flags});
        const std::uint64_t groups = cdiv(nout, 4);
        const std::uint32_t gx = static_cast<std::uint32_t>(
            std::max<std::uint64_t>(1, std::min<std::uint64_t>(groups, 65535)));
        const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(groups, gx));
        const std::uint64_t ptrs[] = {g.a, g.b, g.c, g.bias};
        const std::int32_t ints[] = {g.m, g.n, g.k, g.lda, g.ldb, g.ldc, g.n / 2};
        mir_dispatch_args_3d(d, mk, gx, gy, 1u, ptrs, ints, {});
        return;
    }

    if (coopmat_eligible(d, g)) {
        const std::uint32_t heur_idx = pick_cm_idx(g, nout);
        const std::uint32_t heur_sw = 0u;
        GemmTuneCfg tuned{heur_idx, heur_sw, true};
        if (g_cm_cfg.load(std::memory_order_relaxed) < 0) {
            tuned = autotune_gemm_cm(d, g, nout, tuned);
        }
        const std::uint32_t cfg_idx = tuned.cfg_idx;
        const CmCfg& c = kCmCfgs[cfg_idx];
        const std::uint32_t wout = g.epi == EPI_SWIGLU ? c.bn / 2 : c.bn;
        const std::uint32_t gx = static_cast<std::uint32_t>(
            g.epi == EPI_SWIGLU ? cdiv(nout, wout) : cdiv(static_cast<std::uint64_t>(g.n), c.bn));
        const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(g.m, c.bm));
        // Range-safe BF16 / FP32: the per-row exponents of A, in a stream-ordered
        // scratch (freed here, reused only by later work on the stream).
        const bool scaled = scale_a(g);
        ::brotensor::Tensor exps;
        std::uint64_t ascale = 0;
        if (scaled) {
            const long long rows = static_cast<long long>(g.m) * (g.sa != 0 ? g.batch : 1);
            if (rows > 0x7fffffffLL) fail(g, "too many rows to scale");
            exps = ::brotensor::Tensor::empty_on(::brotensor::Device::vulkan(d.index()), static_cast<int>(rows), 1,
                                                 Dtype::INT32);
            row_scales(d, g, addr(exps.data));
            ascale = addr(exps.data);
        }
        const std::uint32_t op = (cfg_idx & 3u) |
                                 ((static_cast<std::uint32_t>(g.epi) & 0x0fu) << 4) |
                                 ((static_cast<std::uint32_t>(g.act) & 0x0fu) << 8) |
                                 ((static_cast<std::uint32_t>(g.qb) & 0x0fu) << 12);
        const std::uint8_t base_flags = static_cast<std::uint8_t>(
            (g.ta ? 1u : 0u) | (g.nb ? 2u : 0u) | (cm_vec_ok(g) ? 4u : 0u) |
            (g.bias ? 16u : 0u) | (scaled ? 32u : 0u));
        auto kernel = [&](bool direct, bool swizzle) -> MirKernel {
            const std::uint8_t flags = static_cast<std::uint8_t>(
                base_flags | (direct ? 8u : 0u) | (swizzle ? 64u : 0u));
            return mir_kernel(d, {MirKind::GemmCm, op, g.da, g.db, 1, g.dc, flags}, 32u);
        };
        const bool direct = g.da == Dtype::FP16 && g.dc == Dtype::FP16 && g.epi != EPI_GEGLU && g.c % 16 == 0 && g.ldc % 8 == 0 &&
                            g.sc % 8 == 0 && (g.bias == 0 || g.bias % 16 == 0) &&
                            (g.epi != EPI_RESIDUAL_ADD || g.residual % 16 == 0);
        const std::uint32_t fx = direct ? static_cast<std::uint32_t>((g.epi == EPI_SWIGLU ? nout : g.n) / wout) : 0;
        const std::uint32_t fy = direct ? static_cast<std::uint32_t>(g.m / c.bm) : 0;
        const bool use_swizzle = (tuned.swizzle_group > 0u);
        if (fx > 0 && fy > 0) {
            launch_cm_batched(d, kernel(true, use_swizzle), g, ascale, 0u, 0u, fx, fy);
            if (fx < gx) launch_cm_batched(d, kernel(false, false), g, ascale, 0u, fx, gx - fx, gy);
            if (fy < gy) launch_cm_batched(d, kernel(false, false), g, ascale, fy, 0u, fx, gy - fy);
        } else {
            launch_cm_batched(d, kernel(false, use_swizzle), g, ascale, 0u, 0u, gx, gy);
        }
        return;
    }

    if (g.qb != 0) fail(g, "a quantised B needs cooperative matrix and 16-bit A / C");

    // The SIMT kernel runs GeGLU through an FP32 scratch + Glu pass, while
    // SwiGLU and EPI_RESIDUAL_ADD execute directly in registers in a single pass.
    const bool geglu = g.epi == EPI_GEGLU;
    const bool same_all = (g.da == g.db && g.dc == g.da);
    const bool f32_w16 = (g.da == Dtype::FP32 && g.dc == Dtype::FP32 && is16(g.db));
    const bool f16_c32 = (g.da == g.db && is16(g.da) && g.dc == Dtype::FP32);
    if (!same_all && !f32_w16 && !f16_c32) fail(g, "unsupported dtype combination on Vulkan");
    Dtype dc = g.dc;
    std::uint64_t c_addr = g.c;
    std::int32_t ldc = g.ldc;
    ::brotensor::Tensor r;
    if (geglu) {
        if (g.batch != 1) fail(g, "batched GLU epilogue");
        r = ::brotensor::Tensor::empty_on(::brotensor::Device::vulkan(d.index()), g.m, g.n, Dtype::FP32);
        dc = Dtype::FP32;
        c_addr = addr(r.data);
        ldc = g.n;
    }
    const Dtype dt_bias = f16_c32 ? g.da : g.dc;
    const std::uint64_t tiles_big = cdiv(g.m, kSimtCfgs[0].bm) * cdiv(g.n, kSimtCfgs[0].bn) * g.batch;
    const SimtCfg& c = (g.epi != EPI_SWIGLU && tiles_big >= 160) ? kSimtCfgs[0] : kSimtCfgs[1];
    const std::uint32_t epi = static_cast<std::uint32_t>(geglu ? EPI_STORE : g.epi);
    const std::uint32_t op = epi | (static_cast<std::uint32_t>(g.act) << 4) |
                             (static_cast<std::uint32_t>(dt_bias) << 8);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        ((c.bm == 128) ? 1u : 0u) | (g.ta ? 2u : 0u) | (g.nb ? 4u : 0u) | (g.bias ? 8u : 0u));
    const MirKernel mk = mir_kernel(d, {MirKind::GemmSimt, op, g.da, g.db, 1, dc, flags});
    const std::uint32_t gx = static_cast<std::uint32_t>(cdiv(g.epi == EPI_SWIGLU ? nout : static_cast<std::uint32_t>(g.n), c.bn));
    const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(g.m, c.bm));
    if (gx > 65535 || gy > 65535) fail(g, "matrix too large for one dispatch");
    for (int z0 = 0; z0 < g.batch; z0 += 65535) {
        const int nz = std::min(65535, g.batch - z0);
        const std::uint64_t ptrs[] = {
            g.a + static_cast<std::uint64_t>(z0) * g.sa * esize(g.da),
            g.b + static_cast<std::uint64_t>(z0) * g.sb * esize(g.db),
            c_addr + static_cast<std::uint64_t>(z0) * g.sc * esize(dc),
            g.bias,
            g.epi == EPI_RESIDUAL_ADD
                ? (g.residual + static_cast<std::uint64_t>(z0) * g.sc * esize(dc))
                : 0u,
        };
        const std::span<const std::uint64_t> ptr_span(
            ptrs, g.epi == EPI_RESIDUAL_ADD ? 5u : 4u);
        const std::int32_t ints[] = {g.m, g.n, g.k, g.lda, g.ldb, ldc,
                                     static_cast<std::int32_t>(g.sa), static_cast<std::int32_t>(g.sb),
                                     static_cast<std::int32_t>(g.sc)};
        mir_dispatch_args_3d(d, mk, gx, gy, static_cast<std::uint32_t>(nz), ptr_span, ints, {});
    }
    if (geglu) {
        if (g.ldc != static_cast<int>(nout)) fail(g, "GLU output must be contiguous");
        glu_pass(d, GLU_GEGLU_PAIRS, addr(r.data), g.c, g.dc, static_cast<std::uint32_t>(g.m), nout);
    }
}

}  // namespace brotensor::detail::vulkan
