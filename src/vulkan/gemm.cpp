// The GEMM dispatcher behind every Vulkan matrix op: kernel choice, tile
// configuration, alignment checks, batch chunking. Design: detail/gemm.h and
// docs/vulkan.md "Matrix multiply".

#include "detail/gemm.h"
#include "detail/kernels.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;

namespace {

struct GemmPush {
    std::uint64_t a, b, c, bias;
    std::uint32_t m, n, k, lda, ldb, ldc, sa, sb, sc, half_n;
    std::uint32_t tm0, tn0;   // gemm_cm: tile origin of the dispatch
};

struct GluPush { std::uint64_t x, dy, y; std::uint32_t n, d; };

// Cooperative-matrix tiles: BM x BN workgroup tile, K step BK, WM x WN per
// subgroup of 32. Every entry must satisfy the shader's assumptions: chunk
// counts divisible by the workgroup, WN a multiple of 32 (SwiGLU pairing),
// the epilogue scratch (WG * 4 uvec4) inside As, and <= 64 KiB of shared
// memory (the pipeline guard checks the last one too).
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

std::uint32_t cm_wg(const CmCfg& c) { return (c.bm / c.wm) * (c.bn / c.wn) * 32u; }

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
    return vec_ok(g) && g.k > 0 && ea % 8 == 0 && eb % 8 == 0;
}

bool gemv_eligible(DeviceCtx& d, const GemmArgs& g) {
    if (g_override.load(std::memory_order_relaxed) != 0) return false;
    const std::uint32_t sgs = d.info().subgroup_size;
    if (sgs < 4 || sgs > 64 || 64 % sgs != 0) return false;
    if (g.ta || g.nb || g.batch != 1 || g.m > 8) return false;
    if (g.da != g.dc) return false;
    return g.db == g.da || (g.da == Dtype::FP32 && is16(g.db));
}

bool coopmat_eligible(DeviceCtx& d, const GemmArgs& g) {
    return g_override.load(std::memory_order_relaxed) != 2 && d.info().coopmat_f16 && is16(g.da) && g.db == g.da && g.dc == g.da;
}

// The SIMT shader for (A, B, C) dtypes, or kCount when there is none.
ShaderId simt_shader(const GemmArgs& g) {
    if (g.da == g.db && g.dc == g.da) return dt_variant(ShaderId::gemm_simt_f32, g.da, g.op);
    if (g.da == Dtype::FP32 && g.dc == Dtype::FP32 && g.db == Dtype::FP16) return ShaderId::gemm_simt_f32_w16;
    if (g.da == Dtype::FP32 && g.dc == Dtype::FP32 && g.db == Dtype::BF16) return ShaderId::gemm_simt_f32_wbf16;
    // 16-bit operands, FP32 result (attention scores, FP32 projections): the
    // GLU pass's variants. Their bias is in the operands' dtype (gemm.h).
    if (g.da == g.db && is16(g.da) && g.dc == Dtype::FP32) {
        return g.da == Dtype::FP16 ? ShaderId::gemm_simt_f16_c32 : ShaderId::gemm_simt_bf16_c32;
    }
    return ShaderId::kCount;
}

const CmCfg& pick_cm(const GemmArgs& g, std::uint32_t nout) {
    static const CmCfg* forced = []() -> const CmCfg* {
        const char* e = std::getenv("BROTENSOR_VK_GEMM_CFG");
        if (!e || !*e) return nullptr;
        for (const CmCfg& c : kCmCfgs) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%u,%u,%u,%u,%u", c.bm, c.bn, c.bk, c.wm, c.wn);
            if (std::string(buf) == e) return &c;
        }
        return nullptr;
    }();
    if (forced) return *forced;
    const int ci = g_cm_cfg.load(std::memory_order_relaxed);
    if (ci >= 0 && ci < static_cast<int>(std::size(kCmCfgs))) return kCmCfgs[ci];
    // Measured on the spike's shapes (Radeon 8060S, 40 CUs, docs/vulkan.md):
    // 256x128 pays off once there are ~150 tiles or K is long enough to
    // amortise a thinner grid; 128x128 down to ~24 tiles (512x1024 runs best
    // on 32 of them); 64x64 for short M, where a 128-row tile is half empty.
    auto tiles = [&](const CmCfg& c) {
        const std::uint32_t wout = g.epi == EPI_SWIGLU ? c.bn / 2 : c.bn;
        return cdiv(static_cast<std::uint64_t>(g.m), c.bm) * cdiv(nout, wout) * static_cast<std::uint64_t>(g.batch);
    };
    if (g.m >= 512 && (tiles(kCmCfgs[0]) >= 150 || (g.k >= 8192 && tiles(kCmCfgs[0]) >= 32))) return kCmCfgs[0];
    if (g.m > 64 && tiles(kCmCfgs[1]) >= 24) return kCmCfgs[1];
    if (g.m > 64 && tiles(kCmCfgs[2]) >= 24) return kCmCfgs[2];
    return kCmCfgs[3];
}

// Launches `k` over the batch in chunks of at most 65535 workgroups in z.
void launch_batched(DeviceCtx& d, const Kernel& k, GemmPush pc, const GemmArgs& g,
                    std::uint32_t gx, std::uint32_t gy) {
    if (gx > 65535 || gy > 65535) fail(g, "matrix too large for one dispatch");
    for (int z0 = 0; z0 < g.batch; z0 += 65535) {
        const int nz = std::min(65535, g.batch - z0);
        GemmPush p = pc;
        p.a += static_cast<std::uint64_t>(z0) * g.sa * esize(g.da);
        p.b += static_cast<std::uint64_t>(z0) * g.sb * esize(g.db);
        p.c += static_cast<std::uint64_t>(z0) * g.sc * esize(g.dc);
        launch(d, k, p, gx, gy, static_cast<std::uint32_t>(nz));
    }
}

GemmPush make_push(const GemmArgs& g) {
    GemmPush pc{};
    pc.a = g.a; pc.b = g.b; pc.c = g.c; pc.bias = g.bias;
    pc.m = static_cast<std::uint32_t>(g.m);
    pc.n = static_cast<std::uint32_t>(g.n);
    pc.k = static_cast<std::uint32_t>(g.k);
    pc.lda = static_cast<std::uint32_t>(g.lda);
    pc.ldb = static_cast<std::uint32_t>(g.ldb);
    pc.ldc = static_cast<std::uint32_t>(g.ldc);
    pc.sa = static_cast<std::uint32_t>(g.sa);
    pc.sb = static_cast<std::uint32_t>(g.sb);
    pc.sc = static_cast<std::uint32_t>(g.sc);
    pc.half_n = static_cast<std::uint32_t>(g.n / 2);
    return pc;
}

void check_args(const GemmArgs& g) {
    if (g.m < 0 || g.n < 0 || g.k < 0 || g.batch < 1) fail(g, "negative dimension");
    if (g.sa < 0 || g.sb < 0 || g.sc < 0 || g.sa > 0x7fffffffLL || g.sb > 0x7fffffffLL || g.sc > 0x7fffffffLL) {
        fail(g, "batch strides must be in [0, 2^31)");
    }
    if ((g.epi == EPI_GEGLU || g.epi == EPI_SWIGLU) && (g.ta || g.nb || g.n % 2 != 0 || g.act != 0)) {
        fail(g, "the GeGLU / SwiGLU epilogues need the linear (NT) layout, an even width and act 0");
    }
    // Element indices are 32-bit in the kernels (no buffer exceeds 4 GiB).
    auto fits = [](long long rows, long long ld) { return rows * ld < (1LL << 32); };
    if (!fits(g.ta ? g.k : g.m, g.lda) || !fits(g.nb ? g.k : g.n, g.ldb) || !fits(g.m, g.ldc)) {
        fail(g, "operand larger than the 4 GiB buffer limit");
    }
}

// Gates an FP32 r (rows, 2 half) into y of dtype `dt`.
void glu_pass(DeviceCtx& d, std::uint32_t op, std::uint64_t x, std::uint64_t y, Dtype dt,
              std::uint32_t rows, std::uint32_t half) {
    const ShaderId id = dt == Dtype::FP32   ? ShaderId::glu_f32
                        : dt == Dtype::FP16 ? ShaderId::glu_f32_to_f16
                                            : ShaderId::glu_f32_to_bf16;
    const Kernel& k = d.pipelines().get(id, {op});
    const GluPush pc{x, 0, y, rows * half, half};
    launch(d, k, pc, groups_1d(pc.n, k));
}

}  // namespace

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
    const GemmPush pc = make_push(g);

    if (gemv_eligible(d, g)) {
        ShaderId id;
        if (g.db == g.da) id = dt_variant(ShaderId::gemv_f32, g.da, g.op);
        else id = g.db == Dtype::FP16 ? ShaderId::gemv_f32_w16 : ShaderId::gemv_f32_wbf16;
        const Kernel& k = d.pipelines().get(id, {static_cast<std::uint32_t>(g.m),
                                                 static_cast<std::uint32_t>(g.epi),
                                                 static_cast<std::uint32_t>(g.act),
                                                 vec_ok(g) && g.k % 8 == 0 && g.k > 0 ? 1u : 0u});
        const std::uint64_t groups = cdiv(nout, 4);
        const std::uint32_t gx = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535));
        const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(groups, gx));
        launch(d, k, pc, gx, gy);
        return;
    }

    if (coopmat_eligible(d, g)) {
        const CmCfg& c = pick_cm(g, nout);
        const std::uint32_t wout = g.epi == EPI_SWIGLU ? c.bn / 2 : c.bn;
        const std::uint32_t gx = static_cast<std::uint32_t>(
            g.epi == EPI_SWIGLU ? cdiv(nout, wout) : cdiv(static_cast<std::uint64_t>(g.n), c.bn));
        const std::uint32_t gy = static_cast<std::uint32_t>(cdiv(g.m, c.bm));
        const ShaderId id = g.da == Dtype::FP16 ? ShaderId::gemm_cm_f16 : ShaderId::gemm_cm_bf16;
        auto kernel = [&](bool direct) -> const Kernel& {
            const std::uint32_t spec[] = {c.bm, c.bn, c.bk, c.wm, c.wn, cm_wg(c),
                                          static_cast<std::uint32_t>(g.epi), static_cast<std::uint32_t>(g.act),
                                          g.ta ? 1u : 0u, g.nb ? 1u : 0u, cm_vec_ok(g) ? 1u : 0u,
                                          direct ? 1u : 0u, g.bias ? 1u : 0u};
            return d.pipelines().get(id, spec, static_cast<std::uint32_t>(std::size(spec)), 32);
        };
        // The fragment-form epilogue (stores straight from the accumulator)
        // needs FP16 output, an aligned C, and tiles wholly inside the
        // matrix: the full tiles go in one dispatch, the right and bottom
        // edge strips (if any) in up to two more with the general epilogue.
        const bool direct = g.da == Dtype::FP16 && g.epi != EPI_GEGLU && g.c % 16 == 0 && g.ldc % 8 == 0 &&
                            g.sc % 8 == 0 && (g.bias == 0 || g.bias % 16 == 0);
        const std::uint32_t fx = direct ? static_cast<std::uint32_t>((g.epi == EPI_SWIGLU ? nout : g.n) / wout) : 0;
        const std::uint32_t fy = direct ? static_cast<std::uint32_t>(g.m / c.bm) : 0;
        if (fx > 0 && fy > 0) {
            launch_batched(d, kernel(true), pc, g, fx, fy);
            if (fx < gx) {
                GemmPush p = pc;
                p.tn0 = fx;
                launch_batched(d, kernel(false), p, g, gx - fx, gy);
            }
            if (fy < gy) {
                GemmPush p = pc;
                p.tm0 = fy;
                launch_batched(d, kernel(false), p, g, fx, gy - fy);
            }
        } else {
            launch_batched(d, kernel(false), pc, g, gx, gy);
        }
        return;
    }

    // The SIMT kernel has no GLU epilogue: it writes r (bias included) to an
    // FP32 scratch, unrounded, and a separate pass gates it into C.
    const bool glu = g.epi == EPI_GEGLU || g.epi == EPI_SWIGLU;
    ShaderId id = simt_shader(g);
    if (glu && g.da == g.db && g.dc == g.da && is16(g.da)) {
        id = g.da == Dtype::FP16 ? ShaderId::gemm_simt_f16_c32 : ShaderId::gemm_simt_bf16_c32;
    }
    if (id == ShaderId::kCount) fail(g, "unsupported dtype combination on Vulkan");
    ::brotensor::Tensor r;
    GemmPush p = pc;
    if (glu) {
        if (g.batch != 1) fail(g, "batched GLU epilogue");
        r = ::brotensor::Tensor::empty_on(::brotensor::Device::vulkan(d.index()), g.m, g.n, Dtype::FP32);
        p.c = addr(r.data);
        p.ldc = static_cast<std::uint32_t>(g.n);
    }
    const std::uint64_t tiles_big = cdiv(g.m, kSimtCfgs[0].bm) * cdiv(g.n, kSimtCfgs[0].bn) * g.batch;
    const SimtCfg& c = tiles_big >= 160 ? kSimtCfgs[0] : kSimtCfgs[1];
    const std::uint32_t wg = (c.bm / c.tm) * (c.bn / c.tn);
    const std::uint32_t spec[] = {c.bm, c.bn, c.bk, c.tm, c.tn, wg,
                                  static_cast<std::uint32_t>(glu ? EPI_STORE : g.epi),
                                  static_cast<std::uint32_t>(g.act), g.ta ? 1u : 0u, g.nb ? 1u : 0u};
    const Kernel& k = d.pipelines().get(id, spec, static_cast<std::uint32_t>(std::size(spec)));
    launch_batched(d, k, p, g, static_cast<std::uint32_t>(cdiv(g.n, c.bn)),
                   static_cast<std::uint32_t>(cdiv(g.m, c.bm)));
    if (glu) {
        if (g.ldc != static_cast<int>(nout)) fail(g, "GLU output must be contiguous");
        if (g.epi == EPI_GEGLU) {
            glu_pass(d, GLU_GEGLU_PAIRS, addr(r.data), g.c, g.dc, static_cast<std::uint32_t>(g.m), nout);
        } else {
            glu_pass(d, GLU_SWIGLU, addr(r.data), g.c, g.dc, static_cast<std::uint32_t>(g.m), nout);
        }
    }
}

}  // namespace brotensor::detail::vulkan
