#include <brotensor/kernel_cache.h>

#include "kernel_disk_cache.h"
#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brotensor {

namespace {

std::atomic<int> g_autotune_override{-1}; // -1 = check env BROTENSOR_AUTOTUNE

bool env_autotune_on() noexcept {
    if (const char* v = std::getenv("BROTENSOR_AUTOTUNE")) {
        return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
    }
    return false;
}

struct AutotuneScopeGuard {
    int prev;
    explicit AutotuneScopeGuard(bool enable)
        : prev(g_autotune_override.exchange(enable ? 1 : 0, std::memory_order_relaxed)) {}
    ~AutotuneScopeGuard() {
        g_autotune_override.store(prev, std::memory_order_relaxed);
    }
    AutotuneScopeGuard(const AutotuneScopeGuard&) = delete;
    AutotuneScopeGuard& operator=(const AutotuneScopeGuard&) = delete;
};

} // namespace

void set_autotune_enabled(bool enabled) noexcept {
    g_autotune_override.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

bool autotune_enabled() noexcept {
    const int ov = g_autotune_override.load(std::memory_order_relaxed);
    if (ov >= 0) return ov != 0;
    return env_autotune_on();
}

std::string kernel_cache_dir() {
    return jit::KernelDiskCache::instance().cache_dir().string();
}

KernelCacheStats kernel_cache_stats() {
    auto& kdc = jit::KernelDiskCache::instance();
    const auto s = kdc.stats();
    KernelCacheStats out;
    out.dir = kdc.cache_dir().string();
    out.hits = s.hits;
    out.misses = s.misses;
    out.writes = s.writes;
    out.corruptions = s.corruptions;
    out.entries = kdc.entry_count();
    return out;
}

void flush_kernel_caches(Device dev) {
    jit::flush_pipeline_caches_for(dev);
}

WarmupStats warmup_kernels(Device dev, const WarmupOptions& opts) {
    const std::string& p = opts.profile;
    const bool is_minimal = (p == "minimal");
    const bool is_default = (p == "default" || p.empty());
    const bool is_llm = (p == "llm");
    const bool is_diff = (p == "diffusion");
    if (!is_minimal && !is_default && !is_llm && !is_diff) {
        throw std::invalid_argument("warmup_kernels: unknown profile '" + p +
                                    "' (expected minimal, default, llm, or diffusion)");
    }

    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t writes_before = jit::KernelDiskCache::instance().stats().writes;
    AutotuneScopeGuard at_guard(opts.autotune);

    // 1. Minimal warmup: traced elementwise + RMSNorm + basic matmul/matmul_abt.
    {
        Tensor a = Tensor::zeros_on(dev, 8, 32, Dtype::FP32);
        Tensor b = Tensor::zeros_on(dev, 8, 32, Dtype::FP32);
        Tensor c = Tensor::zeros_on(dev, 8, 32, Dtype::FP32);
        Tensor gamma = Tensor::zeros_on(dev, 1, 32, Dtype::FP32);
        Tensor y_ew, y_rn;
        {
            TraceScope ts;
            y_ew = silu(a * b + c);
        }
        {
            TraceScope ts;
            y_rn = rms_norm(a + b, gamma, 1e-5f);
        }
        (void)y_ew;
        (void)y_rn;

        Tensor mm_a = Tensor::zeros_on(dev, 32, 32, Dtype::FP32);
        Tensor mm_b = Tensor::zeros_on(dev, 32, 32, Dtype::FP32);
        Tensor mm_c;
        matmul(mm_a, mm_b, mm_c);

        if (dev.is_gpu()) {
            Tensor a16 = Tensor::zeros_on(dev, 128, 128, Dtype::FP16);
            Tensor b16 = Tensor::zeros_on(dev, 128, 128, Dtype::FP16);
            Tensor c16 = Tensor::zeros_on(dev, 128, 128, Dtype::FP16);
            matmul_abt(a16, b16, c16, 1, 128, 128, 128, 0, 0, 0, nullptr, kLinearActNone);
        }
    }

    // 2. Default & shared variants.
    if (!is_minimal) {
        Tensor x = Tensor::zeros_on(dev, 16, 64, Dtype::FP32);
        Tensor g = Tensor::zeros_on(dev, 1, 64, Dtype::FP32);
        Tensor b = Tensor::zeros_on(dev, 1, 64, Dtype::FP32);
        Tensor y, xh, m, r;
        layernorm_forward_batched_with_caches(x, g, b, y, xh, m, r, 1e-5f);
        softmax_forward(x, y);

        Tensor t_gelu;
        {
            TraceScope ts;
            t_gelu = gelu(x + b);
        }
        (void)t_gelu;

        Tensor cx = Tensor::zeros_on(dev, 1, 16 * 8 * 8, Dtype::FP32);
        Tensor cw = Tensor::zeros_on(dev, 16, 16 * 9, Dtype::FP32);
        Tensor cy;
        conv2d_forward(cx, cw, nullptr, 1, 16, 8, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, cy);

        Tensor q = Tensor::zeros_on(dev, 32, 4 * 64, Dtype::FP32);
        Tensor k = Tensor::zeros_on(dev, 32, 4 * 64, Dtype::FP32);
        Tensor v = Tensor::zeros_on(dev, 32, 4 * 64, Dtype::FP32);
        Tensor ao;
        flash_attention_gqa_forward(q, k, v, nullptr, 4, 4, false, ao);
    }

    // 3. LLM profile: RMSNorm op, SwiGLU linear epilogue (skinny + MMA), decode GEMV, causal FA.
    if (is_llm) {
        const auto warm_llm_dtype = [&](Dtype wdt) {
            Tensor x = Tensor::zeros_on(dev, 16, 128, wdt);
            Tensor g = Tensor::zeros_on(dev, 1, 128, wdt);
            Tensor y;
            rms_norm_forward(x, g, 1e-5f, y);

            Tensor w_glu = Tensor::zeros_on(dev, 256, 128, wdt);
            Tensor out_glu;
            linear_forward_batched_ex(w_glu, nullptr, x, kLinearActNone, kLinearEpiSwiglu, nullptr, out_glu);

            if (dev.is_gpu()) {
                Tensor x_wide = Tensor::zeros_on(dev, 128, 128, wdt);
                Tensor out_wide;
                linear_forward_batched_ex(w_glu, nullptr, x_wide, kLinearActNone, kLinearEpiSwiglu, nullptr, out_wide);
                Tensor b128 = Tensor::zeros_on(dev, 128, 128, wdt);
                Tensor c128 = Tensor::zeros_on(dev, 128, 128, wdt);
                matmul_abt(x_wide, b128, c128, 1, 128, 128, 128, 0, 0, 0, nullptr, kLinearActNone);
            }

            Tensor dec_a = Tensor::zeros_on(dev, 1, 128, wdt);
            Tensor dec_b = Tensor::zeros_on(dev, 128, 128, wdt);
            Tensor dec_c = Tensor::zeros_on(dev, 1, 128, wdt);
            if (wdt == Dtype::FP16 || wdt == Dtype::BF16) {
                matmul_abt(dec_a, dec_b, dec_c, 1, 1, 128, 128, 0, 0, 0, nullptr, kLinearActNone);
            } else {
                matmul(dec_a, dec_b, dec_c);
            }

            for (int hd : {64, 128}) {
                Tensor q = Tensor::zeros_on(dev, 64, 4 * hd, wdt);
                Tensor k = Tensor::zeros_on(dev, 64, 4 * hd, wdt);
                Tensor v = Tensor::zeros_on(dev, 64, 4 * hd, wdt);
                Tensor ao;
                flash_attention_gqa_forward(q, k, v, nullptr, 4, 4, true, ao);
            }
        };
        if (dev.is_gpu()) {
            warm_llm_dtype(Dtype::FP16);
            warm_llm_dtype(Dtype::BF16);
        } else {
            warm_llm_dtype(Dtype::FP32);
        }
    }

    // 4. Diffusion profile: GroupNorm, diffusion sampler steps, timestep embedding, GeGLU + conv2d.
    if (is_diff) {
        Tensor x = Tensor::zeros_on(dev, 1, 16 * 8 * 8, Dtype::FP32);
        Tensor g = Tensor::zeros_on(dev, 1, 16, Dtype::FP32);
        Tensor b = Tensor::zeros_on(dev, 1, 16, Dtype::FP32);
        Tensor y;
        group_norm_forward(x, g, b, 1, 16, 8, 8, 4, 1e-5f, y);

        Tensor eps = Tensor::zeros_on(dev, 1, 16 * 8 * 8, Dtype::FP32);
        Tensor xp = Tensor::zeros_on(dev, 1, 16 * 8 * 8, Dtype::FP32);
        ddim_step(x, eps, 0.9f, 0.8f, 0.0f, xp);
        euler_step(x, eps, 1.0f, 0.5f, xp);

        Tensor ts = Tensor::zeros_on(dev, 2, 1, Dtype::FP32);
        Tensor emb;
        timestep_embedding(ts, 64, 10000.0f, emb);

        if (dev.is_gpu()) {
            for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
                Tensor gx = Tensor::zeros_on(dev, 128, 128, dt);
                Tensor gw = Tensor::zeros_on(dev, 256, 128, dt);
                Tensor gy;
                linear_forward_batched_ex(gw, nullptr, gx, kLinearActNone, kLinearEpiGeglu, nullptr, gy);
            }
            Tensor cx = Tensor::zeros_on(dev, 1, 16 * 8 * 8, Dtype::FP16);
            Tensor cw = Tensor::zeros_on(dev, 16, 16 * 9, Dtype::FP16);
            Tensor cy;
            conv2d_forward(cx, cw, nullptr, 1, 16, 8, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, cy);
        }
    }

    brotensor::sync(dev);
    if (opts.flush) {
        flush_kernel_caches(dev);
    }

    WarmupStats stats;
    const std::size_t writes_after = jit::KernelDiskCache::instance().stats().writes;
    stats.kernels_compiled = (writes_after >= writes_before) ? (writes_after - writes_before) : 0;
    stats.autotune_entries = 0;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return stats;
}

bool export_kernel_bundle(const std::string& bundle_path,
                          Device dev,
                          std::string* err) {
    flush_kernel_caches(dev);
    const std::string meta = std::string("device=") + device_name(dev) +
                             ";dir=" + kernel_cache_dir();
    return jit::KernelDiskCache::instance().export_bundle(bundle_path, meta, nullptr, nullptr, err);
}

bool import_kernel_bundle(const std::string& bundle_path,
                          BundleImportReport* report,
                          std::string* err) {
    std::size_t imp = 0, skip = 0, bytes = 0;
    const bool ok = jit::KernelDiskCache::instance().import_bundle(
        bundle_path, nullptr, &imp, &skip, &bytes, err);
    if (report) {
        report->entries_imported = imp;
        report->entries_skipped = skip;
        report->bytes = bytes;
    }
    return ok;
}

} // namespace brotensor
