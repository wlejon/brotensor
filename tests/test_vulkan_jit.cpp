// The trace JIT's Vulkan compiler (src/jit/trace_compiler_vulkan.cpp): traced
// expressions compiled to SPIR-V by brass and dispatched on the device's
// stream. `brotensor_test_vulkan --only=jit` runs these; `--bench-jit` the
// fused-vs-replay benchmark (test_vulkan_bench_jit.cpp).
//
// Every result is compared with a double-precision host reference over the
// same (dtype-rounded) inputs, at the CUDA trace suites' tolerances relative
// to the largest reference value: 2e-5 FP32, 2e-3 FP16, 1.6e-2 BF16 (one
// rounding step of the output dtype plus the GPU's approximate exp / rcp /
// rsqrt).
//
// Covered: every FP32/FP16/BF16 combination of an input, a (1, D) row, a
// (1, 1) scalar and the output; misaligned views (the scalar entry), with a
// grid past 65535 workgroups; RMSNorm / LayerNorm with and without gain and
// bias and a modulate / SiLU tail, at vector and scalar widths, LayerNorm on
// rows whose mean dwarfs their spread (the two-pass variance), and a row
// count past 65535 row groups (the 2-D grid); in-place residual + RMSNorm;
// replay inside a graph capture; the pipeline cache; the eager fallback for
// a DAG with no fusion and for BROTENSOR_JIT_VULKAN=eager; the capability
// check's error.

#include "test_vulkan_common.h"

#include <brotensor/cuda_graph.h>
#include <brotensor/jit/trace.h>
#include <brotensor/vulkan.h>

#include "../src/jit/trace_cache.h"

#if BROTENSOR_HAS_VULKAN_TRACE_JIT
#include "vulkan_jit.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace vkt {

#if BROTENSOR_HAS_VULKAN_TRACE_JIT

namespace {

namespace jit = brotensor::jit;
using brotensor::TraceHandle;

const Dtype kDtypes[] = {Dtype::FP32, Dtype::FP16, Dtype::BF16};

float tol_for(Dtype d) {
    switch (d) {
        case Dtype::BF16: return 1.6e-2f;
        case Dtype::FP16: return 2e-3f;
        default: return 2e-5f;
    }
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// Overwrites `t` in place with `v` stored as `dt`.
void restore(Tensor& t, const std::vector<float>& v, Dtype dt) {
    Tensor tmp = upload(v, t.rows, t.cols, dt);
    brotensor::copy_d2d(tmp, 0, t, 0, t.size());
}

void expect_fused(const TraceHandle& h, const std::string& tag, const char* fusion = nullptr) {
    const bool ok = h.launch_count() == 1 && (!fusion || std::string(h.fusion_name()) == fusion);
    if (!ok) {
        std::printf("  FAIL  %s: launches %zu, fusion %s (want 1%s%s)\n", tag.c_str(), h.launch_count(),
                    h.fusion_name(), fusion ? ", " : "", fusion ? fusion : "");
        ++failures();
    }
}

// A (rows, cols) view `off` elements into a fresh buffer of `dt`, filled from
// `v` (so the view is deliberately not 16-byte aligned when off != 0).
Tensor offset_view(Tensor& backing, const std::vector<float>& v, int rows, int cols, Dtype dt, int off) {
    const int esz = dt == Dtype::FP32 ? 4 : 2;
    std::vector<float> all(static_cast<std::size_t>(rows) * cols + off, 0.0f);
    std::copy(v.begin(), v.end(), all.begin() + off);
    backing = upload(all, 1, static_cast<int>(all.size()), dt);
    return Tensor::view(backing.device, static_cast<char*>(backing.data) + off * esz, rows, cols, dt);
}

// ── elementwise: every dtype combination ────────────────────────────────────

void test_dtype_matrix() {
    std::printf("[jit] elementwise: input x row x scalar x output dtypes\n");
    const int N = 97, D = 256;
    for (Dtype da : kDtypes) {
        for (Dtype dg : kDtypes) {
            for (Dtype dout : kDtypes) {
                const auto av = random_values(std::size_t(N) * D, 11, -2, 2, da);
                const auto gv = random_values(D, 12, -1, 1, dg);
                const auto sv = random_values(1, 13, 0.25f, 0.75f, da);
                Tensor a = upload(av, N, D, da), g = upload(gv, 1, D, dg), s = upload(sv, 1, 1, da);
                Tensor dst = Tensor::zeros_on(vk(), N, D, dout);

                jit::begin_trace();
                jit::store(dst, jit::silu(a) * g + s * a);
                TraceHandle h = jit::end_trace();

                std::vector<float> want(av.size());
                for (int r = 0; r < N; ++r) {
                    for (int c = 0; c < D; ++c) {
                        const double x = av[std::size_t(r) * D + c];
                        want[std::size_t(r) * D + c] = static_cast<float>(silu(x) * gv[c] + sv[0] * x);
                    }
                }
                const std::string tag = std::string("silu(a)*row + s*a  a=") + dt_name(da) + " row=" +
                                        dt_name(dg) + " out=" + dt_name(dout);
                expect_scaled(download(dst), want, tol_for(dout), tag);
                expect_fused(h, tag, "elementwise-vec");
            }
        }
    }
}

// ── misaligned views take the scalar entry ──────────────────────────────────

void test_misaligned() {
    std::printf("[jit] elementwise: misaligned views (scalar entry)\n");
    for (Dtype dt : kDtypes) {
        const int N = 33, D = 128;
        const auto av = random_values(std::size_t(N) * D, 21, -1, 1, dt);
        const auto bv = random_values(std::size_t(N) * D, 22, -1, 1, dt);
        Tensor back_a;
        Tensor a = offset_view(back_a, av, N, D, dt, 1);
        Tensor b = upload(bv, N, D, dt);
        jit::begin_trace();
        Tensor out = a * b + 1.0f;
        TraceHandle h = jit::end_trace();
        std::vector<float> want(av.size());
        for (std::size_t i = 0; i < av.size(); ++i) want[i] = static_cast<float>(double(av[i]) * bv[i] + 1.0);
        const std::string tag = std::string("misaligned a*b+1 ") + dt_name(dt);
        expect_scaled(download(out), want, tol_for(dt), tag);
        expect_fused(h, tag, "elementwise-scalar");

        // Rebinding the same trace to aligned buffers (a cache hit) picks
        // the vector entry again.
        Tensor a2 = upload(av, N, D, dt);
        jit::begin_trace();
        Tensor out2 = a2 * b + 1.0f;
        TraceHandle h2 = jit::end_trace();
        VKT_CHECK(h2.is_cache_hit());
        expect_scaled(download(out2), want, tol_for(dt), tag + ", rebound aligned");
        expect_fused(h2, tag + ", rebound aligned", "elementwise-vec");
    }

    // 17M slots through the scalar entry: past 65535 workgroups of 256, so
    // the grid-stride loop covers the rest.
    const int N = 4250, D = 4000;
    const auto av = random_values(std::size_t(N) * D, 23, -1, 1, Dtype::FP32);
    Tensor back;
    Tensor a = offset_view(back, av, N, D, Dtype::FP32, 3);
    jit::begin_trace();
    Tensor out = jit::relu(a) * 2.0f;
    TraceHandle h = jit::end_trace();
    std::vector<float> want(av.size());
    for (std::size_t i = 0; i < av.size(); ++i) want[i] = av[i] > 0 ? 2.0f * av[i] : 0.0f;
    expect_scaled(download(out), want, 2e-5f, "17M-element misaligned relu*2 (grid-stride)");
    expect_fused(h, "17M-element misaligned relu*2", "elementwise-scalar");
}

// ── activations and ops ─────────────────────────────────────────────────────

void test_ops() {
    std::printf("[jit] elementwise: every op\n");
    const int N = 64, D = 96;
    const auto av = random_values(std::size_t(N) * D, 31, -3, 3, Dtype::FP32);
    auto bv = random_values(std::size_t(N) * D, 32, 0.5f, 2, Dtype::FP32);
    Tensor a = upload(av, N, D, Dtype::FP32), b = upload(bv, N, D, Dtype::FP32);
    jit::begin_trace();
    Tensor out = (jit::gelu(a) - jit::tanh(a) / b) * jit::sigmoid(a) + (3.0f - a) * 0.5f -
                 jit::relu(a - 1.0f) / 4.0f;
    TraceHandle h = jit::end_trace();
    std::vector<float> want(av.size());
    for (std::size_t i = 0; i < av.size(); ++i) {
        const double x = av[i], y = bv[i];
        const double gelu = 0.5 * x * (1 + std::tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
        const double sig = 1 / (1 + std::exp(-x));
        want[i] = static_cast<float>((gelu - std::tanh(x) / y) * sig + (3 - x) * 0.5 - std::max(x - 1, 0.0) / 4);
    }
    expect_scaled(download(out), want, 2e-5f, "gelu/tanh/sigmoid/relu/div/sub chain");
    expect_fused(h, "op chain", "elementwise-vec");
}

// ── row-norm ────────────────────────────────────────────────────────────────

enum class Norm { RMS, LN, LNPlain };

// y = modulate(norm(x), scale, shift) (or silu(norm(x)) for LNPlain), with
// optional gain / bias, against a double reference.
void run_norm_case(Norm kind, Dtype dt, int N, int D, float mean_offset, const char* want_fusion,
                   float tol = 0.0f) {
    const auto xv0 = random_values(std::size_t(N) * D, 41 + D, -1.5f, 1.5f, Dtype::FP32);
    std::vector<float> xv(xv0.size());
    for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = round_to(dt, xv0[i] + mean_offset);
    const auto gv = random_values(D, 42, 0.5f, 1.5f, dt);
    const auto bv = random_values(D, 43, -0.3f, 0.3f, dt);
    const auto sv = random_values(D, 44, -0.5f, 0.5f, dt);
    const auto hv = random_values(D, 45, -0.5f, 0.5f, dt);
    Tensor x = upload(xv, N, D, dt), g = upload(gv, 1, D, dt), b = upload(bv, 1, D, dt);
    Tensor sc = upload(sv, 1, D, dt), sh = upload(hv, 1, D, dt);

    jit::begin_trace();
    Tensor y;
    if (kind == Norm::RMS) y = jit::modulate(jit::rms_norm(x, g, 1e-6f), sc, sh);
    if (kind == Norm::LN) y = jit::modulate(jit::layernorm(x, g, b, 1e-6f), sc, sh);
    if (kind == Norm::LNPlain) y = jit::silu(jit::layernorm(x, Tensor(), Tensor(), 1e-6f));
    TraceHandle h = jit::end_trace();

    std::vector<float> want(xv.size());
    for (int r = 0; r < N; ++r) {
        const float* row = xv.data() + std::size_t(r) * D;
        double mean = 0, ss = 0;
        for (int c = 0; c < D; ++c) mean += row[c];
        mean /= D;
        for (int c = 0; c < D; ++c) {
            const double d = kind == Norm::RMS ? row[c] : row[c] - mean;
            ss += d * d;
        }
        const double rstd = 1.0 / std::sqrt(ss / D + 1e-6);
        for (int c = 0; c < D; ++c) {
            double v = (kind == Norm::RMS ? row[c] : row[c] - mean) * rstd;
            if (kind == Norm::LNPlain) {
                v = silu(v);
            } else {
                v = kind == Norm::LN ? v * gv[c] + bv[c] : v * gv[c];
                v = v * (1 + sv[c]) + hv[c];
            }
            want[std::size_t(r) * D + c] = static_cast<float>(v);
        }
    }
    const char* kn = kind == Norm::RMS ? "rms_norm+modulate" : kind == Norm::LN ? "layernorm+modulate"
                                                                                : "silu(layernorm)";
    const std::string tag = std::string(kn) + " " + std::to_string(N) + "x" + std::to_string(D) + " " +
                            dt_name(dt) + (mean_offset != 0 ? " (mean " + std::to_string(int(mean_offset)) + ")" : "");
    expect_scaled(download(y), want, tol > 0 ? tol : tol_for(dt), tag);
    expect_fused(h, tag, want_fusion);
}

void test_row_norm() {
    std::printf("[jit] row-norm: RMS / Mean, gain / bias, modulate tail\n");
    for (Dtype dt : kDtypes) {
        const bool f32 = dt == Dtype::FP32;
        run_norm_case(Norm::RMS, dt, 257, 768, 0, "row-rmsnorm-chain");
        run_norm_case(Norm::LN, dt, 193, 768, 0, "row-layernorm-chain");
        run_norm_case(Norm::LNPlain, dt, 512, 384, 0, "row-layernorm-chain");
        // 100 columns: a 16-bit row is not a whole number of 8-lane vectors
        // (the scalar entry); FP32 takes 4 lanes.
        run_norm_case(Norm::RMS, dt, 1000, 100, 0, f32 ? "row-rmsnorm-chain" : "row-rmsnorm-chain-scalar");
        run_norm_case(Norm::LN, dt, 999, 100, 0, f32 ? "row-layernorm-chain" : "row-layernorm-chain-scalar");
        // Wide rows: several 32-thread groups per row (the shared round).
        run_norm_case(Norm::LN, dt, 16, 12288, 0, "row-layernorm-chain");
    }
    // A row mean of 1000 with a spread of ~1: E[x^2] - E[x]^2 in FP32 loses
    // most of the variance here (sums near 2e9 in steps of 128 against a
    // variance sum of ~1500), the two-pass form does not. What remains is
    // the FP32 mean itself, a few ulp of 1000, hence 1e-4.
    run_norm_case(Norm::LN, Dtype::FP32, 64, 2048, 1000.0f, "row-layernorm-chain", 1e-4f);
    // 600000 rows of 64: eight rows per workgroup, 75000 row groups — past
    // one grid dimension, so the 2-D grid.
    run_norm_case(Norm::RMS, Dtype::FP16, 600000, 64, 0, "row-rmsnorm-chain");
}

// x += p; y = rms_norm(x) in place, at every dtype, then a replay.
void test_residual_rmsnorm() {
    std::printf("[jit] row-norm: in-place residual + RMSNorm\n");
    for (Dtype dt : kDtypes) {
        const int N = 129, D = 640;
        const auto xv = random_values(std::size_t(N) * D, 51, -1, 1, dt);
        const auto pv = random_values(std::size_t(N) * D, 52, -0.7f, 0.7f, dt);
        const auto gv = random_values(D, 53, 0.7f, 1.3f, dt);
        Tensor x = upload(xv, N, D, dt), p = upload(pv, N, D, dt), g = upload(gv, 1, D, dt);
        jit::begin_trace();
        x += p;
        Tensor y = jit::rms_norm(x, g, 1e-6f);
        TraceHandle h = jit::end_trace();

        auto reference = [&](int steps, std::vector<float>& xr, std::vector<float>& yr) {
            xr.assign(xv.begin(), xv.end());
            yr.assign(xv.size(), 0.0f);
            for (int s = 0; s < steps; ++s) {
                for (std::size_t i = 0; i < xr.size(); ++i) xr[i] = round_to(dt, xr[i] + pv[i]);
            }
            for (int r = 0; r < N; ++r) {
                double ss = 0;
                for (int c = 0; c < D; ++c) ss += double(xr[std::size_t(r) * D + c]) * xr[std::size_t(r) * D + c];
                const double rstd = 1.0 / std::sqrt(ss / D + 1e-6);
                for (int c = 0; c < D; ++c) {
                    yr[std::size_t(r) * D + c] = static_cast<float>(xr[std::size_t(r) * D + c] * rstd * gv[c]);
                }
            }
        };
        std::vector<float> xr, yr;
        reference(1, xr, yr);
        const std::string tag = std::string("x += p; rms_norm(x) ") + dt_name(dt);
        expect_scaled(download(x), xr, tol_for(dt), tag + " residual");
        expect_scaled(download(y), yr, tol_for(dt), tag + " norm");
        expect_fused(h, tag, "row-rmsnorm-chain");
        h.execute();
        reference(2, xr, yr);
        expect_scaled(download(x), xr, tol_for(dt), tag + " replay residual");
        expect_scaled(download(y), yr, tol_for(dt), tag + " replay norm");
    }
}

// ── graph capture ───────────────────────────────────────────────────────────

void test_capture() {
    std::printf("[jit] replay inside a graph capture\n");
    if (!brotensor::graph_capture_available(vk())) {
        std::printf("  SKIP  no graph capture on this device\n");
        return;
    }
    const int N = 300, D = 512;
    const Dtype dt = Dtype::BF16;
    const auto xv = random_values(std::size_t(N) * D, 61, -1, 1, dt);
    const auto pv = random_values(std::size_t(N) * D, 62, -0.5f, 0.5f, dt);
    const auto gv = random_values(D, 63, 0.5f, 1.5f, dt);
    Tensor x = upload(xv, N, D, dt), p = upload(pv, N, D, dt), g = upload(gv, 1, D, dt);
    Tensor y = Tensor::empty_on(vk(), N, D, dt);

    jit::begin_trace();   // compiled (and run once) before the capture
    x += p;
    jit::store(y, jit::silu(jit::rms_norm(x, g, 1e-6f)));
    TraceHandle h = jit::end_trace();
    expect_fused(h, "captured trace");
    brotensor::sync(vk());

    brotensor::CudaGraph graph;
    {
        brotensor::CudaGraphCapture cap(vk());
        h.execute();
        // A second trace compiled while the capture records: pipeline
        // creation needs no GPU wait, and its first run lands in the graph.
        jit::begin_trace();
        Tensor z = y * 2.0f;
        jit::store(y, z);
        TraceHandle h2 = jit::end_trace();
        expect_fused(h2, "trace compiled during capture");
        graph = cap.finish();
    }

    auto reference = [&](int steps) {
        std::vector<float> xr(xv), yr(xv.size());
        for (int s = 0; s < steps; ++s) {
            for (std::size_t i = 0; i < xr.size(); ++i) xr[i] = round_to(dt, xr[i] + pv[i]);
        }
        for (int r = 0; r < N; ++r) {
            double ss = 0;
            for (int c = 0; c < D; ++c) ss += double(xr[std::size_t(r) * D + c]) * xr[std::size_t(r) * D + c];
            const double rstd = 1.0 / std::sqrt(ss / D + 1e-6);
            for (int c = 0; c < D; ++c) {
                const double n = round_to(dt, static_cast<float>(silu(xr[std::size_t(r) * D + c] * rstd * gv[c])));
                yr[std::size_t(r) * D + c] = static_cast<float>(2 * n);
            }
        }
        return std::make_pair(xr, yr);
    };
    // Recording ran nothing; x holds the one pre-capture step. Reset it and
    // launch: each launch is one more step.
    restore(x, xv, dt);
    graph.launch();
    brotensor::sync(vk());
    auto [x1, y1] = reference(1);
    expect_scaled(download(x), x1, tol_for(dt), "graph launch 1: residual");
    expect_scaled(download(y), y1, tol_for(dt), "graph launch 1: silu(rms_norm) * 2");
    graph.launch();
    brotensor::sync(vk());
    auto [x2, y2] = reference(2);
    expect_scaled(download(x), x2, tol_for(dt), "graph launch 2: residual");
    expect_scaled(download(y), y2, tol_for(dt), "graph launch 2: silu(rms_norm) * 2");
}

// ── pipeline cache, fallbacks, capability check ────────────────────────────

void test_cache_and_fallback() {
    std::printf("[jit] pipeline cache, eager fallback, capability check\n");
    const int N = 64, D = 128;
    const auto av = random_values(std::size_t(N) * D, 71, -1, 1, Dtype::FP32);
    Tensor a = upload(av, N, D, Dtype::FP32);
    auto trace = [&] {
        jit::begin_trace();
        Tensor out = jit::sigmoid(a) * a - 0.125f;
        TraceHandle h = jit::end_trace();
        return std::make_pair(std::move(out), h);
    };
    std::vector<float> want(av.size());
    for (std::size_t i = 0; i < av.size(); ++i) want[i] = static_cast<float>(av[i] / (1 + std::exp(-double(av[i]))) - 0.125);

    auto [o1, h1] = trace();
    const std::size_t modules = brotensor::detail::vulkan::jit_module_count();
    const std::size_t pipes = brotensor::vulkan::stream_stats(vk()).pipelines;
    // A cold trace cache compiles again: the same SPIR-V, so the same
    // interned module and the pipelines already in the cache.
    brotensor::jit::TraceCache::instance().clear();
    auto [o2, h2] = trace();
    VKT_CHECK(!h2.is_cache_hit());
    VKT_CHECK(brotensor::detail::vulkan::jit_module_count() == modules);
    VKT_CHECK(brotensor::vulkan::stream_stats(vk()).pipelines == pipes);
    expect_scaled(download(o2), want, 2e-5f, "recompiled trace reuses its pipelines");
    expect_fused(h2, "recompiled trace");
    VKT_CHECK(h1.compile_us() > 0.0);

    // BROTENSOR_JIT_VULKAN=eager: the op-by-op replay, same numbers.
    brotensor::jit::TraceCache::instance().clear();
    set_env("BROTENSOR_JIT_VULKAN", "eager");
    auto [o3, h3] = trace();
    set_env("BROTENSOR_JIT_VULKAN", nullptr);
    brotensor::jit::TraceCache::instance().clear();
    VKT_CHECK(std::string(h3.fusion_name()) == "eager-unfused" && h3.launch_count() > 1);
    expect_scaled(download(o3), want, 2e-5f, "BROTENSOR_JIT_VULKAN=eager replays op by op");

    // Two reductions in one trace: no fusion, so the replay runs it.
    jit::begin_trace();
    Tensor y = jit::rms_norm(jit::layernorm(a, Tensor(), Tensor(), 1e-6f), Tensor(), 1e-6f);
    TraceHandle h4 = jit::end_trace();
    VKT_CHECK(std::string(h4.fusion_name()) == "eager-unfused" && h4.launch_count() >= 2);
    std::vector<float> want4(av.size());
    for (int r = 0; r < N; ++r) {
        const float* row = av.data() + std::size_t(r) * D;
        double m = 0, v = 0;
        for (int c = 0; c < D; ++c) m += row[c];
        m /= D;
        for (int c = 0; c < D; ++c) v += (row[c] - m) * (row[c] - m);
        const double rs = 1 / std::sqrt(v / D + 1e-6);
        double ss = 0;
        for (int c = 0; c < D; ++c) ss += (row[c] - m) * rs * (row[c] - m) * rs;
        const double rr = 1 / std::sqrt(ss / D + 1e-6);
        for (int c = 0; c < D; ++c) want4[std::size_t(r) * D + c] = static_cast<float>((row[c] - m) * rs * rr);
    }
    expect_scaled(download(y), want4, 1e-4f, "rms_norm(layernorm(x)) falls back to the replay");

    // The capability check names everything the device does not enable.
    brass::target::SpirvKernel k;
    k.entry = "probe";
    k.words = {0x07230203u, 0x00010500u, 0, 16, 0};
    k.capabilities = {"Shader", "Int64", "Int64Atomics", "AtomicFloat32AddEXT"};
    k.push_constant_bytes = 1024;
    auto& ctx = brotensor::detail::vulkan::device(0);
    const std::string missing = brotensor::detail::vulkan::jit_missing_for(ctx, k);
    VKT_CHECK(missing.find("Int64Atomics") != std::string::npos);
    VKT_CHECK(missing.find("AtomicFloat32AddEXT") != std::string::npos);
    VKT_CHECK(missing.find("push constants take 1024") != std::string::npos);
    VKT_CHECK(missing.find("Int64 ") == std::string::npos);
    std::string err;
    try {
        brotensor::detail::vulkan::jit_pipeline(ctx, k, 256);
    } catch (const std::exception& e) {
        err = e.what();
    }
    VKT_CHECK(err.find("cannot run on") != std::string::npos && err.find("Int64Atomics") != std::string::npos);
    std::printf("  PASS  capability check (%zu lines)\n",
                static_cast<std::size_t>(std::count(missing.begin(), missing.end(), '\n')));
}

}  // namespace

void run_jit_tests() {
    // A compile failure must fail the test, not fall back quietly.
    set_env("BROTENSOR_JIT_STRICT", "1");
    test_dtype_matrix();
    test_misaligned();
    test_ops();
    test_row_norm();
    test_residual_rmsnorm();
    test_capture();
    test_cache_and_fallback();
    set_env("BROTENSOR_JIT_STRICT", nullptr);
}

#else

void run_jit_tests() { std::printf("[jit] SKIP  built without the Vulkan trace compiler\n"); }

#endif

}  // namespace vkt
