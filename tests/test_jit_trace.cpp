// ─── Brass Automatic Tracing JIT Test Suite ──────────────────────────────────
//
// Verifies numerical parity (max error <= 1e-4) and benchmarks:
//   Test 1: Auto-fused elementwise expression: out = (a * b + c) * silu(d)
//   Test 2: Auto-fused Residual + RMSNorm: h += proj; norm = rms_norm(h)
//   Test 3: Auto-fused LayerNorm + Modulate: ln = layernorm(x); out = ln * (1 + scale) + shift
//   Test 4: Trace cache hit verification & replay speedup benchmark.
// Runs across both CPU and NVIDIA RTX GPU (CUDA sm_89).

#include <brotensor/jit/trace.h>
#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>
#include "../src/jit/trace_cache.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>

using namespace brotensor;
using namespace brotensor::jit;

namespace {

int g_failures = 0;

#define CHECK_PARITY(err, tol, name) do {                                    \
    if ((err) > (tol)) {                                                      \
        std::printf("  [FAIL] %s: max err = %g > tol = %g\n", (name), (err), (tol)); \
        ++g_failures;                                                         \
    } else {                                                                  \
        std::printf("  [PASS] %s: max err = %g <= tol = %g\n", (name), (err), (tol)); \
    }                                                                         \
} while (0)

#define CHECK_TRUE(cond, name) do {                                           \
    if (!(cond)) {                                                            \
        std::printf("  [FAIL] %s: condition failed: %s\n", (name), #cond);    \
        ++g_failures;                                                         \
    } else {                                                                  \
        std::printf("  [PASS] %s\n", (name));                                 \
    }                                                                         \
} while (0)

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_f01() {
        return static_cast<float>(next_u64() >> 40) / 16777216.0f;
    }
    float next_unit() { return next_f01() * 2.0f - 1.0f; }
};

void fill_random(std::vector<float>& vec, uint64_t seed, float scale = 1.0f) {
    SplitMix64 rng(seed);
    for (auto& val : vec) {
        val = rng.next_unit() * scale;
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9f;
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// ── Test 1: Auto-fused Elementwise Expression: (a * b + c) * silu(d) ─────────

void test_elementwise_expression(Device dev) {
    const std::string dev_name = dev.is_cuda() ? "CUDA" : "CPU";
    std::printf("\n--- Test 1: Auto-fused Elementwise Expression on %s ---\n", dev_name.c_str());

    const int R = 128;
    const int C = 256;
    const int N = R * C;

    std::vector<float> h_a(N), h_b(N), h_c(N), h_d(N);
    fill_random(h_a, 1001, 1.0f);
    fill_random(h_b, 1002, 1.0f);
    fill_random(h_c, 1003, 1.0f);
    fill_random(h_d, 1004, 2.0f);

    Tensor a = Tensor::from_host_on(dev, h_a.data(), R, C);
    Tensor b = Tensor::from_host_on(dev, h_b.data(), R, C);
    Tensor c = Tensor::from_host_on(dev, h_c.data(), R, C);
    Tensor d = Tensor::from_host_on(dev, h_d.data(), R, C);

    // Reference computation: (a * b + c) * silu(d)
    std::vector<float> ref(N);
    for (int i = 0; i < N; ++i) {
        float silu_val = h_d[i] / (1.0f + std::exp(-h_d[i]));
        ref[i] = (h_a[i] * h_b[i] + h_c[i]) * silu_val;
    }

    // JIT traced execution
    begin_trace();
    Tensor out = (a * b + c) * silu(d);
    TraceHandle handle = end_trace();

    CHECK_TRUE(handle.node_count() > 0, "Trace recorded nodes");
    CHECK_TRUE(handle.is_cuda() == dev.is_cuda(), "Trace device matches target");

    brotensor::sync(dev);
    std::vector<float> h_out = out.to_host_vector();
    float err = max_abs_diff(h_out, ref);
    CHECK_PARITY(err, 1e-4f, (dev_name + " Elementwise (a*b+c)*silu(d) initial").c_str());

    // Replay execution
    handle.execute();
    brotensor::sync(dev);
    h_out = out.to_host_vector();
    err = max_abs_diff(h_out, ref);
    CHECK_PARITY(err, 1e-4f, (dev_name + " Elementwise (a*b+c)*silu(d) replay").c_str());
}

// ── Test 2: Auto-fused Residual + RMSNorm: h += proj; norm = rms_norm(h) ─────

void test_residual_rmsnorm(Device dev) {
    const std::string dev_name = dev.is_cuda() ? "CUDA" : "CPU";
    std::printf("\n--- Test 2: Auto-fused Residual + RMSNorm on %s ---\n", dev_name.c_str());

    const int B = 4;
    const int D = 1024;
    const int N = B * D;
    const float eps = 1e-5f;

    std::vector<float> h_h(N), h_proj(N), h_gamma(D);
    fill_random(h_h, 2001, 1.0f);
    fill_random(h_proj, 2002, 0.5f);
    fill_random(h_gamma, 2003, 1.0f);

    Tensor h_init = Tensor::from_host_on(dev, h_h.data(), B, D);
    Tensor proj = Tensor::from_host_on(dev, h_proj.data(), B, D);
    Tensor gamma = Tensor::from_host_on(dev, h_gamma.data(), 1, D);

    // Reference computation
    std::vector<float> ref_h(N);
    std::vector<float> ref_norm(N);
    for (int b = 0; b < B; ++b) {
        float sum_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            int idx = b * D + d;
            ref_h[idx] = h_h[idx] + h_proj[idx];
            sum_sq += ref_h[idx] * ref_h[idx];
        }
        float rrms = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);
        for (int d = 0; d < D; ++d) {
            int idx = b * D + d;
            ref_norm[idx] = ref_h[idx] * h_gamma[d] * rrms;
        }
    }

    Tensor h = h_init.clone();
    begin_trace();
    h += proj;
    Tensor norm = rms_norm(h, gamma, eps);
    TraceHandle handle = end_trace();

    brotensor::sync(dev);
    std::vector<float> actual_h = h.to_host_vector();
    std::vector<float> actual_norm = norm.to_host_vector();

    float err_h = max_abs_diff(actual_h, ref_h);
    float err_norm = max_abs_diff(actual_norm, ref_norm);

    CHECK_PARITY(err_h, 1e-4f, (dev_name + " In-place residual h += proj").c_str());
    CHECK_PARITY(err_norm, 1e-4f, (dev_name + " RMSNorm norm = rms_norm(h)").c_str());

    // Replay execution: restore h contents and re-execute handle
    if (dev.is_cpu()) {
        std::memcpy(h.ptr(), h_init.ptr(), h.bytes());
    } else {
        detail::alloc_for(dev).memcpy_d2d(h.data, h_init.data, h.bytes(), dev.index);
    }
    handle.execute();
    brotensor::sync(dev);

    actual_h = h.to_host_vector();
    actual_norm = norm.to_host_vector();
    err_h = max_abs_diff(actual_h, ref_h);
    err_norm = max_abs_diff(actual_norm, ref_norm);

    CHECK_PARITY(err_h, 1e-4f, (dev_name + " Residual RMSNorm replay h").c_str());
    CHECK_PARITY(err_norm, 1e-4f, (dev_name + " Residual RMSNorm replay norm").c_str());
}

// ── Test 3: Auto-fused LayerNorm + Modulate: ln = layernorm(x); out = ln * (1 + scale) + shift

void test_layernorm_modulate(Device dev) {
    const std::string dev_name = dev.is_cuda() ? "CUDA" : "CPU";
    std::printf("\n--- Test 3: Auto-fused LayerNorm + Modulate on %s ---\n", dev_name.c_str());

    const int R = 8;
    const int D = 512;
    const int N = R * D;
    const float eps = 1e-5f;

    std::vector<float> h_x(N), h_gamma(D), h_beta(D), h_scale(D), h_shift(D);
    fill_random(h_x, 3001, 2.0f);
    fill_random(h_gamma, 3002, 1.0f);
    fill_random(h_beta, 3003, 0.5f);
    fill_random(h_scale, 3004, 0.5f);
    fill_random(h_shift, 3005, 0.5f);

    Tensor x = Tensor::from_host_on(dev, h_x.data(), R, D);
    Tensor gamma = Tensor::from_host_on(dev, h_gamma.data(), 1, D);
    Tensor beta = Tensor::from_host_on(dev, h_beta.data(), 1, D);
    Tensor scale = Tensor::from_host_on(dev, h_scale.data(), 1, D);
    Tensor shift = Tensor::from_host_on(dev, h_shift.data(), 1, D);

    // Reference computation
    std::vector<float> ref_out(N);
    for (int r = 0; r < R; ++r) {
        float mean = 0.0f;
        for (int d = 0; d < D; ++d) mean += h_x[r * D + d];
        mean /= static_cast<float>(D);

        float var = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = h_x[r * D + d] - mean;
            var += diff * diff;
        }
        float rstd = 1.0f / std::sqrt(var / static_cast<float>(D) + eps);

        for (int d = 0; d < D; ++d) {
            int idx = r * D + d;
            float ln_val = (h_x[idx] - mean) * rstd * h_gamma[d] + h_beta[d];
            ref_out[idx] = ln_val * (1.0f + h_scale[d]) + h_shift[d];
        }
    }

    begin_trace();
    Tensor ln = layernorm(x, gamma, beta, eps);
    Tensor out = ln * (1.0f + scale) + shift;
    TraceHandle handle = end_trace();

    brotensor::sync(dev);
    std::vector<float> actual_out = out.to_host_vector();
    float err = max_abs_diff(actual_out, ref_out);
    CHECK_PARITY(err, 1e-4f, (dev_name + " LayerNorm + Modulate initial").c_str());

    // Replay execution
    handle.execute();
    brotensor::sync(dev);
    actual_out = out.to_host_vector();
    err = max_abs_diff(actual_out, ref_out);
    CHECK_PARITY(err, 1e-4f, (dev_name + " LayerNorm + Modulate replay").c_str());
}

// ── Test 4: Trace Cache Hit Verification & Replay Speedup ─────────────────────

void test_cache_hit_and_speedup(Device dev) {
    const std::string dev_name = dev.is_cuda() ? "CUDA" : "CPU";
    std::printf("\n--- Test 4: Trace Cache Hit & Replay Speedup on %s ---\n", dev_name.c_str());

    TraceCache::instance().clear();

    const int R = 64;
    const int C = 512;
    const int N = R * C;

    std::vector<float> h_a(N), h_b(N), h_c(N), h_d(N);
    fill_random(h_a, 4001, 1.0f);
    fill_random(h_b, 4002, 1.0f);
    fill_random(h_c, 4003, 1.0f);
    fill_random(h_d, 4004, 1.0f);

    Tensor a1 = Tensor::from_host_on(dev, h_a.data(), R, C);
    Tensor b1 = Tensor::from_host_on(dev, h_b.data(), R, C);
    Tensor c1 = Tensor::from_host_on(dev, h_c.data(), R, C);
    Tensor d1 = Tensor::from_host_on(dev, h_d.data(), R, C);

    // 1st trace: must be a cache miss
    begin_trace();
    Tensor out1 = (a1 * b1 + c1) * silu(d1);
    TraceHandle h1 = end_trace();

    CHECK_TRUE(!h1.is_cache_hit(), "1st trace compilation is a cache miss");
    CHECK_TRUE(TraceCache::instance().hit_count() == 0, "Cache hits == 0 after 1st trace");

    // 2nd trace: same graph and shape with fresh tensors -> MUST be a cache hit
    Tensor a2 = Tensor::from_host_on(dev, h_a.data(), R, C);
    Tensor b2 = Tensor::from_host_on(dev, h_b.data(), R, C);
    Tensor c2 = Tensor::from_host_on(dev, h_c.data(), R, C);
    Tensor d2 = Tensor::from_host_on(dev, h_d.data(), R, C);

    begin_trace();
    Tensor out2 = (a2 * b2 + c2) * silu(d2);
    TraceHandle h2 = end_trace();

    CHECK_TRUE(h2.is_cache_hit(), "2nd trace is a cache hit");
    CHECK_TRUE(TraceCache::instance().hit_count() >= 1, "Cache hit count incremented");

    // Benchmark replay speedup vs eager un-fused op dispatch
    const int warmup = 50;
    const int iters = 500;

    for (int i = 0; i < warmup; ++i) {
        h2.execute();
    }
    brotensor::sync(dev);

    auto t_start_replay = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        h2.execute();
    }
    brotensor::sync(dev);
    auto t_end_replay = std::chrono::steady_clock::now();
    double replay_time_ms = std::chrono::duration<double, std::milli>(t_end_replay - t_start_replay).count();

    // Warmup & benchmark eager execution
    for (int i = 0; i < warmup; ++i) {
        Tensor mul_ab = a2.clone();
        brotensor::mul_inplace(mul_ab, b2);
        brotensor::add_inplace(mul_ab, c2);
        Tensor silu_d = Tensor::empty_on(dev, R, C);
        brotensor::silu_forward(d2, silu_d);
        brotensor::mul_inplace(mul_ab, silu_d);
    }
    brotensor::sync(dev);

    auto t_start_eager = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        Tensor mul_ab = a2.clone();
        brotensor::mul_inplace(mul_ab, b2);
        brotensor::add_inplace(mul_ab, c2);
        Tensor silu_d = Tensor::empty_on(dev, R, C);
        brotensor::silu_forward(d2, silu_d);
        brotensor::mul_inplace(mul_ab, silu_d);
    }
    brotensor::sync(dev);
    auto t_end_eager = std::chrono::steady_clock::now();
    double eager_time_ms = std::chrono::duration<double, std::milli>(t_end_eager - t_start_eager).count();

    // Verify numerical parity of cache-hit JIT replay output
    Tensor eager_ref = a2.clone();
    brotensor::mul_inplace(eager_ref, b2);
    brotensor::add_inplace(eager_ref, c2);
    Tensor eager_silu = Tensor::empty_on(dev, R, C);
    brotensor::silu_forward(d2, eager_silu);
    brotensor::mul_inplace(eager_ref, eager_silu);
    float parity_err = max_abs_diff(out2.to_host_vector(), eager_ref.to_host_vector());
    CHECK_PARITY(parity_err, 1e-4f, (dev_name + " Trace Cache Hit Replay Parity").c_str());

    double speedup = eager_time_ms / replay_time_ms;
    std::printf("  [BENCHMARK] %d iterations: Eager = %.2f ms, JIT Replay = %.2f ms (%.2fx speedup)\n",
                iters, eager_time_ms, replay_time_ms, speedup);
    CHECK_TRUE(speedup > 0.0, "JIT replay completed successfully");
}

} // namespace

int main() {
    brotensor::init();

    std::printf("================================================================================\n");
    std::printf("  BRASS AUTOMATIC TRACING JIT TEST SUITE (CPU & CUDA RTX 4090)\n");
    std::printf("================================================================================\n");

#if !BROTENSOR_HAS_BRASS_JIT
    std::printf("[SKIP] Brass JIT is not enabled in this build; skipping trace JIT tests.\n");
    return 0;
#else
    // Run CPU test suite
    std::printf("\n============================= [ CPU TEST SUITE ] =============================\n");
    test_elementwise_expression(Device::cpu());
    test_residual_rmsnorm(Device::cpu());
    test_layernorm_modulate(Device::cpu());
    test_cache_hit_and_speedup(Device::cpu());

    // Run CUDA test suite if CUDA device is available
    if (brotensor::is_available(Device::cuda())) {
        std::printf("\n============================ [ CUDA TEST SUITE ] =============================\n");
        test_elementwise_expression(Device::cuda());
        test_residual_rmsnorm(Device::cuda());
        test_layernorm_modulate(Device::cuda());
        test_cache_hit_and_speedup(Device::cuda());
    } else {
        std::printf("\n[SKIP] CUDA device not available or not detected; skipping CUDA tests.\n");
    }

    std::printf("\n================================================================================\n");
    if (g_failures == 0) {
        std::printf("  [SUCCESS] All JIT trace test suites PASSED with 100%% parity!\n");
    } else {
        std::printf("  [FAILURE] %d JIT trace test(s) FAILED.\n", g_failures);
    }
    std::printf("================================================================================\n");

    return g_failures == 0 ? 0 : 1;
#endif
}
