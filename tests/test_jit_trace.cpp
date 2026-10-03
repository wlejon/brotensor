// ─── Brass Automatic Tracing JIT Test Suite ──────────────────────────────────
//
// Verifies numerical parity (max error <= 1e-4) and benchmarks:
//   Test 1: Auto-fused elementwise expression: out = (a * b + c) * silu(d)
//   Test 2: Auto-fused Residual + RMSNorm: h += proj; norm = rms_norm(h)
//   Test 3: Auto-fused LayerNorm + Modulate: ln = layernorm(x); out = ln * (1 + scale) + shift
//   Test 4: Trace cache hit verification & replay speedup benchmark.
//   Test 6: A Vulkan trace never reaches the host code generator.
//   Test 7: Eager divide / modulate and a traced divide on Vulkan.
//   Tests 1-7 run on Vulkan: through the SPIR-V trace compiler when the build
//   has it (BROTENSOR_HAS_VULKAN_TRACE_JIT; every fused pattern is then one
//   launch), else as an op-by-op replay (tests 1-4, 6, 7).
// Runs across CPU, NVIDIA RTX GPU (CUDA sm_89) and Vulkan.

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

std::string device_label(Device dev) {
    if (dev.is_cuda()) return "CUDA";
    if (dev.is_vulkan()) return "Vulkan";
    return "CPU";
}

// Whether `dev`'s traces compile to one fused kernel: CPU and CUDA always,
// Vulkan with the SPIR-V trace compiler.
bool fuses(Device dev) {
#if BROTENSOR_HAS_VULKAN_TRACE_JIT
    (void)dev;
    return true;
#else
    return !dev.is_vulkan();
#endif
}

void check_one_launch(Device dev, const TraceHandle& h, const std::string& what) {
    if (!fuses(dev)) return;
    const std::string name = what + " is one launch (" + h.fusion_name() + ")";
    CHECK_TRUE(h.launch_count() == 1, name.c_str());
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
    const std::string dev_name = device_label(dev);
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
    check_one_launch(dev, handle, dev_name + " (a*b+c)*silu(d)");

    // Replay execution
    handle.execute();
    brotensor::sync(dev);
    h_out = out.to_host_vector();
    err = max_abs_diff(h_out, ref);
    CHECK_PARITY(err, 1e-4f, (dev_name + " Elementwise (a*b+c)*silu(d) replay").c_str());
}

// ── Test 2: Auto-fused Residual + RMSNorm: h += proj; norm = rms_norm(h) ─────

void test_residual_rmsnorm(Device dev) {
    const std::string dev_name = device_label(dev);
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
    check_one_launch(dev, handle, dev_name + " h += proj; rms_norm(h)");

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
    const std::string dev_name = device_label(dev);
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
    check_one_launch(dev, handle, dev_name + " layernorm + modulate");

    // Replay execution
    handle.execute();
    brotensor::sync(dev);
    actual_out = out.to_host_vector();
    err = max_abs_diff(actual_out, ref_out);
    CHECK_PARITY(err, 1e-4f, (dev_name + " LayerNorm + Modulate replay").c_str());
}

// ── Test 4: Trace Cache Hit Verification & Replay Speedup ─────────────────────

void test_cache_hit_and_speedup(Device dev) {
    const std::string dev_name = device_label(dev);
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

// ── Test 5: what tracing allocates, and what it must not ────────────────────
//
// The tracer's whole reason for keeping symbolic tensors is that a fused
// kernel never reads an intermediate out of memory, so it must never buy one.
// brotensor::alloc_stats() counts every backend allocation, so the claim is
// checkable rather than merely intended: an expression that store()s into a
// buffer the caller already owns has to cost ZERO bytes, and an expression
// whose result the caller keeps has to cost exactly the one output buffer.

void test_trace_allocates_nothing(Device dev) {
    const std::string dev_name = device_label(dev);
    std::printf("\n--- Test 5: Trace-time allocation on %s ---\n", dev_name.c_str());

    const int R = 512;
    const int D = 1024;
    const std::size_t N = static_cast<std::size_t>(R) * D;
    const std::uint64_t buf_bytes = static_cast<std::uint64_t>(N) * sizeof(float);

    // Every operand is the full (R, D) shape: the CPU trace compiler has no
    // broadcast form, and this test has to measure the same expression on
    // both backends.
    std::vector<float> h_x(N), h_y(N);
    fill_random(h_x, 5001, 1.5f);
    fill_random(h_y, 5002, 1.0f);

    Tensor x = Tensor::from_host_on(dev, h_x.data(), R, D);
    Tensor y = Tensor::from_host_on(dev, h_y.data(), R, D);
    Tensor dst = Tensor::empty_on(dev, R, D);

    // ── store()d result: nothing at all ──────────────────────────────────
    //
    // silu(x) * y + x is three traced ops. Eagerly that is three full-size
    // (R, D) buffers; traced it must be none — every one of those values is
    // consumed inside the single kernel and never lands in memory.
    const AllocStats before_store = brotensor::alloc_stats();
    begin_trace();
    jit::store(dst, jit::silu(x) * y + x);
    TraceHandle h_store = end_trace();
    const AllocStats after_store = brotensor::alloc_stats();

    const std::uint64_t store_bytes = after_store.bytes - before_store.bytes;
    const std::uint64_t store_count = after_store.count - before_store.count;
    std::printf("  store()d trace: %llu allocation(s), %llu bytes "
                "(eager intermediates would be %llu)\n",
                static_cast<unsigned long long>(store_count),
                static_cast<unsigned long long>(store_bytes),
                static_cast<unsigned long long>(3 * buf_bytes));
    CHECK_TRUE(store_count == 0 && store_bytes == 0,
               (dev_name + " store()d trace allocates no intermediates").c_str());

    // and it still computes the right thing.
    brotensor::sync(dev);
    Tensor ref = Tensor::empty_on(dev, R, D);
    brotensor::silu_forward(x, ref);
    brotensor::mul_inplace(ref, y);
    brotensor::add_inplace(ref, x);
    brotensor::sync(dev);
    CHECK_PARITY(max_abs_diff(dst.to_host_vector(), ref.to_host_vector()), 1e-4f,
                 (dev_name + " store()d trace is numerically correct").c_str());
    CHECK_TRUE(h_store.launch_count() == 1,
               (dev_name + " store()d trace is one launch").c_str());

    // ── kept result: exactly one output buffer ───────────────────────────
    //
    // Same expression, but the caller keeps the value instead of storing it.
    // end_trace() has to hand that one live-at-end node a real buffer — and
    // nothing else: silu(x) and silu(x)*y are still interior to the kernel.
    const AllocStats before_keep = brotensor::alloc_stats();
    begin_trace();
    Tensor kept = jit::silu(x) * y + x;
    TraceHandle h_keep = end_trace();
    const AllocStats after_keep = brotensor::alloc_stats();

    const std::uint64_t keep_bytes = after_keep.bytes - before_keep.bytes;
    const std::uint64_t keep_count = after_keep.count - before_keep.count;
    std::printf("  kept trace:     %llu allocation(s), %llu bytes "
                "(one output buffer is %llu)\n",
                static_cast<unsigned long long>(keep_count),
                static_cast<unsigned long long>(keep_bytes),
                static_cast<unsigned long long>(buf_bytes));
    CHECK_TRUE(keep_count == 1 && keep_bytes == buf_bytes,
               (dev_name + " kept trace allocates exactly its one output").c_str());
    CHECK_TRUE(kept.data != nullptr && kept.jit_slot < 0 &&
                   kept.rows == R && kept.cols == D,
               (dev_name + " kept result owns a real buffer after end_trace()").c_str());

    brotensor::sync(dev);
    CHECK_PARITY(max_abs_diff(kept.to_host_vector(), ref.to_host_vector()), 1e-4f,
                 (dev_name + " kept trace is numerically correct").c_str());

    // ── copying a traced intermediate is rejected ────────────────────────
    //
    // It stands for a node of the trace and owns nothing, so there is no
    // meaning to give a copy. Better a clear throw than an empty tensor.
    bool threw = false;
    begin_trace();
    try {
        Tensor sym = jit::silu(x);
        Tensor copy = sym;   // no meaning: throws
        (void)copy;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    abort_trace();
    CHECK_TRUE(threw, (dev_name + " copying a traced intermediate throws").c_str());

    // ── abort_trace() leaves held tensors sane ───────────────────────────
    //
    // The eager fallback path: nothing was compiled, so nothing holds a
    // value, and every symbolic tensor the caller kept is an empty tensor it
    // can assign over.
    begin_trace();
    Tensor aborted = jit::silu(x) * y;
    abort_trace();
    CHECK_TRUE(aborted.data == nullptr && aborted.jit_slot < 0 && aborted.empty(),
               (dev_name + " abort_trace() neutralises held intermediates").c_str());
    aborted = Tensor::empty_on(dev, R, D);   // and it is an ordinary tensor again
    brotensor::silu_forward(x, aborted);
    brotensor::sync(dev);
    CHECK_TRUE(aborted.rows == R, (dev_name + " a neutralised tensor is reusable").c_str());
}

// ── Test 6: a Vulkan trace never reaches the host code generator ───────────
//
// The CPU compiler emits host code that walks raw pointers, so handing it a Vulkan DAG would dereference
// buffer device addresses (not host pointers at all), with no ordering
// against the device stream: it would read an input the GPU has not
// finished writing. The trace has to stay on the device (the SPIR-V
// compiler, or the op-by-op replay), ordered on its stream.
void test_trace_stays_on_device(Device dev) {
    const std::string dn = device_label(dev);
    std::printf("\n--- Test 6: %s trace stays on the device ---\n", dn.c_str());

    const int R = 2048;
    const int C = 2048;
    const int N = R * C;
    std::vector<float> ones(N, 1.0f);
    Tensor x = Tensor::from_host_on(dev, ones.data(), R, C);
    Tensor y = Tensor::from_host_on(dev, ones.data(), R, C);
    brotensor::sync(dev);

    // Queue enough device work on x that it is certainly still running when
    // the trace is compiled and first executed: x ends up 2^10.
    for (int i = 0; i < 10; ++i) brotensor::scale_inplace(x, 2.0f);

    begin_trace();
    Tensor out = x * y + 1.0f;
    TraceHandle handle = end_trace();
    brotensor::sync(dev);

    CHECK_TRUE(std::string(handle.fusion_name()) != "cpu-avx2-fused",
               (dn + " trace is not compiled by the CPU code generator").c_str());
    std::vector<float> got = out.to_host_vector();
    float err = 0.0f;
    for (int i = 0; i < N; ++i) err = std::fmax(err, std::fabs(got[i] - 1025.0f));
    CHECK_PARITY(err, 0.0f, (dn + " trace sees the device work queued before it").c_str());

    for (int i = 0; i < 2; ++i) brotensor::scale_inplace(x, 0.5f);
    handle.execute();
    brotensor::sync(dev);
    got = out.to_host_vector();
    err = 0.0f;
    for (int i = 0; i < N; ++i) err = std::fmax(err, std::fabs(got[i] - 257.0f));
    CHECK_PARITY(err, 0.0f, (dn + " trace replay is ordered on the stream").c_str());
}

// ── Test 7: eager `/` and modulate() off the host ──────────────────────────
//
// Outside a trace these run immediately. Off CUDA they go through
// div_inplace / modulate. The traced divide replays through the same op.
void test_eager_div_modulate(Device dev) {
    std::printf("\n--- Test 7: eager divide / modulate on %s ---\n", device_label(dev).c_str());

    const int R = 37;
    const int C = 96;
    const int N = R * C;
    std::vector<float> h_a(N), h_b(N), h_scale(C), h_shift(C);
    fill_random(h_a, 7001, 2.0f);
    fill_random(h_b, 7002, 1.0f);
    for (auto& v : h_b) v += (v < 0.0f ? -0.5f : 0.5f);   // keep |b| >= 0.5
    fill_random(h_scale, 7003, 0.5f);
    fill_random(h_shift, 7004, 0.5f);

    Tensor a = Tensor::from_host_on(dev, h_a.data(), R, C);
    Tensor b = Tensor::from_host_on(dev, h_b.data(), R, C);
    Tensor scale = Tensor::from_host_on(dev, h_scale.data(), 1, C);
    Tensor shift = Tensor::from_host_on(dev, h_shift.data(), 1, C);

    std::vector<float> ref_div(N), ref_mod(N);
    for (int i = 0; i < N; ++i) {
        ref_div[i] = h_a[i] / h_b[i];
        ref_mod[i] = h_a[i] * (1.0f + h_scale[i % C]) + h_shift[i % C];
    }

    Tensor q = a / b;
    Tensor m = jit::modulate(a, scale, shift);
    brotensor::sync(dev);
    CHECK_PARITY(max_abs_diff(q.to_host_vector(), ref_div), 1e-5f,
                 (device_label(dev) + " eager a / b").c_str());
    CHECK_PARITY(max_abs_diff(m.to_host_vector(), ref_mod), 1e-5f,
                 (device_label(dev) + " eager modulate").c_str());

    begin_trace();
    Tensor t = (a / b) * scale + 1.0f;
    TraceHandle h = end_trace();
    brotensor::sync(dev);
    std::vector<float> ref_t(N);
    for (int i = 0; i < N; ++i) ref_t[i] = ref_div[i] * h_scale[i % C] + 1.0f;
    CHECK_PARITY(max_abs_diff(t.to_host_vector(), ref_t), 1e-5f,
                 (device_label(dev) + " traced (a / b) * row + 1").c_str());
    (void)h;
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
    test_trace_allocates_nothing(Device::cpu());

    // Run CUDA test suite if CUDA device is available
    if (brotensor::is_available(Device::cuda())) {
        std::printf("\n============================ [ CUDA TEST SUITE ] =============================\n");
        test_elementwise_expression(Device::cuda());
        test_residual_rmsnorm(Device::cuda());
        test_layernorm_modulate(Device::cuda());
        test_cache_hit_and_speedup(Device::cuda());
        test_trace_allocates_nothing(Device::cuda());
    } else {
        std::printf("\n[SKIP] CUDA device not available or not detected; skipping CUDA tests.\n");
    }
    // Vulkan: the SPIR-V trace compiler when the build has it, else an
    // op-by-op replay through the dispatched ops (src/jit/trace_eager.cpp).
    // Test 5 measures the fused kernel's allocation and launch count, which
    // an unfused replay does not have.
    if (brotensor::is_available(Device::vulkan())) {
        std::printf("\n============================ [ VULKAN TEST SUITE ] ===========================\n");
        test_elementwise_expression(Device::vulkan());
        test_residual_rmsnorm(Device::vulkan());
        test_layernorm_modulate(Device::vulkan());
        test_cache_hit_and_speedup(Device::vulkan());
        if (fuses(Device::vulkan())) test_trace_allocates_nothing(Device::vulkan());
        test_trace_stays_on_device(Device::vulkan());
        test_eager_div_modulate(Device::vulkan());
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
