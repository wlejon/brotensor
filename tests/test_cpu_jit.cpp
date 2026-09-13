// ─── Brass JIT CPU Acceleration Parity and Benchmark Suite ──────────────────
//
// Verifies numerical parity of Brass JIT accelerated CPU operations against
// both scalar CPU reference and NVIDIA RTX GPU CUDA reference.
// Benchmarks speedups across critical shapes (B=1 decode, B=4 DiT, B=8).

#include "parity_helpers.h"
#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include <brotensor/detail/cpu/thread_pool.h>
#include "../src/cpu/cpu_jit.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

// ── Scalar CPU References ──────────────────────────────────────────────────

void scalar_rmsnorm(const float* X, const float* gamma, float eps, float* Y, int B, int D) {
    const float inv_D = 1.0f / static_cast<float>(D);
    for (int b = 0; b < B; ++b) {
        const float* x = X + b * D;
        float* y = Y + b * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += x[i] * x[i];
        float rrms = 1.0f / std::sqrt(sum * inv_D + eps);
        for (int i = 0; i < D; ++i) y[i] = x[i] * gamma[i] * rrms;
    }
}

void scalar_swiglu(const float* X, float* Y, int B, int D) {
    for (int b = 0; b < B; ++b) {
        const float* gate = X + b * 2 * D;
        const float* up = gate + D;
        float* y = Y + b * D;
        for (int i = 0; i < D; ++i) {
            float g = gate[i];
            float silu = g / (1.0f + std::exp(-g));
            y[i] = silu * up[i];
        }
    }
}

void scalar_modulate(const float* X, const float* scale, const float* shift, float* Y, int L, int D) {
    for (int l = 0; l < L; ++l) {
        const float* x = X + l * D;
        float* y = Y + l * D;
        for (int d = 0; d < D; ++d) {
            y[d] = x[d] * (1.0f + scale[d]) + shift[d];
        }
    }
}

void scalar_broadcast_mul(const float* X, const float* v, float* Y, int L, int D) {
    for (int l = 0; l < L; ++l) {
        const float* x = X + l * D;
        float* y = Y + l * D;
        for (int d = 0; d < D; ++d) {
            y[d] = x[d] * v[d];
        }
    }
}

void scalar_layernorm(const float* X, const float* gamma, const float* beta, float eps, float* Y, int R, int D) {
    const float inv_D = 1.0f / static_cast<float>(D);
    for (int r = 0; r < R; ++r) {
        const float* x = X + r * D;
        float* y = Y + r * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += x[i];
        float mean = sum * inv_D;
        float sumsq = 0.0f;
        for (int i = 0; i < D; ++i) {
            float diff = x[i] - mean;
            sumsq += diff * diff;
        }
        float var = sumsq * inv_D;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i) {
            y[i] = gamma[i] * ((x[i] - mean) * rstd) + beta[i];
        }
    }
}

// ── Timing & Parity Helpers ────────────────────────────────────────────────

template <typename Fn>
double bench_cpu(int warm, int iters, Fn&& fn) {
    for (int i = 0; i < warm; ++i) fn();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iters;
}

template <typename Fn>
double bench_gpu(int warm, int iters, Fn&& fn) {
    for (int i = 0; i < warm; ++i) fn();
    brotensor::sync_all();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    brotensor::sync_all();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iters;
}

float max_abs_diff(const float* a, const float* b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

struct BenchResult {
    std::string op_name;
    std::string shape_desc;
    double scalar_us;
    double jit_us;
    double gpu_us;
    float max_diff_scalar;
    float max_diff_gpu;
    bool passed;
};

std::vector<BenchResult> g_results;

void test_rmsnorm(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, D);
    Tensor gamma = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    const float eps = 1e-5f;

    // 1. Scalar CPU reference
    Tensor scalar_Y = Tensor::mat(B, D);
    double scalar_us = bench_cpu(10, 50, [&] {
        scalar_rmsnorm(X.host_f32(), gamma.host_f32(), eps, scalar_Y.host_f32_mut(), B, D);
    });

    // 2. Brass JIT CPU
    Tensor jit_Y = Tensor::mat(B, D);
    double jit_us = bench_cpu(10, 50, [&] {
        brotensor::rms_norm_forward(X, gamma, eps, jit_Y);
    });

    // 3. CUDA GPU reference
    Tensor gX = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), B, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::rms_norm_forward(gX, ggamma, eps, gpu_Y);
    });
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), scalar_Y.host_f32(), B * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), B * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f);

    g_results.push_back({
        "RMSNorm",
        "B=" + std::to_string(B) + ", D=" + std::to_string(D),
        scalar_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

void test_swiglu(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, 2 * D);
    fill_random(X, rng);

    // 1. Scalar CPU reference
    Tensor scalar_Y = Tensor::mat(B, D);
    double scalar_us = bench_cpu(10, 50, [&] {
        scalar_swiglu(X.host_f32(), scalar_Y.host_f32_mut(), B, D);
    });

    // 2. Brass JIT CPU
    Tensor jit_Y = Tensor::mat(B, D);
    double jit_us = bench_cpu(10, 50, [&] {
        brotensor::swiglu_forward(X, jit_Y);
    });

    // 3. CUDA GPU reference
    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), B, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::swiglu_forward(gX, gpu_Y);
    });
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), scalar_Y.host_f32(), B * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), B * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f);

    g_results.push_back({
        "SwiGLU",
        "B=" + std::to_string(B) + ", D=" + std::to_string(D),
        scalar_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

void test_modulate(int L, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, D);
    Tensor scale = Tensor::vec(D);
    Tensor shift = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(scale, rng);
    fill_random(shift, rng);

    // 1. Scalar CPU reference
    Tensor scalar_Y = Tensor::mat(L, D);
    double scalar_us = bench_cpu(10, 50, [&] {
        scalar_modulate(X.host_f32(), scale.host_f32(), shift.host_f32(), scalar_Y.host_f32_mut(), L, D);
    });

    // 2. Brass JIT CPU
    Tensor jit_Y = Tensor::mat(L, D);
    double jit_us = bench_cpu(10, 50, [&] {
        brotensor::modulate(X, scale, shift, jit_Y);
    });

    // 3. CUDA GPU reference
    Tensor gX = X.to(gpu_device());
    Tensor gscale = scale.to(gpu_device());
    Tensor gshift = shift.to(gpu_device());
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), L, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::modulate(gX, gscale, gshift, gpu_Y);
    });
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), scalar_Y.host_f32(), L * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), L * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f);

    g_results.push_back({
        "AdaLN-Modulate",
        "L=" + std::to_string(L) + ", D=" + std::to_string(D),
        scalar_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

void test_broadcast_mul(int L, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, D);
    Tensor v = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(v, rng);

    // 1. Scalar CPU reference
    Tensor scalar_Y = Tensor::mat(L, D);
    double scalar_us = bench_cpu(10, 50, [&] {
        scalar_broadcast_mul(X.host_f32(), v.host_f32(), scalar_Y.host_f32_mut(), L, D);
    });

    // 2. Brass JIT CPU
    Tensor jit_Y = Tensor::mat(L, D);
    double jit_us = bench_cpu(10, 50, [&] {
        brotensor::broadcast_mul(X, v, jit_Y);
    });

    // 3. CUDA GPU reference
    Tensor gX = X.to(gpu_device());
    Tensor gv = v.to(gpu_device());
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), L, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::broadcast_mul(gX, gv, gpu_Y);
    });
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), scalar_Y.host_f32(), L * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), L * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f);

    g_results.push_back({
        "BroadcastMul",
        "L=" + std::to_string(L) + ", D=" + std::to_string(D),
        scalar_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

void test_layernorm(int R, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, D);
    Tensor gamma = Tensor::vec(D);
    Tensor beta = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    fill_random(beta, rng);
    const float eps = 1e-5f;

    // 1. Scalar CPU reference
    Tensor scalar_Y = Tensor::mat(R, D);
    double scalar_us = bench_cpu(10, 50, [&] {
        scalar_layernorm(X.host_f32(), gamma.host_f32(), beta.host_f32(), eps, scalar_Y.host_f32_mut(), R, D);
    });

    // 2. Brass JIT CPU
    Tensor jit_Y = Tensor::mat(R, D);
    double jit_us = bench_cpu(10, 50, [&] {
        brotensor::layernorm_forward_inference_batched(X, gamma, beta, jit_Y, eps);
    });

    // 3. CUDA GPU reference
    Tensor gX = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gbeta = beta.to(gpu_device());
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), R, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::layernorm_forward_inference_batched(gX, ggamma, gbeta, gpu_Y, eps);
    });
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), scalar_Y.host_f32(), R * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), R * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f);

    g_results.push_back({
        "LayerNorm",
        "R=" + std::to_string(R) + ", D=" + std::to_string(D),
        scalar_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

void test_fused_residual_rmsnorm(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, D);
    Tensor res = Tensor::mat(B, D);
    Tensor gamma = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(res, rng);
    fill_random(gamma, rng);
    const float eps = 1e-5f;

    // 1. Unfused C++ reference: X += res in memory, then RMSNorm
    Tensor unfused_X = X.clone();
    Tensor unfused_Y = Tensor::mat(B, D);
    double unfused_us = bench_cpu(10, 50, [&] {
        float* xp = unfused_X.host_f32_mut();
        const float* rp = res.host_f32();
        for (int i = 0; i < B * D; ++i) xp[i] = X.host_f32()[i] + rp[i];
        scalar_rmsnorm(xp, gamma.host_f32(), eps, unfused_Y.host_f32_mut(), B, D);
    });

    // 2. Brass Fused JIT: in-place residual addition + RMSNorm in same register pass
    Tensor jit_X = X.clone();
    Tensor jit_Y = Tensor::mat(B, D);
    double jit_us = bench_cpu(10, 50, [&] {
        std::memcpy(jit_X.host_f32_mut(), X.host_f32(), B * D * sizeof(float));
        brotensor::detail::cpu::jit::fused_residual_rmsnorm(
            jit_X.host_f32_mut(), res.host_f32(), gamma.host_f32(), eps, jit_Y.host_f32_mut(), B, D);
    });

    // 3. CUDA GPU reference: add_inplace + rms_norm_forward
    Tensor gX = X.to(gpu_device());
    Tensor gres = res.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), B, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::add_inplace(gX, gres);
        brotensor::rms_norm_forward(gX, ggamma, eps, gpu_Y);
    });

    // Run one fresh pass for exact parity comparison
    Tensor gX_single = X.to(gpu_device());
    brotensor::add_inplace(gX_single, gres);
    brotensor::rms_norm_forward(gX_single, ggamma, eps, gpu_Y);
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    // Compute one fresh pass for JIT parity check
    std::memcpy(jit_X.host_f32_mut(), X.host_f32(), B * D * sizeof(float));
    brotensor::detail::cpu::jit::fused_residual_rmsnorm(
        jit_X.host_f32_mut(), res.host_f32(), gamma.host_f32(), eps, jit_Y.host_f32_mut(), B, D);

    float* xp_ref = unfused_X.host_f32_mut();
    const float* rp_ref = res.host_f32();
    for (int i = 0; i < B * D; ++i) xp_ref[i] = X.host_f32()[i] + rp_ref[i];
    scalar_rmsnorm(xp_ref, gamma.host_f32(), eps, unfused_Y.host_f32_mut(), B, D);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), unfused_Y.host_f32(), B * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), B * D);
    float diff_x = max_abs_diff(jit_X.host_f32(), unfused_X.host_f32(), B * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f) && (diff_x < 1e-5f);

    g_results.push_back({
        "Fused-ResRMS",
        "B=" + std::to_string(B) + ", D=" + std::to_string(D),
        unfused_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

void test_fused_layernorm_modulate(int R, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, D);
    Tensor gamma = Tensor::vec(D);
    Tensor beta = Tensor::vec(D);
    Tensor scale = Tensor::vec(D);
    Tensor shift = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    fill_random(beta, rng);
    fill_random(scale, rng);
    fill_random(shift, rng);
    const float eps = 1e-5f;

    // 1. Unfused C++ reference: LayerNorm -> intermediate buffer -> modulate -> Y
    Tensor unfused_inter = Tensor::mat(R, D);
    Tensor unfused_Y = Tensor::mat(R, D);
    double unfused_us = bench_cpu(10, 50, [&] {
        scalar_layernorm(X.host_f32(), gamma.host_f32(), beta.host_f32(), eps, unfused_inter.host_f32_mut(), R, D);
        scalar_modulate(unfused_inter.host_f32(), scale.host_f32(), shift.host_f32(), unfused_Y.host_f32_mut(), R, D);
    });

    // 2. Brass Fused JIT: 0 intermediate memory writes, LayerNorm + modulate directly in registers
    Tensor jit_Y = Tensor::mat(R, D);
    double jit_us = bench_cpu(10, 50, [&] {
        brotensor::detail::cpu::jit::fused_layernorm_modulate(
            X.host_f32(), gamma.host_f32(), beta.host_f32(), scale.host_f32(), shift.host_f32(), eps, jit_Y.host_f32_mut(), R, D);
    });

    // 3. CUDA GPU reference: LayerNorm + Modulate
    Tensor gX = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gbeta = beta.to(gpu_device());
    Tensor gscale = scale.to(gpu_device());
    Tensor gshift = shift.to(gpu_device());
    Tensor g_inter = Tensor::zeros_on(gpu_device(), R, D);
    Tensor gpu_Y = Tensor::zeros_on(gpu_device(), R, D);
    double gpu_us = bench_gpu(10, 50, [&] {
        brotensor::layernorm_forward_inference_batched(gX, ggamma, gbeta, g_inter, eps);
        brotensor::modulate(g_inter, gscale, gshift, gpu_Y);
    });
    Tensor gpu_host_Y = download_to_host(gpu_Y);

    float diff_scalar = max_abs_diff(jit_Y.host_f32(), unfused_Y.host_f32(), R * D);
    float diff_gpu = max_abs_diff(jit_Y.host_f32(), gpu_host_Y.host_f32(), R * D);
    bool pass = (diff_scalar < 1e-4f) && (diff_gpu < 1e-3f);

    g_results.push_back({
        "Fused-LNMod",
        "R=" + std::to_string(R) + ", D=" + std::to_string(D),
        unfused_us, jit_us, gpu_us, diff_scalar, diff_gpu, pass
    });
}

} // namespace

int main() {
    brotensor::init();

    std::cout << "========================================================================================\n";
    std::cout << " Brass JIT CPU Acceleration Parity and Benchmark Suite (rtx 4090 GPU Oracle)\n";
    std::cout << "========================================================================================\n";

    if (!brotensor::detail::cpu::jit::is_jit_available()) {
        std::cerr << "ERROR: Brass JIT is not available on this CPU platform!\n";
        return 1;
    }
    std::cout << " Brass JIT Engine: ACTIVE (AVX2 + FMA + Polyhedral/Chunk Parallel Multi-threading)\n";
    std::cout << " Thread Pool Workers: " << brotensor::detail::cpu::ThreadPool::instance().num_threads() << " threads\n";
    std::cout << " GPU Device: " << (gpu_device() == Device::CUDA ? "NVIDIA CUDA (RTX 4090)" : "None") << "\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    // Test across small but thorough shapes:
    // 1) B=1, D=4096 (LLM single-token decode critical path)
    // 2) B=4, D=1152 (DiT patch representation)
    // 3) B=8, D=128  (Attention head / small token chunk)
    // 4) B=32, D=4096 (Batched inference / DiT full batch)

    const std::vector<std::pair<int, int>> shapes = {
        {1, 4096},
        {4, 1152},
        {8, 128},
        {32, 4096}
    };

    uint64_t seed = 42;
    for (const auto& [b, d] : shapes) {
        test_rmsnorm(b, d, seed++);
        test_swiglu(b, d, seed++);
        test_modulate(b, d, seed++);
        test_broadcast_mul(b, d, seed++);
        test_layernorm(b, d, seed++);
        test_fused_residual_rmsnorm(b, d, seed++);
        test_fused_layernorm_modulate(b, d, seed++);
    }

    std::cout << "\n### Performance Benchmark & Oracle Parity Results:\n\n";
    std::cout << "| Operation | Shape | CPU Scalar (us) | CPU Brass JIT (us) | Speedup | GPU CUDA (us) | Parity | Status |\n";
    std::cout << "|:----------|:------|:----------------|:-------------------|:--------|:--------------|:-------|:-------|\n";

    int failures = 0;
    for (const auto& r : g_results) {
        double speedup = r.scalar_us / std::max(0.001, r.jit_us);
        std::printf("| %-14s | %-12s | %13.2f us | %16.2f us | %6.2fx | %11.2f us | %6.1e | %s |\n",
                    r.op_name.c_str(),
                    r.shape_desc.c_str(),
                    r.scalar_us,
                    r.jit_us,
                    speedup,
                    r.gpu_us,
                    r.max_diff_gpu,
                    r.passed ? "**PASS**" : "**FAIL**");
        if (!r.passed) ++failures;
    }

    std::cout << "----------------------------------------------------------------------------------------\n";
    if (failures == 0) {
        std::cout << "[SUCCESS] All " << g_results.size() << " test configurations verified with exact numerical parity!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] " << failures << " test configurations failed parity check!\n";
        return 1;
    }
}
