// ─── Brass CUDA Driver JIT Benchmark & Numerical Parity Suite ────────────────
//
// Verifies numerical parity (<= 1e-5 max error) and benchmarks speedups of
// Brass JIT accelerated PTX kernels vs static CUDA kernels on NVIDIA RTX GPUs
// across LLM and DiT inference shapes (B in {1, 8, 32, 64} at D in {4096, 1152}).

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include "../src/cuda/cuda_jit.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <cassert>

using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

int g_failures = 0;

#define CHECK_PARITY(err, tol, name) do {                                    \
    if ((err) > (tol)) {                                                      \
        std::printf("  [FAIL] %s: max err = %g > tol = %g\n", (name), (err), (tol)); \
        ++g_failures;                                                         \
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

float max_abs_error(const float* a, const float* b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

template <typename Fn>
double time_cuda_us(Fn&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (static_cast<double>(ms) * 1000.0) / static_cast<double>(iters);
}

// ── 1. Fused Residual RMSNorm Benchmark & Parity ───────────────────────────

void test_fused_residual_rmsnorm(int B, int D) {
    const float eps = 1e-5f;
    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(D);

    std::vector<float> h_x(total);
    std::vector<float> h_res(total);
    std::vector<float> h_gamma(D);
    fill_random(h_x, 0x12345ULL + static_cast<uint64_t>(B * D));
    fill_random(h_res, 0x54321ULL + static_cast<uint64_t>(B * D));
    fill_random(h_gamma, 0xABCDEULL + static_cast<uint64_t>(D));

    // Calculate baseline expected X_sum on host and upload to GPU
    std::vector<float> h_x_sum(total);
    for (size_t i = 0; i < total; ++i) {
        h_x_sum[i] = h_x[i] + h_res[i];
    }

    Tensor g_x_ref = Tensor::from_host_on(Device::CUDA, h_x_sum.data(), B, D);
    Tensor g_gamma = Tensor::from_host_on(Device::CUDA, h_gamma.data(), D, 1);
    Tensor g_y_ref = Tensor::zeros_on(Device::CUDA, B, D);

    // Static CUDA baseline
    brotensor::rms_norm_forward(g_x_ref, g_gamma, eps, g_y_ref);
    cudaDeviceSynchronize();

    // PTX JIT setup
    Tensor g_x_jit = Tensor::from_host_on(Device::CUDA, h_x.data(), B, D);
    Tensor g_res = Tensor::from_host_on(Device::CUDA, h_res.data(), B, D);
    Tensor g_y_jit = Tensor::zeros_on(Device::CUDA, B, D);

    brotensor::detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
        static_cast<float*>(g_x_jit.data),
        static_cast<const float*>(g_res.data),
        static_cast<const float*>(g_gamma.data),
        static_cast<float*>(g_y_jit.data),
        B, D, eps
    );
    cudaDeviceSynchronize();

    // Verify numerical parity
    std::vector<float> h_y_ref(total);
    std::vector<float> h_y_jit(total);
    std::vector<float> h_x_jit_out(total);

    cudaMemcpy(h_y_ref.data(), g_y_ref.data, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_jit.data(), g_y_jit.data, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_jit_out.data(), g_x_jit.data, total * sizeof(float), cudaMemcpyDeviceToHost);

    float err_x = max_abs_error(h_x_sum.data(), h_x_jit_out.data(), total);
    float err_y = max_abs_error(h_y_ref.data(), h_y_jit.data(), total);

    CHECK_PARITY(err_x, 1e-5f, "Residual X in-place update");
    CHECK_PARITY(err_y, 1e-5f, "FusedResidualRMSNorm Y");

    // Benchmark
    double t_base = time_cuda_us([&]() {
        brotensor::rms_norm_forward(g_x_ref, g_gamma, eps, g_y_ref);
    });

    double t_jit = time_cuda_us([&]() {
        brotensor::detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
            static_cast<float*>(g_x_jit.data),
            static_cast<const float*>(g_res.data),
            static_cast<const float*>(g_gamma.data),
            static_cast<float*>(g_y_jit.data),
            B, D, eps
        );
    });

    double speedup = t_base / t_jit;
    std::printf("  B=%-3d D=%-5d | Static CUDA: %7.2f us | PTX JIT: %7.2f us | Speedup: %5.2fx | MaxErr: %8.2e\n",
                B, D, t_base, t_jit, speedup, std::max(err_x, err_y));
}

// ── 2. Fused LayerNorm + Modulate Benchmark & Parity ───────────────────────

void test_fused_layernorm_modulate(int R, int D) {
    const float eps = 1e-5f;
    const size_t total = static_cast<size_t>(R) * static_cast<size_t>(D);

    std::vector<float> h_x(total);
    std::vector<float> h_gamma(D);
    std::vector<float> h_beta(D);
    std::vector<float> h_scale(D);
    std::vector<float> h_shift(D);

    fill_random(h_x, 0x111ULL + static_cast<uint64_t>(R * D));
    fill_random(h_gamma, 0x222ULL + static_cast<uint64_t>(D));
    fill_random(h_beta, 0x333ULL + static_cast<uint64_t>(D));
    fill_random(h_scale, 0x444ULL + static_cast<uint64_t>(D));
    fill_random(h_shift, 0x555ULL + static_cast<uint64_t>(D));

    Tensor g_x = Tensor::from_host_on(Device::CUDA, h_x.data(), R, D);
    Tensor g_gamma = Tensor::from_host_on(Device::CUDA, h_gamma.data(), D, 1);
    Tensor g_beta = Tensor::from_host_on(Device::CUDA, h_beta.data(), D, 1);
    Tensor g_scale = Tensor::from_host_on(Device::CUDA, h_scale.data(), D, 1);
    Tensor g_shift = Tensor::from_host_on(Device::CUDA, h_shift.data(), D, 1);

    // Static baseline: 2 passes (LayerNorm + AdaLN Modulate)
    Tensor g_normed = Tensor::zeros_on(Device::CUDA, R, D);
    Tensor g_y_ref = Tensor::zeros_on(Device::CUDA, R, D);

    brotensor::layernorm_forward_inference_batched(g_x, g_gamma, g_beta, g_normed, eps);
    brotensor::modulate(g_normed, g_scale, g_shift, g_y_ref);
    cudaDeviceSynchronize();

    // PTX JIT: single pass directly into registers and output Y
    Tensor g_y_jit = Tensor::zeros_on(Device::CUDA, R, D);

    brotensor::detail::cuda::jit::launch_fused_layernorm_modulate_ptx(
        static_cast<const float*>(g_x.data),
        static_cast<const float*>(g_gamma.data),
        static_cast<const float*>(g_beta.data),
        static_cast<const float*>(g_scale.data),
        static_cast<const float*>(g_shift.data),
        static_cast<float*>(g_y_jit.data),
        R, D, eps
    );
    cudaDeviceSynchronize();

    // Numerical parity
    std::vector<float> h_y_ref(total);
    std::vector<float> h_y_jit(total);
    cudaMemcpy(h_y_ref.data(), g_y_ref.data, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_jit.data(), g_y_jit.data, total * sizeof(float), cudaMemcpyDeviceToHost);

    float err_y = max_abs_error(h_y_ref.data(), h_y_jit.data(), total);
    CHECK_PARITY(err_y, 1e-5f, "FusedLayerNormModulate Y");

    // Benchmark
    double t_base = time_cuda_us([&]() {
        brotensor::layernorm_forward_inference_batched(g_x, g_gamma, g_beta, g_normed, eps);
        brotensor::modulate(g_normed, g_scale, g_shift, g_y_ref);
    });

    double t_jit = time_cuda_us([&]() {
        brotensor::detail::cuda::jit::launch_fused_layernorm_modulate_ptx(
            static_cast<const float*>(g_x.data),
            static_cast<const float*>(g_gamma.data),
            static_cast<const float*>(g_beta.data),
            static_cast<const float*>(g_scale.data),
            static_cast<const float*>(g_shift.data),
            static_cast<float*>(g_y_jit.data),
            R, D, eps
        );
    });

    double speedup = t_base / t_jit;
    std::printf("  R=%-3d D=%-5d | Static CUDA: %7.2f us | PTX JIT: %7.2f us | Speedup: %5.2fx | MaxErr: %8.2e\n",
                R, D, t_base, t_jit, speedup, err_y);
}

// ── 3. SwiGLU Benchmark & Parity ───────────────────────────────────────────

void test_swiglu(int B, int D) {
    const size_t in_total = static_cast<size_t>(B) * static_cast<size_t>(2 * D);
    const size_t out_total = static_cast<size_t>(B) * static_cast<size_t>(D);

    std::vector<float> h_x(in_total);
    fill_random(h_x, 0x777ULL + static_cast<uint64_t>(B * D));

    Tensor g_x = Tensor::from_host_on(Device::CUDA, h_x.data(), B, 2 * D);
    Tensor g_y_ref = Tensor::zeros_on(Device::CUDA, B, D);
    Tensor g_y_jit = Tensor::zeros_on(Device::CUDA, B, D);

    // Static baseline
    brotensor::swiglu_forward(g_x, g_y_ref);
    cudaDeviceSynchronize();

    // PTX JIT
    brotensor::detail::cuda::jit::launch_swiglu_ptx(
        static_cast<const float*>(g_x.data),
        static_cast<float*>(g_y_jit.data),
        B, D
    );
    cudaDeviceSynchronize();

    // Parity
    std::vector<float> h_y_ref(out_total);
    std::vector<float> h_y_jit(out_total);
    cudaMemcpy(h_y_ref.data(), g_y_ref.data, out_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_jit.data(), g_y_jit.data, out_total * sizeof(float), cudaMemcpyDeviceToHost);

    float err_y = max_abs_error(h_y_ref.data(), h_y_jit.data(), out_total);
    CHECK_PARITY(err_y, 1e-5f, "SwiGLU Y");

    // Benchmark
    double t_base = time_cuda_us([&]() {
        brotensor::swiglu_forward(g_x, g_y_ref);
    });

    double t_jit = time_cuda_us([&]() {
        brotensor::detail::cuda::jit::launch_swiglu_ptx(
            static_cast<const float*>(g_x.data),
            static_cast<float*>(g_y_jit.data),
            B, D
        );
    });

    double speedup = t_base / t_jit;
    std::printf("  B=%-3d D=%-5d | Static CUDA: %7.2f us | PTX JIT: %7.2f us | Speedup: %5.2fx | MaxErr: %8.2e\n",
                B, D, t_base, t_jit, speedup, err_y);
}

// ── 4. AdaLN Modulate Benchmark & Parity ────────────────────────────────────

void test_modulate(int L, int D) {
    const size_t total = static_cast<size_t>(L) * static_cast<size_t>(D);

    std::vector<float> h_x(total);
    std::vector<float> h_scale(D);
    std::vector<float> h_shift(D);

    fill_random(h_x, 0x888ULL + static_cast<uint64_t>(L * D));
    fill_random(h_scale, 0x999ULL + static_cast<uint64_t>(D));
    fill_random(h_shift, 0xAAAULL + static_cast<uint64_t>(D));

    Tensor g_x = Tensor::from_host_on(Device::CUDA, h_x.data(), L, D);
    Tensor g_scale = Tensor::from_host_on(Device::CUDA, h_scale.data(), D, 1);
    Tensor g_shift = Tensor::from_host_on(Device::CUDA, h_shift.data(), D, 1);

    Tensor g_y_ref = Tensor::zeros_on(Device::CUDA, L, D);
    Tensor g_y_jit = Tensor::zeros_on(Device::CUDA, L, D);

    // Static baseline
    brotensor::modulate(g_x, g_scale, g_shift, g_y_ref);
    cudaDeviceSynchronize();

    // PTX JIT
    brotensor::detail::cuda::jit::launch_modulate_ptx(
        static_cast<const float*>(g_x.data),
        static_cast<const float*>(g_scale.data),
        static_cast<const float*>(g_shift.data),
        static_cast<float*>(g_y_jit.data),
        L, D
    );
    cudaDeviceSynchronize();

    // Parity
    std::vector<float> h_y_ref(total);
    std::vector<float> h_y_jit(total);
    cudaMemcpy(h_y_ref.data(), g_y_ref.data, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_jit.data(), g_y_jit.data, total * sizeof(float), cudaMemcpyDeviceToHost);

    float err_y = max_abs_error(h_y_ref.data(), h_y_jit.data(), total);
    CHECK_PARITY(err_y, 1e-5f, "AdaLN Modulate Y");

    // Benchmark
    double t_base = time_cuda_us([&]() {
        brotensor::modulate(g_x, g_scale, g_shift, g_y_ref);
    });

    double t_jit = time_cuda_us([&]() {
        brotensor::detail::cuda::jit::launch_modulate_ptx(
            static_cast<const float*>(g_x.data),
            static_cast<const float*>(g_scale.data),
            static_cast<const float*>(g_shift.data),
            static_cast<float*>(g_y_jit.data),
            L, D
        );
    });

    double speedup = t_base / t_jit;
    std::printf("  L=%-3d D=%-5d | Static CUDA: %7.2f us | PTX JIT: %7.2f us | Speedup: %5.2fx | MaxErr: %8.2e\n",
                L, D, t_base, t_jit, speedup, err_y);
}

} // namespace

int main() {
    brotensor::init();

    if (!brotensor::is_available(Device::CUDA)) {
        std::printf("CUDA device is not available — skipping test_cuda_jit\n");
        return 0;
    }

    if (!brotensor::detail::cuda::jit::is_cuda_jit_available()) {
        std::printf("CUDA Driver JIT is not available — skipping test_cuda_jit\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("================================================================================\n");
    std::printf("  BRASS CUDA DRIVER JIT BENCHMARK & NUMERICAL PARITY SUITE (RTX SM_%d%d)\n",
                prop.major, prop.minor);
    std::printf("  Device: %s (SMs: %d, Global Mem: %.1f GB)\n",
                prop.name, prop.multiProcessorCount, static_cast<double>(prop.totalGlobalMem) / 1e9);
    std::printf("================================================================================\n\n");

    const std::vector<int> batches = {1, 8, 32, 64};
    const std::vector<int> dims = {4096, 1152};

    // 1. Fused Residual RMSNorm
    std::printf("─── 1. Fused Residual RMSNorm (In-Place Residual + Warp Reduction) ──────────\n");
    for (int D : dims) {
        for (int B : batches) {
            test_fused_residual_rmsnorm(B, D);
        }
    }
    std::printf("\n");

    // 2. Fused LayerNorm + Modulate
    std::printf("─── 2. Fused LayerNorm + AdaLN Modulate (Zero Intermediate DRAM Writes) ─────\n");
    for (int D : dims) {
        for (int B : batches) {
            test_fused_layernorm_modulate(B, D);
        }
    }
    std::printf("\n");

    // 3. SwiGLU
    std::printf("─── 3. SwiGLU (Grid-Strided 128-bit Vector + Fast SiLU) ─────────────────────\n");
    for (int D : dims) {
        for (int B : batches) {
            test_swiglu(B, D);
        }
    }
    std::printf("\n");

    // 4. AdaLN Modulate
    std::printf("─── 4. AdaLN Modulate (Grid-Strided 128-bit Vector Modulate) ────────────────\n");
    for (int D : dims) {
        for (int B : batches) {
            test_modulate(B, D);
        }
    }
    std::printf("\n");

    std::printf("================================================================================\n");
    if (g_failures == 0) {
        std::printf("  [SUCCESS] All CUDA Driver JIT parity and benchmark suites PASSED with 100%% parity!\n");
    } else {
        std::printf("  [FAILED] Found %d parity failures!\n", g_failures);
    }
    std::printf("================================================================================\n");

    return g_failures == 0 ? 0 : 1;
}
