// ─── Brass CUDA Driver JIT Fused GEMV Benchmark & Parity Suite ─────────────
//
// Verifies numerical parity (<= 1e-4 max error) and benchmarks speedups of
// Brass JIT accelerated PTX kernels for Fused GEMV SwiGLU and Fused GEMV Residual
// on NVIDIA RTX GPUs (sm_89) across decode shapes (M=1 at N x K).

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

namespace {

int g_failures = 0;

#define CHECK_PARITY(err, tol, name) do {                                            \
    if ((err) > (tol)) {                                                              \
        std::printf("  [FAIL] %s: max err = %g > tol = %g\n", (name), (err), (tol));  \
        ++g_failures;                                                                 \
    }                                                                                 \
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

    return (static_cast<double>(ms) * 1000.0) / iters;
}

void bench_fused_gemv_swiglu() {
    std::cout << "\n─── 1. Fused GEMV SwiGLU (Gate + Up + SiLU + Mul in Registers) ──────────\n";

    struct Shape { int N; int K; };
    std::vector<Shape> shapes = {
        {11008, 4096}, // LLaMA-7B MLP Up/Gate
        {4096, 4096},  // Square Projection
        {5632, 2048}   // Medium MLP
    };

    for (const auto& sh : shapes) {
        const int N = sh.N;
        const int K = sh.K;

        std::vector<float> h_w_gate(static_cast<size_t>(N) * K);
        std::vector<float> h_w_up(static_cast<size_t>(N) * K);
        std::vector<float> h_x(K);
        std::vector<float> h_ref(N);
        std::vector<float> h_jit(N);

        fill_random(h_w_gate, 1001, 0.02f);
        fill_random(h_w_up, 1002, 0.02f);
        fill_random(h_x, 1003, 0.5f);

        // Reference computation
        for (int n = 0; n < N; ++n) {
            float g_acc = 0.0f;
            float u_acc = 0.0f;
            const size_t row_off = static_cast<size_t>(n) * K;
            for (int k = 0; k < K; ++k) {
                g_acc += h_w_gate[row_off + k] * h_x[k];
                u_acc += h_w_up[row_off + k] * h_x[k];
            }
            float silu_g = g_acc / (1.0f + std::exp(-g_acc));
            h_ref[n] = silu_g * u_acc;
        }

        // Setup GPU tensors
        Tensor t_w_gate = Tensor::from_host_on(Device::CUDA, h_w_gate.data(), N, K);
        Tensor t_w_up   = Tensor::from_host_on(Device::CUDA, h_w_up.data(), N, K);
        Tensor t_x      = Tensor::from_host_on(Device::CUDA, h_x.data(), K, 1);
        Tensor t_zero_b = Tensor::zeros_on(Device::CUDA, N, 1);
        Tensor t_g      = Tensor::zeros_on(Device::CUDA, N, 1);
        Tensor t_u      = Tensor::zeros_on(Device::CUDA, N, 1);
        Tensor t_y_jit  = Tensor::zeros_on(Device::CUDA, N, 1);

        const float* d_w_gate = static_cast<const float*>(t_w_gate.data);
        const float* d_w_up   = static_cast<const float*>(t_w_up.data);
        const float* d_x      = static_cast<const float*>(t_x.data);
        float* d_y_jit        = static_cast<float*>(t_y_jit.data);

        // Run JIT
        brotensor::detail::cuda::jit::launch_fused_gemv_swiglu_ptx(
            d_w_gate, d_w_up, d_x, d_y_jit, N, K, nullptr
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_jit.data(), d_y_jit, N * sizeof(float), cudaMemcpyDeviceToHost);

        float err = max_abs_error(h_ref.data(), h_jit.data(), N);
        CHECK_PARITY(err, 1e-4f, "fused_gemv_swiglu");

        // Benchmark baseline: two separate GEMVs
        double t_base_us = time_cuda_us([&]() {
            brotensor::linear_forward(t_w_gate, t_zero_b, t_x, t_g);
            brotensor::linear_forward(t_w_up, t_zero_b, t_x, t_u);
        });

        // Benchmark JIT
        double t_jit_us = time_cuda_us([&]() {
            brotensor::detail::cuda::jit::launch_fused_gemv_swiglu_ptx(
                d_w_gate, d_w_up, d_x, d_y_jit, N, K, nullptr
            );
        });

        double speedup = t_base_us / t_jit_us;

        std::cout << "  N=" << std::setw(5) << N << " K=" << std::setw(5) << K
                  << " | Baseline (2 GEMV): " << std::setw(7) << std::fixed << std::setprecision(2) << t_base_us << " us"
                  << " | Fused JIT (GEMV+SwiGLU): " << std::setw(7) << std::fixed << std::setprecision(2) << t_jit_us << " us"
                  << " | Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << speedup << "x"
                  << " | MaxErr: " << std::scientific << std::setprecision(2) << err
                  << std::endl;
    }
}

void bench_fused_gemv_residual() {
    std::cout << "\n─── 2. Fused GEMV Residual (Down Matrix + In-Register Residual Add) ───────\n";

    struct Shape { int N; int K; };
    std::vector<Shape> shapes = {
        {4096, 11008}, // LLaMA-7B MLP Down
        {4096, 4096},  // Square Down Projection
        {2048, 5632}   // Medium Down
    };

    for (const auto& sh : shapes) {
        const int N = sh.N;
        const int K = sh.K;

        std::vector<float> h_w_down(static_cast<size_t>(N) * K);
        std::vector<float> h_x(K);
        std::vector<float> h_res(N);
        std::vector<float> h_ref(N);
        std::vector<float> h_jit(N);

        fill_random(h_w_down, 2001, 0.02f);
        fill_random(h_x, 2002, 0.5f);
        fill_random(h_res, 2003, 1.0f);

        // Reference computation
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            const size_t row_off = static_cast<size_t>(n) * K;
            for (int k = 0; k < K; ++k) {
                acc += h_w_down[row_off + k] * h_x[k];
            }
            h_ref[n] = acc + h_res[n];
        }

        // Setup GPU tensors
        Tensor t_w_down = Tensor::from_host_on(Device::CUDA, h_w_down.data(), N, K);
        Tensor t_x      = Tensor::from_host_on(Device::CUDA, h_x.data(), K, 1);
        Tensor t_res    = Tensor::from_host_on(Device::CUDA, h_res.data(), N, 1);
        Tensor t_zero_b = Tensor::zeros_on(Device::CUDA, N, 1);
        Tensor t_out    = Tensor::zeros_on(Device::CUDA, N, 1);
        Tensor t_y_jit  = Tensor::zeros_on(Device::CUDA, N, 1);

        const float* d_w_down = static_cast<const float*>(t_w_down.data);
        const float* d_x      = static_cast<const float*>(t_x.data);
        const float* d_res    = static_cast<const float*>(t_res.data);
        float* d_y_jit        = static_cast<float*>(t_y_jit.data);

        // Run JIT
        brotensor::detail::cuda::jit::launch_fused_gemv_residual_ptx(
            d_w_down, d_x, d_res, d_y_jit, N, K, nullptr
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_jit.data(), d_y_jit, N * sizeof(float), cudaMemcpyDeviceToHost);

        float err = max_abs_error(h_ref.data(), h_jit.data(), N);
        CHECK_PARITY(err, 1e-4f, "fused_gemv_residual");

        // Benchmark baseline: linear + add_inplace
        double t_base_us = time_cuda_us([&]() {
            brotensor::linear_forward(t_w_down, t_zero_b, t_x, t_out);
            brotensor::add_inplace(t_out, t_res);
        });

        // Benchmark JIT
        double t_jit_us = time_cuda_us([&]() {
            brotensor::detail::cuda::jit::launch_fused_gemv_residual_ptx(
                d_w_down, d_x, d_res, d_y_jit, N, K, nullptr
            );
        });

        double speedup = t_base_us / t_jit_us;

        std::cout << "  N=" << std::setw(5) << N << " K=" << std::setw(5) << K
                  << " | Baseline (GEMV+Add): " << std::setw(7) << std::fixed << std::setprecision(2) << t_base_us << " us"
                  << " | Fused JIT (GEMV Res): " << std::setw(7) << std::fixed << std::setprecision(2) << t_jit_us << " us"
                  << " | Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << speedup << "x"
                  << " | MaxErr: " << std::scientific << std::setprecision(2) << err
                  << std::endl;
    }
}

} // namespace

int main() {
    brotensor::init();

    if (!brotensor::is_available(Device::CUDA)) {
        std::cout << "CUDA device is not available — skipping test_cuda_jit_gemv\n";
        return 0;
    }

    if (!brotensor::detail::cuda::jit::is_cuda_jit_available()) {
        std::cout << "CUDA Driver JIT is not available on this system. Skipping.\n";
        return 0;
    }

    std::cout << "================================================================================\n";
    std::cout << "  BRASS CUDA DRIVER JIT FUSED GEMV BENCHMARK & PARITY SUITE (RTX SM_89)\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  Device: " << prop.name
              << " (SMs: " << prop.multiProcessorCount
              << ", Global Mem: " << std::fixed << std::setprecision(1) << (prop.totalGlobalMem / 1e9) << " GB)\n";
    std::cout << "================================================================================\n";

    bench_fused_gemv_swiglu();
    bench_fused_gemv_residual();

    std::cout << "================================================================================\n";
    if (g_failures == 0) {
        std::cout << "  [SUCCESS] All Fused GEMV JIT parity and benchmark suites PASSED with 100% parity!\n";
    } else {
        std::cout << "  [FAILURE] " << g_failures << " tests failed!\n";
    }
    std::cout << "================================================================================\n";

    return g_failures == 0 ? 0 : 1;
}
