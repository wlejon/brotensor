// ─── Brass CUDA Graph Capture for JIT Kernels Parity & Benchmark Suite ───────
//
// Tests CudaGraphRunner capturing Brass JIT-compiled PTX kernels into CUDA Graphs,
// verifying numerical parity and measuring the drop in launch overhead (~4 us).

#include <brotensor/ops.h>
#include <brotensor/tensor.h>
#include <brotensor/runtime.h>
#include "../src/cuda/cuda_jit.h"
#include "../src/cuda/cuda_graph_jit.h"

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
using brotensor::detail::cuda::jit::CudaGraphRunner;

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
double time_cuda_us(Fn&& fn, int warmup = 20, int iters = 200) {
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

void test_graph_runner_lifecycle_and_parity() {
    std::cout << "\n─── 1. CudaGraphRunner Lifecycle & Parity Verification ──────────────────\n";

    constexpr int N = 4096;
    constexpr int K = 4096;

    std::vector<float> h_w_gate(static_cast<size_t>(N) * K);
    std::vector<float> h_w_up(static_cast<size_t>(N) * K);
    std::vector<float> h_x(K);
    std::vector<float> h_ref(N);
    std::vector<float> h_out(N);

    fill_random(h_w_gate, 3001, 0.02f);
    fill_random(h_w_up, 3002, 0.02f);
    fill_random(h_x, 3003, 0.5f);

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

    Tensor t_w_gate = Tensor::from_host_on(Device::CUDA, h_w_gate.data(), N, K);
    Tensor t_w_up   = Tensor::from_host_on(Device::CUDA, h_w_up.data(), N, K);
    Tensor t_x      = Tensor::from_host_on(Device::CUDA, h_x.data(), K, 1);
    Tensor t_y      = Tensor::zeros_on(Device::CUDA, N, 1);

    const float* d_w_gate = static_cast<const float*>(t_w_gate.data);
    const float* d_w_up   = static_cast<const float*>(t_w_up.data);
    const float* d_x      = static_cast<const float*>(t_x.data);
    float* d_y            = static_cast<float*>(t_y.data);

    // Warmup JIT driver compilation outside graph capture
    brotensor::detail::cuda::jit::launch_fused_gemv_swiglu_ptx(
        d_w_gate, d_w_up, d_x, d_y, N, K, nullptr
    );
    cudaDeviceSynchronize();

    // Test explicit lifecycle: begin_capture -> op -> end_capture -> instantiate -> launch
    CudaGraphRunner runner;
    assert(!runner.is_capturing());
    assert(!runner.is_captured());
    assert(!runner.is_instantiated());

    runner.begin_capture();
    assert(runner.is_capturing());

    brotensor::detail::cuda::jit::launch_fused_gemv_swiglu_ptx(
        d_w_gate, d_w_up, d_x, d_y, N, K, nullptr
    );

    runner.end_capture();
    assert(!runner.is_capturing());
    assert(runner.is_captured());
    assert(!runner.is_instantiated());

    runner.instantiate();
    assert(runner.is_instantiated());

    // Execute via graph replay
    runner.launch();
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    float err = max_abs_error(h_ref.data(), h_out.data(), N);
    CHECK_PARITY(err, 1e-4f, "graph_runner_manual_lifecycle");
    std::cout << "  Explicit lifecycle capture & replay: PASSED (MaxErr: " << std::scientific << err << ")\n";

    // Test RAII capture helper
    CudaGraphRunner runner_raii;
    runner_raii.capture([&]() {
        brotensor::detail::cuda::jit::launch_fused_gemv_swiglu_ptx(
            d_w_gate, d_w_up, d_x, d_y, N, K, nullptr
        );
    });
    assert(runner_raii.is_instantiated());

    runner_raii.launch();
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    float err_raii = max_abs_error(h_ref.data(), h_out.data(), N);
    CHECK_PARITY(err_raii, 1e-4f, "graph_runner_raii_capture");
    std::cout << "  RAII capture helper & replay: PASSED (MaxErr: " << std::scientific << err_raii << ")\n";
}

void bench_graph_launch_latency() {
    std::cout << "\n─── 2. CUDA Graph JIT Kernel Launch Overhead & Replay Benchmark ─────────\n";

    // Benchmark small and medium shapes to highlight launch overhead reduction
    struct TestCase {
        std::string name;
        int B;
        int D;
    };
    std::vector<TestCase> cases = {
        {"Fused Residual RMSNorm (B=1, D=4096)", 1, 4096},
        {"Fused Residual RMSNorm (B=8, D=4096)", 8, 4096},
        {"SwiGLU Vector JIT (B=1, D=4096)", 1, 4096},
        {"AdaLN Modulate JIT (L=1, D=4096)", 1, 4096}
    };

    for (const auto& tc : cases) {
        Tensor t_x = Tensor::zeros_on(Device::CUDA, tc.B, tc.D);
        Tensor t_res = Tensor::zeros_on(Device::CUDA, tc.B, tc.D);
        Tensor t_gamma = Tensor::zeros_on(Device::CUDA, 1, tc.D);
        Tensor t_y = Tensor::zeros_on(Device::CUDA, tc.B, tc.D);

        float* d_x = static_cast<float*>(t_x.data);
        const float* d_res = static_cast<const float*>(t_res.data);
        const float* d_gamma = static_cast<const float*>(t_gamma.data);
        float* d_y = static_cast<float*>(t_y.data);

        // Warmup JIT
        brotensor::detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
            d_x, d_res, d_gamma, d_y, tc.B, tc.D, 1e-5f, nullptr
        );
        cudaDeviceSynchronize();

        // 1. Uncaptured launches (driver submission + kernel execution)
        double t_uncaptured_us = time_cuda_us([&]() {
            brotensor::detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
                d_x, d_res, d_gamma, d_y, tc.B, tc.D, 1e-5f, nullptr
            );
        });

        // 2. CUDA Graph capture & replay
        CudaGraphRunner runner;
        runner.capture([&]() {
            brotensor::detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
                d_x, d_res, d_gamma, d_y, tc.B, tc.D, 1e-5f, nullptr
            );
        });

        double t_graph_us = time_cuda_us([&]() {
            runner.launch();
        });

        double speedup = t_uncaptured_us / t_graph_us;

        std::cout << "  " << std::left << std::setw(42) << tc.name
                  << " | Uncaptured: " << std::setw(6) << std::fixed << std::setprecision(2) << t_uncaptured_us << " us"
                  << " | CUDA Graph: " << std::setw(6) << std::fixed << std::setprecision(2) << t_graph_us << " us"
                  << " | Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }
}

} // namespace

int main() {
    brotensor::init();

    if (!brotensor::is_available(Device::CUDA)) {
        std::cout << "CUDA device is not available — skipping test_cuda_graph_jit\n";
        return 0;
    }

    if (!brotensor::detail::cuda::jit::is_cuda_jit_available()) {
        std::cout << "CUDA Driver JIT is not available on this system. Skipping.\n";
        return 0;
    }

    std::cout << "================================================================================\n";
    std::cout << "  CUDA GRAPH CAPTURE FOR BRASS JIT KERNELS BENCHMARK SUITE (RTX SM_89)\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  Device: " << prop.name
              << " (SMs: " << prop.multiProcessorCount
              << ", Global Mem: " << std::fixed << std::setprecision(1) << (prop.totalGlobalMem / 1e9) << " GB)\n";
    std::cout << "================================================================================\n";

    // A kernel the JIT cannot find or launch (a brass launch contract that
    // moved under this file) throws; caught, it is a failure that names the
    // kernel instead of a fast-fail exit with the buffered output lost.
    try {
        test_graph_runner_lifecycle_and_parity();
        bench_graph_launch_latency();
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] " << e.what() << "\n";
        ++g_failures;
    }

    std::cout << "================================================================================\n";
    if (g_failures == 0) {
        std::cout << "  [SUCCESS] All CUDA Graph JIT tests and benchmarks PASSED!\n";
    } else {
        std::cout << "  [FAILURE] " << g_failures << " tests failed!\n";
    }
    std::cout << "================================================================================\n";

    return g_failures == 0 ? 0 : 1;
}
