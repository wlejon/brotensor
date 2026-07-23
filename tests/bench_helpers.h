#pragma once

// Shared timing harness for the CUDA micro-benchmarks.
//
// Why this exists: on an idle desktop GPU the SM clock sits at its P8 floor
// (285 MHz on a 4090 — roughly a ninth of the ~2600 MHz boost clock). A
// micro-benchmark whose whole measured region lasts a few milliseconds never
// gives the driver a reason to ramp, so it samples whatever clock happened to
// be in effect. Measured swings of 6x on the same kernel between consecutive
// runs are the result, which makes before/after comparison worthless.
//
// Locking clocks with `nvidia-smi -lgc` would be the clean fix, but that needs
// administrator rights this session does not have. So the harness does it
// in-process:
//
//   spin_up()      — runs a sustained dense GEMM until the clocks have had
//                    time to ramp. Call once at the start of main().
//   time_min_ms()  — warms up for a fixed wall-clock duration (not a fixed
//                    iteration count, so cheap kernels still warm properly),
//                    then times individual iterations and returns the MINIMUM.
//
// The minimum is the right statistic here: these kernels are deterministic, so
// the true cost is a lower bound and every sample above it is interference
// (clock ramp, another process, driver bookkeeping). A mean folds that noise
// into the number being reported.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace bt_bench {

// Duration of the per-op warm-up. Long enough to cover a P8 -> P0 ramp.
constexpr double kWarmupMs = 120.0;

// Timed samples taken after warm-up.
constexpr int kSamples = 20;

// Run a dense GEMM long enough to pull the SM clock off its idle floor. The
// shape is arbitrary — it just has to keep the tensor cores busy.
inline void spin_up(double ms = 600.0) {
    using brotensor::Device;
    using brotensor::Tensor;
    constexpr int M = 1024, K = 4096, N = 4096;
    std::vector<uint16_t> w(static_cast<size_t>(N) * K, 0x3C00);  // 1.0h
    std::vector<uint16_t> x(static_cast<size_t>(M) * K, 0x3800);  // 0.5h
    Tensor W = Tensor::from_host_fp16_on(Device::CUDA, w.data(), N, K);
    Tensor X = Tensor::from_host_fp16_on(Device::CUDA, x.data(), M, K);
    Tensor Y;
    brotensor::linear_forward_batched_fp16(W, nullptr, X, Y);
    brotensor::sync_all();

    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        for (int i = 0; i < 4; ++i) {
            brotensor::linear_forward_batched_fp16(W, nullptr, X, Y);
        }
        brotensor::sync_all();
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= ms) break;
    }
}

// Warm up for kWarmupMs of wall clock, then return the fastest of kSamples
// individually-timed iterations.
inline float time_min_ms(const std::function<void()>& body) {
    body();
    brotensor::sync_all();

    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        body();
        brotensor::sync_all();
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= kWarmupMs) break;
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e30f;
    for (int s = 0; s < kSamples; ++s) {
        cudaEventRecord(e0);
        body();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best;
}

}  // namespace bt_bench
