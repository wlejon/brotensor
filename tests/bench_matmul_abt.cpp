// Microbench for the public matmul_abt op (batched A @ B^T, FP16 storage /
// FP32 accumulate) on the active GPU backend (CUDA or Metal). Times a few
// representative prefill-shaped GEMMs and reports GFLOP/s. Not a ctest — built
// as a target and run by hand to compare the simdgroup fast path against the
// naive fallback (flip kTiledMin in src/metal/fp16_matmul.mm to force naive).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void bench(Device dev, int M, int N, int K, int iters) {
    std::mt19937 rng(0xC0DEu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(size_t(M) * K), B(size_t(N) * K);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    auto Ah = to_fp16(A), Bh = to_fp16(B);
    Tensor Ag = Tensor::from_host_fp16_on(dev, Ah.data(), M, K);
    Tensor Bg = Tensor::from_host_fp16_on(dev, Bh.data(), N, K);
    Tensor Cg = Tensor::zeros_on(dev, M, N, Dtype::FP16);

    // Back-to-back launches per timed sample, so a short GEMM's time is the
    // kernel's and not the launch / event floor (~10 us under WDDM).
    constexpr int kReps = 20;
    auto run = [&] {
        for (int r = 0; r < kReps; ++r) {
            brotensor::matmul_abt(Ag, Bg, Cg, 1, M, N, K,
                                  (long long)M * K, (long long)N * K, (long long)M * N,
                                  nullptr, 0);
        }
    };
    run();                    // warm-up (also builds the PSO)
    brotensor::sync_all();

    (void)iters;   // the harness picks its own warm-up length and sample count
    const double secs = bt_bench::time_min_ms(run) * 1e-3 / kReps;
    const double gflop = 2.0 * M * N * K / 1e9;
    std::printf("  M=%-5d N=%-5d K=%-5d  %8.3f ms  %8.1f GFLOP/s",
                M, N, K, secs * 1e3, gflop / secs);
    if (dev == Device::CUDA) {
        // Same product through linear_forward_batched_ex with a split-K workspace.
        Tensor ws;
        auto run_ex = [&] {
            for (int r = 0; r < kReps; ++r)
                brotensor::linear_forward_batched_ex(Bg, nullptr, Ag, 0, brotensor::kLinearEpiStore, &ws, Cg);
        };
        run_ex();
        brotensor::sync_all();
        const double s2 = bt_bench::time_min_ms(run_ex) * 1e-3 / kReps;
        std::printf("   | split-K %8.3f ms  %8.1f GFLOP/s", s2 * 1e3, gflop / s2);
        auto run_fast = [&] {
            for (int r = 0; r < kReps; ++r)
                brotensor::linear_forward_batched_ex(Bg, nullptr, Ag, 0,
                                                     brotensor::kLinearEpiStore | brotensor::kLinearEpiFastAccum,
                                                     &ws, Cg);
        };
        run_fast();
        brotensor::sync_all();
        const double s3 = bt_bench::time_min_ms(run_fast) * 1e-3 / kReps;
        std::printf("   | fast-acc %8.1f GFLOP/s", gflop / s3);
    }
    std::printf("\n");
}

int main() try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    brotensor::init();
    Device dev = Device::CPU;
    if (brotensor::is_available(Device::CUDA))       dev = Device::CUDA;
    else if (brotensor::is_available(Device::Metal)) dev = Device::Metal;
    else { std::printf("no GPU backend available - skipping\n"); return 0; }
    std::printf("bench_matmul_abt (device=%s)\n",
                dev == Device::CUDA ? "CUDA" : "Metal");

    // Pull the SM clock off its P8 idle floor before any timing.
    bt_bench::spin_up();

    bench(dev, 512,  512,  512,  50);
    bench(dev, 256,  2048, 2048, 30);   // attention out-proj shape
    bench(dev, 512,  4096, 4096, 20);   // FFN-ish shape
    bench(dev, 1024, 1024, 1024, 30);
    // Packed encoder batches (ModernBERT-large: D 1024, GeGLU F 2624) at a
    // short (112-row), medium (960) and large (4608) packed row count:
    // Wqkv, Wo, Wi, mlp Wo.
    for (int M : {112, 960, 4608}) {
        bench(dev, M, 3072, 1024, 30);
        bench(dev, M, 1024, 1024, 30);
        bench(dev, M, 5248, 1024, 30);
        bench(dev, M, 1024, 2624, 30);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_matmul_abt: %s\n", e.what());
    return 1;
}
