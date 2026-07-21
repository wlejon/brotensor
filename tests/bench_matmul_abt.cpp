// Microbench for the public matmul_abt op (batched A @ B^T, FP16 storage /
// FP32 accumulate) on the active GPU backend (CUDA or Metal). Times a few
// representative prefill-shaped GEMMs and reports GFLOP/s. Not a ctest — built
// as a target and run by hand to compare the simdgroup fast path against the
// naive fallback (flip kTiledMin in src/metal/fp16_matmul.mm to force naive).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

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

    auto run = [&] {
        brotensor::matmul_abt(Ag, Bg, Cg, 1, M, N, K,
                              (long long)M * K, (long long)N * K, (long long)M * N,
                              nullptr, 0);
    };
    run();                    // warm-up (also builds the PSO)
    brotensor::sync_all();

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) run();
    brotensor::sync_all();
    auto t1 = clk::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count() / iters;
    const double gflop = 2.0 * M * N * K / 1e9;
    std::printf("  M=%-5d N=%-5d K=%-5d  %8.3f ms  %8.1f GFLOP/s\n",
                M, N, K, secs * 1e3, gflop / secs);
}

int main() {
    brotensor::init();
    Device dev = Device::CPU;
    if (brotensor::is_available(Device::CUDA))       dev = Device::CUDA;
    else if (brotensor::is_available(Device::Metal)) dev = Device::Metal;
    else { std::printf("no GPU backend available - skipping\n"); return 0; }
    std::printf("bench_matmul_abt (device=%s)\n",
                dev == Device::CUDA ? "CUDA" : "Metal");

    bench(dev, 512,  512,  512,  50);
    bench(dev, 256,  2048, 2048, 30);   // attention out-proj shape
    bench(dev, 512,  4096, 4096, 20);   // FFN-ish shape
    bench(dev, 1024, 1024, 1024, 30);
    return 0;
}
