// Microbench for the plain FP16 matmul WMMA path (matmul_rm_wmma_kernel):
//   C(M,N) = A(M,K) @ B(K,N), row-major, FP16 storage / FP32 accumulate.
// Times a few representative shapes on CUDA. Not a ctest — built as a target
// and run by hand to compare the register-prefetch kernel vs baseline.

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
using brotensor::Tensor;

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void bench(int M, int N, int K, int iters) {
    std::mt19937 rng(0xC0DEu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    auto Ah = to_fp16(A), Bh = to_fp16(B);
    Tensor Ag = Tensor::from_host_fp16_on(Device::CUDA, Ah.data(), M, K);
    Tensor Bg = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(), K, N);

    Tensor Cg;
    brotensor::matmul(Ag, Bg, Cg);  // warm-up (also sizes Cg)
    brotensor::sync_all();

    // Reuse Cg rather than allocating a fresh output per iteration: the timed
    // region should be the GEMM, not the allocator. (The previous loop built a
    // new Tensor every pass, so every sample carried an alloc + free.)
    (void)iters;   // the harness picks its own warm-up length and sample count
    const double ms = bt_bench::time_min_ms([&] {
        brotensor::matmul(Ag, Bg, Cg);
    });
    const double gflop = 2.0 * M * N * K * 1e-9;
    std::printf("  M=%-6d N=%-6d K=%-6d : %8.4f ms/iter  %8.1f GFLOP/s\n",
                M, N, K, ms, gflop / (ms * 1e-3));
}

int main() {
    brotensor::init();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Pull the SM clock off its P8 idle floor before any timing.
    bt_bench::spin_up();
    std::printf("[bench] matmul_rm FP16 WMMA path\n");
    const int it = 100;
    // Square GEMMs across scales.
    bench(512, 512, 512, it);
    bench(1024, 1024, 1024, it);
    bench(2048, 2048, 2048, it);
    bench(4096, 4096, 4096, it);
    // Skewed shapes (short M, wide K/N — transformer-linear-like).
    bench(64, 4096, 4096, it);
    bench(128, 4096, 4096, it);
    bench(256, 14336, 4096, it);
    return 0;
}
