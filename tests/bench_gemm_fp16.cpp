// Microbench for the FP16 GEMM on the active GPU backend, through the public
// ops that reach it, one per operand layout:
//   NN   matmul                      C = A(M,K) @ B(K,N)
//   NT   linear_forward_batched_ex   C = A(M,K) @ B(N,K)^T   (the LLM / DiT linear)
//   ABT  matmul_abt                  C = A(M,K) @ B(N,K)^T   (the attention-style op)
//   TN   conv2d_backward_input 1x1   C = A(K,M)^T @ B(K,N)
//   W16  linear_forward_batched      FP32 X @ FP16 W^T -> FP32 (mixed precision)
// Reports GFLOP/s = 2*M*N*K / time. Not a ctest — run by hand.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

Device g_dev = Device::CPU;

Tensor rand_on(int r, int c, uint32_t seed, Dtype dt = Dtype::FP16) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(static_cast<size_t>(r) * c);
    for (auto& x : v) x = dist(rng);
    if (dt == Dtype::FP32) return Tensor::from_host_on(g_dev, v.data(), r, c);
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return Tensor::from_host_fp16_on(g_dev, h.data(), r, c);
}

void report(const char* tag, int M, int N, int K, int reps, const std::function<void()>& body) {
    auto run = [&] { for (int r = 0; r < reps; ++r) body(); };
    const double secs = bt_bench::time_min_ms(run) * 1e-3 / reps;
    const double gflop = 2.0 * M * N * K / 1e9;
    std::printf("  %-4s M=%-6d N=%-6d K=%-6d %9.3f ms %9.1f GFLOP/s\n", tag, M, N, K, secs * 1e3,
                gflop / secs);
}

int reps_for(int M, int N, int K) {
    const double flop = 2.0 * M * N * K;
    return flop > 5e10 ? 1 : flop > 5e9 ? 4 : 20;
}

void bench_shape(int M, int N, int K) {
    const int reps = reps_for(M, N, K);
    {
        Tensor A = rand_on(M, K, 1), B = rand_on(K, N, 2), C;
        report("NN", M, N, K, reps, [&] { brotensor::matmul(A, B, C); });
    }
    Tensor X = rand_on(M, K, 3), W = rand_on(N, K, 4), Y;
    report("NT", M, N, K, reps, [&] {
        brotensor::linear_forward_batched_ex(W, nullptr, X, 0, brotensor::kLinearEpiStore, nullptr, Y);
    });
    {
        Tensor C = Tensor::zeros_on(g_dev, M, N, Dtype::FP16);
        report("ABT", M, N, K, reps, [&] {
            brotensor::matmul_abt(X, W, C, 1, M, N, K, 0, 0, 0, nullptr, 0);
        });
    }
    {
        Tensor Wt = rand_on(K, M, 5), dY = rand_on(1, K * N, 6), dX;
        report("TN", M, N, K, reps, [&] {
            brotensor::conv2d_backward_input(Wt, dY, 1, M, 1, N, K, 1, 1, 1, 1, 0, 0, 1, 1, dX);
        });
    }
    {
        Tensor X32 = rand_on(M, K, 7, Dtype::FP32), b = rand_on(N, 1, 8, Dtype::FP32), Y32;
        report("W16", M, N, K, reps, [&] { brotensor::linear_forward_batched(W, b, X32, Y32); });
    }
}

}  // namespace

int main() try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    brotensor::init();
    if (brotensor::is_available(Device::CUDA))       g_dev = Device::CUDA;
    else if (brotensor::is_available(Device::Metal)) g_dev = Device::Metal;
    else { std::printf("no GPU backend available - skipping\n"); return 0; }
    std::printf("bench_gemm_fp16 (device=%s)\n", g_dev == Device::CUDA ? "CUDA" : "Metal");
    bt_bench::spin_up();

    const int shapes[][3] = {
        {4096, 4096, 4096}, {1024, 1024, 1024}, {512, 4096, 4096},
        {128, 3072, 768},   {1, 4096, 4096},    {8, 4096, 11008},
    };
    for (const auto& s : shapes) bench_shape(s[0], s[1], s[2]);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_gemm_fp16: %s\n", e.what());
    return 1;
}
