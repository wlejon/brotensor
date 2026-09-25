// Microbench for the FP32 GEMM on the active GPU backend, through the public
// ops that reach it, one per operand layout:
//   NN  matmul                      C = A(M,K) @ B(K,N)
//   NT  linear_forward_batched_ex   C = A(M,K) @ B(N,K)^T
//   TN  conv2d_backward_input 1x1   C = A(K,M)^T @ B(K,N)   (M = C_in, N = HW)
// plus the 1x1-conv shapes StyleGAN3 inversion runs (forward, backward-input,
// and the long-K / small-output backward-weight that splits K). Reports
// GFLOP/s = 2*M*N*K / time. Not a ctest — run by hand.

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

Tensor rand_on(int r, int c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(static_cast<size_t>(r) * c);
    for (auto& x : v) x = dist(rng);
    return Tensor::from_host_on(g_dev, v.data(), r, c);
}

// Times `body` (reps launches back to back per sample) and prints one row.
void report(const char* tag, int M, int N, int K, int reps, const std::function<void()>& body) {
    auto run = [&] { for (int r = 0; r < reps; ++r) body(); };
    const double secs = bt_bench::time_min_ms(run) * 1e-3 / reps;
    const double gflop = 2.0 * M * N * K / 1e9;
    std::printf("  %-8s M=%-6d N=%-6d K=%-6d %9.3f ms %9.1f GFLOP/s\n", tag, M, N, K, secs * 1e3,
                gflop / secs);
}

int reps_for(int M, int N, int K) {
    const double flop = 2.0 * M * N * K;
    return flop > 5e10 ? 1 : flop > 5e9 ? 4 : 20;
}

void bench_nn(int M, int N, int K) {
    Tensor A = rand_on(M, K, 1), B = rand_on(K, N, 2), C;
    report("NN", M, N, K, reps_for(M, N, K), [&] { brotensor::matmul(A, B, C); });
}

void bench_nt(int M, int N, int K) {
    Tensor X = rand_on(M, K, 3), W = rand_on(N, K, 4), Y;
    report("NT", M, N, K, reps_for(M, N, K), [&] {
        brotensor::linear_forward_batched_ex(W, nullptr, X, 0, brotensor::kLinearEpiStore, nullptr, Y);
    });
}

// TN through the 1x1 conv backward-input: dX(C_in, HW) = W(C_out, C_in)^T @ dY(C_out, HW).
void bench_tn(int M, int N, int K) {
    Tensor Wt = rand_on(K, M, 5), dY = rand_on(1, K * N, 6), dX;
    report("TN", M, N, K, reps_for(M, N, K), [&] {
        brotensor::conv2d_backward_input(Wt, dY, 1, M, 1, N, K, 1, 1, 1, 1, 0, 0, 1, 1, dX);
    });
}

// A 1x1 conv (C_in -> C_out over H x W) forward, backward-input, backward-weight.
void bench_conv1x1(int C_in, int C_out, int H, int W) {
    const int HW = H * W;
    Tensor X = rand_on(1, C_in * HW, 7), Wt = rand_on(C_out, C_in, 8), dY = rand_on(1, C_out * HW, 9);
    Tensor Y, dX, dW = Tensor::zeros_on(g_dev, C_out, C_in);
    const int reps = reps_for(C_out, HW, C_in);
    report("conv fw", C_out, HW, C_in, reps, [&] {
        brotensor::conv2d_forward(X, Wt, nullptr, 1, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, Y);
    });
    report("conv dX", C_in, HW, C_out, reps, [&] {
        brotensor::conv2d_backward_input(Wt, dY, 1, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, dX);
    });
    report("conv dW", C_out, C_in, HW, reps, [&] {
        brotensor::conv2d_backward_weight(X, dY, 1, C_in, H, W, C_out, 1, 1, 1, 1, 0, 0, 1, 1, dW);
    });
}

}  // namespace

int main() try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    brotensor::init();
    if (brotensor::is_available(Device::CUDA))       g_dev = Device::CUDA;
    else if (brotensor::is_available(Device::Metal)) g_dev = Device::Metal;
    else { std::printf("no GPU backend available - skipping\n"); return 0; }
    std::printf("bench_gemm_fp32 (device=%s)\n", g_dev == Device::CUDA ? "CUDA" : "Metal");
    bt_bench::spin_up();

    const int shapes[][3] = {
        {4096, 4096, 4096}, {1024, 1024, 1024}, {512, 4096, 4096},
        {1, 4096, 4096},    {128, 3072, 768},   {1000, 999, 1001},
    };
    for (const auto& s : shapes) {
        bench_nn(s[0], s[1], s[2]);
        bench_nt(s[0], s[1], s[2]);
        bench_tn(s[0], s[1], s[2]);
    }
    // StyleGAN3-R 1x1 modulated conv: 181 -> 128 channels over 240 x 320.
    bench_conv1x1(181, 128, 240, 320);
    bench_conv1x1(512, 512, 64, 64);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_gemm_fp32: %s\n", e.what());
    return 1;
}
