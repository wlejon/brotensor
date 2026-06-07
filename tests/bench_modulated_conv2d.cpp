// Micro-benchmark: modulated_conv2d_forward at StyleGAN3-R config-R shapes
// (1×1 conv = per-pixel modulated matmul), FP32 vs FP16 vs BF16. Confirms
// whether the FP16/BF16 path (which routes the inner conv to the WMMA
// implicit-GEMM) is actually faster than FP32 for config-R, or whether the
// per-element half↔float conversion in the weight build dominates.
//
// NOT registered with ctest — invoke manually:
//   ./build-cuda/tests/Release/brotensor_bench_modulated_conv2d

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

namespace {
constexpr int WARMUP = 5;
constexpr int ITERS  = 50;

float time_loop_ms(int iters, const std::function<void()>& body) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < WARMUP; ++i) body();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) body();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

std::vector<float> rnd(size_t n, uint32_t seed, float s = 0.3f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-s, s);
    std::vector<float> v(n); for (auto& x : v) x = d(rng); return v;
}

Tensor up_dtype(const std::vector<float>& v, int r, int c, Dtype dt) {
    Tensor g = Tensor::from_host_on(Device::CPU, v.data(), r, c).to(Device::CUDA);
    if (dt == Dtype::FP32) return g;
    Tensor out; brotensor::cast(g, out, dt); return out;
}

void run(Dtype dt, const char* tag, int Cin, int Cout, int H, int W) {
    const int N = 1, kH = 1, kW = 1, pad = 0;
    Tensor X = up_dtype(rnd((size_t)N * Cin * H * W, 1), N, Cin * H * W, dt);
    Tensor Wt = up_dtype(rnd((size_t)Cout * Cin, 2), Cout, Cin * kH * kW, dt);
    Tensor S = up_dtype(rnd((size_t)N * Cin, 3, 1.0f), N, Cin, dt);
    Tensor dcoef, Y;
    auto fwd = [&]() {
        brotensor::modulated_conv2d_forward(X, Wt, S, N, Cin, H, W, Cout, kH, kW,
                                            pad, pad, /*demod=*/true, 1e-8f, dcoef, Y);
    };
    fwd(); brotensor::sync(Device::CUDA);
    const float t = time_loop_ms(ITERS, fwd);
    std::printf("  %-5s Cin=%-4d Cout=%-4d %dx%-4d  %7.3f ms\n", tag, Cin, Cout, H, W, t);
}
} // namespace

int main() {
    brotensor::init();
    if (!brotensor::is_available(Device::CUDA)) { std::printf("CUDA n/a\n"); return 0; }
    std::printf("modulated_conv2d_forward (config-R 1x1, demod):\n");
    struct S { int Cin, Cout, H, W; };
    for (S s : {S{512, 512, 32, 32}, S{512, 512, 64, 64}, S{256, 256, 128, 128},
                S{128, 128, 256, 256}}) {
        for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
            const char* tag = dt == Dtype::FP32 ? "fp32" : (dt == Dtype::FP16 ? "fp16" : "bf16");
            run(dt, tag, s.Cin, s.Cout, s.H, s.W);
        }
        std::printf("\n");
    }
    return 0;
}
