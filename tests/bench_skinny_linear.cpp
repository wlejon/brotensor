// Microbench for the 16-bit linear at short M — the regime of a transformer
// evaluating a few dozen rows against its whole weight set (a flow-matching
// NAR step, a speculative verify, a short prefill). The product is weight
// bandwidth bound there, so each line reports the time and the weight bytes
// streamed per second.
//
//   brotensor_bench_skinny_linear [M ...]       (default 1 8 16 32 52 64 96 128)
//   SKINNY_ONLY=<name>                          one shape only (for a profiler)
//
// Shapes: one decoder layer of a 2048-wide, GQA 16/8 x 128, 6144-FFN model
// (q, k, v, o, gate+up stacked with the SwiGLU epilogue, down), then the
// layer as a whole. Each shape cycles over enough weight copies to exceed L2,
// so every line is a cold-weight number. BF16. Not a ctest.

#include <brotensor/cuda_graph.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

Tensor rand_bf16(Device dev, int r, int c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<uint16_t> h(static_cast<size_t>(r) * c);
    for (auto& x : h) x = brotensor::fp32_to_bf16_bits(dist(rng));
    return Tensor::from_host_bf16_on(dev, h.data(), r, c);
}

struct Shape {
    const char* name;
    int N, K, epi;
};

}  // namespace

int main(int argc, char** argv) {
    brotensor::init();
    const Device dev = Device::CUDA;
    if (!brotensor::is_available(dev)) {
        std::printf("no GPU backend; skipping\n");
        return 0;
    }
    std::vector<int> Ms;
    for (int i = 1; i < argc; ++i) Ms.push_back(std::atoi(argv[i]));
    if (Ms.empty()) Ms = {1, 8, 16, 32, 52, 64, 96, 128};
    constexpr int kStore = brotensor::kLinearEpiStore, kSwiglu = brotensor::kLinearEpiSwiglu;
    const Shape shapes[] = {{"q", 2048, 2048, kStore},  {"k", 1024, 2048, kStore},
                            {"v", 1024, 2048, kStore},  {"o", 2048, 2048, kStore},
                            {"gu", 12288, 2048, kSwiglu}, {"down", 2048, 6144, kStore}};
    constexpr int S = 6;
    // Copies per shape: at least 192 MB of each, so L2 (72 MB on a 4090)
    // never holds the next copy.
    std::vector<std::vector<Tensor>> W(S);
    for (int i = 0; i < S; ++i) {
        const double mb = 2.0 * shapes[i].N * shapes[i].K / 1e6;
        const int copies = static_cast<int>(192.0 / mb) + 1;
        for (int c = 0; c < copies; ++c) W[i].push_back(rand_bf16(dev, shapes[i].N, shapes[i].K, 10 + 31 * i + c));
    }
    bt_bench::spin_up();
    Tensor ws;
    const char* only = std::getenv("SKINNY_ONLY");
    for (int M : Ms) {
        Tensor X2 = rand_bf16(dev, M, 2048, 1), X6 = rand_bf16(dev, M, 6144, 2);
        std::vector<Tensor> Y(S);
        double layer_bytes = 0;
        auto call = [&](int i, int c) {
            const Tensor& X = shapes[i].K == 2048 ? X2 : X6;
            brotensor::linear_forward_batched_ex(W[i][c % W[i].size()], nullptr, X, brotensor::kLinearActNone,
                                                 shapes[i].epi, &ws, Y[i]);
        };
        // Each timed body replays as one CUDA graph, so the numbers are GPU
        // time (kernels and their dependency gaps), not host launch rate.
        auto graphed = [&](const std::function<void()>& body) {
            body();
            brotensor::sync_all();
            brotensor::CudaGraphCapture cap;
            body();
            brotensor::CudaGraph g = cap.finish();
            return bt_bench::time_min_ms([&] { g.launch(); });
        };
        for (int i = 0; i < S; ++i) {
            const Shape& s = shapes[i];
            if (only && std::strcmp(only, s.name) != 0) continue;
            const int n = static_cast<int>(W[i].size());
            const double ms = graphed([&] {
                for (int c = 0; c < n; ++c) call(i, c);
            }) / n;
            const double bytes = 2.0 * s.N * s.K;
            layer_bytes += bytes;
            std::printf("  M=%-4d %-5s N=%-5d K=%-5d %8.1f us %7.1f GB/s\n", M, s.name, s.N, s.K, ms * 1e3,
                        bytes / (ms * 1e-3) / 1e9);
        }
        if (only) continue;
        // Four layers on different weight copies per replay.
        const double ms = graphed([&] {
            for (int l = 0; l < 4; ++l)
                for (int i = 0; i < S; ++i) call(i, l);
        }) / 4;
        std::printf("  M=%-4d layer             %8.1f us %7.1f GB/s  (x28 = %.2f ms)\n", M, ms * 1e3,
                    layer_bytes / (ms * 1e-3) / 1e9, ms * 28);
    }
    return 0;
}
