// Microbench for the weight-only INT8 / FP8 E4M3 linear against the FP16 one
// at decode shapes: one decoder layer of a 2048-wide, GQA 16/8 x 128,
// 6144-FFN model (q, k, v, o, gate+up stacked with the SwiGLU epilogue, down),
// M = 1, 2, 8 rows, on the default device. Each line reports the layer's
// time and the weight bytes it streams per second (2 bytes a weight for
// FP16, 1 for INT8/FP8). Not a ctest — run by hand.
//
//   brotensor_bench_wq_linear [M ...]       (default 1 2 8)

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "bench_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

namespace {

// Random weights on `dev` as FP16, and the same values as FP32 on the CPU in
// `fp32` when asked (quantize_rows takes FP32 on the CPU).
Tensor rand_fp16(Device dev, int r, int c, uint32_t seed, Tensor* fp32 = nullptr) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<uint16_t> h(static_cast<size_t>(r) * c);
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = brotensor::fp32_to_fp16_bits(dist(rng));
        f[i] = brotensor::fp16_bits_to_fp32(h[i]);
    }
    if (fp32) *fp32 = Tensor::from_host(f.data(), r, c);
    return Tensor::from_host_fp16_on(dev, h.data(), r, c);
}

struct Linear {
    int out, in, epi;
};

}  // namespace

int main(int argc, char** argv) {
    brotensor::init();
    const Device dev = brotensor::default_device();
    std::printf("weight-only linear vs FP16, one 2048-wide decoder layer, on %s\n", brotensor::device_name(dev));
    std::vector<int> ms;
    for (int i = 1; i < argc; ++i) ms.push_back(std::atoi(argv[i]));
    if (ms.empty()) ms = {1, 2, 8};

    const std::vector<Linear> layer = {
        {2048, 2048, brotensor::kLinearEpiStore},    // q
        {1024, 2048, brotensor::kLinearEpiStore},    // k
        {1024, 2048, brotensor::kLinearEpiStore},    // v
        {2048, 2048, brotensor::kLinearEpiStore},    // o
        {12288, 2048, brotensor::kLinearEpiSwiglu},  // gate + up
        {2048, 6144, brotensor::kLinearEpiStore},    // down
    };
    // Weight copies cycled so a run streams more than the caches hold.
    constexpr int kCopies = 3;
    std::vector<std::vector<Tensor>> w16(kCopies), wq8(kCopies), sq8(kCopies), wf8(kCopies), sf8(kCopies);
    double weights = 0;
    for (int c = 0; c < kCopies; ++c)
        for (size_t i = 0; i < layer.size(); ++i) {
            const Linear& l = layer[i];
            Tensor w32;
            w16[c].push_back(rand_fp16(dev, l.out, l.in, 100u * c + static_cast<uint32_t>(i), &w32));
            Tensor q, s;
            brotensor::quantize_rows(w32, Dtype::INT8, q, s);
            wq8[c].push_back(q.to(dev));
            sq8[c].push_back(s.to(dev));
            brotensor::quantize_rows(w32, Dtype::FP8_E4M3, q, s);
            wf8[c].push_back(q.to(dev));
            sf8[c].push_back(s.to(dev));
            if (c == 0) weights += static_cast<double>(l.out) * l.in;
        }
    bt_bench::spin_up();

    for (const int M : ms) {
        std::vector<Tensor> xs, ys(layer.size());
        for (size_t i = 0; i < layer.size(); ++i) xs.push_back(rand_fp16(dev, M, layer[i].in, 7u + static_cast<uint32_t>(i)));
        int copy = 0;
        auto run = [&](int kind) {
            for (size_t i = 0; i < layer.size(); ++i) {
                if (kind == 0)
                    brotensor::linear_forward_batched_ex(w16[copy][i], nullptr, xs[i], 0, layer[i].epi, nullptr, ys[i]);
                else if (kind == 1)
                    brotensor::linear_forward_batched_wq_ex(wq8[copy][i], sq8[copy][i], nullptr, xs[i], 0,
                                                            layer[i].epi, ys[i]);
                else
                    brotensor::linear_forward_batched_wq_ex(wf8[copy][i], sf8[copy][i], nullptr, xs[i], 0,
                                                            layer[i].epi, ys[i]);
            }
            copy = (copy + 1) % kCopies;
        };
        const char* names[] = {"FP16", "INT8", "FP8"};
        for (int kind = 0; kind < 3; ++kind) {
            const double ms_ = bt_bench::time_min_ms([&] { run(kind); });
            const double bytes = weights * (kind == 0 ? 2.0 : 1.0);
            std::printf("  M=%-3d %-5s %8.3f ms  %7.1f GB/s\n", M, names[kind], ms_, bytes / (ms_ * 1e-3) / 1e9);
        }
    }
    return 0;
}
