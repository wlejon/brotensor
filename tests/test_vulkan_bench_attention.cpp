// `brotensor_test_vulkan --bench-attention`: flash attention throughput on the
// vk-spike's shapes (../vk-spike/RESULTS.md: FP16, bidirectional, packed
// heads), next to the spike's cooperative-matrix kernel; then causal prefill,
// decode against a KV cache and the backward. Not part of ctest.
//
// Timing: wall clock around back-to-back launches with one device sync,
// median of 7 batches after warm-up (test_vulkan_bench.cpp's method). TF/s
// counts 4 L^2 hd H flops (2 for Q K^T, 2 for P V), halved for causal.
// BROTENSOR_VK_BENCH_SHAPE=<tag> runs one shape; BROTENSOR_VK_FA_CFG=bc,nsg
// forces the fa_cm tile; BROTENSOR_VK_BENCH_PART=prefill|causal|decode|bwd
// runs one part.

#include "test_vulkan_common.h"

#include "detail/attention.h"

#include <brotensor/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace vkt {

namespace {

template <class F>
double median_ms(Device dev, F&& launch_once) {
    for (int i = 0; i < 3; ++i) launch_once();
    brotensor::sync(dev);
    auto t0 = std::chrono::steady_clock::now();
    launch_once();
    brotensor::sync(dev);
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int reps = std::max(3, std::min(200, static_cast<int>(150.0 / std::max(one, 1e-3))));
    std::vector<double> t;
    for (int b = 0; b < 7; ++b) {
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) launch_once();
        brotensor::sync(dev);
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct Shape {
    int L, H, hd;
    const char* tag;
    double spike_ms;   // vk-spike RESULTS.md, best fa_cm configuration
};

const Shape kShapes[] = {
    {542, 48, 128, "L542-h48-d128", 0.58},
    {1024, 32, 128, "L1024-h32-d128", 1.29},
    {4096, 24, 128, "L4096-h24-d128", 16.1},
    {1024, 16, 64, "L1024-h16-d64", 0.43},
    {4096, 16, 64, "L4096-h16-d64", 5.41},
    {4096, 24, 64, "L4096-h24-d64", 9.17},
    {1024, 16, 80, "L1024-h16-d80", 0},
    {4096, 16, 96, "L4096-h16-d96", 0},
    {1024, 16, 160, "L1024-h16-d160", 0},
    {2048, 8, 256, "L2048-h8-d256", 0},
};

// One attention call on `dev` with fresh random FP16 inputs.
struct Problem {
    Tensor Q, K, V, O;
    Problem(Device dev, int lq, int lk, int Dq, int Dkv) {
        Q = upload(random_values(std::size_t(lq) * Dq, 1, -1.0f, 1.0f, Dtype::FP16), lq, Dq, Dtype::FP16, dev);
        K = upload(random_values(std::size_t(lk) * Dkv, 2, -1.0f, 1.0f, Dtype::FP16), lk, Dkv, Dtype::FP16, dev);
        V = upload(random_values(std::size_t(lk) * Dkv, 3, -1.0f, 1.0f, Dtype::FP16), lk, Dkv, Dtype::FP16, dev);
    }
};

void bench_prefill(bool causal) {
    std::printf("\n%s prefill, FP16, flash_attention_forward (ms / TF/s)\n", causal ? "causal" : "bidirectional");
    std::printf("%-18s %10s %16s %8s\n", "shape", "spike", "vulkan", "vk/spk");
    const char* only = std::getenv("BROTENSOR_VK_BENCH_SHAPE");
    for (const Shape& s : kShapes) {
        if (only && std::string(only) != s.tag) continue;
        const double flop = 4.0 * s.L * double(s.L) * s.hd * s.H * (causal ? 0.5 : 1.0);
        const int D = s.H * s.hd;
        Problem p(vk(), s.L, s.L, D, D);
        const double vk_ms =
            median_ms(vk(), [&] { brotensor::flash_attention_forward(p.Q, p.K, p.V, nullptr, s.H, causal, p.O); });
        const double spk = causal ? 0 : s.spike_ms;
        char a[32], b[32];
        std::snprintf(a, sizeof a, "%.2f", spk);
        std::snprintf(b, sizeof b, "%.3f / %5.2f", vk_ms, flop / (vk_ms * 1e9));
        std::printf("%-18s %10s %16s %8.2f\n", s.tag, spk > 0 ? a : "-", b, spk > 0 ? spk / vk_ms : 0.0);
    }
}

void bench_decode() {
    std::printf("\ndecode, FP16, flash_attention_decode (us / GB/s of K+V read)\n");
    std::printf("%-30s %16s\n", "Lq x hq/hkv/hd x valid_len", "vulkan");
    struct D { int hq, hkv, hd, len, lq = 1; };
    const D shapes[] = {{16, 8, 128, 512},  {16, 8, 128, 4096},     {32, 8, 128, 16384},    {14, 2, 64, 2048},
                        {32, 32, 128, 4096}, {16, 8, 128, 32768},   {16, 8, 128, 16384, 4}, {16, 8, 128, 16384, 8},
                        {16, 8, 128, 16384, 16}, {16, 8, 128, 16384, 64}};
    for (const D& s : shapes) {
        const int Dq = s.hq * s.hd, Dkv = s.hkv * s.hd;
        const double bytes = 2.0 * s.len * Dkv * 2;
        Problem p(vk(), s.lq, s.len, Dq, Dkv);
        const double ms =
            median_ms(vk(), [&] { brotensor::flash_attention_decode(p.Q, p.K, p.V, s.len, s.hq, s.hkv, p.O, 0.0f, 0); });
        char tag[48], a[32];
        std::snprintf(tag, sizeof tag, "%d x %d/%d/%d x %d", s.lq, s.hq, s.hkv, s.hd, s.len);
        std::snprintf(a, sizeof a, "%7.1f / %5.0f", ms * 1e3, bytes / (ms * 1e6));
        std::printf("%-30s %16s\n", tag, a);
    }
}

// flash_attention_backward (FP16), TF/s counting the usual 10 L^2 hd H
// backward flops (halved for causal).
void bench_backward() {
    std::printf("\nbackward, FP16, flash_attention_backward (ms / TF/s)\n");
    std::printf("%-26s %16s\n", "shape", "vulkan");
    struct B { int L, H, hd; bool causal; };
    const B shapes[] = {{512, 8, 64, false}, {1024, 16, 64, false}, {1024, 16, 64, true},
                        {2048, 8, 128, true}, {2048, 16, 64, false}, {1024, 16, 128, false}};
    for (const B& s : shapes) {
        const int D = s.H * s.hd;
        const double flop = 10.0 * s.L * double(s.L) * s.hd * s.H * (s.causal ? 0.5 : 1.0);
        Problem p(vk(), s.L, s.L, D, D);
        Tensor g = upload(random_values(std::size_t(s.L) * D, 4, -1.0f, 1.0f, Dtype::FP16), s.L, D, Dtype::FP16, vk());
        Tensor dQ, dK, dV;
        const double ms = median_ms(vk(), [&] {
            brotensor::flash_attention_backward(p.Q, p.K, p.V, p.Q, g, nullptr, s.H, s.causal, dQ, dK, dV);
        });
        char tag[48], a[32];
        std::snprintf(tag, sizeof tag, "L%d-h%d-d%d%s", s.L, s.H, s.hd, s.causal ? " causal" : "");
        std::snprintf(a, sizeof a, "%.3f / %5.2f", ms, flop / (ms * 1e9));
        std::printf("%-26s %16s\n", tag, a);
    }
}

}  // namespace

void run_attention_bench() {
    const auto info = brotensor::vulkan::device_info(vk());
    std::printf("attention benchmark on %s (%s)\n", info.name.c_str(), info.driver.c_str());
    const char* part = std::getenv("BROTENSOR_VK_BENCH_PART");   // prefill | causal | decode
    if (!part || std::string(part) == "prefill") bench_prefill(false);
    if (!part || std::string(part) == "causal") bench_prefill(true);
    if (!part || std::string(part) == "decode") bench_decode();
    if (!part || std::string(part) == "bwd") bench_backward();
}

}  // namespace vkt
