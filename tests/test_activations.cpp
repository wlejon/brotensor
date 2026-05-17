// CPU↔GPU parity for silu_forward_gpu and gelu_forward_gpu (FP32 + FP16).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static float silu_ref(float v) { return v / (1.0f + std::exp(-v)); }
static float gelu_ref(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(u));
}
static float quick_gelu_ref(float v) { return v / (1.0f + std::exp(-1.702f * v)); }

static void test_fp32(void (*op)(const GpuTensor&, GpuTensor&),
                      float (*ref)(float), const char* name) {
    std::printf("  %s fp32\n", name);
    std::mt19937 rng(0xACE0);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const int N = 257;
    std::vector<float> host(N);
    for (auto& v : host) v = dist(rng);

    GpuTensor x, y;
    brotensor::upload(host.data(), N, 1, x);
    op(x, y);
    CHECK(y.rows == N && y.cols == 1 && y.dtype == Dtype::FP32);

    std::vector<float> got(N);
    brotensor::download(y, got.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float r = ref(host[i]);
        const float e = std::fabs(got[i] - r);
        if (e > max_err) max_err = e;
        if (e > 1e-5f + 1e-5f * std::fabs(r)) ++bad;
    }
    std::printf("    max_err=%g bad=%d\n", max_err, bad);
    CHECK(bad == 0);
}

static void test_fp16(void (*op)(const GpuTensor&, GpuTensor&),
                      float (*ref)(float), const char* name) {
    std::printf("  %s fp16\n", name);
    std::mt19937 rng(0xACE1);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const int N = 257;
    std::vector<float>    host_f(N);
    std::vector<uint16_t> host_h(N);
    for (int i = 0; i < N; ++i) {
        host_f[i] = dist(rng);
        host_h[i] = brotensor::fp32_to_fp16_bits(host_f[i]);
    }
    GpuTensor x, y;
    brotensor::upload_fp16(host_h.data(), N, 1, x);
    op(x, y);
    CHECK(y.rows == N && y.cols == 1 && y.dtype == Dtype::FP16);

    std::vector<uint16_t> got_h(N);
    brotensor::download_fp16(y, got_h.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float in = brotensor::fp16_bits_to_fp32(host_h[i]);
        const float r  = ref(in);
        const float g  = brotensor::fp16_bits_to_fp32(got_h[i]);
        const float e  = std::fabs(g - r);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 1e-2f * std::fabs(r)) ++bad;
    }
    std::printf("    max_err=%g bad=%d\n", max_err, bad);
    CHECK(bad == 0);
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_activations\n");

    test_fp32(brotensor::silu_forward_gpu, silu_ref, "silu");
    test_fp16(brotensor::silu_forward_gpu, silu_ref, "silu");
    test_fp32(brotensor::gelu_forward_gpu, gelu_ref, "gelu");
    test_fp16(brotensor::gelu_forward_gpu, gelu_ref, "gelu");
    test_fp32(brotensor::quick_gelu_forward_gpu, quick_gelu_ref, "quick_gelu");
    test_fp16(brotensor::quick_gelu_forward_gpu, quick_gelu_ref, "quick_gelu");

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll activation checks passed.\n");
    return 0;
}
