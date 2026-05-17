// Minimal smoke test for brotensor GpuTensor lifecycle + upload/download
// round-trip + one hand-checked op (relu_forward_gpu). Deep CPU↔GPU parity
// coverage lives in brogameagent's tests since those need brogameagent's CPU
// reference impls.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/ops.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using brotensor::GpuTensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static void test_lifecycle() {
    std::printf("test_lifecycle\n");
    GpuTensor a(4, 8);
    CHECK(a.rows == 4);
    CHECK(a.cols == 8);
    CHECK(a.size() == 32);
    CHECK(a.data != nullptr);
    a.zero();

    a.resize(2, 3);
    CHECK(a.rows == 2);
    CHECK(a.cols == 3);
    CHECK(a.size() == 6);

    // Move ctor / move assign.
    GpuTensor b = std::move(a);
    CHECK(b.rows == 2 && b.cols == 3);
    CHECK(a.data == nullptr);
    GpuTensor c;
    c = std::move(b);
    CHECK(c.rows == 2 && c.cols == 3);
}

static void test_round_trip() {
    std::printf("test_round_trip\n");
    std::vector<float> host_in = {1.0f, -2.5f, 3.25f, 0.0f, 7.0f, -0.125f};
    GpuTensor g;
    brotensor::upload(host_in.data(), 2, 3, g);
    CHECK(g.rows == 2 && g.cols == 3);

    std::vector<float> host_out(g.size(), 0.0f);
    brotensor::download(g, host_out.data());
    brotensor::cuda_sync();
    for (size_t i = 0; i < host_in.size(); ++i) {
        CHECK(host_in[i] == host_out[i]);
    }
}

static void test_clone() {
    std::printf("test_clone\n");
    std::vector<float> host_in = {5.0f, -1.0f, 2.0f, 4.0f};
    GpuTensor a;
    brotensor::upload(host_in.data(), 2, 2, a);
    GpuTensor b = a.clone();
    CHECK(b.rows == 2 && b.cols == 2);
    CHECK(b.data != a.data);

    std::vector<float> host_out(4, 0.0f);
    brotensor::download(b, host_out.data());
    brotensor::cuda_sync();
    for (size_t i = 0; i < host_in.size(); ++i) {
        CHECK(host_in[i] == host_out[i]);
    }
}

static void test_relu_smoke() {
    std::printf("test_relu_smoke\n");
    std::vector<float> host_in = {-3.0f, -0.5f, 0.0f, 0.25f, 7.0f};
    GpuTensor x, y;
    brotensor::upload(host_in.data(), 5, 1, x);
    y.resize(5, 1);
    brotensor::relu_forward_gpu(x, y);

    std::vector<float> host_out(5, 0.0f);
    brotensor::download(y, host_out.data());
    brotensor::cuda_sync();

    CHECK(host_out[0] == 0.0f);
    CHECK(host_out[1] == 0.0f);
    CHECK(host_out[2] == 0.0f);
    CHECK(host_out[3] == 0.25f);
    CHECK(host_out[4] == 7.0f);
}

static void test_fp16_host_conversion() {
    std::printf("test_fp16_host_conversion\n");
    // Exact representables.
    CHECK(brotensor::fp32_to_fp16_bits(0.0f) == 0x0000);
    CHECK(brotensor::fp32_to_fp16_bits(-0.0f) == 0x8000);
    CHECK(brotensor::fp32_to_fp16_bits(1.0f) == 0x3C00);
    CHECK(brotensor::fp32_to_fp16_bits(-1.0f) == 0xBC00);
    CHECK(brotensor::fp32_to_fp16_bits(2.0f) == 0x4000);
    CHECK(brotensor::fp16_bits_to_fp32(0x3C00) == 1.0f);
    CHECK(brotensor::fp16_bits_to_fp32(0x4000) == 2.0f);
    CHECK(brotensor::fp16_bits_to_fp32(0xBC00) == -1.0f);

    // Round-trip a spread of values; FP16 has ~3 decimal digits.
    const float samples[] = {0.5f, 0.25f, -3.5f, 12.5f, 100.0f, 0.001f};
    for (float v : samples) {
        const uint16_t bits = brotensor::fp32_to_fp16_bits(v);
        const float back = brotensor::fp16_bits_to_fp32(bits);
        const float relerr = std::fabs(back - v) / std::fabs(v);
        CHECK(relerr < 1e-2f);
    }
}

static void test_fp16_round_trip() {
    std::printf("test_fp16_round_trip\n");
    const float src_f32[] = {1.0f, -2.5f, 3.25f, 0.0f, 7.0f, -0.125f};
    std::vector<uint16_t> host_in(6);
    for (int i = 0; i < 6; ++i) {
        host_in[i] = brotensor::fp32_to_fp16_bits(src_f32[i]);
    }
    GpuTensor g;
    brotensor::upload_fp16(host_in.data(), 2, 3, g);
    CHECK(g.rows == 2 && g.cols == 3);
    CHECK(g.dtype == brotensor::Dtype::FP16);
    CHECK(g.bytes() == 12);

    std::vector<uint16_t> host_out(6, 0);
    brotensor::download_fp16(g, host_out.data());
    brotensor::cuda_sync();
    for (int i = 0; i < 6; ++i) {
        CHECK(host_in[i] == host_out[i]);
    }

    // Clone preserves dtype.
    GpuTensor g2 = g.clone();
    CHECK(g2.dtype == brotensor::Dtype::FP16);
    CHECK(g2.bytes() == 12);
    std::vector<uint16_t> host_out2(6, 0);
    brotensor::download_fp16(g2, host_out2.data());
    brotensor::cuda_sync();
    for (int i = 0; i < 6; ++i) {
        CHECK(host_in[i] == host_out2[i]);
    }
}

static void test_fp16_resize_and_zero() {
    std::printf("test_fp16_resize_and_zero\n");
    GpuTensor g(4, 4, brotensor::Dtype::FP16);
    CHECK(g.dtype == brotensor::Dtype::FP16);
    CHECK(g.bytes() == 32);
    g.zero();
    std::vector<uint16_t> host_out(16, 0xFFFF);
    brotensor::download_fp16(g, host_out.data());
    brotensor::cuda_sync();
    for (int i = 0; i < 16; ++i) CHECK(host_out[i] == 0);

    // Switch dtype back to FP32 via resize.
    g.resize(2, 2, brotensor::Dtype::FP32);
    CHECK(g.dtype == brotensor::Dtype::FP32);
    CHECK(g.bytes() == 16);
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }

    test_lifecycle();
    test_round_trip();
    test_clone();
    test_relu_smoke();
    test_fp16_host_conversion();
    test_fp16_round_trip();
    test_fp16_resize_and_zero();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll smoke checks passed.\n");
    return 0;
}
