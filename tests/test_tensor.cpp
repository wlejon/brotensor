// Smoke test for the unified brotensor::Tensor: lifecycle, host↔device
// round-trip, clone, a hand-checked op (relu_forward), and FP16
// host-conversion + round-trip. CUDA-only — guarded out on a CPU-only build
// since the round-trip / op coverage here exercises a GPU backend.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static void test_lifecycle() {
    std::printf("test_lifecycle\n");
    Tensor a = Tensor::zeros_on(Device::CUDA, 4, 8);
    CHECK(a.rows == 4);
    CHECK(a.cols == 8);
    CHECK(a.size() == 32);
    CHECK(a.data != nullptr);
    CHECK(a.device == Device::CUDA);
    a.zero();

    a.resize(2, 3);
    CHECK(a.rows == 2);
    CHECK(a.cols == 3);
    CHECK(a.size() == 6);
    CHECK(a.device == Device::CUDA);  // resize preserves device

    // Move ctor / move assign.
    Tensor b = std::move(a);
    CHECK(b.rows == 2 && b.cols == 3);
    CHECK(a.data == nullptr);
    Tensor c;
    c = std::move(b);
    CHECK(c.rows == 2 && c.cols == 3);
    CHECK(c.device == Device::CUDA);
}

static void test_round_trip() {
    std::printf("test_round_trip\n");
    std::vector<float> host_in = {1.0f, -2.5f, 3.25f, 0.0f, 7.0f, -0.125f};
    Tensor g = Tensor::from_host_on(Device::CUDA, host_in.data(), 2, 3);
    CHECK(g.rows == 2 && g.cols == 3);
    CHECK(g.device == Device::CUDA);

    std::vector<float> host_out = g.to_host_vector();
    CHECK(host_out.size() == host_in.size());
    for (size_t i = 0; i < host_in.size(); ++i) {
        CHECK(host_in[i] == host_out[i]);
    }

    // copy_to_host into a caller-supplied buffer.
    std::vector<float> host_out2(g.size(), 0.0f);
    g.copy_to_host(host_out2.data());
    for (size_t i = 0; i < host_in.size(); ++i) {
        CHECK(host_in[i] == host_out2[i]);
    }
}

static void test_clone() {
    std::printf("test_clone\n");
    std::vector<float> host_in = {5.0f, -1.0f, 2.0f, 4.0f};
    Tensor a = Tensor::from_host_on(Device::CUDA, host_in.data(), 2, 2);
    Tensor b = a.clone();
    CHECK(b.rows == 2 && b.cols == 2);
    CHECK(b.device == Device::CUDA);
    CHECK(b.data != a.data);

    std::vector<float> host_out = b.to_host_vector();
    for (size_t i = 0; i < host_in.size(); ++i) {
        CHECK(host_in[i] == host_out[i]);
    }
}

static void test_to_migration() {
    std::printf("test_to_migration\n");
    std::vector<float> host_in = {1.5f, -2.0f, 0.0f, 9.0f};
    Tensor g = Tensor::from_host_on(Device::CUDA, host_in.data(), 2, 2);

    // Device → host migration.
    Tensor h = g.to(Device::CPU);
    CHECK(h.device == Device::CPU);
    for (int i = 0; i < 4; ++i) CHECK(h.host_f32()[i] == host_in[i]);

    // Host → device round-trips back to the same values.
    Tensor g2 = h.to(Device::CUDA);
    CHECK(g2.device == Device::CUDA);
    std::vector<float> back = g2.to_host_vector();
    for (int i = 0; i < 4; ++i) CHECK(back[i] == host_in[i]);
}

static void test_relu_smoke() {
    std::printf("test_relu_smoke\n");
    std::vector<float> host_in = {-3.0f, -0.5f, 0.0f, 0.25f, 7.0f};
    Tensor x = Tensor::from_host_on(Device::CUDA, host_in.data(), 5, 1);
    Tensor y = Tensor::empty_on(Device::CUDA, 5, 1);
    brotensor::relu_forward(x, y);
    brotensor::sync(Device::CUDA);

    std::vector<float> host_out = y.to_host_vector();
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
    Tensor g = Tensor::from_host_fp16_on(Device::CUDA, host_in.data(), 2, 3);
    CHECK(g.rows == 2 && g.cols == 3);
    CHECK(g.dtype == Dtype::FP16);
    CHECK(g.bytes() == 12);

    std::vector<uint16_t> host_out = g.to_host_vector_fp16();
    CHECK(host_out.size() == host_in.size());
    for (int i = 0; i < 6; ++i) {
        CHECK(host_in[i] == host_out[i]);
    }

    // Clone preserves dtype.
    Tensor g2 = g.clone();
    CHECK(g2.dtype == Dtype::FP16);
    CHECK(g2.bytes() == 12);
    std::vector<uint16_t> host_out2 = g2.to_host_vector_fp16();
    for (int i = 0; i < 6; ++i) {
        CHECK(host_in[i] == host_out2[i]);
    }
}

static void test_fp16_resize_and_zero() {
    std::printf("test_fp16_resize_and_zero\n");
    Tensor g = Tensor::zeros_on(Device::CUDA, 4, 4, Dtype::FP16);
    CHECK(g.dtype == Dtype::FP16);
    CHECK(g.bytes() == 32);
    g.zero();
    std::vector<uint16_t> host_out = g.to_host_vector_fp16();
    for (int i = 0; i < 16; ++i) CHECK(host_out[i] == 0);

    // Switch dtype back to FP32 via resize.
    g.resize(2, 2, Dtype::FP32);
    CHECK(g.dtype == Dtype::FP32);
    CHECK(g.bytes() == 16);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }

    test_lifecycle();
    test_round_trip();
    test_clone();
    test_to_migration();
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
