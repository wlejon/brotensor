// Minimal smoke test for brotensor GpuTensor lifecycle + upload/download
// round-trip + one hand-checked op (relu_forward_gpu). Deep CPU↔GPU parity
// coverage lives in brogameagent's tests since those need brogameagent's CPU
// reference impls.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/ops.h>

#include <cmath>
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

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll smoke checks passed.\n");
    return 0;
}
