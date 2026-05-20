// Streams smoke test.
//
// The legacy public stream API (cuda_set_stream / cuda_current_stream /
// cuda_stream_sync) was removed in the device-free Tensor refactor: stream
// control is now an internal detail of the CUDA backend and not part of the
// public surface (see runtime.h — only init / sync / sync_all are exposed).
//
// As a result the original multi-stream handle assertions can no longer be
// expressed through the public API. This test is therefore reduced to a
// minimal correctness smoke test: it runs matmul on CUDA tensors and verifies
// the result against a CPU reference.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void test_matmul_smoke() {
    std::printf("  matmul fp16 smoke\n");
    const int M = 64, K = 64, N = 64;
    std::mt19937 rng(0x77);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> A(M*K), B(K*N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // CPU reference (FP32).
    std::vector<float> ref(M*N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += static_cast<double>(A[m*K + k]) * B[k*N + n];
            ref[m*N + n] = static_cast<float>(s);
        }

    auto ah = to_fp16(A), bh = to_fp16(B);
    Tensor Ag = Tensor::from_host_fp16_on(Device::CUDA, ah.data(), M, K);
    Tensor Bg = Tensor::from_host_fp16_on(Device::CUDA, bh.data(), K, N);

    Tensor Cg;
    brotensor::matmul(Ag, Bg, Cg);
    CHECK(Cg.rows == M && Cg.cols == N);

    std::vector<uint16_t> got(M*N);
    Cg.copy_to_host_fp16(got.data());
    brotensor::sync_all();

    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M*N; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 1e-2f * std::fabs(ref[i])) ++bad;
    }
    std::printf("    max_err=%g bad=%d/%d\n", max_err, bad, M*N);
    CHECK(bad == 0);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_streams\n");
    test_matmul_smoke();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
