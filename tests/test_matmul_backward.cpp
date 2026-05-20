// Parity for matmul_backward_gpu (FP32 + FP16) against a CPU reference,
// plus accumulate-into-semantics check.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}
static std::vector<float> rq(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v[i]));
    return o;
}

// dA[m, k] = sum_n dC[m, n] * B[k, n]
// dB[k, n] = sum_m A[m, k] * dC[m, n]
static void cpu_matmul_backward(const std::vector<float>& A,
                                const std::vector<float>& B,
                                const std::vector<float>& dC,
                                std::vector<float>& dA,
                                std::vector<float>& dB,
                                int M, int N, int K) {
    dA.assign(M * K, 0.0f);
    dB.assign(K * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            double s = 0.0;
            for (int n = 0; n < N; ++n)
                s += static_cast<double>(dC[m*N + n]) * B[k*N + n];
            dA[m*K + k] = static_cast<float>(s);
        }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int m = 0; m < M; ++m)
                s += static_cast<double>(A[m*K + k]) * dC[m*N + n];
            dB[k*N + n] = static_cast<float>(s);
        }
}

static void test_fp32() {
    std::printf("  matmul_backward_gpu fp32\n");
    const int M = 5, K = 7, N = 3;
    std::mt19937 rng(0xD00Du);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(M*K), B(K*N), dC(M*N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : dC) v = dist(rng);
    std::vector<float> dA_ref, dB_ref;
    cpu_matmul_backward(A, B, dC, dA_ref, dB_ref, M, N, K);

    Tensor Ag = Tensor::from_host_on(Device::CUDA, A.data(),  M, K);
    Tensor Bg = Tensor::from_host_on(Device::CUDA, B.data(),  K, N);
    Tensor dCg = Tensor::from_host_on(Device::CUDA, dC.data(), M, N);
    Tensor dAg = Tensor::zeros_on(Device::CUDA, M, K, Dtype::FP32);
    Tensor dBg = Tensor::zeros_on(Device::CUDA, K, N, Dtype::FP32);
    dAg.zero();
    dBg.zero();
    brotensor::matmul_backward(Ag, Bg, dCg, dAg, dBg);

    std::vector<float> dA_got(M*K), dB_got(K*N);
    brotensor::sync_all();
    dAg.copy_to_host(dA_got.data());
    dBg.copy_to_host(dB_got.data());

    float maxA = 0.0f, maxB = 0.0f;
    int badA = 0, badB = 0;
    for (size_t i = 0; i < dA_ref.size(); ++i) {
        const float e = std::fabs(dA_got[i] - dA_ref[i]);
        if (e > maxA) maxA = e;
        if (e > 1e-3f + 1e-3f * std::fabs(dA_ref[i])) ++badA;
    }
    for (size_t i = 0; i < dB_ref.size(); ++i) {
        const float e = std::fabs(dB_got[i] - dB_ref[i]);
        if (e > maxB) maxB = e;
        if (e > 1e-3f + 1e-3f * std::fabs(dB_ref[i])) ++badB;
    }
    std::printf("    fp32 dA max_err=%g bad=%d/%zu\n", maxA, badA, dA_ref.size());
    std::printf("    fp32 dB max_err=%g bad=%d/%zu\n", maxB, badB, dB_ref.size());
    CHECK(badA == 0);
    CHECK(badB == 0);

    // Accumulate semantics: second call doubles the result.
    brotensor::matmul_backward(Ag, Bg, dCg, dAg, dBg);
    brotensor::sync_all();
    dAg.copy_to_host(dA_got.data());
    dBg.copy_to_host(dB_got.data());
    float maxA2 = 0.0f, maxB2 = 0.0f;
    int badA2 = 0, badB2 = 0;
    for (size_t i = 0; i < dA_ref.size(); ++i) {
        const float e = std::fabs(dA_got[i] - 2.0f * dA_ref[i]);
        if (e > maxA2) maxA2 = e;
        if (e > 1e-3f + 1e-3f * std::fabs(2.0f * dA_ref[i])) ++badA2;
    }
    for (size_t i = 0; i < dB_ref.size(); ++i) {
        const float e = std::fabs(dB_got[i] - 2.0f * dB_ref[i]);
        if (e > maxB2) maxB2 = e;
        if (e > 1e-3f + 1e-3f * std::fabs(2.0f * dB_ref[i])) ++badB2;
    }
    std::printf("    fp32 accum2 dA max_err=%g bad=%d  dB max_err=%g bad=%d\n",
                maxA2, badA2, maxB2, badB2);
    CHECK(badA2 == 0);
    CHECK(badB2 == 0);
}

static void test_fp16() {
    std::printf("  matmul_backward_gpu fp16\n");
    const int M = 5, K = 7, N = 3;
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(M*K), B(K*N), dC(M*N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : dC) v = dist(rng);
    auto Aq = rq(A), Bq = rq(B), dCq = rq(dC);
    std::vector<float> dA_ref, dB_ref;
    cpu_matmul_backward(Aq, Bq, dCq, dA_ref, dB_ref, M, N, K);

    auto Ah = to_fp16(A), Bh = to_fp16(B), dCh = to_fp16(dC);
    Tensor Ag = Tensor::from_host_fp16_on(Device::CUDA, Ah.data(),  M, K);
    Tensor Bg = Tensor::from_host_fp16_on(Device::CUDA, Bh.data(),  K, N);
    Tensor dCg = Tensor::from_host_fp16_on(Device::CUDA, dCh.data(), M, N);
    Tensor dAg = Tensor::zeros_on(Device::CUDA, M, K, Dtype::FP16);
    Tensor dBg = Tensor::zeros_on(Device::CUDA, K, N, Dtype::FP16);
    dAg.zero();
    dBg.zero();
    brotensor::matmul_backward(Ag, Bg, dCg, dAg, dBg);

    std::vector<uint16_t> dA_got_h(M*K), dB_got_h(K*N);
    brotensor::sync_all();
    dAg.copy_to_host_fp16(dA_got_h.data());
    dBg.copy_to_host_fp16(dB_got_h.data());

    float maxA = 0.0f, maxB = 0.0f;
    int badA = 0, badB = 0;
    for (size_t i = 0; i < dA_ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(dA_got_h[i]);
        const float e = std::fabs(g - dA_ref[i]);
        if (e > maxA) maxA = e;
        if (e > 1e-2f + 5e-2f * std::fabs(dA_ref[i])) ++badA;
    }
    for (size_t i = 0; i < dB_ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(dB_got_h[i]);
        const float e = std::fabs(g - dB_ref[i]);
        if (e > maxB) maxB = e;
        if (e > 1e-2f + 5e-2f * std::fabs(dB_ref[i])) ++badB;
    }
    std::printf("    fp16 dA max_err=%g bad=%d/%zu\n", maxA, badA, dA_ref.size());
    std::printf("    fp16 dB max_err=%g bad=%d/%zu\n", maxB, badB, dB_ref.size());
    CHECK(badA == 0);
    CHECK(badB == 0);

    // Accumulate-into: second call doubles.
    brotensor::matmul_backward(Ag, Bg, dCg, dAg, dBg);
    brotensor::sync_all();
    dAg.copy_to_host_fp16(dA_got_h.data());
    dBg.copy_to_host_fp16(dB_got_h.data());
    int badA2 = 0, badB2 = 0;
    float maxA2 = 0.0f, maxB2 = 0.0f;
    for (size_t i = 0; i < dA_ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(dA_got_h[i]);
        const float e = std::fabs(g - 2.0f * dA_ref[i]);
        if (e > maxA2) maxA2 = e;
        if (e > 2e-2f + 1e-1f * std::fabs(2.0f * dA_ref[i])) ++badA2;
    }
    for (size_t i = 0; i < dB_ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(dB_got_h[i]);
        const float e = std::fabs(g - 2.0f * dB_ref[i]);
        if (e > maxB2) maxB2 = e;
        if (e > 2e-2f + 1e-1f * std::fabs(2.0f * dB_ref[i])) ++badB2;
    }
    std::printf("    fp16 accum2 dA max_err=%g bad=%d  dB max_err=%g bad=%d\n",
                maxA2, badA2, maxB2, badB2);
    CHECK(badA2 == 0);
    CHECK(badB2 == 0);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_matmul_backward\n");
    test_fp32();
    test_fp16();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
