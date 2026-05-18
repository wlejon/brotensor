// Parity for matmul_gpu (FP32 and FP16) against a CPU reference.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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

static void cpu_matmul(const std::vector<float>& A,
                       const std::vector<float>& B,
                       std::vector<float>& C,
                       int M, int N, int K) {
    C.assign(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += static_cast<double>(A[m * K + k]) *
                     static_cast<double>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<float>(s);
        }
}

static void test_fp32() {
    std::printf("  matmul_gpu fp32\n");
    const int M = 17, K = 23, N = 11;
    std::mt19937 rng(0xC0DEu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(M * K), B(K * N), ref;
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    cpu_matmul(A, B, ref, M, N, K);

    GpuTensor Ag, Bg, Cg;
    brotensor::upload(A.data(), M, K, Ag);
    brotensor::upload(B.data(), K, N, Bg);
    brotensor::matmul_gpu(Ag, Bg, Cg);
    CHECK(Cg.rows == M && Cg.cols == N && Cg.dtype == Dtype::FP32);
    std::vector<float> got(Cg.size());
    brotensor::download(Cg, got.data());
    brotensor::cuda_sync();

    float max_err = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-3f + 1e-3f * std::fabs(ref[i])) ++bad;
    }
    std::printf("    fp32 max_err=%g bad=%d/%zu\n", max_err, bad, ref.size());
    CHECK(bad == 0);
}

static void test_fp16() {
    std::printf("  matmul_gpu fp16\n");
    const int M = 19, K = 32, N = 13;
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> A(M * K), B(K * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    auto Aq = rq(A), Bq = rq(B);
    std::vector<float> ref;
    cpu_matmul(Aq, Bq, ref, M, N, K);

    GpuTensor Ag, Bg, Cg;
    auto Ah = to_fp16(A), Bh = to_fp16(B);
    brotensor::upload_fp16(Ah.data(), M, K, Ag);
    brotensor::upload_fp16(Bh.data(), K, N, Bg);
    brotensor::matmul_gpu(Ag, Bg, Cg);
    CHECK(Cg.rows == M && Cg.cols == N && Cg.dtype == Dtype::FP16);
    std::vector<uint16_t> got(Cg.size());
    brotensor::download_fp16(Cg, got.data());
    brotensor::cuda_sync();

    float max_err = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 5e-2f * std::fabs(ref[i])) ++bad;
    }
    std::printf("    fp16 max_err=%g bad=%d/%zu\n", max_err, bad, ref.size());
    CHECK(bad == 0);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_matmul\n");
    test_fp32();
    test_fp16();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
