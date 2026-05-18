// CPU↔GPU parity for sum_rows_gpu / sum_cols_gpu / argmax_rows_gpu.

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

static void test_sum_rows_fp32() {
    std::printf("  sum_rows_gpu fp32\n");
    const int M = 7, N = 53;
    std::mt19937 rng(0x11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    std::vector<float> ref(M, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) ref[m] += X[m * N + n];

    GpuTensor Xg, Yg;
    brotensor::upload(X.data(), M, N, Xg);
    brotensor::sum_rows_gpu(Xg, Yg);
    CHECK(Yg.rows == M && Yg.cols == 1 && Yg.dtype == Dtype::FP32);
    std::vector<float> got(M);
    brotensor::download(Yg, got.data());
    brotensor::cuda_sync();
    float max_err = 0.0f;
    for (int m = 0; m < M; ++m)
        max_err = std::max(max_err, std::fabs(got[m] - ref[m]));
    std::printf("    fp32 max_err=%g\n", max_err);
    CHECK(max_err < 1e-3f);
}

static void test_sum_rows_fp16() {
    std::printf("  sum_rows_gpu fp16\n");
    const int M = 5, N = 17;
    std::mt19937 rng(0x22);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    auto Xq = rq(X);
    std::vector<float> ref(M, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) ref[m] += Xq[m * N + n];

    GpuTensor Xg, Yg;
    auto Xh = to_fp16(X);
    brotensor::upload_fp16(Xh.data(), M, N, Xg);
    brotensor::sum_rows_gpu(Xg, Yg);
    CHECK(Yg.dtype == Dtype::FP16 && Yg.rows == M && Yg.cols == 1);
    std::vector<uint16_t> got(M);
    brotensor::download_fp16(Yg, got.data());
    brotensor::cuda_sync();
    float max_err = 0.0f;
    for (int m = 0; m < M; ++m) {
        const float g = brotensor::fp16_bits_to_fp32(got[m]);
        max_err = std::max(max_err, std::fabs(g - ref[m]));
    }
    std::printf("    fp16 max_err=%g\n", max_err);
    CHECK(max_err < 5e-2f);
}

static void test_sum_cols_fp32() {
    std::printf("  sum_cols_gpu fp32\n");
    const int M = 13, N = 9;
    std::mt19937 rng(0x33);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    std::vector<float> ref(N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) ref[n] += X[m * N + n];
    GpuTensor Xg, Yg;
    brotensor::upload(X.data(), M, N, Xg);
    brotensor::sum_cols_gpu(Xg, Yg);
    CHECK(Yg.rows == 1 && Yg.cols == N && Yg.dtype == Dtype::FP32);
    std::vector<float> got(N);
    brotensor::download(Yg, got.data());
    brotensor::cuda_sync();
    float max_err = 0.0f;
    for (int n = 0; n < N; ++n)
        max_err = std::max(max_err, std::fabs(got[n] - ref[n]));
    std::printf("    fp32 max_err=%g\n", max_err);
    CHECK(max_err < 1e-3f);
}

static void test_argmax_rows() {
    std::printf("  argmax_rows_gpu fp32 + fp16\n");
    const int M = 6, N = 23;
    std::mt19937 rng(0x44);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    // Force a unique max per row for unambiguous indices.
    for (int m = 0; m < M; ++m) X[m * N + (m % N)] = 5.0f + static_cast<float>(m);
    std::vector<int> ref(M);
    for (int m = 0; m < M; ++m) {
        int best = 0;
        for (int n = 1; n < N; ++n)
            if (X[m * N + n] > X[m * N + best]) best = n;
        ref[m] = best;
    }
    // FP32
    {
        GpuTensor Xg, Ig;
        brotensor::upload(X.data(), M, N, Xg);
        brotensor::argmax_rows_gpu(Xg, Ig);
        CHECK(Ig.dtype == Dtype::FP32 && Ig.rows == M && Ig.cols == 1);
        std::vector<float> got(M);
        brotensor::download(Ig, got.data());
        brotensor::cuda_sync();
        for (int m = 0; m < M; ++m) CHECK(static_cast<int>(got[m]) == ref[m]);
    }
    // FP16
    {
        GpuTensor Xg, Ig;
        auto Xh = to_fp16(X);
        brotensor::upload_fp16(Xh.data(), M, N, Xg);
        brotensor::argmax_rows_gpu(Xg, Ig);
        CHECK(Ig.dtype == Dtype::FP32);
        std::vector<float> got(M);
        brotensor::download(Ig, got.data());
        brotensor::cuda_sync();
        for (int m = 0; m < M; ++m) CHECK(static_cast<int>(got[m]) == ref[m]);
    }
}

int main() {
    brotensor::cuda_init();
    std::printf("test_public_reductions\n");
    test_sum_rows_fp32();
    test_sum_rows_fp16();
    test_sum_cols_fp32();
    test_argmax_rows();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
