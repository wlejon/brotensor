// CPU↔GPU parity for sum_rows_gpu / sum_cols_gpu / argmax_rows_gpu.

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

// GPU backend selection: prefer CUDA when present, else Metal. Cached after
// the first call (which must happen after brotensor::init()).
static Device gpu_device() {
    static const Device d = [] {
        if (brotensor::is_available(Device::CUDA))  return Device::CUDA;
        if (brotensor::is_available(Device::Metal)) return Device::Metal;
        return Device::CPU;
    }();
    return d;
}

// BF16 host helpers (mirrors parity_helpers.h inline versions).
static std::vector<uint16_t> to_bf16_vec(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp32_to_bf16_bits(v[i]);
    return o;
}
static float bf16_to_f32(uint16_t b) { return brotensor::bf16_bits_to_fp32(b); }

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

    Tensor Yg;
    Tensor Xg = Tensor::from_host_on(gpu_device(), X.data(), M, N);
    brotensor::sum_rows(Xg, Yg);
    CHECK(Yg.rows == M && Yg.cols == 1 && Yg.dtype == Dtype::FP32);
    std::vector<float> got(M);
    brotensor::sync_all();
    Yg.copy_to_host(got.data());
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

    Tensor Yg;
    auto Xh = to_fp16(X);
    Tensor Xg = Tensor::from_host_fp16_on(gpu_device(), Xh.data(), M, N);
    brotensor::sum_rows(Xg, Yg);
    CHECK(Yg.dtype == Dtype::FP16 && Yg.rows == M && Yg.cols == 1);
    std::vector<uint16_t> got(M);
    brotensor::sync_all();
    Yg.copy_to_host_fp16(got.data());
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
    Tensor Yg;
    Tensor Xg = Tensor::from_host_on(gpu_device(), X.data(), M, N);
    brotensor::sum_cols(Xg, Yg);
    CHECK(Yg.rows == 1 && Yg.cols == N && Yg.dtype == Dtype::FP32);
    std::vector<float> got(N);
    brotensor::sync_all();
    Yg.copy_to_host(got.data());
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
        Tensor Ig;
        Tensor Xg = Tensor::from_host_on(gpu_device(), X.data(), M, N);
        brotensor::argmax_rows(Xg, Ig);
        CHECK(Ig.dtype == Dtype::FP32 && Ig.rows == M && Ig.cols == 1);
        std::vector<float> got(M);
        brotensor::sync_all();
        Ig.copy_to_host(got.data());
        for (int m = 0; m < M; ++m) CHECK(static_cast<int>(got[m]) == ref[m]);
    }
    // FP16
    {
        Tensor Ig;
        auto Xh = to_fp16(X);
        Tensor Xg = Tensor::from_host_fp16_on(gpu_device(), Xh.data(), M, N);
        brotensor::argmax_rows(Xg, Ig);
        CHECK(Ig.dtype == Dtype::FP32);
        std::vector<float> got(M);
        brotensor::sync_all();
        Ig.copy_to_host(got.data());
        for (int m = 0; m < M; ++m) CHECK(static_cast<int>(got[m]) == ref[m]);
    }
    // FP32 input, INT32 output (opt-in via a pre-typed INT32 Idx tensor).
    {
        Tensor Ig = Tensor::empty_on(gpu_device(), 1, 1, Dtype::INT32);
        Tensor Xg = Tensor::from_host_on(gpu_device(), X.data(), M, N);
        brotensor::argmax_rows(Xg, Ig);
        CHECK(Ig.dtype == Dtype::INT32 && Ig.rows == M && Ig.cols == 1);
        brotensor::sync_all();
        Tensor Ic = Ig.to(Device::CPU);
        const int32_t* got = static_cast<const int32_t*>(Ic.host_raw());
        for (int m = 0; m < M; ++m) CHECK(got[m] == ref[m]);
    }
}

static void test_sum_rows_bf16() {
    std::printf("  sum_rows_gpu bf16\n");
    const int M = 5, N = 17;
    std::mt19937 rng(0x55);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    // Round-trip through BF16 for reference.
    auto Xb = to_bf16_vec(X);
    std::vector<float> ref(M, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            ref[m] += bf16_to_f32(Xb[m * N + n]);

    Tensor Yg;
    Tensor Xg = Tensor::from_host_bf16_on(gpu_device(), Xb.data(), M, N);
    brotensor::sum_rows(Xg, Yg);
    CHECK(Yg.dtype == Dtype::BF16 && Yg.rows == M && Yg.cols == 1);
    std::vector<uint16_t> got(M);
    brotensor::sync_all();
    Yg.copy_to_host_bf16(got.data());
    float max_err = 0.0f;
    for (int m = 0; m < M; ++m) {
        const float g = bf16_to_f32(got[m]);
        max_err = std::max(max_err, std::fabs(g - ref[m]));
    }
    std::printf("    bf16 max_err=%g\n", max_err);
    CHECK(max_err < 5e-2f);
}

static void test_sum_cols_bf16() {
    std::printf("  sum_cols_gpu bf16\n");
    const int M = 13, N = 9;
    std::mt19937 rng(0x66);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    auto Xb = to_bf16_vec(X);
    std::vector<float> ref(N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            ref[n] += bf16_to_f32(Xb[m * N + n]);

    Tensor Yg;
    Tensor Xg = Tensor::from_host_bf16_on(gpu_device(), Xb.data(), M, N);
    brotensor::sum_cols(Xg, Yg);
    CHECK(Yg.dtype == Dtype::BF16 && Yg.rows == 1 && Yg.cols == N);
    std::vector<uint16_t> got(N);
    brotensor::sync_all();
    Yg.copy_to_host_bf16(got.data());
    float max_err = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float g = bf16_to_f32(got[n]);
        max_err = std::max(max_err, std::fabs(g - ref[n]));
    }
    std::printf("    bf16 max_err=%g\n", max_err);
    CHECK(max_err < 5e-2f);
}

static void test_argmax_rows_bf16() {
    std::printf("  argmax_rows_gpu bf16\n");
    const int M = 6, N = 23;
    std::mt19937 rng(0x77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(M * N);
    for (auto& v : X) v = dist(rng);
    // Force unique max per row — peaks are large so BF16 rounding won't change argmax.
    for (int m = 0; m < M; ++m) X[m * N + (m % N)] = 5.0f + static_cast<float>(m);
    std::vector<int> ref(M);
    for (int m = 0; m < M; ++m) {
        int best = 0;
        for (int n = 1; n < N; ++n)
            if (X[m * N + n] > X[m * N + best]) best = n;
        ref[m] = best;
    }

    auto Xb = to_bf16_vec(X);
    Tensor Ig;
    Tensor Xg = Tensor::from_host_bf16_on(gpu_device(), Xb.data(), M, N);
    brotensor::argmax_rows(Xg, Ig);
    CHECK(Ig.dtype == Dtype::FP32 && Ig.rows == M && Ig.cols == 1);
    std::vector<float> got(M);
    brotensor::sync_all();
    Ig.copy_to_host(got.data());
    for (int m = 0; m < M; ++m) CHECK(static_cast<int>(got[m]) == ref[m]);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(gpu_device())) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_public_reductions\n");
    test_sum_rows_fp32();
    test_sum_rows_fp16();
    test_sum_rows_bf16();
    test_sum_cols_fp32();
    test_sum_cols_bf16();
    test_argmax_rows();
    test_argmax_rows_bf16();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
