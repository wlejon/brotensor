// CPU coverage for the Philox-driven noise ops:
//   randn, rand_uniform, rand_bernoulli, randn_truncated.
//
// Verifies:
//   * determinism — identical (key, counter) -> identical output.
//   * substream addressing — element i depends only on (key, counter + i),
//     not on the surrounding tensor shape.
//   * uniform: range and mean / variance match U[0,1).
//   * normal:  mean ≈ 0, variance ≈ 1 over a large sample.
//   * bernoulli: fraction of 1s ≈ p; p=0 -> all zeros, p=1 -> all ones.
//   * truncated normal: every sample falls within [lo, hi]; mean ≈ 0 for
//     symmetric [-a, a].
//
// CPU-resident, FP32 output. Plain executable; non-zero exit on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  %s\n",                           \
                        __FILE__, __LINE__, (ctx), #cond);                     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_NEAR(actual, expected, tol, ctx)                                \
    do {                                                                       \
        const double _a = (double)(actual);                                    \
        const double _e = (double)(expected);                                  \
        if (std::fabs(_a - _e) > (tol)) {                                      \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.6f expected=%.6f "      \
                        "tol=%.6f\n",                                          \
                        __FILE__, __LINE__, (ctx), _a, _e, (double)(tol));     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static Tensor make(int R, int C) {
    return Tensor::zeros_on(Device::CPU, R, C, Dtype::FP32);
}

// ─── determinism ────────────────────────────────────────────────────────────
static void test_determinism() {
    Tensor a = make(8, 16);
    Tensor b = make(8, 16);
    brotensor::randn(0x1234ull, 7ull, a);
    brotensor::randn(0x1234ull, 7ull, b);
    const float* pa = a.host_f32();
    const float* pb = b.host_f32();
    bool same = true;
    for (int i = 0; i < 128; ++i) {
        if (pa[i] != pb[i]) { same = false; break; }
    }
    EXPECT_TRUE(same, "randn is deterministic in (key, counter)");

    brotensor::rand_uniform(0x1234ull, 7ull, a);
    brotensor::rand_uniform(0x1234ull, 7ull, b);
    pa = a.host_f32(); pb = b.host_f32();
    same = true;
    for (int i = 0; i < 128; ++i) {
        if (pa[i] != pb[i]) { same = false; break; }
    }
    EXPECT_TRUE(same, "rand_uniform is deterministic");
}

// ─── substream addressing: counter-shifted tail equals shorter draw ─────────
static void test_substream_addressing() {
    Tensor full = make(1, 64);
    Tensor tail = make(1, 32);
    brotensor::randn(99ull, 1000ull, full);
    brotensor::randn(99ull, 1032ull, tail);  // start 32 elements in
    const float* pf = full.host_f32();
    const float* pt = tail.host_f32();
    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (pf[32 + i] != pt[i]) { ok = false; break; }
    }
    EXPECT_TRUE(ok, "randn element i uses substream (counter + i)");
}

// ─── uniform: range and moments ─────────────────────────────────────────────
static void test_uniform_moments() {
    const int N = 100000;
    Tensor y = make(1, N);
    brotensor::rand_uniform(0xC0FFEEull, 0ull, y);
    const float* p = y.host_f32();
    double sum = 0.0, sumsq = 0.0;
    bool in_range = true;
    for (int i = 0; i < N; ++i) {
        const float v = p[i];
        if (!(v >= 0.0f && v < 1.0f)) { in_range = false; }
        sum += v;
        sumsq += static_cast<double>(v) * v;
    }
    EXPECT_TRUE(in_range, "rand_uniform stays in [0, 1)");
    const double mean = sum / N;
    const double var  = sumsq / N - mean * mean;
    EXPECT_NEAR(mean, 0.5,        0.01, "rand_uniform mean ≈ 0.5");
    EXPECT_NEAR(var,  1.0 / 12.0, 0.01, "rand_uniform variance ≈ 1/12");
}

// ─── normal: moments ────────────────────────────────────────────────────────
static void test_normal_moments() {
    const int N = 100000;
    Tensor y = make(1, N);
    brotensor::randn(0xBEEFull, 0ull, y);
    const float* p = y.host_f32();
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < N; ++i) {
        const double v = p[i];
        sum += v;
        sumsq += v * v;
    }
    const double mean = sum / N;
    const double var  = sumsq / N - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.02, "randn mean ≈ 0");
    EXPECT_NEAR(var,  1.0, 0.05, "randn variance ≈ 1");
}

// ─── bernoulli: extremes + middle ───────────────────────────────────────────
static void test_bernoulli() {
    const int N = 50000;
    Tensor y = make(1, N);

    brotensor::rand_bernoulli(0.0f, 0xAull, 0ull, y);
    const float* p = y.host_f32();
    bool all_zero = true;
    for (int i = 0; i < N; ++i) if (p[i] != 0.0f) { all_zero = false; break; }
    EXPECT_TRUE(all_zero, "rand_bernoulli(p=0) -> all zeros");

    brotensor::rand_bernoulli(1.0f, 0xBull, 0ull, y);
    p = y.host_f32();
    bool all_one = true;
    for (int i = 0; i < N; ++i) if (p[i] != 1.0f) { all_one = false; break; }
    EXPECT_TRUE(all_one, "rand_bernoulli(p=1) -> all ones");

    brotensor::rand_bernoulli(0.3f, 0xCull, 0ull, y);
    p = y.host_f32();
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += p[i];
    EXPECT_NEAR(sum / N, 0.3, 0.02, "rand_bernoulli(p=0.3) empirical ≈ 0.3");
}

// ─── truncated normal: bounded + mean ───────────────────────────────────────
static void test_truncated_normal() {
    const int N = 50000;
    Tensor y = make(1, N);
    brotensor::randn_truncated(-2.0f, 2.0f, 0xDEADull, 0ull, y);
    const float* p = y.host_f32();
    bool in_range = true;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        if (!(p[i] >= -2.0f && p[i] <= 2.0f)) { in_range = false; }
        sum += p[i];
    }
    EXPECT_TRUE(in_range, "randn_truncated samples lie in [lo, hi]");
    EXPECT_NEAR(sum / N, 0.0, 0.03,
                "randn_truncated symmetric interval has mean ≈ 0");
}

// ─── zero-size tensor is a no-op ────────────────────────────────────────────
static void test_zero_size() {
    Tensor y = make(0, 0);
    try {
        brotensor::randn(1ull, 1ull, y);
        brotensor::rand_uniform(1ull, 1ull, y);
        brotensor::rand_bernoulli(0.5f, 1ull, 1ull, y);
        brotensor::randn_truncated(-1.0f, 1.0f, 1ull, 1ull, y);
    } catch (...) {
        EXPECT_TRUE(false, "zero-size tensors should be a no-op, not throw");
    }
}

int main() {
    try {
        brotensor::init();
        test_determinism();
        test_substream_addressing();
        test_uniform_moments();
        test_normal_moments();
        test_bernoulli();
        test_truncated_normal();
        test_zero_size();
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (g_failures > 0) {
        std::printf("FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
