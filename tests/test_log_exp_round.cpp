// Standalone CPU coverage for the brosoundml log / exp / round elementwise
// ops (CHUNK 6, family G).
//
// Verifies:
//   * log_forward / exp_forward / round_forward against hand-computed
//     references.
//   * FD gradient checks for log_backward and exp_backward.
//   * round_forward exercises the round-half-to-even (banker's rounding)
//     ties: 0.5->0, 1.5->2, 2.5->2, -2.5->-2.
//   * round_backward is the straight-through estimator — dX == dY exactly.
//   * exp(log(x)) round-trips for positive x.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(double a, double b, double abs_eps, double rel_eps) {
    const double d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, abs_eps, rel_eps, ctx)                    \
    do {                                                                       \
        const double _a = (actual);                                            \
        const double _e = (expected);                                          \
        if (!near_(_a, _e, (abs_eps), (rel_eps))) {                            \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g\n",     \
                        __FILE__, __LINE__, (ctx), _a, _e);                     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                      \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float next() {  // uniform in [-1, 1)
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) /
                                  static_cast<double>(1ULL << 53)) * 2.0f - 1.0f;
    }
};

static Tensor cpu_rand(Rng& rng, int r, int c) {
    Tensor t = Tensor::zeros_on(Device::CPU, r, c);
    for (int i = 0; i < r * c; ++i) t.host_f32_mut()[i] = rng.next();
    return t;
}

// ─── log ────────────────────────────────────────────────────────────────────
static void test_log() {
    const char* ctx = "log_forward";
    Rng rng(0x106F00ULL);
    const int R = 3, C = 7;
    // Positive inputs in (0.1, 4.1) — log() precondition is x > 0.
    Tensor X = cpu_rand(rng, R, C);
    for (int i = 0; i < R * C; ++i)
        X.host_f32_mut()[i] = 0.1f + 2.0f * (X.host_f32()[i] + 1.0f);

    Tensor Y;
    brotensor::log_forward(X, Y);
    EXPECT_TRUE(Y.rows == R && Y.cols == C, ctx);
    for (int i = 0; i < R * C; ++i)
        EXPECT_NEAR(Y.host_f32()[i], std::log(X.host_f32()[i]),
                    1e-6, 1e-5, ctx);

    // FD gradient check: loss = sum(dY * log(X)), dlog/dx = 1/x.
    Tensor dY = cpu_rand(rng, R, C);
    Tensor dX;
    brotensor::log_backward(X, dY, dX);
    EXPECT_TRUE(dX.rows == R && dX.cols == C, "log_backward shape");
    auto loss = [&](const Tensor& Xin) -> double {
        Tensor Yt;
        brotensor::log_forward(Xin, Yt);
        double sm = 0.0;
        for (int i = 0; i < Yt.rows * Yt.cols; ++i)
            sm += static_cast<double>(Yt.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };
    const double h = 1e-3;
    for (int i = 0; i < R * C; ++i) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, "log_backward FD");
    }
}

// ─── exp ────────────────────────────────────────────────────────────────────
static void test_exp() {
    const char* ctx = "exp_forward";
    Rng rng(0xE5900ULL);
    const int R = 4, C = 5;
    Tensor X = cpu_rand(rng, R, C);  // inputs in (-1, 1)

    Tensor Y;
    brotensor::exp_forward(X, Y);
    EXPECT_TRUE(Y.rows == R && Y.cols == C, ctx);
    for (int i = 0; i < R * C; ++i)
        EXPECT_NEAR(Y.host_f32()[i], std::exp(X.host_f32()[i]),
                    1e-6, 1e-5, ctx);

    // FD gradient check: dexp/dx = exp(x).
    Tensor dY = cpu_rand(rng, R, C);
    Tensor dX;
    brotensor::exp_backward(X, dY, dX);
    EXPECT_TRUE(dX.rows == R && dX.cols == C, "exp_backward shape");
    auto loss = [&](const Tensor& Xin) -> double {
        Tensor Yt;
        brotensor::exp_forward(Xin, Yt);
        double sm = 0.0;
        for (int i = 0; i < Yt.rows * Yt.cols; ++i)
            sm += static_cast<double>(Yt.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };
    const double h = 1e-3;
    for (int i = 0; i < R * C; ++i) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, "exp_backward FD");
    }
}

// exp(log(x)) round-trips for positive x.
static void test_log_exp_roundtrip() {
    const char* ctx = "exp(log(x)) roundtrip";
    Rng rng(0x4007121ULL);
    const int R = 2, C = 8;
    Tensor X = cpu_rand(rng, R, C);
    for (int i = 0; i < R * C; ++i)
        X.host_f32_mut()[i] = 0.2f + 3.0f * (X.host_f32()[i] + 1.0f);
    Tensor L, E;
    brotensor::log_forward(X, L);
    brotensor::exp_forward(L, E);
    for (int i = 0; i < R * C; ++i)
        EXPECT_NEAR(E.host_f32()[i], X.host_f32()[i], 1e-5, 1e-5, ctx);
}

// ─── round ──────────────────────────────────────────────────────────────────
static void test_round_forward() {
    const char* ctx = "round_forward ties-to-even";
    // Hand-picked values exercising the half-integer ties (banker's rounding).
    const float in[]  = { 0.5f, 1.5f, 2.5f, 3.5f, -0.5f, -1.5f, -2.5f,
                          0.4f, 0.6f, -0.4f, -0.6f, 2.0f, -3.0f };
    const float want[] = { 0.0f, 2.0f, 2.0f, 4.0f,  0.0f, -2.0f, -2.0f,
                          0.0f, 1.0f,  0.0f, -1.0f, 2.0f, -3.0f };
    const int n = static_cast<int>(sizeof(in) / sizeof(in[0]));
    Tensor X = Tensor::zeros_on(Device::CPU, 1, n);
    for (int i = 0; i < n; ++i) X.host_f32_mut()[i] = in[i];

    Tensor Y;
    brotensor::round_forward(X, Y);
    EXPECT_TRUE(Y.rows == 1 && Y.cols == n, ctx);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(Y.host_f32()[i], want[i], 0.0, 0.0, ctx);
}

// round_backward is the straight-through estimator: dX == dY exactly.
static void test_round_backward_ste() {
    const char* ctx = "round_backward STE identity";
    Rng rng(0x5710E0ULL);
    const int R = 5, C = 6;
    Tensor dY = cpu_rand(rng, R, C);
    Tensor dX;
    brotensor::round_backward(dY, dX);
    EXPECT_TRUE(dX.rows == R && dX.cols == C, ctx);
    for (int i = 0; i < R * C; ++i)
        EXPECT_NEAR(dX.host_f32()[i], dY.host_f32()[i], 0.0, 0.0, ctx);
}

// All three ops tolerate an in-place alias (x and y the same tensor).
static void test_aliasing() {
    const char* ctx = "in-place aliasing";
    Rng rng(0xA11A50ULL);
    Tensor X = cpu_rand(rng, 3, 4);
    for (int i = 0; i < 12; ++i)
        X.host_f32_mut()[i] = 0.3f + 2.0f * (X.host_f32()[i] + 1.0f);

    Tensor ref;
    brotensor::log_forward(X, ref);
    Tensor A = X.clone();
    brotensor::log_forward(A, A);                 // alias
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(A.host_f32()[i], ref.host_f32()[i], 1e-6, 1e-6, ctx);

    Tensor dY = cpu_rand(rng, 3, 4);
    Tensor dref;
    brotensor::exp_backward(X, dY, dref);
    Tensor B = dY.clone();
    brotensor::exp_backward(X, B, B);             // dX aliases dY
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(B.host_f32()[i], dref.host_f32()[i], 1e-6, 1e-6, ctx);
}

int main() {
    brotensor::init();
  try {
    test_log();
    test_exp();
    test_log_exp_roundtrip();
    test_round_forward();
    test_round_backward_ste();
    test_aliasing();
  } catch (const std::exception& e) {
    std::printf("test_log_exp_round: uncaught exception: %s\n", e.what());
    return 2;
  }

    if (g_failures == 0) {
        std::printf("test_log_exp_round: all checks passed\n");
        return 0;
    }
    std::printf("test_log_exp_round: %d FAILURE(S)\n", g_failures);
    return 1;
}
