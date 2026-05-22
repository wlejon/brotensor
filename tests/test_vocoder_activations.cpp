// Standalone CPU coverage for the brosoundml vocoder/codec activations
// (CHUNK 4, family C).
//
// Verifies:
//   * snake_forward (plain + snakebeta) against a hand-computed reference,
//     including the per-channel alpha/beta broadcast over the NCL plane.
//   * snake_backward FD gradient checks on dX, dAlpha and dBeta, for both
//     the plain-snake (dBeta == null) and snakebeta cases.
//   * elu_forward against a hand-computed reference (alpha != 1 too) and an
//     FD gradient check on elu_backward.
//   * leaky_relu_forward against a hand-computed reference and an FD
//     gradient check on leaky_relu_backward.
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
static Tensor cpu_zeros(int r, int c) {
    return Tensor::zeros_on(Device::CPU, r, c);
}

// ─── snake forward reference ────────────────────────────────────────────────
static void test_snake_forward(bool snakebeta) {
    const char* ctx = snakebeta ? "snake_forward snakebeta"
                                : "snake_forward plain";
    Rng rng(snakebeta ? 0x5BEE7AULL : 0x5A4E00ULL);
    const int N = 2, C = 3, L = 5;

    Tensor X = cpu_rand(rng, N, C * L);
    // alpha away from zero so the reference is well-conditioned.
    Tensor alpha = cpu_zeros(C, 1);
    for (int c = 0; c < C; ++c) alpha.host_f32_mut()[c] = 0.5f + 0.4f * (c + 1);
    Tensor beta = cpu_zeros(C, 1);
    for (int c = 0; c < C; ++c) beta.host_f32_mut()[c] = 0.7f + 0.3f * (c + 1);

    Tensor Y;
    brotensor::snake_forward(X, alpha, snakebeta ? &beta : nullptr, N, C, L, Y);
    EXPECT_TRUE(Y.rows == N && Y.cols == C * L, ctx);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const double a = alpha.host_f32()[c];
            const double denom = snakebeta ? beta.host_f32()[c] : a;
            const double r = 1.0 / denom;
            for (int l = 0; l < L; ++l) {
                const double x = X.host_f32()[(n * C + c) * L + l];
                const double s = std::sin(a * x);
                const double want = x + r * s * s;
                EXPECT_NEAR(Y.host_f32()[(n * C + c) * L + l], want,
                            1e-6, 1e-5, ctx);
            }
        }
    }
}

// ─── snake backward FD gradient checks ──────────────────────────────────────
static void test_snake_backward_fd(bool snakebeta) {
    const char* ctx = snakebeta ? "snake_backward snakebeta FD"
                                : "snake_backward plain FD";
    Rng rng(snakebeta ? 0xB57A00ULL : 0xB4A1F0ULL);
    const int N = 2, C = 3, L = 4;

    Tensor X = cpu_rand(rng, N, C * L);
    Tensor alpha = cpu_zeros(C, 1);
    for (int c = 0; c < C; ++c) alpha.host_f32_mut()[c] = 0.6f + 0.5f * (c + 1);
    Tensor beta = cpu_zeros(C, 1);
    for (int c = 0; c < C; ++c) beta.host_f32_mut()[c] = 0.8f + 0.35f * (c + 1);

    Tensor* betap = snakebeta ? &beta : nullptr;

    // Random upstream gradient; scalar loss = sum(dY * Y).
    Tensor dY = cpu_rand(rng, N, C * L);

    auto forward_loss = [&](const Tensor& Xin, const Tensor& Ain,
                            const Tensor& Bin) -> double {
        Tensor Y;
        brotensor::snake_forward(Xin, Ain, snakebeta ? &Bin : nullptr,
                                 N, C, L, Y);
        double sm = 0.0;
        for (int i = 0; i < Y.rows * Y.cols; ++i)
            sm += static_cast<double>(Y.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };

    // Analytic gradients. dAlpha / dBeta accumulate — caller zeros first.
    Tensor dX;
    Tensor dAlpha = cpu_zeros(C, 1);
    Tensor dBeta  = cpu_zeros(C, 1);
    brotensor::snake_backward(X, alpha, betap, dY, N, C, L,
                              dX, dAlpha, snakebeta ? &dBeta : nullptr);
    EXPECT_TRUE(dX.rows == N && dX.cols == C * L, ctx);

    const double h = 1e-3;
    // dX
    for (int i = 0; i < N * C * L; ++i) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (forward_loss(Xp, alpha, beta)
                           - forward_loss(Xm, alpha, beta)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, ctx);
    }
    // dAlpha
    for (int c = 0; c < C; ++c) {
        Tensor Ap = alpha.clone(), Am = alpha.clone();
        Ap.host_f32_mut()[c] += static_cast<float>(h);
        Am.host_f32_mut()[c] -= static_cast<float>(h);
        const double fd = (forward_loss(X, Ap, beta)
                           - forward_loss(X, Am, beta)) / (2.0 * h);
        EXPECT_NEAR(dAlpha.host_f32()[c], fd, 3e-3, 3e-3, ctx);
    }
    // dBeta (snakebeta only)
    if (snakebeta) {
        for (int c = 0; c < C; ++c) {
            Tensor Bp = beta.clone(), Bm = beta.clone();
            Bp.host_f32_mut()[c] += static_cast<float>(h);
            Bm.host_f32_mut()[c] -= static_cast<float>(h);
            const double fd = (forward_loss(X, alpha, Bp)
                               - forward_loss(X, alpha, Bm)) / (2.0 * h);
            EXPECT_NEAR(dBeta.host_f32()[c], fd, 3e-3, 3e-3, ctx);
        }
    }
}

// Verify dAlpha / dBeta genuinely ACCUMULATE (caller-zeros contract).
static void test_snake_backward_accumulates() {
    const char* ctx = "snake_backward accumulates";
    Rng rng(0xACC0A1ULL);
    const int N = 1, C = 2, L = 3;
    Tensor X = cpu_rand(rng, N, C * L);
    Tensor alpha = cpu_zeros(C, 1);
    for (int c = 0; c < C; ++c) alpha.host_f32_mut()[c] = 1.0f + 0.3f * c;
    Tensor beta = cpu_zeros(C, 1);
    for (int c = 0; c < C; ++c) beta.host_f32_mut()[c] = 1.1f + 0.2f * c;
    Tensor dY = cpu_rand(rng, N, C * L);

    Tensor dX;
    Tensor dA1 = cpu_zeros(C, 1), dB1 = cpu_zeros(C, 1);
    brotensor::snake_backward(X, alpha, &beta, dY, N, C, L, dX, dA1, &dB1);

    // Run twice into the same (non-zeroed) buffers — expect 2x.
    Tensor dA2 = cpu_zeros(C, 1), dB2 = cpu_zeros(C, 1);
    brotensor::snake_backward(X, alpha, &beta, dY, N, C, L, dX, dA2, &dB2);
    brotensor::snake_backward(X, alpha, &beta, dY, N, C, L, dX, dA2, &dB2);
    for (int c = 0; c < C; ++c) {
        EXPECT_NEAR(dA2.host_f32()[c], 2.0 * dA1.host_f32()[c], 1e-6, 1e-5, ctx);
        EXPECT_NEAR(dB2.host_f32()[c], 2.0 * dB1.host_f32()[c], 1e-6, 1e-5, ctx);
    }
}

// ─── elu ────────────────────────────────────────────────────────────────────
static void test_elu(float alpha) {
    char ctx[64];
    std::snprintf(ctx, sizeof ctx, "elu alpha=%.3g", alpha);
    Rng rng(0xE10000ULL ^ static_cast<uint64_t>(alpha * 1000.0f));
    const int R = 3, Cc = 7;
    Tensor X = cpu_rand(rng, R, Cc);
    // Scale so we exercise both signs with non-tiny magnitudes.
    for (int i = 0; i < R * Cc; ++i) X.host_f32_mut()[i] *= 2.0f;

    Tensor Y;
    brotensor::elu_forward(X, alpha, Y);
    EXPECT_TRUE(Y.rows == R && Y.cols == Cc, ctx);
    for (int i = 0; i < R * Cc; ++i) {
        const double v = X.host_f32()[i];
        const double want = v > 0.0 ? v
                                    : alpha * (std::exp(v) - 1.0);
        EXPECT_NEAR(Y.host_f32()[i], want, 1e-6, 1e-5, ctx);
    }

    // FD gradient check.
    Tensor dY = cpu_rand(rng, R, Cc);
    Tensor dX;
    brotensor::elu_backward(X, dY, alpha, dX);
    auto loss = [&](const Tensor& Xin) -> double {
        Tensor Yt;
        brotensor::elu_forward(Xin, alpha, Yt);
        double sm = 0.0;
        for (int i = 0; i < Yt.rows * Yt.cols; ++i)
            sm += static_cast<double>(Yt.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };
    const double h = 1e-3;
    for (int i = 0; i < R * Cc; ++i) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, ctx);
    }
}

// ─── leaky_relu ─────────────────────────────────────────────────────────────
static void test_leaky_relu(float slope) {
    char ctx[64];
    std::snprintf(ctx, sizeof ctx, "leaky_relu slope=%.3g", slope);
    Rng rng(0x1EA000ULL ^ static_cast<uint64_t>(slope * 100000.0f));
    const int R = 4, Cc = 5;
    Tensor X = cpu_rand(rng, R, Cc);

    Tensor Y;
    brotensor::leaky_relu_forward(X, slope, Y);
    EXPECT_TRUE(Y.rows == R && Y.cols == Cc, ctx);
    for (int i = 0; i < R * Cc; ++i) {
        const double v = X.host_f32()[i];
        const double want = v > 0.0 ? v : slope * v;
        EXPECT_NEAR(Y.host_f32()[i], want, 1e-6, 1e-5, ctx);
    }

    // FD gradient check (avoid samples right at the x==0 kink).
    Tensor dY = cpu_rand(rng, R, Cc);
    Tensor dX;
    brotensor::leaky_relu_backward(X, dY, slope, dX);
    auto loss = [&](const Tensor& Xin) -> double {
        Tensor Yt;
        brotensor::leaky_relu_forward(Xin, slope, Yt);
        double sm = 0.0;
        for (int i = 0; i < Yt.rows * Yt.cols; ++i)
            sm += static_cast<double>(Yt.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };
    const double h = 1e-3;
    for (int i = 0; i < R * Cc; ++i) {
        if (std::fabs(X.host_f32()[i]) < 2.0 * h) continue;  // skip the kink
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, ctx);
    }
}

// elu/leaky_relu default-alpha overloads stay consistent with the explicit form.
static void test_default_overloads() {
    const char* ctx = "default overloads";
    Rng rng(0xDEFA01ULL);
    Tensor X = cpu_rand(rng, 2, 6);
    for (int i = 0; i < 12; ++i) X.host_f32_mut()[i] *= 1.5f;

    Tensor Y1, Y2;
    brotensor::elu_forward(X, Y1);
    brotensor::elu_forward(X, 1.0f, Y2);
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(Y1.host_f32()[i], Y2.host_f32()[i], 1e-7, 1e-6, ctx);

    Tensor dY = cpu_rand(rng, 2, 6);
    Tensor dX1, dX2;
    brotensor::elu_backward(X, dY, dX1);
    brotensor::elu_backward(X, dY, 1.0f, dX2);
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(dX1.host_f32()[i], dX2.host_f32()[i], 1e-7, 1e-6, ctx);
}

int main() {
    brotensor::init();
  try {
    test_snake_forward(/*snakebeta=*/false);
    test_snake_forward(/*snakebeta=*/true);
    test_snake_backward_fd(/*snakebeta=*/false);
    test_snake_backward_fd(/*snakebeta=*/true);
    test_snake_backward_accumulates();

    test_elu(/*alpha=*/1.0f);
    test_elu(/*alpha=*/0.5f);
    test_elu(/*alpha=*/2.0f);

    test_leaky_relu(/*slope=*/0.01f);
    test_leaky_relu(/*slope=*/0.2f);

    test_default_overloads();
  } catch (const std::exception& e) {
    std::printf("test_vocoder_activations: uncaught exception: %s\n", e.what());
    return 2;
  }

    if (g_failures == 0) {
        std::printf("test_vocoder_activations: all checks passed\n");
        return 0;
    }
    std::printf("test_vocoder_activations: %d FAILURE(S)\n", g_failures);
    return 1;
}
