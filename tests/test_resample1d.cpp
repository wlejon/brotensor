// Standalone CPU coverage for the brosoundml 1D resampling ops
// (CHUNK 6, family E).
//
// Verifies:
//   * resample1d_forward against a hand-computed reference for an upsample
//     ratio and a downsample ratio, in both nearest and linear modes.
//   * resample1d to the same L_out is the exact identity (both modes).
//   * resample1d_backward (linear mode) against a finite-difference gradient
//     check, and that backward is the exact adjoint of forward (the
//     <dY, forward(X)> == <backward(dY), X> inner-product identity).
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

// Independent host reference for the forward op — mirrors the documented
// align_corners=False convention.
static double ref_forward(const std::vector<double>& x, int L_in,
                          int L_out, int mode, int dst) {
    const double scale = static_cast<double>(L_in) /
                         static_cast<double>(L_out);
    const double src = (dst + 0.5) * scale - 0.5;
    if (mode == 0) {
        double r = std::nearbyint(src);
        int idx = static_cast<int>(r);
        if (idx < 0) idx = 0;
        if (idx > L_in - 1) idx = L_in - 1;
        return x[idx];
    }
    double s = src;
    if (s < 0.0) s = 0.0;
    if (s > L_in - 1) s = L_in - 1;
    int x0 = static_cast<int>(std::floor(s));
    int x1 = (x0 + 1 < L_in) ? x0 + 1 : L_in - 1;
    double f = s - x0;
    return (1.0 - f) * x[x0] + f * x[x1];
}

// ─── forward references: upsample + downsample, both modes ──────────────────
static void test_forward_ref(int L_in, int L_out, int mode) {
    char ctx[96];
    std::snprintf(ctx, sizeof ctx, "resample1d_forward L_in=%d L_out=%d mode=%d",
                  L_in, L_out, mode);
    Rng rng(0x4E5A1100ULL ^ (static_cast<uint64_t>(L_in) << 16) ^
            (static_cast<uint64_t>(L_out) << 4) ^ static_cast<uint64_t>(mode));
    const int N = 2, C = 3;

    Tensor X = cpu_rand(rng, N, C * L_in);
    Tensor Y;
    brotensor::resample1d_forward(X, N, C, L_in, L_out, mode, Y);
    EXPECT_TRUE(Y.rows == N && Y.cols == C * L_out, ctx);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            std::vector<double> row(L_in);
            for (int l = 0; l < L_in; ++l)
                row[l] = X.host_f32()[(n * C + c) * L_in + l];
            for (int dst = 0; dst < L_out; ++dst) {
                const double want = ref_forward(row, L_in, L_out, mode, dst);
                EXPECT_NEAR(Y.host_f32()[(n * C + c) * L_out + dst], want,
                            1e-6, 1e-5, ctx);
            }
        }
    }
}

// Hand-checked tiny case: L_in=2 -> L_out=4 linear upsample.
// scale = 2/4 = 0.5;  src(dst) = (dst+0.5)*0.5 - 0.5.
//   dst0: src=-0.25 -> clamp 0     -> Y = x0
//   dst1: src= 0.25 -> x0=0 f=0.25 -> Y = 0.75 x0 + 0.25 x1
//   dst2: src= 0.75 -> x0=0 f=0.75 -> Y = 0.25 x0 + 0.75 x1
//   dst3: src= 1.25 -> clamp 1     -> Y = x1
static void test_forward_handcomputed() {
    const char* ctx = "resample1d_forward handcomputed 2->4 linear";
    Tensor X = Tensor::zeros_on(Device::CPU, 1, 2);
    X.host_f32_mut()[0] = 10.0f;
    X.host_f32_mut()[1] = 30.0f;
    Tensor Y;
    brotensor::resample1d_forward(X, /*N=*/1, /*C=*/1, /*L_in=*/2, /*L_out=*/4,
                                  /*mode=*/1, Y);
    EXPECT_TRUE(Y.rows == 1 && Y.cols == 4, ctx);
    EXPECT_NEAR(Y.host_f32()[0], 10.0, 1e-6, 1e-6, ctx);
    EXPECT_NEAR(Y.host_f32()[1], 0.75 * 10.0 + 0.25 * 30.0, 1e-6, 1e-6, ctx);
    EXPECT_NEAR(Y.host_f32()[2], 0.25 * 10.0 + 0.75 * 30.0, 1e-6, 1e-6, ctx);
    EXPECT_NEAR(Y.host_f32()[3], 30.0, 1e-6, 1e-6, ctx);

    // Same case, nearest: src rounds to {0,0,1,1}.
    const char* ctx2 = "resample1d_forward handcomputed 2->4 nearest";
    Tensor Yn;
    brotensor::resample1d_forward(X, 1, 1, 2, 4, /*mode=*/0, Yn);
    EXPECT_NEAR(Yn.host_f32()[0], 10.0, 1e-6, 1e-6, ctx2);
    EXPECT_NEAR(Yn.host_f32()[1], 10.0, 1e-6, 1e-6, ctx2);
    EXPECT_NEAR(Yn.host_f32()[2], 30.0, 1e-6, 1e-6, ctx2);
    EXPECT_NEAR(Yn.host_f32()[3], 30.0, 1e-6, 1e-6, ctx2);
}

// ─── identity: L_out == L_in returns the input exactly ──────────────────────
static void test_identity(int mode) {
    char ctx[64];
    std::snprintf(ctx, sizeof ctx, "resample1d identity mode=%d", mode);
    Rng rng(0x1DE07700ULL ^ static_cast<uint64_t>(mode));
    const int N = 2, C = 4, L = 7;
    Tensor X = cpu_rand(rng, N, C * L);
    Tensor Y;
    brotensor::resample1d_forward(X, N, C, L, L, mode, Y);
    EXPECT_TRUE(Y.rows == N && Y.cols == C * L, ctx);
    for (int i = 0; i < N * C * L; ++i)
        EXPECT_NEAR(Y.host_f32()[i], X.host_f32()[i], 0.0, 0.0, ctx);
}

// ─── backward FD gradient check (linear mode) ───────────────────────────────
static void test_backward_fd(int L_in, int L_out) {
    char ctx[80];
    std::snprintf(ctx, sizeof ctx, "resample1d_backward FD L_in=%d L_out=%d",
                  L_in, L_out);
    Rng rng(0xBADF00DULL ^ (static_cast<uint64_t>(L_in) << 8) ^
            static_cast<uint64_t>(L_out));
    const int N = 2, C = 2, mode = 1;

    Tensor X = cpu_rand(rng, N, C * L_in);
    Tensor dY = cpu_rand(rng, N, C * L_out);

    Tensor dX;
    brotensor::resample1d_backward(dY, N, C, L_in, L_out, mode, dX);
    EXPECT_TRUE(dX.rows == N && dX.cols == C * L_in, ctx);

    // scalar loss = sum(dY * forward(X))
    auto loss = [&](const Tensor& Xin) -> double {
        Tensor Y;
        brotensor::resample1d_forward(Xin, N, C, L_in, L_out, mode, Y);
        double sm = 0.0;
        for (int i = 0; i < Y.rows * Y.cols; ++i)
            sm += static_cast<double>(Y.host_f32()[i]) * dY.host_f32()[i];
        return sm;
    };
    const double h = 1e-3;
    for (int i = 0; i < N * C * L_in; ++i) {
        Tensor Xp = X.clone(), Xm = X.clone();
        Xp.host_f32_mut()[i] += static_cast<float>(h);
        Xm.host_f32_mut()[i] -= static_cast<float>(h);
        const double fd = (loss(Xp) - loss(Xm)) / (2.0 * h);
        EXPECT_NEAR(dX.host_f32()[i], fd, 3e-3, 3e-3, ctx);
    }
}

// Adjoint identity: <dY, forward(X)> == <backward(dY), X>, both modes.
static void test_adjoint(int L_in, int L_out, int mode) {
    char ctx[80];
    std::snprintf(ctx, sizeof ctx, "resample1d adjoint L_in=%d L_out=%d mode=%d",
                  L_in, L_out, mode);
    Rng rng(0xAD701000ULL ^ (static_cast<uint64_t>(L_in) << 12) ^
            (static_cast<uint64_t>(L_out) << 2) ^ static_cast<uint64_t>(mode));
    const int N = 2, C = 3;
    Tensor X = cpu_rand(rng, N, C * L_in);
    Tensor dY = cpu_rand(rng, N, C * L_out);

    Tensor Y;
    brotensor::resample1d_forward(X, N, C, L_in, L_out, mode, Y);
    Tensor dX;
    brotensor::resample1d_backward(dY, N, C, L_in, L_out, mode, dX);

    double lhs = 0.0, rhs = 0.0;
    for (int i = 0; i < N * C * L_out; ++i)
        lhs += static_cast<double>(dY.host_f32()[i]) * Y.host_f32()[i];
    for (int i = 0; i < N * C * L_in; ++i)
        rhs += static_cast<double>(dX.host_f32()[i]) * X.host_f32()[i];
    EXPECT_NEAR(lhs, rhs, 1e-4, 1e-5, ctx);
}

int main() {
    brotensor::init();
  try {
    // upsample (L_in < L_out) and downsample (L_in > L_out), both modes.
    test_forward_ref(/*L_in=*/4, /*L_out=*/10, /*mode=*/0);
    test_forward_ref(/*L_in=*/4, /*L_out=*/10, /*mode=*/1);
    test_forward_ref(/*L_in=*/12, /*L_out=*/5, /*mode=*/0);
    test_forward_ref(/*L_in=*/12, /*L_out=*/5, /*mode=*/1);
    test_forward_handcomputed();

    test_identity(/*mode=*/0);
    test_identity(/*mode=*/1);

    test_backward_fd(/*L_in=*/5, /*L_out=*/11);   // upsample
    test_backward_fd(/*L_in=*/11, /*L_out=*/4);   // downsample

    test_adjoint(/*L_in=*/5, /*L_out=*/13, /*mode=*/0);
    test_adjoint(/*L_in=*/5, /*L_out=*/13, /*mode=*/1);
    test_adjoint(/*L_in=*/13, /*L_out=*/5, /*mode=*/0);
    test_adjoint(/*L_in=*/13, /*L_out=*/5, /*mode=*/1);
  } catch (const std::exception& e) {
    std::printf("test_resample1d: uncaught exception: %s\n", e.what());
    return 2;
  }

    if (g_failures == 0) {
        std::printf("test_resample1d: all checks passed\n");
        return 0;
    }
    std::printf("test_resample1d: %d FAILURE(S)\n", g_failures);
    return 1;
}
