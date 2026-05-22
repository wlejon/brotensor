// Standalone CPU coverage for the brosoundml spectral / FFT core (CHUNK 1).
//
// Verifies:
//   * DFT-definition correctness against a naive O(N^2) reference, for a range
//     of sizes (power-of-2, smooth non-power-of-2 including Whisper's 400,
//     and prime sizes 53 / 401 to exercise the Bluestein fallback).
//   * Round-trip identities ifft(fft(x)) == x and irfft(rfft(x)) == x.
//   * Parseval / energy conservation.
//   * Complex elementwise ops (mul / abs / angle / from_polar) against
//     hand-computed expectations.
//   * Finite-difference gradient checks for every backward op
//     (complex_mul_backward, complex_abs_backward, rfft_backward,
//      irfft_backward) plus the fft/ifft adjoint-via-existing-transform path.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                      \
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

static Tensor cpu_zeros(int r, int c = 1) {
    return Tensor::zeros_on(Device::CPU, r, c);
}

// Deterministic small pseudo-random generator (splitmix64-derived) so the
// tests are reproducible without <random> dependency assumptions.
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

// ─── naive reference DFT in double precision ───────────────────────────────
static void ref_dft(const std::vector<double>& re, const std::vector<double>& im,
                     std::vector<double>& ore, std::vector<double>& oim,
                     int sign) {
    const int N = static_cast<int>(re.size());
    ore.assign(N, 0.0);
    oim.assign(N, 0.0);
    const double s = sign * 2.0 * 3.14159265358979323846 / N;
    for (int k = 0; k < N; ++k) {
        double ar = 0.0, ai = 0.0;
        for (int n = 0; n < N; ++n) {
            const double a = s * (static_cast<long long>(k) * n % N);
            const double c = std::cos(a), si = std::sin(a);
            ar += re[n] * c - im[n] * si;
            ai += re[n] * si + im[n] * c;
        }
        ore[k] = ar;
        oim[k] = ai;
    }
}

// ─── fft / ifft correctness + round-trip + Parseval ────────────────────────
static void test_fft_definition(int N) {
    char ctx[64];
    Rng rng(0x100 + N);

    // Build an (1, 2N) interleaved-complex input.
    Tensor x = cpu_zeros(1, 2 * N);
    std::vector<double> re(N), im(N);
    for (int n = 0; n < N; ++n) {
        re[n] = rng.next();
        im[n] = rng.next();
        x.host_f32_mut()[2 * n]     = static_cast<float>(re[n]);
        x.host_f32_mut()[2 * n + 1] = static_cast<float>(im[n]);
    }

    Tensor y = cpu_zeros(1, 2 * N);
    brotensor::fft(x, y);

    std::vector<double> ore, oim;
    ref_dft(re, im, ore, oim, -1);
    for (int k = 0; k < N; ++k) {
        std::snprintf(ctx, sizeof(ctx), "fft N=%d re[%d]", N, k);
        EXPECT_NEAR(y.host_f32()[2 * k], ore[k], 1e-3, 2e-4, ctx);
        std::snprintf(ctx, sizeof(ctx), "fft N=%d im[%d]", N, k);
        EXPECT_NEAR(y.host_f32()[2 * k + 1], oim[k], 1e-3, 2e-4, ctx);
    }

    // Round trip: ifft(fft(x)) == x.
    Tensor xr = cpu_zeros(1, 2 * N);
    brotensor::ifft(y, xr);
    for (int n = 0; n < N; ++n) {
        std::snprintf(ctx, sizeof(ctx), "ifft(fft) roundtrip N=%d re[%d]", N, n);
        EXPECT_NEAR(xr.host_f32()[2 * n], re[n], 1e-4, 1e-3, ctx);
        std::snprintf(ctx, sizeof(ctx), "ifft(fft) roundtrip N=%d im[%d]", N, n);
        EXPECT_NEAR(xr.host_f32()[2 * n + 1], im[n], 1e-4, 1e-3, ctx);
    }

    // Parseval: sum |x|^2 == (1/N) sum |X|^2.
    double et = 0.0, ef = 0.0;
    for (int n = 0; n < N; ++n) et += re[n] * re[n] + im[n] * im[n];
    for (int k = 0; k < N; ++k) {
        const double yr = y.host_f32()[2 * k], yi = y.host_f32()[2 * k + 1];
        ef += yr * yr + yi * yi;
    }
    ef /= N;
    std::snprintf(ctx, sizeof(ctx), "Parseval N=%d", N);
    EXPECT_NEAR(et, ef, 1e-3, 1e-3, ctx);
}

// ─── rfft / irfft correctness + round-trip + Parseval ──────────────────────
static void test_rfft(int L) {
    char ctx[64];
    Rng rng(0x900 + L);
    const int C = L / 2 + 1;

    Tensor x = cpu_zeros(1, L);
    std::vector<double> re(L), im(L, 0.0);
    for (int n = 0; n < L; ++n) {
        re[n] = rng.next();
        x.host_f32_mut()[n] = static_cast<float>(re[n]);
    }

    Tensor y = cpu_zeros(1, 2 * C);
    brotensor::rfft(x, y);

    // Compare against the first C bins of a full complex DFT.
    std::vector<double> ore, oim;
    ref_dft(re, im, ore, oim, -1);
    for (int k = 0; k < C; ++k) {
        std::snprintf(ctx, sizeof(ctx), "rfft L=%d re[%d]", L, k);
        EXPECT_NEAR(y.host_f32()[2 * k], ore[k], 1e-3, 2e-4, ctx);
        std::snprintf(ctx, sizeof(ctx), "rfft L=%d im[%d]", L, k);
        EXPECT_NEAR(y.host_f32()[2 * k + 1], oim[k], 1e-3, 2e-4, ctx);
    }

    // Round trip irfft(rfft(x)) == x.
    Tensor xr = cpu_zeros(1, L);
    brotensor::irfft(y, L, xr);
    for (int n = 0; n < L; ++n) {
        std::snprintf(ctx, sizeof(ctx), "irfft(rfft) roundtrip L=%d [%d]", L, n);
        EXPECT_NEAR(xr.host_f32()[n], re[n], 1e-4, 1e-3, ctx);
    }
}

// ─── complex elementwise ops ───────────────────────────────────────────────
static void test_complex_elementwise() {
    std::printf("complex elementwise (mul/abs/angle/from_polar)\n");

    // 1 row, 2 complex bins.
    Tensor a = cpu_zeros(1, 4), b = cpu_zeros(1, 4), y = cpu_zeros(1, 4);
    // bin0: (1+2i)*(3+4i) = (3-8) + (4+6)i = -5 + 10i
    // bin1: (-1+0.5i)*(2-1i) = (-2+0.5) + (1+1)i = -1.5 + 2i
    a.host_f32_mut()[0] = 1;  a.host_f32_mut()[1] = 2;
    a.host_f32_mut()[2] = -1; a.host_f32_mut()[3] = 0.5f;
    b.host_f32_mut()[0] = 3;  b.host_f32_mut()[1] = 4;
    b.host_f32_mut()[2] = 2;  b.host_f32_mut()[3] = -1;
    brotensor::complex_mul(a, b, y);
    EXPECT_NEAR(y.host_f32()[0], -5.0, 1e-6, 1e-6, "complex_mul re0");
    EXPECT_NEAR(y.host_f32()[1], 10.0, 1e-6, 1e-6, "complex_mul im0");
    EXPECT_NEAR(y.host_f32()[2], -1.5, 1e-6, 1e-6, "complex_mul re1");
    EXPECT_NEAR(y.host_f32()[3],  2.0, 1e-6, 1e-6, "complex_mul im1");

    // abs: |3+4i| = 5, |(-1+0.5i)| = sqrt(1.25)
    Tensor z = cpu_zeros(1, 4), mag = cpu_zeros(1, 2);
    z.host_f32_mut()[0] = 3;  z.host_f32_mut()[1] = 4;
    z.host_f32_mut()[2] = -1; z.host_f32_mut()[3] = 0.5f;
    brotensor::complex_abs(z, mag);
    EXPECT_NEAR(mag.host_f32()[0], 5.0, 1e-6, 1e-6, "complex_abs 0");
    EXPECT_NEAR(mag.host_f32()[1], std::sqrt(1.25), 1e-6, 1e-6, "complex_abs 1");

    // angle: atan2(4,3), atan2(0.5,-1)
    Tensor ang = cpu_zeros(1, 2);
    brotensor::complex_angle(z, ang);
    EXPECT_NEAR(ang.host_f32()[0], std::atan2(4.0, 3.0), 1e-6, 1e-6, "angle 0");
    EXPECT_NEAR(ang.host_f32()[1], std::atan2(0.5, -1.0), 1e-6, 1e-6, "angle 1");

    // from_polar then abs/angle is identity.
    Tensor zp = cpu_zeros(1, 4);
    brotensor::complex_from_polar(mag, ang, zp);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(zp.host_f32()[i], z.host_f32()[i], 1e-5, 1e-5,
                    "from_polar roundtrip");
    }
}

// ─── generic finite-difference gradient checker ────────────────────────────
template <typename Fn>
static void fd_check(const std::string& name, float* in, int n,
                     const std::vector<float>& analytic, Fn&& loss_at,
                     float h = 1e-3f, float abs_eps = 2e-2f,
                     float rel_eps = 2e-2f) {
    for (int i = 0; i < n; ++i) {
        const float saved = in[i];
        in[i] = saved + h;
        const float lp = loss_at();
        in[i] = saved - h;
        const float lm = loss_at();
        in[i] = saved;
        const float num = (lp - lm) / (2.0f * h);
        if (!near_(analytic[i], num, abs_eps, rel_eps)) {
            std::printf("  FAIL  fd-grad %s  i=%d  analytic=%.6g numeric=%.6g\n",
                        name.c_str(), i, analytic[i], num);
            ++g_failures;
        }
    }
}

// complex_mul_backward: loss = sum(w .* y), y = complex_mul(a,b).
static void test_complex_mul_backward() {
    std::printf("complex_mul_backward (fd grad)\n");
    const int C = 3;
    Tensor a = cpu_zeros(2, 2 * C), b = cpu_zeros(2, 2 * C);
    Tensor w = cpu_zeros(2, 2 * C);  // upstream weights
    Rng rng(0x2222);
    for (int i = 0; i < a.size(); ++i) {
        a.host_f32_mut()[i] = rng.next();
        b.host_f32_mut()[i] = rng.next();
        w.host_f32_mut()[i] = rng.next();
    }
    Tensor y = cpu_zeros(2, 2 * C);
    auto loss = [&]() {
        brotensor::complex_mul(a, b, y);
        float s = 0.0f;
        for (int i = 0; i < y.size(); ++i) s += w.host_f32()[i] * y.host_f32()[i];
        return s;
    };
    loss();
    Tensor dA = cpu_zeros(2, 2 * C), dB = cpu_zeros(2, 2 * C);
    brotensor::complex_mul_backward(a, b, w, dA, dB);
    std::vector<float> gA(dA.host_f32(), dA.host_f32() + dA.size());
    std::vector<float> gB(dB.host_f32(), dB.host_f32() + dB.size());
    fd_check("complex_mul dA", a.host_f32_mut(), a.size(), gA, loss);
    fd_check("complex_mul dB", b.host_f32_mut(), b.size(), gB, loss);
}

// complex_abs_backward: loss = sum(w .* abs(z)).
static void test_complex_abs_backward() {
    std::printf("complex_abs_backward (fd grad)\n");
    const int C = 4;
    Tensor z = cpu_zeros(2, 2 * C), w = cpu_zeros(2, C);
    Rng rng(0x3333);
    for (int i = 0; i < z.size(); ++i) z.host_f32_mut()[i] = rng.next() + 1.5f;
    for (int i = 0; i < w.size(); ++i) w.host_f32_mut()[i] = rng.next();
    Tensor mag = cpu_zeros(2, C);
    auto loss = [&]() {
        brotensor::complex_abs(z, mag);
        float s = 0.0f;
        for (int i = 0; i < mag.size(); ++i)
            s += w.host_f32()[i] * mag.host_f32()[i];
        return s;
    };
    loss();
    Tensor dZ = cpu_zeros(2, 2 * C);
    brotensor::complex_abs_backward(z, w, dZ);
    std::vector<float> g(dZ.host_f32(), dZ.host_f32() + dZ.size());
    fd_check("complex_abs dZ", z.host_f32_mut(), z.size(), g, loss);
}

// rfft_backward: loss = sum(w .* rfft(x)), check grad w.r.t. x.
static void test_rfft_backward(int L) {
    char name[48];
    std::snprintf(name, sizeof(name), "rfft_backward L=%d (fd grad)", L);
    std::printf("%s\n", name);
    const int C = L / 2 + 1;
    Tensor x = cpu_zeros(2, L), w = cpu_zeros(2, 2 * C);
    Rng rng(0x4400 + L);
    for (int i = 0; i < x.size(); ++i) x.host_f32_mut()[i] = rng.next();
    for (int i = 0; i < w.size(); ++i) w.host_f32_mut()[i] = rng.next();
    Tensor y = cpu_zeros(2, 2 * C);
    auto loss = [&]() {
        brotensor::rfft(x, y);
        float s = 0.0f;
        for (int i = 0; i < y.size(); ++i) s += w.host_f32()[i] * y.host_f32()[i];
        return s;
    };
    loss();
    Tensor dX = cpu_zeros(2, L);
    brotensor::rfft_backward(w, L, dX);
    std::vector<float> g(dX.host_f32(), dX.host_f32() + dX.size());
    fd_check(name, x.host_f32_mut(), x.size(), g, loss);
}

// irfft_backward: loss = sum(w .* irfft(x)), check grad w.r.t. x (complex).
static void test_irfft_backward(int L) {
    char name[48];
    std::snprintf(name, sizeof(name), "irfft_backward L=%d (fd grad)", L);
    std::printf("%s\n", name);
    const int C = L / 2 + 1;
    Tensor x = cpu_zeros(2, 2 * C), w = cpu_zeros(2, L);
    Rng rng(0x5500 + L);
    for (int i = 0; i < x.size(); ++i) x.host_f32_mut()[i] = rng.next();
    for (int i = 0; i < w.size(); ++i) w.host_f32_mut()[i] = rng.next();
    Tensor y = cpu_zeros(2, L);
    auto loss = [&]() {
        brotensor::irfft(x, L, y);
        float s = 0.0f;
        for (int i = 0; i < y.size(); ++i) s += w.host_f32()[i] * y.host_f32()[i];
        return s;
    };
    loss();
    Tensor dX = cpu_zeros(2, 2 * C);
    brotensor::irfft_backward(w, dX);
    std::vector<float> g(dX.host_f32(), dX.host_f32() + dX.size());
    fd_check(name, x.host_f32_mut(), x.size(), g, loss);
}

// fft/ifft adjoint path: grad of loss=sum(w.*fft(x)) is ifft(w)*N.
static void test_fft_adjoint(int N) {
    char name[48];
    std::snprintf(name, sizeof(name), "fft adjoint N=%d (fd grad)", N);
    std::printf("%s\n", name);
    Tensor x = cpu_zeros(1, 2 * N), w = cpu_zeros(1, 2 * N);
    Rng rng(0x6600 + N);
    for (int i = 0; i < x.size(); ++i) x.host_f32_mut()[i] = rng.next();
    for (int i = 0; i < w.size(); ++i) w.host_f32_mut()[i] = rng.next();
    Tensor y = cpu_zeros(1, 2 * N);
    auto loss = [&]() {
        brotensor::fft(x, y);
        float s = 0.0f;
        for (int i = 0; i < y.size(); ++i) s += w.host_f32()[i] * y.host_f32()[i];
        return s;
    };
    loss();
    // The loss L = Σ (w.re·y.re + w.im·y.im) for y = fft(x). Treating fft as
    // a complex linear map y = F x, the gradient of this real-valued loss
    // w.r.t. the complex x is the adjoint conj(F)^T applied to w. The DFT
    // matrix is symmetric (F^T = F), so conj(F)^T = conj(F) = N * F^{-1}.
    // Hence dX = N * ifft(w) — the documented "no fft_backward, use ifft +
    // scale by N" gradient path. We verify exactly that recipe here.
    Tensor g = cpu_zeros(1, 2 * N);
    brotensor::ifft(w, g);
    brotensor::scale_inplace(g, static_cast<float>(N));
    std::vector<float> ga(g.host_f32(), g.host_f32() + g.size());
    fd_check(name, x.host_f32_mut(), x.size(), ga, loss);
}

int main() {
    brotensor::init();
    std::printf("test_fft\n");

    // fft / ifft: power-of-2, smooth non-power-of-2 (incl. Whisper 400),
    // and primes (Bluestein).
    for (int N : {1, 2, 4, 8, 16, 6, 9, 12, 15, 400, 5, 7, 53, 401}) {
        test_fft_definition(N);
    }

    // rfft / irfft: even + odd, smooth + prime lengths.
    for (int L : {2, 4, 8, 16, 6, 12, 400, 7, 9, 15, 53, 401}) {
        test_rfft(L);
    }

    test_complex_elementwise();
    test_complex_mul_backward();
    test_complex_abs_backward();

    for (int L : {4, 8, 6, 7, 53}) test_rfft_backward(L);
    for (int L : {4, 8, 6, 7, 53}) test_irfft_backward(L);
    for (int N : {4, 8, 6, 7, 53}) test_fft_adjoint(N);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll FFT / spectral op checks passed.\n");
    return 0;
}
