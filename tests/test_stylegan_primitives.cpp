// CPU FP32 tests for the StyleGAN3 synthesis-input primitives:
//   sin / cos / rsqrt  (forward vs std::, backward vs finite-difference)
//   pixel_norm         (forward vs manual RMS-over-channel, backward vs FD)

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

static Tensor from_vec(const std::vector<float>& v, int r, int c) {
    return Tensor::from_host_on(brotensor::Device::CPU, v.data(), r, c);
}

// ─── sin / cos / rsqrt forward ──────────────────────────────────────────────

static void test_trig_rsqrt_forward() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    const int R = 4, C = 5;
    std::vector<float> x(R * C), xpos(R * C);
    for (auto& v : x)    v = d(rng);
    for (auto& v : xpos) v = std::uniform_real_distribution<float>(0.3f, 4.0f)(rng);

    Tensor X = from_vec(x, R, C), Y;
    brotensor::sin_forward(X, Y);
    for (int i = 0; i < R * C; ++i) CHECK(close(Y[i], std::sin(x[i])));
    brotensor::cos_forward(X, Y);
    for (int i = 0; i < R * C; ++i) CHECK(close(Y[i], std::cos(x[i])));

    Tensor Xp = from_vec(xpos, R, C);
    brotensor::rsqrt_forward(Xp, Y);
    for (int i = 0; i < R * C; ++i) CHECK(close(Y[i], 1.0f / std::sqrt(xpos[i])));
}

// Generic finite-difference check for an elementwise op whose backward reads a
// "primal" tensor p (= x for sin/cos, = y for rsqrt). f maps x->y.
template <class Fwd, class Bwd>
static void fd_elementwise(Fwd fwd, Bwd bwd, bool primal_is_output,
                           float lo, float hi) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(lo, hi);
    const int R = 3, C = 4, n = R * C;
    std::vector<float> x(n);
    for (auto& v : x) v = d(rng);
    Tensor X = from_vec(x, R, C), Y;
    fwd(X, Y);

    // Upstream grad = ones → dX should equal dy/dx.
    std::vector<float> ones(n, 1.0f);
    Tensor dY = from_vec(ones, R, C), dX;
    const Tensor& primal = primal_is_output ? Y : X;
    bwd(primal, dY, dX);

    const float h = 1e-3f;
    for (int i = 0; i < n; ++i) {
        std::vector<float> xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        Tensor Yp, Ym, Xp = from_vec(xp, R, C), Xm = from_vec(xm, R, C);
        fwd(Xp, Yp); fwd(Xm, Ym);
        const float fd = (Yp[i] - Ym[i]) / (2 * h);
        CHECK(close(dX[i], fd, 2e-2f));
    }
}

static void test_trig_rsqrt_backward() {
    fd_elementwise(brotensor::sin_forward, brotensor::sin_backward,
                   /*primal_is_output=*/false, -3.0f, 3.0f);
    fd_elementwise(brotensor::cos_forward, brotensor::cos_backward,
                   /*primal_is_output=*/false, -3.0f, 3.0f);
    fd_elementwise(brotensor::rsqrt_forward, brotensor::rsqrt_backward,
                   /*primal_is_output=*/true, 0.4f, 4.0f);
}

// ─── pixel_norm ─────────────────────────────────────────────────────────────

static void test_pixel_norm() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    const int R = 4, C = 6;
    const float eps = 1e-8f;
    std::vector<float> x(R * C);
    for (auto& v : x) v = d(rng);

    Tensor X = from_vec(x, R, C), Y;
    brotensor::pixel_norm_forward(X, eps, Y);
    for (int r = 0; r < R; ++r) {
        float ss = 0.0f;
        for (int c = 0; c < C; ++c) ss += x[r * C + c] * x[r * C + c];
        const float rinv = 1.0f / std::sqrt(ss / C + eps);
        for (int c = 0; c < C; ++c)
            CHECK(close(Y[r * C + c], x[r * C + c] * rinv));
    }

    // Backward vs finite difference of a scalar loss L = sum(w_i * y_i) for
    // random weights w (so dY = w).
    std::vector<float> w(R * C);
    for (auto& v : w) v = d(rng);
    Tensor dY = from_vec(w, R, C), dX;
    brotensor::pixel_norm_backward(X, dY, eps, dX);

    const float h = 1e-3f;
    for (int i = 0; i < R * C; ++i) {
        std::vector<float> xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        Tensor Yp, Ym, Xp = from_vec(xp, R, C), Xm = from_vec(xm, R, C);
        brotensor::pixel_norm_forward(Xp, eps, Yp);
        brotensor::pixel_norm_forward(Xm, eps, Ym);
        float lp = 0.0f, lm = 0.0f;
        for (int k = 0; k < R * C; ++k) { lp += w[k] * Yp[k]; lm += w[k] * Ym[k]; }
        const float fd = (lp - lm) / (2 * h);
        CHECK(close(dX[i], fd, 2e-2f));
    }
}

int main() {
    test_trig_rsqrt_forward();
    test_trig_rsqrt_backward();
    test_pixel_norm();
    if (g_failures) {
        std::printf("stylegan_primitives: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("stylegan_primitives: all passed\n");
    return 0;
}
