// ─── CPU-only test for interp2d_forward / interp2d_backward ────────────────
//
// Coverage:
//   1. mode 0 (nearest)  scale=2  must match upsample_nearest_2x exactly.
//   2. mode 1 (bilinear) scale=2  must match upsample_bilinear_2x exactly.
//   3. Identity:  H_out==H_in, W_out==W_in, mode 0/1 reproduces input.
//   4. Arbitrary scale:  1024->64 downsample of a ramp — endpoints + monotonicity.
//   5. Non-uniform scale: H_out != W_out independently rescaled.
//   6. Bicubic:  smooth (linear) input must be reproduced near-exactly
//                (Catmull-Rom interpolates linear functions exactly away
//                from the border).
//   7. Backward (bilinear): finite-difference gradcheck on a small tensor.
//   8. Backward (nearest):  each dY pixel scatters onto exactly one dX pixel,
//                column-sum invariant.
//   9. mode 2 backward throws.
//
// CPU is FP32-only; this test exercises the CPU backend only. The CPU↔GPU
// parity test lives separately (test_interp2d_parity.cpp).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static bool approx(float a, float b, float tol = 1e-5f) {
    const float d = std::fabs(a - b);
    return d <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

static Tensor make_cpu(int rows, int cols) {
    Tensor t;
    t.resize(rows, cols, Dtype::FP32);
    return t;
}

static void fill_random(Tensor& t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float* p = t.host_f32_mut();
    const int n = t.rows * t.cols;
    for (int i = 0; i < n; ++i) p[i] = d(rng);
}

// ── 1. nearest@2x parity with upsample_nearest_2x ──────────────────────────
static void test_nearest_2x_parity() {
    const int N = 2, C = 3, H = 4, W = 5;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0xA1);
    Tensor Y_ref = make_cpu(0, 0);
    brotensor::upsample_nearest_2x(X, N, C, H, W, Y_ref);
    Tensor Y_new = make_cpu(0, 0);
    brotensor::interp2d_forward(X, N, C, H, W, 2 * H, 2 * W, /*nearest*/ 0,
                                Y_new);
    CHECK(Y_new.rows == Y_ref.rows && Y_new.cols == Y_ref.cols);
    const float* a = Y_ref.host_f32();
    const float* b = Y_new.host_f32();
    bool all_eq = true;
    for (int i = 0; i < Y_ref.rows * Y_ref.cols; ++i) {
        if (a[i] != b[i]) { all_eq = false; break; }
    }
    CHECK(all_eq);
}

// ── 2. bilinear@2x parity with upsample_bilinear_2x ────────────────────────
static void test_bilinear_2x_parity() {
    const int N = 2, C = 3, H = 4, W = 5;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0xB2);
    Tensor Y_ref = make_cpu(0, 0);
    brotensor::upsample_bilinear_2x(X, N, C, H, W, Y_ref);
    Tensor Y_new = make_cpu(0, 0);
    brotensor::interp2d_forward(X, N, C, H, W, 2 * H, 2 * W, /*bilinear*/ 1,
                                Y_new);
    CHECK(Y_new.rows == Y_ref.rows && Y_new.cols == Y_ref.cols);
    const float* a = Y_ref.host_f32();
    const float* b = Y_new.host_f32();
    bool ok = true;
    for (int i = 0; i < Y_ref.rows * Y_ref.cols; ++i) {
        if (!approx(a[i], b[i])) { ok = false; break; }
    }
    CHECK(ok);
}

// ── 3. identity (mode 0/1, H_out==H_in) ────────────────────────────────────
static void test_identity() {
    const int N = 1, C = 2, H = 7, W = 9;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0xC3);
    for (int mode : {0, 1}) {
        Tensor Y = make_cpu(0, 0);
        brotensor::interp2d_forward(X, N, C, H, W, H, W, mode, Y);
        const float* a = X.host_f32();
        const float* b = Y.host_f32();
        bool ok = true;
        for (int i = 0; i < N * C * H * W; ++i) {
            if (!approx(a[i], b[i], 1e-6f)) { ok = false; break; }
        }
        CHECK(ok);
    }
}

// ── 4. arbitrary-scale downsample of a ramp ────────────────────────────────
static void test_arbitrary_downsample_ramp() {
    // 1x1x1x32 ramp -> 1x1x1x4 bilinear. Output should be monotonically
    // increasing with first sample close to ramp start, last close to end.
    const int N = 1, C = 1, H = 1, W = 32, W_out = 4;
    Tensor X = make_cpu(N, C * H * W);
    float* p = X.host_f32_mut();
    for (int i = 0; i < W; ++i) p[i] = static_cast<float>(i);
    Tensor Y = make_cpu(0, 0);
    brotensor::interp2d_forward(X, N, C, H, W, H, W_out, /*bilinear*/ 1, Y);
    const float* y = Y.host_f32();
    CHECK(Y.rows == 1 && Y.cols == W_out);
    for (int i = 0; i + 1 < W_out; ++i) CHECK(y[i] < y[i + 1]);
    // bilinear half-pixel: first sample at src_x = (0+0.5)*8 - 0.5 = 3.5
    // -> y[0] = 3.5; last at (3+0.5)*8 - 0.5 = 27.5 -> y[3] = 27.5
    CHECK(approx(y[0], 3.5f, 1e-5f));
    CHECK(approx(y[W_out - 1], 27.5f, 1e-5f));
}

// ── 5. non-uniform scale (H_out != W_out independently) ────────────────────
static void test_non_uniform_scale() {
    const int N = 1, C = 1, H = 6, W = 10, H_out = 12, W_out = 5;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0xD4);
    Tensor Y = make_cpu(0, 0);
    brotensor::interp2d_forward(X, N, C, H, W, H_out, W_out,
                                /*bilinear*/ 1, Y);
    CHECK(Y.rows == N && Y.cols == C * H_out * W_out);
    // Just sanity: not all zero (random input shouldn't yield zero output).
    const float* y = Y.host_f32();
    bool any_nonzero = false;
    for (int i = 0; i < Y.cols; ++i)
        if (y[i] != 0.0f) { any_nonzero = true; break; }
    CHECK(any_nonzero);
}

// ── 6. bicubic reproduces linear ramp exactly (away from borders) ──────────
static void test_bicubic_linear_ramp() {
    // f(x) = 2x + 3 is linear; Keys/Catmull-Rom (a=-0.5) reproduces linear
    // functions exactly in the interior. Borders use clamping so we skip
    // the outermost output pixels.
    const int N = 1, C = 1, H = 1, W = 8, W_out = 16;
    Tensor X = make_cpu(N, C * H * W);
    float* p = X.host_f32_mut();
    for (int i = 0; i < W; ++i) p[i] = 2.0f * i + 3.0f;
    Tensor Y = make_cpu(0, 0);
    brotensor::interp2d_forward(X, N, C, H, W, H, W_out, /*bicubic*/ 2, Y);
    const float* y = Y.host_f32();
    // src_x = (ow + 0.5)*0.5 - 0.5 = 0.5*ow - 0.25
    // expected = 2*src_x + 3 = ow - 0.5 + 3 = ow + 2.5.
    // Catmull-Rom needs all 4 taps inside [0, W-1] to reproduce linear data
    // exactly — at the borders, clamped taps double-count an edge sample.
    // For W=8, ow=2 reaches src_x=0.75 (x0=0, tap -1 clamped) and ow=13
    // reaches src_x=6.25 (tap 8 clamped). Safe interior: ow in [3, 12].
    for (int ow = 3; ow + 3 < W_out; ++ow) {
        const float expected = static_cast<float>(ow) + 2.5f;
        CHECK(approx(y[ow], expected, 1e-4f));
    }
}

// ── 7. bilinear backward: finite-difference gradcheck ──────────────────────
static void test_bilinear_backward_gradcheck() {
    const int N = 1, C = 1, H = 3, W = 4, H_out = 5, W_out = 6;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0xE5);

    // Loss L = sum(Y * V) for a fixed random V, so dL/dY = V.
    Tensor V = make_cpu(N, C * H_out * W_out);
    fill_random(V, 0xF6);

    Tensor dX = make_cpu(0, 0);
    brotensor::interp2d_backward(V, N, C, H, W, H_out, W_out,
                                 /*bilinear*/ 1, dX);

    // Numerical gradient: for each input pixel, perturb +/-h, sum(Y*V).
    const float h = 1e-3f;
    const float* dXp = dX.host_f32();
    float* Xp = X.host_f32_mut();
    Tensor Yt = make_cpu(0, 0);
    bool ok = true;
    int max_checks = 20;
    std::mt19937_64 rng(0x77);
    std::uniform_int_distribution<int> pick(0, N * C * H * W - 1);
    for (int k = 0; k < max_checks; ++k) {
        const int i = pick(rng);
        const float orig = Xp[i];
        Xp[i] = orig + h;
        brotensor::interp2d_forward(X, N, C, H, W, H_out, W_out, 1, Yt);
        double Lp = 0.0;
        const float* y = Yt.host_f32();
        const float* v = V.host_f32();
        for (int j = 0; j < Yt.cols; ++j) Lp += static_cast<double>(y[j]) * v[j];
        Xp[i] = orig - h;
        brotensor::interp2d_forward(X, N, C, H, W, H_out, W_out, 1, Yt);
        double Lm = 0.0;
        y = Yt.host_f32();
        for (int j = 0; j < Yt.cols; ++j) Lm += static_cast<double>(y[j]) * v[j];
        Xp[i] = orig;
        const float num = static_cast<float>((Lp - Lm) / (2.0 * h));
        if (!approx(num, dXp[i], 1e-3f)) {
            std::printf("  gradcheck mismatch at i=%d: num=%g ana=%g\n",
                        i, num, dXp[i]);
            ok = false;
        }
    }
    CHECK(ok);
}

// ── 8. nearest backward: each dY scatters to exactly one dX ────────────────
static void test_nearest_backward_count() {
    const int N = 1, C = 1, H = 4, W = 4, H_out = 8, W_out = 8;
    Tensor dY = make_cpu(N, C * H_out * W_out);
    float* dy = dY.host_f32_mut();
    for (int i = 0; i < dY.cols; ++i) dy[i] = 1.0f;
    Tensor dX = make_cpu(0, 0);
    brotensor::interp2d_backward(dY, N, C, H, W, H_out, W_out,
                                 /*nearest*/ 0, dX);
    // Sum of dX must equal sum of dY (every dY pixel deposited exactly once).
    double sum_dx = 0.0;
    const float* dxp = dX.host_f32();
    for (int i = 0; i < dX.cols; ++i) sum_dx += dxp[i];
    CHECK(approx(static_cast<float>(sum_dx),
                 static_cast<float>(H_out * W_out), 1e-5f));
}

// ── 9. bicubic backward throws ─────────────────────────────────────────────
static void test_bicubic_backward_throws() {
    Tensor dY = make_cpu(1, 1 * 2 * 2);
    Tensor dX = make_cpu(0, 0);
    bool threw = false;
    try {
        brotensor::interp2d_backward(dY, 1, 1, 1, 1, 2, 2,
                                     /*bicubic*/ 2, dX);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    brotensor::init();
    std::printf("test_interp2d (CPU FP32):\n");
    test_nearest_2x_parity();
    test_bilinear_2x_parity();
    test_identity();
    test_arbitrary_downsample_ramp();
    test_non_uniform_scale();
    test_bicubic_linear_ramp();
    test_bilinear_backward_gradcheck();
    test_nearest_backward_count();
    test_bicubic_backward_throws();
    if (g_failures == 0) {
        std::printf("  OK  all interp2d CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
