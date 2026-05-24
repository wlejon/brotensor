// ─── CPU-only test for pad2d_forward / pad2d_backward ──────────────────────
//
// Coverage:
//   1. Zero pad reproduces inputs at the interior and zeros elsewhere.
//   2. Replicate pad clamps to edge samples (corners, edges, axes).
//   3. Reflect pad mirrors without repeating the edge sample (matches numpy).
//   4. Reflect rejects pad >= H or pad >= W.
//   5. Zero-pad backward: gradient sum on interior == output corner sum;
//      pad slots contribute zero to dX.
//   6. Replicate backward: gradient accumulates onto the edge sample for
//      every padding row/column that copied it.
//   7. Reflect backward gradcheck (finite-difference on a small tensor).
//   8. Asymmetric pad (pt != pb, pl != pr) shape + corner sanity.

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

// ── 1. zero pad: interior reproduced, surround zero ───────────────────────
static void test_zero_interior_and_zeros() {
    const int N = 1, C = 1, H = 3, W = 4;
    const int pt = 1, pb = 2, pl = 1, pr = 2;
    const int Hp = H + pt + pb, Wp = W + pl + pr;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0x11);
    Tensor Y = make_cpu(0, 0);
    brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr, /*zero*/ 0, Y);
    CHECK(Y.rows == N && Y.cols == C * Hp * Wp);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    // Interior block
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            CHECK(approx(y[(pt + h) * Wp + (pl + w)], x[h * W + w]));
        }
    }
    // Top band zero
    for (int h = 0; h < pt; ++h)
        for (int w = 0; w < Wp; ++w)
            CHECK(y[h * Wp + w] == 0.0f);
    // Bottom band zero
    for (int h = pt + H; h < Hp; ++h)
        for (int w = 0; w < Wp; ++w)
            CHECK(y[h * Wp + w] == 0.0f);
    // Left + right bands zero on interior rows
    for (int h = pt; h < pt + H; ++h) {
        for (int w = 0; w < pl; ++w) CHECK(y[h * Wp + w] == 0.0f);
        for (int w = pl + W; w < Wp; ++w) CHECK(y[h * Wp + w] == 0.0f);
    }
}

// ── 2. replicate pad: edge samples clamp out ──────────────────────────────
static void test_replicate_clamps() {
    const int N = 1, C = 1, H = 2, W = 3;
    const int pt = 2, pb = 1, pl = 1, pr = 2;
    const int Wp = W + pl + pr;
    Tensor X = make_cpu(N, C * H * W);
    float* xp = X.host_f32_mut();
    // Distinct values per pixel for unambiguous checks.
    // X = [[10, 20, 30],
    //      [40, 50, 60]]
    xp[0] = 10; xp[1] = 20; xp[2] = 30;
    xp[3] = 40; xp[4] = 50; xp[5] = 60;
    Tensor Y = make_cpu(0, 0);
    brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr,
                             /*replicate*/ 2, Y);
    const float* y = Y.host_f32();
    // Top-left corner copies X[0,0] = 10.
    CHECK(y[0 * Wp + 0] == 10.0f);
    CHECK(y[1 * Wp + 0] == 10.0f);          // pt-1 row
    CHECK(y[0 * Wp + (pl + W)] == 30.0f);    // top, right-padding sample
    CHECK(y[(pt + H) * Wp + (pl + W + 1)] == 60.0f);  // bottom-right corner
    // Interior preserved.
    CHECK(y[pt * Wp + pl + 1] == 20.0f);
}

// ── 3. reflect pad: mirror without repeating the edge ──────────────────────
static void test_reflect_mirrors() {
    // 1-row case (no H reflection needed): X = [1, 2, 3, 4], pl = pr = 2.
    // Reflect output should be [3, 2, | 1, 2, 3, 4, | 3, 2].
    const int N = 1, C = 1, H = 1, W = 4;
    const int pt = 0, pb = 0, pl = 2, pr = 2;
    const int Wp = W + pl + pr;
    Tensor X = make_cpu(N, C * H * W);
    float* xp = X.host_f32_mut();
    xp[0] = 1; xp[1] = 2; xp[2] = 3; xp[3] = 4;
    Tensor Y = make_cpu(0, 0);
    brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr,
                             /*reflect*/ 1, Y);
    const float* y = Y.host_f32();
    const float expected[] = {3, 2, 1, 2, 3, 4, 3, 2};
    for (int w = 0; w < Wp; ++w) CHECK(y[w] == expected[w]);
}

// ── 4. reflect rejects pad >= H or pad >= W ────────────────────────────────
static void test_reflect_bounds_throw() {
    Tensor X = make_cpu(1, 1 * 2 * 3);
    Tensor Y = make_cpu(0, 0);
    bool threw = false;
    try { brotensor::pad2d_forward(X, 1, 1, 2, 3, 0, 0, 3, 0, 1, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { brotensor::pad2d_forward(X, 1, 1, 2, 3, 2, 0, 0, 0, 1, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 5. zero pad backward: only interior contributes ───────────────────────
static void test_zero_backward() {
    const int N = 1, C = 1, H = 2, W = 3;
    const int pt = 1, pb = 1, pl = 2, pr = 1;
    const int Hp = H + pt + pb, Wp = W + pl + pr;
    Tensor dY = make_cpu(N, C * Hp * Wp);
    float* dy = dY.host_f32_mut();
    for (int i = 0; i < dY.cols; ++i) dy[i] = static_cast<float>(i + 1);
    Tensor dX = make_cpu(0, 0);
    brotensor::pad2d_backward(dY, N, C, H, W, pt, pb, pl, pr,
                              /*zero*/ 0, dX);
    const float* dx = dX.host_f32();
    // dX[h, w] = dY[pt + h, pl + w] for zero padding.
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            CHECK(dx[h * W + w] == dy[(pt + h) * Wp + (pl + w)]);
}

// ── 6. replicate backward: edge samples accumulate the pad rows/cols ──────
static void test_replicate_backward_accumulates_edges() {
    // 1x1x1x3 with pl=2, pr=2 replicate. dY = [1,1,1,1,1,1,1] (length 7).
    // Each pad slot on the left clamps to X[0]; each pad on the right
    // clamps to X[2]. So dX = [1 + pl, 1, 1 + pr] = [3, 1, 3].
    const int N = 1, C = 1, H = 1, W = 3;
    const int pt = 0, pb = 0, pl = 2, pr = 2;
    const int Wp = W + pl + pr;
    Tensor dY = make_cpu(N, C * 1 * Wp);
    float* dy = dY.host_f32_mut();
    for (int i = 0; i < dY.cols; ++i) dy[i] = 1.0f;
    Tensor dX = make_cpu(0, 0);
    brotensor::pad2d_backward(dY, N, C, H, W, pt, pb, pl, pr,
                              /*replicate*/ 2, dX);
    const float* dx = dX.host_f32();
    CHECK(approx(dx[0], 3.0f));
    CHECK(approx(dx[1], 1.0f));
    CHECK(approx(dx[2], 3.0f));
}

// ── 7. reflect backward: finite-difference gradcheck ──────────────────────
static void test_reflect_backward_gradcheck() {
    const int N = 1, C = 1, H = 3, W = 4;
    const int pt = 1, pb = 2, pl = 2, pr = 1;
    const int Hp = H + pt + pb, Wp = W + pl + pr;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0x99);
    // Loss L = sum(Y * V) with random V -> dL/dY = V.
    Tensor V = make_cpu(N, C * Hp * Wp);
    fill_random(V, 0xAA);

    Tensor dX = make_cpu(0, 0);
    brotensor::pad2d_backward(V, N, C, H, W, pt, pb, pl, pr, 1, dX);

    const float h = 1e-3f;
    const float* dxp = dX.host_f32();
    float* xp = X.host_f32_mut();
    Tensor Yt = make_cpu(0, 0);
    bool ok = true;
    std::mt19937_64 rng(0xBB);
    std::uniform_int_distribution<int> pick(0, N * C * H * W - 1);
    for (int k = 0; k < 20; ++k) {
        const int i = pick(rng);
        const float orig = xp[i];
        xp[i] = orig + h;
        brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr, 1, Yt);
        double Lp = 0.0;
        const float* y = Yt.host_f32();
        const float* v = V.host_f32();
        for (int j = 0; j < Yt.cols; ++j) Lp += static_cast<double>(y[j]) * v[j];
        xp[i] = orig - h;
        brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr, 1, Yt);
        double Lm = 0.0;
        y = Yt.host_f32();
        for (int j = 0; j < Yt.cols; ++j) Lm += static_cast<double>(y[j]) * v[j];
        xp[i] = orig;
        const float num = static_cast<float>((Lp - Lm) / (2.0 * h));
        if (!approx(num, dxp[i], 1e-3f)) {
            std::printf("  gradcheck mismatch at i=%d: num=%g ana=%g\n",
                        i, num, dxp[i]);
            ok = false;
        }
    }
    CHECK(ok);
}

// ── 8. asymmetric pad shape + corner sanity ────────────────────────────────
static void test_asymmetric_shape() {
    const int N = 2, C = 3, H = 5, W = 7;
    const int pt = 1, pb = 4, pl = 0, pr = 2;
    const int Hp = H + pt + pb, Wp = W + pl + pr;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0xCC);
    Tensor Y = make_cpu(0, 0);
    brotensor::pad2d_forward(X, N, C, H, W, pt, pb, pl, pr, 0, Y);
    CHECK(Y.rows == N && Y.cols == C * Hp * Wp);
    // Spot-check that (n=1, c=2, h=0+pt, w=0+pl) maps to X[1,2,0,0].
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    const long y_base = (1L * C + 2) * Hp * Wp;
    const long x_base = (1L * C + 2) * H * W;
    CHECK(y[y_base + (pt + 0) * Wp + (pl + 0)] == x[x_base + 0 * W + 0]);
    CHECK(y[y_base + (pt + H - 1) * Wp + (pl + W - 1)]
          == x[x_base + (H - 1) * W + (W - 1)]);
}

int main() {
    brotensor::init();
    std::printf("test_pad2d (CPU FP32):\n");
    test_zero_interior_and_zeros();
    test_replicate_clamps();
    test_reflect_mirrors();
    test_reflect_bounds_throw();
    test_zero_backward();
    test_replicate_backward_accumulates_edges();
    test_reflect_backward_gradcheck();
    test_asymmetric_shape();
    if (g_failures == 0) {
        std::printf("  OK  all pad2d CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
