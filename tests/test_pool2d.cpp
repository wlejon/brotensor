// ─── CPU-only test for adaptive_avg_pool2d + max_pool2d ────────────────────
//
// Coverage:
//   adaptive_avg_pool2d:
//     A1. Identity: H_out == H, W_out == W reproduces X (every region is one
//         pixel of area 1).
//     A2. Global pool: H_out == W_out == 1 produces the per-channel mean.
//     A3. Backward gradcheck on a small non-divisible case (regions overlap
//         when H_out doesn't divide H).
//     A4. Backward preserves total mass: sum(dX) == sum(dY).
//   max_pool2d:
//     M1. Forward picks the max in a known 2x2/stride2 kernel.
//     M2. Idx round-trips: Y == X[..., Idx_h, Idx_w].
//     M3. Padded edges: pad pixels never win (we set non-padded entries
//         strictly greater).
//     M4. Backward scatters dY onto the selected positions; non-selected
//         positions get zero.
//     M5. Overlapping kernels (stride < kernel) — backward sums.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
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

static bool approx(float a, float b, float tol = 1e-5f) {
    const float d = std::fabs(a - b);
    return d <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

static Tensor make_f32(int r, int c) {
    Tensor t;
    t.resize(r, c, Dtype::FP32);
    return t;
}

static void fill_random(Tensor& t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng);
}

// ── A1. adaptive identity ──────────────────────────────────────────────────
static void test_adaptive_identity() {
    const int N = 1, C = 2, H = 4, W = 5;
    Tensor X = make_f32(N, C * H * W);
    fill_random(X, 0x10);
    Tensor Y;
    brotensor::adaptive_avg_pool2d_forward(X, N, C, H, W, H, W, Y);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int i = 0; i < X.cols; ++i) CHECK(approx(x[i], y[i]));
}

// ── A2. global pool == per-channel mean ────────────────────────────────────
static void test_adaptive_global_mean() {
    const int N = 2, C = 3, H = 4, W = 5;
    Tensor X = make_f32(N, C * H * W);
    fill_random(X, 0x11);
    Tensor Y;
    brotensor::adaptive_avg_pool2d_forward(X, N, C, H, W, 1, 1, Y);
    CHECK(Y.rows == N && Y.cols == C);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            double s = 0.0;
            const float* chan = x + (n * C + c) * H * W;
            for (int i = 0; i < H * W; ++i) s += chan[i];
            const float mean = static_cast<float>(s / (H * W));
            CHECK(approx(y[n * C + c], mean));
        }
}

// ── A3. adaptive backward gradcheck (non-divisible) ───────────────────────
static void test_adaptive_backward_gradcheck() {
    const int N = 1, C = 1, H = 5, W = 5, H_out = 2, W_out = 3;
    Tensor X = make_f32(N, C * H * W);
    fill_random(X, 0x12);
    Tensor V = make_f32(N, C * H_out * W_out);
    fill_random(V, 0x13);
    Tensor dX;
    brotensor::adaptive_avg_pool2d_backward(V, N, C, H, W, H_out, W_out, dX);
    const float* dxp = dX.host_f32();
    float* xp = X.host_f32_mut();
    Tensor Yt;
    const float h = 1e-3f;
    bool ok = true;
    std::mt19937_64 rng(0x14);
    std::uniform_int_distribution<int> pick(0, H * W - 1);
    for (int t = 0; t < 15; ++t) {
        const int i = pick(rng);
        const float orig = xp[i];
        xp[i] = orig + h;
        brotensor::adaptive_avg_pool2d_forward(X, N, C, H, W, H_out, W_out, Yt);
        double Lp = 0.0;
        const float* y = Yt.host_f32();
        const float* v = V.host_f32();
        for (int j = 0; j < Yt.cols; ++j) Lp += static_cast<double>(y[j]) * v[j];
        xp[i] = orig - h;
        brotensor::adaptive_avg_pool2d_forward(X, N, C, H, W, H_out, W_out, Yt);
        double Lm = 0.0;
        y = Yt.host_f32();
        for (int j = 0; j < Yt.cols; ++j) Lm += static_cast<double>(y[j]) * v[j];
        xp[i] = orig;
        const float num = static_cast<float>((Lp - Lm) / (2.0 * h));
        if (!approx(num, dxp[i], 5e-3f)) {
            std::printf("  adaptive gradcheck mismatch at i=%d: num=%g ana=%g\n",
                        i, num, dxp[i]);
            ok = false;
        }
    }
    CHECK(ok);
}

// ── A4. adaptive backward preserves total mass ────────────────────────────
static void test_adaptive_backward_mass() {
    const int N = 1, C = 1, H = 4, W = 6, H_out = 2, W_out = 3;
    Tensor dY = make_f32(N, C * H_out * W_out);
    float* dy = dY.host_f32_mut();
    for (int i = 0; i < dY.cols; ++i) dy[i] = static_cast<float>(i + 1);
    Tensor dX;
    brotensor::adaptive_avg_pool2d_backward(dY, N, C, H, W, H_out, W_out, dX);
    double s_dx = 0.0, s_dy = 0.0;
    const float* dxp = dX.host_f32();
    for (int i = 0; i < dX.cols; ++i) s_dx += dxp[i];
    for (int i = 0; i < dY.cols; ++i) s_dy += dy[i];
    CHECK(approx(static_cast<float>(s_dx), static_cast<float>(s_dy), 1e-5f));
}

// ── M1+M2. max_pool forward + index round-trip ────────────────────────────
static void test_maxpool_basic_and_idx() {
    const int N = 1, C = 1, H = 4, W = 4;
    Tensor X = make_f32(N, C * H * W);
    // Distinct values so max is unambiguous.
    float* xp = X.host_f32_mut();
    for (int i = 0; i < X.cols; ++i) xp[i] = static_cast<float>(i + 1);
    Tensor Y, Idx;
    // 2x2 / stride 2, no pad -> (2,2).
    brotensor::max_pool2d_forward(X, N, C, H, W, 2, 2, 2, 2, 0, 0, Y, Idx);
    CHECK(Y.rows == N && Y.cols == 4);
    const float* yp = Y.host_f32();
    const int32_t* ip = static_cast<const int32_t*>(Idx.data);
    // Each 2x2 corner's max is the bottom-right (largest value).
    // Block (0,0): {1,2,5,6} -> max 6 at (1,1) flat = 5.
    // Block (0,1): {3,4,7,8} -> max 8 at (1,3) flat = 7.
    // Block (1,0): {9,10,13,14} -> max 14 at (3,1) flat = 13.
    // Block (1,1): {11,12,15,16} -> max 16 at (3,3) flat = 15.
    CHECK(yp[0] == 6.0f  && ip[0] == 5);
    CHECK(yp[1] == 8.0f  && ip[1] == 7);
    CHECK(yp[2] == 14.0f && ip[2] == 13);
    CHECK(yp[3] == 16.0f && ip[3] == 15);
    for (int i = 0; i < 4; ++i) CHECK(yp[i] == xp[ip[i]]);
}

// ── M3. padded edges: pad pixels lose ─────────────────────────────────────
static void test_maxpool_padded() {
    const int N = 1, C = 1, H = 2, W = 2;
    Tensor X = make_f32(N, C * H * W);
    float* xp = X.host_f32_mut();
    xp[0] = 0.1f; xp[1] = 0.2f; xp[2] = 0.3f; xp[3] = 0.4f;
    Tensor Y, Idx;
    // 2x2 / stride 1 / pad 1 -> output (3, 3).
    brotensor::max_pool2d_forward(X, N, C, H, W, 2, 2, 1, 1, 1, 1, Y, Idx);
    CHECK(Y.rows == 1 && Y.cols == 9);
    const float* yp = Y.host_f32();
    const int32_t* ip = static_cast<const int32_t*>(Idx.data);
    // No output should ever pick a "padding" value (we set ours to -inf
    // implicitly). All idx must be in [0, H*W).
    for (int i = 0; i < 9; ++i) {
        CHECK(ip[i] >= 0 && ip[i] < H * W);
        CHECK(yp[i] == xp[ip[i]]);
    }
}

// ── M4. backward: dY scattered onto selected positions, others zero ───────
static void test_maxpool_backward_scatter() {
    const int N = 1, C = 1, H = 4, W = 4;
    Tensor X = make_f32(N, C * H * W);
    float* xp = X.host_f32_mut();
    for (int i = 0; i < X.cols; ++i) xp[i] = static_cast<float>(i + 1);
    Tensor Y, Idx;
    brotensor::max_pool2d_forward(X, N, C, H, W, 2, 2, 2, 2, 0, 0, Y, Idx);
    Tensor dY = make_f32(N, Y.cols);
    float* dyp = dY.host_f32_mut();
    dyp[0] = 1.0f; dyp[1] = 2.0f; dyp[2] = 3.0f; dyp[3] = 4.0f;
    Tensor dX;
    brotensor::max_pool2d_backward(dY, Idx, N, C, H, W, 2, 2, dX);
    const float* dxp = dX.host_f32();
    const int32_t* ip = static_cast<const int32_t*>(Idx.data);
    // Build expected dX: zero everywhere except dxp[ip[k]] += dyp[k].
    std::vector<float> exp(H * W, 0.0f);
    for (int k = 0; k < 4; ++k) exp[ip[k]] += dyp[k];
    for (int i = 0; i < H * W; ++i) CHECK(approx(dxp[i], exp[i]));
}

// ── M5. overlapping kernels: backward sums ────────────────────────────────
static void test_maxpool_backward_overlap_sums() {
    // 1x1x3x3, kernel 2x2 stride 1 -> output 2x2. Center pixel X[1,1] is in
    // all 4 output kernels; if it's the max of every kernel, dX[1,1] sums
    // all four dY values.
    const int N = 1, C = 1, H = 3, W = 3;
    Tensor X = make_f32(N, C * H * W);
    float* xp = X.host_f32_mut();
    for (int i = 0; i < 9; ++i) xp[i] = 0.0f;
    xp[1 * W + 1] = 100.0f;  // center dominates every kernel
    Tensor Y, Idx;
    brotensor::max_pool2d_forward(X, N, C, H, W, 2, 2, 1, 1, 0, 0, Y, Idx);
    const int32_t* ip = static_cast<const int32_t*>(Idx.data);
    CHECK(Y.cols == 4);
    for (int k = 0; k < 4; ++k) CHECK(ip[k] == 1 * W + 1);  // center
    Tensor dY = make_f32(N, 4);
    float* dyp = dY.host_f32_mut();
    dyp[0] = 1.0f; dyp[1] = 2.0f; dyp[2] = 3.0f; dyp[3] = 4.0f;
    Tensor dX;
    brotensor::max_pool2d_backward(dY, Idx, N, C, H, W, 2, 2, dX);
    const float* dxp = dX.host_f32();
    CHECK(approx(dxp[1 * W + 1], 10.0f));  // 1+2+3+4
    for (int i = 0; i < 9; ++i)
        if (i != 1 * W + 1) CHECK(dxp[i] == 0.0f);
}

int main() {
    brotensor::init();
    std::printf("test_pool2d (CPU FP32):\n");
    test_adaptive_identity();
    test_adaptive_global_mean();
    test_adaptive_backward_gradcheck();
    test_adaptive_backward_mass();
    test_maxpool_basic_and_idx();
    test_maxpool_padded();
    test_maxpool_backward_scatter();
    test_maxpool_backward_overlap_sums();
    if (g_failures == 0) {
        std::printf("  OK  all pool2d CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
