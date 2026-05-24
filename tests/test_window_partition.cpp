// ─── CPU-only test for window_partition / window_reverse ──────────────────
//
// Coverage:
//   1. Output shape: partition(N, C, H, W, window) → (N*nw_h*nw_w, C*w*w).
//   2. Single window (window == H == W) is a row-flatten only — no permute.
//   3. Round-trip: reverse(partition(X)) == X for a random NCHW tensor.
//   4. Inverse direction: partition(reverse(Y)) == Y for a random windowed
//      tensor.
//   5. Partition correctly places known sentinels at the right rows + offsets.
//   6. window not dividing H or W throws.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

using brotensor::Tensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

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

// ── 1. shape ──────────────────────────────────────────────────────────────
static void test_shape() {
    const int N = 2, C = 3, H = 8, W = 12, window = 4;
    Tensor X = make_f32(N, C * H * W);
    fill_random(X, 0x11);
    Tensor Y;
    brotensor::window_partition_forward(X, N, C, H, W, window, Y);
    const int nw_h = H / window, nw_w = W / window;
    CHECK(Y.rows == N * nw_h * nw_w);
    CHECK(Y.cols == C * window * window);
}

// ── 2. window == H == W: just a row-flatten of each (C, H, W) image ──────
static void test_single_window() {
    const int N = 2, C = 3, H = 4, W = 4, window = 4;
    Tensor X = make_f32(N, C * H * W);
    fill_random(X, 0x22);
    Tensor Y;
    brotensor::window_partition_forward(X, N, C, H, W, window, Y);
    CHECK(Y.rows == N && Y.cols == X.cols);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    // With one window per image, partition is the identity.
    for (int i = 0; i < X.rows * X.cols; ++i) CHECK(y[i] == x[i]);
}

// ── 3. round-trip reverse(partition(X)) == X ──────────────────────────────
static void test_round_trip_partition_then_reverse() {
    const int N = 2, C = 3, H = 8, W = 6, window = 2;
    Tensor X = make_f32(N, C * H * W);
    fill_random(X, 0x33);
    Tensor Win;
    brotensor::window_partition_forward(X, N, C, H, W, window, Win);
    Tensor Xback;
    brotensor::window_reverse_forward(Win, N, C, H, W, window, Xback);
    CHECK(Xback.rows == X.rows && Xback.cols == X.cols);
    const float* a = X.host_f32();
    const float* b = Xback.host_f32();
    for (int i = 0; i < X.rows * X.cols; ++i) CHECK(a[i] == b[i]);
}

// ── 4. inverse direction partition(reverse(Y)) == Y ───────────────────────
static void test_round_trip_reverse_then_partition() {
    const int N = 1, C = 2, H = 6, W = 6, window = 3;
    const int nw_h = H / window, nw_w = W / window;
    Tensor Y = make_f32(N * nw_h * nw_w, C * window * window);
    fill_random(Y, 0x44);
    Tensor Img;
    brotensor::window_reverse_forward(Y, N, C, H, W, window, Img);
    Tensor Yback;
    brotensor::window_partition_forward(Img, N, C, H, W, window, Yback);
    CHECK(Yback.rows == Y.rows && Yback.cols == Y.cols);
    const float* a = Y.host_f32();
    const float* b = Yback.host_f32();
    for (int i = 0; i < Y.rows * Y.cols; ++i) CHECK(a[i] == b[i]);
}

// ── 5. sentinel placement check ───────────────────────────────────────────
static void test_sentinel_placement() {
    // 1x1x4x4 with window 2 -> 4 windows of 2x2.
    // Top-left window contains X[0..1, 0..1] = {1, 2, 5, 6} (row-major).
    // Bottom-right window contains X[2..3, 2..3] = {11, 12, 15, 16}.
    const int N = 1, C = 1, H = 4, W = 4, window = 2;
    Tensor X = make_f32(N, C * H * W);
    float* xp = X.host_f32_mut();
    for (int i = 0; i < 16; ++i) xp[i] = static_cast<float>(i + 1);
    Tensor Y;
    brotensor::window_partition_forward(X, N, C, H, W, window, Y);
    const float* y = Y.host_f32();
    // Row 0 = (nh=0, nw=0) -> {1, 2, 5, 6}
    CHECK(y[0] == 1 && y[1] == 2 && y[2] == 5 && y[3] == 6);
    // Row 1 = (nh=0, nw=1) -> {3, 4, 7, 8}
    CHECK(y[4] == 3 && y[5] == 4 && y[6] == 7 && y[7] == 8);
    // Row 2 = (nh=1, nw=0) -> {9, 10, 13, 14}
    CHECK(y[8] == 9 && y[9] == 10 && y[10] == 13 && y[11] == 14);
    // Row 3 = (nh=1, nw=1) -> {11, 12, 15, 16}
    CHECK(y[12] == 11 && y[13] == 12 && y[14] == 15 && y[15] == 16);
}

// ── 6. non-divisible window throws ────────────────────────────────────────
static void test_non_divisible_throws() {
    Tensor X = make_f32(1, 1 * 5 * 6);
    Tensor Y;
    bool threw = false;
    try { brotensor::window_partition_forward(X, 1, 1, 5, 6, 2, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

int main() {
    brotensor::init();
    std::printf("test_window_partition (CPU FP32):\n");
    test_shape();
    test_single_window();
    test_round_trip_partition_then_reverse();
    test_round_trip_reverse_then_partition();
    test_sentinel_placement();
    test_non_divisible_throws();
    if (g_failures == 0) {
        std::printf("  OK  all window_partition CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
