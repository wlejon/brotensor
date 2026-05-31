// ─── CPU-only test for interp2d_align_corners_forward ──────────────────────
//
// align_corners=True maps output pixel o to src = o*(in-1)/(out-1) (out==1 -> 0),
// versus the half-pixel map of interp2d_forward. Coverage:
//   1. Identity (H_out==H_in) for nearest + bilinear reproduces the input.
//   2. Corner alignment: bilinear upsample of a ramp keeps the two endpoints
//      pinned to the input endpoints and stays linear in between (the defining
//      property of align_corners=True).
//   3. Hand-computed bilinear values on a known ramp.
//   4. out_dim==1 degenerate case samples the first input pixel.
//   5. Differs from the half-pixel interp2d_forward on a non-trivial upsample.
//
// CPU is FP32-only; the CPU↔GPU parity case lives in test_interp2d_parity.cpp.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>

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
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

static Tensor cpu(int rows, int cols) {
    Tensor t;
    t.resize(rows, cols, Dtype::FP32);
    return t;
}

// ── 1. identity ────────────────────────────────────────────────────────────
static void test_identity() {
    const int N = 1, C = 2, H = 5, W = 6;
    Tensor X = cpu(N, C * H * W);
    float* p = X.host_f32_mut();
    for (int i = 0; i < N * C * H * W; ++i) p[i] = std::sin(0.1f * i);
    for (int mode : {0, 1}) {
        Tensor Y = cpu(0, 0);
        brotensor::interp2d_align_corners_forward(X, N, C, H, W, H, W, mode, Y);
        const float* a = X.host_f32();
        const float* b = Y.host_f32();
        bool ok = true;
        for (int i = 0; i < N * C * H * W; ++i)
            if (!approx(a[i], b[i], 1e-6f)) { ok = false; break; }
        CHECK(ok);
    }
}

// ── 2. corner alignment: ramp endpoints pinned, linear interior ─────────────
static void test_ramp_endpoints() {
    // 1x1x1x4 ramp value==index -> 1x1x1x7 bilinear align_corners=True.
    // src_x = ow*(4-1)/(7-1) = ow*0.5, and value==index so y[ow] == ow*0.5.
    const int W = 4, W_out = 7;
    Tensor X = cpu(1, W);
    float* p = X.host_f32_mut();
    for (int i = 0; i < W; ++i) p[i] = static_cast<float>(i);
    Tensor Y = cpu(0, 0);
    brotensor::interp2d_align_corners_forward(X, 1, 1, 1, W, 1, W_out, 1, Y);
    const float* y = Y.host_f32();
    CHECK(Y.rows == 1 && Y.cols == W_out);
    CHECK(approx(y[0], 0.0f));              // first output == first input
    CHECK(approx(y[W_out - 1], 3.0f));      // last output  == last input
    for (int ow = 0; ow < W_out; ++ow)
        CHECK(approx(y[ow], 0.5f * ow));
}

// ── 3. 2D corners preserved ─────────────────────────────────────────────────
static void test_2d_corners() {
    const int H = 2, W = 2, H_out = 5, W_out = 5;
    Tensor X = cpu(1, H * W);
    float* p = X.host_f32_mut();
    p[0] = 10.f; p[1] = 20.f; p[2] = 30.f; p[3] = 40.f;   // [[10,20],[30,40]]
    Tensor Y = cpu(0, 0);
    brotensor::interp2d_align_corners_forward(X, 1, 1, H, W, H_out, W_out, 1, Y);
    const float* y = Y.host_f32();
    CHECK(approx(y[0], 10.f));                       // top-left
    CHECK(approx(y[W_out - 1], 20.f));               // top-right
    CHECK(approx(y[(H_out - 1) * W_out], 30.f));     // bottom-left
    CHECK(approx(y[H_out * W_out - 1], 40.f));       // bottom-right
    CHECK(approx(y[2 * W_out + 2], 25.f));           // exact centre = mean
}

// ── 4. out_dim==1 samples the first pixel ───────────────────────────────────
static void test_out_one() {
    const int W = 5;
    Tensor X = cpu(1, W);
    float* p = X.host_f32_mut();
    for (int i = 0; i < W; ++i) p[i] = 100.f + i;
    Tensor Y = cpu(0, 0);
    brotensor::interp2d_align_corners_forward(X, 1, 1, 1, W, 1, 1, 1, Y);
    CHECK(Y.cols == 1);
    CHECK(approx(Y.host_f32()[0], 100.f));   // src = 0 -> X[0]
}

// ── 5. differs from half-pixel interp2d_forward ─────────────────────────────
static void test_differs_from_half_pixel() {
    const int W = 4, W_out = 8;
    Tensor X = cpu(1, W);
    float* p = X.host_f32_mut();
    for (int i = 0; i < W; ++i) p[i] = static_cast<float>(i * i);  // nonlinear
    Tensor Yac = cpu(0, 0), Yhp = cpu(0, 0);
    brotensor::interp2d_align_corners_forward(X, 1, 1, 1, W, 1, W_out, 1, Yac);
    brotensor::interp2d_forward(X, 1, 1, 1, W, 1, W_out, 1, Yhp);
    const float* a = Yac.host_f32();
    const float* b = Yhp.host_f32();
    bool any_diff = false;
    for (int i = 0; i < W_out; ++i)
        if (!approx(a[i], b[i])) { any_diff = true; break; }
    CHECK(any_diff);
    // align_corners pins the last sample to the input end; half-pixel does not.
    CHECK(approx(a[W_out - 1], 9.0f));   // X[3] == 9
}

int main() {
    brotensor::init();
    std::printf("test_interp2d_align_corners (CPU FP32):\n");
    test_identity();
    test_ramp_endpoints();
    test_2d_corners();
    test_out_one();
    test_differs_from_half_pixel();
    if (g_failures == 0) {
        std::printf("  OK  all interp2d_align_corners CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
