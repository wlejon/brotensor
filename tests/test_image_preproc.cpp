// ─── CPU-only test for image preprocessing helpers ────────────────────────
//
// Coverage:
//   1. image_normalize — output matches per-channel (X - mean[c]) / std[c].
//   2. image_normalize rejects std[c] == 0.
//   3. image_u8_to_f32_nhwc_to_nchw — known pixel values produce the expected
//      FP32 NCHW tensor; scale + bias applied; layout reorder correct.
//   4. The two canonical scaling conventions ([0,255]->[0,1], [0,255]->[-1,1])
//      hit the expected boundary values.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
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

static Tensor make_f32(int r, int c) {
    Tensor t;
    t.resize(r, c, Dtype::FP32);
    return t;
}

static void fill_random(Tensor& t, uint64_t seed,
                        float lo = -1.0f, float hi = 1.0f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng);
}

// ── 1. image_normalize — matches per-channel (x - mu) / s ─────────────────
static void test_image_normalize_basic() {
    const int N = 2, C = 3, H = 4, W = 5;
    const int spatial = H * W;
    Tensor X = make_f32(N, C * spatial);
    Tensor mean = make_f32(C, 1);
    Tensor std_ = make_f32(C, 1);
    fill_random(X, 0xE1);
    // ImageNet-ish values.
    float mu[3]  = {0.485f, 0.456f, 0.406f};
    float sig[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < C; ++c) {
        mean.host_f32_mut()[c] = mu[c];
        std_.host_f32_mut()[c] = sig[c];
    }

    Tensor Y;
    brotensor::image_normalize(X, mean, std_, N, C, H, W, Y);
    CHECK(Y.rows == N && Y.cols == C * spatial);

    const float* Xp = X.host_f32();
    const float* Yp = Y.host_f32();
    for (int c = 0; c < C; ++c) {
        const float inv = 1.0f / sig[c];
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const int idx = ((n * C + c) * spatial) + s;
                const float ref = (Xp[idx] - mu[c]) * inv;
                if (std::fabs(Yp[idx] - ref) > 1e-6f) {
                    std::printf("    mismatch idx=%d got=%g ref=%g\n",
                                idx, Yp[idx], ref);
                    CHECK(false);
                    return;
                }
            }
        }
    }
}

// ── 2. image_normalize throws on std[c] == 0 ──────────────────────────────
static void test_image_normalize_zero_std() {
    const int N = 1, C = 2, H = 2, W = 2;
    Tensor X = make_f32(N, C * H * W);
    Tensor mean = make_f32(C, 1);
    Tensor std_ = make_f32(C, 1);
    fill_random(X, 0xE2);
    mean.host_f32_mut()[0] = 0.0f;
    mean.host_f32_mut()[1] = 0.0f;
    std_.host_f32_mut()[0] = 1.0f;
    std_.host_f32_mut()[1] = 0.0f;  // poison

    Tensor Y;
    bool threw = false;
    try {
        brotensor::image_normalize(X, mean, std_, N, C, H, W, Y);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

// ── 3. u8 NHWC -> f32 NCHW: known values + layout swap ────────────────────
static void test_u8_to_f32_layout_and_values() {
    // N=1, H=2, W=2, C=3. Pixels chosen so each (h,w,c) is uniquely
    // identifiable; verify the output ends up at the correct (c,h,w) index.
    const int N = 1, H = 2, W = 2, C = 3;
    std::vector<uint8_t> src = {
        // (h=0,w=0)              (h=0,w=1)
        10, 20, 30,                40, 50, 60,
        // (h=1,w=0)              (h=1,w=1)
        70, 80, 90,                100, 110, 120,
    };
    Tensor Y;
    // scale=1.0, bias=0 — easy to verify.
    brotensor::image_u8_to_f32_nhwc_to_nchw(src.data(), N, H, W, C,
                                            1.0f, 0.0f, Y);
    CHECK(Y.rows == N && Y.cols == C * H * W);

    const float* Yp = Y.host_f32();
    // Y[n,c,h,w]; helper:
    auto y = [&](int c, int h, int w) {
        return Yp[(c * H + h) * W + w];  // N=1
    };
    CHECK(y(0,0,0) == 10.0f);  CHECK(y(1,0,0) == 20.0f);  CHECK(y(2,0,0) == 30.0f);
    CHECK(y(0,0,1) == 40.0f);  CHECK(y(1,0,1) == 50.0f);  CHECK(y(2,0,1) == 60.0f);
    CHECK(y(0,1,0) == 70.0f);  CHECK(y(1,1,0) == 80.0f);  CHECK(y(2,1,0) == 90.0f);
    CHECK(y(0,1,1) == 100.0f); CHECK(y(1,1,1) == 110.0f); CHECK(y(2,1,1) == 120.0f);
}

// ── 4. canonical [0,255] -> [0,1] and [-1,1] mappings ─────────────────────
static void test_u8_to_f32_scaling_conventions() {
    // 1 pixel, 1 channel: 0 and 255 endpoints.
    const int N = 2, H = 1, W = 1, C = 1;
    std::vector<uint8_t> src = {0, 255};

    Tensor Y01;
    brotensor::image_u8_to_f32_nhwc_to_nchw(src.data(), N, H, W, C,
                                            1.0f / 255.0f, 0.0f, Y01);
    CHECK(Y01.rows == N && Y01.cols == 1);
    CHECK(std::fabs(Y01.host_f32()[0] - 0.0f) < 1e-6f);
    CHECK(std::fabs(Y01.host_f32()[1] - 1.0f) < 1e-6f);

    Tensor Ym11;
    brotensor::image_u8_to_f32_nhwc_to_nchw(src.data(), N, H, W, C,
                                            2.0f / 255.0f, -1.0f, Ym11);
    CHECK(std::fabs(Ym11.host_f32()[0] - (-1.0f)) < 1e-6f);
    CHECK(std::fabs(Ym11.host_f32()[1] - 1.0f) < 1e-6f);
}

int main() {
    brotensor::init();
    std::printf("test_image_preproc\n");

    test_image_normalize_basic();
    test_image_normalize_zero_std();
    test_u8_to_f32_layout_and_values();
    test_u8_to_f32_scaling_conventions();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll image_preproc checks passed.\n");
    return 0;
}
