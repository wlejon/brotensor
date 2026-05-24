// ─── CPU-only test for slice2d_forward / slice2d_backward ──────────────────
//
// Coverage:
//   1. Forward extracts the (H_out, W_out) sub-region; N,C pass through.
//   2. Identity slice (h0=0, w0=0, full H/W) is a verbatim copy.
//   3. Empty slice (H_out=0 or W_out=0) is a valid no-op.
//   4. Out-of-bounds slice throws.
//   5. Negative offsets throw.
//   6. Backward zeros dX, then deposits dY at the slice region — pixels
//      outside the region are zero.
//   7. slice followed by its adjoint reproduces the slice region (verified
//      by composing with pad2d's zero mode: slice2d_backward into a zero
//      buffer equals pad2d_forward of the slice in zero mode).

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

static Tensor make_cpu(int rows, int cols) {
    Tensor t;
    t.resize(rows, cols, Dtype::FP32);
    return t;
}

static void fill_random(Tensor& t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng);
}

// ── 1. forward sub-region extraction ──────────────────────────────────────
static void test_extract_subregion() {
    const int N = 2, C = 3, H = 6, W = 8;
    const int h0 = 1, w0 = 2, H_out = 3, W_out = 4;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0x11);
    Tensor Y = make_cpu(0, 0);
    brotensor::slice2d_forward(X, N, C, H, W, h0, w0, H_out, W_out, Y);
    CHECK(Y.rows == N && Y.cols == C * H_out * W_out);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H_out; ++h)
                for (int w = 0; w < W_out; ++w) {
                    const long xi =
                        ((static_cast<long>(n) * C + c) * H + (h0 + h)) * W +
                        (w0 + w);
                    const long yi =
                        ((static_cast<long>(n) * C + c) * H_out + h) * W_out +
                        w;
                    CHECK(y[yi] == x[xi]);
                }
}

// ── 2. identity slice is a verbatim copy ──────────────────────────────────
static void test_identity_slice() {
    const int N = 1, C = 2, H = 5, W = 4;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0x22);
    Tensor Y = make_cpu(0, 0);
    brotensor::slice2d_forward(X, N, C, H, W, 0, 0, H, W, Y);
    CHECK(Y.cols == X.cols);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int i = 0; i < X.cols; ++i) CHECK(y[i] == x[i]);
}

// ── 3. empty slice is a no-op ──────────────────────────────────────────────
static void test_empty_slice() {
    const int N = 1, C = 2, H = 4, W = 4;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0x33);
    Tensor Y = make_cpu(0, 0);
    brotensor::slice2d_forward(X, N, C, H, W, 1, 1, 0, 3, Y);
    CHECK(Y.rows == N && Y.cols == 0);
}

// ── 4. out-of-bounds slice throws ─────────────────────────────────────────
static void test_oob_throws() {
    Tensor X = make_cpu(1, 1 * 4 * 4);
    Tensor Y = make_cpu(0, 0);
    bool threw = false;
    try { brotensor::slice2d_forward(X, 1, 1, 4, 4, 2, 0, 3, 4, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { brotensor::slice2d_forward(X, 1, 1, 4, 4, 0, 2, 4, 3, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 5. negative offsets throw ─────────────────────────────────────────────
static void test_negative_offsets_throw() {
    Tensor X = make_cpu(1, 1 * 4 * 4);
    Tensor Y = make_cpu(0, 0);
    bool threw = false;
    try { brotensor::slice2d_forward(X, 1, 1, 4, 4, -1, 0, 2, 2, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 6. backward zero-then-scatter at the slice region ─────────────────────
static void test_backward_scatter() {
    const int N = 1, C = 2, H = 5, W = 6;
    const int h0 = 1, w0 = 2, H_out = 2, W_out = 3;
    Tensor dY = make_cpu(N, C * H_out * W_out);
    fill_random(dY, 0x44);
    Tensor dX = make_cpu(0, 0);
    brotensor::slice2d_backward(dY, N, C, H, W, h0, w0, H_out, W_out, dX);
    CHECK(dX.rows == N && dX.cols == C * H * W);
    const float* dy = dY.host_f32();
    const float* dx = dX.host_f32();
    // Interior copied from dY; everything else zero.
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    const long xi =
                        ((static_cast<long>(n) * C + c) * H + h) * W + w;
                    const bool in_slice = (h >= h0 && h < h0 + H_out &&
                                           w >= w0 && w < w0 + W_out);
                    if (in_slice) {
                        const long yi =
                            ((static_cast<long>(n) * C + c) * H_out +
                             (h - h0)) * W_out + (w - w0);
                        CHECK(dx[xi] == dy[yi]);
                    } else {
                        CHECK(dx[xi] == 0.0f);
                    }
                }
}

// ── 7. slice2d_backward == pad2d_forward(zero, asymmetric pad) ────────────
static void test_backward_equals_zero_pad() {
    const int N = 1, C = 1, H = 6, W = 6;
    const int h0 = 1, w0 = 2, H_out = 3, W_out = 2;
    // pt = h0, pb = H - h0 - H_out;  pl = w0, pr = W - w0 - W_out.
    const int pt = h0, pb = H - h0 - H_out;
    const int pl = w0, pr = W - w0 - W_out;
    Tensor dY = make_cpu(N, C * H_out * W_out);
    fill_random(dY, 0x55);
    Tensor via_slice = make_cpu(0, 0);
    brotensor::slice2d_backward(dY, N, C, H, W, h0, w0, H_out, W_out,
                                via_slice);
    Tensor via_pad = make_cpu(0, 0);
    brotensor::pad2d_forward(dY, N, C, H_out, W_out, pt, pb, pl, pr,
                             /*zero*/ 0, via_pad);
    CHECK(via_slice.cols == via_pad.cols);
    const float* a = via_slice.host_f32();
    const float* b = via_pad.host_f32();
    for (int i = 0; i < via_slice.cols; ++i) CHECK(a[i] == b[i]);
}

int main() {
    brotensor::init();
    std::printf("test_slice2d (CPU FP32):\n");
    test_extract_subregion();
    test_identity_slice();
    test_empty_slice();
    test_oob_throws();
    test_negative_offsets_throw();
    test_backward_scatter();
    test_backward_equals_zero_pad();
    if (g_failures == 0) {
        std::printf("  OK  all slice2d CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
