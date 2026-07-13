// ─── CPU-only test for pixel_shuffle_upsample_2x_forward ───────────────────
//
// DC-AE up-shortcut: repeat_interleave along channels fused with a 2x
// pixel-shuffle, as one gather:
//     Y[n, c_out, 2h+i, 2w+j] = X[n, (4*c_out + 2i + j) / repeats, h, w]
//     repeats = 4*C_out / C_in,  i,j in {0,1}
// X: (N, C_in*H*W).  Y: (N, C_out*2H*2W).  FP32-only on CPU. Inference-only
// (pure gather — no backward).
//
// Coverage:
//   1. Output shape is (N, C_out * 2H * 2W) and dtype follows X.
//   2. The general case (repeats == 2) matches the index formula element-wise.
//   3. repeats == 1 (C_in == 4*C_out) is a plain pixel-shuffle: the four source
//      channels 4c..4c+3 tile the 2x2 output block in (i,j) raster order.
//   4. repeats == 4 (C_in == C_out) degenerates to a 2x nearest upsample:
//      every 2x2 output block is the source pixel repeated four times.
//   5. It is a pure gather — every output value is some input value, and no
//      input in range is invented or scaled.
//   6. C_in not dividing 4*C_out throws.
//   7. Bad (negative / zero-channel) dimensions throw.
//   8. Degenerate N/H/W == 0 is a valid no-op that still shapes Y.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

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

// Independent reference, written straight from the formula in ops/spatial.h.
static void check_against_formula(const Tensor& X, const Tensor& Y,
                                  int N, int C_in, int H, int W, int C_out) {
    const int repeats = (4 * C_out) / C_in;
    const int H_out = 2 * H, W_out = 2 * W;
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C_out; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j) {
                            const int src_c = (4 * c + 2 * i + j) / repeats;
                            const long xi =
                                ((static_cast<long>(n) * C_in + src_c) * H + h) * W + w;
                            const long yi =
                                ((static_cast<long>(n) * C_out + c) * H_out +
                                 (2 * h + i)) * W_out + (2 * w + j);
                            CHECK(y[yi] == x[xi]);
                        }
}

// ── 1 + 2. shape, and the general repeats == 2 case ───────────────────────
static void test_general_case() {
    // C_in = 8, C_out = 4  ->  repeats = 4*4/8 = 2.
    const int N = 2, C_in = 8, H = 3, W = 5, C_out = 4;
    Tensor X = make_cpu(N, C_in * H * W);
    fill_random(X, 0x11);
    Tensor Y = make_cpu(0, 0);
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, Y);

    CHECK(Y.rows == N);
    CHECK(Y.cols == C_out * (2 * H) * (2 * W));
    CHECK(Y.dtype == Dtype::FP32);
    check_against_formula(X, Y, N, C_in, H, W, C_out);
}

// ── 3. repeats == 1 -> plain pixel-shuffle ────────────────────────────────
static void test_plain_pixel_shuffle() {
    // C_in == 4*C_out  ->  repeats = 1.
    const int N = 1, C_out = 2, C_in = 4 * C_out, H = 2, W = 2;
    Tensor X = make_cpu(N, C_in * H * W);
    fill_random(X, 0x22);
    Tensor Y = make_cpu(0, 0);
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, Y);
    check_against_formula(X, Y, N, C_in, H, W, C_out);

    // With repeats == 1 the four source channels 4c+0..4c+3 land in the 2x2
    // block in (i,j) raster order: (0,0)->4c, (0,1)->4c+1, (1,0)->4c+2, (1,1)->4c+3.
    const int W_out = 2 * W, H_out = 2 * H;
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int c = 0; c < C_out; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j) {
                        const int src_c = 4 * c + 2 * i + j;
                        const long xi = (static_cast<long>(src_c) * H + h) * W + w;
                        const long yi =
                            ((static_cast<long>(c) * H_out) + (2 * h + i)) * W_out
                            + (2 * w + j);
                        CHECK(y[yi] == x[xi]);
                    }
    (void)H_out;
}

// ── 4. repeats == 4 -> 2x nearest upsample ────────────────────────────────
static void test_nearest_upsample() {
    // C_in == C_out  ->  repeats = 4.  (4c + 2i + j) / 4 == c for all i,j in {0,1}.
    const int N = 2, C = 3, H = 4, W = 3;
    Tensor X = make_cpu(N, C * H * W);
    fill_random(X, 0x33);
    Tensor Y = make_cpu(0, 0);
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C, H, W, C, Y);
    CHECK(Y.cols == C * (2 * H) * (2 * W));

    const int H_out = 2 * H, W_out = 2 * W;
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    const long xi =
                        ((static_cast<long>(n) * C + c) * H + h) * W + w;
                    const float src = x[xi];
                    // All four pixels of the 2x2 block equal the source pixel.
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j) {
                            const long yi =
                                ((static_cast<long>(n) * C + c) * H_out +
                                 (2 * h + i)) * W_out + (2 * w + j);
                            CHECK(y[yi] == src);
                        }
                }
}

// ── 5. pure gather — no value is invented or scaled ───────────────────────
static void test_pure_gather() {
    const int N = 1, C_in = 8, H = 2, W = 3, C_out = 4;
    Tensor X = make_cpu(N, C_in * H * W);
    // Give every input a unique, exactly-representable value so we can tell
    // "copied" from "computed" with an exact compare.
    for (int k = 0; k < C_in * H * W; ++k)
        X.host_f32_mut()[k] = static_cast<float>(k + 1);
    Tensor Y = make_cpu(0, 0);
    brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, Y);

    std::vector<bool> is_input(static_cast<size_t>(C_in * H * W) + 2, false);
    for (int k = 0; k < C_in * H * W; ++k) is_input[static_cast<size_t>(k) + 1] = true;

    const float* y = Y.host_f32();
    for (int k = 0; k < Y.cols; ++k) {
        // Every output must be one of the input values verbatim — integral,
        // in range, and never a blend of two of them.
        const float v = y[k];
        const int as_int = static_cast<int>(v);
        CHECK(static_cast<float>(as_int) == v);
        CHECK(as_int >= 1 && as_int <= C_in * H * W);
        CHECK(is_input[static_cast<size_t>(as_int)]);
    }
}

// ── 6. C_in not dividing 4*C_out throws ───────────────────────────────────
static void test_bad_repeats_throws() {
    // C_in = 7, C_out = 4  ->  4*4 = 16, 16 % 7 != 0.
    const int N = 1, C_in = 7, H = 2, W = 2, C_out = 4;
    Tensor X = make_cpu(N, C_in * H * W);
    Tensor Y = make_cpu(0, 0);
    bool threw = false;
    try { brotensor::pixel_shuffle_upsample_2x_forward(X, N, C_in, H, W, C_out, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 7. bad dimensions throw ───────────────────────────────────────────────
static void test_bad_dims_throw() {
    Tensor X = make_cpu(1, 4 * 2 * 2);
    Tensor Y = make_cpu(0, 0);

    bool threw = false;
    try { brotensor::pixel_shuffle_upsample_2x_forward(X, -1, 4, 2, 2, 1, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);                                   // N < 0

    threw = false;
    try { brotensor::pixel_shuffle_upsample_2x_forward(X, 1, 0, 2, 2, 1, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);                                   // C_in == 0

    threw = false;
    try { brotensor::pixel_shuffle_upsample_2x_forward(X, 1, 4, 2, 2, 0, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);                                   // C_out == 0

    threw = false;
    try { brotensor::pixel_shuffle_upsample_2x_forward(X, 1, 4, -2, 2, 1, Y); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);                                   // H < 0
}

// ── 8. degenerate N/H/W == 0 is a shaped no-op ────────────────────────────
static void test_degenerate_noop() {
    // H == 0: Y is still shaped (N, C_out * 0 * 0) == (N, 0), and nothing throws.
    Tensor X = make_cpu(2, 0);
    Tensor Y = make_cpu(0, 0);
    brotensor::pixel_shuffle_upsample_2x_forward(X, 2, 4, 0, 0, 1, Y);
    CHECK(Y.rows == 2 && Y.cols == 0);

    // N == 0: no rows.
    Tensor X0 = make_cpu(0, 0);
    Tensor Y0 = make_cpu(0, 0);
    brotensor::pixel_shuffle_upsample_2x_forward(X0, 0, 4, 2, 2, 1, Y0);
    CHECK(Y0.rows == 0);
}

int main() {
    brotensor::init();
    std::printf("test_pixel_shuffle_upsample (CPU FP32):\n");
    test_general_case();
    test_plain_pixel_shuffle();
    test_nearest_upsample();
    test_pure_gather();
    test_bad_repeats_throws();
    test_bad_dims_throw();
    test_degenerate_noop();
    if (g_failures == 0) {
        std::printf("  OK  all pixel_shuffle_upsample CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
