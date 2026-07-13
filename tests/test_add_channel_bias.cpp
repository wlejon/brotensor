// ─── CPU-only test for add_channel_bias_inplace ────────────────────────────
//
// Channel-broadcast bias add over a channel-major (C, L) buffer:
//     y[c*L + i] += bias[c]
// The per-channel bias every 1x1 (and grouped) conv wants once its GEMM is
// expressed as a plain matmul.
//
// Coverage:
//   1. Every element of channel c gains exactly bias[c]; nothing else moves.
//   2. It accumulates onto y rather than overwriting (the "+=" in the contract).
//   3. Zero bias is a no-op.
//   4. Applying the op twice equals applying 2*bias once (linearity in bias).
//   5. C == 1 broadcasts a single scalar across the whole buffer.
//   6. L == 1 degenerates to a plain elementwise vector add.
//   7. Distinct per-channel biases land on the right channels — a bias vector
//      of [0, 1, 2, ...] over a zeroed y reproduces the channel index at every
//      position, which catches any c/L index transposition.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
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

static bool close(float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) <= tol + 1e-5f * std::fabs(b);
}

// ── 1 + 2. each channel gains exactly bias[c], accumulating onto y ────────
static void test_broadcast_accumulates() {
    const int C = 4, L = 6;
    Tensor y = make_cpu(1, C * L);
    fill_random(y, 0x11);
    Tensor bias = make_cpu(1, C);
    fill_random(bias, 0x12);

    // Snapshot y before the op so we can assert it accumulated, not overwrote.
    const std::vector<float> before(y.host_f32(), y.host_f32() + C * L);
    const std::vector<float> b(bias.host_f32(), bias.host_f32() + C);

    brotensor::add_channel_bias_inplace(y, bias, C, L);

    const float* after = y.host_f32();
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < L; ++i) {
            const int k = c * L + i;
            CHECK(close(after[k], before[k] + b[c]));
        }
}

// ── 3. zero bias is a no-op ───────────────────────────────────────────────
static void test_zero_bias_noop() {
    const int C = 3, L = 5;
    Tensor y = make_cpu(1, C * L);
    fill_random(y, 0x22);
    const std::vector<float> before(y.host_f32(), y.host_f32() + C * L);

    Tensor bias = make_cpu(1, C);
    for (int c = 0; c < C; ++c) bias.host_f32_mut()[c] = 0.0f;

    brotensor::add_channel_bias_inplace(y, bias, C, L);

    const float* after = y.host_f32();
    for (int k = 0; k < C * L; ++k) CHECK(after[k] == before[k]);  // bit-identical
}

// ── 4. applying twice == applying 2*bias once ─────────────────────────────
static void test_twice_equals_double() {
    const int C = 3, L = 7;
    Tensor y_twice = make_cpu(1, C * L);
    fill_random(y_twice, 0x33);
    Tensor y_double = make_cpu(1, C * L);
    // Same starting buffer.
    for (int k = 0; k < C * L; ++k)
        y_double.host_f32_mut()[k] = y_twice.host_f32()[k];

    Tensor bias = make_cpu(1, C);
    fill_random(bias, 0x34);
    Tensor bias2 = make_cpu(1, C);
    for (int c = 0; c < C; ++c)
        bias2.host_f32_mut()[c] = 2.0f * bias.host_f32()[c];

    brotensor::add_channel_bias_inplace(y_twice, bias, C, L);
    brotensor::add_channel_bias_inplace(y_twice, bias, C, L);
    brotensor::add_channel_bias_inplace(y_double, bias2, C, L);

    const float* a = y_twice.host_f32();
    const float* b = y_double.host_f32();
    for (int k = 0; k < C * L; ++k) CHECK(close(a[k], b[k]));
}

// ── 5. C == 1 broadcasts one scalar everywhere ────────────────────────────
static void test_single_channel() {
    const int C = 1, L = 9;
    Tensor y = make_cpu(1, L);
    fill_random(y, 0x44);
    const std::vector<float> before(y.host_f32(), y.host_f32() + L);
    Tensor bias = make_cpu(1, 1);
    bias.host_f32_mut()[0] = 2.5f;

    brotensor::add_channel_bias_inplace(y, bias, C, L);

    const float* after = y.host_f32();
    for (int i = 0; i < L; ++i) CHECK(close(after[i], before[i] + 2.5f));
}

// ── 6. L == 1 is a plain elementwise vector add ───────────────────────────
static void test_single_position() {
    const int C = 5, L = 1;
    Tensor y = make_cpu(1, C);
    fill_random(y, 0x55);
    const std::vector<float> before(y.host_f32(), y.host_f32() + C);
    Tensor bias = make_cpu(1, C);
    fill_random(bias, 0x56);
    const std::vector<float> b(bias.host_f32(), bias.host_f32() + C);

    brotensor::add_channel_bias_inplace(y, bias, C, L);

    const float* after = y.host_f32();
    for (int c = 0; c < C; ++c) CHECK(close(after[c], before[c] + b[c]));
}

// ── 7. channel index lands where it should (catches a c/L transposition) ──
static void test_channel_indexing() {
    const int C = 4, L = 3;
    Tensor y = make_cpu(1, C * L);
    for (int k = 0; k < C * L; ++k) y.host_f32_mut()[k] = 0.0f;
    Tensor bias = make_cpu(1, C);
    for (int c = 0; c < C; ++c)
        bias.host_f32_mut()[c] = static_cast<float>(c);   // 0, 1, 2, 3

    brotensor::add_channel_bias_inplace(y, bias, C, L);

    // Channel-major: every one of the L slots in channel c must read exactly c.
    // A (L, C) transposition would instead produce 0,1,2,3,0,1,2,3,... here.
    const float* after = y.host_f32();
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < L; ++i)
            CHECK(after[c * L + i] == static_cast<float>(c));
}

int main() {
    brotensor::init();
    std::printf("test_add_channel_bias (CPU FP32):\n");
    test_broadcast_accumulates();
    test_zero_bias_noop();
    test_twice_equals_double();
    test_single_channel();
    test_single_position();
    test_channel_indexing();
    if (g_failures == 0) {
        std::printf("  OK  all add_channel_bias CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
