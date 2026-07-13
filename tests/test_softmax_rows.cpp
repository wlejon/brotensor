// ─── CPU-only test for softmax_rows_forward ────────────────────────────────
//
// Row-wise softmax over a flat buffer of `rows` independent rows of `cols`
// each. One call covers every row — the inference primitive behind attention
// score matrices.
//
// Coverage:
//   1. Each row sums to 1 and every probability is in [0, 1].
//   2. Matches an independently-computed max-subtracted reference.
//   3. Rows are independent — perturbing row r leaves every other row bit-identical.
//   4. Shift invariance: adding a constant to a whole row leaves its softmax
//      unchanged (the max-subtraction property).
//   5. A uniform row yields the uniform distribution (1/cols).
//   6. Large-magnitude logits do not overflow to NaN/inf (the max subtraction
//      is what buys this — a naive exp() would produce inf/inf).
//   7. In-place (Y aliases X) matches the out-of-place result — the header
//      explicitly permits Y == X.
//   8. A single row (rows==1) degenerates to a plain softmax.

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

static void fill_random(Tensor& t, uint64_t seed, float scale = 1.0f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng) * scale;
}

static bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol + 1e-4f * std::fabs(b);
}

// Independent reference: max-subtracted softmax, one row at a time.
static std::vector<float> ref_softmax_rows(const float* x, int rows, int cols) {
    std::vector<float> out(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        const float* lp = x + static_cast<size_t>(r) * cols;
        float* pp = out.data() + static_cast<size_t>(r) * cols;
        float m = lp[0];
        for (int i = 1; i < cols; ++i) if (lp[i] > m) m = lp[i];
        double s = 0.0;
        for (int i = 0; i < cols; ++i) {
            pp[i] = std::exp(lp[i] - m);
            s += pp[i];
        }
        for (int i = 0; i < cols; ++i) pp[i] = static_cast<float>(pp[i] / s);
    }
    return out;
}

// ── 1 + 2. rows normalise to 1, and match the reference ───────────────────
static void test_normalised_and_matches_reference() {
    const int rows = 7, cols = 13;
    Tensor X = make_cpu(rows, cols);
    fill_random(X, 0x11, 3.0f);
    Tensor Y = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y, rows, cols);
    CHECK(Y.rows == rows && Y.cols == cols);

    const std::vector<float> ref = ref_softmax_rows(X.host_f32(), rows, cols);
    const float* y = Y.host_f32();
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            const int k = r * cols + i;
            CHECK(y[k] >= 0.0f && y[k] <= 1.0f);
            CHECK(close(y[k], ref[k]));
            sum += y[k];
        }
        CHECK(std::fabs(sum - 1.0) < 1e-5);
    }
}

// ── 3. rows are independent ───────────────────────────────────────────────
static void test_rows_independent() {
    const int rows = 4, cols = 9;
    Tensor X = make_cpu(rows, cols);
    fill_random(X, 0x22, 2.0f);
    Tensor Y0 = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y0, rows, cols);

    // Perturb row 2 only.
    const int perturbed = 2;
    X.host_f32_mut()[perturbed * cols + 3] += 5.0f;
    Tensor Y1 = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y1, rows, cols);

    const float* a = Y0.host_f32();
    const float* b = Y1.host_f32();
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < cols; ++i) {
            const int k = r * cols + i;
            if (r == perturbed) continue;              // this row may change
            CHECK(a[k] == b[k]);                       // others: bit-identical
        }
    }
    // The perturbed row did in fact move.
    bool moved = false;
    for (int i = 0; i < cols; ++i)
        if (a[perturbed * cols + i] != b[perturbed * cols + i]) moved = true;
    CHECK(moved);
}

// ── 4. shift invariance ───────────────────────────────────────────────────
static void test_shift_invariance() {
    const int rows = 3, cols = 8;
    Tensor X = make_cpu(rows, cols);
    fill_random(X, 0x33, 2.0f);
    Tensor Y0 = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y0, rows, cols);

    // Add a different constant to each row — softmax must not move.
    float* x = X.host_f32_mut();
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < cols; ++i)
            x[r * cols + i] += static_cast<float>(r + 1) * 7.5f;

    Tensor Y1 = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y1, rows, cols);
    const float* a = Y0.host_f32();
    const float* b = Y1.host_f32();
    for (int i = 0; i < rows * cols; ++i) CHECK(close(a[i], b[i]));
}

// ── 5. uniform row -> uniform distribution ────────────────────────────────
static void test_uniform_row() {
    const int rows = 2, cols = 5;
    Tensor X = make_cpu(rows, cols);
    float* x = X.host_f32_mut();
    for (int i = 0; i < rows * cols; ++i) x[i] = 0.25f;   // all equal
    Tensor Y = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y, rows, cols);
    const float* y = Y.host_f32();
    const float expect = 1.0f / static_cast<float>(cols);
    for (int i = 0; i < rows * cols; ++i) CHECK(close(y[i], expect));
}

// ── 6. large logits do not overflow ───────────────────────────────────────
static void test_large_logits_no_overflow() {
    const int rows = 2, cols = 4;
    Tensor X = make_cpu(rows, cols);
    float* x = X.host_f32_mut();
    // Row 0: huge positives. Row 1: huge negatives. A naive exp() gives
    // inf/inf and 0/0 respectively; the max-subtraction must keep both finite.
    x[0] = 1000.0f; x[1] = 1001.0f; x[2] = 999.0f;  x[3] = 1000.5f;
    x[4] = -1000.0f; x[5] = -1001.0f; x[6] = -999.0f; x[7] = -1000.5f;
    Tensor Y = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y, rows, cols);
    const float* y = Y.host_f32();
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            const float v = y[r * cols + i];
            CHECK(std::isfinite(v));
            CHECK(v >= 0.0f && v <= 1.0f);
            sum += v;
        }
        CHECK(std::fabs(sum - 1.0) < 1e-5);
    }
    // Row 0's argmax is index 1 (the largest logit) — it must carry the mass.
    CHECK(y[1] > y[0] && y[1] > y[2] && y[1] > y[3]);
    // Row 1's argmax is index 2 (-999 is the largest of the negatives).
    CHECK(y[6] > y[4] && y[6] > y[5] && y[6] > y[7]);
}

// ── 7. in-place (Y aliases X) matches out-of-place ────────────────────────
static void test_in_place() {
    const int rows = 5, cols = 11;
    Tensor X = make_cpu(rows, cols);
    fill_random(X, 0x44, 2.0f);

    Tensor out_of_place = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, out_of_place, rows, cols);

    // Same buffer as both source and destination — the header permits Y == X.
    brotensor::softmax_rows_forward(X, X, rows, cols);

    const float* a = out_of_place.host_f32();
    const float* b = X.host_f32();
    for (int i = 0; i < rows * cols; ++i) CHECK(close(a[i], b[i]));
}

// ── 8. single row degenerates to a plain softmax ──────────────────────────
static void test_single_row() {
    const int cols = 6;
    Tensor X = make_cpu(1, cols);
    fill_random(X, 0x55, 2.0f);
    Tensor Y = make_cpu(0, 0);
    brotensor::softmax_rows_forward(X, Y, 1, cols);
    const std::vector<float> ref = ref_softmax_rows(X.host_f32(), 1, cols);
    const float* y = Y.host_f32();
    double sum = 0.0;
    for (int i = 0; i < cols; ++i) { CHECK(close(y[i], ref[i])); sum += y[i]; }
    CHECK(std::fabs(sum - 1.0) < 1e-5);
}

int main() {
    brotensor::init();
    std::printf("test_softmax_rows (CPU FP32):\n");
    test_normalised_and_matches_reference();
    test_rows_independent();
    test_shift_invariance();
    test_uniform_row();
    test_large_logits_no_overflow();
    test_in_place();
    test_single_row();
    if (g_failures == 0) {
        std::printf("  OK  all softmax_rows CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
