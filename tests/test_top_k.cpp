// ─── CPU-only test for top_k_rows ───────────────────────────────────────────
//
// Coverage:
//   1. Descending value order in output.
//   2. Indices map back to source columns (Vals[r, j] == X[r, Idx[r, j]]).
//   3. Tie-break: smaller column index wins.
//   4. k == C reproduces a full descending sort.
//   5. k == 1 matches argmax_rows.
//   6. k < 1 and k > C throw.
//   7. Random rows: independent reference (std::partial_sort over a copy).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
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

static void fill_random(Tensor& t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng);
}

// ── 1 + 2. descending order, indices map back ──────────────────────────────
static void test_basic_descending() {
    const int R = 1, C = 5, k = 3;
    Tensor X = make_f32(R, C);
    float* x = X.host_f32_mut();
    // Row: [0.5, 9.0, 2.0, 9.0, -1.0]  -> top-3 = [9.0(idx 1), 9.0(idx 3), 2.0]
    x[0] = 0.5f; x[1] = 9.0f; x[2] = 2.0f; x[3] = 9.0f; x[4] = -1.0f;
    Tensor V, I;
    brotensor::top_k_rows(X, k, V, I);
    CHECK(V.rows == R && V.cols == k && V.dtype == Dtype::FP32);
    CHECK(I.rows == R && I.cols == k && I.dtype == Dtype::INT32);
    const float* vp = V.host_f32();
    const int32_t* ip = static_cast<const int32_t*>(I.data);
    // Descending values.
    for (int j = 0; j + 1 < k; ++j) CHECK(vp[j] >= vp[j + 1]);
    // Map back to X.
    for (int j = 0; j < k; ++j) CHECK(vp[j] == x[ip[j]]);
    // Tie-break: smaller idx first.
    CHECK(ip[0] == 1 && ip[1] == 3);
}

// ── 3. tie-break across many duplicates ────────────────────────────────────
static void test_tie_break_smaller_idx() {
    const int R = 1, C = 6, k = 4;
    Tensor X = make_f32(R, C);
    float* x = X.host_f32_mut();
    // All equal -> output indices [0, 1, 2, 3].
    for (int i = 0; i < C; ++i) x[i] = 7.0f;
    Tensor V, I;
    brotensor::top_k_rows(X, k, V, I);
    const int32_t* ip = static_cast<const int32_t*>(I.data);
    for (int j = 0; j < k; ++j) CHECK(ip[j] == j);
}

// ── 4. k == C reproduces a full sort ───────────────────────────────────────
static void test_full_sort() {
    const int R = 3, C = 7, k = C;
    Tensor X = make_f32(R, C);
    fill_random(X, 0x42);
    Tensor V, I;
    brotensor::top_k_rows(X, k, V, I);
    const float* vp = V.host_f32();
    for (int r = 0; r < R; ++r) {
        for (int j = 0; j + 1 < k; ++j)
            CHECK(vp[r * k + j] >= vp[r * k + j + 1]);
    }
}

// ── 5. k == 1 matches argmax_rows ──────────────────────────────────────────
static void test_k1_matches_argmax() {
    const int R = 4, C = 11;
    Tensor X = make_f32(R, C);
    fill_random(X, 0x99);
    Tensor V, I;
    brotensor::top_k_rows(X, 1, V, I);
    // argmax_rows historically returns FP32 indices (one per row).
    Tensor Idx_ref;
    brotensor::argmax_rows(X, Idx_ref);
    const int32_t* a = static_cast<const int32_t*>(I.data);
    const float* b = Idx_ref.host_f32();
    for (int r = 0; r < R; ++r) CHECK(a[r] == static_cast<int32_t>(b[r]));
}

// ── 6. k bounds ────────────────────────────────────────────────────────────
static void test_bounds_throw() {
    Tensor X = make_f32(1, 5);
    Tensor V, I;
    bool threw = false;
    try { brotensor::top_k_rows(X, 0, V, I); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { brotensor::top_k_rows(X, 6, V, I); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 7. random rows vs reference ────────────────────────────────────────────
static void test_random_rows_vs_reference() {
    const int R = 5, C = 32, k = 7;
    Tensor X = make_f32(R, C);
    fill_random(X, 0x1234);
    Tensor V, I;
    brotensor::top_k_rows(X, k, V, I);
    const float* x = X.host_f32();
    const float* vp = V.host_f32();
    const int32_t* ip = static_cast<const int32_t*>(I.data);

    for (int r = 0; r < R; ++r) {
        // Reference: pair (value, idx), sort by (-value, idx).
        std::vector<std::pair<float, int>> rows(C);
        for (int c = 0; c < C; ++c) rows[c] = {x[r * C + c], c};
        std::stable_sort(rows.begin(), rows.end(),
                         [](const auto& a, const auto& b) {
                             if (a.first != b.first) return a.first > b.first;
                             return a.second < b.second;
                         });
        for (int j = 0; j < k; ++j) {
            CHECK(vp[r * k + j] == rows[j].first);
            CHECK(ip[r * k + j] == rows[j].second);
        }
    }
}

int main() {
    brotensor::init();
    std::printf("test_top_k (CPU FP32):\n");
    test_basic_descending();
    test_tie_break_smaller_idx();
    test_full_sort();
    test_k1_matches_argmax();
    test_bounds_throw();
    test_random_rows_vs_reference();
    if (g_failures == 0) {
        std::printf("  OK  all top_k CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
