// ─── CPU-only test for gather_rows / scatter_rows_add ──────────────────────
//
// Coverage:
//   1. Basic gather: Y[m, :] == X[Idx[m], :] across distinct indices.
//   2. Duplicate indices in gather just copy the same row twice.
//   3. Empty M is valid (Y has 0 rows).
//   4. scatter_rows_add zeros dX, then sums dY rows onto dX[Idx[m]].
//   5. Duplicate indices in scatter accumulate.
//   6. gather then scatter (adjoint) gives the loss-vector-product identity:
//        sum(Y * V) where dY = V, dX = scatter(V, Idx)
//        equivalent to sum(X * scatter(V, Idx, X.rows)) at the gathered rows.
//   7. Idx dtype != INT32 throws; Idx.cols != 1 throws.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

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

static bool approx(float a, float b, float tol = 1e-5f) {
    const float d = std::fabs(a - b);
    return d <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

static Tensor make_f32(int r, int c) {
    Tensor t;
    t.resize(r, c, Dtype::FP32);
    return t;
}

static Tensor make_idx(const std::vector<int32_t>& v) {
    Tensor t;
    t.resize(static_cast<int>(v.size()), 1, Dtype::INT32);
    int32_t* p = static_cast<int32_t*>(t.data);
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static void fill_random(Tensor& t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = d(rng);
}

// ── 1. basic gather ───────────────────────────────────────────────────────
static void test_basic_gather() {
    const int R = 5, C = 4;
    Tensor X = make_f32(R, C);
    fill_random(X, 0x11);
    Tensor Idx = make_idx({3, 1, 4, 0});
    Tensor Y;
    brotensor::gather_rows(X, Idx, Y);
    CHECK(Y.rows == 4 && Y.cols == C);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    const int32_t* ip = static_cast<const int32_t*>(Idx.data);
    for (int m = 0; m < Idx.rows; ++m)
        for (int c = 0; c < C; ++c)
            CHECK(y[m * C + c] == x[ip[m] * C + c]);
}

// ── 2. duplicate gather ──────────────────────────────────────────────────
static void test_duplicate_gather() {
    const int R = 3, C = 2;
    Tensor X = make_f32(R, C);
    fill_random(X, 0x22);
    Tensor Idx = make_idx({1, 1, 1});
    Tensor Y;
    brotensor::gather_rows(X, Idx, Y);
    const float* x = X.host_f32();
    const float* y = Y.host_f32();
    for (int m = 0; m < 3; ++m)
        for (int c = 0; c < C; ++c)
            CHECK(y[m * C + c] == x[1 * C + c]);
}

// ── 3. empty M ────────────────────────────────────────────────────────────
static void test_empty_gather() {
    Tensor X = make_f32(4, 3);
    fill_random(X, 0x33);
    Tensor Idx;
    Idx.resize(0, 1, Dtype::INT32);
    Tensor Y;
    brotensor::gather_rows(X, Idx, Y);
    CHECK(Y.rows == 0 && Y.cols == 3);
}

// ── 4. scatter zeros then sums ────────────────────────────────────────────
static void test_scatter_basic() {
    const int R = 4, C = 3, M = 3;
    Tensor dY = make_f32(M, C);
    fill_random(dY, 0x44);
    Tensor Idx = make_idx({2, 0, 3});
    Tensor dX;
    brotensor::scatter_rows_add(dY, Idx, R, dX);
    CHECK(dX.rows == R && dX.cols == C);
    const float* dy = dY.host_f32();
    const float* dx = dX.host_f32();
    // dX[2] = dY[0]; dX[0] = dY[1]; dX[3] = dY[2]; dX[1] = 0.
    for (int c = 0; c < C; ++c) {
        CHECK(approx(dx[2 * C + c], dy[0 * C + c]));
        CHECK(approx(dx[0 * C + c], dy[1 * C + c]));
        CHECK(approx(dx[3 * C + c], dy[2 * C + c]));
        CHECK(dx[1 * C + c] == 0.0f);
    }
}

// ── 5. scatter duplicates accumulate ──────────────────────────────────────
static void test_scatter_duplicates_sum() {
    const int R = 3, C = 2, M = 4;
    Tensor dY = make_f32(M, C);
    float* dyp = dY.host_f32_mut();
    for (int i = 0; i < dY.cols * dY.rows; ++i) dyp[i] = 1.0f;
    Tensor Idx = make_idx({0, 1, 0, 0});  // dX[0] gets 3, dX[1] gets 1, dX[2] zero
    Tensor dX;
    brotensor::scatter_rows_add(dY, Idx, R, dX);
    const float* dx = dX.host_f32();
    for (int c = 0; c < C; ++c) {
        CHECK(approx(dx[0 * C + c], 3.0f));
        CHECK(approx(dx[1 * C + c], 1.0f));
        CHECK(dx[2 * C + c] == 0.0f);
    }
}

// ── 6. gather + scatter adjoint identity ──────────────────────────────────
//   <gather(X, Idx), V> = <X, scatter(V, Idx, X.rows)>
//   (a standard adjoint inner-product check.)
static void test_adjoint_identity() {
    const int R = 6, C = 5, M = 4;
    Tensor X = make_f32(R, C);
    fill_random(X, 0x55);
    Tensor Idx = make_idx({1, 3, 1, 5});  // duplicate index intentional
    Tensor V = make_f32(M, C);
    fill_random(V, 0x66);

    // lhs = sum(gather(X) * V)
    Tensor Y;
    brotensor::gather_rows(X, Idx, Y);
    double lhs = 0.0;
    const float* yp = Y.host_f32();
    const float* vp = V.host_f32();
    for (int i = 0; i < Y.rows * Y.cols; ++i)
        lhs += static_cast<double>(yp[i]) * vp[i];

    // rhs = sum(X * scatter(V, Idx, R))
    Tensor S;
    brotensor::scatter_rows_add(V, Idx, R, S);
    double rhs = 0.0;
    const float* xp = X.host_f32();
    const float* sp = S.host_f32();
    for (int i = 0; i < X.rows * X.cols; ++i)
        rhs += static_cast<double>(xp[i]) * sp[i];

    CHECK(approx(static_cast<float>(lhs),
                 static_cast<float>(rhs), 1e-5f));
}

// ── 7. dtype + shape validation ───────────────────────────────────────────
static void test_validation_throws() {
    Tensor X = make_f32(3, 2);
    Tensor Y;
    {
        Tensor BadIdx = make_f32(2, 1);
        bool threw = false;
        try { brotensor::gather_rows(X, BadIdx, Y); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    {
        Tensor BadIdx;
        BadIdx.resize(1, 2, Dtype::INT32);
        bool threw = false;
        try { brotensor::gather_rows(X, BadIdx, Y); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
}

int main() {
    brotensor::init();
    std::printf("test_gather_rows (CPU FP32):\n");
    test_basic_gather();
    test_duplicate_gather();
    test_empty_gather();
    test_scatter_basic();
    test_scatter_duplicates_sum();
    test_adjoint_identity();
    test_validation_throws();
    if (g_failures == 0) {
        std::printf("  OK  all gather/scatter CPU tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
