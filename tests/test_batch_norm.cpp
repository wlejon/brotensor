// ─── CPU-only test for BatchNorm forward (training + inference) + backward ─
//
// Coverage:
//   1. Training forward — Y matches an inline reference; saved_mean / rstd
//      written; running_mean / running_var updated under PyTorch convention.
//   2. Inference forward — uses running stats, no mutation; matches the
//      per-channel affine y = x * (gamma / sqrt(var+eps))
//                         + (beta - mu * gamma / sqrt(var+eps)).
//   3. Backward — dX / dGamma / dBeta match an inline reference computed
//      from the saved mean/rstd. dGamma / dBeta accumulate.
//   4. Single-element batch (M==1) doesn't blow up the unbiased correction.

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

static void fill_constant(Tensor& t, float v) {
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.rows * t.cols; ++i) p[i] = v;
}

// Inline FP32 reference. Mirrors the kernel math, double accumulation.
static void bn_forward_ref(const std::vector<float>& X,
                           const std::vector<float>& gamma,
                           const std::vector<float>& beta,
                           int N, int C, int H, int W,
                           float eps,
                           std::vector<float>& Y,
                           std::vector<float>& saved_mean,
                           std::vector<float>& saved_rstd,
                           std::vector<float>& batch_var_unb) {
    const int spatial = H * W;
    const int M = N * spatial;
    Y.assign(static_cast<size_t>(N) * C * spatial, 0.0f);
    saved_mean.assign(C, 0.0f);
    saved_rstd.assign(C, 0.0f);
    batch_var_unb.assign(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        double sum = 0.0, sumsq = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const float v = X[((n * C + c) * spatial) + s];
                sum   += v;
                sumsq += static_cast<double>(v) * v;
            }
        }
        const double inv_M = 1.0 / M;
        const double mean = sum * inv_M;
        const double var_b = sumsq * inv_M - mean * mean;
        const double rstd = 1.0 / std::sqrt(var_b + eps);
        saved_mean[c]    = static_cast<float>(mean);
        saved_rstd[c]    = static_cast<float>(rstd);
        const double bessel = (M > 1) ? static_cast<double>(M) / (M - 1) : 1.0;
        batch_var_unb[c] = static_cast<float>(var_b * bessel);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const int idx = ((n * C + c) * spatial) + s;
                const double yn = (X[idx] - mean) * rstd;
                Y[idx] = static_cast<float>(yn * gamma[c] + beta[c]);
            }
        }
    }
}

static void bn_backward_ref(const std::vector<float>& X,
                            const std::vector<float>& gamma,
                            const std::vector<float>& saved_mean,
                            const std::vector<float>& saved_rstd,
                            const std::vector<float>& dY,
                            int N, int C, int H, int W,
                            std::vector<float>& dX,
                            std::vector<float>& dGamma_add,
                            std::vector<float>& dBeta_add) {
    const int spatial = H * W;
    const int M = N * spatial;
    dX.assign(static_cast<size_t>(N) * C * spatial, 0.0f);
    dGamma_add.assign(C, 0.0f);
    dBeta_add.assign(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        const double mean = saved_mean[c];
        const double rstd = saved_rstd[c];
        const double gv   = gamma[c];
        double sum_dY = 0.0, sum_dY_xh = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const int idx = ((n * C + c) * spatial) + s;
                const double xh = (X[idx] - mean) * rstd;
                sum_dY    += dY[idx];
                sum_dY_xh += dY[idx] * xh;
            }
        }
        dGamma_add[c] = static_cast<float>(sum_dY_xh);
        dBeta_add[c]  = static_cast<float>(sum_dY);
        const double sum1 = gv * sum_dY;
        const double sum2 = gv * sum_dY_xh;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const int idx = ((n * C + c) * spatial) + s;
                const double xh  = (X[idx] - mean) * rstd;
                const double dxh = dY[idx] * gv;
                dX[idx] = static_cast<float>(rstd * (dxh - (sum1 + xh * sum2) / M));
            }
        }
    }
}

static bool approx_eq(float a, float b, float atol, float rtol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// ── 1. training forward ───────────────────────────────────────────────────
static void test_training_forward() {
    const int N = 4, C = 3, H = 5, W = 6;
    const float eps = 1e-5f, momentum = 0.1f;
    const int spatial = H * W;

    Tensor X = make_f32(N, C * spatial);
    Tensor gamma = make_f32(C, 1);
    Tensor beta  = make_f32(C, 1);
    Tensor rm    = make_f32(C, 1);
    Tensor rv    = make_f32(C, 1);
    fill_random(X,     0xA1);
    fill_random(gamma, 0xA2, 0.5f, 1.5f);
    fill_random(beta,  0xA3, -0.3f, 0.3f);
    fill_constant(rm, 0.25f);
    fill_constant(rv, 1.50f);

    // Snapshot inputs for the reference.
    std::vector<float> X_h(X.host_f32(), X.host_f32() + N * C * spatial);
    std::vector<float> g_h(gamma.host_f32(), gamma.host_f32() + C);
    std::vector<float> b_h(beta.host_f32(),  beta.host_f32()  + C);
    std::vector<float> rm_h(rm.host_f32(),   rm.host_f32()    + C);
    std::vector<float> rv_h(rv.host_f32(),   rv.host_f32()    + C);

    Tensor Y, saved_mean, saved_rstd;
    brotensor::batch_norm_forward(X, gamma, beta, rm, rv,
                                  N, C, H, W, eps, momentum,
                                  Y, saved_mean, saved_rstd);

    CHECK(Y.rows == N);
    CHECK(Y.cols == C * spatial);
    CHECK(saved_mean.rows == C && saved_mean.cols == 1);
    CHECK(saved_rstd.rows == C && saved_rstd.cols == 1);

    std::vector<float> Y_ref, sm_ref, sr_ref, var_unb_ref;
    bn_forward_ref(X_h, g_h, b_h, N, C, H, W, eps,
                   Y_ref, sm_ref, sr_ref, var_unb_ref);

    const float* Yp = Y.host_f32();
    for (int i = 0; i < N * C * spatial; ++i) {
        if (!approx_eq(Yp[i], Y_ref[i], 1e-5f, 1e-5f)) {
            std::printf("    Y[%d] got=%g ref=%g\n", i, Yp[i], Y_ref[i]);
            CHECK(false);
            break;
        }
    }

    const float* sm = saved_mean.host_f32();
    const float* sr = saved_rstd.host_f32();
    for (int c = 0; c < C; ++c) {
        CHECK(approx_eq(sm[c], sm_ref[c], 1e-5f, 1e-5f));
        CHECK(approx_eq(sr[c], sr_ref[c], 1e-5f, 1e-5f));
    }

    // Running stats: (1 - momentum) * old + momentum * batch.
    const float* rmp = rm.host_f32();
    const float* rvp = rv.host_f32();
    for (int c = 0; c < C; ++c) {
        const float rm_exp = (1.0f - momentum) * rm_h[c] + momentum * sm_ref[c];
        const float rv_exp = (1.0f - momentum) * rv_h[c] + momentum * var_unb_ref[c];
        CHECK(approx_eq(rmp[c], rm_exp, 1e-5f, 1e-5f));
        CHECK(approx_eq(rvp[c], rv_exp, 1e-5f, 1e-5f));
    }
}

// ── 2. inference forward ──────────────────────────────────────────────────
static void test_inference_forward() {
    const int N = 2, C = 4, H = 3, W = 7;
    const float eps = 1e-5f;
    const int spatial = H * W;

    Tensor X = make_f32(N, C * spatial);
    Tensor gamma = make_f32(C, 1), beta = make_f32(C, 1);
    Tensor rm    = make_f32(C, 1), rv   = make_f32(C, 1);
    fill_random(X,     0xB1);
    fill_random(gamma, 0xB2, 0.5f, 1.5f);
    fill_random(beta,  0xB3, -0.3f, 0.3f);
    fill_random(rm,    0xB4);
    fill_random(rv,    0xB5,  0.5f, 2.0f);  // strictly positive

    std::vector<float> X_h(X.host_f32(), X.host_f32() + N * C * spatial);
    std::vector<float> g_h(gamma.host_f32(), gamma.host_f32() + C);
    std::vector<float> b_h(beta.host_f32(),  beta.host_f32()  + C);
    std::vector<float> rm_h(rm.host_f32(),   rm.host_f32()    + C);
    std::vector<float> rv_h(rv.host_f32(),   rv.host_f32()    + C);

    Tensor Y;
    brotensor::batch_norm_inference(X, gamma, beta, rm, rv,
                                    N, C, H, W, eps, Y);
    CHECK(Y.rows == N);
    CHECK(Y.cols == C * spatial);

    // Reference: y = (x - rm) / sqrt(rv + eps) * gamma + beta.
    const float* Yp = Y.host_f32();
    for (int c = 0; c < C; ++c) {
        const float inv = 1.0f / std::sqrt(rv_h[c] + eps);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                const int idx = ((n * C + c) * spatial) + s;
                const float ref = (X_h[idx] - rm_h[c]) * inv * g_h[c] + b_h[c];
                if (!approx_eq(Yp[idx], ref, 1e-5f, 1e-5f)) {
                    std::printf("    inf Y[%d] got=%g ref=%g\n",
                                idx, Yp[idx], ref);
                    CHECK(false);
                    return;
                }
            }
        }
    }

    // rm/rv must NOT have changed.
    const float* rmp = rm.host_f32();
    const float* rvp = rv.host_f32();
    for (int c = 0; c < C; ++c) {
        CHECK(rmp[c] == rm_h[c]);
        CHECK(rvp[c] == rv_h[c]);
    }
}

// ── 3. backward — dX / dGamma / dBeta + accumulation ──────────────────────
static void test_backward() {
    const int N = 3, C = 4, H = 5, W = 6;
    const float eps = 1e-5f;
    const int spatial = H * W;

    Tensor X = make_f32(N, C * spatial);
    Tensor dY = make_f32(N, C * spatial);
    Tensor gamma = make_f32(C, 1), beta = make_f32(C, 1);
    Tensor rm    = make_f32(C, 1), rv   = make_f32(C, 1);
    fill_random(X,     0xC1);
    fill_random(dY,    0xC2);
    fill_random(gamma, 0xC3, 0.5f, 1.5f);
    fill_random(beta,  0xC4, -0.3f, 0.3f);
    fill_constant(rm, 0.0f);
    fill_constant(rv, 1.0f);

    std::vector<float> X_h(X.host_f32(), X.host_f32() + N * C * spatial);
    std::vector<float> dY_h(dY.host_f32(), dY.host_f32() + N * C * spatial);
    std::vector<float> g_h(gamma.host_f32(), gamma.host_f32() + C);
    std::vector<float> b_h(beta.host_f32(),  beta.host_f32()  + C);

    Tensor Y, saved_mean, saved_rstd;
    brotensor::batch_norm_forward(X, gamma, beta, rm, rv,
                                  N, C, H, W, eps, 0.1f,
                                  Y, saved_mean, saved_rstd);

    std::vector<float> sm_h(saved_mean.host_f32(), saved_mean.host_f32() + C);
    std::vector<float> sr_h(saved_rstd.host_f32(), saved_rstd.host_f32() + C);

    // Seed dGamma / dBeta with non-zero values to verify accumulation.
    Tensor dGamma = make_f32(C, 1), dBeta = make_f32(C, 1);
    fill_constant(dGamma, 0.25f);
    fill_constant(dBeta,  -0.5f);
    std::vector<float> dG_pre(dGamma.host_f32(), dGamma.host_f32() + C);
    std::vector<float> dB_pre(dBeta.host_f32(),  dBeta.host_f32()  + C);

    Tensor dX;
    brotensor::batch_norm_backward(X, gamma, saved_mean, saved_rstd, dY,
                                   N, C, H, W, dX, dGamma, dBeta);
    CHECK(dX.rows == N && dX.cols == C * spatial);

    std::vector<float> dX_ref, dG_add, dB_add;
    bn_backward_ref(X_h, g_h, sm_h, sr_h, dY_h, N, C, H, W,
                    dX_ref, dG_add, dB_add);

    const float* dXp = dX.host_f32();
    for (int i = 0; i < N * C * spatial; ++i) {
        if (!approx_eq(dXp[i], dX_ref[i], 1e-4f, 1e-4f)) {
            std::printf("    dX[%d] got=%g ref=%g\n", i, dXp[i], dX_ref[i]);
            CHECK(false);
            return;
        }
    }

    const float* dGp = dGamma.host_f32();
    const float* dBp = dBeta.host_f32();
    for (int c = 0; c < C; ++c) {
        const float dG_exp = dG_pre[c] + dG_add[c];
        const float dB_exp = dB_pre[c] + dB_add[c];
        if (!approx_eq(dGp[c], dG_exp, 1e-4f, 1e-4f)) {
            std::printf("    dGamma[%d] got=%g ref=%g\n", c, dGp[c], dG_exp);
            CHECK(false);
        }
        if (!approx_eq(dBp[c], dB_exp, 1e-4f, 1e-4f)) {
            std::printf("    dBeta[%d] got=%g ref=%g\n", c, dBp[c], dB_exp);
            CHECK(false);
        }
    }

    // Suppress unused-variable warnings for snapshots used only for clarity.
    (void)b_h;
}

// ── 4. M == 1 (N=1, H=1, W=1) — Bessel correction degenerates gracefully ─
static void test_single_element() {
    const int N = 1, C = 3, H = 1, W = 1;
    Tensor X = make_f32(N, C);
    Tensor gamma = make_f32(C, 1), beta = make_f32(C, 1);
    Tensor rm    = make_f32(C, 1), rv   = make_f32(C, 1);
    fill_random(X,     0xD1);
    fill_constant(gamma, 1.0f);
    fill_constant(beta,  0.0f);
    fill_constant(rm,    0.0f);
    fill_constant(rv,    1.0f);

    Tensor Y, saved_mean, saved_rstd;
    // Must not throw / divide by zero (M-1==0).
    try {
        brotensor::batch_norm_forward(X, gamma, beta, rm, rv,
                                      N, C, H, W, 1e-5f, 0.1f,
                                      Y, saved_mean, saved_rstd);
    } catch (const std::exception& e) {
        std::printf("    unexpected throw on M==1: %s\n", e.what());
        CHECK(false);
        return;
    }
    CHECK(Y.rows == N && Y.cols == C);
    // With M==1: mean == x; (x - mean) == 0 ⇒ Y == beta == 0.
    const float* Yp = Y.host_f32();
    for (int c = 0; c < C; ++c) {
        CHECK(std::fabs(Yp[c]) < 1e-5f);
    }
}

int main() {
    brotensor::init();
    std::printf("test_batch_norm\n");

    test_training_forward();
    test_inference_forward();
    test_backward();
    test_single_element();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll batch_norm checks passed.\n");
    return 0;
}
