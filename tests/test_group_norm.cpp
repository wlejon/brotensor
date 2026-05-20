// CPU↔GPU parity for group_norm_forward_gpu (FP16 + FP32) and
// group_norm_backward_gpu (FP16 + FP32).

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
using brotensor::Device;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = brotensor::fp32_to_fp16_bits(src[i]);
    }
    return out;
}

static std::vector<float> requantize(const std::vector<float>& src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(src[i]));
    }
    return out;
}

static void group_norm_cpu(const std::vector<float>& X,
                           const std::vector<float>& gamma,
                           const std::vector<float>& beta,
                           int N, int C, int H, int W,
                           int num_groups, float eps,
                           std::vector<float>& Y) {
    const int spatial = H * W;
    const int cpg = C / num_groups;
    const int tile = cpg * spatial;
    Y.assign(static_cast<size_t>(N) * C * spatial, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            double sum = 0.0, sumsq = 0.0;
            const int chan_base = g * cpg;
            for (int c = 0; c < cpg; ++c) {
                for (int s = 0; s < spatial; ++s) {
                    const float v = X[((n * C + chan_base + c) * spatial) + s];
                    sum   += v;
                    sumsq += static_cast<double>(v) * v;
                }
            }
            const double inv_n = 1.0 / tile;
            const double mean = sum * inv_n;
            const double var  = sumsq * inv_n - mean * mean;
            const double rstd = 1.0 / std::sqrt(var + eps);
            for (int c = 0; c < cpg; ++c) {
                const int channel = chan_base + c;
                const float gv = gamma[channel];
                const float bv = beta[channel];
                for (int s = 0; s < spatial; ++s) {
                    const int idx = ((n * C + channel) * spatial) + s;
                    const double yn = (X[idx] - mean) * rstd;
                    Y[idx] = static_cast<float>(yn * gv + bv);
                }
            }
        }
    }
}

// CPU reference backward, mirrors the kernel math exactly. Computes mean/rstd
// from X per tile (no forward cache). dGamma/dBeta accumulated over batch.
static void group_norm_backward_cpu(const std::vector<float>& X,
                                    const std::vector<float>& gamma,
                                    const std::vector<float>& dY,
                                    int N, int C, int H, int W,
                                    int num_groups, float eps,
                                    std::vector<float>& dX,
                                    std::vector<float>& dGamma,
                                    std::vector<float>& dBeta) {
    const int spatial = H * W;
    const int cpg = C / num_groups;
    const int M = cpg * spatial;
    dX.assign(static_cast<size_t>(N) * C * spatial, 0.0f);
    dGamma.assign(C, 0.0f);
    dBeta.assign(C, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            const int chan_base = g * cpg;
            double sum = 0.0, sumsq = 0.0;
            for (int c = 0; c < cpg; ++c) {
                for (int s = 0; s < spatial; ++s) {
                    const float v = X[((n * C + chan_base + c) * spatial) + s];
                    sum   += v;
                    sumsq += static_cast<double>(v) * v;
                }
            }
            const double inv_n = 1.0 / M;
            const double mean = sum * inv_n;
            const double var  = sumsq * inv_n - mean * mean;
            const double rstd = 1.0 / std::sqrt(var + eps);

            // Pass 2: sum1, sum2 over tile + accumulate channel grads.
            double sum1 = 0.0, sum2 = 0.0;
            for (int c = 0; c < cpg; ++c) {
                const int channel = chan_base + c;
                const float gv = gamma[channel];
                for (int s = 0; s < spatial; ++s) {
                    const int idx = ((n * C + channel) * spatial) + s;
                    const double xh = (X[idx] - mean) * rstd;
                    const double dxh = dY[idx] * gv;
                    sum1 += dxh;
                    sum2 += dxh * xh;
                    dGamma[channel] += static_cast<float>(dY[idx] * xh);
                    dBeta[channel]  += dY[idx];
                }
            }
            // Pass 3.
            for (int c = 0; c < cpg; ++c) {
                const int channel = chan_base + c;
                const float gv = gamma[channel];
                for (int s = 0; s < spatial; ++s) {
                    const int idx = ((n * C + channel) * spatial) + s;
                    const double xh = (X[idx] - mean) * rstd;
                    const double dxh = dY[idx] * gv;
                    const double dx = rstd * (dxh - (sum1 + xh * sum2) / M);
                    dX[idx] = static_cast<float>(dx);
                }
            }
        }
    }
}

// ─── FP16 forward parity (existing) ────────────────────────────────────────
static void run_fwd_fp16(const char* label,
                         int N, int C, int H, int W, int num_groups) {
    std::printf("  fp16 fwd  %s  N=%d C=%d H=%d W=%d groups=%d\n",
                label, N, C, H, W, num_groups);
    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int spatial = H * W;
    std::vector<float> X(static_cast<size_t>(N) * C * spatial);
    std::vector<float> gamma(C), beta(C);
    for (auto& v : X)     v = dist(rng);
    for (auto& v : gamma) v = 0.5f + dist(rng) * 0.5f;
    for (auto& v : beta)  v = dist(rng) * 0.2f;

    auto X_q     = requantize(X);
    auto gamma_q = requantize(gamma);
    auto beta_q  = requantize(beta);

    std::vector<float> Y_cpu;
    group_norm_cpu(X_q, gamma_q, beta_q,
                   N, C, H, W, num_groups, 1e-5f, Y_cpu);

    auto X_h16     = to_fp16(X);
    auto gamma_h16 = to_fp16(gamma);
    auto beta_h16  = to_fp16(beta);

    Tensor Xg, Gg, Bg, Yg;
    Xg = Tensor::from_host_fp16_on(Device::CUDA, X_h16.data(),     N, C * spatial);
    Gg = Tensor::from_host_fp16_on(Device::CUDA, gamma_h16.data(), C, 1);
    Bg = Tensor::from_host_fp16_on(Device::CUDA, beta_h16.data(),  C, 1);

    brotensor::group_norm_forward(Xg, Gg, Bg, N, C, H, W,
                                  num_groups, 1e-5f, Yg);

    CHECK(Yg.rows == N);
    CHECK(Yg.cols == C * spatial);
    CHECK(Yg.dtype == Dtype::FP16);

    std::vector<uint16_t> Y_h16(static_cast<size_t>(Yg.size()), 0);
    Yg.copy_to_host_fp16(Y_h16.data());
    brotensor::sync_all();

    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < Y_cpu.size(); ++i) {
        const float got = brotensor::fp16_bits_to_fp32(Y_h16[i]);
        const float ref = Y_cpu[i];
        const float err = std::fabs(got - ref);
        if (err > max_err) max_err = err;
        const float tol = 1e-2f + 1e-2f * std::fabs(ref);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, got, ref, err);
            }
            ++bad;
        }
    }
    std::printf("    max_err=%g  bad=%d / %zu\n", max_err, bad, Y_cpu.size());
    CHECK(bad == 0);
}

// ─── FP32 forward parity ───────────────────────────────────────────────────
static void run_fwd_fp32(const char* label,
                         int N, int C, int H, int W, int num_groups) {
    std::printf("  fp32 fwd  %s  N=%d C=%d H=%d W=%d groups=%d\n",
                label, N, C, H, W, num_groups);
    std::mt19937 rng(0xC0DE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int spatial = H * W;
    std::vector<float> X(static_cast<size_t>(N) * C * spatial);
    std::vector<float> gamma(C), beta(C);
    for (auto& v : X)     v = dist(rng);
    for (auto& v : gamma) v = 0.5f + dist(rng) * 0.5f;
    for (auto& v : beta)  v = dist(rng) * 0.2f;

    std::vector<float> Y_cpu;
    group_norm_cpu(X, gamma, beta, N, C, H, W, num_groups, 1e-5f, Y_cpu);

    Tensor Xg, Gg, Bg, Yg;
    Xg = Tensor::from_host_on(Device::CUDA, X.data(),     N, C * spatial);
    Gg = Tensor::from_host_on(Device::CUDA, gamma.data(), C, 1);
    Bg = Tensor::from_host_on(Device::CUDA, beta.data(),  C, 1);

    brotensor::group_norm_forward(Xg, Gg, Bg, N, C, H, W,
                                  num_groups, 1e-5f, Yg);

    CHECK(Yg.rows == N);
    CHECK(Yg.cols == C * spatial);
    CHECK(Yg.dtype == Dtype::FP32);

    std::vector<float> got(static_cast<size_t>(Yg.size()), 0.0f);
    Yg.copy_to_host(got.data());
    brotensor::sync_all();

    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < Y_cpu.size(); ++i) {
        const float err = std::fabs(got[i] - Y_cpu[i]);
        if (err > max_err) max_err = err;
        const float tol = 1e-4f + 1e-4f * std::fabs(Y_cpu[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, got[i], Y_cpu[i], err);
            }
            ++bad;
        }
    }
    std::printf("    max_err=%g  bad=%d / %zu\n", max_err, bad, Y_cpu.size());
    CHECK(bad == 0);
}

// ─── FP32 backward parity ──────────────────────────────────────────────────
static void run_bwd_fp32(const char* label,
                         int N, int C, int H, int W, int num_groups) {
    std::printf("  fp32 bwd  %s  N=%d C=%d H=%d W=%d groups=%d\n",
                label, N, C, H, W, num_groups);
    std::mt19937 rng(0xD0D0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int spatial = H * W;
    std::vector<float> X(static_cast<size_t>(N) * C * spatial);
    std::vector<float> dY(static_cast<size_t>(N) * C * spatial);
    std::vector<float> gamma(C);
    for (auto& v : X)     v = dist(rng);
    for (auto& v : dY)    v = dist(rng);
    for (auto& v : gamma) v = 0.5f + dist(rng) * 0.5f;

    std::vector<float> dX_cpu, dG_cpu, dB_cpu;
    group_norm_backward_cpu(X, gamma, dY, N, C, H, W, num_groups, 1e-5f,
                            dX_cpu, dG_cpu, dB_cpu);

    Tensor Xg, Gg, dYg, dXg, dGg, dBg;
    Xg  = Tensor::from_host_on(Device::CUDA, X.data(),     N, C * spatial);
    Gg  = Tensor::from_host_on(Device::CUDA, gamma.data(), C, 1);
    dYg = Tensor::from_host_on(Device::CUDA, dY.data(),    N, C * spatial);
    // Zero-initialize accumulators (caller responsibility).
    std::vector<float> zeros(C, 0.0f);
    dGg = Tensor::from_host_on(Device::CUDA, zeros.data(), C, 1);
    dBg = Tensor::from_host_on(Device::CUDA, zeros.data(), C, 1);

    brotensor::group_norm_backward(Xg, Gg, dYg, N, C, H, W,
                                   num_groups, 1e-5f,
                                   dXg, dGg, dBg);

    CHECK(dXg.rows == N);
    CHECK(dXg.cols == C * spatial);
    CHECK(dXg.dtype == Dtype::FP32);

    std::vector<float> dX_got(static_cast<size_t>(dXg.size()), 0.0f);
    std::vector<float> dG_got(C, 0.0f), dB_got(C, 0.0f);
    dXg.copy_to_host(dX_got.data());
    dGg.copy_to_host(dG_got.data());
    dBg.copy_to_host(dB_got.data());
    brotensor::sync_all();

    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < dX_cpu.size(); ++i) {
        const float err = std::fabs(dX_got[i] - dX_cpu[i]);
        if (err > max_err) max_err = err;
        const float tol = 1e-4f + 1e-4f * std::fabs(dX_cpu[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    dX mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, dX_got[i], dX_cpu[i], err);
            }
            ++bad;
        }
    }
    std::printf("    dX max_err=%g  bad=%d / %zu\n", max_err, bad, dX_cpu.size());
    CHECK(bad == 0);

    int bad_g = 0; float max_err_g = 0.0f;
    int bad_b = 0; float max_err_b = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float eg = std::fabs(dG_got[c] - dG_cpu[c]);
        const float eb = std::fabs(dB_got[c] - dB_cpu[c]);
        if (eg > max_err_g) max_err_g = eg;
        if (eb > max_err_b) max_err_b = eb;
        // Accumulated sums over batch can be larger; scale tol with magnitude.
        const float tol_g = 1e-3f + 1e-4f * std::fabs(dG_cpu[c]);
        const float tol_b = 1e-3f + 1e-4f * std::fabs(dB_cpu[c]);
        if (eg > tol_g) ++bad_g;
        if (eb > tol_b) ++bad_b;
    }
    std::printf("    dGamma max_err=%g bad=%d / %d\n", max_err_g, bad_g, C);
    std::printf("    dBeta  max_err=%g bad=%d / %d\n", max_err_b, bad_b, C);
    CHECK(bad_g == 0);
    CHECK(bad_b == 0);
}

// ─── FP16 backward parity ──────────────────────────────────────────────────
static void run_bwd_fp16(const char* label,
                         int N, int C, int H, int W, int num_groups) {
    std::printf("  fp16 bwd  %s  N=%d C=%d H=%d W=%d groups=%d\n",
                label, N, C, H, W, num_groups);
    std::mt19937 rng(0xD1D1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int spatial = H * W;
    std::vector<float> X(static_cast<size_t>(N) * C * spatial);
    std::vector<float> dY(static_cast<size_t>(N) * C * spatial);
    std::vector<float> gamma(C);
    for (auto& v : X)     v = dist(rng);
    for (auto& v : dY)    v = dist(rng);
    for (auto& v : gamma) v = 0.5f + dist(rng) * 0.5f;

    // CPU ref runs in FP32 against the FP16-quantized inputs (so we compare
    // against what the kernel actually sees).
    auto X_q     = requantize(X);
    auto gamma_q = requantize(gamma);
    auto dY_q    = requantize(dY);
    std::vector<float> dX_cpu, dG_cpu, dB_cpu;
    group_norm_backward_cpu(X_q, gamma_q, dY_q, N, C, H, W, num_groups, 1e-5f,
                            dX_cpu, dG_cpu, dB_cpu);

    auto X_h16     = to_fp16(X);
    auto gamma_h16 = to_fp16(gamma);
    auto dY_h16    = to_fp16(dY);

    Tensor Xg, Gg, dYg, dXg, dGg, dBg;
    Xg  = Tensor::from_host_fp16_on(Device::CUDA, X_h16.data(),     N, C * spatial);
    Gg  = Tensor::from_host_fp16_on(Device::CUDA, gamma_h16.data(), C, 1);
    dYg = Tensor::from_host_fp16_on(Device::CUDA, dY_h16.data(),    N, C * spatial);
    std::vector<uint16_t> zeros_h(C, brotensor::fp32_to_fp16_bits(0.0f));
    dGg = Tensor::from_host_fp16_on(Device::CUDA, zeros_h.data(), C, 1);
    dBg = Tensor::from_host_fp16_on(Device::CUDA, zeros_h.data(), C, 1);

    brotensor::group_norm_backward(Xg, Gg, dYg, N, C, H, W,
                                   num_groups, 1e-5f,
                                   dXg, dGg, dBg);

    CHECK(dXg.rows == N);
    CHECK(dXg.cols == C * spatial);
    CHECK(dXg.dtype == Dtype::FP16);

    std::vector<uint16_t> dX_got_h(static_cast<size_t>(dXg.size()), 0);
    std::vector<uint16_t> dG_got_h(C, 0), dB_got_h(C, 0);
    dXg.copy_to_host_fp16(dX_got_h.data());
    dGg.copy_to_host_fp16(dG_got_h.data());
    dBg.copy_to_host_fp16(dB_got_h.data());
    brotensor::sync_all();

    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < dX_cpu.size(); ++i) {
        const float got = brotensor::fp16_bits_to_fp32(dX_got_h[i]);
        const float err = std::fabs(got - dX_cpu[i]);
        if (err > max_err) max_err = err;
        const float tol = 1e-2f + 1e-2f * std::fabs(dX_cpu[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    dX mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, got, dX_cpu[i], err);
            }
            ++bad;
        }
    }
    std::printf("    dX max_err=%g  bad=%d / %zu\n", max_err, bad, dX_cpu.size());
    CHECK(bad == 0);

    int bad_g = 0; float max_err_g = 0.0f;
    int bad_b = 0; float max_err_b = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float gg = brotensor::fp16_bits_to_fp32(dG_got_h[c]);
        const float bb = brotensor::fp16_bits_to_fp32(dB_got_h[c]);
        const float eg = std::fabs(gg - dG_cpu[c]);
        const float eb = std::fabs(bb - dB_cpu[c]);
        if (eg > max_err_g) max_err_g = eg;
        if (eb > max_err_b) max_err_b = eb;
        // FP16 accumulators across N*H*W elements lose precision; allow a
        // fairly loose absolute tol scaled with magnitude.
        const float tol_g = 5e-2f + 1e-2f * std::fabs(dG_cpu[c]);
        const float tol_b = 5e-2f + 1e-2f * std::fabs(dB_cpu[c]);
        if (eg > tol_g) ++bad_g;
        if (eb > tol_b) ++bad_b;
    }
    std::printf("    dGamma max_err=%g bad=%d / %d\n", max_err_g, bad_g, C);
    std::printf("    dBeta  max_err=%g bad=%d / %d\n", max_err_b, bad_b, C);
    CHECK(bad_g == 0);
    CHECK(bad_b == 0);
}

int main() {
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    std::printf("test_group_norm\n");

    // FP16 forward (regression).
    run_fwd_fp16("tiny",       1, 4,  2, 2, 2);
    run_fwd_fp16("standard",   2, 8,  4, 4, 4);
    run_fwd_fp16("groups=1",   1, 6,  3, 3, 1);
    run_fwd_fp16("groups=C",   1, 4,  5, 5, 4);
    run_fwd_fp16("SD-ish",     1, 32, 8, 8, 8);

    // FP32 forward (new path).
    run_fwd_fp32("tiny",       1, 4,  2, 2, 2);
    run_fwd_fp32("standard",   2, 8,  4, 4, 4);
    run_fwd_fp32("groups=1",   1, 6,  3, 3, 1);
    run_fwd_fp32("groups=C",   1, 4,  5, 5, 4);
    run_fwd_fp32("SD-ish",     1, 32, 8, 8, 8);

    // FP32 backward.
    run_bwd_fp32("tiny",       1, 4,  2, 2, 2);
    run_bwd_fp32("standard",   2, 8,  4, 4, 4);
    run_bwd_fp32("groups=1",   1, 6,  3, 3, 1);
    run_bwd_fp32("groups=C",   1, 4,  5, 5, 4);
    run_bwd_fp32("SD-ish",     1, 32, 8, 8, 8);

    // FP16 backward.
    run_bwd_fp16("tiny",       1, 4,  2, 2, 2);
    run_bwd_fp16("standard",   2, 8,  4, 4, 4);
    run_bwd_fp16("groups=1",   1, 6,  3, 3, 1);
    run_bwd_fp16("groups=C",   1, 4,  5, 5, 4);
    run_bwd_fp16("SD-ish",     1, 32, 8, 8, 8);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll group_norm checks passed.\n");
    return 0;
}
