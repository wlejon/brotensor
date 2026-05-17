// CPU↔GPU parity for group_norm_forward_gpu (FP16).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::GpuTensor;
using brotensor::Dtype;

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

static void run_one(const char* label,
                    int N, int C, int H, int W, int num_groups) {
    std::printf("  %s  N=%d C=%d H=%d W=%d groups=%d\n",
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

    GpuTensor Xg, Gg, Bg, Yg;
    brotensor::upload_fp16(X_h16.data(),     N, C * spatial, Xg);
    brotensor::upload_fp16(gamma_h16.data(), C, 1,           Gg);
    brotensor::upload_fp16(beta_h16.data(),  C, 1,           Bg);

    brotensor::group_norm_forward_gpu(Xg, Gg, Bg, N, C, H, W,
                                      num_groups, 1e-5f, Yg);

    CHECK(Yg.rows == N);
    CHECK(Yg.cols == C * spatial);
    CHECK(Yg.dtype == Dtype::FP16);

    std::vector<uint16_t> Y_h16(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h16.data());
    brotensor::cuda_sync();

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

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_group_norm\n");

    run_one("tiny",       1, 4,  2, 2, 2);
    run_one("standard",   2, 8,  4, 4, 4);
    run_one("groups=1 ≡ layernorm-ish", 1, 6, 3, 3, 1);
    run_one("groups=C ≡ instancenorm-ish", 1, 4, 5, 5, 4);
    run_one("SD-ish",     1, 32, 8, 8, 8);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll group_norm checks passed.\n");
    return 0;
}
