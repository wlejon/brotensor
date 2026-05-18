// CPU↔GPU parity for rms_norm_forward_gpu / rms_norm_backward_gpu.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void rms_cpu_fwd(const std::vector<float>& X,
                        const std::vector<float>& gamma,
                        std::vector<float>& Y,
                        int B, int D, float eps) {
    Y.assign(B * D, 0.0f);
    for (int b = 0; b < B; ++b) {
        double ss = 0.0;
        for (int j = 0; j < D; ++j) ss += static_cast<double>(X[b * D + j]) *
                                         static_cast<double>(X[b * D + j]);
        const float rrms = 1.0f / std::sqrt(static_cast<float>(ss / D) + eps);
        for (int j = 0; j < D; ++j)
            Y[b * D + j] = X[b * D + j] * gamma[j] * rrms;
    }
}

static void rms_cpu_bwd(const std::vector<float>& X,
                        const std::vector<float>& gamma,
                        const std::vector<float>& dY,
                        std::vector<float>& dX,
                        std::vector<float>& dGamma,
                        int B, int D, float eps) {
    dX.assign(B * D, 0.0f);
    dGamma.assign(D, 0.0f);
    for (int b = 0; b < B; ++b) {
        double ss = 0.0;
        for (int j = 0; j < D; ++j) ss += static_cast<double>(X[b * D + j]) *
                                         static_cast<double>(X[b * D + j]);
        const float rrms = 1.0f / std::sqrt(static_cast<float>(ss / D) + eps);
        double sum_xdy = 0.0;
        for (int j = 0; j < D; ++j)
            sum_xdy += static_cast<double>(X[b*D+j]) * dY[b*D+j] * gamma[j];
        const float coeff = static_cast<float>(sum_xdy) * rrms * rrms /
                            static_cast<float>(D);
        for (int j = 0; j < D; ++j) {
            dX[b*D+j] = rrms * (gamma[j] * dY[b*D+j] - X[b*D+j] * coeff);
            dGamma[j] += dY[b*D+j] * X[b*D+j] * rrms;
        }
    }
}

static void test_fp32() {
    std::printf("  rms fp32 fwd+bwd\n");
    const int B = 4, D = 17;
    const float eps = 1e-5f;
    std::mt19937 rng(0x99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(B * D), gamma(D), dY(B * D);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : gamma) v = 0.5f + dist(rng) * 0.2f;
    for (auto& v : dY)   v = dist(rng);

    std::vector<float> Yref, dXref, dGref;
    rms_cpu_fwd(X, gamma, Yref, B, D, eps);
    rms_cpu_bwd(X, gamma, dY, dXref, dGref, B, D, eps);

    GpuTensor Xg, Gg, Yg, dYg, dXg, dGg;
    brotensor::upload(X.data(), B, D, Xg);
    brotensor::upload(gamma.data(), D, 1, Gg);
    brotensor::rms_norm_forward_gpu(Xg, Gg, eps, Yg);
    std::vector<float> got(Yg.size());
    brotensor::download(Yg, got.data());
    brotensor::cuda_sync();
    float me = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
        me = std::max(me, std::fabs(got[i] - Yref[i]));
    std::printf("    fwd max_err=%g\n", me);
    CHECK(me < 1e-4f);

    brotensor::upload(dY.data(), B, D, dYg);
    dGg.resize(D, 1);
    dGg.zero();
    brotensor::rms_norm_backward_gpu(Xg, Gg, dYg, eps, dXg, dGg);
    std::vector<float> got_dx(dXg.size()), got_dg(dGg.size());
    brotensor::download(dXg, got_dx.data());
    brotensor::download(dGg, got_dg.data());
    brotensor::cuda_sync();
    float me_x = 0.0f, me_g = 0.0f;
    for (size_t i = 0; i < got_dx.size(); ++i)
        me_x = std::max(me_x, std::fabs(got_dx[i] - dXref[i]));
    for (size_t i = 0; i < got_dg.size(); ++i)
        me_g = std::max(me_g, std::fabs(got_dg[i] - dGref[i]));
    std::printf("    bwd dX max_err=%g, dGamma max_err=%g\n", me_x, me_g);
    CHECK(me_x < 1e-4f);
    CHECK(me_g < 1e-3f);
}

static void test_fp16() {
    std::printf("  rms fp16 fwd+bwd\n");
    const int B = 3, D = 32;
    const float eps = 1e-5f;
    std::mt19937 rng(0x9A);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> X(B * D), gamma(D), dY(B * D);
    for (auto& v : X)     v = dist(rng);
    for (auto& v : gamma) v = 0.5f + dist(rng) * 0.1f;
    for (auto& v : dY)    v = dist(rng);

    std::vector<float> Yref, dXref, dGref;
    rms_cpu_fwd(X, gamma, Yref, B, D, eps);
    rms_cpu_bwd(X, gamma, dY, dXref, dGref, B, D, eps);

    GpuTensor Xg, Gg, Yg, dYg, dXg, dGg;
    auto Xh = to_fp16(X), Gh = to_fp16(gamma), dYh = to_fp16(dY);
    brotensor::upload_fp16(Xh.data(), B, D, Xg);
    brotensor::upload_fp16(Gh.data(), D, 1, Gg);
    brotensor::rms_norm_forward_gpu(Xg, Gg, eps, Yg);
    std::vector<uint16_t> got(Yg.size());
    brotensor::download_fp16(Yg, got.data());
    brotensor::cuda_sync();
    float me = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        me = std::max(me, std::fabs(g - Yref[i]));
    }
    std::printf("    fwd fp16 max_err=%g\n", me);
    CHECK(me < 2e-2f);

    brotensor::upload_fp16(dYh.data(), B, D, dYg);
    dGg.resize(D, 1, Dtype::FP16);
    dGg.zero();
    brotensor::rms_norm_backward_gpu(Xg, Gg, dYg, eps, dXg, dGg);
    std::vector<uint16_t> got_dx(dXg.size()), got_dg(dGg.size());
    brotensor::download_fp16(dXg, got_dx.data());
    brotensor::download_fp16(dGg, got_dg.data());
    brotensor::cuda_sync();
    float me_x = 0.0f, me_g = 0.0f;
    for (size_t i = 0; i < got_dx.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_dx[i]);
        me_x = std::max(me_x, std::fabs(g - dXref[i]));
    }
    for (size_t i = 0; i < got_dg.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_dg[i]);
        me_g = std::max(me_g, std::fabs(g - dGref[i]));
    }
    std::printf("    bwd fp16 dX_err=%g dG_err=%g\n", me_x, me_g);
    CHECK(me_x < 5e-2f);
    CHECK(me_g < 5e-2f);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_rms_norm\n");
    test_fp32();
    test_fp16();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
