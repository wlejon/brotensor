// CPU↔GPU parity for ddim_step_gpu (FP16).

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
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while(0)

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}
static std::vector<float> rq(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v[i]));
    return o;
}

static void test_basic() {
    std::printf("  ddim_step_gpu basic\n");
    const int R = 9, C = 17;
    const float alpha_t    = 0.6f;
    const float alpha_prev = 0.7f;
    const float sigma_t    = 0.0f;
    std::mt19937 rng(0x1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> xt(R * C), eps(R * C);
    for (auto& v : xt)  v = dist(rng);
    for (auto& v : eps) v = dist(rng);
    auto xtq = rq(xt), epsq = rq(eps);

    const float sqrt_at  = std::sqrt(alpha_t);
    const float sqrt_1ma = std::sqrt(1.0f - alpha_t);
    const float sqrt_ap  = std::sqrt(alpha_prev);
    const float dir_coef = std::sqrt(std::max(0.0f, 1.0f - alpha_prev - sigma_t * sigma_t));
    std::vector<float> ref(R * C);
    for (int i = 0; i < R * C; ++i) {
        const float x0 = (xtq[i] - sqrt_1ma * epsq[i]) / sqrt_at;
        ref[i] = sqrt_ap * x0 + dir_coef * epsq[i];
    }

    GpuTensor Xt, Eps, Xp;
    auto xth = to_fp16(xt), eh = to_fp16(eps);
    brotensor::upload_fp16(xth.data(), R, C, Xt);
    brotensor::upload_fp16(eh.data(),  R, C, Eps);
    brotensor::ddim_step_gpu(Xt, Eps, alpha_t, alpha_prev, sigma_t, Xp);
    CHECK(Xp.dtype == Dtype::FP16 && Xp.rows == R && Xp.cols == C);
    std::vector<uint16_t> got(R * C);
    brotensor::download_fp16(Xp, got.data());
    brotensor::cuda_sync();

    float max_err = 0.0f;
    int bad = 0;
    for (int i = 0; i < R * C; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 5e-3f + 5e-2f * std::fabs(ref[i])) ++bad;
    }
    std::printf("    max_err=%g bad=%d/%d\n", max_err, bad, R * C);
    CHECK(bad == 0);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_ddim_step\n");
    test_basic();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
