// CPU↔GPU parity for the SDXL-prep ops:
//   euler_step_gpu, dpmpp_2m_step_gpu, timestep_embedding_gpu.

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

static void test_euler_step() {
    std::printf("  euler_step_gpu\n");
    const int R = 7, C = 13;
    const float sigma_t    = 1.5f;
    const float sigma_prev = 1.1f;
    std::mt19937 rng(0xE1);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> xt(R * C), eps(R * C);
    for (auto& v : xt)  v = dist(rng);
    for (auto& v : eps) v = dist(rng);
    auto xtq = rq(xt), epsq = rq(eps);

    const float dsigma = sigma_prev - sigma_t;
    std::vector<float> ref(R * C);
    for (int i = 0; i < R * C; ++i) ref[i] = xtq[i] + dsigma * epsq[i];

    GpuTensor Xt, Eps, Xp;
    auto xth = to_fp16(xt), eh = to_fp16(eps);
    brotensor::upload_fp16(xth.data(), R, C, Xt);
    brotensor::upload_fp16(eh.data(),  R, C, Eps);
    brotensor::euler_step_gpu(Xt, Eps, sigma_t, sigma_prev, Xp);
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

static void test_dpmpp_2m_step() {
    std::printf("  dpmpp_2m_step_gpu\n");
    const int R = 9, C = 17;
    const float sigma_t  = 1.4f;
    const float c_xt     = 0.78f;     // sigma_next / sigma_t
    const float c_x0t    = 0.31f;
    const float c_x0prev = -0.07f;
    std::mt19937 rng(0xD2);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> xt(R * C), eps(R * C), x0p(R * C);
    for (auto& v : xt)  v = dist(rng);
    for (auto& v : eps) v = dist(rng);
    for (auto& v : x0p) v = dist(rng);
    auto xtq = rq(xt), epsq = rq(eps), x0pq = rq(x0p);

    std::vector<float> ref_xp(R * C), ref_x0(R * C);
    for (int i = 0; i < R * C; ++i) {
        const float x0t = xtq[i] - sigma_t * epsq[i];
        ref_x0[i] = x0t;
        ref_xp[i] = c_xt * xtq[i] + c_x0t * x0t + c_x0prev * x0pq[i];
    }

    GpuTensor Xt, Eps, X0p, Xp, X0o;
    auto xth = to_fp16(xt), eh = to_fp16(eps), x0ph = to_fp16(x0p);
    brotensor::upload_fp16(xth.data(),  R, C, Xt);
    brotensor::upload_fp16(eh.data(),   R, C, Eps);
    brotensor::upload_fp16(x0ph.data(), R, C, X0p);
    brotensor::dpmpp_2m_step_gpu(Xt, Eps, X0p, sigma_t, c_xt, c_x0t, c_x0prev, Xp, X0o);
    CHECK(Xp.dtype == Dtype::FP16 && Xp.rows == R && Xp.cols == C);
    CHECK(X0o.dtype == Dtype::FP16 && X0o.rows == R && X0o.cols == C);
    std::vector<uint16_t> got_xp(R * C), got_x0(R * C);
    brotensor::download_fp16(Xp,  got_xp.data());
    brotensor::download_fp16(X0o, got_x0.data());
    brotensor::cuda_sync();

    float max_err_xp = 0.0f, max_err_x0 = 0.0f;
    int bad = 0;
    for (int i = 0; i < R * C; ++i) {
        const float gxp = brotensor::fp16_bits_to_fp32(got_xp[i]);
        const float gx0 = brotensor::fp16_bits_to_fp32(got_x0[i]);
        const float e1 = std::fabs(gxp - ref_xp[i]);
        const float e2 = std::fabs(gx0 - ref_x0[i]);
        if (e1 > max_err_xp) max_err_xp = e1;
        if (e2 > max_err_x0) max_err_x0 = e2;
        if (e1 > 5e-3f + 5e-2f * std::fabs(ref_xp[i])) ++bad;
        if (e2 > 5e-3f + 5e-2f * std::fabs(ref_x0[i])) ++bad;
    }
    std::printf("    max_err xp=%g x0=%g bad=%d/%d\n",
                max_err_xp, max_err_x0, bad, 2 * R * C);
    CHECK(bad == 0);
}

static void test_timestep_embedding() {
    std::printf("  timestep_embedding_gpu\n");
    // Test both even and odd dim to exercise the tail-zero path.
    for (int dim : {16, 320, 17}) {
        const int N = 4;
        const float max_period = 10000.0f;
        std::vector<float> ts = {0.0f, 1.0f, 500.0f, 999.0f};

        const int half = dim / 2;
        const float log_mp = std::log(max_period);
        std::vector<float> ref(N * dim);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dim; ++j) {
                if (j >= 2 * half) { ref[i * dim + j] = 0.0f; continue; }
                const int k = j < half ? j : j - half;
                const float freq = std::exp(-log_mp * (float)k / (float)half);
                const float arg  = ts[i] * freq;
                ref[i * dim + j] = j < half ? std::cos(arg) : std::sin(arg);
            }
        }

        GpuTensor T, Y;
        brotensor::upload(ts.data(), N, 1, T);
        brotensor::timestep_embedding_gpu(T, dim, max_period, Y);
        CHECK(Y.dtype == Dtype::FP32 && Y.rows == N && Y.cols == dim);
        std::vector<float> got(N * dim);
        brotensor::download(Y, got.data());
        brotensor::cuda_sync();

        float max_err = 0.0f;
        int bad = 0;
        for (int i = 0; i < N * dim; ++i) {
            const float e = std::fabs(got[i] - ref[i]);
            if (e > max_err) max_err = e;
            if (e > 1e-4f) ++bad;
        }
        std::printf("    dim=%d max_err=%g bad=%d/%d\n", dim, max_err, bad, N * dim);
        CHECK(bad == 0);
    }
}

int main() {
    brotensor::cuda_init();
    std::printf("test_sdxl_schedulers\n");
    test_euler_step();
    test_dpmpp_2m_step();
    test_timestep_embedding();
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
