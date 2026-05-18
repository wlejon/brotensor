// CPU<->GPU parity for attention_token_moments_gpu.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
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
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

static void run_case(const char* label,
                     int h_lat, int w_lat, int Lk,
                     const std::vector<float>& Attn_host,
                     float mass_atol, float centroid_atol) {
    const int Lq = h_lat * w_lat;
    std::printf("  %s  h=%d w=%d Lk=%d\n", label, h_lat, w_lat, Lk);

    auto Attn_fp16 = to_fp16(Attn_host);
    GpuTensor Ag, Mg, Cg;
    brotensor::upload_fp16(Attn_fp16.data(), Lq, Lk, Ag);

    brotensor::attention_token_moments_gpu(Ag, h_lat, w_lat, Mg, Cg);
    CHECK(Mg.rows == Lk && Mg.cols == 1 && Mg.dtype == Dtype::FP32);
    CHECK(Cg.rows == Lk && Cg.cols == 2 && Cg.dtype == Dtype::FP32);

    std::vector<float> mass_gpu(static_cast<size_t>(Lk), 0.0f);
    std::vector<float> cent_gpu(static_cast<size_t>(Lk) * 2, 0.0f);
    brotensor::download(Mg, mass_gpu.data());
    brotensor::download(Cg, cent_gpu.data());
    brotensor::cuda_sync();

    // CPU reference using FP16-rounded inputs (parity with GPU).
    std::vector<float> mass_ref(Lk, 0.0f);
    std::vector<float> cent_ref(static_cast<size_t>(Lk) * 2, 0.0f);
    for (int k = 0; k < Lk; ++k) {
        double m = 0.0, ay = 0.0, ax = 0.0;
        for (int q = 0; q < Lq; ++q) {
            const float a = brotensor::fp16_bits_to_fp32(Attn_fp16[q * Lk + k]);
            const int y = q / w_lat;
            const int x = q - y * w_lat;
            m += a;
            ay += static_cast<double>(y) * a;
            ax += static_cast<double>(x) * a;
        }
        mass_ref[k] = static_cast<float>(m);
        if (m > 1e-8) {
            cent_ref[k * 2 + 0] = static_cast<float>(ay / m);
            cent_ref[k * 2 + 1] = static_cast<float>(ax / m);
        } else {
            cent_ref[k * 2 + 0] = 0.0f;
            cent_ref[k * 2 + 1] = 0.0f;
        }
    }

    float max_mass_err = 0.0f;
    float max_cent_err = 0.0f;
    int bad = 0;
    for (int k = 0; k < Lk; ++k) {
        const float me = std::fabs(mass_gpu[k] - mass_ref[k]);
        max_mass_err = std::max(max_mass_err, me);
        if (me > mass_atol) {
            if (bad < 5)
                std::printf("    mass mismatch k=%d got=%g ref=%g err=%g\n",
                            k, mass_gpu[k], mass_ref[k], me);
            ++bad;
        }
        for (int a = 0; a < 2; ++a) {
            const float e = std::fabs(cent_gpu[k * 2 + a] - cent_ref[k * 2 + a]);
            max_cent_err = std::max(max_cent_err, e);
            if (e > centroid_atol) {
                if (bad < 5)
                    std::printf("    centroid mismatch k=%d axis=%d got=%g ref=%g err=%g\n",
                                k, a, cent_gpu[k * 2 + a], cent_ref[k * 2 + a], e);
                ++bad;
            }
        }
    }
    std::printf("    max_mass_err=%g max_centroid_err=%g bad=%d\n",
                max_mass_err, max_cent_err, bad);
    CHECK(bad == 0);
}

static void test_uniform_single_token() {
    const int h_lat = 8, w_lat = 8, Lk = 4;
    const int Lq = h_lat * w_lat;
    std::vector<float> A(static_cast<size_t>(Lq) * Lk, 0.0f);
    const float u = 1.0f / static_cast<float>(Lq);
    for (int q = 0; q < Lq; ++q) A[q * Lk + 0] = u;

    auto Afp16 = to_fp16(A);
    GpuTensor Ag, Mg, Cg;
    brotensor::upload_fp16(Afp16.data(), Lq, Lk, Ag);
    brotensor::attention_token_moments_gpu(Ag, h_lat, w_lat, Mg, Cg);
    std::vector<float> mg(Lk), cg(Lk * 2);
    brotensor::download(Mg, mg.data());
    brotensor::download(Cg, cg.data());
    brotensor::cuda_sync();

    std::printf("  uniform-single-token\n");
    CHECK(std::fabs(mg[0] - 1.0f) < 1e-3f);
    const float cy_exp = (h_lat - 1) * 0.5f;
    const float cx_exp = (w_lat - 1) * 0.5f;
    std::printf("    mass[0]=%g cy=%g cx=%g (exp %g, %g)\n",
                mg[0], cg[0], cg[1], cy_exp, cx_exp);
    CHECK(std::fabs(cg[0] - cy_exp) < 5e-3f);
    CHECK(std::fabs(cg[1] - cx_exp) < 5e-3f);
    for (int k = 1; k < Lk; ++k) {
        CHECK(std::fabs(mg[k]) < 1e-6f);
        CHECK(std::fabs(cg[k * 2 + 0]) < 1e-6f);
        CHECK(std::fabs(cg[k * 2 + 1]) < 1e-6f);
    }
}

static void test_point_mass() {
    const int h_lat = 8, w_lat = 16, Lk = 3;
    const int Lq = h_lat * w_lat;
    const int y_star = 5, x_star = 11, k_star = 2;
    const int q_star = y_star * w_lat + x_star;
    std::vector<float> A(static_cast<size_t>(Lq) * Lk, 0.0f);
    A[q_star * Lk + k_star] = 1.0f;

    auto Afp16 = to_fp16(A);
    GpuTensor Ag, Mg, Cg;
    brotensor::upload_fp16(Afp16.data(), Lq, Lk, Ag);
    brotensor::attention_token_moments_gpu(Ag, h_lat, w_lat, Mg, Cg);
    std::vector<float> mg(Lk), cg(Lk * 2);
    brotensor::download(Mg, mg.data());
    brotensor::download(Cg, cg.data());
    brotensor::cuda_sync();

    std::printf("  point-mass  y*=%d x*=%d k*=%d\n", y_star, x_star, k_star);
    CHECK(std::fabs(mg[k_star] - 1.0f) < 1e-3f);
    CHECK(std::fabs(cg[k_star * 2 + 0] - y_star) < 1e-4f);
    CHECK(std::fabs(cg[k_star * 2 + 1] - x_star) < 1e-4f);
    for (int k = 0; k < Lk; ++k) if (k != k_star) {
        CHECK(std::fabs(mg[k]) < 1e-6f);
        CHECK(std::fabs(cg[k * 2 + 0]) < 1e-6f);
        CHECK(std::fabs(cg[k * 2 + 1]) < 1e-6f);
    }
}

static void test_two_token_separation() {
    const int h_lat = 16, w_lat = 16, Lk = 2;
    const int Lq = h_lat * w_lat;
    std::vector<float> A(static_cast<size_t>(Lq) * Lk, 0.0f);
    // token 0: top-left quadrant; token 1: bottom-right quadrant.
    int n0 = 0, n1 = 0;
    for (int y = 0; y < h_lat; ++y)
        for (int x = 0; x < w_lat; ++x) {
            if (y < h_lat / 2 && x < w_lat / 2) ++n0;
            if (y >= h_lat / 2 && x >= w_lat / 2) ++n1;
        }
    for (int y = 0; y < h_lat; ++y)
        for (int x = 0; x < w_lat; ++x) {
            const int q = y * w_lat + x;
            if (y < h_lat / 2 && x < w_lat / 2)
                A[q * Lk + 0] = 1.0f / static_cast<float>(n0);
            if (y >= h_lat / 2 && x >= w_lat / 2)
                A[q * Lk + 1] = 1.0f / static_cast<float>(n1);
        }

    auto Afp16 = to_fp16(A);
    GpuTensor Ag, Mg, Cg;
    brotensor::upload_fp16(Afp16.data(), Lq, Lk, Ag);
    brotensor::attention_token_moments_gpu(Ag, h_lat, w_lat, Mg, Cg);
    std::vector<float> mg(Lk), cg(Lk * 2);
    brotensor::download(Mg, mg.data());
    brotensor::download(Cg, cg.data());
    brotensor::cuda_sync();

    std::printf("  two-token-separation  c0=(%g,%g) c1=(%g,%g)\n",
                cg[0], cg[1], cg[2], cg[3]);
    CHECK(std::fabs(mg[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(mg[1] - 1.0f) < 1e-3f);
    CHECK(cg[0 * 2 + 0] < cg[1 * 2 + 0] - 1.0f);  // y separation
    CHECK(cg[0 * 2 + 1] < cg[1 * 2 + 1] - 1.0f);  // x separation
}

static void test_zero_mass_token() {
    const int h_lat = 4, w_lat = 4, Lk = 3;
    const int Lq = h_lat * w_lat;
    std::vector<float> A(static_cast<size_t>(Lq) * Lk, 0.0f);
    // token 1 has all zeros; token 0 has uniform; token 2 has point mass.
    for (int q = 0; q < Lq; ++q) A[q * Lk + 0] = 1.0f / static_cast<float>(Lq);
    A[(2 * w_lat + 1) * Lk + 2] = 0.5f;

    auto Afp16 = to_fp16(A);
    GpuTensor Ag, Mg, Cg;
    brotensor::upload_fp16(Afp16.data(), Lq, Lk, Ag);
    brotensor::attention_token_moments_gpu(Ag, h_lat, w_lat, Mg, Cg);
    std::vector<float> mg(Lk), cg(Lk * 2);
    brotensor::download(Mg, mg.data());
    brotensor::download(Cg, cg.data());
    brotensor::cuda_sync();

    std::printf("  zero-mass-token  mass=(%g,%g,%g)\n", mg[0], mg[1], mg[2]);
    CHECK(std::fabs(mg[1]) < 1e-6f);
    CHECK(std::fabs(cg[1 * 2 + 0]) < 1e-6f);
    CHECK(std::fabs(cg[1 * 2 + 1]) < 1e-6f);
    CHECK(std::isfinite(cg[1 * 2 + 0]));
    CHECK(std::isfinite(cg[1 * 2 + 1]));
}

static void test_realistic_random() {
    const int h_lat = 32, w_lat = 32, Lk = 77;
    const int Lq = h_lat * w_lat;
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Per-row (per query) softmax-like distribution: roll positive scores,
    // normalise along k. Mirrors what a real attention map looks like.
    std::vector<float> A(static_cast<size_t>(Lq) * Lk, 0.0f);
    for (int q = 0; q < Lq; ++q) {
        float s = 0.0f;
        for (int k = 0; k < Lk; ++k) {
            const float v = dist(rng);
            A[q * Lk + k] = v;
            s += v;
        }
        const float inv = 1.0f / s;
        for (int k = 0; k < Lk; ++k) A[q * Lk + k] *= inv;
    }

    run_case("realistic-random  Lq=1024 Lk=77",
             h_lat, w_lat, Lk, A,
             /*mass_atol=*/2e-1f,   // 1024 FP16 sums accumulate noise
             /*centroid_atol=*/2e-1f);

    // Additional finiteness sweep.
    auto Afp16 = to_fp16(A);
    GpuTensor Ag, Mg, Cg;
    brotensor::upload_fp16(Afp16.data(), Lq, Lk, Ag);
    brotensor::attention_token_moments_gpu(Ag, h_lat, w_lat, Mg, Cg);
    std::vector<float> mg(Lk), cg(Lk * 2);
    brotensor::download(Mg, mg.data());
    brotensor::download(Cg, cg.data());
    brotensor::cuda_sync();
    for (int k = 0; k < Lk; ++k) {
        CHECK(std::isfinite(mg[k]));
        CHECK(std::isfinite(cg[k * 2 + 0]));
        CHECK(std::isfinite(cg[k * 2 + 1]));
    }
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_attention_moments\n");

    test_uniform_single_token();
    test_point_mass();
    test_two_token_separation();
    test_zero_mass_token();
    test_realistic_random();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll attention_moments checks passed.\n");
    return 0;
}
