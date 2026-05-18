// CPU↔GPU parity for swiglu_forward_gpu / swiglu_backward_gpu.

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

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
static inline float silu_grad(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return s * (1.0f + x * (1.0f - s));
}

static void test_fp32() {
    std::printf("  swiglu fp32 fwd+bwd\n");
    const int B = 4, D = 7;
    std::mt19937 rng(0x100);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(B * 2 * D), dY(B * D);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> Yref(B * D), dXref(B * 2 * D, 0.0f);
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d) {
            const float a  = X[b * 2 * D + d];
            const float bh = X[b * 2 * D + D + d];
            Yref[b * D + d] = silu(a) * bh;
            const float dy  = dY[b * D + d];
            dXref[b * 2 * D + d]     = dy * bh * silu_grad(a);
            dXref[b * 2 * D + D + d] = dy * silu(a);
        }

    GpuTensor Xg, Yg, dYg, dXg;
    brotensor::upload(X.data(), B, 2 * D, Xg);
    brotensor::swiglu_forward_gpu(Xg, Yg);
    std::vector<float> got(Yg.size());
    brotensor::download(Yg, got.data());
    brotensor::cuda_sync();
    float me = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
        me = std::max(me, std::fabs(got[i] - Yref[i]));
    std::printf("    fwd max_err=%g\n", me);
    CHECK(me < 1e-5f);

    brotensor::upload(dY.data(), B, D, dYg);
    brotensor::swiglu_backward_gpu(Xg, dYg, dXg);
    std::vector<float> got_dx(dXg.size());
    brotensor::download(dXg, got_dx.data());
    brotensor::cuda_sync();
    float me_b = 0.0f;
    for (size_t i = 0; i < got_dx.size(); ++i)
        me_b = std::max(me_b, std::fabs(got_dx[i] - dXref[i]));
    std::printf("    bwd max_err=%g\n", me_b);
    CHECK(me_b < 1e-5f);
}

static void test_fp16() {
    std::printf("  swiglu fp16 fwd+bwd\n");
    const int B = 3, D = 8;
    std::mt19937 rng(0x101);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(B * 2 * D), dY(B * D);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> Yref(B * D), dXref(B * 2 * D, 0.0f);
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d) {
            const float a  = X[b * 2 * D + d];
            const float bh = X[b * 2 * D + D + d];
            Yref[b * D + d] = silu(a) * bh;
            const float dy  = dY[b * D + d];
            dXref[b * 2 * D + d]     = dy * bh * silu_grad(a);
            dXref[b * 2 * D + D + d] = dy * silu(a);
        }

    GpuTensor Xg, Yg, dYg, dXg;
    auto Xh = to_fp16(X), dYh = to_fp16(dY);
    brotensor::upload_fp16(Xh.data(), B, 2 * D, Xg);
    brotensor::swiglu_forward_gpu(Xg, Yg);
    std::vector<uint16_t> got(Yg.size());
    brotensor::download_fp16(Yg, got.data());
    brotensor::cuda_sync();
    float me = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        me = std::max(me, std::fabs(g - Yref[i]));
    }
    std::printf("    fwd fp16 max_err=%g\n", me);
    CHECK(me < 5e-3f);

    brotensor::upload_fp16(dYh.data(), B, D, dYg);
    brotensor::swiglu_backward_gpu(Xg, dYg, dXg);
    std::vector<uint16_t> got_dx(dXg.size());
    brotensor::download_fp16(dXg, got_dx.data());
    brotensor::cuda_sync();
    float me_b = 0.0f;
    for (size_t i = 0; i < got_dx.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_dx[i]);
        me_b = std::max(me_b, std::fabs(g - dXref[i]));
    }
    std::printf("    bwd fp16 max_err=%g\n", me_b);
    CHECK(me_b < 5e-3f);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_swiglu\n");
    test_fp32();
    test_fp16();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
