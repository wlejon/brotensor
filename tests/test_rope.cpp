// CPU↔GPU parity for rope_forward_gpu and rope_backward_gpu (FP32 + FP16).

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

static void rope_cpu_fwd(const std::vector<float>& X,
                         std::vector<float>& Y,
                         int L, int num_heads, int head_dim,
                         int seq_offset, float base) {
    const int D = num_heads * head_dim;
    Y.assign(L * D, 0.0f);
    for (int row = 0; row < L; ++row)
        for (int h = 0; h < num_heads; ++h)
            for (int i = 0; i < head_dim / 2; ++i) {
                const float theta = (row + seq_offset) *
                    std::exp(-2.0f * i / static_cast<float>(head_dim) *
                             std::log(base));
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const int off = row * D + h * head_dim;
                const float x0 = X[off + 2 * i];
                const float x1 = X[off + 2 * i + 1];
                Y[off + 2 * i]     = x0 * c - x1 * s;
                Y[off + 2 * i + 1] = x0 * s + x1 * c;
            }
}

static void rope_cpu_bwd(const std::vector<float>& dY,
                         std::vector<float>& dX,
                         int L, int num_heads, int head_dim,
                         int seq_offset, float base) {
    const int D = num_heads * head_dim;
    dX.assign(L * D, 0.0f);
    for (int row = 0; row < L; ++row)
        for (int h = 0; h < num_heads; ++h)
            for (int i = 0; i < head_dim / 2; ++i) {
                const float theta = (row + seq_offset) *
                    std::exp(-2.0f * i / static_cast<float>(head_dim) *
                             std::log(base));
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                const int off = row * D + h * head_dim;
                const float dy0 = dY[off + 2 * i];
                const float dy1 = dY[off + 2 * i + 1];
                dX[off + 2 * i]     = dy0 * c + dy1 * s;
                dX[off + 2 * i + 1] = -dy0 * s + dy1 * c;
            }
}

static void test_fp32() {
    std::printf("  rope fp32 fwd+bwd\n");
    const int L = 5, nh = 3, hd = 8;
    const int seq_offset = 4;
    const float base = 10000.0f;
    std::mt19937 rng(0x42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(L * nh * hd), dY(L * nh * hd);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> Yref, dXref;
    rope_cpu_fwd(X, Yref, L, nh, hd, seq_offset, base);
    rope_cpu_bwd(dY, dXref, L, nh, hd, seq_offset, base);

    GpuTensor Xg, Yg, dYg, dXg;
    brotensor::upload(X.data(), L, nh * hd, Xg);
    brotensor::rope_forward_gpu(Xg, hd, nh, seq_offset, base, Yg);
    std::vector<float> got(Yg.size());
    brotensor::download(Yg, got.data());
    brotensor::cuda_sync();
    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, std::fabs(got[i] - Yref[i]));
    }
    std::printf("    fwd fp32 max_err=%g\n", max_err);
    CHECK(max_err < 1e-4f);

    brotensor::upload(dY.data(), L, nh * hd, dYg);
    brotensor::rope_backward_gpu(dYg, hd, nh, seq_offset, base, dXg);
    std::vector<float> got_dx(dXg.size());
    brotensor::download(dXg, got_dx.data());
    brotensor::cuda_sync();
    float max_err_b = 0.0f;
    for (size_t i = 0; i < got_dx.size(); ++i) {
        max_err_b = std::max(max_err_b, std::fabs(got_dx[i] - dXref[i]));
    }
    std::printf("    bwd fp32 max_err=%g\n", max_err_b);
    CHECK(max_err_b < 1e-4f);
}

static void test_fp16() {
    std::printf("  rope fp16 fwd+bwd\n");
    const int L = 6, nh = 2, hd = 16;
    const int seq_offset = 2;
    const float base = 10000.0f;
    std::mt19937 rng(0x43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(L * nh * hd), dY(L * nh * hd);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> Yref, dXref;
    rope_cpu_fwd(X, Yref, L, nh, hd, seq_offset, base);
    rope_cpu_bwd(dY, dXref, L, nh, hd, seq_offset, base);

    GpuTensor Xg, Yg, dYg, dXg;
    auto Xh = to_fp16(X);
    brotensor::upload_fp16(Xh.data(), L, nh * hd, Xg);
    brotensor::rope_forward_gpu(Xg, hd, nh, seq_offset, base, Yg);
    std::vector<uint16_t> got(Yg.size());
    brotensor::download_fp16(Yg, got.data());
    brotensor::cuda_sync();
    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        max_err = std::max(max_err, std::fabs(g - Yref[i]));
    }
    std::printf("    fwd fp16 max_err=%g\n", max_err);
    CHECK(max_err < 3e-2f);

    auto dYh = to_fp16(dY);
    brotensor::upload_fp16(dYh.data(), L, nh * hd, dYg);
    brotensor::rope_backward_gpu(dYg, hd, nh, seq_offset, base, dXg);
    std::vector<uint16_t> got_dx(dXg.size());
    brotensor::download_fp16(dXg, got_dx.data());
    brotensor::cuda_sync();
    float max_err_b = 0.0f;
    for (size_t i = 0; i < got_dx.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_dx[i]);
        max_err_b = std::max(max_err_b, std::fabs(g - dXref[i]));
    }
    std::printf("    bwd fp16 max_err=%g\n", max_err_b);
    CHECK(max_err_b < 3e-2f);
}

int main() {
    brotensor::cuda_init();
    std::printf("test_rope\n");
    test_fp32();
    test_fp16();
    std::printf("%s (%d failures)\n",
                g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
