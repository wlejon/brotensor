// CPU↔GPU parity for upsample/downsample 2x ops (FP16 NCHW).

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

static std::vector<uint16_t> to_fp16(const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        out[i] = brotensor::fp32_to_fp16_bits(src[i]);
    return out;
}
static std::vector<float> requantize(const std::vector<float>& src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        out[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(src[i]));
    return out;
}

static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void upsample_nearest_cpu(const std::vector<float>& X,
                                 int N, int C, int H, int W,
                                 std::vector<float>& Y) {
    const int H2 = 2 * H, W2 = 2 * W;
    Y.assign(static_cast<size_t>(N) * C * H2 * W2, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < H2; ++oh)
                for (int ow = 0; ow < W2; ++ow) {
                    const int ih = oh / 2, iw = ow / 2;
                    Y[((n * C + c) * H2 + oh) * W2 + ow] =
                        X[((n * C + c) * H + ih) * W + iw];
                }
}

static void upsample_bilinear_cpu(const std::vector<float>& X,
                                  int N, int C, int H, int W,
                                  std::vector<float>& Y) {
    const int H2 = 2 * H, W2 = 2 * W;
    Y.assign(static_cast<size_t>(N) * C * H2 * W2, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < H2; ++oh)
                for (int ow = 0; ow < W2; ++ow) {
                    const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
                    const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
                    const int y0 = static_cast<int>(std::floor(src_y));
                    const int x0 = static_cast<int>(std::floor(src_x));
                    const float fy = src_y - y0;
                    const float fx = src_x - x0;
                    const int y0c = clamp(y0, 0, H - 1);
                    const int x0c = clamp(x0, 0, W - 1);
                    const int y1c = clamp(y0 + 1, 0, H - 1);
                    const int x1c = clamp(x0 + 1, 0, W - 1);
                    const int base = (n * C + c) * H;
                    const float v00 = X[(base + y0c) * W + x0c];
                    const float v01 = X[(base + y0c) * W + x1c];
                    const float v10 = X[(base + y1c) * W + x0c];
                    const float v11 = X[(base + y1c) * W + x1c];
                    const float top = v00 + (v01 - v00) * fx;
                    const float bot = v10 + (v11 - v10) * fx;
                    Y[((n * C + c) * H2 + oh) * W2 + ow] =
                        top + (bot - top) * fy;
                }
}

static void downsample_avg_cpu(const std::vector<float>& X,
                               int N, int C, int H, int W,
                               std::vector<float>& Y) {
    const int H2 = H / 2, W2 = W / 2;
    Y.assign(static_cast<size_t>(N) * C * H2 * W2, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < H2; ++oh)
                for (int ow = 0; ow < W2; ++ow) {
                    const int ih = oh * 2, iw = ow * 2;
                    const int b = ((n * C + c) * H + ih) * W + iw;
                    Y[((n * C + c) * H2 + oh) * W2 + ow] =
                        0.25f * (X[b] + X[b + 1] + X[b + W] + X[b + W + 1]);
                }
}

static void compare(const std::vector<float>& ref,
                    const std::vector<uint16_t>& got_h, const char* label) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_h[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-2f + 1e-2f * std::fabs(ref[i])) ++bad;
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_resample\n");

    const int N = 1, C = 3, H = 4, W = 4;
    std::mt19937 rng(0xF00D);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(N * C * H * W);
    for (auto& v : X) v = dist(rng);
    auto X_q = requantize(X);
    auto X_h = to_fp16(X);

    GpuTensor Xg, Yg;
    brotensor::upload_fp16(X_h.data(), N, C * H * W, Xg);

    // Nearest upsample.
    std::vector<float> Yref;
    upsample_nearest_cpu(X_q, N, C, H, W, Yref);
    brotensor::upsample_nearest_2x_gpu(Xg, N, C, H, W, Yg);
    CHECK(Yg.rows == N && Yg.cols == C * 4 * H * W && Yg.dtype == Dtype::FP16);
    std::vector<uint16_t> Y_h(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h.data());
    brotensor::cuda_sync();
    compare(Yref, Y_h, "nearest_2x");

    // Bilinear upsample.
    upsample_bilinear_cpu(X_q, N, C, H, W, Yref);
    brotensor::upsample_bilinear_2x_gpu(Xg, N, C, H, W, Yg);
    Y_h.assign(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h.data());
    brotensor::cuda_sync();
    compare(Yref, Y_h, "bilinear_2x");

    // Downsample.
    downsample_avg_cpu(X_q, N, C, H, W, Yref);
    brotensor::downsample_avg_2x_gpu(Xg, N, C, H, W, Yg);
    CHECK(Yg.rows == N && Yg.cols == C * (H / 2) * (W / 2));
    Y_h.assign(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h.data());
    brotensor::cuda_sync();
    compare(Yref, Y_h, "downsample_avg_2x");

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll resample checks passed.\n");
    return 0;
}
