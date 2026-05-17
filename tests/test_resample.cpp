// CPU↔GPU parity for upsample/downsample 2x ops.
//   FP16 forward (regression), FP32 forward (new dispatch).
//   FP32 backward + FP16 backward for all three ops.

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

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ─── CPU forward references ────────────────────────────────────────────────

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
                    const int y0c = clampi(y0, 0, H - 1);
                    const int x0c = clampi(x0, 0, W - 1);
                    const int y1c = clampi(y0 + 1, 0, H - 1);
                    const int x1c = clampi(x0 + 1, 0, W - 1);
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

// ─── CPU backward references ───────────────────────────────────────────────

static void upsample_nearest_backward_cpu(const std::vector<float>& dY,
                                          int N, int C, int H, int W,
                                          std::vector<float>& dX) {
    const int H2 = 2 * H, W2 = 2 * W;
    dX.assign(static_cast<size_t>(N) * C * H * W, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int ih = 0; ih < H; ++ih)
                for (int iw = 0; iw < W; ++iw) {
                    double s = 0.0;
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b)
                            s += dY[((n * C + c) * H2 + 2 * ih + a) * W2 + 2 * iw + b];
                    dX[((n * C + c) * H + ih) * W + iw] = static_cast<float>(s);
                }
}

static void upsample_bilinear_backward_cpu(const std::vector<float>& dY,
                                           int N, int C, int H, int W,
                                           std::vector<float>& dX) {
    const int H2 = 2 * H, W2 = 2 * W;
    dX.assign(static_cast<size_t>(N) * C * H * W, 0.0f);
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
                    const int y0c = clampi(y0, 0, H - 1);
                    const int x0c = clampi(x0, 0, W - 1);
                    const int y1c = clampi(y0 + 1, 0, H - 1);
                    const int x1c = clampi(x0 + 1, 0, W - 1);
                    const float w00 = (1.0f - fy) * (1.0f - fx);
                    const float w01 = (1.0f - fy) * fx;
                    const float w10 = fy * (1.0f - fx);
                    const float w11 = fy * fx;
                    const float g = dY[((n * C + c) * H2 + oh) * W2 + ow];
                    const int base = (n * C + c) * H;
                    dX[(base + y0c) * W + x0c] += w00 * g;
                    dX[(base + y0c) * W + x1c] += w01 * g;
                    dX[(base + y1c) * W + x0c] += w10 * g;
                    dX[(base + y1c) * W + x1c] += w11 * g;
                }
}

static void downsample_avg_backward_cpu(const std::vector<float>& dY,
                                        int N, int C, int H, int W,
                                        std::vector<float>& dX) {
    const int H2 = H / 2, W2 = W / 2;
    dX.assign(static_cast<size_t>(N) * C * H * W, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int ih = 0; ih < H; ++ih)
                for (int iw = 0; iw < W; ++iw) {
                    const int oh = ih / 2, ow = iw / 2;
                    dX[((n * C + c) * H + ih) * W + iw] =
                        0.25f * dY[((n * C + c) * H2 + oh) * W2 + ow];
                }
}

// ─── Compare helpers ───────────────────────────────────────────────────────

static void compare_fp16(const std::vector<float>& ref,
                         const std::vector<uint16_t>& got_h, const char* label,
                         float abs_tol, float rel_tol) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got_h[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > abs_tol + rel_tol * std::fabs(ref[i])) ++bad;
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

static void compare_fp32(const std::vector<float>& ref,
                         const std::vector<float>& got, const char* label,
                         float abs_tol, float rel_tol) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        if (e > max_err) max_err = e;
        if (e > abs_tol + rel_tol * std::fabs(ref[i])) ++bad;
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

// ─── Test runners ──────────────────────────────────────────────────────────

static void run_fwd_fp16(int N, int C, int H, int W) {
    std::printf("  fp16 fwd  N=%d C=%d H=%d W=%d\n", N, C, H, W);
    std::mt19937 rng(0xF00D);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(static_cast<size_t>(N) * C * H * W);
    for (auto& v : X) v = dist(rng);
    auto X_q = requantize(X);
    auto X_h = to_fp16(X);

    GpuTensor Xg, Yg;
    brotensor::upload_fp16(X_h.data(), N, C * H * W, Xg);

    std::vector<float> Yref;
    upsample_nearest_cpu(X_q, N, C, H, W, Yref);
    brotensor::upsample_nearest_2x_gpu(Xg, N, C, H, W, Yg);
    CHECK(Yg.rows == N && Yg.cols == C * 4 * H * W && Yg.dtype == Dtype::FP16);
    std::vector<uint16_t> Y_h(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h.data());
    brotensor::cuda_sync();
    compare_fp16(Yref, Y_h, "nearest_2x", 1e-2f, 1e-2f);

    upsample_bilinear_cpu(X_q, N, C, H, W, Yref);
    brotensor::upsample_bilinear_2x_gpu(Xg, N, C, H, W, Yg);
    Y_h.assign(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h.data());
    brotensor::cuda_sync();
    compare_fp16(Yref, Y_h, "bilinear_2x", 1e-2f, 1e-2f);

    downsample_avg_cpu(X_q, N, C, H, W, Yref);
    brotensor::downsample_avg_2x_gpu(Xg, N, C, H, W, Yg);
    CHECK(Yg.rows == N && Yg.cols == C * (H / 2) * (W / 2));
    Y_h.assign(static_cast<size_t>(Yg.size()), 0);
    brotensor::download_fp16(Yg, Y_h.data());
    brotensor::cuda_sync();
    compare_fp16(Yref, Y_h, "downsample_avg_2x", 1e-2f, 1e-2f);
}

static void run_fwd_fp32(int N, int C, int H, int W) {
    std::printf("  fp32 fwd  N=%d C=%d H=%d W=%d\n", N, C, H, W);
    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X(static_cast<size_t>(N) * C * H * W);
    for (auto& v : X) v = dist(rng);

    GpuTensor Xg, Yg;
    brotensor::upload(X.data(), N, C * H * W, Xg);

    std::vector<float> Yref;
    upsample_nearest_cpu(X, N, C, H, W, Yref);
    brotensor::upsample_nearest_2x_gpu(Xg, N, C, H, W, Yg);
    CHECK(Yg.rows == N && Yg.cols == C * 4 * H * W && Yg.dtype == Dtype::FP32);
    std::vector<float> Y_got(static_cast<size_t>(Yg.size()), 0.0f);
    brotensor::download(Yg, Y_got.data());
    brotensor::cuda_sync();
    compare_fp32(Yref, Y_got, "nearest_2x", 1e-5f, 1e-5f);

    upsample_bilinear_cpu(X, N, C, H, W, Yref);
    brotensor::upsample_bilinear_2x_gpu(Xg, N, C, H, W, Yg);
    Y_got.assign(static_cast<size_t>(Yg.size()), 0.0f);
    brotensor::download(Yg, Y_got.data());
    brotensor::cuda_sync();
    compare_fp32(Yref, Y_got, "bilinear_2x", 1e-5f, 1e-5f);

    downsample_avg_cpu(X, N, C, H, W, Yref);
    brotensor::downsample_avg_2x_gpu(Xg, N, C, H, W, Yg);
    Y_got.assign(static_cast<size_t>(Yg.size()), 0.0f);
    brotensor::download(Yg, Y_got.data());
    brotensor::cuda_sync();
    compare_fp32(Yref, Y_got, "downsample_avg_2x", 1e-5f, 1e-5f);
}

static void run_bwd_fp32(int N, int C, int H, int W) {
    std::printf("  fp32 bwd  N=%d C=%d H=%d W=%d\n", N, C, H, W);
    std::mt19937 rng(0xD0D0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int Hout2 = 2 * H, Wout2 = 2 * W;
    const int Hdn = H / 2, Wdn = W / 2;

    // Nearest 2x upsample backward (dY has up-shape).
    {
        std::vector<float> dY(static_cast<size_t>(N) * C * Hout2 * Wout2);
        for (auto& v : dY) v = dist(rng);
        std::vector<float> dX_ref;
        upsample_nearest_backward_cpu(dY, N, C, H, W, dX_ref);

        GpuTensor dYg, dXg;
        brotensor::upload(dY.data(), N, C * Hout2 * Wout2, dYg);
        brotensor::upsample_nearest_2x_backward_gpu(dYg, N, C, H, W, dXg);
        CHECK(dXg.rows == N && dXg.cols == C * H * W && dXg.dtype == Dtype::FP32);
        std::vector<float> dX_got(static_cast<size_t>(dXg.size()), 0.0f);
        brotensor::download(dXg, dX_got.data());
        brotensor::cuda_sync();
        compare_fp32(dX_ref, dX_got, "nearest_bwd", 1e-5f, 1e-5f);
    }
    // Bilinear 2x upsample backward.
    {
        std::vector<float> dY(static_cast<size_t>(N) * C * Hout2 * Wout2);
        for (auto& v : dY) v = dist(rng);
        std::vector<float> dX_ref;
        upsample_bilinear_backward_cpu(dY, N, C, H, W, dX_ref);

        GpuTensor dYg, dXg;
        brotensor::upload(dY.data(), N, C * Hout2 * Wout2, dYg);
        brotensor::upsample_bilinear_2x_backward_gpu(dYg, N, C, H, W, dXg);
        CHECK(dXg.rows == N && dXg.cols == C * H * W && dXg.dtype == Dtype::FP32);
        std::vector<float> dX_got(static_cast<size_t>(dXg.size()), 0.0f);
        brotensor::download(dXg, dX_got.data());
        brotensor::cuda_sync();
        // Looser tol due to atomic-add summation order.
        compare_fp32(dX_ref, dX_got, "bilinear_bwd", 1e-4f, 1e-4f);
    }
    // Avg 2x downsample backward (dY has down-shape).
    if (Hdn > 0 && Wdn > 0) {
        std::vector<float> dY(static_cast<size_t>(N) * C * Hdn * Wdn);
        for (auto& v : dY) v = dist(rng);
        std::vector<float> dX_ref;
        downsample_avg_backward_cpu(dY, N, C, H, W, dX_ref);

        GpuTensor dYg, dXg;
        brotensor::upload(dY.data(), N, C * Hdn * Wdn, dYg);
        brotensor::downsample_avg_2x_backward_gpu(dYg, N, C, H, W, dXg);
        CHECK(dXg.rows == N && dXg.cols == C * H * W && dXg.dtype == Dtype::FP32);
        std::vector<float> dX_got(static_cast<size_t>(dXg.size()), 0.0f);
        brotensor::download(dXg, dX_got.data());
        brotensor::cuda_sync();
        compare_fp32(dX_ref, dX_got, "downsample_avg_bwd", 1e-5f, 1e-5f);
    }
}

static void run_bwd_fp16(int N, int C, int H, int W) {
    std::printf("  fp16 bwd  N=%d C=%d H=%d W=%d\n", N, C, H, W);
    std::mt19937 rng(0xD1D1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int Hout2 = 2 * H, Wout2 = 2 * W;
    const int Hdn = H / 2, Wdn = W / 2;

    // Nearest.
    {
        std::vector<float> dY(static_cast<size_t>(N) * C * Hout2 * Wout2);
        for (auto& v : dY) v = dist(rng);
        auto dY_q = requantize(dY);
        std::vector<float> dX_ref;
        upsample_nearest_backward_cpu(dY_q, N, C, H, W, dX_ref);

        auto dY_h = to_fp16(dY);
        GpuTensor dYg, dXg;
        brotensor::upload_fp16(dY_h.data(), N, C * Hout2 * Wout2, dYg);
        brotensor::upsample_nearest_2x_backward_gpu(dYg, N, C, H, W, dXg);
        CHECK(dXg.rows == N && dXg.cols == C * H * W && dXg.dtype == Dtype::FP16);
        std::vector<uint16_t> dX_got_h(static_cast<size_t>(dXg.size()), 0);
        brotensor::download_fp16(dXg, dX_got_h.data());
        brotensor::cuda_sync();
        compare_fp16(dX_ref, dX_got_h, "nearest_bwd", 1e-2f, 1e-2f);
    }
    // Bilinear.
    {
        std::vector<float> dY(static_cast<size_t>(N) * C * Hout2 * Wout2);
        for (auto& v : dY) v = dist(rng);
        auto dY_q = requantize(dY);
        std::vector<float> dX_ref;
        upsample_bilinear_backward_cpu(dY_q, N, C, H, W, dX_ref);

        auto dY_h = to_fp16(dY);
        GpuTensor dYg, dXg;
        brotensor::upload_fp16(dY_h.data(), N, C * Hout2 * Wout2, dYg);
        brotensor::upsample_bilinear_2x_backward_gpu(dYg, N, C, H, W, dXg);
        CHECK(dXg.rows == N && dXg.cols == C * H * W && dXg.dtype == Dtype::FP16);
        std::vector<uint16_t> dX_got_h(static_cast<size_t>(dXg.size()), 0);
        brotensor::download_fp16(dXg, dX_got_h.data());
        brotensor::cuda_sync();
        compare_fp16(dX_ref, dX_got_h, "bilinear_bwd", 2e-2f, 2e-2f);
    }
    // Avg.
    if (Hdn > 0 && Wdn > 0) {
        std::vector<float> dY(static_cast<size_t>(N) * C * Hdn * Wdn);
        for (auto& v : dY) v = dist(rng);
        auto dY_q = requantize(dY);
        std::vector<float> dX_ref;
        downsample_avg_backward_cpu(dY_q, N, C, H, W, dX_ref);

        auto dY_h = to_fp16(dY);
        GpuTensor dYg, dXg;
        brotensor::upload_fp16(dY_h.data(), N, C * Hdn * Wdn, dYg);
        brotensor::downsample_avg_2x_backward_gpu(dYg, N, C, H, W, dXg);
        CHECK(dXg.rows == N && dXg.cols == C * H * W && dXg.dtype == Dtype::FP16);
        std::vector<uint16_t> dX_got_h(static_cast<size_t>(dXg.size()), 0);
        brotensor::download_fp16(dXg, dX_got_h.data());
        brotensor::cuda_sync();
        compare_fp16(dX_ref, dX_got_h, "downsample_avg_bwd", 1e-2f, 1e-2f);
    }
}

int main() {
    try {
        brotensor::cuda_init();
    } catch (const std::exception& e) {
        std::printf("brotensor::cuda_init failed: %s\n", e.what());
        return 1;
    }
    std::printf("test_resample\n");

    // FP16 forward (regression).
    run_fwd_fp16(1, 3, 4, 4);
    run_fwd_fp16(2, 4, 6, 6);

    // FP32 forward (new dispatch).
    run_fwd_fp32(1, 3, 4, 4);
    run_fwd_fp32(2, 4, 6, 6);

    // FP32 backward.
    run_bwd_fp32(1, 3, 4, 4);
    run_bwd_fp32(2, 4, 6, 6);

    // FP16 backward.
    run_bwd_fp16(1, 3, 4, 4);
    run_bwd_fp16(2, 4, 6, 6);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll resample checks passed.\n");
    return 0;
}
