// CPU↔GPU parity for conv2d_forward_gpu (FP16). Compares against a naive
// FP32 CPU reference written inline.

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

// Naive NCHW conv2d reference. All inputs are FP32; we'll quantize to FP16
// at the boundary in the test driver to match the GPU's FP16 storage.
static void conv2d_cpu_fp32(const std::vector<float>& X,
                            const std::vector<float>& Wt,
                            const std::vector<float>& bias, bool has_bias,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int H_out, int W_out,
                            std::vector<float>& Y) {
    Y.assign(static_cast<size_t>(N) * C_out * H_out * W_out, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float acc = has_bias ? bias[oc] : 0.0f;
                    for (int ic = 0; ic < C_in; ++ic) {
                        for (int kh = 0; kh < kH; ++kh) {
                            const int in_h = oh * stride_h - pad_h + kh * dil_h;
                            if (in_h < 0 || in_h >= H) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                const int in_w = ow * stride_w - pad_w + kw * dil_w;
                                if (in_w < 0 || in_w >= W) continue;
                                const int x_idx = ((n * C_in + ic) * H + in_h) * W + in_w;
                                const int w_idx = ((oc * C_in + ic) * kH + kh) * kW + kw;
                                acc += X[x_idx] * Wt[w_idx];
                            }
                        }
                    }
                    const int y_idx = ((n * C_out + oc) * H_out + oh) * W_out + ow;
                    Y[y_idx] = acc;
                }
            }
        }
    }
}

static std::vector<uint16_t> to_fp16(const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = brotensor::fp32_to_fp16_bits(src[i]);
    }
    return out;
}

static std::vector<float> quantize_through_fp16(const std::vector<float>& src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(src[i]));
    }
    return out;
}

static void run_one(const char* label,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    bool has_bias) {
    std::printf("  %s  N=%d Cin=%d H=%d W=%d Cout=%d k=%dx%d s=%dx%d p=%dx%d d=%dx%d bias=%d\n",
                label, N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, (int)has_bias);

    std::mt19937 rng(0xC0DE);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int x_n = N * C_in * H * W;
    const int w_n = C_out * C_in * kH * kW;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    std::vector<float> X(x_n), Wt(w_n), bias(has_bias ? C_out : 0);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : Wt)   v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    // Quantize the CPU reference inputs to FP16 too, so we compare like-for-like
    // (the GPU sees FP16 storage of those exact bit patterns).
    auto X_q  = quantize_through_fp16(X);
    auto Wt_q = quantize_through_fp16(Wt);
    auto B_q  = quantize_through_fp16(bias);

    std::vector<float> Y_cpu;
    conv2d_cpu_fp32(X_q, Wt_q, B_q, has_bias,
                    N, C_in, H, W, C_out, kH, kW,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, Y_cpu);

    auto X_h16  = to_fp16(X);
    auto Wt_h16 = to_fp16(Wt);
    auto B_h16  = to_fp16(bias);

    GpuTensor Xg, Wg, Bg, Yg;
    brotensor::upload_fp16(X_h16.data(),  N,     C_in * H * W,  Xg);
    brotensor::upload_fp16(Wt_h16.data(), C_out, C_in * kH * kW, Wg);
    GpuTensor* Bptr = nullptr;
    if (has_bias) {
        brotensor::upload_fp16(B_h16.data(), C_out, 1, Bg);
        Bptr = &Bg;
    }

    brotensor::conv2d_forward_gpu(Xg, Wg, Bptr,
                                  N, C_in, H, W, C_out, kH, kW,
                                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                  Yg);

    CHECK(Yg.rows == N);
    CHECK(Yg.cols == C_out * H_out * W_out);
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
    std::printf("test_conv2d\n");

    // (1) Smallest meaningful case: 1x1 conv on a single pixel.
    run_one("trivial 1x1",
            /*N*/1, /*C_in*/2, /*H*/3, /*W*/3,
            /*C_out*/2, /*kH*/1, /*kW*/1,
            /*stride*/1, 1, /*pad*/0, 0, /*dil*/1, 1,
            /*bias*/true);

    // (2) Standard 3x3 same-pad with bias.
    run_one("3x3 same-pad",
            2, 3, 5, 5,
            4, 3, 3,
            1, 1, 1, 1, 1, 1,
            true);

    // (3) Stride 2 — downsample.
    run_one("3x3 stride2",
            1, 4, 8, 8,
            8, 3, 3,
            2, 2, 1, 1, 1, 1,
            false);

    // (4) Dilation 2.
    run_one("3x3 dilation2",
            1, 2, 7, 7,
            3, 3, 3,
            1, 1, 2, 2, 2, 2,
            true);

    // (5) Asymmetric kernel + asymmetric stride/pad.
    run_one("1x3 stride 1x2",
            1, 2, 4, 6,
            2, 1, 3,
            1, 2, 0, 1, 1, 1,
            true);

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll conv2d checks passed.\n");
    return 0;
}
