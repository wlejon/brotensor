// CPU↔GPU parity for conv2d_forward_gpu (FP16 + FP32) and
// conv2d_backward_input_gpu (FP32). Compares against naive FP32 CPU
// references written inline. Also runs a finite-difference check on a small
// shape to validate the analytic backward.

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

// CPU reference for conv2d_backward_input_gpu — gather form, matches the
// kernel index inversion exactly.
static void conv2d_backward_input_cpu_fp32(const std::vector<float>& Wt,
                                           const std::vector<float>& dY,
                                           int N, int C_in, int H, int W,
                                           int C_out, int kH, int kW,
                                           int stride_h, int stride_w,
                                           int pad_h, int pad_w,
                                           int dil_h, int dil_w,
                                           int H_out, int W_out,
                                           std::vector<float>& dX) {
    dX.assign(static_cast<size_t>(N) * C_in * H * W, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int num_h = i + pad_h - dil_h * kh;
                        if (num_h < 0 || num_h % stride_h != 0) continue;
                        const int i_out = num_h / stride_h;
                        if (i_out < 0 || i_out >= H_out) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int num_w = j + pad_w - dil_w * kw;
                            if (num_w < 0 || num_w % stride_w != 0) continue;
                            const int j_out = num_w / stride_w;
                            if (j_out < 0 || j_out >= W_out) continue;
                            for (int c_out = 0; c_out < C_out; ++c_out) {
                                const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                                const int w_idx  = ((c_out * C_in + c_in) * kH + kh) * kW + kw;
                                acc += dY[dy_idx] * Wt[w_idx];
                            }
                        }
                    }
                    const int dx_idx = ((n * C_in + c_in) * H + i) * W + j;
                    dX[dx_idx] = acc;
                }
            }
        }
    }
}

static void run_one_fp32(const char* label,
                         int N, int C_in, int H, int W,
                         int C_out, int kH, int kW,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dil_h, int dil_w,
                         bool has_bias) {
    std::printf("  [fp32 fwd] %s  N=%d Cin=%d H=%d W=%d Cout=%d k=%dx%d s=%dx%d p=%dx%d d=%dx%d bias=%d\n",
                label, N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, (int)has_bias);

    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int x_n = N * C_in * H * W;
    const int w_n = C_out * C_in * kH * kW;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    std::vector<float> X(x_n), Wt(w_n), bias(has_bias ? C_out : 0);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : Wt)   v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    std::vector<float> Y_cpu;
    conv2d_cpu_fp32(X, Wt, bias, has_bias,
                    N, C_in, H, W, C_out, kH, kW,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, Y_cpu);

    GpuTensor Xg, Wg, Bg, Yg;
    brotensor::upload(X.data(),  N,     C_in * H * W,   Xg);
    brotensor::upload(Wt.data(), C_out, C_in * kH * kW, Wg);
    GpuTensor* Bptr = nullptr;
    if (has_bias) {
        brotensor::upload(bias.data(), C_out, 1, Bg);
        Bptr = &Bg;
    }

    brotensor::conv2d_forward_gpu(Xg, Wg, Bptr,
                                  N, C_in, H, W, C_out, kH, kW,
                                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                  Yg);

    CHECK(Yg.rows == N);
    CHECK(Yg.cols == C_out * H_out * W_out);
    CHECK(Yg.dtype == Dtype::FP32);

    std::vector<float> Y_gpu(static_cast<size_t>(Yg.size()), 0.0f);
    brotensor::download(Yg, Y_gpu.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    // Accumulation can drift; bigger kernels get more terms. Allow 1e-4 abs.
    const float tol_abs = 1e-4f;
    const float tol_rel = 1e-5f;
    for (size_t i = 0; i < Y_cpu.size(); ++i) {
        const float err = std::fabs(Y_gpu[i] - Y_cpu[i]);
        if (err > max_err) max_err = err;
        const float tol = tol_abs + tol_rel * std::fabs(Y_cpu[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, Y_gpu[i], Y_cpu[i], err);
            }
            ++bad;
        }
    }
    std::printf("    fp32-fwd max_err=%g  bad=%d / %zu\n", max_err, bad, Y_cpu.size());
    CHECK(bad == 0);
}

static void run_one_bwd_input(const char* label,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dil_h, int dil_w) {
    std::printf("  [fp32 bwd-input] %s  N=%d Cin=%d H=%d W=%d Cout=%d k=%dx%d s=%dx%d p=%dx%d d=%dx%d\n",
                label, N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);

    std::mt19937 rng(0xFEEDU);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int w_n = C_out * C_in * kH * kW;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> Wt(w_n), dY(dy_n);
    for (auto& v : Wt) v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> dX_cpu;
    conv2d_backward_input_cpu_fp32(Wt, dY,
                                   N, C_in, H, W, C_out, kH, kW,
                                   stride_h, stride_w, pad_h, pad_w,
                                   dil_h, dil_w, H_out, W_out, dX_cpu);

    GpuTensor Wg, dYg, dXg;
    brotensor::upload(Wt.data(), C_out, C_in * kH * kW, Wg);
    brotensor::upload(dY.data(), N, C_out * H_out * W_out, dYg);

    brotensor::conv2d_backward_input_gpu(Wg, dYg,
                                         N, C_in, H, W, C_out, kH, kW,
                                         stride_h, stride_w, pad_h, pad_w,
                                         dil_h, dil_w, dXg);

    CHECK(dXg.rows == N);
    CHECK(dXg.cols == C_in * H * W);
    CHECK(dXg.dtype == Dtype::FP32);

    std::vector<float> dX_gpu(static_cast<size_t>(dXg.size()), 0.0f);
    brotensor::download(dXg, dX_gpu.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    const float tol_abs = 1e-4f;
    const float tol_rel = 1e-5f;
    for (size_t i = 0; i < dX_cpu.size(); ++i) {
        const float err = std::fabs(dX_gpu[i] - dX_cpu[i]);
        if (err > max_err) max_err = err;
        const float tol = tol_abs + tol_rel * std::fabs(dX_cpu[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, dX_gpu[i], dX_cpu[i], err);
            }
            ++bad;
        }
    }
    std::printf("    fp32-bwd-input max_err=%g  bad=%d / %zu\n",
                max_err, bad, dX_cpu.size());
    CHECK(bad == 0);
}

// CPU reference for conv2d_backward_weight_gpu — direct evaluation of the
// math in ops.h, looping over (c_out, c_in, kh, kw) outside and the output
// extent inside.
static void conv2d_backward_weight_cpu_fp32(const std::vector<float>& X,
                                            const std::vector<float>& dY,
                                            int N, int C_in, int H, int W,
                                            int C_out, int kH, int kW,
                                            int stride_h, int stride_w,
                                            int pad_h, int pad_w,
                                            int dil_h, int dil_w,
                                            int H_out, int W_out,
                                            std::vector<float>& dWt) {
    dWt.assign(static_cast<size_t>(C_out) * C_in * kH * kW, 0.0f);
    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    float acc = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        for (int i_out = 0; i_out < H_out; ++i_out) {
                            const int in_h = i_out * stride_h - pad_h + kh * dil_h;
                            if (in_h < 0 || in_h >= H) continue;
                            for (int j_out = 0; j_out < W_out; ++j_out) {
                                const int in_w = j_out * stride_w - pad_w + kw * dil_w;
                                if (in_w < 0 || in_w >= W) continue;
                                const int x_idx  = ((n * C_in  + c_in)  * H     + in_h)  * W     + in_w;
                                const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                                acc += dY[dy_idx] * X[x_idx];
                            }
                        }
                    }
                    const int w_idx = ((c_out * C_in + c_in) * kH + kh) * kW + kw;
                    dWt[w_idx] = acc;
                }
            }
        }
    }
}

static void conv2d_backward_bias_cpu_fp32(const std::vector<float>& dY,
                                          int N, int C_out, int H_out, int W_out,
                                          std::vector<float>& dB) {
    dB.assign(static_cast<size_t>(C_out), 0.0f);
    for (int c_out = 0; c_out < C_out; ++c_out) {
        double acc = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int i_out = 0; i_out < H_out; ++i_out) {
                for (int j_out = 0; j_out < W_out; ++j_out) {
                    const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                    acc += dY[dy_idx];
                }
            }
        }
        dB[c_out] = static_cast<float>(acc);
    }
}

static void run_one_bwd_weight(const char* label,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w) {
    std::printf("  [fp32 bwd-weight] %s  N=%d Cin=%d H=%d W=%d Cout=%d k=%dx%d s=%dx%d p=%dx%d d=%dx%d\n",
                label, N, C_in, H, W, C_out, kH, kW,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w);

    std::mt19937 rng(0xD00DU);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int x_n = N * C_in * H * W;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> X(x_n), dY(dy_n);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> dW_cpu;
    conv2d_backward_weight_cpu_fp32(X, dY,
                                    N, C_in, H, W, C_out, kH, kW,
                                    stride_h, stride_w, pad_h, pad_w,
                                    dil_h, dil_w, H_out, W_out, dW_cpu);

    GpuTensor Xg, dYg, dWg;
    brotensor::upload(X.data(),  N, C_in * H * W, Xg);
    brotensor::upload(dY.data(), N, C_out * H_out * W_out, dYg);
    // Pre-allocate dWg shape (C_out, C_in*kH*kW), zeroed — op accumulates.
    std::vector<float> zeros(C_out * C_in * kH * kW, 0.0f);
    brotensor::upload(zeros.data(), C_out, C_in * kH * kW, dWg);

    brotensor::conv2d_backward_weight_gpu(Xg, dYg,
                                          N, C_in, H, W, C_out, kH, kW,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dil_h, dil_w, dWg);

    CHECK(dWg.rows == C_out);
    CHECK(dWg.cols == C_in * kH * kW);
    CHECK(dWg.dtype == Dtype::FP32);

    std::vector<float> dW_gpu(static_cast<size_t>(dWg.size()), 0.0f);
    brotensor::download(dWg, dW_gpu.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    const float tol_abs = 1e-4f;
    const float tol_rel = 1e-5f;
    for (size_t i = 0; i < dW_cpu.size(); ++i) {
        const float err = std::fabs(dW_gpu[i] - dW_cpu[i]);
        if (err > max_err) max_err = err;
        const float tol = tol_abs + tol_rel * std::fabs(dW_cpu[i]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                            i, dW_gpu[i], dW_cpu[i], err);
            }
            ++bad;
        }
    }
    std::printf("    fp32-bwd-weight max_err=%g  bad=%d / %zu\n",
                max_err, bad, dW_cpu.size());
    CHECK(bad == 0);
}

static void run_one_bwd_bias(const char* label,
                             int N, int C_out, int H_out, int W_out) {
    std::printf("  [fp32 bwd-bias] %s  N=%d Cout=%d Hout=%d Wout=%d\n",
                label, N, C_out, H_out, W_out);

    std::mt19937 rng(0xB1A5U);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int dy_n = N * C_out * H_out * W_out;
    std::vector<float> dY(dy_n);
    for (auto& v : dY) v = dist(rng);

    std::vector<float> dB_cpu;
    conv2d_backward_bias_cpu_fp32(dY, N, C_out, H_out, W_out, dB_cpu);

    GpuTensor dYg, dBg;
    brotensor::upload(dY.data(), N, C_out * H_out * W_out, dYg);
    std::vector<float> zeros(C_out, 0.0f);
    brotensor::upload(zeros.data(), C_out, 1, dBg);

    brotensor::conv2d_backward_bias_gpu(dYg, N, C_out, H_out, W_out, dBg);

    CHECK(dBg.rows == C_out);
    CHECK(dBg.cols == 1);
    CHECK(dBg.dtype == Dtype::FP32);

    std::vector<float> dB_gpu(static_cast<size_t>(dBg.size()), 0.0f);
    brotensor::download(dBg, dB_gpu.data());
    brotensor::cuda_sync();

    int bad = 0;
    float max_err = 0.0f;
    const float tol_abs = 1e-4f;
    const float tol_rel = 1e-5f;
    for (int c = 0; c < C_out; ++c) {
        const float err = std::fabs(dB_gpu[c] - dB_cpu[c]);
        if (err > max_err) max_err = err;
        const float tol = tol_abs + tol_rel * std::fabs(dB_cpu[c]);
        if (err > tol) {
            if (bad < 5) {
                std::printf("    mismatch c=%d got=%g ref=%g err=%g\n",
                            c, dB_gpu[c], dB_cpu[c], err);
            }
            ++bad;
        }
    }
    std::printf("    fp32-bwd-bias max_err=%g  bad=%d / %d\n",
                max_err, bad, C_out);
    CHECK(bad == 0);
}

// Finite-difference check for dW on the same tiny shape as the dX FD check.
static void run_finite_diff_check_weight() {
    const int N = 1, C_in = 2, C_out = 2, H = 4, W = 4;
    const int kH = 3, kW = 3;
    const int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    const int dil_h = 1, dil_w = 1;
    std::printf("  [fp32 bwd-weight finite-diff] tiny 3x3 same-pad\n");

    std::mt19937 rng(0xCAFEU);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    const int x_n = N * C_in * H * W;
    const int w_n = C_out * C_in * kH * kW;
    const int H_out = H, W_out = W;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> X(x_n), Wt(w_n), dY(dy_n);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : Wt) v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    // Analytic dW from the GPU (zero dWg first, op accumulates).
    GpuTensor Xg, dYg, dWg;
    brotensor::upload(X.data(),  N, C_in * H * W, Xg);
    brotensor::upload(dY.data(), N, C_out * H_out * W_out, dYg);
    std::vector<float> zeros(w_n, 0.0f);
    brotensor::upload(zeros.data(), C_out, C_in * kH * kW, dWg);

    brotensor::conv2d_backward_weight_gpu(Xg, dYg,
                                          N, C_in, H, W, C_out, kH, kW,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dil_h, dil_w, dWg);
    std::vector<float> dW_gpu(static_cast<size_t>(dWg.size()), 0.0f);
    brotensor::download(dWg, dW_gpu.data());
    brotensor::cuda_sync();

    auto loss_for_W = [&](const std::vector<float>& W_pert) {
        std::vector<float> Y;
        conv2d_cpu_fp32(X, W_pert, /*bias*/std::vector<float>{}, /*has_bias*/false,
                        N, C_in, H, W, C_out, kH, kW,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                        H_out, W_out, Y);
        double s = 0.0;
        for (size_t i = 0; i < Y.size(); ++i) s += static_cast<double>(dY[i]) * Y[i];
        return s;
    };

    const float eps = 1e-3f;
    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < w_n; ++i) {
        std::vector<float> Wp = Wt, Wm = Wt;
        Wp[i] += eps;
        Wm[i] -= eps;
        const double lp = loss_for_W(Wp);
        const double lm = loss_for_W(Wm);
        const float num = static_cast<float>((lp - lm) / (2.0 * eps));
        const float ana = dW_gpu[i];
        const float err = std::fabs(num - ana);
        if (err > max_err) max_err = err;
        if (err > 1e-3f + 1e-3f * std::fabs(ana)) {
            if (bad < 5) {
                std::printf("    fd mismatch i=%d num=%g ana=%g err=%g\n",
                            i, num, ana, err);
            }
            ++bad;
        }
    }
    std::printf("    finite-diff dW max_err=%g  bad=%d / %d\n", max_err, bad, w_n);
    CHECK(bad == 0);
}

// Finite-difference check for dB: perturb each bias, run forward with bias,
// numerical grad ≈ (L(b+h) - L(b-h)) / (2h); compare against analytic dB.
static void run_finite_diff_check_bias() {
    // dB depends only on dY, not on X / W / conv hyperparams — the FD check
    // perturbs bias and the bias contribution to Y is purely additive per
    // channel. We only need C_out, H_out, W_out, N to define dY.
    const int N = 1, C_out = 2, H_out = 4, W_out = 4;
    std::printf("  [fp32 bwd-bias finite-diff] tiny 3x3 same-pad\n");

    std::mt19937 rng(0xCAFEU);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> dY(dy_n), bias(C_out);
    for (auto& v : dY)   v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    // Analytic dB from the GPU (does not depend on bias).
    GpuTensor dYg, dBg;
    brotensor::upload(dY.data(), N, C_out * H_out * W_out, dYg);
    std::vector<float> zeros(C_out, 0.0f);
    brotensor::upload(zeros.data(), C_out, 1, dBg);
    brotensor::conv2d_backward_bias_gpu(dYg, N, C_out, H_out, W_out, dBg);
    std::vector<float> dB_gpu(static_cast<size_t>(dBg.size()), 0.0f);
    brotensor::download(dBg, dB_gpu.data());
    brotensor::cuda_sync();

    // For bias FD: forward Y depends on bias additively per channel, so the
    // analytic gradient equals sum_{n,i,j} dY[n,c,i,j], independent of X/W.
    // Use a "bias-only" forward: Y[n,c,i,j] = bias[c].
    auto loss_for_bias = [&](const std::vector<float>& b_pert) {
        double s = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C_out; ++c) {
                for (int i = 0; i < H_out; ++i) {
                    for (int j = 0; j < W_out; ++j) {
                        const int dy_idx = ((n * C_out + c) * H_out + i) * W_out + j;
                        s += static_cast<double>(dY[dy_idx]) * b_pert[c];
                    }
                }
            }
        }
        return s;
    };

    const float eps = 1e-3f;
    int bad = 0;
    float max_err = 0.0f;
    for (int c = 0; c < C_out; ++c) {
        std::vector<float> bp = bias, bm = bias;
        bp[c] += eps;
        bm[c] -= eps;
        const double lp = loss_for_bias(bp);
        const double lm = loss_for_bias(bm);
        const float num = static_cast<float>((lp - lm) / (2.0 * eps));
        const float ana = dB_gpu[c];
        const float err = std::fabs(num - ana);
        if (err > max_err) max_err = err;
        if (err > 1e-3f + 1e-3f * std::fabs(ana)) {
            if (bad < 5) {
                std::printf("    fd mismatch c=%d num=%g ana=%g err=%g\n",
                            c, num, ana, err);
            }
            ++bad;
        }
    }
    std::printf("    finite-diff dB max_err=%g  bad=%d / %d\n", max_err, bad, C_out);
    CHECK(bad == 0);
}

// Finite-difference check on a tiny shape: confirms the analytic dX matches
// d/dX of sum(dY * Y(X)), where dY is a fixed random tensor and Y(X) is the
// forward conv. So gradient[i] = sum_j dY[j] * dY/dX[i] which equals our
// returned dX[i].
static void run_finite_diff_check() {
    const int N = 1, C_in = 2, C_out = 2, H = 4, W = 4;
    const int kH = 3, kW = 3;
    const int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    const int dil_h = 1, dil_w = 1;
    std::printf("  [fp32 bwd-input finite-diff] tiny 3x3 same-pad\n");

    std::mt19937 rng(0xCAFE);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    const int x_n = N * C_in * H * W;
    const int w_n = C_out * C_in * kH * kW;
    const int H_out = H, W_out = W;
    const int dy_n = N * C_out * H_out * W_out;

    std::vector<float> X(x_n), Wt(w_n), dY(dy_n);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : Wt) v = dist(rng);
    for (auto& v : dY) v = dist(rng);

    // Analytic dX from the GPU.
    GpuTensor Wg, dYg, dXg;
    brotensor::upload(Wt.data(), C_out, C_in * kH * kW, Wg);
    brotensor::upload(dY.data(), N, C_out * H_out * W_out, dYg);
    brotensor::conv2d_backward_input_gpu(Wg, dYg,
                                         N, C_in, H, W, C_out, kH, kW,
                                         stride_h, stride_w, pad_h, pad_w,
                                         dil_h, dil_w, dXg);
    std::vector<float> dX_gpu(static_cast<size_t>(dXg.size()), 0.0f);
    brotensor::download(dXg, dX_gpu.data());
    brotensor::cuda_sync();

    // CPU forward closure: returns sum(dY * Y(X_perturbed)).
    auto loss_for_X = [&](const std::vector<float>& X_pert) {
        std::vector<float> Y;
        conv2d_cpu_fp32(X_pert, Wt, /*bias*/std::vector<float>{}, /*has_bias*/false,
                        N, C_in, H, W, C_out, kH, kW,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                        H_out, W_out, Y);
        double s = 0.0;
        for (size_t i = 0; i < Y.size(); ++i) s += static_cast<double>(dY[i]) * Y[i];
        return s;
    };

    const float eps = 1e-3f;
    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < x_n; ++i) {
        std::vector<float> Xp = X, Xm = X;
        Xp[i] += eps;
        Xm[i] -= eps;
        const double lp = loss_for_X(Xp);
        const double lm = loss_for_X(Xm);
        const float num = static_cast<float>((lp - lm) / (2.0 * eps));
        const float ana = dX_gpu[i];
        const float err = std::fabs(num - ana);
        if (err > max_err) max_err = err;
        if (err > 1e-3f + 1e-3f * std::fabs(ana)) {
            if (bad < 5) {
                std::printf("    fd mismatch i=%d num=%g ana=%g err=%g\n",
                            i, num, ana, err);
            }
            ++bad;
        }
    }
    std::printf("    finite-diff max_err=%g  bad=%d / %d\n", max_err, bad, x_n);
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

    // ─── FP32 forward parity ─────────────────────────────────────────────
    run_one_fp32("3x3 same-pad",
                 2, 3, 5, 5, 4, 3, 3,
                 1, 1, 1, 1, 1, 1, true);
    run_one_fp32("3x3 stride2",
                 1, 4, 8, 8, 8, 3, 3,
                 2, 2, 1, 1, 1, 1, false);
    run_one_fp32("1x1 pointwise",
                 1, 2, 3, 3, 2, 1, 1,
                 1, 1, 0, 0, 1, 1, true);

    // ─── FP32 backward-input parity ──────────────────────────────────────
    run_one_bwd_input("3x3 same-pad",
                      2, 3, 5, 5, 4, 3, 3,
                      1, 1, 1, 1, 1, 1);
    run_one_bwd_input("3x3 stride2 half",
                      1, 4, 8, 8, 8, 3, 3,
                      2, 2, 1, 1, 1, 1);
    run_one_bwd_input("1x1 pointwise",
                      1, 2, 3, 3, 2, 1, 1,
                      1, 1, 0, 0, 1, 1);

    // ─── FP32 backward-weight parity ─────────────────────────────────────
    run_one_bwd_weight("3x3 same-pad",
                       2, 3, 5, 5, 4, 3, 3,
                       1, 1, 1, 1, 1, 1);
    run_one_bwd_weight("3x3 stride2 half",
                       1, 4, 8, 8, 8, 3, 3,
                       2, 2, 1, 1, 1, 1);
    run_one_bwd_weight("1x1 pointwise",
                       1, 2, 3, 3, 2, 1, 1,
                       1, 1, 0, 0, 1, 1);

    // ─── FP32 backward-bias parity ───────────────────────────────────────
    run_one_bwd_bias("typical", 2, 4, 5, 5);
    run_one_bwd_bias("wide",    1, 8, 4, 4);
    run_one_bwd_bias("tall",    1, 2, 3, 3);

    // ─── Gold-standard finite-difference check ───────────────────────────
    run_finite_diff_check();
    run_finite_diff_check_weight();
    run_finite_diff_check_bias();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll conv2d checks passed.\n");
    return 0;
}
