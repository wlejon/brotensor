#pragma once

// Shared by test_conv2d.cpp (dense conv2d parity, backward, finite
// differences) and test_conv2d_grouped.cpp (grouped / depthwise): the CHECK
// counter, the FP16 helpers and the naive FP32 CPU references.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#include "gpu_select.h"

using brotensor::Tensor;
using brotensor::Dtype;

inline int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// Naive NCHW conv2d reference. All inputs are FP32; we'll quantize to FP16
// at the boundary in the test driver to match the GPU's FP16 storage.
inline void conv2d_cpu_fp32(const std::vector<float>& X,
                            const std::vector<float>& Wt,
                            const std::vector<float>& bias, bool has_bias,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int H_out, int W_out,
                            std::vector<float>& Y,
                            int groups = 1) {
    Y.assign(static_cast<size_t>(N) * C_out * H_out * W_out, 0.0f);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < C_out; ++oc) {
            const int g = oc / Cg_out;
            const int ic_base = g * Cg_in;
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float acc = has_bias ? bias[oc] : 0.0f;
                    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
                        const int ic = ic_base + ic_local;
                        for (int kh = 0; kh < kH; ++kh) {
                            const int in_h = oh * stride_h - pad_h + kh * dil_h;
                            if (in_h < 0 || in_h >= H) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                const int in_w = ow * stride_w - pad_w + kw * dil_w;
                                if (in_w < 0 || in_w >= W) continue;
                                const int x_idx = ((n * C_in + ic) * H + in_h) * W + in_w;
                                const int w_idx = ((oc * Cg_in + ic_local) * kH + kh) * kW + kw;
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

inline std::vector<uint16_t> to_fp16(const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = brotensor::fp32_to_fp16_bits(src[i]);
    }
    return out;
}

inline std::vector<float> quantize_through_fp16(const std::vector<float>& src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(src[i]));
    }
    return out;
}

// CPU reference for conv2d_backward_input_gpu — gather form, matches the
// kernel index inversion exactly.
inline void conv2d_backward_input_cpu_fp32(const std::vector<float>& Wt,
                                           const std::vector<float>& dY,
                                           int N, int C_in, int H, int W,
                                           int C_out, int kH, int kW,
                                           int stride_h, int stride_w,
                                           int pad_h, int pad_w,
                                           int dil_h, int dil_w,
                                           int H_out, int W_out,
                                           std::vector<float>& dX,
                                           int groups = 1) {
    dX.assign(static_cast<size_t>(N) * C_in * H * W, 0.0f);
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    for (int n = 0; n < N; ++n) {
        for (int c_in = 0; c_in < C_in; ++c_in) {
            const int g = c_in / Cg_in;
            const int c_in_local = c_in - g * Cg_in;
            const int oc_lo = g * Cg_out;
            const int oc_hi = oc_lo + Cg_out;
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
                            for (int c_out = oc_lo; c_out < oc_hi; ++c_out) {
                                const int dy_idx = ((n * C_out + c_out) * H_out + i_out) * W_out + j_out;
                                const int w_idx  = ((c_out * Cg_in + c_in_local) * kH + kh) * kW + kw;
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

// CPU reference for conv2d_backward_weight_gpu — direct evaluation of the
// math in ops.h, looping over (c_out, c_in, kh, kw) outside and the output
// extent inside.
inline void conv2d_backward_weight_cpu_fp32(const std::vector<float>& X,
                                            const std::vector<float>& dY,
                                            int N, int C_in, int H, int W,
                                            int C_out, int kH, int kW,
                                            int stride_h, int stride_w,
                                            int pad_h, int pad_w,
                                            int dil_h, int dil_w,
                                            int H_out, int W_out,
                                            std::vector<float>& dWt,
                                            int groups = 1) {
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    dWt.assign(static_cast<size_t>(C_out) * Cg_in * kH * kW, 0.0f);
    for (int c_out = 0; c_out < C_out; ++c_out) {
        const int g = c_out / Cg_out;
        for (int c_in_local = 0; c_in_local < Cg_in; ++c_in_local) {
            const int c_in = g * Cg_in + c_in_local;
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
                    const int w_idx = ((c_out * Cg_in + c_in_local) * kH + kh) * kW + kw;
                    dWt[w_idx] = acc;
                }
            }
        }
    }
}

inline void conv2d_backward_bias_cpu_fp32(const std::vector<float>& dY,
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

inline void check_fp16_against(const std::vector<uint16_t>& got,
                               const std::vector<float>& ref,
                               const char* label,
                               int& bad_out, float& max_err_out) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float err = std::fabs(g - ref[i]);
        if (err > max_err) max_err = err;
        const float tol = 1e-2f + 1e-2f * std::fabs(ref[i]);
        if (err > tol) {
            if (bad < 5)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, g, ref[i], err);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
    bad_out = bad;
    max_err_out = max_err;
}

// test_conv2d_grouped.cpp
void run_grouped_fp32(const char* label, int N, int C_in, int H, int W, int C_out, int kH, int kW,
                      int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups);
void run_grouped_fp16(const char* label, int N, int C_in, int H, int W, int C_out, int kH, int kW,
                      int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w, int groups);
void run_depthwise_finite_diff();
