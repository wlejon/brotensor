#pragma once

// Shared by test_flash_attention.cpp (forward parity, windowed, decode) and
// test_flash_attention_bwd.cpp (the backward cases): the CHECK counter and
// the FP16 helpers.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "gpu_select.h"

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

inline int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

inline std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}
inline std::vector<float> rq(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v[i]));
    return o;
}

inline void check_fp16(const std::vector<uint16_t>& got,
                       const std::vector<float>& ref,
                       const char* label,
                       float atol = 1e-2f, float rtol = 1e-2f) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = brotensor::fp16_bits_to_fp32(got[i]);
        const float e = std::fabs(g - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) {
            if (bad < 3)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, g, ref[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

// test_flash_attention_bwd.cpp
void run_bwd_self_vs_mha(const char* label, int L, int D, int nh);
void run_bwd_cross_vs_cx(const char* label, int Lq, int Lk, int D, int D_ctx, int nh);
void run_bwd_cross_with_biases_cpu(const char* label, int Lq, int Lk, int D, int D_ctx, int nh);
void run_bwd_causal_cpu(const char* label, int L, int D, int nh);
