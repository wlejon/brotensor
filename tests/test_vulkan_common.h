#pragma once

// Shared harness for the Vulkan backend tests (test_vulkan.cpp: runtime,
// allocator, transfers, events, replay; test_vulkan_ops.cpp: op parity
// against the CPU backend). One executable, brotensor_test_vulkan; the suite
// skips (exit 0) when no Vulkan device registers.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vkt {

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

inline int& failures() {
    static int f = 0;
    return f;
}

#define VKT_CHECK(cond)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            ++::vkt::failures();                                               \
        }                                                                      \
    } while (0)

// True if `fn` throws std::exception.
template <class F>
bool throws(F&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

inline Device vk() { return Device::vulkan(0); }

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * static_cast<float>(next() >> 40) / 16777216.0f;
    }
};

inline const char* dt_name(Dtype dt) {
    switch (dt) {
        case Dtype::FP32: return "f32";
        case Dtype::FP16: return "f16";
        case Dtype::BF16: return "bf16";
        case Dtype::INT32: return "i32";
        case Dtype::INT8: return "i8";
        default: return "?";
    }
}

// Value as stored in `dt` and read back as float (the input both backends see).
inline float round_to(Dtype dt, float v) {
    if (dt == Dtype::FP16) return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
    if (dt == Dtype::BF16) return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
    return v;
}

inline std::vector<float> random_values(std::size_t n, std::uint64_t seed, float lo, float hi, Dtype dt) {
    Rng r(seed);
    std::vector<float> v(n);
    for (auto& x : v) x = round_to(dt, r.uniform(lo, hi));
    return v;
}

// Upload FP32 values to `d` stored as `dt`.
inline Tensor upload(const std::vector<float>& v, int rows, int cols, Dtype dt, Device d = vk()) {
    if (dt == Dtype::FP32) return Tensor::from_host_on(d, v.data(), rows, cols);
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        bits[i] = dt == Dtype::FP16 ? brotensor::fp32_to_fp16_bits(v[i]) : brotensor::fp32_to_bf16_bits(v[i]);
    }
    return dt == Dtype::FP16 ? Tensor::from_host_fp16_on(d, bits.data(), rows, cols)
                             : Tensor::from_host_bf16_on(d, bits.data(), rows, cols);
}

// Any FP32 / FP16 / BF16 tensor, on any device, as FP32 host values.
inline std::vector<float> download(const Tensor& t) {
    if (t.dtype == Dtype::FP32) return t.to_host_vector();
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(t.size()));
    t.copy_to_host_raw(bits.data(), bits.size() * 2);
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        out[i] = t.dtype == Dtype::FP16 ? brotensor::fp16_bits_to_fp32(bits[i]) : brotensor::bf16_bits_to_fp32(bits[i]);
    }
    return out;
}

// Relative step of the dtype's mantissa (one ulp at 1.0).
inline float dtype_eps(Dtype dt) {
    if (dt == Dtype::FP16) return 1.0f / 1024.0f;
    if (dt == Dtype::BF16) return 1.0f / 128.0f;
    return 1.0f / 8388608.0f;
}

// |got - want| <= atol + rtol * |want| elementwise; NaN must match NaN and
// infinities must match exactly. Prints one PASS / FAIL line.
inline bool expect_close(const std::vector<float>& got, const std::vector<float>& want,
                         float atol, float rtol, const std::string& tag) {
    if (got.size() != want.size()) {
        std::printf("  FAIL  %s: size %zu vs %zu\n", tag.c_str(), got.size(), want.size());
        ++failures();
        return false;
    }
    std::size_t worst = 0;
    double worst_excess = 0.0, worst_diff = 0.0;
    bool bad = false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float g = got[i], w = want[i];
        if (std::isnan(w) || std::isnan(g)) {
            if (std::isnan(w) != std::isnan(g)) { bad = true; worst = i; worst_excess = 1e30; }
            continue;
        }
        if (std::isinf(w) || std::isinf(g)) {
            if (g != w) { bad = true; worst = i; worst_excess = 1e30; }
            continue;
        }
        const double diff = std::fabs(double(g) - double(w));
        const double tol = atol + rtol * std::fabs(double(w));
        if (diff > tol && diff - tol > worst_excess) {
            bad = true;
            worst = i;
            worst_excess = diff - tol;
        }
        if (diff > worst_diff) worst_diff = diff;
    }
    if (bad) {
        std::printf("  FAIL  %s: at %zu got %.9g want %.9g (atol %g rtol %g)\n", tag.c_str(), worst,
                    got[worst], want[worst], atol, rtol);
        ++failures();
        return false;
    }
    std::printf("  PASS  %s (max |diff| %.3g)\n", tag.c_str(), worst_diff);
    return true;
}

// |got - want| <= rel * max|want| + 1e-7 (sums of many products in another
// order: the error scales with the largest output). Prints the worst
// difference relative to max|want|.
inline bool expect_scaled(const std::vector<float>& got, const std::vector<float>& want, float rel,
                          const std::string& tag) {
    float m = 0;
    for (float v : want) m = std::max(m, std::fabs(v));
    double worst = 0;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (std::isfinite(got[i]) && std::isfinite(want[i])) worst = std::max(worst, std::fabs(double(got[i]) - want[i]));
    }
    std::printf("        %s: max|diff| / max|want| = %.3g\n", tag.c_str(), m > 0 ? worst / m : worst);
    return expect_close(got, want, rel * m + 1e-7f, 0, tag);
}

inline bool expect_equal_bits(const void* a, const void* b, std::size_t n, const std::string& tag) {
    if (std::memcmp(a, b, n) != 0) {
        std::printf("  FAIL  %s: bytes differ\n", tag.c_str());
        ++failures();
        return false;
    }
    std::printf("  PASS  %s\n", tag.c_str());
    return true;
}

// Defined in test_vulkan_ops.cpp, test_vulkan_gemm.cpp, test_vulkan_norm.cpp,
// test_vulkan_attention.cpp, test_vulkan_bench.cpp and
// test_vulkan_bench_attention.cpp.
void run_op_tests();
void run_gemm_tests();
void run_norm_tests();
void run_attention_tests();
void run_gemm_bench();
void run_attention_bench();
void run_conv_tests();      // test_vulkan_conv.cpp
void run_spatial_tests();   // test_vulkan_spatial.cpp
void run_conv_bench();      // test_vulkan_bench_conv.cpp
void run_quant_tests();     // test_vulkan_quant.cpp
void run_audio_tests();     // test_vulkan_audio.cpp
void run_quant_bench();     // test_vulkan_bench_quant.cpp
void run_audio_bench();     // test_vulkan_bench_quant.cpp
void run_misc_tests();      // test_vulkan_misc.cpp
void run_vision_tests();    // test_vulkan_vision.cpp
void run_capture_tests();   // test_vulkan_capture.cpp
void run_train_tests();     // test_vulkan_train.cpp (calls the run_train_*_tests below)
void run_train_fa_tests();  // test_vulkan_train_fa.cpp
void run_train_spatial_tests();  // test_vulkan_train_spatial.cpp
void run_train_spatial2_tests(); // test_vulkan_train_spatial2.cpp
void run_jit_tests();       // test_vulkan_jit.cpp (the trace JIT's SPIR-V compiler)
void run_jit_bench();       // test_vulkan_bench_jit.cpp

}  // namespace vkt
