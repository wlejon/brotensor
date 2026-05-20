// Regression test for the "capped grid + early-return" bug that silently
// dropped output elements past index 4096*256 = 1,048,576 in several
// per-element CUDA kernels.
//
// SD1.5's first GEGLU op produces B*D = 4096*1280 = 5,242,880 output
// elements. With the old capped grid_for (4096 blocks * 256 threads =
// 1,048,576 threads max) and an `if (idx >= total) return;` kernel pattern,
// roughly 80% of the output was left at whatever was in fresh device memory
// (typically zero). This test launches every affected op at >=5M elements
// and verifies that every output position — especially indices in
// [1_048_576, n) — matches the host reference.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// SD1.5 first FFN GEGLU shape.
static constexpr int kBigB = 4096;
static constexpr int kBigD = 1280;        // post-GEGLU feature dim
static constexpr int kCeil = 1048576;     // 4096 * 256: where the bug used
                                          // to start dropping elements.

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        out[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return out;
}

// Match the device's tanh-approx GELU exactly so the host reference uses
// the same formula the kernel uses (otherwise FP16 rounding noise around
// large indices is indistinguishable from the bug).
static inline float gelu_tanh_host(float v) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + std::tanh(u));
}

static inline float gelu_exact_host(float v) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
}

static inline float gelu_exact_grad_host(float v) {
    constexpr float kInvSqrt2  = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    const float cdf_term = 0.5f * (1.0f + std::erf(v * kInvSqrt2));
    const float pdf      = kInvSqrt2Pi * std::exp(-0.5f * v * v);
    return cdf_term + v * pdf;
}

// Bounded inputs so FP16 doesn't overflow / lose all precision at 5M
// distinct indices.
static inline float pattern_val(int idx) {
    // small, non-zero, varies per index — important so a dropped element
    // (left at the post-allocation zero) is clearly distinguishable from a
    // computed one.
    return 0.25f + 0.5f * std::sin(static_cast<float>(idx) * 1.0e-4f);
}

// Round-trip a host float through fp16 to model the device storage we'll
// compare against.
static inline float roundtrip_fp16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

// Count dropped (== exact zero, when reference is non-zero) elements past
// the old 1M ceiling — clearest possible signal of the original bug.
static int count_dropped_past_ceiling(const std::vector<float>& got,
                                      const std::vector<float>& ref,
                                      int n) {
    int dropped = 0;
    for (int i = kCeil; i < n; ++i) {
        if (got[i] == 0.0f && std::fabs(ref[i]) > 1e-3f) ++dropped;
    }
    return dropped;
}

static void test_geglu_forward_fp16_big() {
    std::printf("  geglu_forward_gpu  FP16  B=%d D=%d  (n=%d)\n",
                kBigB, kBigD, kBigB * kBigD);
    const int two_d = 2 * kBigD;
    const int in_n  = kBigB * two_d;
    const int out_n = kBigB * kBigD;

    std::vector<float> x_f(in_n);
    for (int i = 0; i < in_n; ++i) x_f[i] = pattern_val(i);
    auto x_h = to_fp16(x_f);

    Tensor X = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                         x_h.data(), kBigB, two_d);
    Tensor Y;
    brotensor::geglu_forward(X, Y);

    std::vector<uint16_t> y_h(out_n);
    Y.copy_to_host_fp16(y_h.data());
    brotensor::sync_all();

    std::vector<float> got(out_n), ref(out_n);
    for (int b = 0; b < kBigB; ++b) {
        for (int d = 0; d < kBigD; ++d) {
            const int idx = b * kBigD + d;
            const float a  = brotensor::fp16_bits_to_fp32(x_h[b * two_d + d]);
            const float gv = brotensor::fp16_bits_to_fp32(x_h[b * two_d + kBigD + d]);
            ref[idx] = roundtrip_fp16(a * gelu_tanh_host(gv));
            got[idx] = brotensor::fp16_bits_to_fp32(y_h[idx]);
        }
    }

    const int dropped = count_dropped_past_ceiling(got, ref, out_n);
    if (dropped > 0) {
        std::printf("  FAIL  geglu_forward FP16: %d elements past index %d "
                    "left at zero (capped-grid bug)\n", dropped, kCeil);
        ++g_failures;
    }
    int bad = 0;
    for (int i = 0; i < out_n && bad < 5; ++i) {
        if (std::fabs(got[i] - ref[i]) > 5e-3f) {
            std::printf("  FAIL  geglu_forward FP16 mismatch at i=%d "
                        "(got=%.6f ref=%.6f)\n", i, got[i], ref[i]);
            ++g_failures;
            ++bad;
        }
    }
}

static void test_geglu_exact_forward_fp16_big() {
    std::printf("  geglu_exact_forward_gpu  FP16  B=%d D=%d  (n=%d)\n",
                kBigB, kBigD, kBigB * kBigD);
    const int two_d = 2 * kBigD;
    const int in_n  = kBigB * two_d;
    const int out_n = kBigB * kBigD;

    std::vector<float> x_f(in_n);
    for (int i = 0; i < in_n; ++i) x_f[i] = pattern_val(i);
    auto x_h = to_fp16(x_f);

    Tensor X = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                         x_h.data(), kBigB, two_d);
    Tensor Y;
    brotensor::geglu_exact_forward(X, Y);

    std::vector<uint16_t> y_h(out_n);
    Y.copy_to_host_fp16(y_h.data());
    brotensor::sync_all();

    std::vector<float> got(out_n), ref(out_n);
    for (int b = 0; b < kBigB; ++b) {
        for (int d = 0; d < kBigD; ++d) {
            const int idx = b * kBigD + d;
            const float a  = brotensor::fp16_bits_to_fp32(x_h[b * two_d + d]);
            const float gv = brotensor::fp16_bits_to_fp32(x_h[b * two_d + kBigD + d]);
            ref[idx] = roundtrip_fp16(a * gelu_exact_host(gv));
            got[idx] = brotensor::fp16_bits_to_fp32(y_h[idx]);
        }
    }

    const int dropped = count_dropped_past_ceiling(got, ref, out_n);
    if (dropped > 0) {
        std::printf("  FAIL  geglu_exact_forward FP16: %d elements past index %d "
                    "left at zero (capped-grid bug)\n", dropped, kCeil);
        ++g_failures;
    }
    int bad = 0;
    for (int i = 0; i < out_n && bad < 5; ++i) {
        if (std::fabs(got[i] - ref[i]) > 5e-3f) {
            std::printf("  FAIL  geglu_exact_forward FP16 mismatch at i=%d "
                        "(got=%.6f ref=%.6f)\n", i, got[i], ref[i]);
            ++g_failures;
            ++bad;
        }
    }
}

static void test_geglu_exact_backward_fp16_big() {
    std::printf("  geglu_exact_backward_gpu  FP16  B=%d D=%d  (n=%d)\n",
                kBigB, kBigD, kBigB * kBigD);
    const int two_d = 2 * kBigD;
    const int in_n  = kBigB * two_d;
    const int out_n = kBigB * kBigD;

    std::vector<float> x_f(in_n);
    for (int i = 0; i < in_n; ++i) x_f[i] = pattern_val(i);
    auto x_h = to_fp16(x_f);

    std::vector<float> dy_f(out_n);
    for (int i = 0; i < out_n; ++i) dy_f[i] = pattern_val(in_n + i);
    auto dy_h = to_fp16(dy_f);

    Tensor X = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                         x_h.data(), kBigB, two_d);
    Tensor dY = Tensor::from_host_fp16_on(brotensor::Device::CUDA,
                                          dy_h.data(), kBigB, kBigD);
    Tensor dX;
    brotensor::geglu_exact_backward(X, dY, dX);

    std::vector<uint16_t> dx_h(in_n);
    dX.copy_to_host_fp16(dx_h.data());
    brotensor::sync_all();

    // dX has shape (B, 2D). dA = dy * g; dB_half = dy * a * gelu'(b).
    std::vector<float> got_a(out_n), ref_a(out_n);
    std::vector<float> got_b(out_n), ref_b(out_n);
    for (int b = 0; b < kBigB; ++b) {
        for (int d = 0; d < kBigD; ++d) {
            const int oi = b * kBigD + d;
            const float a  = brotensor::fp16_bits_to_fp32(x_h[b * two_d + d]);
            const float bh = brotensor::fp16_bits_to_fp32(x_h[b * two_d + kBigD + d]);
            const float dy = brotensor::fp16_bits_to_fp32(dy_h[oi]);
            const float g      = gelu_exact_host(bh);
            const float gprime = gelu_exact_grad_host(bh);
            ref_a[oi] = roundtrip_fp16(dy * g);
            ref_b[oi] = roundtrip_fp16(dy * a * gprime);
            got_a[oi] = brotensor::fp16_bits_to_fp32(dx_h[b * two_d + d]);
            got_b[oi] = brotensor::fp16_bits_to_fp32(dx_h[b * two_d + kBigD + d]);
        }
    }

    const int dropped_a = count_dropped_past_ceiling(got_a, ref_a, out_n);
    const int dropped_b = count_dropped_past_ceiling(got_b, ref_b, out_n);
    if (dropped_a > 0 || dropped_b > 0) {
        std::printf("  FAIL  geglu_exact_backward FP16: dA dropped=%d, "
                    "dB_half dropped=%d (capped-grid bug)\n",
                    dropped_a, dropped_b);
        ++g_failures;
    }
    int bad = 0;
    for (int i = 0; i < out_n && bad < 5; ++i) {
        if (std::fabs(got_a[i] - ref_a[i]) > 5e-3f ||
            std::fabs(got_b[i] - ref_b[i]) > 5e-3f) {
            std::printf("  FAIL  geglu_exact_backward FP16 mismatch at i=%d "
                        "(got_a=%.6f ref_a=%.6f got_b=%.6f ref_b=%.6f)\n",
                        i, got_a[i], ref_a[i], got_b[i], ref_b[i]);
            ++g_failures;
            ++bad;
        }
    }
}

int main() {
    std::printf("test_large_elementwise: exercising kernels at >1M elements\n");
    brotensor::init();
    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping\n");
        return 0;
    }
    try {
        test_geglu_forward_fp16_big();
        test_geglu_exact_forward_fp16_big();
        test_geglu_exact_backward_fp16_big();
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (g_failures) {
        std::printf("FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
