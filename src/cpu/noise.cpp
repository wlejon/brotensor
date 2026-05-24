// ─── CPU counter-based noise generation ─────────────────────────────────────
//
// Philox 4x32-10 stream (PyTorch / JAX compatible) implementing four host-side
// FP32 noise ops:
//   randn            — N(0, 1) via Box-Muller
//   rand_uniform     — U[0, 1)
//   rand_bernoulli   — 0/1 mask at probability p
//   randn_truncated  — rejection-sampled normal in [lo, hi]
//
// All four require Y FP32 and pre-sized; the op fills rows*cols elements in
// row-major linear order. Element i draws from substream (counter + i); see
// ops.h for the full ABI contract.
//
// The Philox construction here is byte-identical to src/cpu/sample_logits.cpp
// and the CUDA / Metal noise implementations, so a given (key, counter) yields
// the same draws on every backend.

#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// ── Philox 4x32-10 ──────────────────────────────────────────────────────────
// Same construction as sample_logits.cpp; duplicated rather than shared so the
// translation units stay independent.

inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
}

inline void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, ctr[0], hi0, lo0);
    mulhilo32(0xCD9E8D57u, ctr[2], hi1, lo1);
    const uint32_t n0 = hi1 ^ ctr[1] ^ key[0];
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ ctr[3] ^ key[1];
    const uint32_t n3 = lo0;
    ctr[0] = n0; ctr[1] = n1; ctr[2] = n2; ctr[3] = n3;
}

// Run all ten rounds for substream `substream` under `key64`. Output is the
// four mixed counter words ctr[0..3] (each a uniformly distributed uint32).
inline void philox4x32(uint64_t key64, uint64_t substream, uint32_t out[4]) {
    uint32_t key[2] = {
        static_cast<uint32_t>(key64 & 0xFFFFFFFFull),
        static_cast<uint32_t>(key64 >> 32),
    };
    uint32_t ctr[4] = {
        static_cast<uint32_t>(substream & 0xFFFFFFFFull),
        static_cast<uint32_t>(substream >> 32),
        0u, 0u,
    };
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr, key);
        if (r < 9) {
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
    }
    out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}

// Top 24 bits / 2^24 -> uniform in [0, 1). Matches xavier_init / sample_logits.
inline float u01_from(uint32_t w) {
    return static_cast<float>(w >> 8) * (1.0f / 16777216.0f);
}

// One uniform draw for substream `substream` (use ctr[0] only).
inline float philox_uniform(uint64_t key64, uint64_t substream) {
    uint32_t ctr[4];
    philox4x32(key64, substream, ctr);
    return u01_from(ctr[0]);
}

// One standard-normal draw via Box-Muller on (ctr[0], ctr[1]). We use
// 1 - u1 so log() stays finite for u1 == 0.
inline float philox_normal(uint64_t key64, uint64_t substream) {
    uint32_t ctr[4];
    philox4x32(key64, substream, ctr);
    const float u1 = 1.0f - u01_from(ctr[0]);   // (0, 1]
    const float u2 = u01_from(ctr[1]);          // [0, 1)
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float theta =
        static_cast<float>(kTwoPi * static_cast<double>(u2));
    return radius * std::cos(theta);
}

// Common entry validation for the four ops.
inline std::size_t check_y(const char* op, const ::brotensor::Tensor& Y) {
    if (Y.dtype != ::brotensor::Dtype::FP32) {
        fail(op, "Y must be FP32");
    }
    if (Y.rows < 0 || Y.cols < 0) {
        fail(op, "Y has negative dimension");
    }
    const std::size_t n = static_cast<std::size_t>(Y.rows) *
                          static_cast<std::size_t>(Y.cols);
    if (n != 0 && Y.data == nullptr) {
        fail(op, "Y is uncommitted; pre-allocate before calling");
    }
    return n;
}

} // namespace

// ─── randn ──────────────────────────────────────────────────────────────────
void randn(uint64_t key, uint64_t counter, ::brotensor::Tensor& Y) {
    const std::size_t n = check_y("randn", Y);
    if (n == 0) return;
    float* y = Y.host_f32_mut();
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = philox_normal(key, counter + static_cast<uint64_t>(i));
    }
}

// ─── rand_uniform ───────────────────────────────────────────────────────────
void rand_uniform(uint64_t key, uint64_t counter, ::brotensor::Tensor& Y) {
    const std::size_t n = check_y("rand_uniform", Y);
    if (n == 0) return;
    float* y = Y.host_f32_mut();
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = philox_uniform(key, counter + static_cast<uint64_t>(i));
    }
}

// ─── rand_bernoulli ─────────────────────────────────────────────────────────
void rand_bernoulli(float p, uint64_t key, uint64_t counter,
                    ::brotensor::Tensor& Y) {
    const char* op = "rand_bernoulli";
    if (!(p >= 0.0f && p <= 1.0f)) {
        fail(op, "p must be in [0, 1]");
    }
    const std::size_t n = check_y(op, Y);
    if (n == 0) return;
    float* y = Y.host_f32_mut();
    for (std::size_t i = 0; i < n; ++i) {
        const float u = philox_uniform(key, counter + static_cast<uint64_t>(i));
        y[i] = (u < p) ? 1.0f : 0.0f;
    }
}

// ─── randn_truncated ────────────────────────────────────────────────────────
void randn_truncated(float lo, float hi, uint64_t key, uint64_t counter,
                     ::brotensor::Tensor& Y) {
    const char* op = "randn_truncated";
    if (!(lo < hi)) fail(op, "lo must be < hi");
    const std::size_t n = check_y(op, Y);
    if (n == 0) return;
    float* y = Y.host_f32_mut();
    constexpr int kMaxRetries = 64;
    for (std::size_t i = 0; i < n; ++i) {
        float z = 0.0f;
        for (int r = 0; r < kMaxRetries; ++r) {
            const uint64_t sub = counter + static_cast<uint64_t>(i) +
                                 static_cast<uint64_t>(r) *
                                 static_cast<uint64_t>(n);
            z = philox_normal(key, sub);
            if (z >= lo && z <= hi) break;
        }
        if (z < lo) z = lo;
        if (z > hi) z = hi;
        y[i] = z;
    }
}

} // namespace brotensor::detail::cpu
