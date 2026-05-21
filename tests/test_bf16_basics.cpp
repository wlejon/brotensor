// Host-side unit tests for the BF16 conversion routines:
//   brotensor::fp32_to_bf16_bits  — FP32 → BF16 (round-to-nearest-even)
//   brotensor::bf16_bits_to_fp32  — BF16 → FP32 (lossless widening)
//
// These run on the CPU with no GPU backend — they exercise the scalar
// conversion helpers that every BF16 path (host plumbing, safetensors,
// parity-test helpers) is built on. Edge cases: exact values, sign of zero,
// ±Inf, NaN, round-to-nearest vs truncation, tie-to-even, denormals, and the
// half-ULP relative-error bound.

#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using brotensor::fp32_to_bf16_bits;
using brotensor::bf16_bits_to_fp32;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// Round x to BF16 and widen straight back to FP32.
static float roundtrip(float x) {
    return bf16_bits_to_fp32(fp32_to_bf16_bits(x));
}

int main() {
    std::printf("test_bf16_basics\n");

    // ── Exactly representable values round-trip with no loss ───────────────
    // Integers 1..4, powers of two and their negatives all fit in BF16's
    // 8-bit significand exactly.
    for (float v : {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 0.25f, -1.0f, -2.0f,
                    -0.5f, 128.0f, -256.0f}) {
        CHECK(roundtrip(v) == v);
    }

    // ── Sign of zero is preserved ──────────────────────────────────────────
    CHECK(roundtrip(0.0f) == 0.0f);
    CHECK(roundtrip(-0.0f) == 0.0f);
    CHECK(std::signbit(roundtrip(-0.0f)));
    CHECK(!std::signbit(roundtrip(0.0f)));

    // ── Infinities survive intact ──────────────────────────────────────────
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(roundtrip(inf) == inf);
    CHECK(roundtrip(-inf) == -inf);

    // ── NaN maps to a NaN (never to Inf or a finite value) ─────────────────
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(std::isnan(roundtrip(nan)));
    // A NaN with a non-trivial payload must not collapse to Inf via a
    // rounding carry into the exponent.
    uint32_t nan_bits = 0x7F800001u;  // exponent all-ones, mantissa = 1
    float payload_nan;
    std::memcpy(&payload_nan, &nan_bits, sizeof(float));
    CHECK(std::isnan(roundtrip(payload_nan)));

    // ── bf16_bits_to_fp32 is the inverse of the exact path ─────────────────
    // The result of a round is itself BF16-representable, so re-rounding it
    // is idempotent.
    for (float v : {1.005859375f, -7.1f, 1e6f, 3.14159265f, 1e-4f}) {
        const float r = roundtrip(v);
        CHECK(roundtrip(r) == r);
    }

    // ── Round-to-nearest, not truncation ───────────────────────────────────
    // The BF16 grid step just above 1.0 is 2^-7 = 1/128. A value 3/4 of the
    // way up must round UP (truncation would round down).
    const float step = 1.0f / 128.0f;
    CHECK(roundtrip(1.0f + 0.75f * step) == 1.0f + step);
    CHECK(roundtrip(1.0f + 0.25f * step) == 1.0f);

    // ── Tie-to-even at the exact half-way point ────────────────────────────
    // 1.0 has an even significand (LSB 0): a tie rounds DOWN to it.
    CHECK(roundtrip(1.0f + 0.5f * step) == 1.0f);
    // 1.0 + step has an odd significand (LSB 1): a tie rounds UP to the even
    // neighbour 1.0 + 2*step.
    CHECK(roundtrip((1.0f + step) + 0.5f * step) == 1.0f + 2.0f * step);

    // ── Half-ULP relative-error bound over a deterministic sweep ───────────
    // Round-to-nearest keeps the error within half a BF16 ULP; the ULP is
    // 2^-7 relative, so |err| <= 2^-8 * |x|.
    uint64_t s = 0x9E3779B97F4A7C15ull;
    int swept = 0;
    for (int i = 0; i < 4096; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        // Map to a value in roughly [-1e3, 1e3], avoiding subnormals.
        const float frac = static_cast<float>(s >> 40) / 16777216.0f;  // [0,1)
        const float x = (frac * 2.0f - 1.0f) * 1000.0f;
        const float r = roundtrip(x);
        const float err = std::fabs(r - x);
        CHECK(err <= (1.0f / 256.0f) * std::fabs(x) + 1e-30f);
        ++swept;
    }
    CHECK(swept == 4096);

    // ── Small (subnormal-ish) magnitudes still round sanely ────────────────
    // BF16's smallest normal is 2^-126; values near it must not become Inf
    // or NaN.
    for (float v : {1e-30f, -1e-30f, 1e-20f, 5e-39f}) {
        const float r = roundtrip(v);
        CHECK(std::isfinite(r));
        CHECK(std::signbit(r) == std::signbit(v) || r == 0.0f);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "\nbf16_basics: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("bf16_basics: OK\n");
    return 0;
}
