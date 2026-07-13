// ─── CPU-only coverage for matmul_abt (batched A @ B^T, FP16 / BF16) ───────
//
// tests/test_matmul_abt.cpp is CUDA-gated, so the CPU implementation in
// src/cpu/matmul.cpp never executed. This drives it directly on Device::CPU.
//
// The op computes, per batch b:
//     C[b][m,n] = act( sum_k A[b][m,k] * B[b][n,k]  (+ bias[n]) )
// with FP32 accumulation, FP16/BF16 storage, element strides between batch
// slices, an optional per-N bias, and an activation code
// (0 none, 1 relu, 2 gelu-tanh, 3 gelu-exact, 4 silu, 5 quick-gelu; anything
// else is identity). C is NOT auto-resized — the caller pre-sizes it.
//
// Coverage:
//   1. FP16 + BF16 × batch=1 / batch>1 × bias / no bias × act 0..5.
//   2. An unrecognised act code falls through to identity.
//   3. Non-trivial (padded) strides for A, B and C; bytes of C outside the
//      written (M, N) tiles must be left untouched.
//   4. batch=1 with all-zero strides — the documented non-batched case.
//   5. K == 0 is not an early-out: C = act(bias[n]) / act(0).
//   6. Degenerate early-outs (batch <= 0, M == 0, N == 0) leave C untouched.
//   7. Error paths: A/B/C dtype disagreement, a non-16-bit dtype, and a bias
//      whose dtype does not match the operands.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

// ─── 16-bit storage helpers ────────────────────────────────────────────────

static uint16_t enc(Dtype dt, float v) {
    return dt == Dtype::FP16 ? brotensor::fp32_to_fp16_bits(v)
                             : brotensor::fp32_to_bf16_bits(v);
}

static float dec(Dtype dt, uint16_t bits) {
    return dt == Dtype::FP16 ? brotensor::fp16_bits_to_fp32(bits)
                             : brotensor::bf16_bits_to_fp32(bits);
}

static uint16_t* h16_mut(Tensor& t) {
    return t.dtype == Dtype::FP16 ? t.host_fp16_mut() : t.host_bf16_mut();
}

static const uint16_t* h16(const Tensor& t) {
    return t.dtype == Dtype::FP16 ? t.host_fp16() : t.host_bf16();
}

static const char* dtype_label(Dtype dt) {
    return dt == Dtype::FP16 ? "fp16" : "bf16";
}

// Independent reference for the activation codes the op honours.
static float ref_act(int act, float v) {
    switch (act) {
        case 0: return v;
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: {  // gelu, tanh approximation
            const float c = 0.7978845608028654f;  // sqrt(2/pi)
            const float t = c * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(t));
        }
        case 3:  // gelu, exact
            return 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f));
        case 4:  // silu
            return v / (1.0f + std::exp(-v));
        case 5:  // quick gelu
            return v / (1.0f + std::exp(-1.702f * v));
        default: return v;  // unrecognised code == identity
    }
}

// Allocate a CPU tensor of `dt` holding `bits` laid out as a single row.
static Tensor make16(Dtype dt, const std::vector<uint16_t>& bits) {
    Tensor t = Tensor::zeros_on(Device::CPU, 1,
                                static_cast<int>(bits.size()), dt);
    if (bits.empty()) return t;
    uint16_t* p = h16_mut(t);
    for (std::size_t i = 0; i < bits.size(); ++i) p[i] = bits[i];
    return t;
}

// ─── the workhorse ─────────────────────────────────────────────────────────
//
// Builds A / B / (bias) / C on the CPU, runs matmul_abt, and compares every
// written element against a straight FP32 reference computed from the *stored*
// (already-rounded) operand bits — so the only error left is the final
// round-to-16-bit of the result. C is pre-filled with a sentinel; every slot
// the op is not supposed to write must still hold it afterwards.
static void run_case(Dtype dt, int batch, int M, int N, int K,
                     long long strideA, long long strideB, long long strideC,
                     bool with_bias, int act, const char* label) {
    std::printf("  %-22s %s batch=%d M=%d N=%d K=%d bias=%d act=%d\n",
                label, dtype_label(dt), batch, M, N, K,
                with_bias ? 1 : 0, act);

    // Minimum span each buffer must cover, plus 4 trailing sentinel slots on C
    // so an overrun past the last batch slice is caught too.
    const long long spanA =
        static_cast<long long>(batch - 1) * strideA + static_cast<long long>(M) * K;
    const long long spanB =
        static_cast<long long>(batch - 1) * strideB + static_cast<long long>(N) * K;
    const long long spanC =
        static_cast<long long>(batch - 1) * strideC +
        static_cast<long long>(M) * N + 4;

    std::mt19937 rng(0xAB70u ^ static_cast<unsigned>(
        batch * 1000003 + M * 10007 + N * 101 + K * 7 + act * 3 +
        (dt == Dtype::FP16 ? 1 : 2) + (with_bias ? 512 : 0)));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> Abits(static_cast<std::size_t>(spanA));
    std::vector<uint16_t> Bbits(static_cast<std::size_t>(spanB));
    std::vector<uint16_t> biasbits(static_cast<std::size_t>(N));
    for (auto& v : Abits)    v = enc(dt, dist(rng));
    for (auto& v : Bbits)    v = enc(dt, dist(rng));
    for (auto& v : biasbits) v = enc(dt, dist(rng));

    Tensor A = make16(dt, Abits);
    Tensor B = make16(dt, Bbits);
    Tensor bias = make16(dt, biasbits);

    const uint16_t sentinel = enc(dt, -7.5f);  // exact in both fp16 and bf16
    std::vector<uint16_t> Cinit(static_cast<std::size_t>(spanC), sentinel);
    Tensor C = make16(dt, Cinit);

    brotensor::matmul_abt(A, B, C, batch, M, N, K,
                          strideA, strideB, strideC,
                          with_bias ? &bias : nullptr, act);

    const uint16_t* Cp = h16(C);

    // Which slots the op is contractually allowed to write.
    std::vector<char> written(static_cast<std::size_t>(spanC), 0);

    // BF16 carries ~8 mantissa bits; FP16 ~11. Tolerances cover the final
    // store rounding (the accumulation itself is FP32 in both impl and ref).
    const float atol = (dt == Dtype::FP16) ? 2e-2f : 6e-2f;
    const float rtol = (dt == Dtype::FP16) ? 1e-2f : 5e-2f;

    float max_err = 0.0f;
    for (int b = 0; b < batch; ++b) {
        const std::size_t oa = static_cast<std::size_t>(b * strideA);
        const std::size_t ob = static_cast<std::size_t>(b * strideB);
        const std::size_t oc = static_cast<std::size_t>(b * strideC);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += dec(dt, Abits[oa + static_cast<std::size_t>(m) * K + k]) *
                           dec(dt, Bbits[ob + static_cast<std::size_t>(n) * K + k]);
                }
                if (with_bias) acc += dec(dt, biasbits[static_cast<std::size_t>(n)]);
                const float want = ref_act(act, acc);

                const std::size_t ci = oc + static_cast<std::size_t>(m) * N + n;
                written[ci] = 1;
                const float got = dec(dt, Cp[ci]);
                const float err = std::fabs(got - want);
                if (err > max_err) max_err = err;
                CHECK(err <= atol + rtol * std::fabs(want));
            }
        }
    }

    // Stride padding + trailing slack: untouched.
    for (std::size_t i = 0; i < written.size(); ++i) {
        if (!written[i]) CHECK(Cp[i] == sentinel);
    }
    std::printf("    max_err=%g\n", static_cast<double>(max_err));
}

// ─── 1. FP16 / BF16 × batch × bias × every activation code ─────────────────
static void test_acts_and_dtypes() {
    std::printf("test_acts_and_dtypes\n");
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        for (int act = 0; act <= 5; ++act) {
            // batch = 1, tight strides, no bias.
            run_case(dt, /*batch=*/1, /*M=*/3, /*N=*/4, /*K=*/8,
                     /*strideA=*/3 * 8, /*strideB=*/4 * 8, /*strideC=*/3 * 4,
                     /*with_bias=*/false, act, "batch1-tight");
            // batch = 3, padded strides, with bias.
            run_case(dt, /*batch=*/3, /*M=*/2, /*N=*/5, /*K=*/4,
                     /*strideA=*/2 * 4 + 3, /*strideB=*/5 * 4 + 2,
                     /*strideC=*/2 * 5 + 7,
                     /*with_bias=*/true, act, "batch3-padded-bias");
        }
    }
}

// ─── 2. an unrecognised activation code is the identity ────────────────────
static void test_unknown_act_is_identity() {
    std::printf("test_unknown_act_is_identity\n");
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        run_case(dt, /*batch=*/2, /*M=*/2, /*N=*/3, /*K=*/6,
                 /*strideA=*/2 * 6 + 1, /*strideB=*/3 * 6 + 1,
                 /*strideC=*/2 * 3 + 1,
                 /*with_bias=*/true, /*act=*/99, "unknown-act");
    }
}

// ─── 3. batch=1 with zero strides (the documented non-batched case) ────────
static void test_zero_strides() {
    std::printf("test_zero_strides\n");
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        run_case(dt, /*batch=*/1, /*M=*/4, /*N=*/3, /*K=*/16,
                 /*strideA=*/0, /*strideB=*/0, /*strideC=*/0,
                 /*with_bias=*/true, /*act=*/1, "batch1-zero-stride");
    }
}

// ─── 4. K == 0: the k loop never runs, so C = act(bias[n]) / act(0) ────────
static void test_k_zero() {
    std::printf("test_k_zero\n");
    const int M = 2, N = 3;
    const float bvals[3] = {0.5f, -0.25f, 2.0f};
    for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        // A / B are never dereferenced when K == 0, but must still carry the
        // operand dtype (the impl takes a typed host pointer up front).
        Tensor A = Tensor::zeros_on(Device::CPU, 1, 1, dt);
        Tensor B = Tensor::zeros_on(Device::CPU, 1, 1, dt);
        Tensor C = Tensor::zeros_on(Device::CPU, 1, M * N, dt);

        std::vector<uint16_t> bbits(static_cast<std::size_t>(N));
        for (int n = 0; n < N; ++n) bbits[static_cast<std::size_t>(n)] = enc(dt, bvals[n]);
        Tensor bias = make16(dt, bbits);

        // act = 1 (relu) so the negative bias entry clamps to zero.
        brotensor::matmul_abt(A, B, C, /*batch=*/1, M, N, /*K=*/0,
                              0, 0, 0, &bias, /*act=*/1);
        const uint16_t* Cp = h16(C);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const float want =
                    ref_act(1, dec(dt, bbits[static_cast<std::size_t>(n)]));
                const float got = dec(dt, Cp[static_cast<std::size_t>(m) * N + n]);
                CHECK(std::fabs(got - want) <= 1e-2f + 1e-2f * std::fabs(want));
            }
        }

        // No bias: acc stays 0, and every activation maps 0 -> 0 exactly.
        for (int act = 0; act <= 5; ++act) {
            brotensor::matmul_abt(A, B, C, /*batch=*/1, M, N, /*K=*/0,
                                  0, 0, 0, nullptr, act);
            const uint16_t* p = h16(C);
            for (int i = 0; i < M * N; ++i) {
                CHECK(dec(dt, p[static_cast<std::size_t>(i)]) == 0.0f);
            }
        }
    }
}

// ─── 5. degenerate shapes return before touching C ─────────────────────────
static void test_degenerate_shapes() {
    std::printf("test_degenerate_shapes\n");
    const Dtype dt = Dtype::FP16;
    const int n_slots = 8;
    Tensor A = Tensor::zeros_on(Device::CPU, 1, 8, dt);
    Tensor B = Tensor::zeros_on(Device::CPU, 1, 8, dt);
    Tensor C = Tensor::zeros_on(Device::CPU, 1, n_slots, dt);
    const uint16_t sentinel = enc(dt, -7.5f);

    auto reset = [&]() {
        uint16_t* p = h16_mut(C);
        for (int i = 0; i < n_slots; ++i) p[i] = sentinel;
    };
    auto untouched = [&]() {
        const uint16_t* p = h16(C);
        for (int i = 0; i < n_slots; ++i) {
            if (p[i] != sentinel) return false;
        }
        return true;
    };

    reset();
    brotensor::matmul_abt(A, B, C, /*batch=*/0, 2, 2, 2, 4, 4, 4, nullptr, 0);
    CHECK(untouched());

    reset();
    brotensor::matmul_abt(A, B, C, /*batch=*/-1, 2, 2, 2, 4, 4, 4, nullptr, 0);
    CHECK(untouched());

    reset();
    brotensor::matmul_abt(A, B, C, 1, /*M=*/0, 2, 2, 4, 4, 4, nullptr, 0);
    CHECK(untouched());

    reset();
    brotensor::matmul_abt(A, B, C, 1, 2, /*N=*/0, 2, 4, 4, 4, nullptr, 0);
    CHECK(untouched());
}

// ─── 6. error paths ────────────────────────────────────────────────────────
static void test_dtype_errors() {
    std::printf("test_dtype_errors\n");
    Tensor A16 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP16);
    Tensor B16 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP16);
    Tensor C16 = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP16);
    Tensor Bbf = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::BF16);
    Tensor Cbf = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::BF16);

    // C's dtype disagrees with A/B.
    bool threw = false;
    try {
        brotensor::matmul_abt(A16, B16, Cbf, 1, 2, 2, 2, 4, 4, 4, nullptr, 0);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // B's dtype disagrees with A.
    threw = false;
    try {
        brotensor::matmul_abt(A16, Bbf, C16, 1, 2, 2, 2, 4, 4, 4, nullptr, 0);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // A consistent but unsupported dtype (FP32) is rejected.
    Tensor Af = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP32);
    Tensor Bf = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP32);
    Tensor Cf = Tensor::zeros_on(Device::CPU, 2, 2, Dtype::FP32);
    threw = false;
    try {
        brotensor::matmul_abt(Af, Bf, Cf, 1, 2, 2, 2, 4, 4, 4, nullptr, 0);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // A bias whose dtype does not match the operands.
    Tensor bias_bf = Tensor::zeros_on(Device::CPU, 1, 2, Dtype::BF16);
    threw = false;
    try {
        brotensor::matmul_abt(A16, B16, C16, 1, 2, 2, 2, 4, 4, 4, &bias_bf, 0);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Sanity: the same call with a matching bias dtype succeeds.
    Tensor bias_16 = Tensor::zeros_on(Device::CPU, 1, 2, Dtype::FP16);
    threw = false;
    try {
        brotensor::matmul_abt(A16, B16, C16, 1, 2, 2, 2, 4, 4, 4, &bias_16, 0);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(!threw);
}

int main() {
    brotensor::init();
    brotensor::DeviceScope cpu_only(Device::CPU);
    std::printf("test_matmul_abt_cpu\n");

    test_acts_and_dtypes();
    test_unknown_act_is_identity();
    test_zero_strides();
    test_k_zero();
    test_degenerate_shapes();
    test_dtype_errors();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll matmul_abt CPU checks passed.\n");
    return 0;
}
