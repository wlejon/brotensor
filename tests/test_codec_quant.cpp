// Standalone CPU coverage for the brosoundml codec quantization ops
// (CHUNK 5, family D):
//   vq_encode_forward / vq_encode_backward      — vector quantization
//   fsq_quantize_forward / fsq_quantize_backward — finite scalar quantization
//
// Verifies:
//   * vq_encode_forward picks the L2-nearest codeword against a brute-force
//     reference, and the quantized rows equal codebook[indices[n], :].
//   * vq_encode tie-breaking keeps the lowest index.
//   * vq_encode indices feed straight into embedding_lookup_forward as a
//     decode step (round trip: encode -> decode == quantized).
//   * fsq_quantize round-trips a value already sitting on a level (it
//     quantizes to itself), bounds out-of-range inputs into [-1, 1], and the
//     packed mixed-radix index decodes back to the per-dim level tuple.
//   * Both backward ops are the straight-through identity: dX == dQuantized.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(double a, double b, double abs_eps, double rel_eps) {
    const double d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, abs_eps, rel_eps, ctx)                    \
    do {                                                                       \
        const double _a = (actual);                                            \
        const double _e = (expected);                                          \
        if (!near_(_a, _e, (abs_eps), (rel_eps))) {                            \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g\n",     \
                        __FILE__, __LINE__, (ctx), _a, _e);                     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                      \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_EQ_INT(actual, expected, ctx)                                   \
    do {                                                                       \
        const long long _a = (long long)(actual);                              \
        const long long _e = (long long)(expected);                            \
        if (_a != _e) {                                                        \
            std::printf("  FAIL  %s:%d  [%s]  actual=%lld expected=%lld\n",     \
                        __FILE__, __LINE__, (ctx), _a, _e);                     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// Tiny deterministic xorshift RNG so the test is reproducible.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    uint64_t next_u64() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    // Uniform in [-1, 1).
    float next_signed() {
        return static_cast<float>(
                   static_cast<double>(next_u64() >> 11) /
                   static_cast<double>(1ull << 53)) *
                   2.0f - 1.0f;
    }
};

static Tensor cpu_fp32(int r, int c) {
    return Tensor::zeros_on(Device::CPU, r, c, Dtype::FP32);
}

static Tensor cpu_int32(const std::vector<int32_t>& v) {
    Tensor t = Tensor::zeros_on(Device::CPU, static_cast<int>(v.size()), 1,
                                Dtype::INT32);
    auto* p = static_cast<int32_t*>(t.host_raw_mut());
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

// ─── vq_encode ──────────────────────────────────────────────────────────────

static void test_vq_encode_forward() {
    const int N = 17;
    const int D = 6;
    const int K = 11;
    Rng rng(0xABCDEF01u);

    Tensor x = cpu_fp32(N, D);
    Tensor codebook = cpu_fp32(K, D);
    for (int i = 0; i < N * D; ++i) x.host_f32_mut()[i] = rng.next_signed() * 3.0f;
    for (int i = 0; i < K * D; ++i)
        codebook.host_f32_mut()[i] = rng.next_signed() * 3.0f;

    Tensor indices;    // uncommitted — op resizes + sets INT32.
    Tensor quantized;  // uncommitted — op resizes + sets FP32.
    brotensor::vq_encode_forward(x, codebook, indices, quantized);

    EXPECT_EQ_INT(indices.rows, N, "vq indices rows");
    EXPECT_EQ_INT(indices.cols, 1, "vq indices cols");
    EXPECT_TRUE(indices.dtype == Dtype::INT32, "vq indices dtype INT32");
    EXPECT_EQ_INT(quantized.rows, N, "vq quantized rows");
    EXPECT_EQ_INT(quantized.cols, D, "vq quantized cols");
    EXPECT_TRUE(quantized.dtype == Dtype::FP32, "vq quantized dtype FP32");

    const int32_t* ip = static_cast<const int32_t*>(indices.host_raw());
    const float* xp = x.host_f32();
    const float* cp = codebook.host_f32();
    const float* qp = quantized.host_f32();

    for (int n = 0; n < N; ++n) {
        // Brute-force reference nearest codeword.
        double best = 1e300;
        int best_k = -1;
        for (int k = 0; k < K; ++k) {
            double d2 = 0.0;
            for (int j = 0; j < D; ++j) {
                const double diff = xp[n * D + j] - cp[k * D + j];
                d2 += diff * diff;
            }
            if (d2 < best) { best = d2; best_k = k; }
        }
        EXPECT_EQ_INT(ip[n], best_k, "vq nearest index");
        // Quantized row must equal the chosen codeword.
        for (int j = 0; j < D; ++j) {
            EXPECT_NEAR(qp[n * D + j], cp[ip[n] * D + j], 0.0, 0.0,
                        "vq quantized == codebook row");
        }
    }
}

static void test_vq_encode_tie_lowest_index() {
    // Two identical codewords; an input equidistant (exactly equal) must pick
    // the lowest index.
    const int D = 3;
    Tensor x = cpu_fp32(1, D);
    Tensor codebook = cpu_fp32(4, D);
    for (int j = 0; j < D; ++j) {
        x.host_f32_mut()[j] = 0.5f;
        // codewords 1 and 2 both exactly equal x.
        codebook.host_f32_mut()[0 * D + j] = 9.0f;
        codebook.host_f32_mut()[1 * D + j] = 0.5f;
        codebook.host_f32_mut()[2 * D + j] = 0.5f;
        codebook.host_f32_mut()[3 * D + j] = 9.0f;
    }
    Tensor indices, quantized;
    brotensor::vq_encode_forward(x, codebook, indices, quantized);
    const int32_t* ip = static_cast<const int32_t*>(indices.host_raw());
    EXPECT_EQ_INT(ip[0], 1, "vq tie keeps lowest index");
}

static void test_vq_decode_via_embedding() {
    // vq_encode indices decode back to vectors via embedding_lookup_forward;
    // the decoded rows must equal the quantized output.
    const int N = 9;
    const int D = 5;
    const int K = 7;
    Rng rng(0x13572468u);

    Tensor x = cpu_fp32(N, D);
    Tensor codebook = cpu_fp32(K, D);
    for (int i = 0; i < N * D; ++i) x.host_f32_mut()[i] = rng.next_signed() * 2.0f;
    for (int i = 0; i < K * D; ++i)
        codebook.host_f32_mut()[i] = rng.next_signed() * 2.0f;

    Tensor indices, quantized;
    brotensor::vq_encode_forward(x, codebook, indices, quantized);

    // The INT32 indices tensor's data pointer is the d_idx arg directly.
    const int32_t* d_idx = static_cast<const int32_t*>(indices.host_raw());
    Tensor decoded;
    brotensor::embedding_lookup_forward(codebook, d_idx, N, decoded);

    EXPECT_EQ_INT(decoded.rows, N, "vq decode rows");
    EXPECT_EQ_INT(decoded.cols, D, "vq decode cols");
    for (int i = 0; i < N * D; ++i) {
        EXPECT_NEAR(decoded.host_f32()[i], quantized.host_f32()[i], 0.0, 0.0,
                    "vq decode == quantized");
    }
}

static void test_vq_encode_backward_identity() {
    // Straight-through estimator: dX == dQuantized (overwrite, not accumulate).
    const int N = 8;
    const int D = 4;
    Rng rng(0xC0FFEEu);
    Tensor dQ = cpu_fp32(N, D);
    for (int i = 0; i < N * D; ++i) dQ.host_f32_mut()[i] = rng.next_signed();

    Tensor dX;  // uncommitted
    brotensor::vq_encode_backward(dQ, dX);
    EXPECT_EQ_INT(dX.rows, N, "vq bwd dX rows");
    EXPECT_EQ_INT(dX.cols, D, "vq bwd dX cols");
    for (int i = 0; i < N * D; ++i) {
        EXPECT_NEAR(dX.host_f32()[i], dQ.host_f32()[i], 0.0, 0.0,
                    "vq bwd STE identity");
    }

    // Pre-seed dX with garbage to confirm it is OVERWRITTEN, not accumulated.
    Tensor dX2 = cpu_fp32(N, D);
    for (int i = 0; i < N * D; ++i) dX2.host_f32_mut()[i] = 123.0f;
    brotensor::vq_encode_backward(dQ, dX2);
    for (int i = 0; i < N * D; ++i) {
        EXPECT_NEAR(dX2.host_f32()[i], dQ.host_f32()[i], 0.0, 0.0,
                    "vq bwd overwrites (not accumulates)");
    }
}

// ─── fsq_quantize ───────────────────────────────────────────────────────────

// Reference decode of a packed mixed-radix code back to the per-dim tuple.
static std::vector<int> fsq_unpack(long long packed,
                                   const std::vector<int32_t>& levels) {
    std::vector<int> tuple(levels.size(), 0);
    for (std::size_t d = 0; d < levels.size(); ++d) {
        const int L = levels[d];
        tuple[d] = static_cast<int>(packed % L);
        packed /= L;
    }
    return tuple;
}

static void test_fsq_round_trips_grid() {
    // Build x whose every coordinate already sits exactly on an FSQ level:
    //   level index i -> value i/h - 1, h = (L-1)/2.
    // Such a value must quantize to itself, and the packed index must decode
    // back to the (i_0, i_1, ...) tuple we put in.
    const std::vector<int32_t> levels_v = {3, 4, 5, 8};
    const int D = static_cast<int>(levels_v.size());
    Tensor levels = cpu_int32(levels_v);

    const int N = 6;
    Tensor x = cpu_fp32(N, D);
    // Per-row chosen level indices.
    std::vector<std::vector<int>> chosen(N, std::vector<int>(D));
    Rng rng(0x5151AAu);
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            const int L = levels_v[d];
            const int i = static_cast<int>(rng.next_u64() % L);
            chosen[n][d] = i;
            const float h = (L - 1) * 0.5f;
            x.host_f32_mut()[n * D + d] = static_cast<float>(i) / h - 1.0f;
        }
    }

    Tensor quantized, packed;  // uncommitted
    brotensor::fsq_quantize_forward(x, levels, quantized, packed);

    EXPECT_EQ_INT(quantized.rows, N, "fsq quantized rows");
    EXPECT_EQ_INT(quantized.cols, D, "fsq quantized cols");
    EXPECT_TRUE(quantized.dtype == Dtype::FP32, "fsq quantized dtype FP32");
    EXPECT_EQ_INT(packed.rows, N, "fsq packed rows");
    EXPECT_EQ_INT(packed.cols, 1, "fsq packed cols");
    EXPECT_TRUE(packed.dtype == Dtype::INT32, "fsq packed dtype INT32");

    const int32_t* pp = static_cast<const int32_t*>(packed.host_raw());
    for (int n = 0; n < N; ++n) {
        // A value already on a level round-trips to itself.
        for (int d = 0; d < D; ++d) {
            EXPECT_NEAR(quantized.host_f32()[n * D + d], x.host_f32()[n * D + d],
                        1e-6, 1e-6, "fsq on-grid round-trips to itself");
        }
        // Packed index decodes back to the per-dim level tuple.
        std::vector<int> got = fsq_unpack(pp[n], levels_v);
        for (int d = 0; d < D; ++d) {
            EXPECT_EQ_INT(got[d], chosen[n][d], "fsq packed decodes to tuple");
        }
        // Cross-check the explicit Horner formula:
        //   packed = i0 + L0*(i1 + L1*(i2 + L2*i3))
        long long ref = 0;
        for (int d = D - 1; d >= 0; --d)
            ref = ref * levels_v[d] + chosen[n][d];
        EXPECT_EQ_INT(pp[n], ref, "fsq packed == mixed-radix reference");
    }
}

static void test_fsq_bounds_and_rounding() {
    // Out-of-range inputs clamp into [-1, 1]; an arbitrary in-range value
    // snaps to its nearest level.
    const std::vector<int32_t> levels_v = {5};  // levels: -1, -0.5, 0, 0.5, 1
    Tensor levels = cpu_int32(levels_v);

    const int N = 5;
    Tensor x = cpu_fp32(N, 1);
    x.host_f32_mut()[0] = -10.0f;   // clamps to -1  -> level 0
    x.host_f32_mut()[1] = 10.0f;    // clamps to +1  -> level 4
    x.host_f32_mut()[2] = 0.24f;    // nearest 0.5? -> (0.24+1)/2*4 = 2.48 -> 2 -> 0.0
    x.host_f32_mut()[3] = 0.26f;    // (0.26+1)/2*4 = 2.52 -> 3 -> 0.5
    x.host_f32_mut()[4] = 0.0f;     // exactly level 2 -> 0.0

    Tensor quantized, packed;
    brotensor::fsq_quantize_forward(x, levels, quantized, packed);
    const float* q = quantized.host_f32();
    EXPECT_NEAR(q[0], -1.0f, 1e-6, 1e-6, "fsq clamp low");
    EXPECT_NEAR(q[1],  1.0f, 1e-6, 1e-6, "fsq clamp high");
    EXPECT_NEAR(q[2],  0.0f, 1e-6, 1e-6, "fsq round down");
    EXPECT_NEAR(q[3],  0.5f, 1e-6, 1e-6, "fsq round up");
    EXPECT_NEAR(q[4],  0.0f, 1e-6, 1e-6, "fsq exact level");

    const int32_t* pp = static_cast<const int32_t*>(packed.host_raw());
    EXPECT_EQ_INT(pp[0], 0, "fsq packed clamp low");
    EXPECT_EQ_INT(pp[1], 4, "fsq packed clamp high");
    EXPECT_EQ_INT(pp[2], 2, "fsq packed round down");
    EXPECT_EQ_INT(pp[3], 3, "fsq packed round up");
    EXPECT_EQ_INT(pp[4], 2, "fsq packed exact level");
}

static void test_fsq_quantize_backward_identity() {
    // Straight-through estimator: dX == dQuantized (overwrite, not accumulate).
    const int N = 7;
    const int D = 5;
    Rng rng(0xBADF00Du);
    Tensor dQ = cpu_fp32(N, D);
    for (int i = 0; i < N * D; ++i) dQ.host_f32_mut()[i] = rng.next_signed();

    Tensor dX;  // uncommitted
    brotensor::fsq_quantize_backward(dQ, dX);
    EXPECT_EQ_INT(dX.rows, N, "fsq bwd dX rows");
    EXPECT_EQ_INT(dX.cols, D, "fsq bwd dX cols");
    for (int i = 0; i < N * D; ++i) {
        EXPECT_NEAR(dX.host_f32()[i], dQ.host_f32()[i], 0.0, 0.0,
                    "fsq bwd STE identity");
    }

    // Pre-seed dX with garbage to confirm overwrite (not accumulate).
    Tensor dX2 = cpu_fp32(N, D);
    for (int i = 0; i < N * D; ++i) dX2.host_f32_mut()[i] = -777.0f;
    brotensor::fsq_quantize_backward(dQ, dX2);
    for (int i = 0; i < N * D; ++i) {
        EXPECT_NEAR(dX2.host_f32()[i], dQ.host_f32()[i], 0.0, 0.0,
                    "fsq bwd overwrites (not accumulates)");
    }
}

int main() {
    brotensor::init();
  try {
    test_vq_encode_forward();
    test_vq_encode_tie_lowest_index();
    test_vq_decode_via_embedding();
    test_vq_encode_backward_identity();

    test_fsq_round_trips_grid();
    test_fsq_bounds_and_rounding();
    test_fsq_quantize_backward_identity();
  } catch (const std::exception& e) {
    std::printf("test_codec_quant: uncaught exception: %s\n", e.what());
    return 2;
  }

    if (g_failures == 0) {
        std::printf("test_codec_quant: all checks passed\n");
        return 0;
    }
    std::printf("test_codec_quant: %d FAILURE(S)\n", g_failures);
    return 1;
}
