// Standalone CPU coverage for the brosoundml autoregressive logit sampler
// (CHUNK 7, family F):
//   sample_logits — temperature / top-k / top-p (nucleus) next-token sampling.
//
// Verifies:
//   * temperature == 0 returns the per-row argmax (greedy, no RNG).
//   * determinism — identical (key, counter) => identical output; a different
//     counter generally produces a different draw.
//   * top_k == 1 always returns the argmax regardless of the RNG.
//   * top-p (nucleus) filtering keeps the correct set on a hand-built
//     distribution (a sharp two-token nucleus is never escaped).
//   * statistical: over many draws on a known 2- and 3-token distribution the
//     empirical token frequencies approximately match the softmax probs.
//   * per-row substream independence: a given row's draw depends only on
//     (key, counter + row), never on the other rows or the row count N.
//
// CPU-resident, FP32 input / INT32 output. Plain executable; the main() body
// is wrapped in try/catch and returns non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                     \
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

// ─── helpers ────────────────────────────────────────────────────────────────

// Build an (R, C) FP32 logit tensor from a flat row-major vector.
static Tensor logits_from(int R, int C, const std::vector<float>& v) {
    Tensor t = Tensor::zeros_on(Device::CPU, R, C, Dtype::FP32);
    float* p = t.host_f32_mut();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

static int32_t idx_at(const Tensor& indices, int row) {
    return static_cast<const int32_t*>(indices.host_raw())[row];
}

// ─── tests ──────────────────────────────────────────────────────────────────

// temperature == 0 returns the per-row argmax (ties -> lowest index).
static void test_greedy_argmax() {
    // 3 rows, vocab 5; known argmax per row.
    Tensor logits = logits_from(3, 5, {
        0.1f, 2.5f, -1.0f, 0.3f, 1.2f,   // argmax 1
        4.0f, 4.0f, 1.0f, 0.0f, -2.0f,   // tie at 0 & 1 -> 0
        -3.0f, -1.0f, -0.5f, 9.9f, 0.0f, // argmax 3
    });
    Tensor indices;
    brotensor::sample_logits(logits, /*temperature=*/0.0f, /*top_k=*/0,
                             /*top_p=*/1.0f, /*key=*/123, /*counter=*/0,
                             indices);
    EXPECT_EQ_INT(indices.rows, 3, "greedy: indices.rows");
    EXPECT_EQ_INT(indices.cols, 1, "greedy: indices.cols");
    EXPECT_TRUE(indices.dtype == Dtype::INT32, "greedy: indices dtype INT32");
    EXPECT_EQ_INT(idx_at(indices, 0), 1, "greedy row0 argmax");
    EXPECT_EQ_INT(idx_at(indices, 1), 0, "greedy row1 tie -> lowest");
    EXPECT_EQ_INT(idx_at(indices, 2), 3, "greedy row2 argmax");
}

// Same (key, counter) => identical output; different counter generally differs.
static void test_determinism() {
    // A diffuse distribution so the draw is genuinely random.
    Tensor logits = logits_from(1, 8, {
        0.5f, 0.4f, 0.6f, 0.45f, 0.55f, 0.5f, 0.5f, 0.5f,
    });
    Tensor a, b, c;
    brotensor::sample_logits(logits, 1.0f, 0, 1.0f, /*key=*/777, /*counter=*/10, a);
    brotensor::sample_logits(logits, 1.0f, 0, 1.0f, /*key=*/777, /*counter=*/10, b);
    EXPECT_EQ_INT(idx_at(a, 0), idx_at(b, 0),
                  "determinism: same key/counter -> identical draw");

    // Different counter values should not all collapse to one token: sweep a
    // range and confirm at least two distinct results appear.
    int distinct_seen[8] = {0,0,0,0,0,0,0,0};
    int n_distinct = 0;
    for (uint64_t ctr = 0; ctr < 64; ++ctr) {
        brotensor::sample_logits(logits, 1.0f, 0, 1.0f, 777, ctr, c);
        const int tok = idx_at(c, 0);
        if (tok >= 0 && tok < 8 && !distinct_seen[tok]) {
            distinct_seen[tok] = 1;
            ++n_distinct;
        }
    }
    EXPECT_TRUE(n_distinct >= 2,
                "determinism: distinct counters produce varied draws");
}

// top_k == 1 collapses to the argmax regardless of the RNG.
static void test_top_k_one() {
    Tensor logits = logits_from(2, 6, {
        1.0f, 1.0f, 5.0f, 1.0f, 1.0f, 1.0f,    // argmax 2
        2.0f, 8.0f, 2.0f, 2.0f, 2.0f, 2.0f,    // argmax 1
    });
    for (uint64_t ctr = 0; ctr < 32; ++ctr) {
        Tensor indices;
        brotensor::sample_logits(logits, 1.0f, /*top_k=*/1, 1.0f, 42, ctr,
                                 indices);
        EXPECT_EQ_INT(idx_at(indices, 0), 2, "top_k=1 row0 -> argmax");
        EXPECT_EQ_INT(idx_at(indices, 1), 1, "top_k=1 row1 -> argmax");
    }
}

// top-p nucleus: a hand-built sharp distribution where two tokens carry almost
// all the mass. With top_p = 0.9 only those two may ever be drawn.
static void test_top_p_nucleus() {
    // Logits chosen so softmax mass concentrates on tokens 4 and 2.
    //   token 4 highest, token 2 second; the rest negligible.
    Tensor logits = logits_from(1, 6, {
        -8.0f, -8.0f, 4.0f, -8.0f, 5.0f, -8.0f,
    });
    // softmax: p4 ~ e^5 / (e^5+e^4+~0) ~ 0.731, p2 ~ 0.269; cumulative of the
    // top two ~ 1.0, well past 0.9 — nucleus is exactly {4, 2}.
    for (uint64_t ctr = 0; ctr < 128; ++ctr) {
        Tensor indices;
        brotensor::sample_logits(logits, 1.0f, /*top_k=*/0, /*top_p=*/0.9f,
                                 99, ctr, indices);
        const int tok = idx_at(indices, 0);
        EXPECT_TRUE(tok == 4 || tok == 2,
                    "top_p nucleus draw stays inside {4,2}");
    }

    // A tiny top_p must still keep at least one token (the argmax).
    Tensor one;
    brotensor::sample_logits(logits, 1.0f, 0, /*top_p=*/0.0f, 99, 0, one);
    EXPECT_EQ_INT(idx_at(one, 0), 4, "top_p=0 keeps the argmax only");
}

// Statistical: empirical frequencies match softmax probabilities (loose).
static void test_statistical_frequencies() {
    // Two-token distribution: logits {ln(3), 0} over a vocab of 2 ->
    // p0 = 0.75, p1 = 0.25.
    {
        Tensor logits = logits_from(1, 2, { std::log(3.0f), 0.0f });
        const int draws = 20000;
        int count0 = 0;
        for (int i = 0; i < draws; ++i) {
            Tensor indices;
            brotensor::sample_logits(logits, 1.0f, 0, 1.0f, 2024,
                                     static_cast<uint64_t>(i), indices);
            if (idx_at(indices, 0) == 0) ++count0;
        }
        const double freq0 = static_cast<double>(count0) / draws;
        std::printf("  stat 2-token: p0=0.75 empirical=%.4f\n", freq0);
        EXPECT_TRUE(std::fabs(freq0 - 0.75) < 0.03,
                    "2-token empirical freq ~ 0.75");
    }
    // Three-token distribution: logits {ln(1), ln(2), ln(7)} over vocab 3 ->
    // p = {0.1, 0.2, 0.7}.
    {
        Tensor logits = logits_from(1, 3, {
            std::log(1.0f), std::log(2.0f), std::log(7.0f),
        });
        const int draws = 30000;
        int count[3] = {0, 0, 0};
        for (int i = 0; i < draws; ++i) {
            Tensor indices;
            brotensor::sample_logits(logits, 1.0f, 0, 1.0f, 31337,
                                     static_cast<uint64_t>(i), indices);
            const int tok = idx_at(indices, 0);
            if (tok >= 0 && tok < 3) ++count[tok];
        }
        const double f0 = static_cast<double>(count[0]) / draws;
        const double f1 = static_cast<double>(count[1]) / draws;
        const double f2 = static_cast<double>(count[2]) / draws;
        std::printf("  stat 3-token: p={0.1,0.2,0.7} empirical={%.4f,%.4f,%.4f}\n",
                    f0, f1, f2);
        EXPECT_TRUE(std::fabs(f0 - 0.1) < 0.03, "3-token freq0 ~ 0.1");
        EXPECT_TRUE(std::fabs(f1 - 0.2) < 0.03, "3-token freq1 ~ 0.2");
        EXPECT_TRUE(std::fabs(f2 - 0.7) < 0.03, "3-token freq2 ~ 0.7");
    }
}

// Per-row substream independence: row n's draw must depend only on
// (key, counter + n), not on N or the values of other rows.
static void test_row_substream_independence() {
    // A 4-row tensor; every row has the SAME diffuse logits. Row n must draw
    // exactly what a single-row call at counter (base + n) draws.
    const std::vector<float> row = {
        0.3f, 0.7f, 0.1f, 0.9f, 0.5f, 0.2f, 0.8f, 0.4f,
    };
    std::vector<float> flat;
    for (int r = 0; r < 4; ++r)
        for (float v : row) flat.push_back(v);
    Tensor multi = logits_from(4, 8, flat);

    const uint64_t key = 555;
    const uint64_t base = 1000;
    Tensor multi_idx;
    brotensor::sample_logits(multi, 1.0f, 0, 1.0f, key, base, multi_idx);

    for (int r = 0; r < 4; ++r) {
        Tensor single = logits_from(1, 8, row);
        Tensor single_idx;
        brotensor::sample_logits(single, 1.0f, 0, 1.0f, key,
                                 base + static_cast<uint64_t>(r), single_idx);
        EXPECT_EQ_INT(idx_at(multi_idx, r), idx_at(single_idx, 0),
                      "row substream == single-row at counter+n");
    }

    // The substream is independent of the OTHER rows' logits: change rows
    // 0,1,3 entirely and row 2's draw must be unchanged.
    std::vector<float> flat2 = flat;
    for (int r = 0; r < 4; ++r) {
        if (r == 2) continue;
        for (int c = 0; c < 8; ++c) flat2[r * 8 + c] = -static_cast<float>(c);
    }
    Tensor multi2 = logits_from(4, 8, flat2);
    Tensor multi2_idx;
    brotensor::sample_logits(multi2, 1.0f, 0, 1.0f, key, base, multi2_idx);
    EXPECT_EQ_INT(idx_at(multi2_idx, 2), idx_at(multi_idx, 2),
                  "row 2 draw independent of other rows' logits");
}

// Error handling: negative temperature / top_k / top_p must throw.
static void test_error_paths() {
    Tensor logits = logits_from(1, 4, { 1.0f, 2.0f, 3.0f, 4.0f });
    Tensor indices;
    bool threw;

    threw = false;
    try { brotensor::sample_logits(logits, -1.0f, 0, 1.0f, 0, 0, indices); }
    catch (const std::exception&) { threw = true; }
    EXPECT_TRUE(threw, "negative temperature throws");

    threw = false;
    try { brotensor::sample_logits(logits, 1.0f, -2, 1.0f, 0, 0, indices); }
    catch (const std::exception&) { threw = true; }
    EXPECT_TRUE(threw, "negative top_k throws");

    threw = false;
    try { brotensor::sample_logits(logits, 1.0f, 0, -0.5f, 0, 0, indices); }
    catch (const std::exception&) { threw = true; }
    EXPECT_TRUE(threw, "negative top_p throws");
}

int main() {
    try {
        brotensor::init();

        test_greedy_argmax();
        test_determinism();
        test_top_k_one();
        test_top_p_nucleus();
        test_statistical_frequencies();
        test_row_substream_independence();
        test_error_paths();

        if (g_failures == 0) {
            std::printf("test_sample_logits: ALL PASS\n");
            return 0;
        }
        std::printf("test_sample_logits: %d FAILURE(S)\n", g_failures);
        return 1;
    } catch (const std::exception& e) {
        std::printf("test_sample_logits: uncaught exception: %s\n", e.what());
        return 2;
    }
}
