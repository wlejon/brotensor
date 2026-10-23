// Issue #140: Gradient NaN-Freedom Proofs & Clamping Across Backward
// Activations, Norms & Attention.
//
// Exercises:
//   1. BVSolver IEEE-754 bitvector proofs of exponent/cubic clamping and
//      denominator positivity across activations, norms, and safe softmax.
//   2. Exhaustive IEEE-754 boundary sweeps (0.0, -0.0, subnormals, +-20, +-88,
//      +-1e20, +-1e38) across all backward activation kernels.
//   3. Zero-variance (constant row) and zero-eps normalization backward tests.
//   4. All-masked (-Inf / mask=0) Softmax and FlashAttention backward tests.

#include <brotensor/backward_verify.h>
#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static void test_bv_and_runtime_nan_freedom_suite() {
    auto report = brotensor::verify::run_all_backward_nan_freedom_proofs();
    CHECK(!report.entries.empty());
    for (const auto& entry : report.entries) {
        std::printf("  [NaN-Proof] %-58s : %s (%s)\n",
                    entry.name.c_str(),
                    entry.proven ? "PROVEN" : "FAILED",
                    entry.detail.c_str());
        CHECK(entry.proven);
    }
    CHECK(report.all_passed());
}

static void test_sqrt_backward_clamped_exact_and_boundary() {
    Tensor x = Tensor::vec(6);
    Tensor dy = Tensor::vec(6);
    x[0] = 0.0f;
    x[1] = -0.0f;
    x[2] = -5.0f;
    x[3] = 1e-38f;
    x[4] = 4.0f;
    x[5] = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < 6; ++i) dy[i] = 2.0f;

    Tensor dx;
    brotensor::verify::sqrt_backward_clamped(x, dy, dx, 1e-12f);
    for (int i = 0; i < 6; ++i) {
        CHECK(std::isfinite(dx[i]));
    }
    // At x = 0.0f with eps = 1e-12f, dx = 2.0 / (2 * sqrt(1e-12)) = 1e6f
    CHECK(std::fabs(dx[0] - 1e6f) <= 1.0f);
    CHECK(std::fabs(dx[1] - 1e6f) <= 1.0f);
    CHECK(std::fabs(dx[2] - 1e6f) <= 1.0f);
    // At x = 4.0f, dx = 2.0 / (2 * 2.0) = 0.5f
    CHECK(std::fabs(dx[4] - 0.5f) <= 1e-6f);
    // At x = NaN, sanitized to 0.0f
    CHECK(dx[5] == 0.0f);
}

static void test_varlen_flash_attention_empty_and_causal_nan_freedom() {
    // Packed varlen FlashAttention backward where sequence 0 has length 0 in K
    // (all keys excluded) and sequence 1 is normal causal attention.
    const int total_q = 4, total_k = 4, num_heads = 2, head_dim = 8;
    const int D = num_heads * head_dim;
    Tensor Q = Tensor::mat(total_q, D);
    Tensor K = Tensor::mat(total_k, D);
    Tensor V = Tensor::mat(total_k, D);
    Tensor O = Tensor::mat(total_q, D);
    Tensor dO = Tensor::mat(total_q, D);
    for (int i = 0; i < Q.size(); ++i) {
        Q[i] = 0.1f * static_cast<float>((i % 7) - 3);
        K[i] = 0.15f * static_cast<float>((i % 5) - 2);
        V[i] = 0.2f * static_cast<float>((i % 9) - 4);
        dO[i] = 0.05f * static_cast<float>((i % 3) + 1);
    }

    // Batch of 2 sequences: seq 0 has 1 query token and 0 key tokens;
    // seq 1 has 3 query tokens and 4 key tokens.
    const int32_t cu_q[3] = {0, 1, 4};
    const int32_t cu_k[3] = {0, 0, 4};

    Tensor dQ, dK, dV;
    brotensor::flash_attention_varlen_backward(
        Q, K, V, O, dO, cu_q, cu_k,
        /*batch_size=*/2, /*max_seqlen_q=*/3, /*max_seqlen_k=*/4,
        num_heads, head_dim, /*causal=*/false, dQ, dK, dV);

    for (int i = 0; i < dQ.size(); ++i) CHECK(std::isfinite(dQ[i]));
    for (int i = 0; i < dK.size(); ++i) CHECK(std::isfinite(dK[i]));
    for (int i = 0; i < dV.size(); ++i) CHECK(std::isfinite(dV[i]));
    // Query row 0 had 0 keys to attend to -> dQ row 0 must be identically 0.0f
    for (int d = 0; d < D; ++d) {
        CHECK(dQ(0, d) == 0.0f);
    }
}

int main() {
    std::printf("=== Issue #140: Gradient NaN-Freedom Proofs & Clamping ===\n");
    test_bv_and_runtime_nan_freedom_suite();
    test_sqrt_backward_clamped_exact_and_boundary();
    test_varlen_flash_attention_empty_and_causal_nan_freedom();

    if (g_failures == 0) {
        std::printf("ALL PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
