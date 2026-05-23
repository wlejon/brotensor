// CPU smoke test for flash_attention_varlen_forward (FP32).
//
// Validates packed variable-length attention against an explicit reference
// that runs attention independently on each sequence's row slice. Covers:
//   * 1 sequence — equivalent to flash_attention_forward.
//   * 2 sequences of equal length.
//   * 3 sequences of varying length.
//   * non-causal and causal (causal requires per-sequence Lq == Lk).
//
// CPU-only — brotensor CPU is FP32-only. The CPU/CUDA parity test
// (test_flash_attention_varlen_parity.cpp) covers FP16 cross-backend
// agreement.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;
using brotensor::Device;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

// Reference: independent per-sequence attention. Mirrors the CPU impl's
// attention_core math.
void varlen_ref(const std::vector<float>& Q,
                const std::vector<float>& K,
                const std::vector<float>& V,
                const std::vector<int32_t>& cq,
                const std::vector<int32_t>& ck,
                int B, int num_heads, int head_dim, bool causal,
                std::vector<float>& O) {
    const int D = num_heads * head_dim;
    const int total_q = cq.back();
    O.assign(static_cast<size_t>(total_q) * D, 0.0f);
    const int hd = head_dim;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int b = 0; b < B; ++b) {
        const int qb = cq[b], qe = cq[b + 1];
        const int kb = ck[b], ke = ck[b + 1];
        const int Lq = qe - qb;
        const int Lk = ke - kb;
        if (Lq == 0) continue;
        if (Lk == 0) continue; // O rows already zero
        for (int q = 0; q < Lq; ++q) {
            for (int h = 0; h < num_heads; ++h) {
                const int off = h * hd;
                std::vector<float> scores(Lk);
                float maxv = -1e30f;
                for (int k = 0; k < Lk; ++k) {
                    if (causal && k > q) { scores[k] = -1e30f; continue; }
                    double dot = 0.0;
                    for (int d = 0; d < hd; ++d)
                        dot += static_cast<double>(Q[(qb + q)*D + off + d]) *
                               K[(kb + k)*D + off + d];
                    float s = static_cast<float>(dot) * inv_sqrt;
                    scores[k] = s;
                    if (s > maxv) maxv = s;
                }
                const bool empty = (maxv <= -1e29f);
                float sum = 0.0f;
                for (int k = 0; k < Lk; ++k) {
                    const float e = empty ? 0.0f : std::exp(scores[k] - maxv);
                    scores[k] = e;
                    sum += e;
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int d = 0; d < hd; ++d) {
                    double acc = 0.0;
                    for (int k = 0; k < Lk; ++k)
                        acc += static_cast<double>(scores[k]) * inv *
                               V[(kb + k)*D + off + d];
                    O[(qb + q)*D + off + d] = static_cast<float>(acc);
                }
            }
        }
    }
}

void check_close(const std::vector<float>& got,
                 const std::vector<float>& ref,
                 const char* label,
                 float atol = 1e-5f, float rtol = 1e-5f) {
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * std::fabs(ref[i])) {
            if (bad < 3)
                std::printf("    %s mismatch i=%zu got=%g ref=%g err=%g\n",
                            label, i, got[i], ref[i], e);
            ++bad;
        }
    }
    std::printf("    %s max_err=%g bad=%d / %zu\n", label, max_err, bad, ref.size());
    CHECK(bad == 0);
}

void run_one(const char* label,
             const std::vector<int>& seq_lens_q,
             const std::vector<int>& seq_lens_k,
             int num_heads, int head_dim, bool causal) {
    const int B = static_cast<int>(seq_lens_q.size());
    std::vector<int32_t> cq(B + 1, 0), ck(B + 1, 0);
    for (int b = 0; b < B; ++b) {
        cq[b + 1] = cq[b] + seq_lens_q[b];
        ck[b + 1] = ck[b] + seq_lens_k[b];
    }
    const int total_q = cq.back();
    const int total_k = ck.back();
    int max_q = 0, max_k = 0;
    for (int b = 0; b < B; ++b) {
        if (seq_lens_q[b] > max_q) max_q = seq_lens_q[b];
        if (seq_lens_k[b] > max_k) max_k = seq_lens_k[b];
    }
    const int D = num_heads * head_dim;
    std::printf("  %s  B=%d total_q=%d total_k=%d nh=%d hd=%d causal=%d\n",
                label, B, total_q, total_k, num_heads, head_dim, (int)causal);

    std::mt19937 rng(0xA11C);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(static_cast<size_t>(total_q) * D);
    std::vector<float> K(static_cast<size_t>(total_k) * D);
    std::vector<float> V(static_cast<size_t>(total_k) * D);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> O_ref;
    varlen_ref(Q, K, V, cq, ck, B, num_heads, head_dim, causal, O_ref);

    Tensor Qt = Tensor::from_host_on(Device::CPU, Q.data(), total_q, D);
    Tensor Kt = Tensor::from_host_on(Device::CPU, K.data(), total_k, D);
    Tensor Vt = Tensor::from_host_on(Device::CPU, V.data(), total_k, D);
    Tensor Ot;

    brotensor::flash_attention_varlen_forward(
        Qt, Kt, Vt, cq.data(), ck.data(),
        B, max_q, max_k, num_heads, head_dim, causal, Ot);

    CHECK(Ot.rows == total_q && Ot.cols == D && Ot.dtype == Dtype::FP32);
    std::vector<float> got(static_cast<size_t>(total_q) * D);
    for (size_t i = 0; i < got.size(); ++i)
        got[i] = Ot.host_f32()[i];

    check_close(got, O_ref, label);
}

} // namespace

int main() {
    std::printf("flash_attention_varlen_forward CPU smoke\n");

    // 1 sequence
    run_one("1seq-noncausal", {7}, {7}, 2, 4, false);
    run_one("1seq-causal",    {7}, {7}, 2, 4, true);

    // 2 sequences, equal length
    run_one("2seq-equal-noncausal", {5, 5}, {5, 5}, 2, 4, false);
    run_one("2seq-equal-causal",    {5, 5}, {5, 5}, 2, 4, true);

    // 3 sequences, varying length (non-causal — varied Lq vs Lk allowed)
    run_one("3seq-varying-noncausal", {3, 6, 4}, {3, 8, 4}, 2, 4, false);
    run_one("3seq-varying-causal",    {3, 6, 4}, {3, 6, 4}, 2, 4, true);

    // Single sequence, head_dim 8 (a more realistic shape)
    run_one("1seq-larger", {12}, {12}, 4, 8, false);

    if (g_failures == 0) {
        std::printf("  PASS\n");
        return 0;
    }
    std::printf("  FAIL  %d failures\n", g_failures);
    return 1;
}
