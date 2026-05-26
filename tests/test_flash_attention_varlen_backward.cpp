// CPU smoke test for flash_attention_varlen_backward (FP32).
//
// Validates packed variable-length attention backward against an explicit
// reference that runs the attention backward independently on each
// sequence's row slice (mirrors test_flash_attention_varlen.cpp's forward
// reference style). Covers the same shape matrix:
//   * 1 sequence — equivalent to flash_attention_backward.
//   * 2 sequences of equal length.
//   * 3 sequences of varying length (non-causal allows Lq != Lk).
//   * causal and non-causal.
//
// CPU-only — brotensor CPU is FP32-only. CPU↔GPU parity ships in
// test_flash_attention_varlen_backward_parity.cpp alongside the CUDA impl.

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

// Reference per-sequence attention backward. Mirrors the CPU impl's
// attention_core_backward math: recompute P, then dV = P^T·dO,
// dP = dO·V^T, dS = P*(dP - D_q)*inv_sqrt, dQ = dS·K, dK = dS^T·Q.
void varlen_backward_ref(const std::vector<float>& Q,
                         const std::vector<float>& K,
                         const std::vector<float>& V,
                         const std::vector<float>& dO,
                         const std::vector<int32_t>& cq,
                         const std::vector<int32_t>& ck,
                         int B, int num_heads, int head_dim, bool causal,
                         std::vector<float>& dQ,
                         std::vector<float>& dK,
                         std::vector<float>& dV) {
    const int D = num_heads * head_dim;
    const int total_q = cq.back();
    const int total_k = ck.back();
    dQ.assign(static_cast<size_t>(total_q) * D, 0.0f);
    dK.assign(static_cast<size_t>(total_k) * D, 0.0f);
    dV.assign(static_cast<size_t>(total_k) * D, 0.0f);
    const int hd = head_dim;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int b = 0; b < B; ++b) {
        const int qb = cq[b], qe = cq[b + 1];
        const int kb = ck[b], ke = ck[b + 1];
        const int Lq = qe - qb;
        const int Lk = ke - kb;
        if (Lq <= 0 || Lk <= 0) continue;

        for (int h = 0; h < num_heads; ++h) {
            const int off = h * hd;
            std::vector<float> P(static_cast<size_t>(Lq) * Lk, 0.0f);
            // Recompute P[q,k] for this sequence/head.
            for (int q = 0; q < Lq; ++q) {
                float maxv = -1e30f;
                float* prow = P.data() + static_cast<size_t>(q) * Lk;
                for (int k = 0; k < Lk; ++k) {
                    if (causal && k > q) { prow[k] = -1e30f; continue; }
                    double dot = 0.0;
                    for (int d = 0; d < hd; ++d)
                        dot += static_cast<double>(Q[(qb + q)*D + off + d]) *
                               K[(kb + k)*D + off + d];
                    float s = static_cast<float>(dot) * inv_sqrt;
                    prow[k] = s;
                    if (s > maxv) maxv = s;
                }
                const bool empty = (maxv <= -1e29f);
                float sum = 0.0f;
                for (int k = 0; k < Lk; ++k) {
                    const float e = empty ? 0.0f : std::exp(prow[k] - maxv);
                    prow[k] = e;
                    sum += e;
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int k = 0; k < Lk; ++k) prow[k] *= inv;
            }
            // dV[k, off+d] = sum_q P[q,k] * dO[q, off+d].
            for (int k = 0; k < Lk; ++k) {
                for (int d = 0; d < hd; ++d) {
                    double acc = 0.0;
                    for (int q = 0; q < Lq; ++q)
                        acc += static_cast<double>(P[(size_t)q*Lk + k]) *
                               dO[(qb + q)*D + off + d];
                    dV[(kb + k)*D + off + d] = static_cast<float>(acc);
                }
            }
            // dP[q,k] = sum_d dO[q, off+d] * V[k, off+d].
            std::vector<float> dP(static_cast<size_t>(Lq) * Lk, 0.0f);
            for (int q = 0; q < Lq; ++q) {
                for (int k = 0; k < Lk; ++k) {
                    double acc = 0.0;
                    for (int d = 0; d < hd; ++d)
                        acc += static_cast<double>(dO[(qb + q)*D + off + d]) *
                               V[(kb + k)*D + off + d];
                    dP[(size_t)q*Lk + k] = static_cast<float>(acc);
                }
            }
            // dS[q,k] = P[q,k] * (dP[q,k] - D_q) * inv_sqrt, where
            // D_q = sum_k P[q,k] * dP[q,k].
            std::vector<float> dS(static_cast<size_t>(Lq) * Lk, 0.0f);
            for (int q = 0; q < Lq; ++q) {
                const float* prow = P.data()  + static_cast<size_t>(q) * Lk;
                const float* dpr  = dP.data() + static_cast<size_t>(q) * Lk;
                float* dsr        = dS.data() + static_cast<size_t>(q) * Lk;
                double Dq = 0.0;
                for (int k = 0; k < Lk; ++k) Dq += static_cast<double>(prow[k]) * dpr[k];
                for (int k = 0; k < Lk; ++k)
                    dsr[k] = prow[k] * (dpr[k] - static_cast<float>(Dq)) * inv_sqrt;
            }
            // dQ[q, off+d] = sum_k dS[q,k] * K[k, off+d].
            for (int q = 0; q < Lq; ++q) {
                for (int d = 0; d < hd; ++d) {
                    double acc = 0.0;
                    for (int k = 0; k < Lk; ++k)
                        acc += static_cast<double>(dS[(size_t)q*Lk + k]) *
                               K[(kb + k)*D + off + d];
                    dQ[(qb + q)*D + off + d] = static_cast<float>(acc);
                }
            }
            // dK[k, off+d] = sum_q dS[q,k] * Q[q, off+d].
            for (int k = 0; k < Lk; ++k) {
                for (int d = 0; d < hd; ++d) {
                    double acc = 0.0;
                    for (int q = 0; q < Lq; ++q)
                        acc += static_cast<double>(dS[(size_t)q*Lk + k]) *
                               Q[(qb + q)*D + off + d];
                    dK[(kb + k)*D + off + d] = static_cast<float>(acc);
                }
            }
        }
    }
}

void check_close(const std::vector<float>& got,
                 const std::vector<float>& ref,
                 const char* label,
                 float atol = 5e-5f, float rtol = 5e-5f) {
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

    std::mt19937 rng(0xB1A5);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> Q(static_cast<size_t>(total_q) * D);
    std::vector<float> K(static_cast<size_t>(total_k) * D);
    std::vector<float> V(static_cast<size_t>(total_k) * D);
    std::vector<float> dO(static_cast<size_t>(total_q) * D);
    for (auto& v : Q)  v = dist(rng);
    for (auto& v : K)  v = dist(rng);
    for (auto& v : V)  v = dist(rng);
    for (auto& v : dO) v = dist(rng);

    std::vector<float> dQ_ref, dK_ref, dV_ref;
    varlen_backward_ref(Q, K, V, dO, cq, ck, B, num_heads, head_dim, causal,
                        dQ_ref, dK_ref, dV_ref);

    Tensor Qt  = Tensor::from_host_on(Device::CPU, Q.data(),  total_q, D);
    Tensor Kt  = Tensor::from_host_on(Device::CPU, K.data(),  total_k, D);
    Tensor Vt  = Tensor::from_host_on(Device::CPU, V.data(),  total_k, D);
    Tensor dOt = Tensor::from_host_on(Device::CPU, dO.data(), total_q, D);
    // O is unused by the recompute backward — pass an empty tensor sized to
    // match Q so the op signature is satisfied and the input checks pass.
    Tensor Ot  = Tensor::from_host_on(Device::CPU, Q.data(),  total_q, D);
    Tensor dQt, dKt, dVt;

    brotensor::flash_attention_varlen_backward(
        Qt, Kt, Vt, Ot, dOt, cq.data(), ck.data(),
        B, max_q, max_k, num_heads, head_dim, causal, dQt, dKt, dVt);

    CHECK(dQt.rows == total_q && dQt.cols == D && dQt.dtype == Dtype::FP32);
    CHECK(dKt.rows == total_k && dKt.cols == D && dKt.dtype == Dtype::FP32);
    CHECK(dVt.rows == total_k && dVt.cols == D && dVt.dtype == Dtype::FP32);

    std::vector<float> dQ_got(static_cast<size_t>(total_q) * D);
    std::vector<float> dK_got(static_cast<size_t>(total_k) * D);
    std::vector<float> dV_got(static_cast<size_t>(total_k) * D);
    for (size_t i = 0; i < dQ_got.size(); ++i) dQ_got[i] = dQt.host_f32()[i];
    for (size_t i = 0; i < dK_got.size(); ++i) dK_got[i] = dKt.host_f32()[i];
    for (size_t i = 0; i < dV_got.size(); ++i) dV_got[i] = dVt.host_f32()[i];

    check_close(dQ_got, dQ_ref, "dQ");
    check_close(dK_got, dK_ref, "dK");
    check_close(dV_got, dV_ref, "dV");
}

} // namespace

int main() {
    std::printf("flash_attention_varlen_backward CPU smoke\n");

    // 1 sequence
    run_one("1seq-noncausal", {7}, {7}, 2, 4, false);
    run_one("1seq-causal",    {7}, {7}, 2, 4, true);

    // 2 sequences, equal length
    run_one("2seq-equal-noncausal", {5, 5}, {5, 5}, 2, 4, false);
    run_one("2seq-equal-causal",    {5, 5}, {5, 5}, 2, 4, true);

    // 3 sequences, varying length (non-causal allows Lq != Lk)
    run_one("3seq-varying-noncausal", {3, 6, 4}, {3, 8, 4}, 2, 4, false);
    run_one("3seq-varying-causal",    {3, 6, 4}, {3, 6, 4}, 2, 4, true);

    // Single sequence, larger head_dim
    run_one("1seq-larger", {12}, {12}, 4, 8, false);

    if (g_failures == 0) {
        std::printf("  PASS\n");
        return 0;
    }
    std::printf("  FAIL  %d failures\n", g_failures);
    return 1;
}
