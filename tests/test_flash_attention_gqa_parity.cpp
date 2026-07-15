// Correctness + CPU<->GPU parity for flash_attention_gqa_forward.
//
// flash_attention_gqa_forward generalises flash_attention_forward to grouped-
// query attention (Q num_q_heads-wide, K/V num_kv_heads-wide) with a causal flag
// — causal == false is the full bidirectional prefill an LLM2Vec-style encoder
// needs. Three checks:
//   1. CPU vs an INDEPENDENT naive FP32 GQA reference (not just CPU==GPU, which
//      could be wrong the same way) — causal + bidirectional, MHA + GQA ratios,
//      with/without a key mask.
//   2. The MHA special case (num_kv == num_q) must reproduce the existing
//      flash_attention_forward, both causal and non-causal.
//   3. GPU vs the naive reference — FP32 (tight) and FP16 (loose), so the shared
//      windowed kernel's new bidirectional bound is exercised on the GPU too.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <cmath>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

// Independent naive GQA attention (FP32), causal or bidirectional, key-masked.
//   Q: (Lq, num_q*hd);  K, V: (Lk, num_kv*hd);  group = num_q / num_kv.
// Query head h reads K/V head h/group (the GQA mapping). causal drops keys k > q.
Tensor naive_gqa(const Tensor& Q, const Tensor& K, const Tensor& V,
                 const std::vector<float>* mask,
                 int num_q, int num_kv, bool causal) {
    const int Lq = Q.rows, Lk = K.rows;
    const int Dq = Q.cols, Dkv = K.cols;
    const int hd = Dq / num_q;
    const int group = num_q / num_kv;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    Tensor O = Tensor::mat(Lq, Dq);
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < num_q; ++h) {
            const int off    = h * hd;
            const int off_kv = (h / group) * hd;
            std::vector<float> sc(static_cast<size_t>(Lk), -1e30f);
            float m = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                if (mask && (*mask)[k] <= 0.5f) continue;
                if (causal && k > q) continue;
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d)
                    dot += Q[q * Dq + off + d] * K[k * Dkv + off_kv + d];
                sc[k] = dot * inv_sqrt;
                if (sc[k] > m) m = sc[k];
            }
            const bool empty = (m <= -1e29f);
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                const float e = empty ? 0.0f : std::exp(sc[k] - m);
                sc[k] = e;
                sum += e;
            }
            const float rinv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < Lk; ++k)
                    acc += sc[k] * rinv * V[k * Dkv + off_kv + d];
                O.ptr()[q * Dq + off + d] = acc;
            }
        }
    }
    return O;
}

void make_qkv(int Lq, int Lk, int num_q, int num_kv, int hd, SplitMix64& rng,
              Tensor& Q, Tensor& K, Tensor& V) {
    Q = Tensor::mat(Lq, num_q * hd);
    K = Tensor::mat(Lk, num_kv * hd);
    V = Tensor::mat(Lk, num_kv * hd);
    fill_random(Q, rng, 0.5f);
    fill_random(K, rng, 0.5f);
    fill_random(V, rng, 0.5f);
}

// 1. CPU FP32 vs the naive reference.
void run_cpu_ref(int Lq, int Lk, int num_q, int num_kv, int hd, bool causal,
                 const std::vector<float>* mask, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q, K, V;
    make_qkv(Lq, Lk, num_q, num_kv, hd, rng, Q, K, V);
    const float* hm = mask ? mask->data() : nullptr;

    Tensor O;
    brotensor::flash_attention_gqa_forward(Q, K, V, hm, num_q, num_kv, causal, O);
    Tensor ref = naive_gqa(Q, K, V, mask, num_q, num_kv, causal);
    compare_tensors(ref, O, "gqa.cpu_ref", 1e-4f, 1e-3f);
}

// 2. MHA special case reproduces flash_attention_forward (CPU FP32).
void run_mha_equiv(int L, int num_heads, int hd, bool causal, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q, K, V;
    make_qkv(L, L, num_heads, num_heads, hd, rng, Q, K, V);

    Tensor O_gqa;
    brotensor::flash_attention_gqa_forward(Q, K, V, nullptr, num_heads,
                                           num_heads, causal, O_gqa);
    Tensor O_fa;
    brotensor::flash_attention_forward(Q, K, V, nullptr, num_heads, causal, O_fa);
    compare_tensors(O_fa, O_gqa, "gqa.mha_equiv", 1e-5f, 1e-4f);
}

// 3a. GPU FP32 vs the naive reference (tight — same math, FP32 both sides).
void run_gpu_fp32(int Lq, int Lk, int num_q, int num_kv, int hd, bool causal,
                  const std::vector<float>* mask, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q, K, V;
    make_qkv(Lq, Lk, num_q, num_kv, hd, rng, Q, K, V);
    Tensor ref = naive_gqa(Q, K, V, mask, num_q, num_kv, causal);

    Tensor gQ = Q.to(gpu_device()), gK = K.to(gpu_device()),
           gV = V.to(gpu_device());
    Tensor dmb = upload_mask(mask);
    const float* dm = mask ? static_cast<const float*>(dmb.data) : nullptr;

    Tensor gO;
    brotensor::flash_attention_gqa_forward(gQ, gK, gV, dm, num_q, num_kv,
                                           causal, gO);
    compare_tensors(ref, download_to_host(gO), "gqa.gpu_fp32", 1e-3f, 1e-3f);
}

// 3b. GPU FP16 vs the naive FP32 reference (loose — FP16 storage).
void run_gpu_fp16(int Lq, int Lk, int num_q, int num_kv, int hd, bool causal,
                  const std::vector<float>* mask, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor Q, K, V;
    make_qkv(Lq, Lk, num_q, num_kv, hd, rng, Q, K, V);
    Tensor ref = naive_gqa(Q, K, V, mask, num_q, num_kv, causal);

    Tensor gQ = to_fp16_gpu(Q), gK = to_fp16_gpu(K), gV = to_fp16_gpu(V);
    Tensor dmb = upload_mask(mask);
    const float* dm = mask ? static_cast<const float*>(dmb.data) : nullptr;

    Tensor gO;
    brotensor::flash_attention_gqa_forward(gQ, gK, gV, dm, num_q, num_kv,
                                           causal, gO);
    compare_tensors(ref, fp16_host_to_f32(download_to_host(gO)),
                    "gqa.gpu_fp16", 2e-2f, 2e-2f);
}

std::vector<float> partial_mask(int n) {
    std::vector<float> m(n, 1.0f);
    for (int i = n / 2; i < n; ++i) m[i] = 0.0f;
    if (n >= 2) m[1] = 0.0f;
    return m;
}

}  // namespace

// ─── 1. CPU vs naive reference ─────────────────────────────────────────────
// Bidirectional (the LLM2Vec case): MHA and GQA ratios, with/without mask.
BT_PARITY_TEST(gqa_cpu_bidir_mha_L8_h4_hd16) {
    run_cpu_ref(8, 8, 4, 4, 16, false, nullptr, 0x700ull);
}
BT_PARITY_TEST(gqa_cpu_bidir_gqa_L8_q8_kv2_hd16) {
    run_cpu_ref(8, 8, 8, 2, 16, false, nullptr, 0x701ull);
}
BT_PARITY_TEST(gqa_cpu_bidir_gqa_L12_q8_kv4_hd32) {
    run_cpu_ref(12, 12, 8, 4, 32, false, nullptr, 0x702ull);
}
BT_PARITY_TEST(gqa_cpu_bidir_gqa_L8_q8_kv2_hd16_mask) {
    auto m = partial_mask(8);
    run_cpu_ref(8, 8, 8, 2, 16, false, &m, 0x703ull);
}
// Causal GQA (requires Lq == Lk).
BT_PARITY_TEST(gqa_cpu_causal_gqa_L8_q8_kv2_hd16) {
    run_cpu_ref(8, 8, 8, 2, 16, true, nullptr, 0x704ull);
}
BT_PARITY_TEST(gqa_cpu_causal_gqa_L12_q8_kv4_hd32_mask) {
    auto m = partial_mask(12);
    run_cpu_ref(12, 12, 8, 4, 32, true, &m, 0x705ull);
}

// ─── 2. MHA special case == flash_attention_forward ────────────────────────
BT_PARITY_TEST(gqa_mha_equiv_bidir_L8_h4_hd16)  { run_mha_equiv(8, 4, 16, false, 0x710ull); }
BT_PARITY_TEST(gqa_mha_equiv_causal_L8_h4_hd16) { run_mha_equiv(8, 4, 16, true,  0x711ull); }
BT_PARITY_TEST(gqa_mha_equiv_bidir_L16_h8_hd32) { run_mha_equiv(16, 8, 32, false, 0x712ull); }

// ─── 3a. GPU FP32 vs naive reference ───────────────────────────────────────
BT_PARITY_TEST(gqa_gpu_fp32_bidir_gqa_L8_q8_kv2_hd16) {
    run_gpu_fp32(8, 8, 8, 2, 16, false, nullptr, 0x720ull);
}
BT_PARITY_TEST(gqa_gpu_fp32_bidir_gqa_L12_q8_kv4_hd32_mask) {
    auto m = partial_mask(12);
    run_gpu_fp32(12, 12, 8, 4, 32, false, &m, 0x721ull);
}
BT_PARITY_TEST(gqa_gpu_fp32_causal_gqa_L8_q8_kv2_hd16) {
    run_gpu_fp32(8, 8, 8, 2, 16, true, nullptr, 0x722ull);
}

// ─── 3b. GPU FP16 vs naive reference (the production dtype) ─────────────────
BT_PARITY_TEST(gqa_gpu_fp16_bidir_gqa_L12_q8_kv4_hd32) {
    run_gpu_fp16(12, 12, 8, 4, 32, false, nullptr, 0x730ull);
}
BT_PARITY_TEST(gqa_gpu_fp16_bidir_gqa_L16_q8_kv2_hd64) {
    run_gpu_fp16(16, 16, 8, 2, 64, false, nullptr, 0x731ull);
}
BT_PARITY_TEST(gqa_gpu_fp16_bidir_gqa_L12_q8_kv4_hd32_mask) {
    auto m = partial_mask(12);
    run_gpu_fp16(12, 12, 8, 4, 32, false, &m, 0x732ull);
}

int main() { return run_all("flash_attention_gqa correctness + parity"); }
