// CPU<->GPU parity for flash_attention_varlen_forward (Qwen3-VL window
// attention).
//
// DTYPE NOTES
//   GPU runs FP16 internally, CPU is FP32. We quantise all inputs through
//   FP16 so both backends start from identical values; compare with a loose
//   FP16-scale tolerance (atol 1e-2, rtol 1e-2 — same envelope as
//   test_flash_attention_parity.cpp's flash_fwd parity).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

inline float q16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

Tensor make_q16_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = q16(rng.next_unit() * scale);
    return t;
}

Tensor to_fp16_gpu(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(n);
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_fp16_bits(cpu[i]);
    return Tensor::from_host_fp16_on(gpu_device(), h.data(), cpu.rows, cpu.cols);
}

Tensor fp16_gpu_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_fp16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return out;
}

void run_varlen(const std::vector<int>& seq_lens_q,
                const std::vector<int>& seq_lens_k,
                int num_heads, int head_dim, bool causal,
                uint64_t seed) {
    const int B = static_cast<int>(seq_lens_q.size());
    std::vector<int32_t> cq(B + 1, 0), ck(B + 1, 0);
    int max_q = 0, max_k = 0;
    for (int b = 0; b < B; ++b) {
        cq[b + 1] = cq[b] + seq_lens_q[b];
        ck[b + 1] = ck[b] + seq_lens_k[b];
        if (seq_lens_q[b] > max_q) max_q = seq_lens_q[b];
        if (seq_lens_k[b] > max_k) max_k = seq_lens_k[b];
    }
    const int total_q = cq.back();
    const int total_k = ck.back();
    const int D = num_heads * head_dim;

    SplitMix64 rng(seed);
    Tensor Q = make_q16_cpu(total_q, D, rng, 0.3f);
    Tensor K = make_q16_cpu(total_k, D, rng, 0.3f);
    Tensor V = make_q16_cpu(total_k, D, rng, 0.3f);

    // CPU run: cu_seqlens are host pointers on CPU.
    Tensor O_c;
    brotensor::flash_attention_varlen_forward(
        Q, K, V, cq.data(), ck.data(),
        B, max_q, max_k, num_heads, head_dim, causal, O_c);

    // GPU run: upload Q/K/V and cu_seqlens.
    Tensor gQ = to_fp16_gpu(Q), gK = to_fp16_gpu(K), gV = to_fp16_gpu(V);
    std::vector<int32_t> cq_vec(cq.begin(), cq.end());
    std::vector<int32_t> ck_vec(ck.begin(), ck.end());
    Tensor gCQ = upload_indices(cq_vec);
    Tensor gCK = upload_indices(ck_vec);
    const int32_t* d_cq = static_cast<const int32_t*>(gCQ.data);
    const int32_t* d_ck = static_cast<const int32_t*>(gCK.data);

    Tensor gO;
    brotensor::flash_attention_varlen_forward(
        gQ, gK, gV, d_cq, d_ck,
        B, max_q, max_k, num_heads, head_dim, causal, gO);

    compare_tensors(O_c, fp16_gpu_to_cpu(gO), "varlen.O", 1e-2f, 1e-2f);
}

BT_PARITY_TEST(varlen_one_seq_noncausal) {
    run_varlen({7}, {7}, 2, 4, false, 0xA1);
}
BT_PARITY_TEST(varlen_one_seq_causal) {
    run_varlen({7}, {7}, 2, 4, true, 0xA2);
}
BT_PARITY_TEST(varlen_two_seq_equal_noncausal) {
    run_varlen({5, 5}, {5, 5}, 2, 4, false, 0xA3);
}
BT_PARITY_TEST(varlen_two_seq_equal_causal) {
    run_varlen({5, 5}, {5, 5}, 2, 4, true, 0xA4);
}
BT_PARITY_TEST(varlen_three_seq_varying_noncausal) {
    run_varlen({3, 9, 4}, {3, 11, 4}, 2, 4, false, 0xA5);
}
BT_PARITY_TEST(varlen_three_seq_varying_causal) {
    run_varlen({3, 9, 4}, {3, 9, 4}, 2, 4, true, 0xA6);
}
BT_PARITY_TEST(varlen_larger_head_dim) {
    run_varlen({16, 16}, {16, 16}, 4, 16, false, 0xA7);
}

// ─── FP32 path (CPU and GPU both FP32 — tight tolerance) ──────────────────

void run_varlen_fp32(const std::vector<int>& seq_lens_q,
                     const std::vector<int>& seq_lens_k,
                     int num_heads, int head_dim, bool causal,
                     uint64_t seed) {
    const int B = static_cast<int>(seq_lens_q.size());
    std::vector<int32_t> cq(B + 1, 0), ck(B + 1, 0);
    int max_q = 0, max_k = 0;
    for (int b = 0; b < B; ++b) {
        cq[b + 1] = cq[b] + seq_lens_q[b];
        ck[b + 1] = ck[b] + seq_lens_k[b];
        if (seq_lens_q[b] > max_q) max_q = seq_lens_q[b];
        if (seq_lens_k[b] > max_k) max_k = seq_lens_k[b];
    }
    const int total_q = cq.back();
    const int total_k = ck.back();
    const int D = num_heads * head_dim;

    SplitMix64 rng(seed);
    Tensor Q = Tensor::mat(total_q, D);
    Tensor K = Tensor::mat(total_k, D);
    Tensor V = Tensor::mat(total_k, D);
    for (int i = 0; i < Q.size(); ++i) Q.ptr()[i] = rng.next_unit() * 0.3f;
    for (int i = 0; i < K.size(); ++i) K.ptr()[i] = rng.next_unit() * 0.3f;
    for (int i = 0; i < V.size(); ++i) V.ptr()[i] = rng.next_unit() * 0.3f;

    Tensor O_c;
    brotensor::flash_attention_varlen_forward(
        Q, K, V, cq.data(), ck.data(),
        B, max_q, max_k, num_heads, head_dim, causal, O_c);

    Tensor gQ = Q.to(gpu_device());
    Tensor gK = K.to(gpu_device());
    Tensor gV = V.to(gpu_device());
    Tensor gCQ = upload_indices(std::vector<int32_t>(cq.begin(), cq.end()));
    Tensor gCK = upload_indices(std::vector<int32_t>(ck.begin(), ck.end()));
    const int32_t* d_cq = static_cast<const int32_t*>(gCQ.data);
    const int32_t* d_ck = static_cast<const int32_t*>(gCK.data);

    Tensor gO;
    brotensor::flash_attention_varlen_forward(
        gQ, gK, gV, d_cq, d_ck,
        B, max_q, max_k, num_heads, head_dim, causal, gO);

    brotensor::sync_all();
    compare_tensors(O_c, gO.to(brotensor::Device::CPU), "varlen_fp32.O",
                    5e-5f, 5e-5f);
}

BT_PARITY_TEST(varlen_fp32_one_seq_noncausal) {
    run_varlen_fp32({7}, {7}, 2, 4, false, 0xA8);
}
BT_PARITY_TEST(varlen_fp32_three_seq_varying_causal) {
    run_varlen_fp32({3, 9, 4}, {3, 9, 4}, 2, 4, true, 0xA9);
}
BT_PARITY_TEST(varlen_fp32_larger_head_dim) {
    run_varlen_fp32({16, 16}, {16, 16}, 4, 16, false, 0xAA);
}

} // namespace

int main() {
    return run_all("flash_attention_varlen_forward CPU<->GPU parity");
}
