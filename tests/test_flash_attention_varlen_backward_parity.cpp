// CPU<->GPU parity for flash_attention_varlen_backward.
//
// Mirrors test_flash_attention_varlen_parity.cpp's shape matrix on the
// backward, using the same FP16-quantised input convention so CPU (FP32)
// and GPU (FP16) start from identical values; compare with loose FP16-scale
// tolerance (atol 1e-2 / rtol 1e-2 — same envelope as the existing flash
// attention parity tests).

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

void run_varlen_backward(const std::vector<int>& seq_lens_q,
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
    Tensor Q  = make_q16_cpu(total_q, D, rng, 0.3f);
    Tensor K  = make_q16_cpu(total_k, D, rng, 0.3f);
    Tensor V  = make_q16_cpu(total_k, D, rng, 0.3f);
    Tensor dO = make_q16_cpu(total_q, D, rng, 0.3f);
    // O is unused by the recompute backward — pass a same-shape tensor that
    // satisfies the signature checks. Reuse Q's storage shape; values don't matter.
    Tensor O  = make_q16_cpu(total_q, D, rng, 0.3f);

    // CPU run: cu_seqlens are host pointers on CPU.
    Tensor dQ_c, dK_c, dV_c;
    brotensor::flash_attention_varlen_backward(
        Q, K, V, O, dO, cq.data(), ck.data(),
        B, max_q, max_k, num_heads, head_dim, causal,
        dQ_c, dK_c, dV_c);

    // GPU run: upload Q/K/V/O/dO and cu_seqlens.
    Tensor gQ  = to_fp16_gpu(Q);
    Tensor gK  = to_fp16_gpu(K);
    Tensor gV  = to_fp16_gpu(V);
    Tensor gO  = to_fp16_gpu(O);
    Tensor gdO = to_fp16_gpu(dO);
    Tensor gCQ = upload_indices(std::vector<int32_t>(cq.begin(), cq.end()));
    Tensor gCK = upload_indices(std::vector<int32_t>(ck.begin(), ck.end()));
    const int32_t* d_cq = static_cast<const int32_t*>(gCQ.data);
    const int32_t* d_ck = static_cast<const int32_t*>(gCK.data);

    Tensor gdQ, gdK, gdV;
    brotensor::flash_attention_varlen_backward(
        gQ, gK, gV, gO, gdO, d_cq, d_ck,
        B, max_q, max_k, num_heads, head_dim, causal,
        gdQ, gdK, gdV);

    compare_tensors(dQ_c, fp16_gpu_to_cpu(gdQ), "varlen_bwd.dQ", 1e-2f, 1e-2f);
    compare_tensors(dK_c, fp16_gpu_to_cpu(gdK), "varlen_bwd.dK", 1e-2f, 1e-2f);
    compare_tensors(dV_c, fp16_gpu_to_cpu(gdV), "varlen_bwd.dV", 1e-2f, 1e-2f);
}

BT_PARITY_TEST(varlen_bwd_one_seq_noncausal) {
    run_varlen_backward({7}, {7}, 2, 4, false, 0xB1);
}
BT_PARITY_TEST(varlen_bwd_one_seq_causal) {
    run_varlen_backward({7}, {7}, 2, 4, true, 0xB2);
}
BT_PARITY_TEST(varlen_bwd_two_seq_equal_noncausal) {
    run_varlen_backward({5, 5}, {5, 5}, 2, 4, false, 0xB3);
}
BT_PARITY_TEST(varlen_bwd_two_seq_equal_causal) {
    run_varlen_backward({5, 5}, {5, 5}, 2, 4, true, 0xB4);
}
BT_PARITY_TEST(varlen_bwd_three_seq_varying_noncausal) {
    run_varlen_backward({3, 9, 4}, {3, 11, 4}, 2, 4, false, 0xB5);
}
BT_PARITY_TEST(varlen_bwd_three_seq_varying_causal) {
    run_varlen_backward({3, 9, 4}, {3, 9, 4}, 2, 4, true, 0xB6);
}
BT_PARITY_TEST(varlen_bwd_larger_head_dim) {
    run_varlen_backward({16, 16}, {16, 16}, 4, 16, false, 0xB7);
}

} // namespace

int main() {
    return run_all("flash_attention_varlen_backward CPU<->GPU parity");
}
