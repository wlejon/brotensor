// CPU↔GPU parity tests for the KV-cache ops (CHUNK 4).
//
//   kv_cache_append       — OVERWRITES the [cur_len, cur_len+L_new) cache
//                           slice; rows outside the slice are untouched.
//   flash_attention_decode— O OVERWRITTEN. Causal attention of Lq query rows
//                           against the first valid_len cache rows.
//
// DTYPE NOTE: both GPU kv-cache ops run FP16 — their tensors MUST be FP16.
// The CPU backend is FP32-only. So the harness quantises inputs through FP16
// (so CPU and GPU see identical input values), feeds FP16 to the GPU and FP32
// to the CPU, and compares with a loose FP16-driven tolerance.
//   * kv_cache_append is a pure copy — once inputs are FP16-quantised the
//     result is bit-exact, so a small tol suffices.
//   * flash_attention_decode does an FP16 dot-product + softmax reduction;
//     the GPU also uses fast-math expf/rsqrtf. Tolerance is relaxed to
//     atol=5e-2, rtol=5e-2 to absorb the FP16 + reduction-order gap. This is
//     the same magnitude the GPU smoke test (test_kv_cache.cpp) uses when
//     comparing decode against the causal forward kernel.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <limits>
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

Tensor to_fp16_cuda(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_fp16_bits(cpu[i]);
    return Tensor::from_host_fp16_on(gpu_device(), h.data(),
                                     cpu.rows, cpu.cols);
}

Tensor fp16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_fp16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return out;
}

// ─── kv_cache_append ───────────────────────────────────────────────────────
// Appends two chunks at increasing offsets; verifies the whole cache (written
// slices + untouched zero tail) matches between CPU and GPU.
void run_append(int L_max, int D, int len0, int len1, uint64_t seed) {
    SplitMix64 rng(seed);

    Tensor K0 = make_q16_cpu(len0, D, rng, 1.0f);
    Tensor V0 = make_q16_cpu(len0, D, rng, 1.0f);
    Tensor K1 = make_q16_cpu(len1, D, rng, 1.0f);
    Tensor V1 = make_q16_cpu(len1, D, rng, 1.0f);

    // CPU: zero-initialised caches.
    Tensor cpu_Kc = Tensor::mat(L_max, D);
    Tensor cpu_Vc = Tensor::mat(L_max, D);
    cpu_Kc.zero();
    cpu_Vc.zero();
    brotensor::kv_cache_append(K0, V0, 0,    cpu_Kc, cpu_Vc);
    brotensor::kv_cache_append(K1, V1, len0, cpu_Kc, cpu_Vc);

    // GPU: FP16 caches zero-initialised.
    Tensor gpu_Kc = Tensor::zeros_on(gpu_device(), L_max, D, Dtype::FP16);
    Tensor gpu_Vc = Tensor::zeros_on(gpu_device(), L_max, D, Dtype::FP16);
    Tensor gK0 = to_fp16_cuda(K0), gV0 = to_fp16_cuda(V0);
    Tensor gK1 = to_fp16_cuda(K1), gV1 = to_fp16_cuda(V1);
    brotensor::kv_cache_append(gK0, gV0, 0,    gpu_Kc, gpu_Vc);
    brotensor::kv_cache_append(gK1, gV1, len0, gpu_Kc, gpu_Vc);

    // Pure copy of FP16-quantised inputs — bit-exact match expected.
    compare_tensors(cpu_Kc, fp16_cuda_to_cpu(gpu_Kc), "kv_append_K",
                    1e-3f, 1e-3f);
    compare_tensors(cpu_Vc, fp16_cuda_to_cpu(gpu_Vc), "kv_append_V",
                    1e-3f, 1e-3f);
}

// ─── flash_attention_decode ────────────────────────────────────────────────
// valid_len cache rows, Lq query rows (the tail). Q is (Lq, num_q_heads*head_dim);
// the K/V cache is (valid_len, num_kv_heads*head_dim). num_kv_heads == num_q_heads
// is MHA; num_kv_heads < num_q_heads is GQA (each KV head serves
// num_q_heads/num_kv_heads consecutive query heads).
void run_decode(int valid_len, int Lq, int num_q_heads, int num_kv_heads,
                int head_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Dq  = num_q_heads  * head_dim;
    const int Dkv = num_kv_heads * head_dim;
    // Small magnitude keeps FP16 softmax well-conditioned.
    Tensor Q  = make_q16_cpu(Lq, Dq, rng, 0.4f);
    Tensor Kc = make_q16_cpu(valid_len, Dkv, rng, 0.4f);
    Tensor Vc = make_q16_cpu(valid_len, Dkv, rng, 0.4f);

    Tensor cpu_O;
    brotensor::flash_attention_decode(Q, Kc, Vc, valid_len,
                                      num_q_heads, num_kv_heads, cpu_O);

    Tensor gQ  = to_fp16_cuda(Q);
    Tensor gKc = to_fp16_cuda(Kc);
    Tensor gVc = to_fp16_cuda(Vc);
    Tensor gpu_O;
    brotensor::flash_attention_decode(gQ, gKc, gVc, valid_len,
                                      num_q_heads, num_kv_heads, gpu_O);

    // FP16 dot-product + softmax reduction + fast-math expf/rsqrtf.
    compare_tensors(cpu_O, fp16_cuda_to_cpu(gpu_O), "flash_decode",
                    5e-2f, 5e-2f);
}

// ─── flash_attention_decode_masked ─────────────────────────────────────────
// Fixed-capacity masked decode vs the length-truncated op over the SAME cache
// buffers. Rows past valid_len are filled with NaN to prove masked keys are
// never read: any leak poisons the output. The masked GPU result must match
// the truncated GPU result bit-for-bit (identical tile math, masked keys
// reduce to exact zeros); CPU masked vs CPU truncated likewise.
void run_decode_masked(int cap, int valid_len, int num_q_heads,
                       int num_kv_heads, int head_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Dq  = num_q_heads  * head_dim;
    const int Dkv = num_kv_heads * head_dim;

    Tensor Q  = make_q16_cpu(1, Dq, rng, 0.4f);
    Tensor Kc = make_q16_cpu(cap, Dkv, rng, 0.4f);
    Tensor Vc = make_q16_cpu(cap, Dkv, rng, 0.4f);
    // Poison the masked-out rows.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int r = valid_len; r < cap; ++r) {
        for (int c = 0; c < Dkv; ++c) {
            Kc.host_f32_mut()[static_cast<size_t>(r) * Dkv + c] = nan;
            Vc.host_f32_mut()[static_cast<size_t>(r) * Dkv + c] = nan;
        }
    }
    std::vector<float> mask(static_cast<size_t>(cap), 0.0f);
    for (int r = 0; r < valid_len; ++r) mask[static_cast<size_t>(r)] = 1.0f;

    // CPU: masked over the full cap == truncated at valid_len, exactly.
    Tensor cpu_O_ref, cpu_O_masked;
    brotensor::flash_attention_decode(Q, Kc, Vc, valid_len,
                                      num_q_heads, num_kv_heads, cpu_O_ref);
    brotensor::flash_attention_decode_masked(Q, Kc, Vc, mask.data(),
                                             num_q_heads, num_kv_heads,
                                             cpu_O_masked);
    compare_tensors(cpu_O_ref, cpu_O_masked, "decode_masked_cpu_exact",
                    1e-7f, 1e-7f);

    // GPU: same equivalence on the FP16 path, expected bit-identical.
    Tensor gQ  = to_fp16_cuda(Q);
    Tensor gKc = to_fp16_cuda(Kc);
    Tensor gVc = to_fp16_cuda(Vc);
    Tensor gmask = Tensor::from_host_on(gpu_device(), mask.data(), cap, 1);
    Tensor gpu_O_ref, gpu_O_masked;
    brotensor::flash_attention_decode(gQ, gKc, gVc, valid_len,
                                      num_q_heads, num_kv_heads, gpu_O_ref);
    brotensor::flash_attention_decode_masked(
        gQ, gKc, gVc, static_cast<const float*>(gmask.data),
        num_q_heads, num_kv_heads, gpu_O_masked);
    compare_tensors(fp16_cuda_to_cpu(gpu_O_ref), fp16_cuda_to_cpu(gpu_O_masked),
                    "decode_masked_gpu_exact", 0.0f, 0.0f);

    // And the usual CPU-vs-GPU parity for the masked op itself.
    compare_tensors(cpu_O_masked, fp16_cuda_to_cpu(gpu_O_masked),
                    "decode_masked_parity", 5e-2f, 5e-2f);
}

} // namespace

// ─── kv_cache_append ───────────────────────────────────────────────────────
BT_PARITY_TEST(kv_cache_append_small)  { run_append(8,  4,  3, 2, 0x7300ull); }
BT_PARITY_TEST(kv_cache_append_wide)   { run_append(16, 32, 5, 4, 0x7301ull); }
BT_PARITY_TEST(kv_cache_append_full)   { run_append(10, 8,  6, 4, 0x7302ull); }
BT_PARITY_TEST(kv_cache_append_single) { run_append(12, 16, 1, 1, 0x7303ull); }

// ─── flash_attention_decode (MHA: num_kv_heads == num_q_heads) ─────────────
//   args: valid_len, Lq, num_q_heads, num_kv_heads, head_dim, seed
BT_PARITY_TEST(kv_decode_single_query) { run_decode(12, 1, 2, 2, 8,  0x7310ull); }
BT_PARITY_TEST(kv_decode_tail)         { run_decode(12, 3, 2, 2, 8,  0x7311ull); }
BT_PARITY_TEST(kv_decode_one_head)     { run_decode(10, 2, 1, 1, 8,  0x7312ull); }
BT_PARITY_TEST(kv_decode_full_prefill) { run_decode(8,  8, 4, 4, 8,  0x7313ull); }
BT_PARITY_TEST(kv_decode_wide_head)    { run_decode(20, 2, 1, 1, 64, 0x7314ull); }

// ─── flash_attention_decode (GQA: num_kv_heads < num_q_heads) ──────────────
// Qwen3-shaped 8q/2kv (group 4), 2q/1kv MQA, and a wider 6q/3kv (group 2).
BT_PARITY_TEST(kv_decode_gqa_qwen3)    { run_decode(16, 1, 8, 2, 16, 0x7315ull); }
BT_PARITY_TEST(kv_decode_gqa_tail)     { run_decode(16, 3, 8, 2, 16, 0x7316ull); }
BT_PARITY_TEST(kv_decode_mqa)          { run_decode(12, 1, 2, 1, 8,  0x7317ull); }
BT_PARITY_TEST(kv_decode_gqa_group2)   { run_decode(14, 2, 6, 3, 8,  0x7318ull); }

// ─── flash_attention_decode_masked (fixed-cap + key mask) ──────────────────
//   args: cap, valid_len, num_q_heads, num_kv_heads, head_dim, seed
BT_PARITY_TEST(kv_decode_masked_small)     { run_decode_masked(16,  9,  2, 2, 8,   0x7320ull); }
BT_PARITY_TEST(kv_decode_masked_gqa)       { run_decode_masked(96,  41, 8, 2, 16,  0x7321ull); }
BT_PARITY_TEST(kv_decode_masked_multitile) { run_decode_masked(256, 130, 4, 2, 32, 0x7322ull); }
BT_PARITY_TEST(kv_decode_masked_tile_edge) { run_decode_masked(192, 64,  4, 4, 16, 0x7323ull); }
BT_PARITY_TEST(kv_decode_masked_one_key)   { run_decode_masked(128, 1,   8, 2, 64, 0x7324ull); }

// ─── BF16: BF16-on-CUDA vs FP32 CPU reference ─────────────────────────────
// Same harness as FP16 but quantise through BF16 (7-bit mantissa).
// kv_cache_append is a pure copy — atol/rtol=2e-2 is ample.
// flash_attention_decode accumulates via softmax — use 5e-2 like the FP16 case.

namespace {

inline float q_bf16(float v) {
    return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
}

Tensor make_q_bf16_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = q_bf16(rng.next_unit() * scale);
    return t;
}

Tensor to_bf16_gpu_from_cpu(const Tensor& cpu) {
    return to_bf16_gpu(cpu);
}

Tensor bf16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    return bf16_host_to_f32(download_to_host(g));
}

void run_append_bf16(int L_max, int D, int len0, int len1, uint64_t seed) {
    SplitMix64 rng(seed);

    Tensor K0 = make_q_bf16_cpu(len0, D, rng, 1.0f);
    Tensor V0 = make_q_bf16_cpu(len0, D, rng, 1.0f);
    Tensor K1 = make_q_bf16_cpu(len1, D, rng, 1.0f);
    Tensor V1 = make_q_bf16_cpu(len1, D, rng, 1.0f);

    // CPU: FP32 reference caches.
    Tensor cpu_Kc = Tensor::mat(L_max, D);
    Tensor cpu_Vc = Tensor::mat(L_max, D);
    cpu_Kc.zero();
    cpu_Vc.zero();
    brotensor::kv_cache_append(K0, V0, 0,    cpu_Kc, cpu_Vc);
    brotensor::kv_cache_append(K1, V1, len0, cpu_Kc, cpu_Vc);

    // GPU: BF16 caches.
    Tensor gpu_Kc = Tensor::zeros_on(gpu_device(), L_max, D, Dtype::BF16);
    Tensor gpu_Vc = Tensor::zeros_on(gpu_device(), L_max, D, Dtype::BF16);
    Tensor gK0 = to_bf16_gpu_from_cpu(K0), gV0 = to_bf16_gpu_from_cpu(V0);
    Tensor gK1 = to_bf16_gpu_from_cpu(K1), gV1 = to_bf16_gpu_from_cpu(V1);
    brotensor::kv_cache_append(gK0, gV0, 0,    gpu_Kc, gpu_Vc);
    brotensor::kv_cache_append(gK1, gV1, len0, gpu_Kc, gpu_Vc);

    compare_tensors(cpu_Kc, bf16_cuda_to_cpu(gpu_Kc), "kv_append_bf16_K",
                    2e-2f, 2e-2f);
    compare_tensors(cpu_Vc, bf16_cuda_to_cpu(gpu_Vc), "kv_append_bf16_V",
                    2e-2f, 2e-2f);
}

void run_decode_bf16(int valid_len, int Lq, int num_q_heads, int num_kv_heads,
                     int head_dim, uint64_t seed) {
    SplitMix64 rng(seed);
    const int Dq  = num_q_heads  * head_dim;
    const int Dkv = num_kv_heads * head_dim;
    Tensor Q  = make_q_bf16_cpu(Lq, Dq, rng, 0.4f);
    Tensor Kc = make_q_bf16_cpu(valid_len, Dkv, rng, 0.4f);
    Tensor Vc = make_q_bf16_cpu(valid_len, Dkv, rng, 0.4f);

    Tensor cpu_O;
    brotensor::flash_attention_decode(Q, Kc, Vc, valid_len,
                                      num_q_heads, num_kv_heads, cpu_O);

    Tensor gQ  = to_bf16_gpu_from_cpu(Q);
    Tensor gKc = to_bf16_gpu_from_cpu(Kc);
    Tensor gVc = to_bf16_gpu_from_cpu(Vc);
    Tensor gpu_O;
    brotensor::flash_attention_decode(gQ, gKc, gVc, valid_len,
                                      num_q_heads, num_kv_heads, gpu_O);

    compare_tensors(cpu_O, bf16_cuda_to_cpu(gpu_O), "flash_decode_bf16",
                    5e-2f, 5e-2f);
}

} // namespace (bf16 helpers)

BT_PARITY_TEST(kv_cache_bf16_append_small)  { run_append_bf16(8,  4,  3, 2, 0x7320ull); }
BT_PARITY_TEST(kv_cache_bf16_append_wide)   { run_append_bf16(16, 32, 5, 4, 0x7321ull); }
BT_PARITY_TEST(kv_cache_bf16_append_full)   { run_append_bf16(10, 8,  6, 4, 0x7322ull); }

//   args: valid_len, Lq, num_q_heads, num_kv_heads, head_dim, seed
BT_PARITY_TEST(kv_decode_bf16_single_query) { run_decode_bf16(12, 1, 2, 2, 8,  0x7330ull); }
BT_PARITY_TEST(kv_decode_bf16_tail)         { run_decode_bf16(12, 3, 2, 2, 8,  0x7331ull); }
BT_PARITY_TEST(kv_decode_bf16_one_head)     { run_decode_bf16(10, 2, 1, 1, 8,  0x7332ull); }
BT_PARITY_TEST(kv_decode_bf16_full_prefill) { run_decode_bf16(8,  8, 4, 4, 8,  0x7333ull); }
// GQA (num_kv_heads < num_q_heads): Qwen3-shaped 8q/2kv and 2q/1kv MQA.
BT_PARITY_TEST(kv_decode_bf16_gqa_qwen3)    { run_decode_bf16(16, 1, 8, 2, 16, 0x7334ull); }
BT_PARITY_TEST(kv_decode_bf16_mqa)          { run_decode_bf16(12, 1, 2, 1, 8,  0x7335ull); }

int main() { return run_all("kv-cache cpu/gpu parity"); }
