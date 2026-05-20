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
// valid_len cache rows, Lq query rows (the tail). num_heads divides D.
void run_decode(int valid_len, int Lq, int D, int num_heads, uint64_t seed) {
    SplitMix64 rng(seed);
    // Small magnitude keeps FP16 softmax well-conditioned.
    Tensor Q  = make_q16_cpu(Lq, D, rng, 0.4f);
    Tensor Kc = make_q16_cpu(valid_len, D, rng, 0.4f);
    Tensor Vc = make_q16_cpu(valid_len, D, rng, 0.4f);

    Tensor cpu_O;
    brotensor::flash_attention_decode(Q, Kc, Vc, valid_len, num_heads, cpu_O);

    Tensor gQ  = to_fp16_cuda(Q);
    Tensor gKc = to_fp16_cuda(Kc);
    Tensor gVc = to_fp16_cuda(Vc);
    Tensor gpu_O;
    brotensor::flash_attention_decode(gQ, gKc, gVc, valid_len, num_heads,
                                      gpu_O);

    // FP16 dot-product + softmax reduction + fast-math expf/rsqrtf.
    compare_tensors(cpu_O, fp16_cuda_to_cpu(gpu_O), "flash_decode",
                    5e-2f, 5e-2f);
}

} // namespace

// ─── kv_cache_append ───────────────────────────────────────────────────────
BT_PARITY_TEST(kv_cache_append_small)  { run_append(8,  4,  3, 2, 0x7300ull); }
BT_PARITY_TEST(kv_cache_append_wide)   { run_append(16, 32, 5, 4, 0x7301ull); }
BT_PARITY_TEST(kv_cache_append_full)   { run_append(10, 8,  6, 4, 0x7302ull); }
BT_PARITY_TEST(kv_cache_append_single) { run_append(12, 16, 1, 1, 0x7303ull); }

// ─── flash_attention_decode ────────────────────────────────────────────────
BT_PARITY_TEST(kv_decode_single_query) { run_decode(12, 1, 16, 2, 0x7310ull); }
BT_PARITY_TEST(kv_decode_tail)         { run_decode(12, 3, 16, 2, 0x7311ull); }
BT_PARITY_TEST(kv_decode_one_head)     { run_decode(10, 2, 8,  1, 0x7312ull); }
BT_PARITY_TEST(kv_decode_full_prefill) { run_decode(8,  8, 32, 4, 0x7313ull); }
BT_PARITY_TEST(kv_decode_wide_head)    { run_decode(20, 2, 64, 1, 0x7314ull); }

int main() { return run_all("kv-cache cpu/gpu parity"); }
