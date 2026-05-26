// CPU↔GPU parity tests for the newly-CPU-ported embedding ops:
//   embedding_lookup_forward / embedding_lookup_backward.
//
// CHUNK 1. test_embedding_parity.cpp compares the GPU against an inline CPU
// reference; this file runs the actual CPU backend ops on both sides.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_embedding(int V, int D, const std::vector<int32_t>& idx,
                   uint64_t seed) {
    const int B = static_cast<int>(idx.size());
    SplitMix64 rng(seed);

    Tensor table = Tensor::mat(V, D), dOut = Tensor::mat(B, D);
    fill_random(table, rng);
    fill_random(dOut, rng);

    // Backward accumulates: pre-fill dTable with non-zero to verify +=.
    Tensor dTable_init = Tensor::mat(V, D);
    fill_random(dTable_init, rng, 0.25f);

    // CPU backend run.
    Tensor out_cpu = Tensor::mat(B, D);
    brotensor::embedding_lookup_forward(table, idx.data(), B, out_cpu);
    Tensor dTable_cpu = dTable_init;  // deep copy
    brotensor::embedding_lookup_backward(dOut, idx.data(), B, dTable_cpu);

    // GPU backend run.
    Tensor gtable = table.to(gpu_device());
    Tensor gdOut = dOut.to(gpu_device());
    Tensor gout = Tensor::zeros_on(gpu_device(), B, D);
    Tensor gdTable = dTable_init.to(gpu_device());
    Tensor d_idx_buf = upload_indices(idx);
    const int32_t* d_idx = static_cast<const int32_t*>(d_idx_buf.data);
    brotensor::embedding_lookup_forward(gtable, d_idx, B, gout);
    brotensor::embedding_lookup_backward(gdOut, d_idx, B, gdTable);

    Tensor out_gpu = download_to_host(gout);
    Tensor dTable_gpu = download_to_host(gdTable);
    compare_tensors(out_cpu, out_gpu, "emb.out");
    compare_tensors(dTable_cpu, dTable_gpu, "emb.dTable");
}

} // namespace

BT_PARITY_TEST(emb_extra_distinct) {
    run_embedding(8, 4, {0, 1, 2, 3, 4, 5, 6, 7}, 0x900ull);
}
BT_PARITY_TEST(emb_extra_repeats) {
    run_embedding(16, 32, {3, 3, 3, 7, 7, 1, 0, 15, 15, 15}, 0x901ull);
}
BT_PARITY_TEST(emb_extra_single) {
    run_embedding(4, 16, {2}, 0x902ull);
}
BT_PARITY_TEST(emb_extra_random) {
    std::vector<int32_t> idx;
    SplitMix64 rng(0x903ull);
    for (int i = 0; i < 24; ++i)
        idx.push_back(static_cast<int32_t>(rng.next_u64() % 32));
    run_embedding(32, 8, idx, 0x903ull);
}

// ─── BF16: BF16-on-CUDA vs FP32 CPU ops ───────────────────────────────────
// Run the actual CPU backend on both sides with BF16-rounded values, compare
// GPU BF16 result against CPU FP32 reference. Use 4e-2 for backward
// (scatter accumulates: FP32 scratch + fold-back into BF16).

namespace {

void run_embedding_bf16(int V, int D, const std::vector<int32_t>& idx,
                        uint64_t seed) {
    const int B = static_cast<int>(idx.size());
    SplitMix64 rng(seed);

    Tensor table_f32 = Tensor::mat(V, D), dOut_f32 = Tensor::mat(B, D);
    fill_random(table_f32, rng);
    fill_random(dOut_f32, rng);
    Tensor dTable_init = Tensor::mat(V, D);
    fill_random(dTable_init, rng, 0.25f);

    // CPU backend: run with FP32 tables (CPU is FP32-only).
    Tensor out_cpu = Tensor::mat(B, D);
    brotensor::embedding_lookup_forward(table_f32, idx.data(), B, out_cpu);
    Tensor dTable_cpu = dTable_init;
    brotensor::embedding_lookup_backward(dOut_f32, idx.data(), B, dTable_cpu);

    // GPU: BF16 tensors. dTable starts as zero (CPU pre-fill not comparable).
    Tensor gtable  = to_bf16_gpu(table_f32);
    Tensor gdOut   = to_bf16_gpu(dOut_f32);
    Tensor gout    = Tensor::zeros_on(gpu_device(), B, D,
                                      brotensor::Dtype::BF16);
    Tensor gdTable = Tensor::zeros_on(gpu_device(), V, D,
                                      brotensor::Dtype::BF16);

    Tensor d_idx_buf = upload_indices(idx);
    const int32_t* d_idx = static_cast<const int32_t*>(d_idx_buf.data);
    brotensor::embedding_lookup_forward(gtable, d_idx, B, gout);
    brotensor::embedding_lookup_backward(gdOut, d_idx, B, gdTable);

    Tensor out_gpu    = bf16_host_to_f32(download_to_host(gout));
    Tensor dTable_gpu = bf16_host_to_f32(download_to_host(gdTable));

    compare_tensors(out_cpu, out_gpu, "emb_extra_bf16.out", 2e-2f, 2e-2f);

    // For dTable: re-run CPU with zero init to match GPU zero-init baseline.
    Tensor dTable_cpu_zero = Tensor::mat(V, D);
    dTable_cpu_zero.zero();
    brotensor::embedding_lookup_backward(dOut_f32, idx.data(), B, dTable_cpu_zero);
    compare_tensors(dTable_cpu_zero, dTable_gpu, "emb_extra_bf16.dTable", 4e-2f, 4e-2f);
}

} // namespace (bf16 helpers)

BT_PARITY_TEST(emb_extra_bf16_distinct) {
    run_embedding_bf16(8, 4, {0, 1, 2, 3, 4, 5, 6, 7}, 0x910ull);
}
BT_PARITY_TEST(emb_extra_bf16_repeats) {
    run_embedding_bf16(16, 32, {3, 3, 3, 7, 7, 1, 0, 15, 15, 15}, 0x911ull);
}
BT_PARITY_TEST(emb_extra_bf16_random) {
    std::vector<int32_t> idx;
    SplitMix64 rng(0x912ull);
    for (int i = 0; i < 24; ++i)
        idx.push_back(static_cast<int32_t>(rng.next_u64() % 32));
    run_embedding_bf16(32, 8, idx, 0x912ull);
}

// Regression: embedding_lookup_forward must reject unsupported table dtypes
// (e.g. Q8_0) with a clear throw rather than silently reading the bytes as
// floats. Earlier behavior fell through to the FP32 kernel and produced a
// garbage output tagged with the quant dtype, which then surfaced far
// downstream as an unrelated "dtype mismatch" error in the first rms_norm
// of a model loaded straight from a Q8_0 gguf.
BT_PARITY_TEST(emb_rejects_q8_0_table) {
    using brotensor::Dtype;
    // Empty allocation is enough — the dtype guard fires before any data read,
    // so neither the table bytes nor the index pointer are dereferenced.
    Tensor table = Tensor::empty_on(gpu_device(), /*rows=*/8, /*cols=*/32,
                                    Dtype::Q8_0);
    std::vector<int32_t> idx = {0, 1, 2};
    Tensor out;
    bool threw = false;
    try {
        brotensor::embedding_lookup_forward(table, idx.data(),
                                            static_cast<int>(idx.size()), out);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    BT_CHECK(threw);
}

int main() { return run_all("embedding-extra cpu/gpu parity"); }
