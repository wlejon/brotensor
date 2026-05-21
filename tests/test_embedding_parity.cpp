// CPU↔GPU parity tests for brotensor::embedding_lookup_forward / _backward.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
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

    // CPU forward: gather rows.
    Tensor out_cpu = Tensor::mat(B, D);
    for (int b = 0; b < B; ++b) {
        const int row = idx[b];
        for (int j = 0; j < D; ++j) out_cpu(b, j) = table(row, j);
    }
    // CPU backward: scatter-accumulate. Pre-fill with non-zero to verify +=.
    Tensor dTable_init = Tensor::mat(V, D);
    fill_random(dTable_init, rng, 0.25f);
    Tensor dTable_cpu = dTable_init;
    for (int b = 0; b < B; ++b) {
        const int row = idx[b];
        for (int j = 0; j < D; ++j) dTable_cpu(row, j) += dOut(b, j);
    }

    // GPU.
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

BT_PARITY_TEST(emb_V8_D4_distinct) {
    run_embedding(8, 4, {0, 1, 2, 3, 4, 5, 6, 7}, 0x500ull);
}
BT_PARITY_TEST(emb_V16_D32_repeats) {
    // Repeated indices to exercise scatter accumulation.
    run_embedding(16, 32, {3, 3, 3, 7, 7, 1, 0, 15, 15, 15}, 0x501ull);
}
BT_PARITY_TEST(emb_V64_D8_random) {
    std::vector<int32_t> idx;
    SplitMix64 rng(0x502ull);
    for (int i = 0; i < 32; ++i) {
        idx.push_back(static_cast<int32_t>(rng.next_u64() % 64));
    }
    run_embedding(64, 8, idx, 0x502ull);
}
BT_PARITY_TEST(emb_V4_D16_all_same) {
    run_embedding(4, 16, {2, 2, 2, 2, 2, 2}, 0x503ull);
}

// ─── BF16: BF16-on-CUDA vs FP32 CPU reference ─────────────────────────────
// Forward is a pure gather — atol/rtol=2e-2 covers BF16 rounding on the
// table values. Backward scatter-accumulates: use 4e-2.

namespace {

void run_embedding_bf16(int V, int D, const std::vector<int32_t>& idx,
                        uint64_t seed) {
    const int B = static_cast<int>(idx.size());
    SplitMix64 rng(seed);

    Tensor table_f32 = Tensor::mat(V, D), dOut_f32 = Tensor::mat(B, D);
    fill_random(table_f32, rng);
    fill_random(dOut_f32, rng);

    // CPU forward reference (FP32).
    Tensor out_cpu = Tensor::mat(B, D);
    for (int b = 0; b < B; ++b) {
        const int row = idx[b];
        for (int j = 0; j < D; ++j) out_cpu(b, j) = table_f32(row, j);
    }
    // CPU backward reference: pre-fill zero (GPU dTable starts zero too).
    Tensor dTable_cpu = Tensor::mat(V, D);
    dTable_cpu.zero();
    for (int b = 0; b < B; ++b) {
        const int row = idx[b];
        for (int j = 0; j < D; ++j) dTable_cpu(row, j) += dOut_f32(b, j);
    }

    // GPU: convert table and dOut to BF16.
    Tensor gtable  = to_bf16_cuda(table_f32);
    Tensor gdOut   = to_bf16_cuda(dOut_f32);
    Tensor gout    = Tensor::zeros_on(Device::CUDA, B, D,
                                      brotensor::Dtype::BF16);
    Tensor gdTable = Tensor::zeros_on(Device::CUDA, V, D,
                                      brotensor::Dtype::BF16);

    Tensor d_idx_buf = upload_indices(idx);
    const int32_t* d_idx = static_cast<const int32_t*>(d_idx_buf.data);

    brotensor::embedding_lookup_forward(gtable, d_idx, B, gout);
    brotensor::embedding_lookup_backward(gdOut, d_idx, B, gdTable);

    Tensor out_gpu    = bf16_host_to_f32(download_to_host(gout));
    Tensor dTable_gpu = bf16_host_to_f32(download_to_host(gdTable));

    compare_tensors(out_cpu, out_gpu, "emb_bf16.out", 2e-2f, 2e-2f);
    compare_tensors(dTable_cpu, dTable_gpu, "emb_bf16.dTable", 4e-2f, 4e-2f);
}

} // namespace (bf16 helpers)

BT_PARITY_TEST(emb_bf16_distinct) {
    run_embedding_bf16(8, 4, {0, 1, 2, 3, 4, 5, 6, 7}, 0x510ull);
}
BT_PARITY_TEST(emb_bf16_repeats) {
    run_embedding_bf16(16, 32, {3, 3, 3, 7, 7, 1, 0, 15, 15, 15}, 0x511ull);
}
BT_PARITY_TEST(emb_bf16_random) {
    std::vector<int32_t> idx;
    SplitMix64 rng(0x512ull);
    for (int i = 0; i < 32; ++i)
        idx.push_back(static_cast<int32_t>(rng.next_u64() % 64));
    run_embedding_bf16(64, 8, idx, 0x512ull);
}
BT_PARITY_TEST(emb_bf16_all_same) {
    run_embedding_bf16(4, 16, {2, 2, 2, 2, 2, 2}, 0x513ull);
}

int main() { return run_all("embedding cpu/gpu parity"); }
