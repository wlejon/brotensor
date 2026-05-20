// CPU↔GPU parity tests for the newly-CPU-ported embedding ops:
//   embedding_lookup_forward / embedding_lookup_backward.
//
// CHUNK 1. test_embedding_parity.cpp compares the GPU against an inline CPU
// reference; this file runs the actual CPU backend ops on both sides.

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

    // Backward accumulates: pre-fill dTable with non-zero to verify +=.
    Tensor dTable_init = Tensor::mat(V, D);
    fill_random(dTable_init, rng, 0.25f);

    // CPU backend run.
    Tensor out_cpu = Tensor::mat(B, D);
    brotensor::embedding_lookup_forward(table, idx.data(), B, out_cpu);
    Tensor dTable_cpu = dTable_init;  // deep copy
    brotensor::embedding_lookup_backward(dOut, idx.data(), B, dTable_cpu);

    // GPU backend run.
    Tensor gtable = table.to(Device::CUDA);
    Tensor gdOut = dOut.to(Device::CUDA);
    Tensor gout = Tensor::zeros_on(Device::CUDA, B, D);
    Tensor gdTable = dTable_init.to(Device::CUDA);
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

int main() { return run_all("embedding-extra cpu/gpu parity"); }
