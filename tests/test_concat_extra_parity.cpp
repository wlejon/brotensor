// CPU↔GPU parity tests for the newly-CPU-ported concat ops:
//   concat_batched_rows / concat_nchw_channels / concat_nchw_channels_backward.
//
// CHUNK 1. test_concat_parity.cpp covers concat_rows/split_rows; this file
// covers the batched-rows and NCHW-channel variants.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

// ─── concat_batched_rows: parts (B, c_i) -> out (B, sum c_i) ────────────────

void run_batched_rows(int B, const std::vector<int>& cols, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<Tensor> parts_cpu;
    for (int c : cols) {
        Tensor t = Tensor::mat(B, c);
        fill_random(t, rng);
        parts_cpu.push_back(std::move(t));
    }
    std::vector<const Tensor*> cpu_ptr;
    for (const auto& p : parts_cpu) cpu_ptr.push_back(&p);

    Tensor out_cpu;
    brotensor::concat_batched_rows(cpu_ptr, out_cpu);

    std::vector<Tensor> parts_gpu(cols.size());
    std::vector<const Tensor*> gpu_ptr;
    for (size_t i = 0; i < cols.size(); ++i) {
        parts_gpu[i] = parts_cpu[i].to(Device::CUDA);
        gpu_ptr.push_back(&parts_gpu[i]);
    }
    Tensor out_gpu;
    brotensor::concat_batched_rows(gpu_ptr, out_gpu);
    brotensor::sync_all();

    Tensor out_gpu_h = download_to_host(out_gpu);
    compare_tensors(out_cpu, out_gpu_h, "concat_batched_rows");
}

// ─── concat_nchw_channels + backward ───────────────────────────────────────

void run_nchw(int N, int H, int W, const std::vector<int>& C_per_part,
              uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<Tensor> parts_cpu;
    for (int Ci : C_per_part) {
        Tensor t = Tensor::mat(N, Ci * H * W);
        fill_random(t, rng);
        parts_cpu.push_back(std::move(t));
    }
    std::vector<const Tensor*> cpu_ptr;
    for (const auto& p : parts_cpu) cpu_ptr.push_back(&p);

    Tensor out_cpu;
    brotensor::concat_nchw_channels(cpu_ptr, N, H, W, C_per_part, out_cpu);

    std::vector<Tensor> parts_gpu(C_per_part.size());
    std::vector<const Tensor*> gpu_ptr;
    for (size_t i = 0; i < C_per_part.size(); ++i) {
        parts_gpu[i] = parts_cpu[i].to(Device::CUDA);
        gpu_ptr.push_back(&parts_gpu[i]);
    }
    Tensor out_gpu;
    brotensor::concat_nchw_channels(gpu_ptr, N, H, W, C_per_part, out_gpu);
    brotensor::sync_all();

    Tensor out_gpu_h = download_to_host(out_gpu);
    compare_tensors(out_cpu, out_gpu_h, "concat_nchw_channels");

    // Backward: split dY back into per-part tensors. Use out as a stand-in
    // for dY (an arbitrary correctly-shaped gradient).
    std::vector<Tensor> dparts_cpu(C_per_part.size());
    std::vector<Tensor*> dcpu_ptr;
    for (auto& t : dparts_cpu) dcpu_ptr.push_back(&t);
    brotensor::concat_nchw_channels_backward(out_cpu, N, H, W, C_per_part,
                                             dcpu_ptr);

    std::vector<Tensor> dparts_gpu(C_per_part.size());
    std::vector<Tensor*> dgpu_ptr;
    for (auto& t : dparts_gpu) dgpu_ptr.push_back(&t);
    brotensor::concat_nchw_channels_backward(out_gpu, N, H, W, C_per_part,
                                             dgpu_ptr);
    brotensor::sync_all();

    for (size_t i = 0; i < C_per_part.size(); ++i) {
        Tensor seg = download_to_host(dparts_gpu[i]);
        compare_tensors(dparts_cpu[i], seg, "concat_nchw_backward.part");
        // Backward must reproduce the original part exactly (round-trip).
        compare_tensors(parts_cpu[i], dparts_cpu[i], "concat_nchw_roundtrip");
    }
}

} // namespace

BT_PARITY_TEST(cbr_equal)    { run_batched_rows(4, {8, 8, 8}, 0xA00ull); }
BT_PARITY_TEST(cbr_varying)  { run_batched_rows(7, {3, 17, 5}, 0xA01ull); }
BT_PARITY_TEST(cbr_single)   { run_batched_rows(1, {16}, 0xA02ull); }
BT_PARITY_TEST(cbr_two)      { run_batched_rows(16, {32, 8}, 0xA03ull); }

BT_PARITY_TEST(nchw_equal)   { run_nchw(2, 4, 4, {3, 3}, 0xA10ull); }
BT_PARITY_TEST(nchw_varying) { run_nchw(3, 2, 5, {1, 4, 2}, 0xA11ull); }
BT_PARITY_TEST(nchw_single)  { run_nchw(1, 8, 8, {16}, 0xA12ull); }
BT_PARITY_TEST(nchw_1x1)     { run_nchw(1, 1, 1, {2, 3}, 0xA13ull); }

int main() { return run_all("concat-extra cpu/gpu parity"); }
