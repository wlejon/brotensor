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
        parts_gpu[i] = parts_cpu[i].to(gpu_device());
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
        parts_gpu[i] = parts_cpu[i].to(gpu_device());
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

// ─── Regression: concat must drain pending GPU work before its host memcpy ──
//
// The Metal concat ops copy buffer memory with a host memcpy (valid on
// unified memory) rather than a compute kernel — but a host memcpy is not
// ordered against the Metal command queue. If a concat input is the still-
// pending output of an earlier GPU kernel, an unsynchronised memcpy reads
// stale data. The tests above never caught this: they concat freshly
// uploaded tensors, with no GPU compute in flight.
//
// These tests queue several async matmuls (matmul submits to the queue
// without waiting), then concat the still-in-flight outputs *twice*: once
// immediately (racy) and once after an explicit sync (settled). A correct
// concat drains the queue itself, so the two results are bit-identical. An
// unsynchronised concat reads stale data on the first call and the two
// disagree. The comparison is exact and dtype-agnostic — both operands are
// GPU results of identical provenance, so no arithmetic tolerance applies.

// Bitwise equality of two device tensors (downloaded; dtype-agnostic).
bool gpu_tensors_bit_equal(const Tensor& a_dev, const Tensor& b_dev) {
    brotensor::sync_all();
    Tensor a = a_dev.to(brotensor::Device::CPU);
    Tensor b = b_dev.to(brotensor::Device::CPU);
    if (a.rows != b.rows || a.cols != b.cols || a.dtype != b.dtype) return false;
    return std::memcmp(a.host_raw(), b.host_raw(), a.bytes()) == 0;
}

void run_nchw_after_compute(int N, int H, int W,
                            const std::vector<int>& C_per_part,
                            int K, uint64_t seed) {
    SplitMix64 rng(seed);
    const int np = static_cast<int>(C_per_part.size());
    const int warmup = 8;
    // Every GPU operand lives here for the whole function so its buffer
    // outlives the async matmul that references it.
    std::vector<Tensor> keep;
    keep.reserve(static_cast<size_t>(warmup + np) * 2);

    // Queue one async matmul (M,K)@(K,Nmm) and return its (still in-flight)
    // output of shape (M, Nmm).
    auto queue_one = [&](int M, int Nmm) -> Tensor {
        Tensor a = Tensor::mat(M, K);   fill_random(a, rng);
        Tensor b = Tensor::mat(K, Nmm); fill_random(b, rng);
        keep.push_back(a.to(gpu_device()));
        keep.push_back(b.to(gpu_device()));
        Tensor out;
        brotensor::matmul(keep[keep.size() - 2], keep[keep.size() - 1], out);
        return out;
    };

    // Deepen the command queue, then produce the concat parts — all async.
    std::vector<Tensor> discard;
    for (int i = 0; i < warmup; ++i) discard.push_back(queue_one(N, 64 * 64));
    std::vector<Tensor> parts;
    for (int Ci : C_per_part) parts.push_back(queue_one(N, Ci * H * W));

    std::vector<const Tensor*> ptr;
    for (const auto& p : parts) ptr.push_back(&p);

    Tensor out_racy;     // concat while the matmuls are still in flight
    brotensor::concat_nchw_channels(ptr, N, H, W, C_per_part, out_racy);
    brotensor::sync_all();
    Tensor out_settled;  // concat again, queue now fully drained
    brotensor::concat_nchw_channels(ptr, N, H, W, C_per_part, out_settled);

    BT_CHECK(gpu_tensors_bit_equal(out_racy, out_settled));
}

void run_batched_after_compute(int B, const std::vector<int>& cols,
                               int K, uint64_t seed) {
    SplitMix64 rng(seed);
    const int np = static_cast<int>(cols.size());
    const int warmup = 8;
    std::vector<Tensor> keep;
    keep.reserve(static_cast<size_t>(warmup + np) * 2);

    auto queue_one = [&](int M, int Nmm) -> Tensor {
        Tensor a = Tensor::mat(M, K);   fill_random(a, rng);
        Tensor b = Tensor::mat(K, Nmm); fill_random(b, rng);
        keep.push_back(a.to(gpu_device()));
        keep.push_back(b.to(gpu_device()));
        Tensor out;
        brotensor::matmul(keep[keep.size() - 2], keep[keep.size() - 1], out);
        return out;
    };

    std::vector<Tensor> discard;
    for (int i = 0; i < warmup; ++i) discard.push_back(queue_one(B, 256));
    std::vector<Tensor> parts;
    for (int c : cols) parts.push_back(queue_one(B, c));

    std::vector<const Tensor*> ptr;
    for (const auto& p : parts) ptr.push_back(&p);

    Tensor out_racy;
    brotensor::concat_batched_rows(ptr, out_racy);
    brotensor::sync_all();
    Tensor out_settled;
    brotensor::concat_batched_rows(ptr, out_settled);

    BT_CHECK(gpu_tensors_bit_equal(out_racy, out_settled));
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

// ─── BF16: BF16-on-CUDA vs FP32 CPU reference ─────────────────────────────
// concat_batched_rows and concat_nchw_channels are pure memcpy; the BF16
// rounding is on the input values only. atol/rtol=2e-2 covers that.

namespace {

void run_batched_rows_bf16(int B, const std::vector<int>& cols, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<Tensor> parts_cpu;
    for (int c : cols) {
        Tensor t = Tensor::mat(B, c);
        fill_random(t, rng);
        parts_cpu.push_back(std::move(t));
    }

    // FP32 CPU reference with BF16-rounded inputs.
    std::vector<Tensor> parts_f32(cols.size());
    for (size_t i = 0; i < cols.size(); ++i)
        parts_f32[i] = bf16_host_to_f32(to_bf16_host(parts_cpu[i]));
    std::vector<const Tensor*> cpu_ptr;
    for (const auto& p : parts_f32) cpu_ptr.push_back(&p);
    Tensor out_cpu;
    brotensor::concat_batched_rows(cpu_ptr, out_cpu);

    // GPU: BF16 parts.
    std::vector<Tensor> parts_gpu(cols.size());
    std::vector<const Tensor*> gpu_ptr;
    for (size_t i = 0; i < cols.size(); ++i) {
        parts_gpu[i] = to_bf16_cuda(parts_cpu[i]);
        gpu_ptr.push_back(&parts_gpu[i]);
    }
    Tensor out_gpu_bf16;
    brotensor::concat_batched_rows(gpu_ptr, out_gpu_bf16);
    brotensor::sync_all();

    Tensor out_gpu = bf16_host_to_f32(download_to_host(out_gpu_bf16));
    compare_tensors(out_cpu, out_gpu, "cbr_bf16", 2e-2f, 2e-2f);
}

void run_nchw_bf16(int N, int H, int W, const std::vector<int>& C_per_part,
                   uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<Tensor> parts_cpu;
    for (int Ci : C_per_part) {
        Tensor t = Tensor::mat(N, Ci * H * W);
        fill_random(t, rng);
        parts_cpu.push_back(std::move(t));
    }

    // FP32 CPU reference with BF16-rounded inputs.
    std::vector<Tensor> parts_f32(C_per_part.size());
    for (size_t i = 0; i < C_per_part.size(); ++i)
        parts_f32[i] = bf16_host_to_f32(to_bf16_host(parts_cpu[i]));
    std::vector<const Tensor*> cpu_ptr;
    for (const auto& p : parts_f32) cpu_ptr.push_back(&p);
    Tensor out_cpu;
    brotensor::concat_nchw_channels(cpu_ptr, N, H, W, C_per_part, out_cpu);

    // GPU: BF16 parts.
    std::vector<Tensor> parts_gpu(C_per_part.size());
    std::vector<const Tensor*> gpu_ptr;
    for (size_t i = 0; i < C_per_part.size(); ++i) {
        parts_gpu[i] = to_bf16_cuda(parts_cpu[i]);
        gpu_ptr.push_back(&parts_gpu[i]);
    }
    Tensor out_gpu_bf16;
    brotensor::concat_nchw_channels(gpu_ptr, N, H, W, C_per_part, out_gpu_bf16);
    brotensor::sync_all();

    Tensor out_gpu = bf16_host_to_f32(download_to_host(out_gpu_bf16));
    compare_tensors(out_cpu, out_gpu, "nchw_bf16", 2e-2f, 2e-2f);

    // Backward.
    std::vector<Tensor> dparts_cpu(C_per_part.size());
    std::vector<Tensor*> dcpu_ptr;
    for (auto& t : dparts_cpu) dcpu_ptr.push_back(&t);
    brotensor::concat_nchw_channels_backward(out_cpu, N, H, W, C_per_part,
                                             dcpu_ptr);

    std::vector<Tensor> dparts_gpu_bf16(C_per_part.size());
    std::vector<Tensor*> dgpu_ptr;
    for (auto& t : dparts_gpu_bf16) dgpu_ptr.push_back(&t);
    brotensor::concat_nchw_channels_backward(out_gpu_bf16, N, H, W, C_per_part,
                                             dgpu_ptr);
    brotensor::sync_all();

    for (size_t i = 0; i < C_per_part.size(); ++i) {
        Tensor seg = bf16_host_to_f32(download_to_host(dparts_gpu_bf16[i]));
        compare_tensors(dparts_cpu[i], seg, "nchw_bwd_bf16", 2e-2f, 2e-2f);
    }
}

} // namespace (bf16 helpers)

BT_PARITY_TEST(cbr_bf16_equal)   { run_batched_rows_bf16(4, {8, 8, 8}, 0xA20ull); }
BT_PARITY_TEST(cbr_bf16_varying) { run_batched_rows_bf16(7, {3, 17, 5}, 0xA21ull); }
BT_PARITY_TEST(cbr_bf16_two)     { run_batched_rows_bf16(16, {32, 8}, 0xA22ull); }

BT_PARITY_TEST(nchw_bf16_equal)   { run_nchw_bf16(2, 4, 4, {3, 3}, 0xA30ull); }
BT_PARITY_TEST(nchw_bf16_varying) { run_nchw_bf16(3, 2, 5, {1, 4, 2}, 0xA31ull); }
BT_PARITY_TEST(nchw_bf16_single)  { run_nchw_bf16(1, 8, 8, {16}, 0xA32ull); }

// Regression tests for the pending-GPU-work race (see comment above).
BT_PARITY_TEST(nchw_after_compute_a) {
    run_nchw_after_compute(1, 16, 16, {16, 8, 8}, 512, 0xA20ull);
}
BT_PARITY_TEST(nchw_after_compute_b) {
    run_nchw_after_compute(2, 8, 8, {8, 8}, 384, 0xA21ull);
}
BT_PARITY_TEST(cbr_after_compute) {
    run_batched_after_compute(4, {512, 256, 128}, 512, 0xA22ull);
}

int main() { return run_all("concat-extra cpu/gpu parity"); }
