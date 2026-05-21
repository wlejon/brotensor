// CPU↔GPU parity tests for the tensor-utility ops that had no parity
// coverage: scale_inplace / cast / copy_d2d / build_slot_mask.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

// ─── scale_inplace ─────────────────────────────────────────────────────────
//
// y *= s, in place, FP32.

void run_scale_inplace(int n, float s, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor y = Tensor::vec(n);
    fill_random(y, rng);

    Tensor y_cpu = y;
    brotensor::scale_inplace(y_cpu, s);

    Tensor gy = y.to(gpu_device());
    brotensor::scale_inplace(gy, s);

    compare_tensors(y_cpu, download_to_host(gy), "scale_inplace");
}

// ─── cast ──────────────────────────────────────────────────────────────────
//
// dst = src converted to out_dtype. Tested for FP32->FP16, FP16->FP32, and
// same-dtype passthrough. FP16 storage uses loose tolerances.

void run_cast_fp32_to_fp16(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor src = Tensor::mat(r, c);
    fill_random(src, rng);

    Tensor dst_cpu;
    brotensor::cast(src, dst_cpu, Dtype::FP16);
    BT_CHECK(dst_cpu.dtype == Dtype::FP16);
    BT_CHECK(dst_cpu.rows == r && dst_cpu.cols == c);

    Tensor gsrc = src.to(gpu_device());
    Tensor gdst;
    brotensor::cast(gsrc, gdst, Dtype::FP16);
    BT_CHECK(gdst.dtype == Dtype::FP16);

    // Compare the resulting FP16 bit patterns: both should be the IEEE-754
    // binary16 rounding of the same FP32 source, so they must match exactly.
    std::vector<uint16_t> cpu_bits = dst_cpu.to_host_vector_fp16();
    std::vector<uint16_t> gpu_bits = gdst.to_host_vector_fp16();
    BT_CHECK(cpu_bits.size() == gpu_bits.size());
    for (size_t i = 0; i < cpu_bits.size(); ++i) {
        const float a = brotensor::fp16_bits_to_fp32(cpu_bits[i]);
        const float b = brotensor::fp16_bits_to_fp32(gpu_bits[i]);
        BT_CHECK(std::fabs(a - b) <= 2e-3f + 2e-3f * std::fabs(a));
    }
}

void run_cast_fp16_to_fp32(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    // Build an FP16 source by casting a random FP32 tensor down first.
    Tensor src_fp32 = Tensor::mat(r, c);
    fill_random(src_fp32, rng);
    Tensor src_fp16;
    brotensor::cast(src_fp32, src_fp16, Dtype::FP16);

    Tensor dst_cpu;
    brotensor::cast(src_fp16, dst_cpu, Dtype::FP32);
    BT_CHECK(dst_cpu.dtype == Dtype::FP32);

    Tensor gsrc = src_fp16.to(gpu_device());
    Tensor gdst;
    brotensor::cast(gsrc, gdst, Dtype::FP32);
    BT_CHECK(gdst.dtype == Dtype::FP32);

    compare_tensors(dst_cpu, download_to_host(gdst), "cast.fp16_to_fp32",
                    2e-3f, 2e-3f);
}

void run_cast_passthrough(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor src = Tensor::mat(r, c);
    fill_random(src, rng);

    Tensor dst_cpu;
    brotensor::cast(src, dst_cpu, Dtype::FP32);
    BT_CHECK(dst_cpu.dtype == Dtype::FP32);

    Tensor gsrc = src.to(gpu_device());
    Tensor gdst;
    brotensor::cast(gsrc, gdst, Dtype::FP32);

    // Same-dtype passthrough must reproduce the source exactly.
    compare_tensors(src, dst_cpu, "cast.passthrough_cpu");
    compare_tensors(dst_cpu, download_to_host(gdst), "cast.passthrough");
}

void run_cast_fp32_to_bf16(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor src = Tensor::mat(r, c);
    fill_random(src, rng);

    Tensor dst_cpu;
    brotensor::cast(src, dst_cpu, Dtype::BF16);
    BT_CHECK(dst_cpu.dtype == Dtype::BF16);
    BT_CHECK(dst_cpu.rows == r && dst_cpu.cols == c);

    Tensor gsrc = src.to(gpu_device());
    Tensor gdst;
    brotensor::cast(gsrc, gdst, Dtype::BF16);
    BT_CHECK(gdst.dtype == Dtype::BF16);

    // CPU and GPU both round the same FP32 source to BF16 with round-to-
    // nearest-even, so the bit patterns must match exactly.
    std::vector<uint16_t> cpu_bits = dst_cpu.to_host_vector_bf16();
    std::vector<uint16_t> gpu_bits = gdst.to_host_vector_bf16();
    BT_CHECK(cpu_bits.size() == gpu_bits.size());
    for (size_t i = 0; i < cpu_bits.size(); ++i) {
        BT_CHECK(cpu_bits[i] == gpu_bits[i]);
    }
}

void run_cast_bf16_to_fp32(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    // Build a BF16 source by casting a random FP32 tensor down first.
    Tensor src_fp32 = Tensor::mat(r, c);
    fill_random(src_fp32, rng);
    Tensor src_bf16;
    brotensor::cast(src_fp32, src_bf16, Dtype::BF16);

    Tensor dst_cpu;
    brotensor::cast(src_bf16, dst_cpu, Dtype::FP32);
    BT_CHECK(dst_cpu.dtype == Dtype::FP32);

    Tensor gsrc = src_bf16.to(gpu_device());
    Tensor gdst;
    brotensor::cast(gsrc, gdst, Dtype::FP32);
    BT_CHECK(gdst.dtype == Dtype::FP32);

    // BF16->FP32 widening is lossless: CPU and GPU must agree exactly.
    compare_tensors(dst_cpu, download_to_host(gdst), "cast.bf16_to_fp32",
                    0.0f, 0.0f);
}

// ─── copy_d2d ──────────────────────────────────────────────────────────────
//
// Flat-buffer sub-range copy. Verify a partial copy leaves the untouched
// region of dst intact.

void run_copy_d2d(int src_n, int dst_n, int src_off, int dst_off, int n,
                  uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor src = Tensor::vec(src_n);
    Tensor dst_init = Tensor::vec(dst_n);
    fill_random(src, rng);
    fill_random(dst_init, rng);

    // CPU path.
    Tensor dst_cpu = dst_init;
    brotensor::copy_d2d(src, src_off, dst_cpu, dst_off, n);

    // GPU path.
    Tensor gsrc = src.to(gpu_device());
    Tensor gdst = dst_init.to(gpu_device());
    brotensor::copy_d2d(gsrc, src_off, gdst, dst_off, n);

    Tensor dst_gpu = download_to_host(gdst);
    compare_tensors(dst_cpu, dst_gpu, "copy_d2d");

    // Explicitly check: copied range matches src, untouched range unchanged.
    for (int i = 0; i < dst_n; ++i) {
        if (i >= dst_off && i < dst_off + n) {
            BT_CHECK(dst_gpu[i] == src[src_off + (i - dst_off)]);
        } else {
            BT_CHECK(dst_gpu[i] == dst_init[i]);
        }
    }
}

// ─── build_slot_mask ───────────────────────────────────────────────────────
//
// mask[k] = (x[offset + k*stride] > 0.5f) ? 1 : 0; mask resized to (K, 1).

void run_build_slot_mask(int x_n, int offset, int K, int stride,
                         uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(x_n);
    // Scale by 2 so values straddle the 0.5 threshold roughly evenly.
    fill_random(x, rng, 2.0f);

    Tensor mask_cpu;
    brotensor::build_slot_mask(x, offset, K, stride, mask_cpu);
    BT_CHECK(mask_cpu.rows == K && mask_cpu.cols == 1);

    Tensor gx = x.to(gpu_device());
    Tensor gmask;
    brotensor::build_slot_mask(gx, offset, K, stride, gmask);
    BT_CHECK(gmask.rows == K && gmask.cols == 1);

    Tensor mask_gpu = download_to_host(gmask);
    compare_tensors(mask_cpu, mask_gpu, "build_slot_mask");

    // Independent reference check.
    for (int k = 0; k < K; ++k) {
        const float expect =
            (x[offset + k * stride] > 0.5f) ? 1.0f : 0.0f;
        BT_CHECK(mask_gpu[k] == expect);
    }
}

} // namespace

// ─── scale_inplace ─────────────────────────────────────────────────────────
BT_PARITY_TEST(scale_inplace_n1)    { run_scale_inplace(1, 0.375f, 0x200ull); }
BT_PARITY_TEST(scale_inplace_n64)   { run_scale_inplace(64, -1.5f, 0x201ull); }
BT_PARITY_TEST(scale_inplace_n1024) { run_scale_inplace(1024, 2.0f, 0x202ull); }

// ─── cast ──────────────────────────────────────────────────────────────────
BT_PARITY_TEST(cast_fp32_to_fp16_small)  { run_cast_fp32_to_fp16(4, 8, 0x210ull); }
BT_PARITY_TEST(cast_fp32_to_fp16_medium) { run_cast_fp32_to_fp16(16, 64, 0x211ull); }
BT_PARITY_TEST(cast_fp16_to_fp32_small)  { run_cast_fp16_to_fp32(4, 8, 0x212ull); }
BT_PARITY_TEST(cast_fp16_to_fp32_medium) { run_cast_fp16_to_fp32(16, 64, 0x213ull); }
BT_PARITY_TEST(cast_passthrough_small)   { run_cast_passthrough(4, 8, 0x214ull); }
BT_PARITY_TEST(cast_passthrough_medium)  { run_cast_passthrough(32, 32, 0x215ull); }
BT_PARITY_TEST(cast_fp32_to_bf16_small)  { run_cast_fp32_to_bf16(4, 8, 0x216ull); }
BT_PARITY_TEST(cast_fp32_to_bf16_medium) { run_cast_fp32_to_bf16(16, 64, 0x217ull); }
BT_PARITY_TEST(cast_bf16_to_fp32_small)  { run_cast_bf16_to_fp32(4, 8, 0x218ull); }
BT_PARITY_TEST(cast_bf16_to_fp32_medium) { run_cast_bf16_to_fp32(16, 64, 0x219ull); }

// ─── copy_d2d ──────────────────────────────────────────────────────────────
BT_PARITY_TEST(copy_d2d_partial)  { run_copy_d2d(16, 16, 3, 5, 6, 0x220ull); }
BT_PARITY_TEST(copy_d2d_full)     { run_copy_d2d(32, 32, 0, 0, 32, 0x221ull); }
BT_PARITY_TEST(copy_d2d_offset)   { run_copy_d2d(64, 40, 20, 8, 12, 0x222ull); }
BT_PARITY_TEST(copy_d2d_single)   { run_copy_d2d(8, 8, 7, 0, 1, 0x223ull); }

// ─── build_slot_mask ───────────────────────────────────────────────────────
BT_PARITY_TEST(build_slot_mask_contig)  { run_build_slot_mask(32, 0, 32, 1, 0x230ull); }
BT_PARITY_TEST(build_slot_mask_strided) { run_build_slot_mask(64, 2, 16, 3, 0x231ull); }
BT_PARITY_TEST(build_slot_mask_offset)  { run_build_slot_mask(128, 10, 24, 4, 0x232ull); }
BT_PARITY_TEST(build_slot_mask_single)  { run_build_slot_mask(8, 3, 1, 1, 0x233ull); }

int main() { return run_all("tensor-ops cpu/gpu parity"); }
