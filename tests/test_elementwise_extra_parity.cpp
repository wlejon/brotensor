// CPU↔GPU parity tests for brotensor::clamp and brotensor::mul_inplace.
//
// CHUNK 1. test_elementwise_parity.cpp already covers the activation ops;
// this file covers the two newly-CPU-ported elementwise ops.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_clamp(int r, int c, float lo, float hi, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor base = Tensor::mat(r, c);
    fill_random(base, rng, 3.0f);  // wide range so clamping actually bites

    Tensor cpu = base;             // deep copy (CPU)
    brotensor::clamp(cpu, lo, hi);

    Tensor gpu = base.to(gpu_device());
    brotensor::clamp(gpu, lo, hi);

    Tensor gpu_h = download_to_host(gpu);
    compare_tensors(cpu, gpu_h, "clamp");
}

void run_clamp_bf16(int r, int c, float lo, float hi, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor base = Tensor::mat(r, c);
    fill_random(base, rng, 3.0f);

    Tensor cpu = base;
    brotensor::clamp(cpu, lo, hi);  // FP32 CPU reference

    Tensor gpu = to_bf16_cuda(base);
    brotensor::clamp(gpu, lo, hi);

    compare_tensors(cpu, bf16_host_to_f32(download_to_host(gpu)), "clamp_bf16", 2e-2f, 2e-2f);
}

void run_mul(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a = Tensor::mat(r, c), b = Tensor::mat(r, c);
    fill_random(a, rng);
    fill_random(b, rng);

    Tensor cpu = a;                // deep copy (CPU)
    brotensor::mul_inplace(cpu, b);

    Tensor ga = a.to(gpu_device());
    Tensor gb = b.to(gpu_device());
    brotensor::mul_inplace(ga, gb);

    Tensor gpu_h = download_to_host(ga);
    compare_tensors(cpu, gpu_h, "mul_inplace");
}

void run_mul_bf16(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a = Tensor::mat(r, c), b = Tensor::mat(r, c);
    fill_random(a, rng);
    fill_random(b, rng);

    Tensor cpu = a;
    brotensor::mul_inplace(cpu, b);  // FP32 CPU reference

    Tensor ga = to_bf16_cuda(a);
    Tensor gb = to_bf16_cuda(b);
    brotensor::mul_inplace(ga, gb);

    compare_tensors(cpu, bf16_host_to_f32(download_to_host(ga)), "mul_inplace_bf16", 2e-2f, 2e-2f);
}

} // namespace

BT_PARITY_TEST(clamp_1x1)        { run_clamp(1, 1, -0.5f, 0.5f, 0x700ull); }
BT_PARITY_TEST(clamp_8x32)       { run_clamp(8, 32, -1.0f, 1.0f, 0x701ull); }
BT_PARITY_TEST(clamp_asym)       { run_clamp(16, 16, -0.25f, 0.75f, 0x702ull); }
BT_PARITY_TEST(clamp_relu_like)  { run_clamp(7, 13, 0.0f, 3.4e38f, 0x703ull); }

BT_PARITY_TEST(mul_1x1)          { run_mul(1, 1, 0x710ull); }
BT_PARITY_TEST(mul_8x32)         { run_mul(8, 32, 0x711ull); }
BT_PARITY_TEST(mul_vec)          { run_mul(64, 1, 0x712ull); }

// ─── BF16 parity tests ─────────────────────────────────────────────────────
BT_PARITY_TEST(clamp_bf16_8x32)   { run_clamp_bf16(8, 32, -1.0f, 1.0f, 0x720ull); }
BT_PARITY_TEST(clamp_bf16_16x16)  { run_clamp_bf16(16, 16, -0.25f, 0.75f, 0x721ull); }

BT_PARITY_TEST(mul_bf16_8x32)     { run_mul_bf16(8, 32, 0x730ull); }
BT_PARITY_TEST(mul_bf16_64x1)     { run_mul_bf16(64, 1, 0x731ull); }

int main() { return run_all("clamp/mul_inplace cpu/gpu parity"); }
