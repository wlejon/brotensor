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

    Tensor gpu = base.to(Device::CUDA);
    brotensor::clamp(gpu, lo, hi);

    Tensor gpu_h = download_to_host(gpu);
    compare_tensors(cpu, gpu_h, "clamp");
}

void run_mul(int r, int c, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a = Tensor::mat(r, c), b = Tensor::mat(r, c);
    fill_random(a, rng);
    fill_random(b, rng);

    Tensor cpu = a;                // deep copy (CPU)
    brotensor::mul_inplace(cpu, b);

    Tensor ga = a.to(Device::CUDA);
    Tensor gb = b.to(Device::CUDA);
    brotensor::mul_inplace(ga, gb);

    Tensor gpu_h = download_to_host(ga);
    compare_tensors(cpu, gpu_h, "mul_inplace");
}

} // namespace

BT_PARITY_TEST(clamp_1x1)        { run_clamp(1, 1, -0.5f, 0.5f, 0x700ull); }
BT_PARITY_TEST(clamp_8x32)       { run_clamp(8, 32, -1.0f, 1.0f, 0x701ull); }
BT_PARITY_TEST(clamp_asym)       { run_clamp(16, 16, -0.25f, 0.75f, 0x702ull); }
BT_PARITY_TEST(clamp_relu_like)  { run_clamp(7, 13, 0.0f, 3.4e38f, 0x703ull); }

BT_PARITY_TEST(mul_1x1)          { run_mul(1, 1, 0x710ull); }
BT_PARITY_TEST(mul_8x32)         { run_mul(8, 32, 0x711ull); }
BT_PARITY_TEST(mul_vec)          { run_mul(64, 1, 0x712ull); }

int main() { return run_all("clamp/mul_inplace cpu/gpu parity"); }
