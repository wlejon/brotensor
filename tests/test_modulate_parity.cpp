// CPU↔GPU parity tests for brotensor::modulate and brotensor::broadcast_mul.
//
// AdaLN modulation (DiT / SD3 / Flux):
//   modulate:      Y[l,d] = X[l,d] * (1 + scale[d]) + shift[d]
//   broadcast_mul: Y[l,d] = X[l,d] * v[d]
// scale / shift / v are length-D vectors broadcast across token rows.
//
// FP32: tight tolerance. BF16: GPU-only, round inputs to BF16, run on CUDA,
// widen back and compare against the FP32 CPU reference with loose tolerance.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_modulate(int L, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X     = Tensor::mat(L, D);
    Tensor scale = Tensor::vec(D);
    Tensor shift = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(scale, rng);
    fill_random(shift, rng);

    Tensor cpu_Y;
    brotensor::modulate(X, scale, shift, cpu_Y);

    Tensor gX     = X.to(gpu_device());
    Tensor gscale = scale.to(gpu_device());
    Tensor gshift = shift.to(gpu_device());
    Tensor gpu_Y;
    brotensor::modulate(gX, gscale, gshift, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "modulate", 1e-4f, 1e-3f);
}

void run_broadcast_mul(int L, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, D);
    Tensor v = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(v, rng);

    Tensor cpu_Y;
    brotensor::broadcast_mul(X, v, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gv = v.to(gpu_device());
    Tensor gpu_Y;
    brotensor::broadcast_mul(gX, gv, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "broadcast_mul", 1e-4f, 1e-3f);
}

void run_modulate_bf16(int L, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X     = Tensor::mat(L, D);
    Tensor scale = Tensor::vec(D);
    Tensor shift = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(scale, rng);
    fill_random(shift, rng);

    Tensor cpu_Y;
    brotensor::modulate(X, scale, shift, cpu_Y);

    Tensor gX     = to_bf16_gpu(X);
    Tensor gscale = to_bf16_gpu(scale);
    Tensor gshift = to_bf16_gpu(shift);
    Tensor gpu_Y;
    brotensor::modulate(gX, gscale, gshift, gpu_Y);
    brotensor::sync_all();

    compare_tensors(cpu_Y, bf16_host_to_f32(download_to_host(gpu_Y)),
                    "modulate_bf16", 3e-2f, 3e-2f);
}

void run_broadcast_mul_bf16(int L, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, D);
    Tensor v = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(v, rng);

    Tensor cpu_Y;
    brotensor::broadcast_mul(X, v, cpu_Y);

    Tensor gX = to_bf16_gpu(X);
    Tensor gv = to_bf16_gpu(v);
    Tensor gpu_Y;
    brotensor::broadcast_mul(gX, gv, gpu_Y);
    brotensor::sync_all();

    compare_tensors(cpu_Y, bf16_host_to_f32(download_to_host(gpu_Y)),
                    "broadcast_mul_bf16", 3e-2f, 3e-2f);
}

} // namespace

// ─── FP32 parity ───────────────────────────────────────────────────────────
BT_PARITY_TEST(modulate_1x1)        { run_modulate(1, 1, 0xAD00ull); }
BT_PARITY_TEST(modulate_8x32)       { run_modulate(8, 32, 0xAD01ull); }
BT_PARITY_TEST(modulate_5x7)        { run_modulate(5, 7, 0xAD02ull); }
BT_PARITY_TEST(modulate_wide)       { run_modulate(3, 1153, 0xAD03ull); }
BT_PARITY_TEST(broadcast_mul_8x32)  { run_broadcast_mul(8, 32, 0xAD10ull); }
BT_PARITY_TEST(broadcast_mul_5x7)   { run_broadcast_mul(5, 7, 0xAD11ull); }
BT_PARITY_TEST(broadcast_mul_wide)  { run_broadcast_mul(3, 1153, 0xAD12ull); }

// ─── BF16 parity ───────────────────────────────────────────────────────────
BT_PARITY_TEST(modulate_bf16_8x32)      { run_modulate_bf16(8, 32, 0xAD80ull); }
BT_PARITY_TEST(modulate_bf16_wide)      { run_modulate_bf16(4, 257, 0xAD81ull); }
BT_PARITY_TEST(broadcast_mul_bf16_8x32) { run_broadcast_mul_bf16(8, 32, 0xAD90ull); }
BT_PARITY_TEST(broadcast_mul_bf16_wide) { run_broadcast_mul_bf16(4, 257, 0xAD91ull); }

int main() { return run_all("modulate cpu/gpu parity"); }
