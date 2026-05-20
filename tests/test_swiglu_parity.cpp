// CPU↔GPU parity tests for the brotensor SwiGLU op: forward + backward.
//
// CHUNK 2. X is (B, 2D); first half is the silu-gated value, second half is
// the linear half.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_fwd(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, 2 * D);
    fill_random(X, rng, 3.0f);

    Tensor cpu_Y;
    brotensor::swiglu_forward(X, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::swiglu_forward(gX, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "swiglu_fwd");
}

void run_bwd(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(B, 2 * D);
    Tensor dY = Tensor::mat(B, D);
    fill_random(X, rng, 3.0f);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::swiglu_backward(X, dY, cpu_dX);

    Tensor gX  = X.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::swiglu_backward(gX, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "swiglu_bwd");
}

void run_fwd_bf16(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(B, 2 * D);
    fill_random(X, rng, 3.0f);

    Tensor cpu_Y;
    brotensor::swiglu_forward(X, cpu_Y);  // FP32 CPU reference

    Tensor gX = to_bf16_cuda(X);
    Tensor gpu_Y;
    brotensor::swiglu_forward(gX, gpu_Y);

    compare_tensors(cpu_Y, bf16_host_to_f32(download_to_host(gpu_Y)), "swiglu_fwd_bf16", 2e-2f, 2e-2f);
}

void run_bwd_bf16(int B, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(B, 2 * D);
    Tensor dY = Tensor::mat(B, D);
    fill_random(X, rng, 3.0f);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::swiglu_backward(X, dY, cpu_dX);  // FP32 CPU reference

    Tensor gX  = to_bf16_cuda(X);
    Tensor gdY = to_bf16_cuda(dY);
    Tensor gpu_dX;
    brotensor::swiglu_backward(gX, gdY, gpu_dX);

    compare_tensors(cpu_dX, bf16_host_to_f32(download_to_host(gpu_dX)), "swiglu_bwd_bf16", 2e-2f, 2e-2f);
}

} // namespace

BT_PARITY_TEST(swiglu_fwd_1x2)   { run_fwd(1, 1, 0x4000ull); }
BT_PARITY_TEST(swiglu_fwd_8x64)  { run_fwd(8, 32, 0x4001ull); }
BT_PARITY_TEST(swiglu_fwd_5x14)  { run_fwd(5, 7, 0x4002ull); }
BT_PARITY_TEST(swiglu_bwd_1x2)   { run_bwd(1, 1, 0x4003ull); }
BT_PARITY_TEST(swiglu_bwd_8x64)  { run_bwd(8, 32, 0x4004ull); }
BT_PARITY_TEST(swiglu_bwd_5x14)  { run_bwd(5, 7, 0x4005ull); }

// ─── BF16 parity tests ─────────────────────────────────────────────────────
BT_PARITY_TEST(swiglu_fwd_bf16_8x64)  { run_fwd_bf16(8, 32, 0x4100ull); }
BT_PARITY_TEST(swiglu_fwd_bf16_5x14)  { run_fwd_bf16(5, 7, 0x4101ull); }
BT_PARITY_TEST(swiglu_bwd_bf16_8x64)  { run_bwd_bf16(8, 32, 0x4102ull); }
BT_PARITY_TEST(swiglu_bwd_bf16_5x14)  { run_bwd_bf16(5, 7, 0x4103ull); }

int main() { return run_all("swiglu cpu/gpu parity"); }
