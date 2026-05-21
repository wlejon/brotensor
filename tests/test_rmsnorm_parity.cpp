// CPU↔GPU parity tests for brotensor::rms_norm_forward and rms_norm_backward.
//
// CHUNK 2. Per-row RMSNorm over (B, D). Backward: dX is OVERWRITTEN, dGamma
// ACCUMULATES (+=) across the batch — the accumulation test pre-fills dGamma
// with a non-zero baseline to verify that contract holds on both backends.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_fwd(int B, int D, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X     = Tensor::mat(B, D);
    Tensor gamma = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);

    Tensor cpu_Y;
    brotensor::rms_norm_forward(X, gamma, eps, cpu_Y);

    Tensor gX     = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gpu_Y;
    brotensor::rms_norm_forward(gX, ggamma, eps, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "rms_norm_fwd", 1e-4f, 1e-3f);
}

void run_bwd(int B, int D, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X     = Tensor::mat(B, D);
    Tensor gamma = Tensor::vec(D);
    Tensor dY    = Tensor::mat(B, D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    fill_random(dY, rng);

    Tensor cpu_dX;
    Tensor cpu_dGamma = Tensor::vec(D);  // zero-initialised
    brotensor::rms_norm_backward(X, gamma, dY, eps, cpu_dX, cpu_dGamma);

    Tensor gX     = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gdY    = dY.to(gpu_device());
    Tensor gpu_dX;
    Tensor gpu_dGamma = Tensor::zeros_on(gpu_device(), D, 1);
    brotensor::rms_norm_backward(gX, ggamma, gdY, eps, gpu_dX, gpu_dGamma);

    compare_tensors(cpu_dX, download_to_host(gpu_dX), "rms_norm_bwd_dX", 1e-4f, 1e-3f);
    compare_tensors(cpu_dGamma, download_to_host(gpu_dGamma),
                    "rms_norm_bwd_dGamma", 1e-4f, 1e-3f);
}

// dGamma carries a pre-existing baseline — verifies the += accumulation.
void run_bwd_accum(int B, int D, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X     = Tensor::mat(B, D);
    Tensor gamma = Tensor::vec(D);
    Tensor dY    = Tensor::mat(B, D);
    Tensor dG0   = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    fill_random(dY, rng);
    fill_random(dG0, rng);

    Tensor cpu_dX;
    Tensor cpu_dGamma = dG0;            // deep copy of baseline (CPU)
    brotensor::rms_norm_backward(X, gamma, dY, eps, cpu_dX, cpu_dGamma);

    Tensor gX     = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gdY    = dY.to(gpu_device());
    Tensor gpu_dX;
    Tensor gpu_dGamma = dG0.to(gpu_device());  // same baseline on GPU
    brotensor::rms_norm_backward(gX, ggamma, gdY, eps, gpu_dX, gpu_dGamma);

    compare_tensors(cpu_dX, download_to_host(gpu_dX),
                    "rms_norm_bwd_accum_dX", 1e-4f, 1e-3f);
    compare_tensors(cpu_dGamma, download_to_host(gpu_dGamma),
                    "rms_norm_bwd_accum_dGamma", 1e-4f, 1e-3f);
}

} // namespace

// ─── forward ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(rms_norm_fwd_1x1)   { run_fwd(1, 1, 1e-5f, 0x7000ull); }
BT_PARITY_TEST(rms_norm_fwd_8x32)  { run_fwd(8, 32, 1e-5f, 0x7001ull); }
BT_PARITY_TEST(rms_norm_fwd_5x7)   { run_fwd(5, 7, 1e-6f, 0x7002ull); }
BT_PARITY_TEST(rms_norm_fwd_wide)  { run_fwd(3, 257, 1e-5f, 0x7003ull); }

// ─── backward ──────────────────────────────────────────────────────────────
BT_PARITY_TEST(rms_norm_bwd_1x1)   { run_bwd(1, 1, 1e-5f, 0x7010ull); }
BT_PARITY_TEST(rms_norm_bwd_8x32)  { run_bwd(8, 32, 1e-5f, 0x7011ull); }
BT_PARITY_TEST(rms_norm_bwd_5x7)   { run_bwd(5, 7, 1e-6f, 0x7012ull); }
BT_PARITY_TEST(rms_norm_bwd_accum) { run_bwd_accum(6, 11, 1e-5f, 0x7013ull); }

// ─── BF16 parity ─────────────────────────────────────────────────────────────
// BF16 is GPU-only. Round FP32 inputs to BF16, run on CUDA, widen back and
// compare against the FP32 CPU reference with loose tolerances.

namespace {

void run_fwd_bf16(int B, int D, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_f32     = Tensor::mat(B, D);
    Tensor gamma_f32 = Tensor::vec(D);
    fill_random(X_f32, rng);
    fill_random(gamma_f32, rng);

    // CPU FP32 reference.
    Tensor cpu_Y;
    brotensor::rms_norm_forward(X_f32, gamma_f32, eps, cpu_Y);

    // BF16 GPU path.
    Tensor gX     = to_bf16_gpu(X_f32);
    Tensor ggamma = to_bf16_gpu(gamma_f32);
    Tensor gpu_Y;
    brotensor::rms_norm_forward(gX, ggamma, eps, gpu_Y);
    brotensor::sync_all();

    Tensor Y_h = bf16_host_to_f32(download_to_host(gpu_Y));
    compare_tensors(cpu_Y, Y_h, "rms_norm_bf16_fwd", 3e-2f, 3e-2f);
}

void run_bwd_bf16(int B, int D, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X_f32     = Tensor::mat(B, D);
    Tensor gamma_f32 = Tensor::vec(D);
    Tensor dY_f32    = Tensor::mat(B, D);
    fill_random(X_f32, rng);
    fill_random(gamma_f32, rng);
    fill_random(dY_f32, rng);

    // CPU FP32 reference (zero-init dGamma).
    Tensor cpu_dX;
    Tensor cpu_dGamma = Tensor::vec(D);
    brotensor::rms_norm_backward(X_f32, gamma_f32, dY_f32, eps, cpu_dX, cpu_dGamma);

    // BF16 GPU path.
    Tensor gX     = to_bf16_gpu(X_f32);
    Tensor ggamma = to_bf16_gpu(gamma_f32);
    Tensor gdY    = to_bf16_gpu(dY_f32);
    Tensor gpu_dX;
    Tensor gpu_dGamma = Tensor::zeros_on(gpu_device(), D, 1, brotensor::Dtype::BF16);
    brotensor::rms_norm_backward(gX, ggamma, gdY, eps, gpu_dX, gpu_dGamma);
    brotensor::sync_all();

    Tensor dX_h     = bf16_host_to_f32(download_to_host(gpu_dX));
    Tensor dGamma_h = bf16_host_to_f32(download_to_host(gpu_dGamma));

    compare_tensors(cpu_dX,     dX_h,     "rms_norm_bf16_bwd_dX",     3e-2f, 3e-2f);
    compare_tensors(cpu_dGamma, dGamma_h, "rms_norm_bf16_bwd_dGamma", 6e-2f, 6e-2f);
}

} // namespace (BF16 helpers)

BT_PARITY_TEST(rms_norm_bf16_fwd_8x32)  { run_fwd_bf16(8, 32, 1e-5f,  0x7080ull); }
BT_PARITY_TEST(rms_norm_bf16_fwd_5x7)   { run_fwd_bf16(5, 7, 1e-6f,   0x7081ull); }
BT_PARITY_TEST(rms_norm_bf16_fwd_wide)  { run_fwd_bf16(3, 257, 1e-5f, 0x7082ull); }

BT_PARITY_TEST(rms_norm_bf16_bwd_8x32)  { run_bwd_bf16(8, 32, 1e-5f,  0x7090ull); }
BT_PARITY_TEST(rms_norm_bf16_bwd_5x7)   { run_bwd_bf16(5, 7, 1e-6f,   0x7091ull); }
BT_PARITY_TEST(rms_norm_bf16_bwd_wide)  { run_bwd_bf16(3, 257, 1e-5f, 0x7092ull); }

int main() { return run_all("rms_norm cpu/gpu parity"); }
