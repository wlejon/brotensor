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

    Tensor gX     = X.to(Device::CUDA);
    Tensor ggamma = gamma.to(Device::CUDA);
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

    Tensor gX     = X.to(Device::CUDA);
    Tensor ggamma = gamma.to(Device::CUDA);
    Tensor gdY    = dY.to(Device::CUDA);
    Tensor gpu_dX;
    Tensor gpu_dGamma = Tensor::zeros_on(Device::CUDA, D, 1);
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

    Tensor gX     = X.to(Device::CUDA);
    Tensor ggamma = gamma.to(Device::CUDA);
    Tensor gdY    = dY.to(Device::CUDA);
    Tensor gpu_dX;
    Tensor gpu_dGamma = dG0.to(Device::CUDA);  // same baseline on GPU
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

int main() { return run_all("rms_norm cpu/gpu parity"); }
