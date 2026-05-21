// CPU↔GPU parity tests for brotensor::layernorm_forward / layernorm_backward.
//
// Both paths call the same device-neutral op; it dispatches to the CPU or
// CUDA backend by its operands' Device tag.

#include "parity_helpers.h"

#include <brotensor/ops.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_layernorm(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::vec(n), dY = Tensor::vec(n);
    fill_random(x, rng);
    fill_random(dY, rng);

    // Random gamma/beta (non-trivial).
    Tensor gamma = Tensor::vec(n), beta = Tensor::vec(n);
    for (int i = 0; i < n; ++i) {
        gamma[i] = 0.5f + 0.5f * rng.next_f01();   // [0.5, 1.0)
        beta[i]  = rng.next_unit() * 0.25f;
    }

    // Pre-fill dGamma/dBeta to validate accumulation.
    Tensor dGamma_init = Tensor::vec(n), dBeta_init = Tensor::vec(n);
    fill_random(dGamma_init, rng, 0.25f);
    fill_random(dBeta_init, rng, 0.25f);

    // CPU path — the device-neutral op dispatched on CPU tensors.
    Tensor y_cpu = Tensor::vec(n), xhat_cpu = Tensor::vec(n),
           dX_cpu = Tensor::vec(n);
    Tensor dGamma_cpu = dGamma_init, dBeta_cpu = dBeta_init;
    float mean_cpu = 0.0f, rstd_cpu = 0.0f;
    brotensor::layernorm_forward(
        x, gamma, beta, y_cpu, xhat_cpu, mean_cpu, rstd_cpu, 1e-5f);
    brotensor::layernorm_backward(
        dY, xhat_cpu, gamma, rstd_cpu, dX_cpu, dGamma_cpu, dBeta_cpu);

    // GPU path — the same ops on CUDA-resident tensors.
    Tensor gx = x.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gbeta = beta.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gy = Tensor::zeros_on(gpu_device(), n, 1);
    Tensor gxhat = Tensor::zeros_on(gpu_device(), n, 1);
    Tensor gdX = Tensor::zeros_on(gpu_device(), n, 1);
    Tensor gdGamma = dGamma_init.to(gpu_device());
    Tensor gdBeta = dBeta_init.to(gpu_device());
    float mean_gpu = 0.0f, rstd_gpu = 0.0f;
    brotensor::layernorm_forward(
        gx, ggamma, gbeta, gy, gxhat, mean_gpu, rstd_gpu, 1e-5f);
    brotensor::layernorm_backward(
        gdY, gxhat, ggamma, rstd_gpu, gdX, gdGamma, gdBeta);

    compare_tensors(y_cpu, download_to_host(gy), "layernorm.y");
    compare_tensors(dX_cpu, download_to_host(gdX), "layernorm.dX");
    compare_tensors(dGamma_cpu, download_to_host(gdGamma), "layernorm.dGamma");
    compare_tensors(dBeta_cpu, download_to_host(gdBeta), "layernorm.dBeta");
}

} // namespace

BT_PARITY_TEST(layernorm_n16)  { run_layernorm(16,  0x200ull); }
BT_PARITY_TEST(layernorm_n64)  { run_layernorm(64,  0x201ull); }
BT_PARITY_TEST(layernorm_n256) { run_layernorm(256, 0x202ull); }

// ─── BF16 backward parity ────────────────────────────────────────────────────
// BF16 is GPU-only. We run the FP32 CPU reference, convert inputs to BF16,
// run on CUDA, widen back to FP32, and compare with loose tolerances.
// layernorm_forward is FP32-only (no FP16/BF16 forward path), so only the
// backward path is tested here.
namespace {

void run_layernorm_bf16_bwd(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY_f32 = Tensor::vec(n), xhat_f32 = Tensor::vec(n);
    Tensor gamma_f32 = Tensor::vec(n), beta_f32 = Tensor::vec(n);
    Tensor dGamma_init_f32 = Tensor::vec(n), dBeta_init_f32 = Tensor::vec(n);
    fill_random(dY_f32, rng);
    fill_random(xhat_f32, rng);
    for (int i = 0; i < n; ++i) {
        gamma_f32[i] = 0.5f + 0.5f * rng.next_f01();
        beta_f32[i]  = rng.next_unit() * 0.25f;
    }
    fill_random(dGamma_init_f32, rng, 0.25f);
    fill_random(dBeta_init_f32,  rng, 0.25f);

    // CPU FP32 reference backward (pre-fill dGamma/dBeta).
    const float rstd = 0.5f + rng.next_f01() * 0.5f;  // synthetic rstd ~[0.5, 1.0)
    Tensor cpu_dX = Tensor::vec(n);
    Tensor cpu_dGamma = dGamma_init_f32, cpu_dBeta = dBeta_init_f32;
    brotensor::layernorm_backward(dY_f32, xhat_f32, gamma_f32, rstd,
                                  cpu_dX, cpu_dGamma, cpu_dBeta);

    // BF16 GPU path.
    Tensor gdY    = to_bf16_gpu(dY_f32);
    Tensor gxhat  = to_bf16_gpu(xhat_f32);
    Tensor ggamma = to_bf16_gpu(gamma_f32);
    Tensor gpu_dX = Tensor::zeros_on(gpu_device(), n, 1, brotensor::Dtype::BF16);
    Tensor gpu_dGamma = to_bf16_gpu(dGamma_init_f32);
    Tensor gpu_dBeta  = to_bf16_gpu(dBeta_init_f32);
    brotensor::layernorm_backward(gdY, gxhat, ggamma, rstd,
                                  gpu_dX, gpu_dGamma, gpu_dBeta);
    brotensor::sync_all();

    // Widen BF16 results back to FP32 for comparison.
    Tensor dX_h    = bf16_host_to_f32(download_to_host(gpu_dX));
    Tensor dGamma_h = bf16_host_to_f32(download_to_host(gpu_dGamma));
    Tensor dBeta_h  = bf16_host_to_f32(download_to_host(gpu_dBeta));

    compare_tensors(cpu_dX,     dX_h,     "layernorm_bf16_bwd.dX",     3e-2f, 3e-2f);
    compare_tensors(cpu_dGamma, dGamma_h, "layernorm_bf16_bwd.dGamma", 6e-2f, 6e-2f);
    compare_tensors(cpu_dBeta,  dBeta_h,  "layernorm_bf16_bwd.dBeta",  6e-2f, 6e-2f);
}

} // namespace

BT_PARITY_TEST(layernorm_bf16_bwd_n16)  { run_layernorm_bf16_bwd(16,  0x280ull); }
BT_PARITY_TEST(layernorm_bf16_bwd_n64)  { run_layernorm_bf16_bwd(64,  0x281ull); }
BT_PARITY_TEST(layernorm_bf16_bwd_n256) { run_layernorm_bf16_bwd(256, 0x282ull); }

int main() { return run_all("layernorm cpu/gpu parity"); }
