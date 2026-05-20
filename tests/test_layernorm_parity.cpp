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
    Tensor gx = x.to(Device::CUDA);
    Tensor ggamma = gamma.to(Device::CUDA);
    Tensor gbeta = beta.to(Device::CUDA);
    Tensor gdY = dY.to(Device::CUDA);
    Tensor gy = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gxhat = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gdX = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gdGamma = dGamma_init.to(Device::CUDA);
    Tensor gdBeta = dBeta_init.to(Device::CUDA);
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

int main() { return run_all("layernorm cpu/gpu parity"); }
