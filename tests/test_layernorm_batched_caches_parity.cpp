// CPU↔GPU parity tests for the new training-mode batched LayerNorm:
//   layernorm_forward_batched_with_caches  + corresponding backward.
//
// The CPU backend is FP32 only; the GPU backend covers FP32 / FP16 / BF16.
// FP16/BF16 are validated against the FP32 CPU reference with widened
// tolerances (BF16 has only 8 mantissa bits).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

// Helpers: FP32 host → FP16 host → GPU.
Tensor to_fp16_host(const Tensor& f32cpu) {
    Tensor out = Tensor::zeros_on(Device::CPU, f32cpu.rows, f32cpu.cols, Dtype::FP16);
    const float* s = f32cpu.host_f32();
    uint16_t* d = out.host_fp16_mut();
    for (int i = 0; i < f32cpu.size(); ++i) d[i] = brotensor::fp32_to_fp16_bits(s[i]);
    return out;
}
Tensor fp16_host_to_f32(const Tensor& fp16cpu) {
    Tensor out = Tensor::zeros_on(Device::CPU, fp16cpu.rows, fp16cpu.cols, Dtype::FP32);
    const uint16_t* s = fp16cpu.host_fp16();
    float* d = out.host_f32_mut();
    for (int i = 0; i < fp16cpu.size(); ++i) d[i] = brotensor::fp16_bits_to_fp32(s[i]);
    return out;
}
Tensor to_fp16_gpu(const Tensor& f32cpu) {
    return to_fp16_host(f32cpu).to(gpu_device());
}

// FP32 CPU↔GPU parity for forward + backward.
void run_fp32(int R, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(R, D);
    Tensor dY = Tensor::mat(R, D);
    fill_random(X, rng);
    fill_random(dY, rng);
    Tensor gamma = Tensor::vec(D), beta = Tensor::vec(D);
    for (int i = 0; i < D; ++i) {
        gamma[i] = 0.5f + 0.5f * rng.next_f01();
        beta[i]  = rng.next_unit() * 0.25f;
    }
    Tensor dGamma_init = Tensor::vec(D), dBeta_init = Tensor::vec(D);
    fill_random(dGamma_init, rng, 0.25f);
    fill_random(dBeta_init,  rng, 0.25f);

    // CPU reference.
    Tensor Y_cpu, Xhat_cpu, Mean_cpu, Rstd_cpu, dX_cpu = Tensor::mat(R, D);
    Tensor dGamma_cpu = dGamma_init, dBeta_cpu = dBeta_init;
    brotensor::layernorm_forward_batched_with_caches(
        X, gamma, beta, Y_cpu, Xhat_cpu, Mean_cpu, Rstd_cpu, 1e-5f);
    brotensor::layernorm_backward_batched_with_caches(
        dY, Xhat_cpu, gamma, Rstd_cpu, dX_cpu, dGamma_cpu, dBeta_cpu);

    // GPU FP32.
    Tensor gX = X.to(gpu_device());
    Tensor gg = gamma.to(gpu_device()), gb = beta.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gY, gXhat, gMean, gRstd;
    Tensor gdX = Tensor::zeros_on(gpu_device(), R, D);
    Tensor gdGamma = dGamma_init.to(gpu_device());
    Tensor gdBeta  = dBeta_init.to(gpu_device());
    brotensor::layernorm_forward_batched_with_caches(
        gX, gg, gb, gY, gXhat, gMean, gRstd, 1e-5f);
    brotensor::layernorm_backward_batched_with_caches(
        gdY, gXhat, gg, gRstd, gdX, gdGamma, gdBeta);

    compare_tensors(Y_cpu,    download_to_host(gY),    "ln_batched.fp32.Y");
    compare_tensors(Xhat_cpu, download_to_host(gXhat), "ln_batched.fp32.Xhat");
    compare_tensors(Mean_cpu, download_to_host(gMean), "ln_batched.fp32.Mean", 1e-4f, 1e-4f);
    compare_tensors(Rstd_cpu, download_to_host(gRstd), "ln_batched.fp32.Rstd", 1e-4f, 1e-4f);
    compare_tensors(dX_cpu,     download_to_host(gdX),     "ln_batched.fp32.dX");
    compare_tensors(dGamma_cpu, download_to_host(gdGamma), "ln_batched.fp32.dGamma");
    compare_tensors(dBeta_cpu,  download_to_host(gdBeta),  "ln_batched.fp32.dBeta");
}

// FP16 GPU forward+backward vs FP32 CPU reference, widened tolerances.
void run_fp16(int R, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(R, D);
    Tensor dY = Tensor::mat(R, D);
    fill_random(X, rng);
    fill_random(dY, rng);
    Tensor gamma = Tensor::vec(D), beta = Tensor::vec(D);
    for (int i = 0; i < D; ++i) {
        gamma[i] = 0.5f + 0.5f * rng.next_f01();
        beta[i]  = rng.next_unit() * 0.25f;
    }
    Tensor dGamma_init = Tensor::vec(D), dBeta_init = Tensor::vec(D);
    fill_random(dGamma_init, rng, 0.25f);
    fill_random(dBeta_init,  rng, 0.25f);

    Tensor Y_cpu, Xhat_cpu, Mean_cpu, Rstd_cpu, dX_cpu = Tensor::mat(R, D);
    Tensor dGamma_cpu = dGamma_init, dBeta_cpu = dBeta_init;
    brotensor::layernorm_forward_batched_with_caches(
        X, gamma, beta, Y_cpu, Xhat_cpu, Mean_cpu, Rstd_cpu, 1e-5f);
    brotensor::layernorm_backward_batched_with_caches(
        dY, Xhat_cpu, gamma, Rstd_cpu, dX_cpu, dGamma_cpu, dBeta_cpu);

    Tensor gX = to_fp16_gpu(X), gg = to_fp16_gpu(gamma), gb = to_fp16_gpu(beta);
    Tensor gdY = to_fp16_gpu(dY);
    Tensor gY, gXhat, gMean, gRstd;
    Tensor gdX = Tensor::zeros_on(gpu_device(), R, D, Dtype::FP16);
    Tensor gdGamma = to_fp16_gpu(dGamma_init), gdBeta = to_fp16_gpu(dBeta_init);
    brotensor::layernorm_forward_batched_with_caches(
        gX, gg, gb, gY, gXhat, gMean, gRstd, 1e-5f);
    brotensor::layernorm_backward_batched_with_caches(
        gdY, gXhat, gg, gRstd, gdX, gdGamma, gdBeta);
    brotensor::sync_all();

    compare_tensors(Y_cpu,    fp16_host_to_f32(download_to_host(gY)),    "ln_batched.fp16.Y",    1e-2f, 1e-2f);
    compare_tensors(Mean_cpu, download_to_host(gMean),                   "ln_batched.fp16.Mean", 1e-3f, 1e-3f);
    compare_tensors(Rstd_cpu, download_to_host(gRstd),                   "ln_batched.fp16.Rstd", 1e-3f, 1e-3f);
    compare_tensors(dX_cpu,     fp16_host_to_f32(download_to_host(gdX)),     "ln_batched.fp16.dX",     1e-2f, 1e-2f);
    compare_tensors(dGamma_cpu, fp16_host_to_f32(download_to_host(gdGamma)), "ln_batched.fp16.dGamma", 3e-2f, 3e-2f);
    compare_tensors(dBeta_cpu,  fp16_host_to_f32(download_to_host(gdBeta)),  "ln_batched.fp16.dBeta",  3e-2f, 3e-2f);
}

// BF16 GPU forward+backward vs FP32 CPU reference, BF16 tolerances.
void run_bf16(int R, int D, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(R, D);
    Tensor dY = Tensor::mat(R, D);
    fill_random(X, rng);
    fill_random(dY, rng);
    Tensor gamma = Tensor::vec(D), beta = Tensor::vec(D);
    for (int i = 0; i < D; ++i) {
        gamma[i] = 0.5f + 0.5f * rng.next_f01();
        beta[i]  = rng.next_unit() * 0.25f;
    }
    Tensor dGamma_init = Tensor::vec(D), dBeta_init = Tensor::vec(D);
    fill_random(dGamma_init, rng, 0.25f);
    fill_random(dBeta_init,  rng, 0.25f);

    Tensor Y_cpu, Xhat_cpu, Mean_cpu, Rstd_cpu, dX_cpu = Tensor::mat(R, D);
    Tensor dGamma_cpu = dGamma_init, dBeta_cpu = dBeta_init;
    brotensor::layernorm_forward_batched_with_caches(
        X, gamma, beta, Y_cpu, Xhat_cpu, Mean_cpu, Rstd_cpu, 1e-5f);
    brotensor::layernorm_backward_batched_with_caches(
        dY, Xhat_cpu, gamma, Rstd_cpu, dX_cpu, dGamma_cpu, dBeta_cpu);

    Tensor gX = to_bf16_gpu(X), gg = to_bf16_gpu(gamma), gb = to_bf16_gpu(beta);
    Tensor gdY = to_bf16_gpu(dY);
    Tensor gY, gXhat, gMean, gRstd;
    Tensor gdX = Tensor::zeros_on(gpu_device(), R, D, Dtype::BF16);
    Tensor gdGamma = to_bf16_gpu(dGamma_init), gdBeta = to_bf16_gpu(dBeta_init);
    brotensor::layernorm_forward_batched_with_caches(
        gX, gg, gb, gY, gXhat, gMean, gRstd, 1e-5f);
    brotensor::layernorm_backward_batched_with_caches(
        gdY, gXhat, gg, gRstd, gdX, gdGamma, gdBeta);
    brotensor::sync_all();

    compare_tensors(Y_cpu,    bf16_host_to_f32(download_to_host(gY)),    "ln_batched.bf16.Y",    3e-2f, 3e-2f);
    compare_tensors(Mean_cpu, download_to_host(gMean),                   "ln_batched.bf16.Mean", 1e-3f, 1e-3f);
    compare_tensors(Rstd_cpu, download_to_host(gRstd),                   "ln_batched.bf16.Rstd", 1e-3f, 1e-3f);
    compare_tensors(dX_cpu,     bf16_host_to_f32(download_to_host(gdX)),     "ln_batched.bf16.dX",     3e-2f, 3e-2f);
    compare_tensors(dGamma_cpu, bf16_host_to_f32(download_to_host(gdGamma)), "ln_batched.bf16.dGamma", 6e-2f, 6e-2f);
    compare_tensors(dBeta_cpu,  bf16_host_to_f32(download_to_host(gdBeta)),  "ln_batched.bf16.dBeta",  6e-2f, 6e-2f);
}

} // namespace

BT_PARITY_TEST(ln_batched_fp32_R4_D16)   { run_fp32(4, 16,    0x300ull); }
BT_PARITY_TEST(ln_batched_fp32_R8_D64)   { run_fp32(8, 64,    0x301ull); }
BT_PARITY_TEST(ln_batched_fp32_R32_D256) { run_fp32(32, 256,  0x302ull); }
BT_PARITY_TEST(ln_batched_fp32_R1_D17)   { run_fp32(1, 17,    0x303ull); }  // tiny + non-power-of-2

BT_PARITY_TEST(ln_batched_fp16_R8_D64)   { run_fp16(8, 64,    0x310ull); }
BT_PARITY_TEST(ln_batched_fp16_R32_D256) { run_fp16(32, 256,  0x311ull); }

BT_PARITY_TEST(ln_batched_bf16_R8_D64)   { run_bf16(8, 64,    0x320ull); }
BT_PARITY_TEST(ln_batched_bf16_R32_D256) { run_bf16(32, 256,  0x321ull); }

int main() { return run_all("layernorm batched-with-caches cpu/gpu parity"); }
