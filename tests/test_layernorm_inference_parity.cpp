// CPU↔GPU parity tests for the newly-CPU-ported ops:
//   layernorm_forward_inference_batched / build_causal_mask_row.
//
// CHUNK 1. test_layernorm_parity.cpp covers the training layernorm; this
// file covers the inference-batched variant and the causal-mask helper.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_ln_inf(int R, int D, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, D);
    Tensor gamma = Tensor::vec(D), beta = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    fill_random(beta, rng);

    Tensor y_cpu;
    brotensor::layernorm_forward_inference_batched(X, gamma, beta, y_cpu, eps);

    Tensor gX = X.to(gpu_device());
    Tensor ggamma = gamma.to(gpu_device());
    Tensor gbeta = beta.to(gpu_device());
    Tensor gy;
    brotensor::layernorm_forward_inference_batched(gX, ggamma, gbeta, gy, eps);

    Tensor y_gpu = download_to_host(gy);
    compare_tensors(y_cpu, y_gpu, "layernorm_inference");
}

// Mixed precision (a ViT's FP32 residual stream feeding FP16 GEMMs, e.g.
// brovisionml's DINOv2): FP32 X into a pre-typed FP16 Y keeps Y FP16, and
// the FP16 branch output re-enters the FP32 residual with add_inplace.
// CUDA and Vulkan; the Metal backend has neither form yet.
void run_ln_inf_mixed(int R, int D, uint64_t seed) {
    if (gpu_device().type == brotensor::DeviceType::Metal) {
        std::printf("    (Metal: no FP32->FP16 layernorm / mixed add_inplace; skipped)\n");
        return;
    }
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(R, D);
    Tensor gamma = Tensor::vec(D), beta = Tensor::vec(D);
    fill_random(X, rng);
    fill_random(gamma, rng);
    fill_random(beta, rng);

    Tensor y_cpu;
    brotensor::layernorm_forward_inference_batched(X, gamma, beta, y_cpu, 1e-5f);

    Tensor gX = X.to(gpu_device());
    Tensor gy = Tensor::empty_on(gpu_device(), R, D, brotensor::Dtype::FP16);
    brotensor::layernorm_forward_inference_batched(gX, gamma.to(gpu_device()), beta.to(gpu_device()), gy,
                                                   1e-5f);
    if (gy.dtype != brotensor::Dtype::FP16) {
        std::printf("    [layernorm_inference FP32->FP16] Y lost its FP16 dtype\n");
        throw 0;
    }
    compare_tensors(y_cpu, fp16_host_to_f32(download_to_host(gy)), "layernorm_inference FP32->FP16",
                    1e-2f, 1e-2f);

    // residual += FP16 branch, on the GPU; the CPU reference adds the same
    // FP16-rounded values in FP32.
    Tensor& ref = X;  // X is done with: it becomes the CPU residual
    Tensor y16_host = fp16_host_to_f32(download_to_host(gy));
    brotensor::add_inplace(ref, y16_host);
    brotensor::add_inplace(gX, gy);
    if (gX.dtype != brotensor::Dtype::FP32) {
        std::printf("    [add_inplace FP32 += FP16] the residual lost its FP32 dtype\n");
        throw 0;
    }
    compare_tensors(ref, download_to_host(gX), "add_inplace FP32 += FP16");
}

void run_causal_mask(int L, int q) {
    Tensor m_cpu;
    brotensor::build_causal_mask_row(L, q, m_cpu);

    Tensor m_gpu;
    // Pin the output to CUDA so the op dispatches to the GPU backend; its
    // sole Tensor operand is the output mask.
    m_gpu = Tensor::zeros_on(gpu_device(), L, 1);
    brotensor::build_causal_mask_row(L, q, m_gpu);

    Tensor m_gpu_h = download_to_host(m_gpu);
    compare_tensors(m_cpu, m_gpu_h, "build_causal_mask_row", 0.0f, 0.0f);
}

} // namespace

BT_PARITY_TEST(ln_inf_1x1)    { run_ln_inf(1, 1, 1e-5f, 0xC00ull); }
BT_PARITY_TEST(ln_inf_4x16)   { run_ln_inf(4, 16, 1e-5f, 0xC01ull); }
BT_PARITY_TEST(ln_inf_8x256)  { run_ln_inf(8, 256, 1e-5f, 0xC02ull); }
BT_PARITY_TEST(ln_inf_3x257)  { run_ln_inf(3, 257, 1e-6f, 0xC03ull); }
BT_PARITY_TEST(ln_inf_mixed_8x384)  { run_ln_inf_mixed(8, 384, 0xC04ull); }
BT_PARITY_TEST(ln_inf_mixed_37x257) { run_ln_inf_mixed(37, 257, 0xC05ull); }

BT_PARITY_TEST(causal_q0_L8)    { run_causal_mask(8, 0); }
BT_PARITY_TEST(causal_q3_L8)    { run_causal_mask(8, 3); }
BT_PARITY_TEST(causal_qlast_L8) { run_causal_mask(8, 7); }
BT_PARITY_TEST(causal_L1)       { run_causal_mask(1, 0); }
BT_PARITY_TEST(causal_L300)     { run_causal_mask(300, 150); }

int main() { return run_all("layernorm-inference cpu/gpu parity"); }
