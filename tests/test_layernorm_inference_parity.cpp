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

BT_PARITY_TEST(causal_q0_L8)    { run_causal_mask(8, 0); }
BT_PARITY_TEST(causal_q3_L8)    { run_causal_mask(8, 3); }
BT_PARITY_TEST(causal_qlast_L8) { run_causal_mask(8, 7); }
BT_PARITY_TEST(causal_L1)       { run_causal_mask(1, 0); }
BT_PARITY_TEST(causal_L300)     { run_causal_mask(300, 150); }

int main() { return run_all("layernorm-inference cpu/gpu parity"); }
