// CPU↔GPU parity tests for the newly-CPU-ported loss ops:
//   mse_vec_forward / mse_vec_backward / softmax_xent_fused.
//
// CHUNK 1. test_loss_parity.cpp compares the GPU against an *inline* CPU
// reference; this file runs the actual CPU backend ops and the GPU ops on
// the same inputs and compares directly.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_mse(int n, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor pred = Tensor::vec(n), target = Tensor::vec(n);
    fill_random(pred, rng);
    fill_random(target, rng);

    Tensor dPred_cpu = Tensor::vec(n);
    const float loss_cpu = brotensor::mse_vec_forward(pred, target);
    brotensor::mse_vec_backward(pred, target, dPred_cpu);

    Tensor gpred = pred.to(Device::CUDA);
    Tensor gtarget = target.to(Device::CUDA);
    Tensor gdPred = Tensor::zeros_on(Device::CUDA, n, 1);
    const float loss_gpu = brotensor::mse_vec_forward(gpred, gtarget);
    brotensor::mse_vec_backward(gpred, gtarget, gdPred);

    Tensor dPred_gpu = download_to_host(gdPred);
    BT_CHECK(std::fabs(loss_cpu - loss_gpu) < 1e-5f + 1e-4f * std::fabs(loss_cpu));
    compare_tensors(dPred_cpu, dPred_gpu, "mse.dPred");
}

void run_xent(int n, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor logits = Tensor::vec(n), target = Tensor::vec(n);
    fill_random(logits, rng);

    // Soft target normalised over valid entries.
    target.zero();
    float tsum = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && (*mask)[i] == 0.0f) continue;
        const float v = rng.next_f01() + 0.05f;
        target[i] = v;
        tsum += v;
    }
    if (tsum > 0.0f) for (int i = 0; i < n; ++i) target[i] /= tsum;

    Tensor probs_cpu = Tensor::vec(n), dLogits_cpu = Tensor::vec(n);
    const float loss_cpu = brotensor::softmax_xent_fused(
        logits, target, mask ? mask->data() : nullptr,
        probs_cpu, dLogits_cpu);

    Tensor glogits = logits.to(Device::CUDA);
    Tensor gtarget = target.to(Device::CUDA);
    Tensor gprobs = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor gdLogits = Tensor::zeros_on(Device::CUDA, n, 1);
    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);
    const float loss_gpu = brotensor::softmax_xent_fused(
        glogits, gtarget, d_mask, gprobs, gdLogits);

    Tensor probs_gpu = download_to_host(gprobs);
    Tensor dLogits_gpu = download_to_host(gdLogits);
    BT_CHECK(std::fabs(loss_cpu - loss_gpu) < 1e-5f + 1e-4f * std::fabs(loss_cpu));
    compare_tensors(probs_cpu, probs_gpu, "xent.probs");
    compare_tensors(dLogits_cpu, dLogits_gpu, "xent.dLogits");
}

std::vector<float> mask_all(int n)  { return std::vector<float>(n, 1.0f); }
std::vector<float> mask_half(int n) {
    std::vector<float> m(n, 0.0f);
    for (int i = 0; i < n; ++i) m[i] = (i < n / 2) ? 1.0f : 0.0f;
    if (n / 2 == 0) m[0] = 1.0f;
    return m;
}

} // namespace

BT_PARITY_TEST(fused_mse_n1)   { run_mse(1,   0x800ull); }
BT_PARITY_TEST(fused_mse_n8)   { run_mse(8,   0x801ull); }
BT_PARITY_TEST(fused_mse_n256) { run_mse(256, 0x802ull); }

BT_PARITY_TEST(fused_xent_n1)         { run_xent(1,   0x810ull, nullptr); }
BT_PARITY_TEST(fused_xent_n16)        { run_xent(16,  0x811ull, nullptr); }
BT_PARITY_TEST(fused_xent_n256)       { run_xent(256, 0x812ull, nullptr); }
BT_PARITY_TEST(fused_xent_mask_all)   { auto m = mask_all(32);  run_xent(32, 0x820ull, &m); }
BT_PARITY_TEST(fused_xent_mask_half)  { auto m = mask_half(64); run_xent(64, 0x821ull, &m); }

int main() { return run_all("loss-fused cpu/gpu parity"); }
