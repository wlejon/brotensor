// CPU<->GPU parity for the LoRA adapter primitive (ops/lora.h).
//
// The primitive is a pure composition of linear_forward/linear_backward and
// the elementwise gate, so this confirms the composition stays bit-close
// across backends for both forward (y, bottleneck) and backward (dA/dB/dG/dX),
// gated and ungated.

#include "parity_helpers.h"

#include <brotensor/ops/lora.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

using brotensor::Tensor;
using bt_parity::gpu_device;
using bt_parity::compare_tensors;
using bt_parity::download_to_host;
using bt_parity::SplitMix64;
using bt_parity::fill_random;

namespace {

// out != in != r so a transposed dispatch can't pass by luck.
constexpr int kOut = 6, kIn = 8, kR = 4;
constexpr float kScale = 0.5f;

// Build a fresh CPU tensor filled with deterministic noise.
Tensor cpu_rand(int r, int c, SplitMix64& rng, float scale) {
    Tensor t = Tensor::zeros_on(brotensor::Device::CPU, r, c);
    fill_random(t, rng, scale);
    return t;
}

void run(bool use_g) {
    SplitMix64 rng(use_g ? 0xDA7Au : 0x5EEDu);
    Tensor W = cpu_rand(kOut, kIn, rng, 0.5f);
    Tensor b = cpu_rand(kOut, 1,  rng, 0.3f);
    Tensor x = cpu_rand(kIn, 1,   rng, 0.7f);
    Tensor A = cpu_rand(kR, kIn,  rng, 0.5f);
    Tensor B = cpu_rand(kOut, kR, rng, 0.5f);
    Tensor g = cpu_rand(kR, 1,    rng, 0.8f);
    Tensor dY = cpu_rand(kOut, 1, rng, 0.4f);

    const auto dev = gpu_device();

    // ── CPU forward + backward ──
    const Tensor* gp_c = use_g ? &g : nullptr;
    Tensor yc, hc, hgc;
    brotensor::lora_forward(W, b, x, A, B, kScale, gp_c, yc, hc, hgc);
    Tensor dAc = Tensor::zeros_on(brotensor::Device::CPU, kR, kIn);
    Tensor dBc = Tensor::zeros_on(brotensor::Device::CPU, kOut, kR);
    Tensor dGc = Tensor::zeros_on(brotensor::Device::CPU, kR, 1);
    Tensor dXc = Tensor::zeros_on(brotensor::Device::CPU, kIn, 1);
    brotensor::lora_backward(W, x, A, B, kScale, gp_c, hc, hgc, dY,
                             dAc, dBc, use_g ? &dGc : nullptr, &dXc);

    // ── GPU forward + backward (same inputs, uploaded) ──
    Tensor Wg = W.to(dev), bg = b.to(dev), xg = x.to(dev),
           Ag = A.to(dev), Bg = B.to(dev), gg = g.to(dev), dYg = dY.to(dev);
    const Tensor* gp_g = use_g ? &gg : nullptr;
    Tensor yg, hg2, hgg;
    brotensor::lora_forward(Wg, bg, xg, Ag, Bg, kScale, gp_g, yg, hg2, hgg);
    Tensor dAg = Tensor::zeros_on(dev, kR, kIn);
    Tensor dBg = Tensor::zeros_on(dev, kOut, kR);
    Tensor dGg = Tensor::zeros_on(dev, kR, 1);
    Tensor dXg = Tensor::zeros_on(dev, kIn, 1);
    brotensor::lora_backward(Wg, xg, Ag, Bg, kScale, gp_g, hg2, hgg, dYg,
                             dAg, dBg, use_g ? &dGg : nullptr, &dXg);

    const char* t = use_g ? "gated" : "plain";
    compare_tensors(yc,  download_to_host(yg),  t);
    compare_tensors(hgc, download_to_host(hgg), t);
    compare_tensors(dAc, download_to_host(dAg), t);
    compare_tensors(dBc, download_to_host(dBg), t);
    if (use_g) compare_tensors(dGc, download_to_host(dGg), t);
    compare_tensors(dXc, download_to_host(dXg), t);
}

BT_PARITY_TEST(lora_gated)   { run(/*use_g=*/true); }
BT_PARITY_TEST(lora_ungated) { run(/*use_g=*/false); }

}  // namespace

int main() { return bt_parity::run_all("lora parity"); }
