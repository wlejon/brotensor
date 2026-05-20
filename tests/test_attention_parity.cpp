// CPU↔GPU parity tests for brotensor::attention_forward / attention_backward.
//
// Both paths call the same device-neutral op; it dispatches to the CPU or
// CUDA backend by its operands' Device tag.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_attention(int N, int D, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, D), dO = Tensor::mat(N, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    fill_random(Wq, rng, 0.5f);
    fill_random(Wk, rng, 0.5f);
    fill_random(Wv, rng, 0.5f);
    fill_random(Wo, rng, 0.5f);

    // Pre-fill dW* to validate accumulation behavior.
    Tensor dWq_init = Tensor::mat(D, D), dWk_init = Tensor::mat(D, D),
           dWv_init = Tensor::mat(D, D), dWo_init = Tensor::mat(D, D);
    fill_random(dWq_init, rng, 0.1f);
    fill_random(dWk_init, rng, 0.1f);
    fill_random(dWv_init, rng, 0.1f);
    fill_random(dWo_init, rng, 0.1f);

    const float* host_mask = mask ? mask->data() : nullptr;

    // CPU path — the device-neutral op dispatched on CPU tensors.
    Tensor Q = Tensor::mat(N, D), K = Tensor::mat(N, D), V = Tensor::mat(N, D);
    Tensor Attn = Tensor::mat(N, N), Ypre = Tensor::mat(N, D);
    Tensor O_cpu = Tensor::mat(N, D);
    brotensor::attention_forward(
        X, Wq, Wk, Wv, Wo, host_mask, Q, K, V, Attn, Ypre, O_cpu);

    Tensor dX_cpu = Tensor::mat(N, D);
    Tensor dWq_cpu = dWq_init, dWk_cpu = dWk_init,
           dWv_cpu = dWv_init, dWo_cpu = dWo_init;
    brotensor::attention_backward(
        dO, X, Q, K, V, Attn, Ypre, Wq, Wk, Wv, Wo, host_mask,
        dX_cpu, dWq_cpu, dWk_cpu, dWv_cpu, dWo_cpu);

    // GPU path — the same ops on CUDA-resident tensors.
    Tensor gX = X.to(Device::CUDA);
    Tensor gWq = Wq.to(Device::CUDA), gWk = Wk.to(Device::CUDA),
           gWv = Wv.to(Device::CUDA), gWo = Wo.to(Device::CUDA);

    Tensor gQ = Tensor::zeros_on(Device::CUDA, N, D);
    Tensor gK = Tensor::zeros_on(Device::CUDA, N, D);
    Tensor gV = Tensor::zeros_on(Device::CUDA, N, D);
    Tensor gAttn = Tensor::zeros_on(Device::CUDA, N, N);
    Tensor gYpre = Tensor::zeros_on(Device::CUDA, N, D);
    Tensor gO = Tensor::zeros_on(Device::CUDA, N, D);

    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);
    brotensor::attention_forward(
        gX, gWq, gWk, gWv, gWo, d_mask, gQ, gK, gV, gAttn, gYpre, gO);

    Tensor O_gpu = download_to_host(gO);

    Tensor gdO = dO.to(Device::CUDA);
    Tensor gdX = Tensor::zeros_on(Device::CUDA, N, D);
    Tensor gdWq = dWq_init.to(Device::CUDA);
    Tensor gdWk = dWk_init.to(Device::CUDA);
    Tensor gdWv = dWv_init.to(Device::CUDA);
    Tensor gdWo = dWo_init.to(Device::CUDA);

    brotensor::attention_backward(
        gdO, gX, gQ, gK, gV, gAttn, gYpre,
        gWq, gWk, gWv, gWo, d_mask,
        gdX, gdWq, gdWk, gdWv, gdWo);

    // Slightly looser tolerance — attention chains many GEMMs and softmaxes.
    const float atol = 5e-5f, rtol = 5e-4f;
    compare_tensors(O_cpu,  O_gpu,  "attn.O", atol, rtol);
    compare_tensors(dX_cpu, download_to_host(gdX), "attn.dX", atol, rtol);
    compare_tensors(dWq_cpu, download_to_host(gdWq), "attn.dWq", atol, rtol);
    compare_tensors(dWk_cpu, download_to_host(gdWk), "attn.dWk", atol, rtol);
    compare_tensors(dWv_cpu, download_to_host(gdWv), "attn.dWv", atol, rtol);
    compare_tensors(dWo_cpu, download_to_host(gdWo), "attn.dWo", atol, rtol);

    // Invalid rows in O should be zero on the GPU.
    if (mask) {
        for (int i = 0; i < N; ++i) {
            if ((*mask)[i] < 0.5f) {
                for (int c = 0; c < D; ++c) {
                    BT_CHECK(O_gpu(i, c) == 0.0f);
                }
            }
        }
    }
}

std::vector<float> full_mask(int n) { return std::vector<float>(n, 1.0f); }
std::vector<float> partial_mask(int n) {
    std::vector<float> m(n, 1.0f);
    // Mask out roughly the second half but keep the first valid.
    for (int i = n / 2; i < n; ++i) m[i] = 0.0f;
    if (n >= 2) m[1] = 0.0f;  // also poke a hole near the front
    return m;
}

} // namespace

// Unmasked.
BT_PARITY_TEST(attn_K2_D16)  { run_attention(2,  16, 0x300ull, nullptr); }
BT_PARITY_TEST(attn_K8_D16)  { run_attention(8,  16, 0x301ull, nullptr); }
BT_PARITY_TEST(attn_K8_D64)  { run_attention(8,  64, 0x302ull, nullptr); }
BT_PARITY_TEST(attn_K16_D64) { run_attention(16, 64, 0x303ull, nullptr); }

// Full-valid mask (should match unmasked).
BT_PARITY_TEST(attn_K8_D16_full_mask) {
    auto m = full_mask(8); run_attention(8, 16, 0x310ull, &m);
}

// Partial mask.
BT_PARITY_TEST(attn_K8_D16_partial_mask) {
    auto m = partial_mask(8); run_attention(8, 16, 0x320ull, &m);
}
BT_PARITY_TEST(attn_K16_D64_partial_mask) {
    auto m = partial_mask(16); run_attention(16, 64, 0x321ull, &m);
}

int main() { return run_all("attention cpu/gpu parity"); }
