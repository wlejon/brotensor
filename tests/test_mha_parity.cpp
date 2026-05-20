// CPU↔GPU parity tests for brotensor::mha_forward / mha_backward.
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

void run_mha(int K, int D, int H, uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(K, D), dO = Tensor::mat(K, D);
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

    const int dh = D / H;
    const float* host_mask = mask ? mask->data() : nullptr;

    // CPU path — the device-neutral op dispatched on CPU tensors.
    Tensor Qh = Tensor::mat(H * K, dh), Kh = Tensor::mat(H * K, dh),
           Vh = Tensor::mat(H * K, dh);
    Tensor Attnh = Tensor::mat(H * K, K);
    Tensor Yconcat = Tensor::mat(K, D);
    Tensor O_cpu = Tensor::mat(K, D);
    brotensor::mha_forward(
        X, Wq, Wk, Wv, Wo, host_mask, H,
        Qh, Kh, Vh, Attnh, Yconcat, O_cpu);

    Tensor dX_cpu = Tensor::mat(K, D);
    Tensor dWq_cpu = dWq_init, dWk_cpu = dWk_init,
           dWv_cpu = dWv_init, dWo_cpu = dWo_init;
    brotensor::mha_backward(
        dO, X, Qh, Kh, Vh, Attnh, Yconcat,
        Wq, Wk, Wv, Wo, host_mask, H,
        dX_cpu, dWq_cpu, dWk_cpu, dWv_cpu, dWo_cpu);

    // GPU path — the same ops on CUDA-resident tensors.
    Tensor gX = X.to(gpu_device());
    Tensor gWq = Wq.to(gpu_device()), gWk = Wk.to(gpu_device()),
           gWv = Wv.to(gpu_device()), gWo = Wo.to(gpu_device());

    Tensor gQh = Tensor::zeros_on(gpu_device(), H * K, dh);
    Tensor gKh = Tensor::zeros_on(gpu_device(), H * K, dh);
    Tensor gVh = Tensor::zeros_on(gpu_device(), H * K, dh);
    Tensor gAttnh = Tensor::zeros_on(gpu_device(), H * K, K);
    Tensor gYconcat = Tensor::zeros_on(gpu_device(), K, D);
    Tensor gO = Tensor::zeros_on(gpu_device(), K, D);

    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);
    brotensor::mha_forward(
        gX, gWq, gWk, gWv, gWo, d_mask, H,
        gQh, gKh, gVh, gAttnh, gYconcat, gO);

    Tensor O_gpu = download_to_host(gO);

    Tensor gdO = dO.to(gpu_device());
    Tensor gdX = Tensor::zeros_on(gpu_device(), K, D);
    Tensor gdWq = dWq_init.to(gpu_device());
    Tensor gdWk = dWk_init.to(gpu_device());
    Tensor gdWv = dWv_init.to(gpu_device());
    Tensor gdWo = dWo_init.to(gpu_device());

    brotensor::mha_backward(
        gdO, gX, gQh, gKh, gVh, gAttnh, gYconcat,
        gWq, gWk, gWv, gWo, d_mask, H,
        gdX, gdWq, gdWk, gdWv, gdWo);

    // Slightly looser tolerance — multi-head attention chains many GEMMs and
    // softmaxes; FMA reordering between CPU and GPU pushes tighter tolerances
    // over.
    const float atol = 5e-5f, rtol = 5e-4f;
    compare_tensors(O_cpu,  O_gpu,  "mha.O", atol, rtol);
    compare_tensors(dX_cpu, download_to_host(gdX), "mha.dX", atol, rtol);
    compare_tensors(dWq_cpu, download_to_host(gdWq), "mha.dWq", atol, rtol);
    compare_tensors(dWk_cpu, download_to_host(gdWk), "mha.dWk", atol, rtol);
    compare_tensors(dWv_cpu, download_to_host(gdWv), "mha.dWv", atol, rtol);
    compare_tensors(dWo_cpu, download_to_host(gdWo), "mha.dWo", atol, rtol);

    if (mask) {
        for (int i = 0; i < K; ++i) {
            if ((*mask)[i] < 0.5f) {
                for (int c = 0; c < D; ++c) BT_CHECK(O_gpu(i, c) == 0.0f);
            }
        }
    }
}

std::vector<float> partial_mask(int n) {
    std::vector<float> m(n, 1.0f);
    for (int i = n / 2; i < n; ++i) m[i] = 0.0f;
    if (n >= 2) m[1] = 0.0f;
    return m;
}

} // namespace

// Unmasked: cover h ∈ {1, 2, 4}, D ∈ {32, 64}, K ∈ {4, 16}.
BT_PARITY_TEST(mha_h1_K4_D32)   { run_mha(4,  32, 1, 0x600ull, nullptr); }
BT_PARITY_TEST(mha_h2_K4_D32)   { run_mha(4,  32, 2, 0x601ull, nullptr); }
BT_PARITY_TEST(mha_h4_K4_D32)   { run_mha(4,  32, 4, 0x602ull, nullptr); }
BT_PARITY_TEST(mha_h2_K16_D32)  { run_mha(16, 32, 2, 0x603ull, nullptr); }
BT_PARITY_TEST(mha_h4_K16_D64)  { run_mha(16, 64, 4, 0x604ull, nullptr); }
BT_PARITY_TEST(mha_h1_K16_D64)  { run_mha(16, 64, 1, 0x605ull, nullptr); }

// Masked.
BT_PARITY_TEST(mha_h2_K4_D32_mask)  { auto m = partial_mask(4);  run_mha(4,  32, 2, 0x610ull, &m); }
BT_PARITY_TEST(mha_h4_K16_D64_mask) { auto m = partial_mask(16); run_mha(16, 64, 4, 0x611ull, &m); }

int main() { return run_all("mha cpu/gpu parity"); }
