// CPU<->GPU parity for self_attention_forward_train / self_attention_backward.
//
// Both ops are FP32 on both backends (the CUDA path delegates to mha_*; the
// CPU path likewise). Straightforward FP32<->FP32 parity, moderate tolerance.
//
// The test exercises the forward + backward pairing: run forward_train to get
// the per-head caches (Qh, Kh, Vh, Attnh, Yconcat, O), then feed them to
// backward. Gradient buffers dWq/dWk/dWv/dWo ACCUMULATE — they are pre-filled
// with a non-zero baseline (identical on CPU and GPU) so parity also verifies
// the accumulation contract. dX is overwritten.

#include "parity_helpers.h"

#include <brotensor/ops.h>

#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

void run_self_attn(int L, int D, int num_heads, uint64_t seed,
                   const std::vector<float>* mask) {
    SplitMix64 rng(seed);

    Tensor X  = Tensor::mat(L, D);
    Tensor dO = Tensor::mat(L, D);
    fill_random(X, rng);
    fill_random(dO, rng);

    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    fill_random(Wq, rng, 0.5f);
    fill_random(Wk, rng, 0.5f);
    fill_random(Wv, rng, 0.5f);
    fill_random(Wo, rng, 0.5f);

    // Non-zero gradient baselines to verify accumulation.
    Tensor dWq_init = Tensor::mat(D, D), dWk_init = Tensor::mat(D, D),
           dWv_init = Tensor::mat(D, D), dWo_init = Tensor::mat(D, D);
    fill_random(dWq_init, rng, 0.1f);
    fill_random(dWk_init, rng, 0.1f);
    fill_random(dWv_init, rng, 0.1f);
    fill_random(dWo_init, rng, 0.1f);

    const float* host_mask = mask ? mask->data() : nullptr;

    // CPU path.
    Tensor Qh_c, Kh_c, Vh_c, Attnh_c, Yconcat_c, O_c;
    brotensor::self_attention_forward_train(
        X, Wq, Wk, Wv, Wo, host_mask, num_heads,
        Qh_c, Kh_c, Vh_c, Attnh_c, Yconcat_c, O_c);

    Tensor dX_c = Tensor::mat(L, D);
    Tensor dWq_c = dWq_init, dWk_c = dWk_init,
           dWv_c = dWv_init, dWo_c = dWo_init;
    brotensor::self_attention_backward(
        dO, X, Qh_c, Kh_c, Vh_c, Attnh_c, Yconcat_c,
        Wq, Wk, Wv, Wo, host_mask, num_heads,
        dX_c, dWq_c, dWk_c, dWv_c, dWo_c);

    // GPU path.
    Tensor gX  = X.to(Device::CUDA);
    Tensor gWq = Wq.to(Device::CUDA), gWk = Wk.to(Device::CUDA),
           gWv = Wv.to(Device::CUDA), gWo = Wo.to(Device::CUDA);

    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);

    Tensor gQh, gKh, gVh, gAttnh, gYconcat, gO;
    brotensor::self_attention_forward_train(
        gX, gWq, gWk, gWv, gWo, d_mask, num_heads,
        gQh, gKh, gVh, gAttnh, gYconcat, gO);

    Tensor gdO  = dO.to(Device::CUDA);
    Tensor gdX  = Tensor::zeros_on(Device::CUDA, L, D);
    Tensor gdWq = dWq_init.to(Device::CUDA);
    Tensor gdWk = dWk_init.to(Device::CUDA);
    Tensor gdWv = dWv_init.to(Device::CUDA);
    Tensor gdWo = dWo_init.to(Device::CUDA);
    brotensor::self_attention_backward(
        gdO, gX, gQh, gKh, gVh, gAttnh, gYconcat,
        gWq, gWk, gWv, gWo, d_mask, num_heads,
        gdX, gdWq, gdWk, gdWv, gdWo);

    const float atol = 1e-4f, rtol = 1e-3f;
    compare_tensors(O_c,   download_to_host(gO),       "self_attn.O",   atol, rtol);
    compare_tensors(dX_c,  download_to_host(gdX),      "self_attn.dX",  atol, rtol);
    compare_tensors(dWq_c, download_to_host(gdWq),     "self_attn.dWq", atol, rtol);
    compare_tensors(dWk_c, download_to_host(gdWk),     "self_attn.dWk", atol, rtol);
    compare_tensors(dWv_c, download_to_host(gdWv),     "self_attn.dWv", atol, rtol);
    compare_tensors(dWo_c, download_to_host(gdWo),     "self_attn.dWo", atol, rtol);
}

std::vector<float> partial_mask(int n) {
    std::vector<float> m(n, 1.0f);
    for (int i = n / 2; i < n; ++i) m[i] = 0.0f;
    if (n >= 2) m[1] = 0.0f;
    return m;
}

} // namespace

BT_PARITY_TEST(self_attn_L4_D16_h1)  { run_self_attn(4,  16, 1, 0x500ull, nullptr); }
BT_PARITY_TEST(self_attn_L8_D16_h4)  { run_self_attn(8,  16, 4, 0x501ull, nullptr); }
BT_PARITY_TEST(self_attn_L8_D32_h8)  { run_self_attn(8,  32, 8, 0x502ull, nullptr); }
BT_PARITY_TEST(self_attn_L16_D64_h8) { run_self_attn(16, 64, 8, 0x503ull, nullptr); }
BT_PARITY_TEST(self_attn_L6_D48_h6)  { run_self_attn(6,  48, 6, 0x504ull, nullptr); }

BT_PARITY_TEST(self_attn_L8_D32_h4_mask) {
    auto m = partial_mask(8); run_self_attn(8, 32, 4, 0x510ull, &m);
}
BT_PARITY_TEST(self_attn_L16_D64_h8_mask) {
    auto m = partial_mask(16); run_self_attn(16, 64, 8, 0x511ull, &m);
}

int main() { return run_all("self_attention cpu/gpu parity"); }
