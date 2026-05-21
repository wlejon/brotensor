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
    Tensor gX  = X.to(gpu_device());
    Tensor gWq = Wq.to(gpu_device()), gWk = Wk.to(gpu_device()),
           gWv = Wv.to(gpu_device()), gWo = Wo.to(gpu_device());

    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);

    Tensor gQh, gKh, gVh, gAttnh, gYconcat, gO;
    brotensor::self_attention_forward_train(
        gX, gWq, gWk, gWv, gWo, d_mask, num_heads,
        gQh, gKh, gVh, gAttnh, gYconcat, gO);

    Tensor gdO  = dO.to(gpu_device());
    Tensor gdX  = Tensor::zeros_on(gpu_device(), L, D);
    Tensor gdWq = dWq_init.to(gpu_device());
    Tensor gdWk = dWk_init.to(gpu_device());
    Tensor gdWv = dWv_init.to(gpu_device());
    Tensor gdWo = dWo_init.to(gpu_device());
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

// ─── BF16 self_attention_forward parity (GPU-only) ────────────────────────
//
// self_attention_forward_train / _backward are FP32-only (mha path), so they
// have no BF16 twin. self_attention_forward, however, is dtype-dispatched:
// its BF16 path delegates to the flash route. BF16 is GPU-only, so we compare
// BF16-on-CUDA against the FP32 flash composition (flash_attention_qkvo_forward
// drives the CPU reference — same keys-only masking the flash path uses).

inline float qbf(float v) {
    return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(v));
}

Tensor make_qbf_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = qbf(rng.next_unit() * scale);
    return t;
}

Tensor to_bf16_gpu_t(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_bf16_bits(cpu[i]);
    return Tensor::from_host_bf16_on(gpu_device(), h.data(),
                                     cpu.rows, cpu.cols);
}

Tensor bf16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_bf16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::bf16_bits_to_fp32(h[i]);
    return out;
}

void run_self_attn_forward_bf16(int L, int D, int num_heads, uint64_t seed,
                                const std::vector<float>* mask) {
    SplitMix64 rng(seed);
    Tensor X  = make_qbf_cpu(L, D, rng, 0.3f);
    Tensor Wq = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wk = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wv = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wo = make_qbf_cpu(D, D, rng, 0.3f);

    const float* host_mask = mask ? mask->data() : nullptr;

    Tensor O_c;
    brotensor::flash_attention_qkvo_forward(
        X, nullptr, Wq, nullptr, Wk, nullptr, Wv, nullptr, Wo, nullptr,
        host_mask, num_heads, /*causal=*/false, O_c);

    Tensor gX  = to_bf16_gpu_t(X);
    Tensor gWq = to_bf16_gpu_t(Wq), gWk = to_bf16_gpu_t(Wk),
           gWv = to_bf16_gpu_t(Wv), gWo = to_bf16_gpu_t(Wo);
    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);

    Tensor gO;
    brotensor::self_attention_forward(gX, gWq, gWk, gWv, gWo, d_mask,
                                      num_heads, gO);

    compare_tensors(O_c, bf16_cuda_to_cpu(gO), "self_attn_fwd.O.bf16",
                    8e-2f, 8e-2f);
}

// ─── self_attention_bias_forward parity ───────────────────────────────────
//
// Materialised multi-head self-attention with an optional per-head (L,L)
// additive pre-softmax bias. FP32 CPU<->CUDA parity (with/without bias, mask,
// and at scale = 1/sqrt(dh) and scale = 1.0 for the T5 path); plus a cross-
// check that the no-bias scale=1/sqrt(dh) case matches self_attention_forward
// _train; plus BF16-on-CUDA vs the FP32 CPU reference.

void run_sab(int L, int D, int num_heads, float scale, bool with_bias,
             const std::vector<float>* mask, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(L, D);
    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    fill_random(X, rng, 0.5f);
    fill_random(Wq, rng, 0.3f); fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f); fill_random(Wo, rng, 0.3f);

    Tensor bias;
    if (with_bias) {
        bias = Tensor::mat(num_heads * L, L);
        fill_random(bias, rng, 0.5f);
    }
    const Tensor* bias_cpu = with_bias ? &bias : nullptr;
    const float* host_mask = mask ? mask->data() : nullptr;

    Tensor O_c;
    brotensor::self_attention_bias_forward(X, Wq, Wk, Wv, Wo, host_mask,
                                           bias_cpu, num_heads, scale, O_c);

    Tensor gX  = X.to(gpu_device());
    Tensor gWq = Wq.to(gpu_device()), gWk = Wk.to(gpu_device()),
           gWv = Wv.to(gpu_device()), gWo = Wo.to(gpu_device());
    Tensor gbias;
    const Tensor* bias_gpu = nullptr;
    if (with_bias) { gbias = bias.to(gpu_device()); bias_gpu = &gbias; }
    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);

    Tensor gO;
    brotensor::self_attention_bias_forward(gX, gWq, gWk, gWv, gWo, d_mask,
                                           bias_gpu, num_heads, scale, gO);

    compare_tensors(O_c, download_to_host(gO), "sab.O", 1e-4f, 1e-3f);
}

// No-bias, scale = 1/sqrt(dh) must reproduce ordinary self-attention.
void run_sab_vs_mha(int L, int D, int num_heads, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(L, D);
    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, D),
           Wv = Tensor::mat(D, D), Wo = Tensor::mat(D, D);
    fill_random(X, rng, 0.5f);
    fill_random(Wq, rng, 0.3f); fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f); fill_random(Wo, rng, 0.3f);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D / num_heads));

    Tensor O_sab;
    brotensor::self_attention_bias_forward(X, Wq, Wk, Wv, Wo, nullptr,
                                           nullptr, num_heads, scale, O_sab);

    Tensor Qh, Kh, Vh, Attnh, Yconcat, O_mha;
    brotensor::self_attention_forward_train(
        X, Wq, Wk, Wv, Wo, nullptr, num_heads,
        Qh, Kh, Vh, Attnh, Yconcat, O_mha);

    compare_tensors(O_mha, O_sab, "sab_vs_mha", 1e-4f, 1e-3f);
}

void run_sab_bf16(int L, int D, int num_heads, float scale, bool with_bias,
                  uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = make_qbf_cpu(L, D, rng, 0.3f);
    Tensor Wq = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wk = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wv = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wo = make_qbf_cpu(D, D, rng, 0.3f);

    Tensor bias;
    if (with_bias) {
        bias = Tensor::mat(num_heads * L, L);
        fill_random(bias, rng, 0.5f);
    }
    const Tensor* bias_cpu = with_bias ? &bias : nullptr;

    Tensor O_c;
    brotensor::self_attention_bias_forward(X, Wq, Wk, Wv, Wo, nullptr,
                                           bias_cpu, num_heads, scale, O_c);

    Tensor gX  = to_bf16_gpu_t(X);
    Tensor gWq = to_bf16_gpu_t(Wq), gWk = to_bf16_gpu_t(Wk),
           gWv = to_bf16_gpu_t(Wv), gWo = to_bf16_gpu_t(Wo);
    Tensor gbias;
    const Tensor* bias_gpu = nullptr;
    if (with_bias) { gbias = bias.to(gpu_device()); bias_gpu = &gbias; }

    Tensor gO;
    brotensor::self_attention_bias_forward(gX, gWq, gWk, gWv, gWo, nullptr,
                                           bias_gpu, num_heads, scale, gO);

    compare_tensors(O_c, bf16_cuda_to_cpu(gO), "sab.O.bf16", 8e-2f, 8e-2f);
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

// BF16 self_attention_forward (GPU BF16 vs FP32 flash reference).
BT_PARITY_TEST(self_attn_fwd_bf16_L8_D32_h4)  { run_self_attn_forward_bf16(8,  32, 4, 0x520ull, nullptr); }
BT_PARITY_TEST(self_attn_fwd_bf16_L12_D64_h8) { run_self_attn_forward_bf16(12, 64, 8, 0x521ull, nullptr); }
BT_PARITY_TEST(self_attn_fwd_bf16_L8_D32_h4_mask) {
    auto m = partial_mask(8); run_self_attn_forward_bf16(8, 32, 4, 0x522ull, &m);
}

// ─── self_attention_bias_forward (FP32 CPU<->CUDA parity) ──────────────────
BT_PARITY_TEST(sab_L8_D32_h4_nobias) {
    run_sab(8, 32, 4, 0.176776695f, false, nullptr, 0x600ull);
}
BT_PARITY_TEST(sab_L8_D32_h4_bias) {
    run_sab(8, 32, 4, 0.176776695f, true, nullptr, 0x601ull);
}
BT_PARITY_TEST(sab_L16_D64_h8_bias) {
    run_sab(16, 64, 8, 0.125f, true, nullptr, 0x602ull);
}
BT_PARITY_TEST(sab_L6_D48_h6_bias_t5scale) {
    run_sab(6, 48, 6, 1.0f, true, nullptr, 0x603ull);  // T5: unscaled scores
}
BT_PARITY_TEST(sab_L8_D32_h4_bias_mask) {
    auto m = partial_mask(8);
    run_sab(8, 32, 4, 0.176776695f, true, &m, 0x604ull);
}
BT_PARITY_TEST(sab_L16_D64_h8_bias_mask) {
    auto m = partial_mask(16);
    run_sab(16, 64, 8, 0.125f, true, &m, 0x605ull);
}

// No-bias scale=1/sqrt(dh) reproduces ordinary self-attention.
BT_PARITY_TEST(sab_vs_mha_L8_D32_h4)  { run_sab_vs_mha(8, 32, 4, 0x610ull); }
BT_PARITY_TEST(sab_vs_mha_L12_D48_h6) { run_sab_vs_mha(12, 48, 6, 0x611ull); }

// BF16 on CUDA vs FP32 CPU reference.
BT_PARITY_TEST(sab_bf16_L8_D32_h4_bias) {
    run_sab_bf16(8, 32, 4, 0.176776695f, true, 0x620ull);
}
BT_PARITY_TEST(sab_bf16_L12_D64_h8_bias) {
    run_sab_bf16(12, 64, 8, 0.125f, true, 0x621ull);
}
BT_PARITY_TEST(sab_bf16_L8_D32_h4_t5scale) {
    run_sab_bf16(8, 32, 4, 1.0f, true, 0x622ull);
}

int main() { return run_all("self_attention cpu/gpu parity"); }
