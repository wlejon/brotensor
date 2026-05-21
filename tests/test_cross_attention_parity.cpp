// CPU<->GPU parity for the cross-attention family (CHUNK 5).
//
//   cross_attention_forward_train  — FP32 on both backends.
//   cross_attention_backward       — FP32 on both backends.
//   cross_attention_forward        — GPU FP16-inference, CPU FP32.
//   cross_attention_forward_with_attn — GPU FP16-only, CPU FP32.
//
// DTYPE NOTES
//   * cross_attention_forward_train / cross_attention_backward run FP32 on
//     both backends — tested FP32<->FP32 with atol=1e-4, rtol=1e-3.
//   * cross_attention_forward: the CUDA op delegates to the FP16 flash path
//     for FP16 inputs. The CPU op is FP32-only. We feed FP16 tensors to the
//     GPU and FP32 (FP16-quantised) tensors to the CPU, comparing with a
//     loose FP16-scale tolerance (atol 2e-2, rtol 3e-2) — the same envelope
//     test_cross_attention.cpp uses for this op.
//   * cross_attention_forward_with_attn: the CUDA op is FP16-only. Same
//     FP16-vs-FP32 strategy and loose tolerance.
//
// ACCUMULATION: cross_attention_backward accumulates dWq/dWk/dWv/dWo (+=) and
// overwrites dX/dCtx. Gradient buffers are pre-filled with a non-zero
// baseline (identical on both backends) so the test verifies accumulation.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

inline float q16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

// CPU FP32 tensor with FP16-quantised random values.
Tensor make_q16_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = q16(rng.next_unit() * scale);
    return t;
}

// Upload a CPU FP32 tensor as an FP16 CUDA tensor of the same shape.
Tensor to_fp16_cuda(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_fp16_bits(cpu[i]);
    return Tensor::from_host_fp16_on(gpu_device(), h.data(),
                                     cpu.rows, cpu.cols);
}

// Download an FP16 CUDA tensor into a CPU FP32 tensor.
Tensor fp16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_fp16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return out;
}

std::vector<float> key_mask(int Lk) {
    std::vector<float> m(Lk, 1.0f);
    for (int k = 3 * Lk / 4; k < Lk; ++k) m[k] = 0.0f;
    return m;
}

// ─── BF16 helpers (GPU-only: BF16-on-CUDA vs FP32 CPU reference) ───────────

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
    return Tensor::from_host_bf16_on(gpu_device(), h.data(), cpu.rows, cpu.cols);
}

Tensor bf16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_bf16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::bf16_bits_to_fp32(h[i]);
    return out;
}

// ─── FP32 train forward + backward parity ─────────────────────────────────

void run_cross_train(int Lq, int Lk, int D, int Dctx, int num_heads,
                     uint64_t seed, const std::vector<float>* mask) {
    SplitMix64 rng(seed);

    Tensor X   = Tensor::mat(Lq, D);
    Tensor Ctx = Tensor::mat(Lk, Dctx);
    Tensor dO  = Tensor::mat(Lq, D);
    fill_random(X, rng, 0.3f);
    fill_random(Ctx, rng, 0.3f);
    fill_random(dO, rng, 0.3f);

    Tensor Wq = Tensor::mat(D, D), Wk = Tensor::mat(D, Dctx),
           Wv = Tensor::mat(D, Dctx), Wo = Tensor::mat(D, D);
    fill_random(Wq, rng, 0.3f);
    fill_random(Wk, rng, 0.3f);
    fill_random(Wv, rng, 0.3f);
    fill_random(Wo, rng, 0.3f);

    // Non-zero gradient baselines (accumulation contract).
    Tensor dWq_init = Tensor::mat(D, D), dWk_init = Tensor::mat(D, Dctx),
           dWv_init = Tensor::mat(D, Dctx), dWo_init = Tensor::mat(D, D);
    fill_random(dWq_init, rng, 0.1f);
    fill_random(dWk_init, rng, 0.1f);
    fill_random(dWv_init, rng, 0.1f);
    fill_random(dWo_init, rng, 0.1f);

    const float* host_mask = mask ? mask->data() : nullptr;

    // CPU path.
    Tensor Qh_c, Kh_c, Vh_c, Attnh_c, Yconcat_c, O_c;
    brotensor::cross_attention_forward_train(
        X, Ctx, Wq, Wk, Wv, Wo, host_mask, num_heads,
        Qh_c, Kh_c, Vh_c, Attnh_c, Yconcat_c, O_c);

    Tensor dX_c = Tensor::mat(Lq, D), dCtx_c = Tensor::mat(Lk, Dctx);
    Tensor dWq_c = dWq_init, dWk_c = dWk_init,
           dWv_c = dWv_init, dWo_c = dWo_init;
    brotensor::cross_attention_backward(
        dO, X, Ctx, Qh_c, Kh_c, Vh_c, Attnh_c, Yconcat_c,
        Wq, Wk, Wv, Wo, host_mask, num_heads,
        dX_c, dCtx_c, dWq_c, dWk_c, dWv_c, dWo_c);

    // GPU path.
    Tensor gX   = X.to(gpu_device());
    Tensor gCtx = Ctx.to(gpu_device());
    Tensor gWq  = Wq.to(gpu_device()), gWk = Wk.to(gpu_device()),
           gWv  = Wv.to(gpu_device()), gWo = Wo.to(gpu_device());

    Tensor d_mask_buf = upload_mask(mask);
    const float* d_mask = static_cast<const float*>(d_mask_buf.data);

    Tensor gQh, gKh, gVh, gAttnh, gYconcat, gO;
    brotensor::cross_attention_forward_train(
        gX, gCtx, gWq, gWk, gWv, gWo, d_mask, num_heads,
        gQh, gKh, gVh, gAttnh, gYconcat, gO);

    Tensor gdO   = dO.to(gpu_device());
    Tensor gdX   = Tensor::zeros_on(gpu_device(), Lq, D);
    Tensor gdCtx = Tensor::zeros_on(gpu_device(), Lk, Dctx);
    Tensor gdWq  = dWq_init.to(gpu_device());
    Tensor gdWk  = dWk_init.to(gpu_device());
    Tensor gdWv  = dWv_init.to(gpu_device());
    Tensor gdWo  = dWo_init.to(gpu_device());
    brotensor::cross_attention_backward(
        gdO, gX, gCtx, gQh, gKh, gVh, gAttnh, gYconcat,
        gWq, gWk, gWv, gWo, d_mask, num_heads,
        gdX, gdCtx, gdWq, gdWk, gdWv, gdWo);

    const float atol = 1e-4f, rtol = 1e-3f;
    compare_tensors(O_c,    download_to_host(gO),    "cross.O",    atol, rtol);
    compare_tensors(dX_c,   download_to_host(gdX),   "cross.dX",   atol, rtol);
    compare_tensors(dCtx_c, download_to_host(gdCtx), "cross.dCtx", atol, rtol);
    compare_tensors(dWq_c,  download_to_host(gdWq),  "cross.dWq",  atol, rtol);
    compare_tensors(dWk_c,  download_to_host(gdWk),  "cross.dWk",  atol, rtol);
    compare_tensors(dWv_c,  download_to_host(gdWv),  "cross.dWv",  atol, rtol);
    compare_tensors(dWo_c,  download_to_host(gdWo),  "cross.dWo",  atol, rtol);
}

// ─── FP16-inference forward parity (CPU FP32 vs GPU FP16) ─────────────────

void run_cross_forward(int Lq, int Lk, int D, int num_heads,
                       uint64_t seed, bool use_mask) {
    SplitMix64 rng(seed);

    // FP16-quantised inputs so CPU and GPU start from identical values.
    Tensor X   = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor Ctx = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor Wq  = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wk  = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wv  = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wo  = make_q16_cpu(D, D, rng, 0.3f);

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    // CPU FP32 path.
    Tensor O_c;
    brotensor::cross_attention_forward(
        X, Ctx, Wq, Wk, Wv, Wo, host_mask, num_heads, O_c);

    // GPU FP16 path.
    Tensor gX  = to_fp16_cuda(X);
    Tensor gCtx = to_fp16_cuda(Ctx);
    Tensor gWq = to_fp16_cuda(Wq), gWk = to_fp16_cuda(Wk),
           gWv = to_fp16_cuda(Wv), gWo = to_fp16_cuda(Wo);

    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(gpu_device(), mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gO;
    brotensor::cross_attention_forward(
        gX, gCtx, gWq, gWk, gWv, gWo, d_mask, num_heads, gO);

    // Loose FP16-scale tolerance — three matmuls + softmax compound rounding.
    compare_tensors(O_c, fp16_cuda_to_cpu(gO), "cross_fwd.O", 2e-2f, 3e-2f);
}

// ─── FP16-only forward_with_attn parity (CPU FP32 vs GPU FP16) ────────────

void run_cross_forward_with_attn(int Lq, int Lk, int D, int num_heads,
                                 uint64_t seed, bool use_mask, bool use_bias) {
    SplitMix64 rng(seed);

    Tensor X   = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor Ctx = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor Wq  = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wk  = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wv  = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wo  = make_q16_cpu(D, D, rng, 0.3f);

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    // attn_logit_bias is FP32 on both backends (a (Lq, Lk) pre-softmax bias).
    Tensor bias_cpu;
    const Tensor* bias_cpu_ptr = nullptr;
    Tensor bias_gpu;
    const Tensor* bias_gpu_ptr = nullptr;
    if (use_bias) {
        bias_cpu = Tensor::mat(Lq, Lk);
        fill_random(bias_cpu, rng, 0.2f);
        bias_cpu_ptr = &bias_cpu;
        bias_gpu = bias_cpu.to(gpu_device());
        bias_gpu_ptr = &bias_gpu;
    }

    // CPU FP32 path.
    Tensor O_c, AttnAvg_c;
    brotensor::cross_attention_forward_with_attn(
        X, Ctx, Wq, Wk, Wv, Wo, host_mask, bias_cpu_ptr, num_heads,
        O_c, AttnAvg_c);

    // GPU FP16 path.
    Tensor gX  = to_fp16_cuda(X);
    Tensor gCtx = to_fp16_cuda(Ctx);
    Tensor gWq = to_fp16_cuda(Wq), gWk = to_fp16_cuda(Wk),
           gWv = to_fp16_cuda(Wv), gWo = to_fp16_cuda(Wo);

    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(gpu_device(), mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gO, gAttnAvg;
    brotensor::cross_attention_forward_with_attn(
        gX, gCtx, gWq, gWk, gWv, gWo, d_mask, bias_gpu_ptr, num_heads,
        gO, gAttnAvg);

    compare_tensors(O_c, fp16_cuda_to_cpu(gO),
                    "cross_attn.O", 2e-2f, 3e-2f);
    // AttnAvg is a probability average — smaller magnitude, tighter atol.
    compare_tensors(AttnAvg_c, fp16_cuda_to_cpu(gAttnAvg),
                    "cross_attn.AttnAvg", 5e-3f, 3e-2f);
}

// ─── BF16-inference forward parity (CPU FP32 vs GPU BF16) ─────────────────

void run_cross_forward_bf16(int Lq, int Lk, int D, int num_heads,
                            uint64_t seed, bool use_mask) {
    SplitMix64 rng(seed);

    Tensor X   = make_qbf_cpu(Lq, D, rng, 0.3f);
    Tensor Ctx = make_qbf_cpu(Lk, D, rng, 0.3f);
    Tensor Wq  = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wk  = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wv  = make_qbf_cpu(D, D, rng, 0.3f);
    Tensor Wo  = make_qbf_cpu(D, D, rng, 0.3f);

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    // CPU FP32 path.
    Tensor O_c;
    brotensor::cross_attention_forward(
        X, Ctx, Wq, Wk, Wv, Wo, host_mask, num_heads, O_c);

    // GPU BF16 path — cross_attention_forward delegates to the flash path.
    Tensor gX  = to_bf16_gpu_t(X);
    Tensor gCtx = to_bf16_gpu_t(Ctx);
    Tensor gWq = to_bf16_gpu_t(Wq), gWk = to_bf16_gpu_t(Wk),
           gWv = to_bf16_gpu_t(Wv), gWo = to_bf16_gpu_t(Wo);

    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(gpu_device(), mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gO;
    brotensor::cross_attention_forward(
        gX, gCtx, gWq, gWk, gWv, gWo, d_mask, num_heads, gO);

    compare_tensors(O_c, bf16_cuda_to_cpu(gO), "cross_fwd.O.bf16", 8e-2f, 8e-2f);
}

} // namespace

// FP32 train forward + backward (Lq != Lk and Lq == Lk).
BT_PARITY_TEST(cross_train_8x8_D32_h4) {
    run_cross_train(8, 8, 32, 32, 4, 0x600ull, nullptr);
}
BT_PARITY_TEST(cross_train_6x10_D48_Dctx24_h6) {
    run_cross_train(6, 10, 48, 24, 6, 0x601ull, nullptr);
}
BT_PARITY_TEST(cross_train_12x20_D64_Dctx48_h8) {
    run_cross_train(12, 20, 64, 48, 8, 0x602ull, nullptr);
}
BT_PARITY_TEST(cross_train_6x10_D48_Dctx24_h6_mask) {
    auto m = key_mask(10);
    run_cross_train(6, 10, 48, 24, 6, 0x603ull, &m);
}
BT_PARITY_TEST(cross_train_8x8_D32_h4_mask) {
    auto m = key_mask(8);
    run_cross_train(8, 8, 32, 32, 4, 0x604ull, &m);
}

// FP16 inference forward.
BT_PARITY_TEST(cross_fwd_4x5_D8_h1)   { run_cross_forward(4,  5,  8,  1, 0x610ull, false); }
BT_PARITY_TEST(cross_fwd_6x7_D16_h4)  { run_cross_forward(6,  7,  16, 4, 0x611ull, false); }
BT_PARITY_TEST(cross_fwd_8x8_D32_h4)  { run_cross_forward(8,  8,  32, 4, 0x612ull, false); }
BT_PARITY_TEST(cross_fwd_4x8_D16_h2_mask) {
    run_cross_forward(4, 8, 16, 2, 0x613ull, true);
}

// FP16 forward_with_attn (mask + optional bias variants).
BT_PARITY_TEST(cross_attn_6x7_D16_h4)        { run_cross_forward_with_attn(6, 7,  16, 4, 0x620ull, false, false); }
BT_PARITY_TEST(cross_attn_8x8_D32_h4)        { run_cross_forward_with_attn(8, 8,  32, 4, 0x621ull, false, false); }
BT_PARITY_TEST(cross_attn_4x8_D16_h2_mask)   { run_cross_forward_with_attn(4, 8,  16, 2, 0x622ull, true,  false); }
BT_PARITY_TEST(cross_attn_6x10_D48_h6_bias)  { run_cross_forward_with_attn(6, 10, 48, 6, 0x623ull, false, true); }
BT_PARITY_TEST(cross_attn_6x10_D48_h6_mask_bias) {
    run_cross_forward_with_attn(6, 10, 48, 6, 0x624ull, true, true);
}

// BF16 inference forward (CPU FP32 vs GPU BF16).
BT_PARITY_TEST(cross_fwd_bf16_4x5_D8_h1)   { run_cross_forward_bf16(4,  5,  8,  1, 0x6A0ull, false); }
BT_PARITY_TEST(cross_fwd_bf16_6x7_D16_h4)  { run_cross_forward_bf16(6,  7,  16, 4, 0x6A1ull, false); }
BT_PARITY_TEST(cross_fwd_bf16_8x8_D32_h4)  { run_cross_forward_bf16(8,  8,  32, 4, 0x6A2ull, false); }
BT_PARITY_TEST(cross_fwd_bf16_4x8_D16_h2_mask) {
    run_cross_forward_bf16(4, 8, 16, 2, 0x6A3ull, true);
}

int main() { return run_all("cross_attention cpu/gpu parity"); }
