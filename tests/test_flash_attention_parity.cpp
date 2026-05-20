// CPU<->GPU parity for the flash-attention family + self_attention_forward
// (CHUNK 6).
//
//   flash_attention_forward                  — SDPA over pre-projected QKV.
//   flash_attention_qkvo_forward             — fused QKVO projections + SDPA.
//   flash_attention_qkvo_backward            — full backward of the above.
//   flash_attention_backward                 — backward over pre-projected QKV.
//   flash_attention_project_kv               — project ctx → K_out, V_out.
//   flash_attention_q_with_kv_cached_forward — Q projection + cached-KV attn.
//   self_attention_forward                   — forward-only self-attention.
//
// DTYPE NOTES
//   Every op in this family runs FP16 internally on the GPU and FP32 on the
//   CPU. We quantise all inputs through FP16 so both backends start from
//   identical values, feed FP16 to the GPU and FP32 to the CPU, and compare
//   with a loose FP16-scale tolerance — the same envelope test_flash_attention.cpp
//   uses (atol 1e-2, rtol 1e-2; relaxed to 2e-2/3e-2 for qkvo, which chains
//   three FP16 matmuls + softmax).
//
// ACCUMULATION
//   * flash_attention_backward:      dQ/dK/dV OVERWRITTEN.
//   * flash_attention_qkvo_backward: dX/dCtx OVERWRITTEN; dWq/dWk/dWv/dWo and
//                                    dbq/dbk/dbv/dbo ACCUMULATE (+=). The
//                                    weight/bias grad buffers are pre-filled
//                                    with a non-zero baseline (identical on
//                                    both backends) to verify accumulation.

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
    return Tensor::from_host_fp16_on(Device::CUDA, h.data(), cpu.rows, cpu.cols);
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

// ─── flash_attention_forward parity ───────────────────────────────────────

void run_flash_forward(int Lq, int Lk, int D, int num_heads, uint64_t seed,
                       bool use_mask, bool causal) {
    SplitMix64 rng(seed);
    Tensor Q = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor K = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor V = make_q16_cpu(Lk, D, rng, 0.3f);

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    Tensor O_c;
    brotensor::flash_attention_forward(Q, K, V, host_mask, num_heads, causal,
                                       O_c);

    Tensor gQ = to_fp16_cuda(Q), gK = to_fp16_cuda(K), gV = to_fp16_cuda(V);
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gO;
    brotensor::flash_attention_forward(gQ, gK, gV, d_mask, num_heads, causal,
                                       gO);

    compare_tensors(O_c, fp16_cuda_to_cpu(gO), "flash_fwd.O", 1e-2f, 1e-2f);
}

// ─── flash_attention_qkvo_forward parity (self + cross) ───────────────────

void run_qkvo_forward(int Lq, int Lk, int D, int D_ctx, int num_heads,
                      uint64_t seed, bool cross, bool use_bias,
                      bool use_mask, bool causal) {
    SplitMix64 rng(seed);
    Tensor X  = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor Cx = cross ? make_q16_cpu(Lk, D_ctx, rng, 0.3f) : Tensor{};
    const int kv_cols = cross ? D_ctx : D;
    Tensor Wq = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wk = make_q16_cpu(D, kv_cols, rng, 0.3f);
    Tensor Wv = make_q16_cpu(D, kv_cols, rng, 0.3f);
    Tensor Wo = make_q16_cpu(D, D, rng, 0.3f);

    Tensor bq, bk, bv, bo;
    if (use_bias) {
        bq = make_q16_cpu(D, 1, rng, 0.2f);
        bk = make_q16_cpu(D, 1, rng, 0.2f);
        bv = make_q16_cpu(D, 1, rng, 0.2f);
        bo = make_q16_cpu(D, 1, rng, 0.2f);
    }

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    // CPU path.
    const Tensor* Cx_c  = cross ? &Cx : nullptr;
    const Tensor* bq_c  = use_bias ? &bq : nullptr;
    const Tensor* bk_c  = use_bias ? &bk : nullptr;
    const Tensor* bv_c  = use_bias ? &bv : nullptr;
    const Tensor* bo_c  = use_bias ? &bo : nullptr;
    Tensor O_c;
    brotensor::flash_attention_qkvo_forward(
        X, Cx_c, Wq, bq_c, Wk, bk_c, Wv, bv_c, Wo, bo_c,
        host_mask, num_heads, causal, O_c);

    // GPU path.
    Tensor gX = to_fp16_cuda(X);
    Tensor gCx; if (cross) gCx = to_fp16_cuda(Cx);
    Tensor gWq = to_fp16_cuda(Wq), gWk = to_fp16_cuda(Wk),
           gWv = to_fp16_cuda(Wv), gWo = to_fp16_cuda(Wo);
    Tensor gbq, gbk, gbv, gbo;
    if (use_bias) {
        gbq = to_fp16_cuda(bq); gbk = to_fp16_cuda(bk);
        gbv = to_fp16_cuda(bv); gbo = to_fp16_cuda(bo);
    }
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    const Tensor* gCx_p = cross ? &gCx : nullptr;
    const Tensor* gbq_p = use_bias ? &gbq : nullptr;
    const Tensor* gbk_p = use_bias ? &gbk : nullptr;
    const Tensor* gbv_p = use_bias ? &gbv : nullptr;
    const Tensor* gbo_p = use_bias ? &gbo : nullptr;
    Tensor gO;
    brotensor::flash_attention_qkvo_forward(
        gX, gCx_p, gWq, gbq_p, gWk, gbk_p, gWv, gbv_p, gWo, gbo_p,
        d_mask, num_heads, causal, gO);

    compare_tensors(O_c, fp16_cuda_to_cpu(gO), "qkvo_fwd.O", 2e-2f, 3e-2f);
}

// ─── flash_attention_project_kv parity ────────────────────────────────────

void run_project_kv(int Lk, int D, int D_ctx, uint64_t seed, bool use_bias) {
    SplitMix64 rng(seed);
    Tensor ctx = make_q16_cpu(Lk, D_ctx, rng, 0.3f);
    Tensor Wk  = make_q16_cpu(D, D_ctx, rng, 0.3f);
    Tensor Wv  = make_q16_cpu(D, D_ctx, rng, 0.3f);
    Tensor bk, bv;
    if (use_bias) {
        bk = make_q16_cpu(D, 1, rng, 0.2f);
        bv = make_q16_cpu(D, 1, rng, 0.2f);
    }
    const Tensor* bk_c = use_bias ? &bk : nullptr;
    const Tensor* bv_c = use_bias ? &bv : nullptr;

    Tensor K_c, V_c;
    brotensor::flash_attention_project_kv(ctx, Wk, bk_c, Wv, bv_c, K_c, V_c);

    Tensor gctx = to_fp16_cuda(ctx);
    Tensor gWk = to_fp16_cuda(Wk), gWv = to_fp16_cuda(Wv);
    Tensor gbk, gbv;
    if (use_bias) { gbk = to_fp16_cuda(bk); gbv = to_fp16_cuda(bv); }
    const Tensor* gbk_p = use_bias ? &gbk : nullptr;
    const Tensor* gbv_p = use_bias ? &gbv : nullptr;
    Tensor gK, gV;
    brotensor::flash_attention_project_kv(gctx, gWk, gbk_p, gWv, gbv_p,
                                          gK, gV);

    compare_tensors(K_c, fp16_cuda_to_cpu(gK), "project_kv.K", 1e-2f, 1e-2f);
    compare_tensors(V_c, fp16_cuda_to_cpu(gV), "project_kv.V", 1e-2f, 1e-2f);
}

// ─── flash_attention_q_with_kv_cached_forward parity ──────────────────────

void run_q_cached(int Lq, int Lk, int D, int num_heads, uint64_t seed,
                  bool use_bias, bool use_mask) {
    SplitMix64 rng(seed);
    Tensor X  = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor K  = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor V  = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor Wq = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wo = make_q16_cpu(D, D, rng, 0.3f);
    Tensor bq, bo;
    if (use_bias) {
        bq = make_q16_cpu(D, 1, rng, 0.2f);
        bo = make_q16_cpu(D, 1, rng, 0.2f);
    }
    const Tensor* bq_c = use_bias ? &bq : nullptr;
    const Tensor* bo_c = use_bias ? &bo : nullptr;

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    Tensor O_c;
    brotensor::flash_attention_q_with_kv_cached_forward(
        X, K, V, Wq, bq_c, Wo, bo_c, host_mask, num_heads,
        /*causal=*/false, O_c);

    Tensor gX = to_fp16_cuda(X), gK = to_fp16_cuda(K), gV = to_fp16_cuda(V);
    Tensor gWq = to_fp16_cuda(Wq), gWo = to_fp16_cuda(Wo);
    Tensor gbq, gbo;
    if (use_bias) { gbq = to_fp16_cuda(bq); gbo = to_fp16_cuda(bo); }
    const Tensor* gbq_p = use_bias ? &gbq : nullptr;
    const Tensor* gbo_p = use_bias ? &gbo : nullptr;
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gO;
    brotensor::flash_attention_q_with_kv_cached_forward(
        gX, gK, gV, gWq, gbq_p, gWo, gbo_p, d_mask, num_heads,
        /*causal=*/false, gO);

    compare_tensors(O_c, fp16_cuda_to_cpu(gO), "q_cached.O", 2e-2f, 3e-2f);
}

// ─── self_attention_forward parity ────────────────────────────────────────

void run_self_attention(int L, int D, int num_heads, uint64_t seed,
                        bool use_mask) {
    SplitMix64 rng(seed);
    Tensor X  = make_q16_cpu(L, D, rng, 0.3f);
    Tensor Wq = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wk = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wv = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wo = make_q16_cpu(D, D, rng, 0.3f);

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(L); host_mask = mask_host.data(); }

    Tensor O_c;
    brotensor::self_attention_forward(X, Wq, Wk, Wv, Wo, host_mask, num_heads,
                                      O_c);

    // self_attention_forward is the one flash-family op with an FP32 GPU
    // path: for FP32 input it falls back to mha_forward (which gates masked
    // query rows), whereas the FP16 path takes the flash route that masks
    // keys only. The FP32-only CPU op delegates to mha_forward, so parity is
    // compared FP32<->FP32 — both backends then run the identical mha path.
    Tensor gX = X.to(Device::CUDA);
    Tensor gWq = Wq.to(Device::CUDA), gWk = Wk.to(Device::CUDA),
           gWv = Wv.to(Device::CUDA), gWo = Wo.to(Device::CUDA);
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), L, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gO;
    brotensor::self_attention_forward(gX, gWq, gWk, gWv, gWo, d_mask,
                                      num_heads, gO);

    compare_tensors(O_c, download_to_host(gO), "self_attn.O", 1e-4f, 1e-3f);
}

// ─── flash_attention_backward parity ──────────────────────────────────────

void run_flash_backward(int Lq, int Lk, int D, int num_heads, uint64_t seed,
                        bool use_mask, bool causal) {
    SplitMix64 rng(seed);
    Tensor Q  = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor K  = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor V  = make_q16_cpu(Lk, D, rng, 0.3f);
    Tensor dO = make_q16_cpu(Lq, D, rng, 0.3f);

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    // CPU path. O is recompute-based — pass an empty tensor.
    Tensor O_dummy;
    Tensor dQ_c, dK_c, dV_c;
    brotensor::flash_attention_backward(Q, K, V, O_dummy, dO, host_mask,
                                        num_heads, causal, dQ_c, dK_c, dV_c);

    // GPU path.
    Tensor gQ = to_fp16_cuda(Q), gK = to_fp16_cuda(K), gV = to_fp16_cuda(V);
    Tensor gdO = to_fp16_cuda(dO);
    Tensor gO_dummy = Tensor::empty_on(Device::CUDA, 0, 0, Dtype::FP16);
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gdQ = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    Tensor gdK = Tensor::empty_on(Device::CUDA, Lk, D, Dtype::FP16);
    Tensor gdV = Tensor::empty_on(Device::CUDA, Lk, D, Dtype::FP16);
    brotensor::flash_attention_backward(gQ, gK, gV, gO_dummy, gdO, d_mask,
                                        num_heads, causal, gdQ, gdK, gdV);

    compare_tensors(dQ_c, fp16_cuda_to_cpu(gdQ), "flash_bwd.dQ", 2e-2f, 3e-2f);
    compare_tensors(dK_c, fp16_cuda_to_cpu(gdK), "flash_bwd.dK", 2e-2f, 3e-2f);
    compare_tensors(dV_c, fp16_cuda_to_cpu(gdV), "flash_bwd.dV", 2e-2f, 3e-2f);
}

// ─── flash_attention_qkvo_backward parity (self + cross, accumulation) ────

void run_qkvo_backward(int Lq, int Lk, int D, int D_ctx, int num_heads,
                       uint64_t seed, bool cross, bool use_bias,
                       bool use_mask, bool causal) {
    SplitMix64 rng(seed);
    Tensor X  = make_q16_cpu(Lq, D, rng, 0.3f);
    Tensor Cx = cross ? make_q16_cpu(Lk, D_ctx, rng, 0.3f) : Tensor{};
    const int kv_cols = cross ? D_ctx : D;
    Tensor Wq = make_q16_cpu(D, D, rng, 0.3f);
    Tensor Wk = make_q16_cpu(D, kv_cols, rng, 0.3f);
    Tensor Wv = make_q16_cpu(D, kv_cols, rng, 0.3f);
    Tensor Wo = make_q16_cpu(D, D, rng, 0.3f);
    Tensor dO = make_q16_cpu(Lq, D, rng, 0.3f);

    Tensor bq, bk, bv, bo;
    if (use_bias) {
        bq = make_q16_cpu(D, 1, rng, 0.2f);
        bk = make_q16_cpu(D, 1, rng, 0.2f);
        bv = make_q16_cpu(D, 1, rng, 0.2f);
        bo = make_q16_cpu(D, 1, rng, 0.2f);
    }

    // Non-zero gradient baselines (accumulation contract for weight/bias grads).
    Tensor dWq_init = make_q16_cpu(D, D, rng, 0.1f);
    Tensor dWk_init = make_q16_cpu(D, kv_cols, rng, 0.1f);
    Tensor dWv_init = make_q16_cpu(D, kv_cols, rng, 0.1f);
    Tensor dWo_init = make_q16_cpu(D, D, rng, 0.1f);
    Tensor dbq_init, dbk_init, dbv_init, dbo_init;
    if (use_bias) {
        dbq_init = make_q16_cpu(D, 1, rng, 0.1f);
        dbk_init = make_q16_cpu(D, 1, rng, 0.1f);
        dbv_init = make_q16_cpu(D, 1, rng, 0.1f);
        dbo_init = make_q16_cpu(D, 1, rng, 0.1f);
    }

    std::vector<float> mask_host;
    const float* host_mask = nullptr;
    if (use_mask) { mask_host = key_mask(Lk); host_mask = mask_host.data(); }

    const Tensor* Cx_c = cross ? &Cx : nullptr;
    const Tensor* bq_c = use_bias ? &bq : nullptr;
    const Tensor* bk_c = use_bias ? &bk : nullptr;
    const Tensor* bv_c = use_bias ? &bv : nullptr;
    const Tensor* bo_c = use_bias ? &bo : nullptr;

    // CPU path.
    Tensor dX_c = Tensor::mat(Lq, D);
    Tensor dCtx_c = cross ? Tensor::mat(Lk, D_ctx) : Tensor{};
    Tensor* dCtx_c_p = cross ? &dCtx_c : nullptr;
    Tensor dWq_c = dWq_init, dWk_c = dWk_init,
           dWv_c = dWv_init, dWo_c = dWo_init;
    Tensor dbq_c, dbk_c, dbv_c, dbo_c;
    Tensor *dbq_cp = nullptr, *dbk_cp = nullptr,
           *dbv_cp = nullptr, *dbo_cp = nullptr;
    if (use_bias) {
        dbq_c = dbq_init; dbk_c = dbk_init;
        dbv_c = dbv_init; dbo_c = dbo_init;
        dbq_cp = &dbq_c; dbk_cp = &dbk_c;
        dbv_cp = &dbv_c; dbo_cp = &dbo_c;
    }
    brotensor::flash_attention_qkvo_backward(
        X, Cx_c, Wq, bq_c, Wk, bk_c, Wv, bv_c, Wo, bo_c,
        host_mask, num_heads, causal, dO,
        dX_c, dCtx_c_p,
        dWq_c, dbq_cp, dWk_c, dbk_cp, dWv_c, dbv_cp, dWo_c, dbo_cp);

    // GPU path.
    Tensor gX = to_fp16_cuda(X);
    Tensor gCx; if (cross) gCx = to_fp16_cuda(Cx);
    Tensor gWq = to_fp16_cuda(Wq), gWk = to_fp16_cuda(Wk),
           gWv = to_fp16_cuda(Wv), gWo = to_fp16_cuda(Wo);
    Tensor gdO = to_fp16_cuda(dO);
    Tensor gbq, gbk, gbv, gbo;
    if (use_bias) {
        gbq = to_fp16_cuda(bq); gbk = to_fp16_cuda(bk);
        gbv = to_fp16_cuda(bv); gbo = to_fp16_cuda(bo);
    }
    const Tensor* gCx_p = cross ? &gCx : nullptr;
    const Tensor* gbq_p = use_bias ? &gbq : nullptr;
    const Tensor* gbk_p = use_bias ? &gbk : nullptr;
    const Tensor* gbv_p = use_bias ? &gbv : nullptr;
    const Tensor* gbo_p = use_bias ? &gbo : nullptr;
    Tensor mg;
    const float* d_mask = nullptr;
    if (use_mask) {
        mg = Tensor::from_host_on(Device::CUDA, mask_host.data(), Lk, 1);
        d_mask = static_cast<const float*>(mg.data);
    }
    Tensor gdX = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    Tensor gdCtx;
    Tensor* gdCtx_p = nullptr;
    if (cross) {
        gdCtx = Tensor::empty_on(Device::CUDA, Lk, D_ctx, Dtype::FP16);
        gdCtx_p = &gdCtx;
    }
    Tensor gdWq = to_fp16_cuda(dWq_init);
    Tensor gdWk = to_fp16_cuda(dWk_init);
    Tensor gdWv = to_fp16_cuda(dWv_init);
    Tensor gdWo = to_fp16_cuda(dWo_init);
    Tensor gdbq, gdbk, gdbv, gdbo;
    Tensor *gdbq_p = nullptr, *gdbk_p = nullptr,
           *gdbv_p = nullptr, *gdbo_p = nullptr;
    if (use_bias) {
        gdbq = to_fp16_cuda(dbq_init); gdbk = to_fp16_cuda(dbk_init);
        gdbv = to_fp16_cuda(dbv_init); gdbo = to_fp16_cuda(dbo_init);
        gdbq_p = &gdbq; gdbk_p = &gdbk; gdbv_p = &gdbv; gdbo_p = &gdbo;
    }
    brotensor::flash_attention_qkvo_backward(
        gX, gCx_p, gWq, gbq_p, gWk, gbk_p, gWv, gbv_p, gWo, gbo_p,
        d_mask, num_heads, causal, gdO,
        gdX, gdCtx_p,
        gdWq, gdbq_p, gdWk, gdbk_p, gdWv, gdbv_p, gdWo, gdbo_p);

    const float atol = 2e-2f, rtol = 3e-2f;
    compare_tensors(dX_c,  fp16_cuda_to_cpu(gdX),  "qkvo_bwd.dX",  atol, rtol);
    if (cross)
        compare_tensors(dCtx_c, fp16_cuda_to_cpu(gdCtx),
                        "qkvo_bwd.dCtx", atol, rtol);
    compare_tensors(dWq_c, fp16_cuda_to_cpu(gdWq), "qkvo_bwd.dWq", atol, rtol);
    compare_tensors(dWk_c, fp16_cuda_to_cpu(gdWk), "qkvo_bwd.dWk", atol, rtol);
    compare_tensors(dWv_c, fp16_cuda_to_cpu(gdWv), "qkvo_bwd.dWv", atol, rtol);
    compare_tensors(dWo_c, fp16_cuda_to_cpu(gdWo), "qkvo_bwd.dWo", atol, rtol);
    if (use_bias) {
        compare_tensors(dbq_c, fp16_cuda_to_cpu(gdbq),
                        "qkvo_bwd.dbq", atol, rtol);
        compare_tensors(dbk_c, fp16_cuda_to_cpu(gdbk),
                        "qkvo_bwd.dbk", atol, rtol);
        compare_tensors(dbv_c, fp16_cuda_to_cpu(gdbv),
                        "qkvo_bwd.dbv", atol, rtol);
        compare_tensors(dbo_c, fp16_cuda_to_cpu(gdbo),
                        "qkvo_bwd.dbo", atol, rtol);
    }
}

} // namespace

// ─── flash_attention_forward ──────────────────────────────────────────────
BT_PARITY_TEST(flash_fwd_8x8_D32_h4)   { run_flash_forward(8,  8,  32, 4, 0x700ull, false, false); }
BT_PARITY_TEST(flash_fwd_6x10_D48_h6)  { run_flash_forward(6,  10, 48, 6, 0x701ull, false, false); }
BT_PARITY_TEST(flash_fwd_12x20_D64_h8) { run_flash_forward(12, 20, 64, 8, 0x702ull, false, false); }
BT_PARITY_TEST(flash_fwd_8x8_D32_h4_mask)   { run_flash_forward(8, 8, 32, 4, 0x703ull, true,  false); }
BT_PARITY_TEST(flash_fwd_10x10_D32_h2_causal){ run_flash_forward(10, 10, 32, 2, 0x704ull, false, true); }

// ─── flash_attention_qkvo_forward (self + cross, bias, mask, causal) ──────
BT_PARITY_TEST(qkvo_fwd_self_8x8_D32_h4) {
    run_qkvo_forward(8, 8, 32, 32, 4, 0x710ull, false, false, false, false);
}
BT_PARITY_TEST(qkvo_fwd_self_8x8_D32_h4_bias) {
    run_qkvo_forward(8, 8, 32, 32, 4, 0x711ull, false, true, false, false);
}
BT_PARITY_TEST(qkvo_fwd_cross_6x10_D48_Dctx24_h6) {
    run_qkvo_forward(6, 10, 48, 24, 6, 0x712ull, true, false, false, false);
}
BT_PARITY_TEST(qkvo_fwd_cross_6x10_D48_Dctx24_h6_bias_mask) {
    run_qkvo_forward(6, 10, 48, 24, 6, 0x713ull, true, true, true, false);
}
BT_PARITY_TEST(qkvo_fwd_self_10x10_D32_h2_causal) {
    run_qkvo_forward(10, 10, 32, 32, 2, 0x714ull, false, true, false, true);
}

// ─── flash_attention_project_kv ───────────────────────────────────────────
BT_PARITY_TEST(project_kv_10_D32_Dctx24)      { run_project_kv(10, 32, 24, 0x720ull, false); }
BT_PARITY_TEST(project_kv_16_D48_Dctx48_bias) { run_project_kv(16, 48, 48, 0x721ull, true); }

// ─── flash_attention_q_with_kv_cached_forward ─────────────────────────────
BT_PARITY_TEST(q_cached_8x12_D32_h4)        { run_q_cached(8, 12, 32, 4, 0x730ull, false, false); }
BT_PARITY_TEST(q_cached_6x10_D48_h6_bias)   { run_q_cached(6, 10, 48, 6, 0x731ull, true,  false); }
BT_PARITY_TEST(q_cached_8x12_D32_h4_mask)   { run_q_cached(8, 12, 32, 4, 0x732ull, false, true); }

// ─── self_attention_forward ───────────────────────────────────────────────
BT_PARITY_TEST(self_attn_8_D32_h4)      { run_self_attention(8,  32, 4, 0x740ull, false); }
BT_PARITY_TEST(self_attn_12_D64_h8)     { run_self_attention(12, 64, 8, 0x741ull, false); }
BT_PARITY_TEST(self_attn_8_D32_h4_mask) { run_self_attention(8,  32, 4, 0x742ull, true); }

// ─── flash_attention_backward ─────────────────────────────────────────────
BT_PARITY_TEST(flash_bwd_8x8_D32_h4)   { run_flash_backward(8,  8,  32, 4, 0x750ull, false, false); }
BT_PARITY_TEST(flash_bwd_6x10_D48_h6)  { run_flash_backward(6,  10, 48, 6, 0x751ull, false, false); }
BT_PARITY_TEST(flash_bwd_8x8_D32_h4_mask)    { run_flash_backward(8, 8, 32, 4, 0x752ull, true,  false); }
BT_PARITY_TEST(flash_bwd_10x10_D32_h2_causal){ run_flash_backward(10, 10, 32, 2, 0x753ull, false, true); }

// ─── flash_attention_qkvo_backward (self + cross, bias, mask, causal) ─────
BT_PARITY_TEST(qkvo_bwd_self_8x8_D32_h4) {
    run_qkvo_backward(8, 8, 32, 32, 4, 0x760ull, false, false, false, false);
}
BT_PARITY_TEST(qkvo_bwd_self_8x8_D32_h4_bias) {
    run_qkvo_backward(8, 8, 32, 32, 4, 0x761ull, false, true, false, false);
}
BT_PARITY_TEST(qkvo_bwd_cross_6x10_D48_Dctx24_h6) {
    run_qkvo_backward(6, 10, 48, 24, 6, 0x762ull, true, false, false, false);
}
BT_PARITY_TEST(qkvo_bwd_cross_6x10_D48_Dctx24_h6_bias_mask) {
    run_qkvo_backward(6, 10, 48, 24, 6, 0x763ull, true, true, true, false);
}
BT_PARITY_TEST(qkvo_bwd_self_10x10_D32_h2_causal) {
    run_qkvo_backward(10, 10, 32, 32, 2, 0x764ull, false, true, false, true);
}

int main() { return run_all("flash_attention cpu/gpu parity"); }
