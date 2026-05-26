// CPU backend — flash-attention family (CHUNK 6).
//
// Ground truth: src/cuda/flash_attention.cu and
// src/cuda/flash_attention_backward.cu.
//
// Six ops:
//   * flash_attention_forward                  — SDPA over pre-projected QKV.
//   * flash_attention_qkvo_forward             — fused QKVO projections + SDPA.
//   * flash_attention_qkvo_backward            — full backward of the above.
//   * flash_attention_backward                 — backward over pre-projected QKV.
//   * flash_attention_project_kv               — project ctx → K_out, V_out.
//   * flash_attention_q_with_kv_cached_forward — project X→Q, attend cached K/V,
//                                                project through Wo.
//
// DTYPE DECISIONS
//   Every CUDA op in this family runs FP16 internally (Q/K/V/weights all FP16,
//   attention core in FP16). The CPU backend is FP32-only (per CLAUDE.md), so
//   all six CPU impls run the straightforward FP32 scalar math that produces
//   the SAME mathematical result as the GPU's tiled flash kernels — flash
//   tiling is a memory optimisation, not a different result, so a plain
//   materialised attention is correct. The parity tests quantise inputs
//   through FP16 (so both backends start identical), feed FP16 to the GPU and
//   FP32 to the CPU, and compare with a loose FP16-scale tolerance.
//
// CONVENTIONS (verified against flash_attention.cu / *_backward.cu)
//   * Q/K/V are (Lq|Lk, D) with D = num_heads * head_dim, the head dimension
//     contiguous within each row: element (l, h*head_dim + d).
//   * Softmax scale: 1/sqrt(head_dim).
//   * Mask: length-Lk key-validity buffer (1=valid, 0=invalid). mask[k] <= 0.5
//     drops key k from the softmax (score → -inf). A fully-masked row yields a
//     zero output row (rsum == 0 ⇒ inv == 0). Mask never gates query rows.
//   * Causal: key k contributes to query q only when k <= q (requires Lq==Lk).
//   * Projection weight layout (linear_forward_batched_fp16): Wq/Wo are (D, D);
//     Wk/Wv are (D, D_ctx). out(i, n) = sum_k In(i, k) * W(n, k) + b(n) — i.e.
//     In @ W^T, optional bias broadcast over rows.
//   * flash_attention_qkvo_forward: Ctx==null ⇒ self-attention (kv_src = X);
//     non-null ⇒ cross-attention.
//
// ACCUMULATION (verified against the CUDA backward kernels)
//   * flash_attention_backward:        dQ/dK/dV OVERWRITTEN (CUDA zeros then
//                                      fills each per-head slot).
//   * flash_attention_qkvo_backward:   dX OVERWRITTEN (CUDA zeros it, then the
//                                      single fa_fp16_add of dX_from_Q makes
//                                      it equal that path; self-attn adds the
//                                      K/V paths on top). dCtx OVERWRITTEN
//                                      (zeroed then K+V paths added).
//                                      dWq/dWk/dWv/dWo and dbq/dbk/dbv/dbo
//                                      ACCUMULATE (+=) — linear_backward_batched
//                                      folds into the caller's grad buffers.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

// out(i, n) = sum_k In(i, k) * W(n, k) + (bias ? bias[n] : 0).
//   In : (M, Din)   W : (Dout, Din)   bias : (Dout,) optional   out : (M, Dout)
void linear_proj(const float* In, const float* W, const float* bias,
                 float* Out, int M, int Din, int Dout) {
    for (int i = 0; i < M; ++i) {
        const float* xr = In + static_cast<std::size_t>(i) * Din;
        float* orow = Out + static_cast<std::size_t>(i) * Dout;
        for (int n = 0; n < Dout; ++n) {
            const float* wr = W + static_cast<std::size_t>(n) * Din;
            float acc = bias ? bias[n] : 0.0f;
            for (int k = 0; k < Din; ++k) acc += xr[k] * wr[k];
            orow[n] = acc;
        }
    }
}

// Backward of linear_proj. forward: Out = In @ W^T + b.
//   dIn(i, k)  = sum_n dOut(i, n) * W(n, k)        — accumulate into dIn.
//   dW(n, k)  += sum_i dOut(i, n) * In(i, k)       — accumulate.
//   db(n)     += sum_i dOut(i, n)                  — accumulate (if db != null).
void linear_proj_backward(const float* In, const float* W, const float* dOut,
                          float* dIn, float* dW, float* db,
                          int M, int Din, int Dout) {
    for (int i = 0; i < M; ++i) {
        const float* dor = dOut + static_cast<std::size_t>(i) * Dout;
        float* dir = dIn + static_cast<std::size_t>(i) * Din;
        for (int k = 0; k < Din; ++k) {
            float acc = 0.0f;
            for (int n = 0; n < Dout; ++n)
                acc += dor[n] * W[static_cast<std::size_t>(n) * Din + k];
            dir[k] += acc;
        }
    }
    for (int n = 0; n < Dout; ++n) {
        float dbacc = 0.0f;
        for (int k = 0; k < Din; ++k) {
            float dw = 0.0f;
            for (int i = 0; i < M; ++i)
                dw += dOut[static_cast<std::size_t>(i) * Dout + n] *
                      In[static_cast<std::size_t>(i) * Din + k];
            dW[static_cast<std::size_t>(n) * Din + k] += dw;
        }
        for (int i = 0; i < M; ++i)
            dbacc += dOut[static_cast<std::size_t>(i) * Dout + n];
        if (db) db[n] += dbacc;
    }
}

// Materialised multi-head attention over pre-projected Q/K/V.
//   Q : (Lq, D)   K, V : (Lk, D)   D = H * hd.
//   O : (Lq, D) — written.
//   P (optional, may be null): (H*Lq, Lk) per-head softmax probabilities.
// scale = 1/sqrt(hd); key mask + causal applied during the row softmax.
void attention_core(const float* Q, const float* K, const float* V,
                    const float* mask, int Lq, int Lk, int D, int H,
                    bool causal, float* O, float* P) {
    const int hd = D / H;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> scores(static_cast<std::size_t>(Lk));
    for (int q = 0; q < Lq; ++q) {
        for (int h = 0; h < H; ++h) {
            const int off = h * hd;
            float m = -1e30f;
            for (int k = 0; k < Lk; ++k) {
                if (mask && mask[k] <= 0.5f) { scores[k] = -1e30f; continue; }
                if (causal && k > q)         { scores[k] = -1e30f; continue; }
                const float* qr = Q + static_cast<std::size_t>(q) * D + off;
                const float* kr = K + static_cast<std::size_t>(k) * D + off;
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += qr[d] * kr[d];
                const float s = dot * inv_sqrt;
                scores[k] = s;
                if (s > m) m = s;
            }
            const bool empty = (m <= -1e29f);
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                const float e = empty ? 0.0f : std::exp(scores[k] - m);
                scores[k] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            float* orow = O + static_cast<std::size_t>(q) * D + off;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < Lk; ++k) {
                    acc += scores[k] * inv *
                           V[static_cast<std::size_t>(k) * D + off + d];
                }
                orow[d] = acc;
            }
            if (P) {
                float* prow =
                    P + (static_cast<std::size_t>(h) * Lq + q) * Lk;
                for (int k = 0; k < Lk; ++k) prow[k] = scores[k] * inv;
            }
        }
    }
}

// Per-head flash-attention backward over pre-projected Q/K/V. Recompute P,
// then dV = P^T·dO_attn, dP = dO_attn·V^T, dS = P*(dP - D_q)*scale,
// dQ = dS·K, dK = dS^T·Q. dQ/dK/dV are OVERWRITTEN.
void attention_core_backward(const float* Q, const float* K, const float* V,
                             const float* dO, const float* mask,
                             int Lq, int Lk, int D, int H, bool causal,
                             float* dQ, float* dK, float* dV) {
    const int hd = D / H;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    for (std::size_t i = 0; i < static_cast<std::size_t>(Lq) * D; ++i)
        dQ[i] = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Lk) * D; ++i) {
        dK[i] = 0.0f;
        dV[i] = 0.0f;
    }
    std::vector<float> P(static_cast<std::size_t>(Lq) * Lk);
    std::vector<float> dP(static_cast<std::size_t>(Lq) * Lk);
    std::vector<float> dS(static_cast<std::size_t>(Lq) * Lk);
    for (int h = 0; h < H; ++h) {
        const int off = h * hd;
        // Recompute P for this head.
        for (int q = 0; q < Lq; ++q) {
            float m = -1e30f;
            float* prow = P.data() + static_cast<std::size_t>(q) * Lk;
            for (int k = 0; k < Lk; ++k) {
                if (mask && mask[k] <= 0.5f) { prow[k] = -1e30f; continue; }
                if (causal && k > q)         { prow[k] = -1e30f; continue; }
                const float* qr = Q + static_cast<std::size_t>(q) * D + off;
                const float* kr = K + static_cast<std::size_t>(k) * D + off;
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += qr[d] * kr[d];
                prow[k] = dot * inv_sqrt;
                if (prow[k] > m) m = prow[k];
            }
            const bool empty = (m <= -1e29f);
            float sum = 0.0f;
            for (int k = 0; k < Lk; ++k) {
                const float e = empty ? 0.0f : std::exp(prow[k] - m);
                prow[k] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < Lk; ++k) prow[k] *= inv;
        }
        // dV[k, off+d] = sum_q P[q,k] * dO[q, off+d].
        for (int k = 0; k < Lk; ++k) {
            float* dvr = dV + static_cast<std::size_t>(k) * D + off;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int q = 0; q < Lq; ++q)
                    acc += P[static_cast<std::size_t>(q) * Lk + k] *
                           dO[static_cast<std::size_t>(q) * D + off + d];
                dvr[d] = acc;
            }
        }
        // dP[q,k] = sum_d dO[q, off+d] * V[k, off+d].
        for (int q = 0; q < Lq; ++q) {
            for (int k = 0; k < Lk; ++k) {
                float acc = 0.0f;
                for (int d = 0; d < hd; ++d)
                    acc += dO[static_cast<std::size_t>(q) * D + off + d] *
                           V[static_cast<std::size_t>(k) * D + off + d];
                dP[static_cast<std::size_t>(q) * Lk + k] = acc;
            }
        }
        // dS[q,k] = P[q,k] * (dP[q,k] - D_q) * inv_sqrt.
        for (int q = 0; q < Lq; ++q) {
            const float* prow = P.data() + static_cast<std::size_t>(q) * Lk;
            const float* dpr  = dP.data() + static_cast<std::size_t>(q) * Lk;
            float* dsr = dS.data() + static_cast<std::size_t>(q) * Lk;
            float Dq = 0.0f;
            for (int k = 0; k < Lk; ++k) Dq += prow[k] * dpr[k];
            for (int k = 0; k < Lk; ++k)
                dsr[k] = prow[k] * (dpr[k] - Dq) * inv_sqrt;
        }
        // dQ[q, off+d] = sum_k dS[q,k] * K[k, off+d].
        for (int q = 0; q < Lq; ++q) {
            float* dqr = dQ + static_cast<std::size_t>(q) * D + off;
            const float* dsr = dS.data() + static_cast<std::size_t>(q) * Lk;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < Lk; ++k)
                    acc += dsr[k] *
                           K[static_cast<std::size_t>(k) * D + off + d];
                dqr[d] = acc;
            }
        }
        // dK[k, off+d] = sum_q dS[q,k] * Q[q, off+d].
        for (int k = 0; k < Lk; ++k) {
            float* dkr = dK + static_cast<std::size_t>(k) * D + off;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int q = 0; q < Lq; ++q)
                    acc += dS[static_cast<std::size_t>(q) * Lk + k] *
                           Q[static_cast<std::size_t>(q) * D + off + d];
                dkr[d] = acc;
            }
        }
    }
}

inline void ensure_f32(Tensor& t, int r, int c) {
    if (t.rows != r || t.cols != c || t.dtype != Dtype::FP32)
        t.resize(r, c, Dtype::FP32);
}

} // namespace

// ─── flash_attention_forward ───────────────────────────────────────────────

void flash_attention_forward(const ::brotensor::Tensor& Q,
                             const ::brotensor::Tensor& K,
                             const ::brotensor::Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             ::brotensor::Tensor& O) {
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk)
        throw std::runtime_error("flash_attention_forward: shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error("flash_attention_forward: num_heads must divide D");
    if (causal && Lq != Lk)
        throw std::runtime_error("flash_attention_forward: causal requires Lq == Lk");
    ensure_f32(O, Lq, D);
    if (Lq == 0 || Lk == 0 || D == 0) return;

    attention_core(Q.host_f32(), K.host_f32(), V.host_f32(), d_mask,
                   Lq, Lk, D, num_heads, causal, O.host_f32_mut(), nullptr);
}

// ─── flash_attention_varlen_forward ────────────────────────────────────────
//
// Packed variable-length attention (Qwen3-VL window attention). Q/K/V are one
// big (total_tokens, num_heads*head_dim) tensor each; cu_seqlens_q/k are
// length B+1 INT32 prefix sums delimiting per-sequence row ranges. Sequence b
// runs attention_core over its own Q[Lq_b, D] and K/V[Lk_b, D] slice — no
// cross-sequence attention.
//
// On CPU cu_seqlens_q/k are raw host pointers (matches the existing d_mask
// convention: device pointer on GPU, host pointer on CPU). max_seqlen_q/k are
// only used by the GPU kernel for block sizing and are ignored here.
void flash_attention_varlen_forward(const ::brotensor::Tensor& Q,
                                    const ::brotensor::Tensor& K,
                                    const ::brotensor::Tensor& V,
                                    const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k,
                                    int batch_size,
                                    int /*max_seqlen_q*/,
                                    int /*max_seqlen_k*/,
                                    int num_heads,
                                    int head_dim,
                                    bool causal,
                                    ::brotensor::Tensor& O) {
    const int total_q = Q.rows;
    const int total_k = K.rows;
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k)
        throw std::runtime_error("flash_attention_varlen_forward: shape mismatch");
    if (num_heads <= 0 || head_dim <= 0)
        throw std::runtime_error("flash_attention_varlen_forward: num_heads/head_dim must be positive");
    if (batch_size < 0)
        throw std::runtime_error("flash_attention_varlen_forward: batch_size must be non-negative");
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k))
        throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_q/k required when batch_size > 0");
    ensure_f32(O, total_q, D);
    if (total_q == 0 || D == 0) return;
    if (cu_seqlens_q && cu_seqlens_q[0] != 0)
        throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_q[0] must be 0");
    if (cu_seqlens_k && cu_seqlens_k[0] != 0)
        throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_k[0] must be 0");
    if (batch_size > 0) {
        if (cu_seqlens_q[batch_size] != total_q)
            throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_q[B] != total_tokens_q");
        if (cu_seqlens_k[batch_size] != total_k)
            throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens_k[B] != total_tokens_k");
    }

    const float* Qp = Q.host_f32();
    const float* Kp = K.host_f32();
    const float* Vp = V.host_f32();
    float* Op = O.host_f32_mut();

    for (int b = 0; b < batch_size; ++b) {
        const int q_beg = cu_seqlens_q[b];
        const int q_end = cu_seqlens_q[b + 1];
        const int k_beg = cu_seqlens_k[b];
        const int k_end = cu_seqlens_k[b + 1];
        const int Lq = q_end - q_beg;
        const int Lk = k_end - k_beg;
        if (Lq < 0 || Lk < 0)
            throw std::runtime_error("flash_attention_varlen_forward: cu_seqlens must be non-decreasing");
        if (causal && Lq != Lk)
            throw std::runtime_error("flash_attention_varlen_forward: causal requires per-sequence Lq == Lk");
        if (Lq == 0) continue;
        if (Lk == 0) {
            // Fully empty K range — output rows are zero (no keys to attend to).
            for (std::size_t i = 0; i < static_cast<std::size_t>(Lq) * D; ++i)
                Op[static_cast<std::size_t>(q_beg) * D + i] = 0.0f;
            continue;
        }
        attention_core(Qp + static_cast<std::size_t>(q_beg) * D,
                       Kp + static_cast<std::size_t>(k_beg) * D,
                       Vp + static_cast<std::size_t>(k_beg) * D,
                       /*mask=*/nullptr,
                       Lq, Lk, D, num_heads, causal,
                       Op + static_cast<std::size_t>(q_beg) * D,
                       /*P=*/nullptr);
    }
}

// ─── flash_attention_project_kv ────────────────────────────────────────────

void flash_attention_project_kv(const ::brotensor::Tensor& ctx,
                                const ::brotensor::Tensor& Wk,
                                const ::brotensor::Tensor* bk,
                                const ::brotensor::Tensor& Wv,
                                const ::brotensor::Tensor* bv,
                                ::brotensor::Tensor& K_out,
                                ::brotensor::Tensor& V_out) {
    const int Lk    = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D     = Wk.rows;
    if (Wk.cols != D_ctx || Wv.rows != D || Wv.cols != D_ctx)
        throw std::runtime_error("flash_attention_project_kv: Wk/Wv shape mismatch");
    ensure_f32(K_out, Lk, D);
    ensure_f32(V_out, Lk, D);
    if (Lk == 0 || D == 0) return;

    linear_proj(ctx.host_f32(), Wk.host_f32(),
                bk ? bk->host_f32() : nullptr,
                K_out.host_f32_mut(), Lk, D_ctx, D);
    linear_proj(ctx.host_f32(), Wv.host_f32(),
                bv ? bv->host_f32() : nullptr,
                V_out.host_f32_mut(), Lk, D_ctx, D);
}

// ─── flash_attention_q_with_kv_cached_forward ──────────────────────────────

void flash_attention_q_with_kv_cached_forward(const ::brotensor::Tensor& X,
                                              const ::brotensor::Tensor& K,
                                              const ::brotensor::Tensor& V,
                                              const ::brotensor::Tensor& Wq,
                                              const ::brotensor::Tensor* bq,
                                              const ::brotensor::Tensor& Wo,
                                              const ::brotensor::Tensor* bo,
                                              const float* d_mask,
                                              int num_heads,
                                              bool causal,
                                              ::brotensor::Tensor& O) {
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D)
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: K/V shape mismatch");
    if (Wq.rows != D || Wq.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: Wq/Wo shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: num_heads must divide D");
    ensure_f32(O, Lq, D);
    if (Lq == 0 || Lk == 0 || D == 0) return;

    std::vector<float> Qp(static_cast<std::size_t>(Lq) * D);
    std::vector<float> Op(static_cast<std::size_t>(Lq) * D);
    linear_proj(X.host_f32(), Wq.host_f32(), bq ? bq->host_f32() : nullptr,
                Qp.data(), Lq, D, D);
    attention_core(Qp.data(), K.host_f32(), V.host_f32(), d_mask,
                   Lq, Lk, D, num_heads, causal, Op.data(), nullptr);
    linear_proj(Op.data(), Wo.host_f32(), bo ? bo->host_f32() : nullptr,
                O.host_f32_mut(), Lq, D, D);
}

// ─── flash_attention_qkvo_forward ──────────────────────────────────────────

void flash_attention_qkvo_forward(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor* Ctx,
                                  const ::brotensor::Tensor& Wq,
                                  const ::brotensor::Tensor* bq,
                                  const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor* bk,
                                  const ::brotensor::Tensor& Wv,
                                  const ::brotensor::Tensor* bv,
                                  const ::brotensor::Tensor& Wo,
                                  const ::brotensor::Tensor* bo,
                                  const float* d_mask,
                                  int num_heads,
                                  bool causal,
                                  ::brotensor::Tensor& O) {
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk    = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D)
        throw std::runtime_error("flash_attention_qkvo_forward: shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error("flash_attention_qkvo_forward: num_heads must divide D");
    ensure_f32(O, Lq, D);
    if (Lq == 0 || Lk == 0 || D == 0) return;

    std::vector<float> Qp(static_cast<std::size_t>(Lq) * D);
    std::vector<float> Kp(static_cast<std::size_t>(Lk) * D);
    std::vector<float> Vp(static_cast<std::size_t>(Lk) * D);
    std::vector<float> Op(static_cast<std::size_t>(Lq) * D);

    linear_proj(X.host_f32(), Wq.host_f32(), bq ? bq->host_f32() : nullptr,
                Qp.data(), Lq, D, D);
    linear_proj(kv_src.host_f32(), Wk.host_f32(),
                bk ? bk->host_f32() : nullptr, Kp.data(), Lk, D_ctx, D);
    linear_proj(kv_src.host_f32(), Wv.host_f32(),
                bv ? bv->host_f32() : nullptr, Vp.data(), Lk, D_ctx, D);
    attention_core(Qp.data(), Kp.data(), Vp.data(), d_mask,
                   Lq, Lk, D, num_heads, causal, Op.data(), nullptr);
    linear_proj(Op.data(), Wo.host_f32(), bo ? bo->host_f32() : nullptr,
                O.host_f32_mut(), Lq, D, D);
}

// ─── flash_attention_backward ──────────────────────────────────────────────

void flash_attention_backward(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& O,
                              const ::brotensor::Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              ::brotensor::Tensor& dQ,
                              ::brotensor::Tensor& dK,
                              ::brotensor::Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry.
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk)
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    if (dO.rows != Lq || dO.cols != D)
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    if (causal && Lq != Lk)
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    ensure_f32(dQ, Lq, D);
    ensure_f32(dK, Lk, D);
    ensure_f32(dV, Lk, D);
    if (Lq == 0 || Lk == 0 || D == 0) return;

    attention_core_backward(Q.host_f32(), K.host_f32(), V.host_f32(),
                            dO.host_f32(), d_mask, Lq, Lk, D, num_heads,
                            causal, dQ.host_f32_mut(), dK.host_f32_mut(),
                            dV.host_f32_mut());
}

// ─── flash_attention_varlen_backward ───────────────────────────────────────
//
// Per-sequence backward over the packed (total_tokens, D) layout. Same
// recompute math as flash_attention_backward (attention_core_backward) run
// once per sequence on its [cu_seqlens_q[b], cu_seqlens_q[b+1]) Q slice and
// [cu_seqlens_k[b], cu_seqlens_k[b+1]) K/V slice. dQ/dK/dV are OVERWRITTEN —
// attention_core_backward zeros its per-call output region, and rows outside
// any sequence's range (which can't happen for a well-formed cu_seqlens) plus
// rows whose K range is empty are explicitly zeroed here so the contract
// holds globally.
void flash_attention_varlen_backward(const ::brotensor::Tensor& Q,
                                     const ::brotensor::Tensor& K,
                                     const ::brotensor::Tensor& V,
                                     const ::brotensor::Tensor& O,
                                     const ::brotensor::Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size,
                                     int /*max_seqlen_q*/,
                                     int /*max_seqlen_k*/,
                                     int num_heads,
                                     int head_dim,
                                     bool causal,
                                     ::brotensor::Tensor& dQ,
                                     ::brotensor::Tensor& dK,
                                     ::brotensor::Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry.
    const int total_q = Q.rows;
    const int total_k = K.rows;
    const int D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k)
        throw std::runtime_error("flash_attention_varlen_backward: Q/K/V shape mismatch");
    if (dO.rows != total_q || dO.cols != D)
        throw std::runtime_error("flash_attention_varlen_backward: dO shape mismatch");
    if (num_heads <= 0 || head_dim <= 0)
        throw std::runtime_error("flash_attention_varlen_backward: num_heads/head_dim must be positive");
    if (batch_size < 0)
        throw std::runtime_error("flash_attention_varlen_backward: batch_size must be non-negative");
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k))
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q/k required when batch_size > 0");
    ensure_f32(dQ, total_q, D);
    ensure_f32(dK, total_k, D);
    ensure_f32(dV, total_k, D);
    if (total_q == 0 && total_k == 0) return;
    if (D == 0) return;
    if (cu_seqlens_q && cu_seqlens_q[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[0] must be 0");
    if (cu_seqlens_k && cu_seqlens_k[0] != 0)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[0] must be 0");
    if (batch_size > 0) {
        if (cu_seqlens_q[batch_size] != total_q)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[B] != total_tokens_q");
        if (cu_seqlens_k[batch_size] != total_k)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[B] != total_tokens_k");
    }

    // Zero all grads upfront so any rows not covered by a sequence (or covered
    // by an empty-K sequence) end up at exactly 0.0f without depending on the
    // per-sequence path to touch them.
    float* dQp = dQ.host_f32_mut();
    float* dKp = dK.host_f32_mut();
    float* dVp = dV.host_f32_mut();
    for (std::size_t i = 0; i < static_cast<std::size_t>(total_q) * D; ++i) dQp[i] = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(total_k) * D; ++i) {
        dKp[i] = 0.0f;
        dVp[i] = 0.0f;
    }
    if (batch_size == 0) return;

    const float* Qp  = Q.host_f32();
    const float* Kp  = K.host_f32();
    const float* Vp  = V.host_f32();
    const float* dOp = dO.host_f32();

    for (int b = 0; b < batch_size; ++b) {
        const int q_beg = cu_seqlens_q[b];
        const int q_end = cu_seqlens_q[b + 1];
        const int k_beg = cu_seqlens_k[b];
        const int k_end = cu_seqlens_k[b + 1];
        const int Lq = q_end - q_beg;
        const int Lk = k_end - k_beg;
        if (Lq < 0 || Lk < 0)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens must be non-decreasing");
        if (causal && Lq != Lk)
            throw std::runtime_error("flash_attention_varlen_backward: causal requires per-sequence Lq == Lk");
        if (Lq == 0 || Lk == 0) continue;  // grad rows already zero.

        attention_core_backward(
            Qp  + static_cast<std::size_t>(q_beg) * D,
            Kp  + static_cast<std::size_t>(k_beg) * D,
            Vp  + static_cast<std::size_t>(k_beg) * D,
            dOp + static_cast<std::size_t>(q_beg) * D,
            /*mask=*/nullptr,
            Lq, Lk, D, num_heads, causal,
            dQp + static_cast<std::size_t>(q_beg) * D,
            dKp + static_cast<std::size_t>(k_beg) * D,
            dVp + static_cast<std::size_t>(k_beg) * D);
    }
}

// ─── flash_attention_qkvo_backward ─────────────────────────────────────────

void flash_attention_qkvo_backward(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor* Ctx,
                                   const ::brotensor::Tensor& Wq,
                                   const ::brotensor::Tensor* bq,
                                   const ::brotensor::Tensor& Wk,
                                   const ::brotensor::Tensor* bk,
                                   const ::brotensor::Tensor& Wv,
                                   const ::brotensor::Tensor* bv,
                                   const ::brotensor::Tensor& Wo,
                                   const ::brotensor::Tensor* bo,
                                   const float* d_mask,
                                   int num_heads,
                                   bool causal,
                                   const ::brotensor::Tensor& dO,
                                   ::brotensor::Tensor& dX,
                                   ::brotensor::Tensor* dCtx,
                                   ::brotensor::Tensor& dWq,
                                   ::brotensor::Tensor* dbq,
                                   ::brotensor::Tensor& dWk,
                                   ::brotensor::Tensor* dbk,
                                   ::brotensor::Tensor& dWv,
                                   ::brotensor::Tensor* dbv,
                                   ::brotensor::Tensor& dWo,
                                   ::brotensor::Tensor* dbo) {
    const bool self_attn = (Ctx == nullptr);
    if (self_attn && dCtx != nullptr)
        throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be null when Ctx is null");
    if (!self_attn && dCtx == nullptr)
        throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be non-null when Ctx is non-null");
    auto bias_ok = [](const Tensor* b, const Tensor* db) {
        return static_cast<bool>(b) == static_cast<bool>(db);
    };
    if (!bias_ok(bq, dbq) || !bias_ok(bk, dbk) ||
        !bias_ok(bv, dbv) || !bias_ok(bo, dbo))
        throw std::runtime_error("flash_attention_qkvo_backward: bias/grad-bias presence mismatch");

    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk    = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D)
        throw std::runtime_error("flash_attention_qkvo_backward: shape mismatch");
    if (dO.rows != Lq || dO.cols != D)
        throw std::runtime_error("flash_attention_qkvo_backward: dO shape mismatch");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error("flash_attention_qkvo_backward: num_heads must divide D");
    if (causal && Lq != Lk)
        throw std::runtime_error("flash_attention_qkvo_backward: causal requires Lq == Lk");

    ensure_f32(dX, Lq, D);
    dX.zero();
    if (!self_attn) {
        ensure_f32(*dCtx, Lk, D_ctx);
        dCtx->zero();
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    const float* Xp  = X.host_f32();
    const float* kvp = kv_src.host_f32();

    // ── 1. Recompute forward projections. ──
    std::vector<float> Q(static_cast<std::size_t>(Lq) * D);
    std::vector<float> K(static_cast<std::size_t>(Lk) * D);
    std::vector<float> V(static_cast<std::size_t>(Lk) * D);
    linear_proj(Xp,  Wq.host_f32(), bq ? bq->host_f32() : nullptr,
                Q.data(), Lq, D, D);
    linear_proj(kvp, Wk.host_f32(), bk ? bk->host_f32() : nullptr,
                K.data(), Lk, D_ctx, D);
    linear_proj(kvp, Wv.host_f32(), bv ? bv->host_f32() : nullptr,
                V.data(), Lk, D_ctx, D);

    // ── 2. Recompute O_attn (post-attention, pre-Wo). ──
    std::vector<float> O_attn(static_cast<std::size_t>(Lq) * D);
    attention_core(Q.data(), K.data(), V.data(), d_mask,
                   Lq, Lk, D, num_heads, causal, O_attn.data(), nullptr);

    // ── 3. Wo backward: dO_attn = dO·Wo, dWo += dO^T·O_attn, dbo += colsum. ──
    std::vector<float> dO_attn(static_cast<std::size_t>(Lq) * D, 0.0f);
    linear_proj_backward(O_attn.data(), Wo.host_f32(), dO.host_f32(),
                         dO_attn.data(), dWo.host_f32_mut(),
                         dbo ? dbo->host_f32_mut() : nullptr,
                         Lq, D, D);

    // ── 4. Attention core backward → dQ, dK, dV. ──
    std::vector<float> dQ(static_cast<std::size_t>(Lq) * D);
    std::vector<float> dK(static_cast<std::size_t>(Lk) * D);
    std::vector<float> dV(static_cast<std::size_t>(Lk) * D);
    attention_core_backward(Q.data(), K.data(), V.data(), dO_attn.data(),
                            d_mask, Lq, Lk, D, num_heads, causal,
                            dQ.data(), dK.data(), dV.data());

    // ── 5. Q/K/V projection backward. ──
    //   dX accumulates the Q path (and, for self-attn, the K and V paths).
    //   dWq/dWk/dWv and dbq/dbk/dbv accumulate.
    float* dXp = dX.host_f32_mut();
    linear_proj_backward(Xp, Wq.host_f32(), dQ.data(), dXp,
                         dWq.host_f32_mut(),
                         dbq ? dbq->host_f32_mut() : nullptr,
                         Lq, D, D);
    if (self_attn) {
        linear_proj_backward(Xp, Wk.host_f32(), dK.data(), dXp,
                             dWk.host_f32_mut(),
                             dbk ? dbk->host_f32_mut() : nullptr,
                             Lk, D_ctx, D);
        linear_proj_backward(Xp, Wv.host_f32(), dV.data(), dXp,
                             dWv.host_f32_mut(),
                             dbv ? dbv->host_f32_mut() : nullptr,
                             Lk, D_ctx, D);
    } else {
        float* dCp = dCtx->host_f32_mut();
        linear_proj_backward(kvp, Wk.host_f32(), dK.data(), dCp,
                             dWk.host_f32_mut(),
                             dbk ? dbk->host_f32_mut() : nullptr,
                             Lk, D_ctx, D);
        linear_proj_backward(kvp, Wv.host_f32(), dV.data(), dCp,
                             dWv.host_f32_mut(),
                             dbv ? dbv->host_f32_mut() : nullptr,
                             Lk, D_ctx, D);
    }
}

} // namespace brotensor::detail::cpu
