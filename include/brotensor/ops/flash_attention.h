#pragma once

// brotensor ops/flash_attention.h — Flash-attention family: tiled/windowed/varlen/qkvo/decode + KV cache.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// Flash-attention-style fused attention (FP16, inference-only). Q, K, V are
// already projected. Tiled online softmax over Lk (no Lk-long materialisation);
// FP32 accumulation.
//   Q: (Lq,D); K, V: (Lk,D) — FP16.
//   d_mask: optional length-Lk FP32 mask (1 valid / 0 invalid); may be null.
//   num_heads divides D.
//   causal: if true, key k attends to query q only when k <= q (requires
//           Lq == Lk); combines multiplicatively with d_mask.
//   O: (Lq,D) FP16, resized as needed.
void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O);


// Sliding-window causal self-attention (FP32, inference-only) — the local
// attention of streaming neural codecs (e.g. Qwen3-TTS / Mimi) and the
// autoregressive decode step. Q, K, V already projected, (L, num_heads*head_dim).
// Always causal. The Lq queries occupy the last Lq positions of a length-Lk
// causal sequence (q_offset = Lk - Lq): query row r is at absolute position
// r + q_offset and attends keys [max(0, pos-window+1), pos]. window <= 0 means
// unbounded causal — identical to flash_attention_forward with causal=true.
//   - Lq == Lk: self-attention (prefill / codec sliding window).
//   - Lq  < Lk: incremental decode of an Lq-token block over a K/V cache;
//     Lq == 1 with window <= 0 attends every cached key (full cache attention),
//     replacing a varlen call with no cu_seqlens upload. Requires Lk >= Lq.
//   d_mask: optional length-Lk FP32 key mask (1 valid / 0 invalid), combined
//           multiplicatively with the window; may be null.
//   num_heads divides D.  O: (Lq, num_heads*head_dim), resized as needed.
void flash_attention_windowed_forward(const Tensor& Q,
                                      const Tensor& K,
                                      const Tensor& V,
                                      const float* d_mask,
                                      int num_heads,
                                      int window,
                                      Tensor& O);


// Packed variable-length multi-head attention, forward only (Qwen3-VL window
// attention). All sequences in a batch live contiguously in one packed tensor;
// per-sequence boundaries come from `cu_seqlens_*` INT32 prefix-sum buffers of
// length `batch_size + 1`. Sequence b covers Q rows
// [cu_seqlens_q[b], cu_seqlens_q[b+1]) and attends to K/V rows
// [cu_seqlens_k[b], cu_seqlens_k[b+1]); no cross-sequence attention. Mirrors
// flash_attn_varlen_func semantics.
//   Q: (total_tokens_q, num_heads * head_dim) — FP16/BF16/FP32 (GPU) / FP32 (CPU).
//   K, V: (total_tokens_k, num_heads * head_dim) — same packing, num_heads
//         matches Q (no GQA — that's a future cleanup).
//   cu_seqlens_q, cu_seqlens_k: DEVICE pointers (CUDA/Metal device, raw host
//         pointers on CPU), length batch_size + 1, INT32 prefix sums. Same
//         convention as `const float* d_mask` elsewhere — caller owns the
//         buffer; the op does not allocate or copy.
//   max_seqlen_q, max_seqlen_k: bound the longest sequence's length — used by
//         the GPU kernel for block sizing; the CPU impl ignores them.
//   causal: if true, key k in sequence b attends to query q only when
//         (k - cu_seqlens_k[b]) <= (q - cu_seqlens_q[b]); only meaningful when
//         the per-sequence Q and K lengths match.
//   O: (total_tokens_q, num_heads * head_dim) — overwritten, resized + dtype-set.
void flash_attention_varlen_forward(const Tensor& Q,
                                    const Tensor& K,
                                    const Tensor& V,
                                    const int32_t* cu_seqlens_q,
                                    const int32_t* cu_seqlens_k,
                                    int batch_size,
                                    int max_seqlen_q,
                                    int max_seqlen_k,
                                    int num_heads,
                                    int head_dim,
                                    bool causal,
                                    Tensor& O);


// Backward of flash_attention_varlen_forward — packed variable-length attention
// over pre-projected Q/K/V (no projection weights, no biases — projections are
// handled by the caller's linear layer; bias gradients belong to that layer).
// Recompute-based: consumes no forward caches, re-runs the per-sequence
// softmax then reverses it. Matches flash_attention_backward's per-sequence
// math; only the cu_seqlens scatter/gather and per-sequence causal differ.
//   Q: (total_tokens_q, num_heads*head_dim) — FP16/BF16/FP32 (GPU) / FP32 (CPU).
//   K, V: (total_tokens_k, num_heads*head_dim) — same dtype, same packing.
//   O:  (total_tokens_q, ...) forward output — currently unused (kept for API
//       symmetry with flash_attention_backward).
//   dO: (total_tokens_q, ...) upstream gradient, same dtype as Q/K/V.
//   cu_seqlens_q, cu_seqlens_k: DEVICE pointers on GPU, host pointers on CPU,
//       length batch_size + 1, INT32 prefix sums. Same convention as the
//       forward.
//   max_seqlen_q, max_seqlen_k: bound the longest per-sequence length — used
//       by the GPU kernel for block/scratch sizing; the CPU impl ignores them.
//   causal: per-sequence; only meaningful when per-sequence Lq == Lk.
//   dQ: (total_tokens_q, ...); dK, dV: (total_tokens_k, ...) — OVERWRITTEN
//       (resized + dtype-set), matching flash_attention_backward's contract.
// Out-of-range positions (causal-excluded, or in a fully empty K sequence)
// contribute nothing to any gradient. No cross-sequence attention.
void flash_attention_varlen_backward(const Tensor& Q,
                                     const Tensor& K,
                                     const Tensor& V,
                                     const Tensor& O,
                                     const Tensor& dO,
                                     const int32_t* cu_seqlens_q,
                                     const int32_t* cu_seqlens_k,
                                     int batch_size,
                                     int max_seqlen_q,
                                     int max_seqlen_k,
                                     int num_heads,
                                     int head_dim,
                                     bool causal,
                                     Tensor& dQ,
                                     Tensor& dK,
                                     Tensor& dV);


// Flash-attention with QKV and output projections fused at the boundary.
// Projects X->Q, Ctx->K,V (or X->Q,K,V when Ctx is null), runs the tiled core,
// then projects with Wo. FP16 throughout.
//   X:   (Lq,D) FP16, query source.
//   Ctx: (Lk,D_ctx) FP16 or null. Null => self-attention (Ctx<-X). D_ctx may
//        differ from D (e.g. SD1.5 cross-attention).
//   Wq, Wo: (D,D); Wk, Wv: (D,D_ctx) — FP16.
//   bq, bk, bv, bo: optional (D,1) FP16 biases; null to skip.
//   d_mask: optional length-Lk FP32 mask.  num_heads divides D.
//   causal: see flash_attention_forward (typically with Ctx == null).
//   O: (Lq,D) FP16, resized as needed.
void flash_attention_qkvo_forward(const Tensor& X,
                                  const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask,
                                  int num_heads,
                                  bool causal,
                                  Tensor& O);


// Backward of flash_attention_qkvo_forward. Recompute-style: consumes no
// forward caches — re-runs the attention math from the inputs, then reverses
// it. FP16 storage, FP32 accumulation. All shape / dtype / Ctx-null /
// rectangular-Wk-Wv / causal / optional-bias rules match the forward; pass the
// same values.
//   dO:   (Lq,D) FP16 upstream.
//   dX:   (Lq,D) FP16 overwritten. For self-attention (Ctx null) dX absorbs
//         the K/V-projection gradients too: dX = dQ.Wq + dK.Wk + dV.Wv.
//   dCtx: (Lk,D_ctx) FP16 overwritten; must be null iff Ctx is null. For
//         cross-attention dCtx = dK.Wk + dV.Wv.
//   dWq, dWo: (D,D); dWk, dWv: (D,D_ctx) — FP16, accumulated (caller zeros).
//   dbq, dbk, dbv, dbo: (D,1) FP16, accumulated iff the matching forward bias
//         was non-null; pass null otherwise (the null/non-null symmetry must
//         be exact — a mismatch is rejected).
// Causal- and mask-excluded positions contribute nothing to any gradient.
void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask,
    int num_heads,
    bool causal,
    const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo);


// Backward of flash_attention_forward — bare attention core, no projection
// weights (what LoRA-style adapters need; projections are wrapped externally).
// Recompute-based; FP16 storage, FP32 accumulation.
//   Q: (Lq,D); K, V: (Lk,D) — pre-projected forward inputs, FP16.
//   O: (Lq,D) forward output — currently unused (kept for API symmetry).
//   dO: (Lq,D) FP16 upstream.
//   d_mask: optional length-Lk FP32 mask (null for unmasked); positions with
//           mask[k] <= 0.5 are dropped.
//   num_heads divides D.  causal: match the forward (requires Lq == Lk).
//   dQ: (Lq,D); dK, dV: (Lk,D) — FP16, overwritten (resized + dtype-set).
// Causal- and mask-excluded positions contribute nothing to dQ/dK/dV.
void flash_attention_backward(const Tensor& Q,
                              const Tensor& K,
                              const Tensor& V,
                              const Tensor& O,
                              const Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              Tensor& dQ,
                              Tensor& dK,
                              Tensor& dV);


// Project a context tensor through Wk/Wv into the exact (Lk,D) FP16 K/V buffers
// flash_attention_forward consumes. Used to precompute cross-attention K/V once
// per generate() (text context is fixed across denoising steps). Numerically
// identical to the K/V projection stage of flash_attention_qkvo_forward.
//   ctx: (Lk,D_ctx) FP16.  Wk, Wv: (D,D_ctx) FP16.  bk, bv: (D,1) FP16 or null.
//   K_out, V_out: (Lk,D) FP16, resized as needed.
void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out,
                                Tensor& V_out);


// Like flash_attention_qkvo_forward but K and V are already projected by the
// caller (typically via flash_attention_project_kv). Projects X->Q with Wq/bq,
// runs the tiled core against the supplied K/V, applies Wo/bo. FP16 throughout.
//   X: (Lq,D); K, V: (Lk,D) — FP16.  Wq, Wo: (D,D) FP16; bq, bo: (D,1) FP16 or null.
//   d_mask: optional length-Lk FP32 mask.  num_heads divides D.
//   causal: see flash_attention_forward.  O: (Lq,D) FP16, resized as needed.
void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                              const Tensor& K,
                                              const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask,
                                              int num_heads,
                                              bool causal,
                                              Tensor& O);


// KV-cache append (FP16): copy K_new, V_new into rows [cur_len, cur_len+L_new)
// of K_cache, V_cache.
//   K_new, V_new: (L_new,D) FP16.  K_cache, V_cache: (L_max,D) FP16 — must be
//   pre-sized (not resized); cur_len + L_new <= L_max.
void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache);


// Causal flash-attention against a partially-filled KV cache (FP16, fwd-only).
// Runs the tiled core against rows [0, valid_len) of the caches. Query position
// p_q = (valid_len - L_q) + i attends to cache positions [0, p_q].
//   Q: (L_q, num_q_heads*head_dim) FP16 — L_q == 1 for token-by-token decode,
//       L_q > 1 supported.
//   K_cache, V_cache: (L_max, num_kv_heads*head_dim) FP16 — only rows
//       [0, valid_len) are read.
//   valid_len >= L_q.  num_kv_heads must divide num_q_heads (GQA); each KV head
//       serves num_q_heads/num_kv_heads consecutive query heads. num_kv_heads ==
//       num_q_heads is plain MHA. head_dim = Q.cols/num_q_heads must equal
//       K_cache.cols/num_kv_heads.
//   O: (L_q, num_q_heads*head_dim) FP16, resized as needed.
// CUDA and Metal currently support num_kv_heads == num_q_heads only and throw
// "brotensor: flash_attention_decode: GQA not yet implemented on <backend>"
// otherwise; CPU supports GQA fully.
void flash_attention_decode(const Tensor& Q,
                           const Tensor& K_cache, const Tensor& V_cache,
                           int valid_len, int num_q_heads, int num_kv_heads,
                           Tensor& O);


// Back-compat overload: num_kv_heads defaults to num_heads (plain MHA).
inline void flash_attention_decode(const Tensor& Q,
                                   const Tensor& K_cache, const Tensor& V_cache,
                                   int valid_len, int num_heads, Tensor& O) {
    flash_attention_decode(Q, K_cache, V_cache, valid_len,
                           num_heads, /*num_kv_heads=*/num_heads, O);
}


// ─── W8A16 flash-attention variants ────────────────────────────────────────
//
// Same composition as flash_attention_project_kv /
// flash_attention_q_with_kv_cached_forward / flash_attention_qkvo_forward, but
// every projection takes an INT8 weight + per-output-row FP32 scale instead of
// an FP16 weight. The attention core stays FP16 (activations are never
// quantised). Each W*_int8 is (D, in_dim) with a matching scale (D,1); biases
// stay FP16 (D,1), optional. Masks, causal flag, and num_heads match the FP16
// versions.
void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8,
                                           const Tensor& sk,
                                           const Tensor* bk,
                                           const Tensor& Wv_int8,
                                           const Tensor& sv,
                                           const Tensor* bv,
                                           Tensor& K_out,
                                           Tensor& V_out);


void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X,
                                                 const Tensor& K,
                                                 const Tensor& V,
                                                 const Tensor& Wq_int8,
                                                 const Tensor& sq,
                                                 const Tensor* bq,
                                                 const Tensor& Wo_int8,
                                                 const Tensor& so,
                                                 const Tensor* bo,
                                                 const float* d_mask,
                                                 int num_heads,
                                                 bool causal,
                                                 Tensor& O);


void flash_attention_qkvo_int8w_fp16(const Tensor& X,
                                     const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask,
                                     int num_heads,
                                     bool causal,
                                     Tensor& O);

}  // namespace brotensor
