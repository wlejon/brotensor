#pragma once

// brotensor ops/attention.h — Attention (non-flash): single-head, MHA, self/cross, bias, decomposed rel-pos, masks.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// Build a slot-validity mask on-device. For k in [0, K):
//   mask[k] = (x[offset + k*stride] > 0.5f) ? 1.0f : 0.0f
// mask resized to (K, 1).
void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask);


// Single-head scaled dot-product self-attention. Square (D,D) projections,
// no biases.
//   X: (N,D).  Wq, Wk, Wv, Wo: each (D,D).
//   d_mask: optional length-N device mask (1 valid / 0 invalid); may be null.
//           Invalid keys are excluded from the softmax denominator; invalid
//           query rows produce zero output.
//   O: (N,D) output, resized if mis-shaped.
//   Backward caches (out-params): Q, K, V each (N,D); Attn (N,N) post-softmax
//   weights; Y_pre_Wo (N,D) = Attn @ V before the output projection.
void attention_forward(const Tensor& X,
                       const Tensor& Wq, const Tensor& Wk,
                       const Tensor& Wv, const Tensor& Wo,
                       const float* d_mask,
                       Tensor& Q, Tensor& K, Tensor& V,
                       Tensor& Attn, Tensor& Y_pre_Wo,
                       Tensor& O);


// Backward of attention_forward.
//   dO: (N,D) upstream.  X, Q, K, V, Attn, Y_pre_Wo: forward caches.
//   Wq, Wk, Wv, Wo: (D,D) forward weights.  d_mask: as forward (or null).
//   dX: (N,D) overwritten.  dWq, dWk, dWv, dWo: (D,D) accumulated — caller zeros.
void attention_backward(const Tensor& dO,
                        const Tensor& X,
                        const Tensor& Q, const Tensor& K,
                        const Tensor& V, const Tensor& Attn,
                        const Tensor& Y_pre_Wo,
                        const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo,
                        const float* d_mask,
                        Tensor& dX,
                        Tensor& dWq, Tensor& dWk,
                        Tensor& dWv, Tensor& dWo);


// ─── Multi-head self-attention ─────────────────────────────────────────────

// Multi-head scaled dot-product self-attention. Square (D,D) projections,
// split into num_heads heads of head_dim = D / num_heads; num_heads must
// divide D.
//   X: (K,D).  Wq, Wk, Wv, Wo: each (D,D).
//   bq, bk, bv, bo: optional length-D bias vectors (any shape with D
//                   elements, FP32). Added row-wise after the matching
//                   projection: Q/K/V get bq/bk/bv post-projection, O gets
//                   bo post-Wo. Any of the four may be null to skip that
//                   bias term.
//   d_mask: optional length-K device mask (1 valid / 0 invalid); may be null.
//           Same semantics as single-head attention.
//   O: (K,D) output, resized if mis-shaped.
//   Backward caches (out-params, resized if mis-shaped): Qh, Kh, Vh
//   (num_heads*K, head_dim) with head h in rows [h*K, (h+1)*K); Attnh
//   (num_heads*K, K) per-head softmax weights; Yconcat (K,D) pre-Wo concat
//   (does NOT include bo — bo is folded into O directly).
void mha_forward(const Tensor& X,
                 const Tensor& Wq, const Tensor& Wk,
                 const Tensor& Wv, const Tensor& Wo,
                 const Tensor* bq, const Tensor* bk,
                 const Tensor* bv, const Tensor* bo,
                 const float* d_mask,
                 int num_heads,
                 Tensor& Qh, Tensor& Kh, Tensor& Vh,
                 Tensor& Attnh, Tensor& Yconcat,
                 Tensor& O);


// Bias-less convenience overload — forwards to the bias-aware mha_forward
// with bq/bk/bv/bo == nullptr. Preserves the original call shape so existing
// callers don't change.
inline void mha_forward(const Tensor& X,
                        const Tensor& Wq, const Tensor& Wk,
                        const Tensor& Wv, const Tensor& Wo,
                        const float* d_mask,
                        int num_heads,
                        Tensor& Qh, Tensor& Kh, Tensor& Vh,
                        Tensor& Attnh, Tensor& Yconcat,
                        Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo,
                nullptr, nullptr, nullptr, nullptr,
                d_mask, num_heads,
                Qh, Kh, Vh, Attnh, Yconcat, O);
}


// Backward of mha_forward.
//   dO: (K,D) upstream.  X, Qh, Kh, Vh, Attnh, Yconcat: forward caches.
//   Wq, Wk, Wv, Wo: (D,D) forward weights.  d_mask: as forward (or null).
//   num_heads must match forward.
//   dX: (K,D) overwritten.  dWq, dWk, dWv, dWo: (D,D) accumulated — caller zeros.
//   dbq, dbk, dbv, dbo: optional length-D bias gradients, accumulated
//                       (caller zeros). Pass null to skip — must match the
//                       null/non-null pattern of the forward biases.
void mha_backward(const Tensor& dO,
                  const Tensor& X,
                  const Tensor& Qh, const Tensor& Kh,
                  const Tensor& Vh, const Tensor& Attnh,
                  const Tensor& Yconcat,
                  const Tensor& Wq, const Tensor& Wk,
                  const Tensor& Wv, const Tensor& Wo,
                  const float* d_mask,
                  int num_heads,
                  Tensor& dX,
                  Tensor& dWq, Tensor& dWk,
                  Tensor& dWv, Tensor& dWo,
                  Tensor* dbq = nullptr, Tensor* dbk = nullptr,
                  Tensor* dbv = nullptr, Tensor* dbo = nullptr);


// ─── Attention: cross / self / flash ───────────────────────────────────────

// Causal mask helper: fills the length-L FP32 buffer for query row q,
//   mask[k] = (k <= q) ? 1.0f : 0.0f
// resized to (L,1) if mis-shaped. The attention kernels consume a per-row
// length-Lk mask; for fully causal self-attention launch attention per query.
void build_causal_mask_row(int L, int q, Tensor& mask);


// Cross-attention: like mha_forward but K and V are projected from a separate
// context tensor. Dispatched on X.dtype:
//   FP16 — flash-attention inference path; caches not exposed (use
//          cross_attention_forward_train if you need them).
//   FP32 — training-aware path (allocates scratch caches internally).
//   X:   (Lq,D)      query input.
//   Ctx: (Lk,D_ctx)  key/value input; Lk, D_ctx may differ from Lq, D.
//                    Ctx.dtype must match X.dtype.
//   Wq, Wo: (D,D).  Wk, Wv: (D,D_ctx) (rectangular for cross-attention).
//   d_mask: optional length-Lk FP32 mask (1 valid / 0 invalid); may be null.
//   num_heads divides D.
//   O: (Lq,D) output, same dtype as X, resized if mis-shaped.
void cross_attention_forward(const Tensor& X,
                             const Tensor& Ctx,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             Tensor& O);


// Cross-attention with a head-averaged attention map and an optional
// pre-softmax logit bias. FP16 only, FP32 accumulation, no backward. Same math
// as cross_attention_forward, plus:
//   * if attn_logit_bias is non-null it is added to the scaled QK^T scores
//     before softmax, broadcast across heads;
//   * AttnAvg receives the across-head average of the softmax weights.
//   X: (Lq,D); Ctx: (Lk,D_ctx); Wq, Wo: (D,D); Wk, Wv: (D,D_ctx) — all FP16.
//   d_mask: optional length-Lk FP32 mask (1 valid / 0 invalid); may be null.
//   attn_logit_bias: optional (Lq,Lk) FP32 pre-softmax bias; may be null.
//   num_heads divides D.
//   O: (Lq,D) FP16; AttnAvg: (Lq,Lk) FP16 — both resized + dtype-set if needed.
void cross_attention_forward_with_attn(const Tensor& X,
                                       const Tensor& Ctx,
                                       const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo,
                                       const float* d_mask,
                                       const Tensor* attn_logit_bias,
                                       int num_heads,
                                       Tensor& O,
                                       Tensor& AttnAvg);


// FP32 training-side self-attention forward — thin wrapper over mha_forward
// (the mha case with Ctx == X). All tensors FP32.
//   X, O: (L,D).  Wq, Wk, Wv, Wo: (D,D).
//   d_mask: optional length-L FP32 mask; may be null.  num_heads divides D.
//   Caches (resized if mis-shaped): Qh, Kh, Vh (num_heads*L, D/num_heads);
//   Attnh (num_heads*L, L); Yconcat (L,D).
void self_attention_forward_train(const Tensor& X,
                                  const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo,
                                  const float* d_mask,
                                  int num_heads,
                                  Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                  Tensor& Attnh, Tensor& Yconcat,
                                  Tensor& O);


// FP32 training-side self-attention backward — thin wrapper over mha_backward.
// All tensors FP32.
//   dO: (L,D) upstream.  X, Qh, Kh, Vh, Attnh, Yconcat: forward caches.
//   Wq, Wk, Wv, Wo: (D,D) forward weights.  d_mask: as forward (or null).
//   num_heads must match forward.
//   dX: (L,D) overwritten.  dWq, dWk, dWv, dWo: (D,D) accumulated — caller zeros.
void self_attention_backward(const Tensor& dO,
                             const Tensor& X,
                             const Tensor& Qh, const Tensor& Kh,
                             const Tensor& Vh, const Tensor& Attnh,
                             const Tensor& Yconcat,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             Tensor& dX,
                             Tensor& dWq, Tensor& dWk,
                             Tensor& dWv, Tensor& dWo);


// Per-text-token spatial moments of a cross-attention map. Given Attn (Lq,Lk)
// with Lq = h_lat*w_lat a flattened row-major image-token grid (q = y*w_lat+x),
// for each text token k:
//   mass[k]       = sum_q Attn[q,k]
//   centroid[k,0] = sum_q y(q)*Attn[q,k] / max(mass[k], 1e-8)   (y)
//   centroid[k,1] = sum_q x(q)*Attn[q,k] / max(mass[k], 1e-8)   (x)
//   (centroid set to (0,0) when mass[k] is ~ 0).
//   Attn: (Lq,Lk) FP16.  mass: (Lk,1) FP32.  centroid: (Lk,2) FP32 [y,x].
//   mass and centroid resized if mis-shaped. FP32 reductions over FP16 input.
void attention_token_moments(const Tensor& Attn,
                             int h_lat, int w_lat,
                             Tensor& mass,
                             Tensor& centroid);


// FP32 training-side cross-attention forward. mha_forward math with a separate
// Ctx for K/V and rectangular Wk/Wv. All tensors FP32.
//   X: (Lq,D); Ctx: (Lk,D_ctx); Wq, Wo: (D,D); Wk, Wv: (D,D_ctx).
//   d_mask: optional length-Lk FP32 mask; may be null.  num_heads divides D.
//   Caches (resized if mis-shaped): Qh (num_heads*Lq, D/num_heads);
//   Kh, Vh (num_heads*Lk, D/num_heads); Attnh (num_heads*Lq, Lk);
//   Yconcat (Lq,D).  O: (Lq,D), resized if mis-shaped.
void cross_attention_forward_train(const Tensor& X,
                                   const Tensor& Ctx,
                                   const Tensor& Wq, const Tensor& Wk,
                                   const Tensor& Wv, const Tensor& Wo,
                                   const float* d_mask,
                                   int num_heads,
                                   Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                   Tensor& Attnh, Tensor& Yconcat,
                                   Tensor& O);


// FP32 training-side cross-attention backward.
//   dO: (Lq,D) upstream.  X, Ctx, Qh, Kh, Vh, Attnh, Yconcat: forward caches.
//   Wq, Wo: (D,D); Wk, Wv: (D,D_ctx).  d_mask: as forward (or null).
//   num_heads must match forward.
//   dX: (Lq,D), dCtx: (Lk,D_ctx) — overwritten.
//   dWq, dWo: (D,D); dWk, dWv: (D,D_ctx) — accumulated, caller zeros.
void cross_attention_backward(const Tensor& dO,
                              const Tensor& X,
                              const Tensor& Ctx,
                              const Tensor& Qh, const Tensor& Kh,
                              const Tensor& Vh, const Tensor& Attnh,
                              const Tensor& Yconcat,
                              const Tensor& Wq, const Tensor& Wk,
                              const Tensor& Wv, const Tensor& Wo,
                              const float* d_mask,
                              int num_heads,
                              Tensor& dX,
                              Tensor& dCtx,
                              Tensor& dWq, Tensor& dWk,
                              Tensor& dWv, Tensor& dWo);


// FP16 self-attention — thin wrapper over the cross-attention kernel with
// Ctx = X. Conventions as cross_attention_forward (FP16 path).
//   X, O: (L,D) FP16.  Wq, Wk, Wv, Wo: (D,D) FP16.
//   d_mask: optional length-L FP32 mask; may be null.  num_heads divides D.
//   O resized if mis-shaped.
void self_attention_forward(const Tensor& X,
                            const Tensor& Wq, const Tensor& Wk,
                            const Tensor& Wv, const Tensor& Wo,
                            const float* d_mask,
                            int num_heads,
                            Tensor& O);


// Multi-head self-attention with an optional additive pre-softmax bias. Per
// head h:
//   S[q,k] = scale*(Q_h[q].K_h[k]) + attn_bias[h*L+q, k]
//   O = (softmax_k S) @ V_h, concatenated over heads, projected by Wo.
// The additive bias is the primitive behind T5 relative-position bias and
// ALiBi. `scale` multiplies the raw dot product BEFORE the bias: pass
// 1/sqrt(head_dim) for standard attention, or 1.0 for T5.
//   X, O: (L,D).  Wq, Wk, Wv, Wo: (D,D), same dtype as X.
//   d_mask: optional length-L FP32 key mask (also gates padded query rows);
//           may be null.
//   attn_bias: optional (num_heads*L, L) FP32 — row h*L+q is head h query q's
//              length-L bias. Null => plain scaled self-attention.
//   num_heads divides D.  O resized + dtype-set to match X.
// Dispatched on X.dtype (FP32/FP16/BF16); FP32 math; bias is FP32 on every
// backend. Scores are materialised (L,L) per head — for encoder-length seqs.
void self_attention_bias_forward(const Tensor& X,
                                 const Tensor& Wq, const Tensor& Wk,
                                 const Tensor& Wv, const Tensor& Wo,
                                 const float* d_mask,
                                 const Tensor* attn_bias,
                                 int num_heads, float scale,
                                 Tensor& O);


// Multi-head self-attention with a DECOMPOSED 2D relative-position bias — the
// SAM / ViTDet image-encoder attention (segment_anything add_decomposed_rel_pos).
// A token index t maps to grid coords (t/grid_w, t%grid_w) over a grid_h*grid_w
// patch grid (so X.rows == grid_h*grid_w). Per head, with r_q the projected,
// UNSCALED query:
//   bias[q,k] = r_q . rel_pos_h[(qh-kh)+grid_h-1] + r_q . rel_pos_w[(qw-kw)+grid_w-1]
//   S[q,k]    = scale*(Q[q].K[k]) + bias[q,k]   (scale multiplies the dot only)
//   O = concat_h( softmax_k S @ V_h ) @ Wo
// Unlike self_attention_bias_forward the bias is data-dependent (reads Q) and is
// never materialised as (num_heads*L, L) — it's factored into length-grid_h and
// length-grid_w terms. Windowed blocks call this per window (grid == window);
// global blocks call it once over the full grid.
//   X, O: (L, D).  Wq/Wk/Wv/Wo: (D, D).  bq/bk/bv/bo: optional (D,1), null to skip.
//   rel_pos_h: (2*grid_h-1, head_dim).  rel_pos_w: (2*grid_w-1, head_dim).
//   num_heads divides D.  scale: typically 1/sqrt(head_dim).
// Dispatched on X.dtype; O resized + dtype-set to match X.
void self_attention_decomposed_rel_pos_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale,
        Tensor& O);


// Windowed multi-head self-attention with a decomposed 2D relative-position
// bias — the SAM/ViTDet *windowed* encoder block (segment_anything runs most
// blocks over non-overlapping local windows, a few globally). Splits the
// (grid_h, grid_w) token grid into window x window tiles and runs the
// decomposed-rel-pos attention above INDEPENDENTLY within each tile, sharing
// one set of weights and rel-pos tables. The bottom/right of the grid is
// zero-padded up to a multiple of `window` (SAM's window_partition pad) and the
// padding is cropped back off the output, so grid_h/grid_w need not be
// multiples of `window`. For a grid that is exactly one window this is the
// plain decomposed-rel-pos op.
//   X, O: (grid_h*grid_w, D), token-major (row = h*grid_w + w).
//   Wq/Wk/Wv/Wo: (D,D).  bq/bk/bv/bo: optional (D,1), null to skip.
//   rel_pos_h, rel_pos_w: (2*window-1, head_dim) — sized for the window, not
//                         the full grid.
//   num_heads divides D.  scale: typically 1/sqrt(head_dim).
// Dispatched on X.dtype; O resized + dtype-set to match X.
void self_attention_decomposed_rel_pos_windowed_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        Tensor& O);


// W8A16 variant of self_attention_bias_forward — quantised T5-bias attention.
// Identical math and semantics, but each projection weight is an INT8 (D,D)
// matrix paired with an FP32 (D,1) per-output-row dequant scale (the
// quantize_int8_per_row_host convention). Activations stay FP16; the attention
// core is FP32 internally. GPU-only.
//   X, O: (L,D) FP16.  Wq/Wk/Wv/Wo_int8: (D,D) INT8.  sq/sk/sv/so: (D,1) FP32.
//   d_mask: optional length-L FP32 key mask; may be null.
//   attn_bias: optional (num_heads*L, L) FP32 bias; may be null.
//   num_heads divides D.  scale: as in self_attention_bias_forward.
//   O resized as needed.
void self_attention_bias_int8w_fp16(const Tensor& X,
                                    const Tensor& Wq_int8, const Tensor& sq,
                                    const Tensor& Wk_int8, const Tensor& sk,
                                    const Tensor& Wv_int8, const Tensor& sv,
                                    const Tensor& Wo_int8, const Tensor& so,
                                    const float* d_mask,
                                    const Tensor* attn_bias,
                                    int num_heads, float scale,
                                    Tensor& O);

}  // namespace brotensor
