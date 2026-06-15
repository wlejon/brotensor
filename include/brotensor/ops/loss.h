#pragma once

// brotensor ops/loss.h — Losses: softmax, softmax+xent, MSE, BCE (incl. fused/batched).

#include "../tensor.h"
#include <cstdint>

namespace brotensor {

// ─── Softmax ───────────────────────────────────────────────────────────────

// Numerically stable softmax over a flat length-N vector.
//   logits, probs: (N,1) or (1,N), treated flat; probs resized to match.
//   mask: optional N-float device pointer (1 valid / 0 invalid); may be null.
//         Invalid positions contribute 0 to the normaliser and receive 0 in
//         probs. Caller guarantees >=1 valid entry when masking.
void softmax_forward(const Tensor& logits, Tensor& probs,
                     const float* mask = nullptr);


// Row-batched stable softmax: Y[r, :] = softmax(X[r, :]) over `cols`, for each
// of `rows` independent rows (X/Y carry rows*cols elements, row-major). One
// kernel launch covers all rows — the inference primitive for attention score
// matrices, replacing a per-row softmax_forward loop. May be in-place (Y == X).
void softmax_rows_forward(const Tensor& X, Tensor& Y, int rows, int cols);


// Full-Jacobian softmax backward:
//   dLogits[i] = sum_j dProbs[j] * probs[j] * (delta_ij - probs[i]).
// All length-N; dLogits resized to match.
void softmax_backward(const Tensor& probs, const Tensor& dProbs,
                      Tensor& dLogits);


// Vector MSE: returns mean((pred-target)^2) over the N flat elements.
// pred and target are length-N flat tensors (any 2D shape with N elements).
float mse_vec_forward(const Tensor& pred, const Tensor& target);


// Backward of mse_vec: dPred[i] = (2/N)*(pred[i]-target[i]), overwritten.
void mse_vec_backward(const Tensor& pred, const Tensor& target,
                      Tensor& dPred);


// Scalar value-head MSE: pred and target are size 1. Returns 0.5*(pred-target)^2
// and sets dPred = pred-target. CPU-only; GPU paths use mse_vec_per_sample on
// (B,1) tensors instead.
float mse_scalar(float pred, float target, float& dPred);


// Fused softmax + cross-entropy backward for a one-hot or soft target; the
// gradient collapses to (probs - target). Returns -sum_i target_i*log(p_i).
//   logits, target, probs, dLogits: length-N.
//   mask: optional legal-action mask — illegal entries get 0 in probs and are
//         ignored by the gradient.
// CPU-style arg order; see softmax_xent_fused for the GPU-style ordering.
float softmax_xent(const Tensor& logits, const Tensor& target,
                   Tensor& probs, Tensor& dLogits,
                   const float* mask = nullptr);


// Pointer/length form of softmax_xent over n contiguous floats. Lets callers
// apply xent to a segment of a larger buffer without temporary Tensors.
// CPU-only (host pointers); GPU backends throw "not implemented".
float softmax_xent_segment(const float* logits, const float* target,
                           float* probs, float* dLogits,
                           int n, const float* mask = nullptr);


// Fused softmax + cross-entropy (GPU-style ordering of softmax_xent).
//   logits, target: length-N (target soft or one-hot).
//   d_mask: optional length-N mask (1 valid / 0 invalid); may be null.
//   probs:   length-N softmax over valid entries (0 on masked); resized.
//   dLogits: length-N (probs - target on valid, 0 on masked); resized.
// Returns -sum_i mask[i]*target[i]*log(max(probs[i],1e-12)). Caller
// guarantees >=1 valid entry under the mask.
float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask,
                         Tensor& probs, Tensor& dLogits);


// ─── Batched per-sample loss kernels ───────────────────────────────────────

// Per-sample MSE matching mse_scalar (loss 0.5*d^2, dPred = d).
//   pred, target, dPred, loss_per_sample: (B,1).
//   dPred = pred-target;  loss_per_sample = 0.5*(pred-target)^2. Both overwritten.
void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample);


// Batched fused softmax + cross-entropy over (sample, head) tiles. For each
// row b and head h, runs softmax-xent on the slice
//   [d_head_offsets[h], d_head_offsets[h+1]) of row b:
//   probs_BL[b,slice]   = softmax(logits_BL[b,slice])  (0 on masked)
//   dLogits_BL[b,slice] = probs - target               (0 on masked)
//   loss_per_sample[b]  = sum_h -sum_{valid} target*log p
//   logits_BL, target_BL, probs_BL, dLogits_BL: (B, n_act_total).
//   d_mask_BL: optional (B, n_act_total) mask (1 valid / 0 invalid).
//   d_head_offsets: device int*, length n_heads+1, cumulative.
//   loss_per_sample: (B,1), overwritten.
// The caller applies any mean-over-heads scaling — loss and dLogits are not
// divided by n_heads.
void softmax_xent_fused_batched(const Tensor& logits_BL,
                                const Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets,
                                int n_heads,
                                Tensor& probs_BL,
                                Tensor& dLogits_BL,
                                Tensor& loss_per_sample);


// Batched fused sigmoid + binary cross-entropy with per-sample loss reduction.
// Numerically stable via softplus:
//   loss(b,i) = w*y*softplus(-z) + (1-y)*softplus(z),  w = pos_weight,  s = sigmoid(z)
//   dLogits   = s*(w*y + 1 - y) - w*y    (reduces to s - y when w == 1)
//   probs_BL[b,i] = s on valid, 0 on masked.
//   logits_BL, target_BL, probs_BL, dLogits_BL: (B, L).
//   d_mask_BL: optional (B, L) mask (1 valid / 0 invalid); masked entries get
//              probs=0, dLogits=0, and contribute 0 to loss.
//   loss_per_sample: (B, 1), overwritten with sum-over-L per-element loss.
// pos_weight == 1.0f gives standard unweighted BCE.
void bce_with_logits_fused_batched(const Tensor& logits_BL,
                                   const Tensor& target_BL,
                                   const float* d_mask_BL,
                                   float pos_weight,
                                   Tensor& probs_BL,
                                   Tensor& dLogits_BL,
                                   Tensor& loss_per_sample);

}  // namespace brotensor
