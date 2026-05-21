#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace brotensor {

// ─── brotensor ops (declarations only) ─────────────────────────────────────
//
// Every op is declared once. Backend dispatch is runtime: each op forwards
// to the registered backend (CPU / CUDA / Metal) for the device its operand
// tensors live on. Shape contracts, accumulation semantics for backward
// (caller zeros dW/dB; op accumulates), and dtype-dispatch rules are
// documented per-op below — they are the contract every backend must honour.
//
// Tensors are row-major (rows, cols) and carry both a Dtype and a Device.
// Output tensors are resized (and dtype-set, where the contract calls for
// that) by the implementation if their shape doesn't match the expected
// output shape — except for accumulation outputs (dW, dB) which the caller
// must size and zero appropriately.
//
// Synchronisation: every op is implicitly on its backend's default queue /
// stream. Synchronisation is the caller's responsibility; use
// `brotensor::sync(device)` before reading results back to host. CPU ops are
// synchronous.

// ─── Subagent 2: dense layers + elementwise activations ────────────────────

// y = W * x + b.
//   W: (out_dim, in_dim)
//   b: (out_dim, 1)         (vector; cols == 1)
//   x: (in_dim, 1)
//   y: (out_dim, 1)         (resized if mis-shaped)
void linear_forward(const Tensor& W, const Tensor& b,
                    const Tensor& x, Tensor& y);

// Backward of linear_forward.
//   W:   (out_dim, in_dim)   (forward weights, read-only)
//   x:   (in_dim, 1)         (forward input, read-only)
//   dY:  (out_dim, 1)        (upstream gradient)
//   dX:  (in_dim, 1)         (output, *overwritten*)
//   dW:  (out_dim, in_dim)   (output, *accumulated into* — caller zeros)
//   dB:  (out_dim, 1)        (output, *accumulated into* — caller zeros)
void linear_backward(const Tensor& W, const Tensor& x,
                     const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB);

// y = max(x, 0). x and y may alias (same buffer) for in-place ReLU.
// Shapes match exactly; y resized if mis-shaped.
void relu_forward(const Tensor& x, Tensor& y);

// dX = dY * (x > 0). dX resized to match x if mis-shaped. dX may alias dY.
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// y = tanh(x). y resized to match x if mis-shaped.
void tanh_forward(const Tensor& x, Tensor& y);

// dX = dY * (1 - y*y). `y` is the cached forward output (NOT raw x).
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX);

// y = 1 / (1 + exp(-x)).
void sigmoid_forward(const Tensor& x, Tensor& y);

// dX = dY * y * (1 - y). `y` is the cached forward output.
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX);

// y[i] += x[i]. y and x must have identical shape.
void add_inplace(Tensor& y, const Tensor& x);

// y[i] += s for all i. Dispatches FP32/FP16 on y.dtype.
void add_scalar_inplace(Tensor& y, float s);

// y[i] *= s for all i. Dispatches FP32/FP16 on y.dtype.
void scale_inplace(Tensor& y, float s);

// y[i] = min(max(y[i], lo), hi). In-place. Dispatches FP32/FP16 on y.dtype.
// Used for VAE output rescale-and-clamp and any saturating epilogue.
void clamp(Tensor& y, float lo, float hi);

// Build a slot-validity mask on-device. For k in [0, K):
//   mask[k] = (x[offset + k*stride] > 0.5f) ? 1.0f : 0.0f
// `mask` is resized to (K, 1). Used by DeepSetsEncoder to avoid a host sync
// when constructing per-slot validity masks for masked_mean_pool_*.
void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask);

// ─── Subagent 3: reductions, norm, attention, optimiser ────────────────────

// Numerically stable softmax over a flat vector of length N = logits.size().
//
//   logits: (N, 1) or (1, N) — treated as flat length-N buffer.
//   probs:  same shape as logits; resized if mis-shaped.
//   mask:   optional device pointer to N floats (1 valid, 0 invalid). May be
//           null. Invalid positions contribute 0 to the normaliser AND
//           receive 0 in `probs`. Caller guarantees at least one valid entry
//           when masking — the kernel does not check.
void softmax_forward(const Tensor& logits, Tensor& probs,
                     const float* mask = nullptr);

// Full Jacobian softmax backward:
//   dLogits[i] = sum_j dProbs[j] * probs[j] * (delta_ij - probs[i]).
// All tensors length-N; dLogits resized to match if mis-shaped.
void softmax_backward(const Tensor& probs, const Tensor& dProbs,
                      Tensor& dLogits);

// LayerNorm forward (single-vector, matches CPU LayerNorm).
//   x:     (N, 1)            input vector
//   gamma: (N, 1)            learnable scale
//   beta:  (N, 1)            learnable shift
//   y:     (N, 1)            output, resized if mis-shaped
//   xhat:  (N, 1)            cached normalised x = (x - mean) * rstd, resized
//   mean_out: scalar host-side cache, written by op
//   rstd_out: scalar host-side cache (1 / sqrt(var + eps)), written by op
//   eps:   variance epsilon, typically 1e-5f
//
// The backward consumes (xhat, gamma, mean, rstd) — the signature here is
// intentionally rich so backward needs no recomputation. Subagent 3 may
// revise these caches (e.g. promote mean/rstd to a tiny Tensor) if it's
// cleaner; document any change here.
void layernorm_forward(const Tensor& x,
                       const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat,
                       float& mean_out, float& rstd_out,
                       float eps);

// LayerNorm backward. Dtype-dispatched (FP32 or FP16); all tensors share dtype.
// Internal accumulation in FP32; for FP16, dGamma/dBeta use an FP32 scratch +
// fold-back epilogue (Bundle 2 pattern) so atomic adds are safe.
//   dY:     (N, 1) upstream
//   xhat:   (N, 1) cached from forward
//   gamma:  (N, 1) forward scale
//   rstd:   scalar from forward
//   dX:     (N, 1) output, overwritten (resized + dtype-set to match dY)
//   dGamma: (N, 1) accumulated into — caller zeros
//   dBeta:  (N, 1) accumulated into — caller zeros
void layernorm_backward(const Tensor& dY, const Tensor& xhat,
                        const Tensor& gamma, float rstd,
                        Tensor& dX,
                        Tensor& dGamma, Tensor& dBeta);

// Single-head scaled dot-product self-attention (mirrors CPU
// ScaledDotProductAttention). All projections are square (D, D), no biases.
//
//   X:  (N, D) input
//   Wq, Wk, Wv, Wo: each (D, D)
//   d_mask: optional device pointer, length N (1 valid, 0 invalid). May be
//           null (all valid). Invalid keys are excluded from the softmax
//           denominator (additive -inf on the score row pre-softmax) and
//           invalid query rows produce zero output. Same semantics as
//           softmax mask + the CPU attention impl.
//   O:  (N, D) output, resized if mis-shaped
//
// Caches needed for backward (subagent 3 chooses representation):
//   Q, K, V: each (N, D)
//   Attn:   (N, N)  post-softmax weights
//   Y_pre_Wo: (N, D)  Attn @ V (before output projection)
// Pass these as out-parameters so backward can consume them.
void attention_forward(const Tensor& X,
                       const Tensor& Wq, const Tensor& Wk,
                       const Tensor& Wv, const Tensor& Wo,
                       const float* d_mask,
                       Tensor& Q, Tensor& K, Tensor& V,
                       Tensor& Attn, Tensor& Y_pre_Wo,
                       Tensor& O);

// Attention backward.
//   dO: (N, D) upstream
//   X, Q, K, V, Attn, Y_pre_Wo: forward caches
//   Wq, Wk, Wv, Wo: forward weights
//   d_mask: same mask used in forward (or null)
//   dX: (N, D) output, overwritten
//   dWq, dWk, dWv, dWo: (D, D) accumulated into — caller zeros
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

// ─── Subagent A: multi-head self-attention ─────────────────────────────────

// Multi-head scaled dot-product self-attention. Mirrors the CPU
// MultiHeadAttention class. Square (D, D) projections internally split into
// h heads of head_dim = D / h. h must divide D.
//
//   X:   (K, D) input
//   Wq, Wk, Wv, Wo: each (D, D). Each Wq/Wk/Wv is treated as h stacked
//                   per-head row slices of shape (head_dim, D).
//   d_mask: optional length-K device pointer (1 valid, 0 invalid). May be
//           null. Same semantics as single-head: invalid keys excluded from
//           softmax denom; invalid query rows produce zero output.
//   num_heads: number of attention heads.
//   O:   (K, D) output, resized if mis-shaped.
//
// Caches for backward (out-parameters; resized if mis-shaped):
//   Qh:    (h * K, head_dim) — head h occupies rows [h*K, (h+1)*K)
//   Kh:    (h * K, head_dim)
//   Vh:    (h * K, head_dim)
//   Attnh: (h * K, K)        — per-head softmax weights, same row partition
//   Yconcat: (K, D)          — pre-Wo concat of per-head outputs
void mha_forward(const Tensor& X,
                 const Tensor& Wq, const Tensor& Wk,
                 const Tensor& Wv, const Tensor& Wo,
                 const float* d_mask,
                 int num_heads,
                 Tensor& Qh, Tensor& Kh, Tensor& Vh,
                 Tensor& Attnh, Tensor& Yconcat,
                 Tensor& O);

// Multi-head attention backward.
//   dO: (K, D) upstream
//   X, Qh, Kh, Vh, Attnh, Yconcat: forward caches
//   Wq, Wk, Wv, Wo: forward weights (each (D, D))
//   d_mask: same mask used in forward (or null)
//   num_heads: must match forward
//   dX: (K, D) output, *overwritten*
//   dWq, dWk, dWv, dWo: (D, D) accumulated into — caller zeros
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
                  Tensor& dWv, Tensor& dWo);

// ─── Subagent C: pooling, losses, embedding, concat ────────────────────────

// Masked mean-pool forward over rows of a (K, D) matrix.
//   X:    (K, D)  input matrix
//   mask: device pointer to K floats (1.0 valid, 0.0 invalid). May be null —
//         null means all rows are valid.
//   y:    (D, 1)  output vector, resized if mis-shaped.
//
// Semantics: y[j] = (1 / num_valid) * sum_{k : mask[k]==1} X[k, j].
// If num_valid == 0 the output is filled with zeros (matching how
// `DeepSetsEncoder` and `SetTransformerEncoder` skip the *= inv step when
// the count is zero).
void masked_mean_pool_forward(const Tensor& X, const float* d_mask,
                              Tensor& y);

// Masked mean-pool backward.
//   dY:   (D, 1) upstream gradient
//   mask: same mask used in forward (or null)
//   K:    number of rows in the original X (we don't carry X around)
//   dX:   (K, D) output, *overwritten* (NOT accumulated). Invalid rows are
//         set to exactly zero. Valid rows receive dY / num_valid.
//         If num_valid == 0, dX is zeroed entirely.
void masked_mean_pool_backward(const Tensor& dY, const float* d_mask,
                               int K, Tensor& dX);

// Vector MSE forward.
//   pred, target: length-N flat tensors (any 2D shape with N elements).
// Returns: scalar loss = mean((pred - target)^2) = (1/N) * sum (p - t)^2.
// (Note: scalar `mse_scalar` is per-scalar 0.5*d^2 with grad = d. We adopt
// MEAN-of-squared-diffs for the vector form because that's the standard
// autoencoder reconstruction loss and decouples the gradient magnitude from
// N. The backward gradient is dPred = (2 / N) * (pred - target).)
float mse_vec_forward(const Tensor& pred, const Tensor& target);

// Vector MSE backward.
//   pred, target: forward inputs
//   dPred: same shape as pred, *overwritten*
//   dPred[i] = (2 / N) * (pred[i] - target[i]).
void mse_vec_backward(const Tensor& pred, const Tensor& target,
                      Tensor& dPred);

// Mean-squared error for scalar value head. pred and target are both size 1.
// Returns 0.5 * (pred - target)^2; dPred = (pred - target).
//
// CPU-only — used by single-sample value-head losses; GPU paths use
// mse_vec_per_sample on (B, 1) tensors instead.
float mse_scalar(float pred, float target, float& dPred);

// Combined softmax + cross-entropy backward for a one-hot or soft target.
// Convenient because the gradient collapses to (p - target). Mask is the
// same legal-action mask — illegal entries are set to 0 in `probs` and the
// gradient ignores them.
//
// Returns scalar loss = -sum_i target_i * log(p_i) (illegal ignored).
//
// CPU-style ordering (logits, target, probs, dLogits, mask). For the
// GPU-style ordering with separate `d_mask` parameter and fused on-device
// loss reduction, see `softmax_xent_fused`.
float softmax_xent(const Tensor& logits, const Tensor& target,
                   Tensor& probs, Tensor& dLogits,
                   const float* mask = nullptr);

// Pointer/length form of softmax_xent. Operates on n contiguous floats
// starting at the supplied pointers. Used by callers that want to apply
// xent to a segment of a larger logit/target buffer (e.g. the per-head
// policy loss in GenericExItTrainer) without copying through temporary
// Tensors. Same return value semantics as softmax_xent.
//
// CPU-only. Callers operating on host pointers should always be on the CPU
// backend; GPU backends throw "not implemented".
float softmax_xent_segment(const float* logits, const float* target,
                           float* probs, float* dLogits,
                           int n, const float* mask = nullptr);

// Fused softmax + cross-entropy, mirroring CPU `softmax_xent_segment`.
//   logits:  length-N
//   target:  length-N (soft or one-hot; values typically in [0,1] summing to
//            1 over valid entries)
//   d_mask:  optional length-N mask (1 valid / 0 invalid). May be null.
//   probs:   length-N output (softmax over valid entries, 0 on invalid).
//            Resized if mis-shaped.
//   dLogits: length-N output (probs - target on valid; 0 on invalid).
//            Resized if mis-shaped.
// Returns: scalar loss = -sum_i (mask[i] ? target[i] * log(max(probs[i],
// 1e-12)) : 0). Caller guarantees at least one valid entry under mask.
float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask,
                         Tensor& probs, Tensor& dLogits);

// Embedding lookup forward.
//   table:    (V, D) embedding matrix (FP32 or FP16).
//   d_idx:    device pointer to B int32 indices, each in [0, V).
//   B:        number of indices (== rows of `out`).
//   out:      (B, D), resized AND dtype-set to match `table` if mis-shaped/typed.
//             out[b, :] = table[d_idx[b], :].
void embedding_lookup_forward(const Tensor& table,
                              const int32_t* d_idx, int B,
                              Tensor& out);

// Embedding lookup backward — scatter-accumulate. Dtype-dispatched (FP32 or
// FP16); dOut and dTable share dtype. For FP16, an FP32 scratch buffer is
// used for the atomicAdds (FP16 atomicAdd is not portable across CUDA
// compute capabilities) and folded into dTable as an FP32-into-FP16 add.
//   dOut:   (B, D) upstream
//   d_idx:  same indices used in forward (length B)
//   B:      number of indices
//   dTable: (V, D), accumulated into (caller zeros). Multiple lookups of the
//           same row sum their grads via atomicAdd.
void embedding_lookup_backward(const Tensor& dOut,
                               const int32_t* d_idx, int B,
                               Tensor& dTable);

// Concatenate flat tensors end-to-end.
//   parts: list of tensors, each treated as a flat buffer of size parts[i]->size().
//   out:   resized to (total, 1) where total = sum of part sizes.
// Layout: out[off_i .. off_i + size_i) = parts[i] flattened.
void concat_rows(const std::vector<const Tensor*>& parts,
                 Tensor& out);

// Inverse of concat_rows: copy disjoint segments of `in` back into the
// flat buffers of `parts`. Each parts[i] is *overwritten* (not accumulated)
// with the corresponding segment of `in`. Sizes of `parts` must be unchanged
// from the concat call. The function assumes parts[i]->size() segments laid
// end-to-end starting at offset 0 in `in`.
void split_rows(const Tensor& in,
                const std::vector<Tensor*>& parts);

// Batched column-block concat. Each part is shape (B, d_i) for the same B;
// out becomes (B, sum_i d_i) with parts laid as column blocks per row:
//   out[b, off_i + j] = parts[i][b, j].
// Implemented via cudaMemcpy2DAsync per part — bandwidth-bound, no kernel
// launches. Use for batched per-row concat in inference.
void concat_batched_rows(const std::vector<const Tensor*>& parts,
                         Tensor& out);

// Channel-axis concat over NCHW tensors. Each part i is shape
// (N, C_i * H * W) (flat NCHW); out becomes (N, sum_i C_i * H * W) with the
// channel blocks regrouped per sample:
//
//   out[n, (off_i + c) * H*W + h*W + w] = parts[i][n, c * H*W + h*W + w]
//
// where off_i = sum_{j < i} C_j.
//
// Implemented via cudaMemcpy2DAsync per part — dtype-dispatched (FP16/FP32),
// bandwidth-bound, no kernel launches. This is the correct U-Net skip-merge
// concat for N >= 1; a flat byte concat (concat_rows) would interleave
// samples incorrectly for N > 1.
//
// C_per_part.size() must equal parts.size(); part i must have size
// N * C_per_part[i] * H * W. All parts share dtype.
void concat_nchw_channels(const std::vector<const Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          Tensor& out);

// Inverse of concat_nchw_channels: copy disjoint channel-axis slices of
// dY back into per-source gradient buffers. Each parts[i] is *overwritten*
// (not accumulated) with the channels [off_i, off_i + C_per_part[i]) of dY,
// where off_i = sum_{j < i} C_per_part[j].
//
//   dY[n, (off_i + c) * H*W + h*W + w] -> parts[i][n, c * H*W + h*W + w]
//
// Dtype-dispatched (FP32 + FP16); all parts are resized AND dtype-set to
// match dY.dtype if mis-shaped/-typed. Implemented via cudaMemcpy2DAsync per
// part — pure bandwidth, no kernel launches.
//
// C_per_part.size() must equal parts.size(); dY.cols must be N * sum(C_per_part) * H * W.
void concat_nchw_channels_backward(const Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts);

// Single-stream device-to-device chunk copy. Copies `n` floats from
// src.data + src_off into dst.data + dst_off. Both tensors are treated as
// flat float buffers regardless of (rows, cols). Async on the default stream.
void copy_d2d(const Tensor& src, int src_off,
              Tensor& dst,       int dst_off,
              int n);

// Dtype cast: dst = src converted to out_dtype. dst is resized to
// (src.rows, src.cols, out_dtype) and ends up on src's device. Supports the
// FP32 <-> FP16 and FP32 <-> BF16 pairs (and a same-dtype passthrough copy);
// other pairs throw. Device-polymorphic — the standard mixed-precision
// primitive (low-precision weight <-> FP32 master copy) for optimizers built
// on brotensor.
void cast(const Tensor& src, Tensor& dst, Dtype out_dtype);

// Inference-only batched LayerNorm forward. Processes R independent rows
// of length D in a single launch (one block per row). Does not cache xhat
// or read mean/rstd back to host — no syncs. Use when backward isn't
// needed; the existing layernorm_forward remains for training.
//   X_RD:   (R, D) input
//   gamma:  (D,) scale
//   beta:   (D,) shift
//   Y_RD:   (R, D), resized if mis-shaped
void layernorm_forward_inference_batched(const Tensor& X_RD,
                                         const Tensor& gamma,
                                         const Tensor& beta,
                                         Tensor& Y_RD,
                                         float eps);

// SGD with momentum, in-place:
//   velocity = momentum * velocity + grad
//   param   -= lr * velocity
// All three tensors must have identical shape. velocity is updated in place;
// caller is responsible for grad zeroing between batches.
void sgd_step(Tensor& param, Tensor& grad, Tensor& velocity,
              float lr, float momentum);

// Adam optimizer step, in-place. Mirrors adam_step_cpu in circuits.h:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   param -= lr * (m / (1 - beta1^step)) / (sqrt(v / (1 - beta2^step)) + eps)
// `step` is a 1-based step counter for bias correction. All four tensors must
// have identical shape.
void adam_step(Tensor& param, const Tensor& grad,
               Tensor& m, Tensor& v,
               float lr, float beta1, float beta2, float eps, int step);

// Deterministic xavier-uniform init for a Linear weight matrix.
// rng_state is a 64-bit splitmix state advanced in place.
//
// CPU-only — weight initialisation happens before training begins, while
// weights still live on the host. Move them to a GPU backend with `to()`
// afterward.
void xavier_init(Tensor& W, uint64_t& rng_state);

// ─── Batched (inference-only) variants ─────────────────────────────────────
//
// These run B independent forward passes in a single kernel launch. They are
// forward-only; backward is not provided. Used by the BatchedInferenceServer
// to amortise per-kernel-launch latency across many concurrent requests.
//
// Layout convention: tensors carrying B rows are shaped (B, D) row-major, so
// row b at columns 0..D-1 holds the b'th sample. This is the natural shape
// for staging samples end-to-end into a flat host buffer and uploading once.

// Y[b, :] = W * X[b, :] + b   for b in [0, B).
//   W:    (out_dim, in_dim)
//   bias: (out_dim, 1)
//   X_BD: (B, in_dim)
//   Y_BD: (B, out_dim) — resized if mis-shaped.
void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD);

// Elementwise ReLU/Tanh over (B, D). Y resized to match X if mis-shaped.
// X and Y may alias.
void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD);
void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD);

// Y[i] += X[i] over (B, D). Identical shape required.
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD);

// ─── Batched (training) backward variants ──────────────────────────────────
//
// Backward partners for the batched-train path used by GenericExItTrainer.
// Match the math of the single-sample versions summed across B.

// Linear backward over a B-row minibatch. Dtype-dispatched (FP32 or FP16);
// all tensors share dtype. For FP16, internal accumulation is FP32 and
// dW/dB use FP32 scratch + fold-back (Bundle 2 pattern).
//   W:    (out_dim, in_dim) — read-only forward weights
//   X_BD: (B, in_dim)       — forward input (cached by caller)
//   dY_BD:(B, out_dim)      — upstream gradient
//   dX_BD:(B, in_dim)       — output, *overwritten* (resized + dtype-set if
//                             mis-shaped/-typed)
//   dW:   (out_dim, in_dim) — *accumulated*; caller zeros before the step
//   dB:   (out_dim, 1)      — *accumulated*; caller zeros before the step
// Math: dX[b] = W^T * dY[b], dW += sum_b dY[b] * X[b]^T, dB += sum_b dY[b].
void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD,
                             Tensor& dW, Tensor& dB);

// Elementwise ReLU/Tanh backward over (B, D). Same shapes throughout.
//   relu:  dX = dY * (X > 0); reads X_BD (the forward input).
//   tanh:  dX = dY * (1 - Y*Y); reads Y_BD (the forward output).
void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD,
                           Tensor& dX_BD);
void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD,
                           Tensor& dX_BD);

// ─── Batched per-sample loss kernels (training) ────────────────────────────
//
// Used by GenericExItTrainer to fuse loss + grad across the whole minibatch
// in a single launch, with no per-sample host roundtrips.

// Per-sample MSE matching CPU `mse_scalar` (loss = 0.5 * d², dPred = d).
//   pred:   (B, 1)
//   target: (B, 1)
//   dPred:  (B, 1) — overwritten with (pred - target)
//   loss_per_sample: (B, 1) — overwritten with 0.5 * (pred - target)^2
void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample);

// Batched fused softmax + cross-entropy across (sample, head) tiles.
//
// For each row b in [0, B) and head h in [0, n_heads), runs a numerically
// stable softmax-xent over the slice
//     [d_head_offsets[h], d_head_offsets[h+1])
// of the b'th row of `logits_BL`. Writes:
//     probs_BL[b, slice]   = softmax(logits_BL[b, slice]) (0 on masked)
//     dLogits_BL[b, slice] = probs - target on valid; 0 on masked
//     loss_per_sample[b]  += sum_h (-sum_{i in slice, valid} target * log p)
// loss_per_sample is *overwritten* before the per-head accumulation begins.
//
//   logits_BL, target_BL, probs_BL, dLogits_BL: (B, n_act_total)
//   d_mask_BL: optional (B, n_act_total) device pointer (1 valid / 0 invalid)
//   d_head_offsets: device int* of length n_heads + 1 (cumulative per spec).
//   loss_per_sample: (B, 1) — overwritten with the sum-over-heads loss.
//
// The caller is responsible for any mean-over-heads reduction on the loss
// (CPU formulation divides by n_heads). The kernel does not scale dLogits
// by 1/n_heads either — the caller applies that with scale_inplace.
void softmax_xent_fused_batched(const Tensor& logits_BL,
                                const Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets,
                                int n_heads,
                                Tensor& probs_BL,
                                Tensor& dLogits_BL,
                                Tensor& loss_per_sample);

// ─── Diffusion / vision ops (FP16, inference-only) ─────────────────────────
//
// These ops are the GPU primitives needed to run a diffusion U-Net + VAE
// end-to-end. They take FP16 tensors (X.dtype == Dtype::FP16) and produce
// FP16 outputs. Internal accumulation is in FP32. NCHW layout: the Tensor
// is treated as a flat buffer; (N, C, H, W) dimensions are passed as
// integer arguments. Output tensors are resized (and dtype-set) by the op.
// No backward; downstream brodiff drives these for inference only.

// 2D convolution, NCHW. Dispatched on X.dtype (FP32 or FP16);
// Wt, bias (if non-null), and Y must all share X.dtype. Internal
// accumulation in FP32 regardless of storage dtype.
//   X:      (N, C_in * H * W)        input
//   Wt:     (C_out, (C_in/groups) * kH * kW)  weights, OIHW filter layout
//   bias:   (C_out, 1)               optional bias, may be null
//   Y:      (N, C_out * H_out * W_out)  output, resized AND dtype-set to
//                                       match X if mis-shaped/-typed.
//   groups: must divide both C_in and C_out. Default 1 (standard conv).
//           Wt shape becomes (C_out, (C_in/groups) * kH * kW). Output channel
//           c_out belongs to group g = c_out / (C_out / groups), and reads
//           only input channels [g*(C_in/groups), (g+1)*(C_in/groups)).
//           groups=1 reduces to the original full-channel convolution.
//           Depthwise is groups == C_in == C_out (Wt shape (C_out, kH*kW),
//           one filter per channel).
// Output dims (standard PyTorch formula):
//   H_out = (H + 2*pad_h - dil_h * (kH - 1) - 1) / stride_h + 1
//   W_out = (W + 2*pad_w - dil_w * (kW - 1) - 1) / stride_w + 1
void conv2d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    Tensor& Y);
// Convenience overload: groups defaults to 1 (full convolution).
inline void conv2d_forward(const Tensor& X,
                           const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           Tensor& Y) {
    conv2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                   /*groups=*/1, Y);
}

// 2D convolution backward w.r.t. input (dX). Mirrors conv2d_forward's
// groups parameter: input channel c_in only sees output channels in its
// own group g = c_in / (C_in / groups). At groups=1 the math reduces to a
// full sum over c_out.
//
// Gather form: one thread per input pixel, iterates the kernel positions
// (kh, kw) and inverts the forward index relation to find which output
// pixel (i_out, j_out) read this input pixel through this kernel tap:
//   i_out = (i + pad_h - dil_h * kh) / stride_h
//   j_out = (j + pad_w - dil_w * kw) / stride_w
// Only valid when divisible by stride and in [0, H_out) × [0, W_out).
// For each valid (kh, kw), accumulate sum over c_out in c_in's group of
// dY[n, c_out, i_out, j_out] * Wt[c_out, c_in_local, kh, kw], where
// c_in_local = c_in - g * (C_in / groups). FP32 accumulator.
// No atomics — each dX pixel is written by exactly one thread.
// Dtype-dispatched (FP32 or FP16); Wt and dY share dtype, dX matches.
//
// All conv hyperparams (H, W, stride/pad/dil, groups) match the forward call.
//
//   Wt:   (C_out, (C_in/groups) * kH * kW)  forward filter, OIHW
//   dY:   (N, C_out * H_out * W_out)        upstream gradient
//   dX:   (N, C_in  * H * W)                output, *overwritten*. Resized AND
//                                           dtype-set to match dY if
//                                           mis-shaped/-typed.
//   groups: must divide both C_in and C_out. Depthwise is groups == C_in == C_out.
void conv2d_backward_input(const Tensor& Wt,
                           const Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           Tensor& dX);
// Convenience overload: groups defaults to 1.
inline void conv2d_backward_input(const Tensor& Wt,
                                  const Tensor& dY,
                                  int N, int C_in, int H, int W,
                                  int C_out, int kH, int kW,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dil_h, int dil_w,
                                  Tensor& dX) {
    conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                          /*groups=*/1, dX);
}

// 2D convolution backward w.r.t. weights (dW). Dtype-dispatched (FP32 or
// FP16); X, dY, dWt all share dtype. Mirrors conv2d_forward's groups
// parameter; for groups > 1 the filter has shape (C_out, (C_in/groups) * kH * kW)
// and each output channel only sees the inputs in its own group.
//
// One thread per (c_out, c_in_local, kh, kw) element of dWt. Each thread iterates
// (n, i_out, j_out) over the output spatial extent + batch, looks up the
// corresponding input pixel from group g = c_out / (C_out/groups) at absolute
// channel c_in = g * (C_in/groups) + c_in_local (skipping OOB reads — treat OOB
// as zero), and accumulates into a single dWt slot. No atomics — each dWt
// element is owned by exactly one thread. FP32 accumulator; the per-thread sum
// is *added* into the caller's dWt (caller is responsible for zeroing first).
// For FP16 storage, an FP32 scratch + fold-back epilogue is used.
//
// Math (Cg_in = C_in/groups, Cg_out = C_out/groups,
//       g = c_out / Cg_out, c_in = g * Cg_in + c_in_local):
//   dWt[c_out, c_in_local, kh, kw] +=
//     sum over (n, i_out, j_out) of
//       dY[n, c_out, i_out, j_out] *
//       X [n, c_in, stride_h*i_out - pad_h + dil_h*kh,
//                    stride_w*j_out - pad_w + dil_w*kw]
//
//   X:    (N, C_in  * H * W)                 forward input
//   dY:   (N, C_out * H_out * W_out)         upstream gradient
//   dWt:  (C_out, (C_in/groups) * kH * kW)   *accumulated into* — caller zeros
//   groups: must divide both C_in and C_out.
// All conv hyperparams match the forward call.
void conv2d_backward_weight(const Tensor& X,
                            const Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            Tensor& dWt);
// Convenience overload: groups defaults to 1.
inline void conv2d_backward_weight(const Tensor& X,
                                   const Tensor& dY,
                                   int N, int C_in, int H, int W,
                                   int C_out, int kH, int kW,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dil_h, int dil_w,
                                   Tensor& dWt) {
    conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                           /*groups=*/1, dWt);
}

// 2D convolution backward w.r.t. bias (dB). Dtype-dispatched (FP32 or FP16);
// dY and dB share dtype. No groups parameter — bias is per-output-channel
// and the dB math is identical regardless of how the spatial conv is grouped.
//
// One block per c_out, parallel reduction across (n, i_out, j_out). FP32
// accumulator; the block-reduced partial sum is added into dB[c_out]. For
// FP16 storage, an FP32 scratch + fold-back epilogue is used.
//
//   dB[c_out] += sum over (n, i_out, j_out) of dY[n, c_out, i_out, j_out].
//
//   dY:  (N, C_out * H_out * W_out)   upstream gradient
//   dB:  (C_out, 1)                    *accumulated into* — caller zeros
void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB);

// GroupNorm forward, NCHW. Dtype-dispatched on X.dtype (FP32 or FP16);
// gamma, beta, and Y all share X.dtype. Internal accumulation in FP32.
//   X:     (N, C * H * W)   input
//   gamma: (C, 1)           per-channel scale (same dtype as X)
//   beta:  (C, 1)           per-channel shift (same dtype as X)
//   Y:     (N, C * H * W)   output, resized AND dtype-set to match X if
//                           mis-shaped/-typed.
//   num_groups must divide C. eps typically 1e-5f. Mean and variance are
//   computed over (C/num_groups, H, W) within each (n, group) tile.
void group_norm_forward(const Tensor& X,
                        const Tensor& gamma,
                        const Tensor& beta,
                        int N, int C, int H, int W,
                        int num_groups,
                        float eps,
                        Tensor& Y);

// GroupNorm backward, NCHW. Dtype-dispatched on X.dtype (FP32 or FP16);
// gamma and dY share X.dtype. Internal accumulation in FP32 regardless of
// storage dtype. Mean and rstd are recomputed per (n, group) tile inside the
// kernel from X (no forward cache is required — GroupNorm tiles are small).
//
// For each (n, g) tile spanning M = (C/num_groups) * H * W elements:
//   rstd = 1 / sqrt(var + eps),   x̂ = (x - mean) * rstd
//   dx̂ = dY * γ_c
//   sum1 = Σ dx̂   over tile
//   sum2 = Σ dx̂ · x̂   over tile
//   dX = rstd * (dx̂ - (sum1 + x̂ · sum2) / M)
// Per-channel grads accumulated across the whole batch:
//   dGamma_c += Σ_{n,h,w} dY * x̂
//   dBeta_c  += Σ_{n,h,w} dY
//
//   X:      (N, C * H * W)   forward input (same dtype as gamma, dY)
//   gamma:  (C, 1)           forward scale
//   dY:     (N, C * H * W)   upstream gradient
//   dX:     (N, C * H * W)   output, *overwritten*. Resized AND dtype-set to
//                            match X if mis-shaped/-typed.
//   dGamma: (C, 1)           *accumulated into* — caller zeros. Same dtype as X.
//   dBeta:  (C, 1)           *accumulated into* — caller zeros. Same dtype as X.
void group_norm_backward(const Tensor& X,
                         const Tensor& gamma,
                         const Tensor& dY,
                         int N, int C, int H, int W,
                         int num_groups,
                         float eps,
                         Tensor& dX,
                         Tensor& dGamma,
                         Tensor& dBeta);

// SiLU / Swish: y = x * sigmoid(x). FP32 and FP16 variants. y resized to
// match x.shape AND x.dtype if mis-shaped/mis-typed. x and y may alias.
void silu_forward(const Tensor& x, Tensor& y);

// SiLU backward. Reads the raw forward input x (NOT the forward output y).
//   dX[i] = dY[i] * sigmoid(x[i]) * (1 + x[i] * (1 - sigmoid(x[i])))
// FP32 and FP16 variants dispatched on x.dtype (FP16 accumulates in FP32).
// dX is resized AND dtype-set to match x if mis-shaped/-typed. dX may alias dY.
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// GELU (tanh approximation, matching PyTorch's `approximate="tanh"`):
//   y = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
// FP32 and FP16 variants dispatched on x.dtype.
void gelu_forward(const Tensor& x, Tensor& y);

// GELU backward (tanh approximation). Reads the raw forward input x.
//   k = sqrt(2/pi); u = k * (x + 0.044715 * x^3); t = tanh(u)
//   du/dx = k * (1 + 3 * 0.044715 * x^2)
//   dy/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * du/dx
//   dX[i] = dY[i] * dy/dx
// FP32 and FP16 dispatch on x.dtype (FP16 accumulates in FP32). dX resized AND
// dtype-set to match x if mis-shaped/-typed. dX may alias dY.
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// Exact GELU (erf formulation, matches PyTorch `torch.nn.functional.gelu`
// with default `approximate="none"` and HuggingFace `diffusers`' default):
//   y = 0.5 * x * (1 + erf(x / sqrt(2)))
// This is the *exact* Gaussian-CDF GELU, distinct from the tanh-approximation
// `gelu_forward`. Provided as a separate op so downstream call sites
// (e.g. brodiffusion's UNet GEGLU FFNs) can swap activations without
// disturbing the existing tanh-approx path. FP32 and FP16 variants dispatched
// on x.dtype (FP16 accumulates in FP32). y resized AND dtype-set to match x
// if mis-shaped/-typed. x and y may alias.
void gelu_exact_forward(const Tensor& x, Tensor& y);

// Exact-GELU backward. Reads the raw forward input x.
//   dy/dx = 0.5 * (1 + erf(x/√2)) + (x / √(2π)) * exp(-x²/2)
//         = 0.5 * (1 + erf(x/√2)) + x * φ(x)
//   dX[i] = dY[i] * dy/dx
// FP32 and FP16 dispatch on x.dtype (FP16 accumulates in FP32). dX resized AND
// dtype-set to match x if mis-shaped/-typed. dX may alias dY.
void gelu_exact_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX);

// QuickGELU: y = x * sigmoid(1.702 * x). Matches OpenAI CLIP's activation
// (used in SD1.5's CLIP ViT-L/14 text encoder). FP32 and FP16 variants
// dispatched on x.dtype. y resized to match x.shape AND x.dtype if
// mis-shaped/-typed. x and y may alias.
void quick_gelu_forward(const Tensor& x, Tensor& y);

// QuickGELU backward. Reads the raw forward input x. Let s = sigmoid(1.702*x).
//   dy/dx = s + x * 1.702 * s * (1 - s)
//   dX[i] = dY[i] * dy/dx
// FP32 and FP16 dispatch on x.dtype (FP16 accumulates in FP32). dX resized AND
// dtype-set to match x if mis-shaped/-typed. dX may alias dY.
void quick_gelu_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX);

// 2x nearest-neighbour upsample over the spatial dims of an NCHW tensor.
// Dtype-dispatched on X.dtype (FP32 or FP16); Y resized AND dtype-set to
// match X if mis-shaped/-typed.
//   X: (N, C * H * W)
//   Y: (N, C * 2H * 2W)
// Each output pixel (i, j) reads X at (i/2, j/2).
void upsample_nearest_2x(const Tensor& X,
                         int N, int C, int H, int W,
                         Tensor& Y);

// 2x bilinear upsample with align_corners=False (PyTorch default for
// interpolate(scale_factor=2)). NCHW. Dtype-dispatched on X.dtype (FP32 or
// FP16); Y resized AND dtype-set to match X if mis-shaped/-typed. Internal
// math in FP32.
void upsample_bilinear_2x(const Tensor& X,
                          int N, int C, int H, int W,
                          Tensor& Y);

// 2x average-pool downsample over NCHW. Stride 2, kernel 2, no padding.
// Dtype-dispatched on X.dtype (FP32 or FP16); Y resized AND dtype-set to
// match X if mis-shaped/-typed. Internal math in FP32.
//   X: (N, C * H * W);  H and W must be even.
//   Y: (N, C * H/2 * W/2)
void downsample_avg_2x(const Tensor& X,
                       int N, int C, int H, int W,
                       Tensor& Y);

// Backward of upsample_nearest_2x. Each input pixel sums the 4 output-pixel
// gradients that copied from it:
//   dX[n,c,i,j] = sum_{a,b in {0,1}} dY[n,c, 2i+a, 2j+b]
// Dispatched FP32/FP16 on dY.dtype; dX resized AND dtype-set to match dY if
// mis-shaped/-typed. Internal accumulation in FP32. One thread per input
// pixel — no atomics.
//   dY: (N, C * 2H * 2W)  upstream gradient
//   N, C, H, W: INPUT (pre-upsample) dims (so output dims are 2H, 2W)
//   dX: (N, C * H * W)    output, *overwritten*
void upsample_nearest_2x_backward(const Tensor& dY,
                                  int N, int C, int H, int W,
                                  Tensor& dX);

// Backward of upsample_bilinear_2x (align_corners=False). Scatters each
// dY[n,c,i_out,j_out] into the 4 input pixels it bilinear-sampled from,
// weighted by the same bilinear weights (replaying the forward's
// y_in=(i_out+0.5)/2-0.5, x_in=(j_out+0.5)/2-0.5 mapping with [0,H-1]x[0,W-1]
// clamping). Dispatched FP32/FP16 on dY.dtype; dX resized AND dtype-set to
// match dY if mis-shaped/-typed. Internal accumulation in FP32.
// Implementation: one thread per output pixel, atomicAdd into dX. FP16 path
// uses an FP32 scratch buffer + fold kernel (FP16 atomicAdd is not portable).
//   dY: (N, C * 2H * 2W)  upstream gradient
//   N, C, H, W: INPUT (pre-upsample) dims
//   dX: (N, C * H * W)    output, *overwritten*
void upsample_bilinear_2x_backward(const Tensor& dY,
                                   int N, int C, int H, int W,
                                   Tensor& dX);

// Backward of downsample_avg_2x. Each input pixel receives 1/4 of the single
// output pixel's gradient that averaged over it:
//   dX[n,c,2*i_out+a, 2*j_out+b] = (1/4) * dY[n,c,i_out,j_out]
// Dispatched FP32/FP16 on dY.dtype; dX resized AND dtype-set to match dY if
// mis-shaped/-typed. Internal accumulation in FP32. One thread per input
// pixel — no atomics. H, W (input dims) must be even.
//   dY: (N, C * H/2 * W/2)  upstream gradient
//   N, C, H, W: INPUT (pre-downsample) dims; H, W even
//   dX: (N, C * H * W)      output, *overwritten*
void downsample_avg_2x_backward(const Tensor& dY,
                                int N, int C, int H, int W,
                                Tensor& dX);

// FP16 batched linear forward, inference-only. Mirrors
// linear_forward_batched but with FP16 storage on X / W / bias / Y.
//   W:    (out_dim, in_dim)  FP16
//   bias: (out_dim, 1)       FP16; may be null for bias-free linears
//   X_BD: (B, in_dim)        FP16
//   Y_BD: (B, out_dim)       FP16; resized if mis-shaped/-typed.
void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD);

// y[i] *= x[i]. Identical shape and dtype required. Dispatches FP32/FP16
// on y.dtype. Used by GEGLU and by gating paths in transformer FFNs.
void mul_inplace(Tensor& y, const Tensor& x);

// ─── AdaLN modulation (DiT / SD3 / Flux) ───────────────────────────────────
//
// Broadcast affine modulation: Y = X * (1 + scale) + shift, with the
// per-channel scale / shift broadcast across every token row. This is the
// adaptive-LayerNorm modulation every DiT block applies after norm():
//   modulate(norm(x), scale, shift)  ≡  x_hat * (1 + scale) + shift
//
//   X:     (L, D)   token activations
//   scale: a length-D vector ((1,D) or (D,1)) — modulation scale
//   shift: a length-D vector ((1,D) or (D,1)) — modulation shift
//   Y:     (L, D)   output, resized AND dtype-set to match X if mis-shaped.
// scale / shift must share X's dtype and device. Dispatched on X.dtype
// (FP32 / FP16 / BF16); FP32 internal math.
void modulate(const Tensor& X, const Tensor& scale, const Tensor& shift,
              Tensor& Y);

// Broadcast channel-wise multiply: Y[l, d] = X[l, d] * v[d], with the
// length-D vector v broadcast across every token row. This is the DiT
// residual gate — `x = x + broadcast_mul(sublayer_out, gate)` — and any
// per-channel rescale.
//
//   X: (L, D)   token activations
//   v: a length-D vector ((1,D) or (D,1)) — per-channel multiplier
//   Y: (L, D)   output, resized AND dtype-set to match X if mis-shaped.
// v must share X's dtype and device. Dispatched on X.dtype
// (FP32 / FP16 / BF16); FP32 internal math.
void broadcast_mul(const Tensor& X, const Tensor& v, Tensor& Y);

// GEGLU activation: input (B, 2*D) is split along the last dim into halves
// A=(B, D) and B_half=(B, D); output (B, D) = A * gelu(B_half). FP32 and FP16
// variants dispatched on X.dtype (FP16 accumulates in FP32).
//   X:  (B, 2*D)  input
//   Y:  (B, D)    output, resized AND dtype-set to match X if mis-shaped/-typed.
void geglu_forward(const Tensor& X, Tensor& Y);

// GEGLU backward. Splits X along the last dim into halves A and B_half (each
// (B, D)). Let g = gelu(B_half) (tanh-approx).
//   dA      = dY * g
//   dB_half = dY * A * gelu'(B_half)   (same derivative as gelu_backward)
// dX = concat(dA, dB_half) along the last dim with layout matching the
// forward (A then B_half). FP32 and FP16 dispatch on X.dtype (FP16 accumulates
// in FP32). dX is resized AND dtype-set to match X if mis-shaped/-typed.
void geglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX);

// Exact-GELU GEGLU activation: same shape contract as `geglu_forward`
// (input (B, 2*D) split along last dim into A=(B, D) and B_half=(B, D),
// output (B, D) = A * gelu_exact(B_half)), but uses the exact erf-based
// GELU instead of the tanh approximation. Matches HuggingFace `diffusers`'
// default GEGLU. FP32 and FP16 dispatched on X.dtype (FP16 accumulates in
// FP32).
//   X:  (B, 2*D)  input
//   Y:  (B, D)    output, resized AND dtype-set to match X if mis-shaped/-typed.
void geglu_exact_forward(const Tensor& X, Tensor& Y);

// Exact-GELU GEGLU backward. Splits X into A and B_half (each (B, D)). Let
// g = gelu_exact(B_half).
//   dA      = dY * g
//   dB_half = dY * A * gelu_exact'(B_half)   (see gelu_exact_backward)
// dX = concat(dA, dB_half) along the last dim, layout matches the forward
// (A then B_half). FP32 and FP16 dispatch on X.dtype (FP16 accumulates in
// FP32). dX is resized AND dtype-set to match X if mis-shaped/-typed.
void geglu_exact_backward(const Tensor& X, const Tensor& dY,
                          Tensor& dX);

// Causal mask helper for transformer self-attention. Produces an (L, L)
// FP32 mask where mask[q*L + k] = (k <= q) ? 1.0f : 0.0f. The existing
// attention kernels consume a length-Lk mask per attention row; for fully
// causal self-attention you launch the attention per query separately, so
// in practice we expose the simpler diagonal-cumulative form below.
//
// build_causal_mask_row fills the length-L FP32 buffer for row q:
//   mask[k] = (k <= q) ? 1.0f : 0.0f
// Resized to (L, 1) if mis-shaped. Useful for CLIP-text-encoder masking.
void build_causal_mask_row(int L, int q, Tensor& mask);

// Cross-attention: like mha_forward but K and V are projected from a
// separate context tensor instead of from X. Used in diffusion U-Nets to
// inject text conditioning. Dispatched on X.dtype:
//   * FP16: flash-attention path (inference). Caches are NOT exposed; if you
//     want them, use cross_attention_forward_train (FP32 only).
//   * FP32: training-aware path. Internally allocates scratch caches and
//     calls cross_attention_forward_train; scratch is discarded.
//
//   X:    (Lq, D)      query input (image tokens)
//   Ctx:  (Lk, D_ctx)  key/value input (text tokens). Lk and D_ctx may differ
//                      from Lq and D respectively. Ctx.dtype must match X.dtype.
//   Wq:   (D, D)       projects X → Q
//   Wk:   (D, D_ctx)   projects Ctx → K  (rectangular for cross-attn)
//   Wv:   (D, D_ctx)   projects Ctx → V
//   Wo:   (D, D)       output projection
//   d_mask: optional FP32 mask of length Lk (1 valid, 0 invalid). May be null.
//   num_heads: must divide D.
//   O:    (Lq, D) output, same dtype as X. Resized if mis-shaped.
void cross_attention_forward(const Tensor& X,
                             const Tensor& Ctx,
                             const Tensor& Wq, const Tensor& Wk,
                             const Tensor& Wv, const Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             Tensor& O);

// Cross-attention with explicit attention-map output and optional pre-softmax
// logit bias. Same math as cross_attention_forward (FP16) but:
//   * if `attn_logit_bias` is non-null it is added (FP32) to the scaled QKᵀ
//     scores *before* softmax, broadcast across heads (shape (Lq, Lk));
//   * after softmax the attention probabilities are averaged across heads
//     and written to AttnAvg (FP16, shape (Lq, Lk)).
// Designed for Cross-Attention Tree Search: the diffusion inference loop
// inspects head-averaged attention maps and may inject VLM/CLIP-derived
// logit biases. FP16 only, FP32 accumulation throughout. No backward.
//
//   X:    (Lq, D)      FP16 query input
//   Ctx:  (Lk, D_ctx)  FP16 key/value input
//   Wq:   (D, D)       FP16 — projects X → Q
//   Wk:   (D, D_ctx)   FP16 — projects Ctx → K
//   Wv:   (D, D_ctx)   FP16 — projects Ctx → V
//   Wo:   (D, D)       FP16 — output projection
//   d_mask: optional length-Lk FP32 mask (1 valid, 0 invalid). May be null.
//   attn_logit_bias: optional (Lq, Lk) FP32 pre-softmax bias. May be null.
//   num_heads: must divide D.
//   O:       (Lq, D)  FP16 output, resized AND dtype-set if mis-shaped/-typed.
//   AttnAvg: (Lq, Lk) FP16 head-averaged softmax, resized AND dtype-set
//                     if mis-shaped/-typed.
void cross_attention_forward_with_attn(const Tensor& X,
                                       const Tensor& Ctx,
                                       const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo,
                                       const float* d_mask,
                                       const Tensor* attn_logit_bias,
                                       int num_heads,
                                       Tensor& O,
                                       Tensor& AttnAvg);

// FP32 training-side self-attention forward. Thin wrapper over mha_forward
// (signatures match exactly when D_ctx == D and Ctx == X).
//   X:   (L, D) FP32
//   Wq, Wk, Wv, Wo: each (D, D) FP32
//   d_mask: optional length-L FP32 mask. May be null.
//   num_heads: must divide D.
//   Caches (resized if mis-shaped, written by op):
//     Qh, Kh, Vh: (h*L, D/h)
//     Attnh:      (h*L, L)
//     Yconcat:    (L, D)
//   O:   (L, D) FP32; resized if mis-shaped.
void self_attention_forward_train(const Tensor& X,
                                  const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo,
                                  const float* d_mask,
                                  int num_heads,
                                  Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                  Tensor& Attnh, Tensor& Yconcat,
                                  Tensor& O);

// FP32 training-side self-attention backward. Thin wrapper over
// mha_backward.
//   dO: (L, D) upstream
//   X, Qh, Kh, Vh, Attnh, Yconcat: forward caches
//   Wq, Wk, Wv, Wo: forward weights (each (D, D))
//   d_mask: same mask used in forward (or null)
//   num_heads: must match forward
//   dX: (L, D) output, *overwritten*
//   dWq, dWk, dWv, dWo: (D, D) accumulated into — caller zeros.
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

// Per-text-token spatial moments of a cross-attention map. Given an attention
// matrix Attn(Lq, Lk) where Lq = h_lat * w_lat is a flattened image-token grid
// in row-major (q = y * w_lat + x), compute for each text token k:
//   mass[k]        = sum_q Attn[q, k]
//   centroid[k, 0] = (sum_q y(q) * Attn[q, k]) / max(mass[k], 1e-8)
//   centroid[k, 1] = (sum_q x(q) * Attn[q, k]) / max(mass[k], 1e-8)
// When mass[k] is effectively zero the centroid is set to (0, 0). Used as a
// MCTS reward primitive to check whether different text tokens attend to
// physically separated latent regions. FP32 reductions over FP16 input.
//   Attn:     (Lq, Lk) FP16, Lq = h_lat * w_lat
//   mass:     (Lk, 1)  FP32, resized if mis-shaped
//   centroid: (Lk, 2)  FP32, resized if mis-shaped, [y, x] per row
void attention_token_moments(const Tensor& Attn,
                             int h_lat, int w_lat,
                             Tensor& mass,
                             Tensor& centroid);

// FP32 training-side cross-attention forward. Mirrors mha_forward math
// but accepts a separate Ctx tensor for K/V projection and rectangular
// Wk/Wv: (D, D_ctx).
//
//   X:    (Lq, D)      FP32 query input
//   Ctx:  (Lk, D_ctx)  FP32 key/value input
//   Wq:   (D, D)       FP32 — projects X → Q
//   Wk:   (D, D_ctx)   FP32 — projects Ctx → K
//   Wv:   (D, D_ctx)   FP32 — projects Ctx → V
//   Wo:   (D, D)       FP32 — output projection
//   d_mask: optional length-Lk FP32 mask. May be null.
//   num_heads: must divide D.
//   Caches (resized if mis-shaped):
//     Qh:    (h*Lq, D/h)
//     Kh:    (h*Lk, D/h)
//     Vh:    (h*Lk, D/h)
//     Attnh: (h*Lq, Lk)
//     Yconcat: (Lq, D)
//   O:    (Lq, D) FP32; resized if mis-shaped.
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
//   dO: (Lq, D) upstream
//   X, Ctx, Qh, Kh, Vh, Attnh, Yconcat: forward caches
//   Wq: (D, D), Wk/Wv: (D, D_ctx), Wo: (D, D)
//   d_mask: same mask used in forward (or null)
//   num_heads: must match forward
//   dX:   (Lq, D)     output, *overwritten*
//   dCtx: (Lk, D_ctx) output, *overwritten*
//   dWq:  (D, D)      accumulated into — caller zeros
//   dWk:  (D, D_ctx)  accumulated into — caller zeros
//   dWv:  (D, D_ctx)  accumulated into — caller zeros
//   dWo:  (D, D)      accumulated into — caller zeros
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

// FP16 LayerNorm forward, inference-only. Processes R independent rows of
// length D in a single launch (one block per row). FP32 accumulation.
//   X_RD:  (R, D)  FP16
//   gamma: (D,)    FP16 scale
//   beta:  (D,)    FP16 shift
//   Y_RD:  (R, D)  FP16; resized as needed.
void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              const Tensor& beta,
                                              Tensor& Y_RD,
                                              float eps);

// FP16 self-attention. Thin wrapper over the cross-attention kernel with
// Ctx = X (so Lk = Lq). Same shape/dtype conventions as
// cross_attention_forward otherwise.
//   X:   (L, D)  FP16
//   Wq, Wk, Wv, Wo: each (D, D), FP16
//   d_mask: optional FP32 mask of length L (1 valid, 0 invalid). May be
//           null.
//   num_heads: must divide D.
//   O:   (L, D)  FP16; resized if mis-shaped.
void self_attention_forward(const Tensor& X,
                            const Tensor& Wq, const Tensor& Wk,
                            const Tensor& Wv, const Tensor& Wo,
                            const float* d_mask,
                            int num_heads,
                            Tensor& O);

// Flash-attention-style fused attention (FP16, inference-only).
//
// Q, K, V are already projected (caller does the matmuls externally). Tiles
// over Lk with online softmax — no Lk-long materialisation in shared mem;
// works for Lq, Lk up to global-memory limits. FP32 accumulation throughout.
//
//   Q:  (Lq, D)   FP16
//   K:  (Lk, D)   FP16
//   V:  (Lk, D)   FP16
//   d_mask: optional length-Lk FP32 mask (1 valid, 0 invalid). May be null.
//   num_heads: must divide D.
//   causal: if true, apply autoregressive causal masking (key index k
//           may only attend to query index q where k ≤ q). Requires Lq == Lk.
//           Combines multiplicatively with d_mask when both supplied. The
//           kernel skips fully-future tiles outright (one less full tile cost
//           per row beyond the diagonal) and masks within the boundary tile.
//   O:  (Lq, D)   FP16; resized as needed.
void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O);

// Flash-attention with projections fused in at the boundary. Projects
// X → Q, Ctx → K, V (or X → Q,K,V when Ctx == nullptr), runs the tiled
// attention core, then projects the output with Wo. FP16 throughout.
//
//   X:   (Lq, D)            FP16, query source
//   Ctx: (Lk, D_ctx) or null  FP16, key/value source; null means
//                              self-attention (Ctx ← X, Lk ← Lq, D_ctx ← D).
//                              D_ctx may differ from D — e.g. SD1.5
//                              cross-attention has D_ctx = 768 (CLIP) while
//                              D varies per U-Net stage.
//   Wq: (D, D)              FP16 — projects X → Q
//   Wk: (D, D_ctx)          FP16 — projects Ctx → K
//   Wv: (D, D_ctx)          FP16 — projects Ctx → V
//   Wo: (D, D)              FP16 — projects attn output → O
//   bq, bk, bv, bo: optional (D, 1) FP16 biases for the corresponding
//                   projection. Pass nullptr to skip. SD1.5 CLIP attention
//                   has all four; UNet/VAE attention typically has only bo.
//   d_mask: optional length-Lk FP32 mask.
//   num_heads: must divide D.
//   causal: see flash_attention_forward. Typically paired with Ctx ==
//           nullptr (causal self-attention, e.g. CLIP text encoder).
//   O:   (Lq, D)  FP16; resized as needed.
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

// Backward partner of flash_attention_qkvo_forward. "Recompute-style":
// no caches consumed from the forward call — backward re-runs the attention
// math (Q,K,V projection + per-head softmax PV) from the inputs, then
// reverses the math to produce gradients. This matches the xformers /
// flash-attn upstream convention: no API change on the forward, no extra
// per-call buffers saved by the caller.
//
// All FP16. Numerics: FP32 accumulation throughout the per-head sweep
// (matmuls, softmax recompute, D_q reduction, dS, projection-bw scratch).
//
// All shape / dtype / Ctx-may-be-null / rectangular-Wk-Wv / causal /
// optional-biases semantics exactly match flash_attention_qkvo_forward.
//
//   X, Ctx (or null), Wq, bq, Wk, bk, Wv, bv, Wo, bo, d_mask, num_heads,
//   causal:        same as forward. Pass the same values used in the
//                  forward call.
//   dO:            (Lq, D)         FP16, upstream gradient of forward O.
//   dX:            (Lq, D)         FP16, *overwritten*. For self-attn
//                                  (Ctx == null) dX absorbs the K/V-projection
//                                  gradients in addition to the Q-projection
//                                  path: dX = dQ·Wq + dK·Wk + dV·Wv.
//   dCtx:          (Lk, D_ctx)     FP16, *overwritten*. Must be nullptr iff
//                                  Ctx is nullptr. For cross-attn
//                                  (Ctx != null) dCtx = dK·Wk + dV·Wv.
//   dWq:           (D, D)          FP16, *accumulated into* (caller zeros).
//   dWk:           (D, D_ctx)      FP16, *accumulated*. Rectangular for cross.
//   dWv:           (D, D_ctx)      FP16, *accumulated*. Rectangular for cross.
//   dWo:           (D, D)          FP16, *accumulated*.
//   dbq, dbk, dbv, dbo: (D, 1)     FP16, *accumulated* iff the corresponding
//                                  forward bias was non-null. Pass nullptr
//                                  for any whose forward bias was nullptr;
//                                  passing a non-null grad alongside a null
//                                  forward bias is rejected (the symmetry
//                                  must be exact, matching what an autograd
//                                  caller will naturally produce).
//
// Causal masking: as in the forward, position k > q contributes nothing to
// the softmax — dV[k]/dK[k]/dQ[q]/dCtx[k] receive no contribution from those
// (q, k) pairs.
//
// Mask: positions k with d_mask[k] <= 0.5 are dropped in the recompute
// (probability 0), so they contribute nothing to dV/dK/dCtx and dQ/dX
// degrades naturally.
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

// Backward of flash_attention_forward — bare attention core, no
// projection weights. Recompute-based: reproduces the per-head softmax to
// obtain P, then runs the standard FlashAttention-2 backward to produce
// dQ / dK / dV. The "bare" variant is what LoRA-style adapters need —
// projections are wrapped externally (typically via matmul /
// matmul_backward) so adapter parameters can be folded in without
// disturbing the attention core.
//
// All tensors FP16. Numerics: FP32 accumulation throughout the per-head
// sweep (recompute matmuls, softmax, D_q reduction, dS, dQ/dK/dV).
//
//   Q:        (Lq, D)         FP16  — pre-projected queries (forward input).
//   K:        (Lk, D)         FP16  — pre-projected keys.
//   V:        (Lk, D)         FP16  — pre-projected values.
//   O:        (Lq, D)         FP16  — forward output. Currently unused by the
//                                     recompute path but retained in the API
//                                     for symmetry with standard flash-attn
//                                     bwd signatures (and to allow a future
//                                     cache-based shortcut).
//   dO:       (Lq, D)         FP16  — upstream gradient of O.
//   d_mask:   optional length-Lk FP32 mask (nullptr for unmasked). Same
//             semantics as flash_attention_forward's d_mask: positions
//             with mask[k] <= 0.5 are dropped (probability 0); they
//             contribute nothing to dV / dK and the corresponding rows of
//             dQ degrade naturally.
//   num_heads: must divide D.
//   causal:   match the forward's causal flag. Causal masking: position
//             k > q contributes nothing to the softmax — dV[k] / dK[k] /
//             dQ[q] receive no contribution from those (q, k) pairs.
//             Requires Lq == Lk when true.
//   dQ:       (Lq, D)         FP16, *overwritten* (resized + dtype-set if
//                                    mis-shaped).
//   dK:       (Lk, D)         FP16, *overwritten*.
//   dV:       (Lk, D)         FP16, *overwritten*.
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

// Project a key/value context tensor through Wk/Wv (with optional biases),
// producing the exact (Lk, D) FP16 buffers that flash_attention_forward
// consumes. Used to pre-compute cross-attention K/V once per generate() in
// diffusion U-Nets — the text context is fixed across all denoising steps so
// these projections are otherwise pure waste.
//
// Numerically identical to the K/V projection stage inside
// flash_attention_qkvo_forward (same linear_forward_batched_fp16 call).
//
//   ctx:    (Lk, D_ctx)  FP16
//   Wk:     (D, D_ctx)   FP16 — projects ctx → K
//   bk:     (D, 1)       FP16, optional (nullptr to skip)
//   Wv:     (D, D_ctx)   FP16 — projects ctx → V
//   bv:     (D, 1)       FP16, optional
//   K_out:  (Lk, D)      FP16; resized as needed
//   V_out:  (Lk, D)      FP16; resized as needed
void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out,
                                Tensor& V_out);

// Like flash_attention_qkvo_forward but K and V are already projected by
// the caller (typically via flash_attention_project_kv). Projects X → Q
// with Wq/bq, runs the tiled attention core against the supplied K/V, then
// applies Wo/bo. Equivalent (bitwise) to the cached path of
// flash_attention_qkvo_forward.
//
//   X:      (Lq, D)     FP16, query source
//   K:      (Lk, D)     FP16, pre-projected keys (layout = flash_attention_forward's K arg)
//   V:      (Lk, D)     FP16, pre-projected values
//   Wq:     (D, D)      FP16
//   bq:     optional FP16 (D, 1)
//   Wo:     (D, D)      FP16
//   bo:     optional FP16 (D, 1)
//   d_mask: optional length-Lk FP32 mask
//   num_heads: must divide D
//   causal: see flash_attention_forward (false for diffusion cross-attn)
//   O:      (Lq, D)     FP16; resized as needed
void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                              const Tensor& K,
                                              const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask,
                                              int num_heads,
                                              bool causal,
                                              Tensor& O);

// NCHW ↔ sequence layout transpose. Lets ops that expect a (L, D) token
// layout (flash_attention_*, self/cross attention wrappers) consume tensors
// produced by NCHW primitives (conv2d, group_norm, resblock). Per-element
// gather/scatter — no math, no padding.
//
// FP32 and FP16 are both supported; dispatched on X.dtype. Y is resized AND
// dtype-set to match X.dtype if mis-shaped/-typed. X and Y must not alias.
//
// nchw_to_sequence:
//   X:  (N, C * H * W)        any dtype, treated as NCHW
//   Y:  (N * H * W, C)        same dtype; Y[n*H*W + h*W + w, c] = X[n,c,h,w]
//
// sequence_to_nchw (inverse):
//   X:  (N * H * W, C)        any dtype, sequence layout
//   Y:  (N, C * H * W)        same dtype; Y[n,c,h,w] = X[n*H*W + h*W + w, c]
//
// For SD VAE mid-block self-attention (N=1) this gives the (H*W, C) token
// layout the flash kernels want. For N>1, the sequence form is (N*H*W, C);
// callers wanting a separate per-batch attention pass slice the rows.
void nchw_to_sequence(const Tensor& X,
                      int N, int C, int H, int W,
                      Tensor& Y);

void sequence_to_nchw(const Tensor& X,
                      int N, int C, int H, int W,
                      Tensor& Y);

// Fused diffusion ResBlock (FP16, inference-only). Computes the standard
// SD U-Net residual block in a single op:
//
//   h = silu(group_norm(X, gamma1, beta1))
//   h = conv2d_3x3_same(h, W1, b1)
//   if t_emb_shift: h += broadcast(t_emb_shift, NCHW)   // (N, C_out) or (C_out,)
//   h = silu(group_norm(h, gamma2, beta2))
//   h = conv2d_3x3_same(h, W2, b2)
//   if C_in == C_out and Wskip == nullptr:
//       Y = h + X
//   else:
//       Y = h + conv2d_1x1(X, Wskip, bskip)
//
// All tensors FP16. Conv layout matches conv2d_forward (OIHW filter
// layout). The two `GN → SiLU` legs are fused into single kernels; the
// convs remain separate launches but the skip add is folded into the
// second conv's epilogue when shapes permit.
//
//   X:       (N, C_in  * H * W)   FP16 input activation
//   gamma1, beta1: (C_in,  1)     FP16 — applied to the first GN over C_in
//   W1:      (C_out, C_in  * 9)   FP16 OIHW, kH=kW=3
//   b1:      (C_out, 1) or null   FP16
//   t_emb_shift: (N, C_out) or (C_out, 1) or null — additive shift between legs
//   gamma2, beta2: (C_out, 1)     FP16 — applied to second GN over C_out
//   W2:      (C_out, C_out * 9)   FP16 OIHW, kH=kW=3
//   b2:      (C_out, 1) or null   FP16
//   Wskip:   (C_out, C_in * 1)    FP16 OIHW 1x1, or null when C_in == C_out
//   bskip:   (C_out, 1) or null
//   Y:       (N, C_out * H * W)   FP16 output, resized as needed.
//
// num_groups must divide both C_in and C_out (typically 32). eps default 1e-5.
void resblock_forward(const Tensor& X,
                      const Tensor& gamma1, const Tensor& beta1,
                      const Tensor& W1, const Tensor* b1,
                      const Tensor* t_emb_shift,
                      const Tensor& gamma2, const Tensor& beta2,
                      const Tensor& W2, const Tensor* b2,
                      const Tensor* Wskip, const Tensor* bskip,
                      int N, int C_in, int C_out, int H, int W,
                      int num_groups, float eps,
                      Tensor& Y);

// W8A16 variant of resblock_forward (inference-only). Identical math to
// the FP16 op, but conv1, conv2, and the optional 1x1 skip conv consume INT8
// weights with per-output-row symmetric FP32 scales (matching the W8A16
// contract used by conv2d_int8w_fp16_forward). Activations, GN params,
// biases, and the t_emb shift stay FP16. No backward — quantised weights are
// frozen at inference time.
//
//   X:           (N, C_in  * H * W)   FP16 input activation
//   gamma1, beta1: (C_in,  1)         FP16
//   W1_int8:     (C_out, C_in  * 9)   INT8 OIHW, kH=kW=3
//   s1:          (C_out, 1)           FP32 — per-output-row dequant scales for W1
//   b1:          (C_out, 1) or null   FP16
//   t_emb_shift: (N, C_out) or (C_out, 1) or null  FP16
//   gamma2, beta2: (C_out, 1)         FP16
//   W2_int8:     (C_out, C_out * 9)   INT8 OIHW, kH=kW=3
//   s2:          (C_out, 1)           FP32 — per-output-row dequant scales for W2
//   b2:          (C_out, 1) or null   FP16
//   Wskip_int8:  (C_out, C_in * 1)    INT8 OIHW 1x1, or null when C_in == C_out
//   sskip:       (C_out, 1)           FP32 — required iff Wskip_int8 != null
//   bskip:       (C_out, 1) or null   FP16
//   Y:           (N, C_out * H * W)   FP16 output, resized as needed.
//
// num_groups must divide both C_in and C_out (typically 32). eps default 1e-5.
void resblock_forward_int8w_fp16(const Tensor& X,
                                 const Tensor& gamma1, const Tensor& beta1,
                                 const Tensor& W1_int8, const Tensor& s1,
                                 const Tensor* b1,
                                 const Tensor* t_emb_shift,
                                 const Tensor& gamma2, const Tensor& beta2,
                                 const Tensor& W2_int8, const Tensor& s2,
                                 const Tensor* b2,
                                 const Tensor* Wskip_int8, const Tensor* sskip,
                                 const Tensor* bskip,
                                 int N, int C_in, int C_out, int H, int W,
                                 int num_groups, float eps,
                                 Tensor& Y);

// Composite backward of resblock_forward. All tensors FP16. Implemented
// purely by composition of the existing public ops (group_norm forward+
// backward, silu forward+backward, conv2d forward+backward_input/weight/bias,
// add_inplace). The forward intermediates are NOT cached by the forward op,
// so the backward recomputes them from X and the cached parameter tensors:
//
//   h1_pre_silu = group_norm_forward(X, gamma1, beta1)
//   h1          = silu(h1_pre_silu)
//   h2_pre_t    = conv2d_3x3_same(h1, W1, b1)
//   h2          = h2_pre_t + broadcast(t_emb_shift)  (if t_emb_shift)
//   h3_pre_silu = group_norm_forward(h2, gamma2, beta2)
//   h3          = silu(h3_pre_silu)
//
// then routes dY back through SiLU/GN/conv1/conv2/t_emb_shift, summing with
// the residual gradient (identity-skip or conv1x1-skip).
//
// Argument semantics:
//   X, gamma1, beta1, W1, b1, t_emb_shift, gamma2, beta2, W2, b2,
//   Wskip, bskip:                   forward inputs (read-only).
//   dY:    (N, C_out * H * W)       upstream gradient (FP16).
//   dX:    (N, C_in  * H * W)       *overwritten* (resized/retyped as needed).
//   dGamma1, dBeta1: (C_in,  1)     *accumulated* — caller zeros.
//   dW1:   (C_out, C_in * 9)        *accumulated* — caller zeros.
//   db1:   (C_out, 1) or null       *accumulated* if non-null and b1 was used.
//   dt_emb_shift: same shape as     *accumulated* if non-null and t_emb_shift
//                 t_emb_shift, or null. was used.
//   dGamma2, dBeta2: (C_out, 1)     *accumulated* — caller zeros.
//   dW2:   (C_out, C_out * 9)       *accumulated* — caller zeros.
//   db2:   (C_out, 1) or null       *accumulated* if non-null and b2 was used.
//   dWskip: (C_out, C_in * 1) or    *accumulated* if non-null and Wskip was
//           null.                   used (i.e. C_in != C_out path).
//   dbskip: (C_out, 1) or null      *accumulated* if non-null and bskip was used.
//
// num_groups/eps must match the forward call. All FP16.
void resblock_backward(const Tensor& X,
                       const Tensor& gamma1, const Tensor& beta1,
                       const Tensor& W1, const Tensor* b1,
                       const Tensor* t_emb_shift,
                       const Tensor& gamma2, const Tensor& beta2,
                       const Tensor& W2, const Tensor* b2,
                       const Tensor* Wskip, const Tensor* bskip,
                       int N, int C_in, int C_out, int H, int W,
                       int num_groups, float eps,
                       const Tensor& dY,
                       Tensor& dX,
                       Tensor& dGamma1, Tensor& dBeta1,
                       Tensor& dW1, Tensor* db1,
                       Tensor* dt_emb_shift,
                       Tensor& dGamma2, Tensor& dBeta2,
                       Tensor& dW2, Tensor* db2,
                       Tensor* dWskip, Tensor* dbskip);

// ─── Llama-style transformer ops (forward + backward where noted) ──────────

// Plain row-major matrix multiply with no bias:
//   C(M, N) = A(M, K) @ B(K, N)
// Dispatched on A.dtype; B and C must share A.dtype (C is resized AND
// dtype-set to match A if mis-shaped/-typed). Internal accumulation is in
// FP32 for both FP32 and FP16 paths.
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Backward of matmul. Row-major, no bias.
//   forward: C(M, N) = A(M, K) @ B(K, N)
//   dA(M, K) += dC(M, N) @ B^T(N, K)
//   dB(K, N) += A^T(K, M) @ dC(M, N)
//
// Dtype-dispatched FP32 + FP16. All five tensors must share the same dtype.
// FP16 dA/dB accumulators use FP32 scratch + fold (atomic-add into FP16 is
// unsafe across blocks). Caller-zeros-and-passes-presized, op-accumulates-into
// convention (mirrors linear_backward): dA must be (M, K) and dB must be
// (K, N), both pre-allocated and pre-zeroed by the caller; this op adds its
// contribution to whatever's already there. dC is read-only and must be
// (M, N).
//
//   A:   (M, K)   forward input
//   B:   (K, N)   forward weight
//   dC:  (M, N)   upstream gradient
//   dA:  (M, K)   *accumulated into*
//   dB:  (K, N)   *accumulated into*
void matmul_backward(const Tensor& A,
                     const Tensor& B,
                     const Tensor& dC,
                     Tensor& dA,
                     Tensor& dB);

// RoPE (rotary position embedding) forward. Applied per head:
//   x_{2i}   ← x_{2i} * cos(θ) - x_{2i+1} * sin(θ)
//   x_{2i+1} ← x_{2i} * sin(θ) + x_{2i+1} * cos(θ)
// where θ = pos * theta_base^{-2i/head_dim} for token position `pos` and
// dimension pair index i in [0, head_dim/2). seq_offset shifts the position
// of row 0 (KV-cache decode starts at the previous valid length).
//
//   X: (L, num_heads * head_dim)  — input
//   Y: (L, num_heads * head_dim)  — output, resized AND dtype-set to match X
// head_dim must be even. Dispatched on X.dtype (FP32 or FP16).
void rope_forward(const Tensor& X, int head_dim, int num_heads,
                 int seq_offset, float theta_base, Tensor& Y);

// RoPE backward. Equivalent to applying the inverse (transpose) rotation to
// dY pair-wise per head:
//   dX_{2i}   ← dY_{2i} * cos(θ) + dY_{2i+1} * sin(θ)
//   dX_{2i+1} ← -dY_{2i} * sin(θ) + dY_{2i+1} * cos(θ)
//   dY: (L, num_heads * head_dim)
//   dX: (L, num_heads * head_dim) — resized AND dtype-set to match dY.
// Dispatched on dY.dtype.
void rope_backward(const Tensor& dY, int head_dim, int num_heads,
                  int seq_offset, float theta_base, Tensor& dX);

// RoPE with explicit, caller-supplied cos / sin rotation tables.
//
// Unlike rope_forward — which derives the angle from row index + seq_offset —
// this variant takes a precomputed per-(row, pair) table, so the caller owns
// all position semantics. It handles arbitrary explicit position ids and,
// crucially, 2D axial RoPE (Flux / SD3): the caller assigns each frequency
// pair to a height / width / text axis and bakes the resulting angle into the
// table. brotensor stays layout-agnostic; the kernel just applies the rotation.
//
//   x_{2i}   ← x_{2i} * cos_tbl[row,i] - x_{2i+1} * sin_tbl[row,i]
//   x_{2i+1} ← x_{2i} * sin_tbl[row,i] + x_{2i+1} * cos_tbl[row,i]
//
//   X:       (L, num_heads * head_dim)  input
//   cos_tbl: (L, head_dim/2)  FP32 — cos of the rotation angle per (row, pair)
//   sin_tbl: (L, head_dim/2)  FP32 — sin of the rotation angle per (row, pair)
//   Y:       (L, num_heads * head_dim)  output, resized + dtype-set to match X
// The same cos/sin row is shared across all heads. head_dim must be even.
// cos_tbl / sin_tbl are FP32 on any backend; X / Y dispatch on X.dtype
// (FP32 / FP16 / BF16). FP32 internal math.
void rope_apply(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                int head_dim, int num_heads, Tensor& Y);

// RoPE-with-tables backward — applies the inverse (transpose) rotation:
//   dX_{2i}   ←  dY_{2i} * cos_tbl[row,i] + dY_{2i+1} * sin_tbl[row,i]
//   dX_{2i+1} ← -dY_{2i} * sin_tbl[row,i] + dY_{2i+1} * cos_tbl[row,i]
//   dY: (L, num_heads * head_dim)
//   dX: (L, num_heads * head_dim) — resized + dtype-set to match dY.
// cos_tbl / sin_tbl as in rope_apply. Dispatched on dY.dtype.
void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl,
                         const Tensor& sin_tbl, int head_dim, int num_heads,
                         Tensor& dX);

// RMSNorm forward, per row:
//   rms[b] = sqrt(mean_j x[b, j]^2 + eps)
//   y[b, j] = x[b, j] * gamma[j] / rms[b]
//   X:     (B, D) input
//   gamma: (D, 1) scale (same dtype as X)
//   Y:     (B, D) output, resized AND dtype-set to match X if mis-shaped/-typed.
// Dispatched on X.dtype (FP32 or FP16). FP32 accumulation internally.
void rms_norm_forward(const Tensor& X, const Tensor& gamma,
                     float eps, Tensor& Y);

// RMSNorm backward.
//   X:      (B, D) forward input
//   gamma:  (D, 1) forward scale
//   dY:     (B, D) upstream gradient (same dtype as X)
//   dX:     (B, D) overwritten (resized + dtype-set to match X if mis-shaped/-typed)
//   dGamma: (D, 1) *accumulated* — caller zeros. Same dtype as X. For FP16
//                  storage, an FP32 scratch + fold epilogue is used.
void rms_norm_backward(const Tensor& X, const Tensor& gamma,
                      const Tensor& dY, float eps,
                      Tensor& dX, Tensor& dGamma);

// SwiGLU (Llama FFN gate). Input (B, 2*D) is split along the last dim into
// halves A=(B, D) and B_half=(B, D); output (B, D) = silu(A) * B_half.
//   X:  (B, 2*D)  input
//   Y:  (B, D)    output, resized AND dtype-set to match X if mis-shaped/-typed.
// Dispatched on X.dtype (FP32 or FP16; FP16 accumulates in FP32).
void swiglu_forward(const Tensor& X, Tensor& Y);

// SwiGLU backward. Splits X into A and B_half. Let s = silu(A).
//   dA      = dY * B_half * silu'(A)
//   dB_half = dY * s
// dX = concat(dA, dB_half) along the last dim with layout matching the
// forward (A then B_half). Dispatched on X.dtype.
void swiglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX);

// KV-cache append (FP16). Copies K_new and V_new into rows
// [cur_len, cur_len + L_new) of K_cache and V_cache respectively. Both new
// tensors and both caches are FP16 with matching cols (== D). L_new + cur_len
// must fit within K_cache.rows / V_cache.rows (caller pre-allocates).
//   K_new, V_new:     (L_new, D)  FP16
//   K_cache, V_cache: (L_max, D)  FP16 — must already be sized; not resized.
void kv_cache_append(const Tensor& K_new, const Tensor& V_new,
                     int cur_len, Tensor& K_cache, Tensor& V_cache);

// Causal flash-attention against a partially-filled KV cache (FP16, fwd-only).
// Runs the tiled attention core (same as flash_attention_forward) against
// the first `valid_len` rows of K_cache and V_cache. Query position
// p_q = seq_offset + i (with seq_offset = valid_len - L_q) attends to cache
// positions [0, p_q]; entries with cache index > p_q are masked out
// (causal mask).
//
//   Q:        (L_q, D)        FP16  — typically L_q == 1 for token-by-token
//                                     decoding but L_q > 1 is supported.
//   K_cache:  (L_max, D)      FP16  — only rows [0, valid_len) are read.
//   V_cache:  (L_max, D)      FP16
//   valid_len: number of valid cache rows (>= L_q).
//   num_heads: must divide D.
//   O:        (L_q, D)        FP16  — resized as needed.
void flash_attention_decode(const Tensor& Q,
                           const Tensor& K_cache, const Tensor& V_cache,
                           int valid_len, int num_heads, Tensor& O);

// ─── Public reductions ─────────────────────────────────────────────────────

// Row-wise sum: Y[m, 0] = sum_n X[m, n].
//   X: (M, N)  FP32 or FP16
//   Y: (M, 1)  same dtype as X — resized as needed.
void sum_rows(const Tensor& X, Tensor& Y);

// Column-wise sum: Y[0, n] = sum_m X[m, n].
//   X: (M, N)  FP32 or FP16
//   Y: (1, N)  same dtype as X — resized as needed.
void sum_cols(const Tensor& X, Tensor& Y);

// Row-wise argmax: Idx[m, 0] = argmax_n X[m, n], stored as FP32 holding the
// integer index cast to float (keeps the type system uniform).
//   X:   (M, N)  FP32 or FP16
//   Idx: (M, 1)  FP32 — resized as needed.
void argmax_rows(const Tensor& X, Tensor& Idx);

// ─── Fused DDIM step (FP16) ────────────────────────────────────────────────
//
// One-shot DDIM update applied element-wise to a noisy latent:
//   x0_pred = (x_t - sqrt(1 - alpha_t) * eps_pred) / sqrt(alpha_t)
//   dir     = sqrt(1 - alpha_prev - sigma_t^2) * eps_pred
//   x_prev  = sqrt(alpha_prev) * x0_pred + dir
// sigma_t = 0 yields deterministic DDIM; the formula still holds. FP16
// inputs and outputs; FP32 internal math. x_t and eps_pred must share shape;
// x_prev is resized to match.
void ddim_step(const Tensor& x_t, const Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               Tensor& x_prev);

// ─── Fused Euler-discrete sampler step (FP16) ──────────────────────────────
//
// ε-prediction, σ convention matching diffusers' EulerDiscreteScheduler:
//   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred
// FP16 inputs and outputs; FP32 internal math. x_t and eps_pred must share
// shape; x_prev is resized to match.
void euler_step(const Tensor& x_t, const Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                Tensor& x_prev);

// ─── Fused DPM-Solver++ 2M sampler step (FP16) ─────────────────────────────
//
// Multistep, ε-prediction. The caller maintains a running x0 cache and
// computes the three linear-combination coefficients host-side from the
// scheduler's σ / log-SNR schedule. The kernel reconstructs
//   x0_t   = x_t - sigma_t * eps_pred
//   x_prev = c_xt * x_t + c_x0t * x0_t + c_x0prev * x0_prev
//   x0_out = x0_t            (caller copies into x0_prev for the next step)
//
// Coefficient derivation (k-diffusion / DPM++ 2M, ε-prediction, α≡1):
//   h_last = lambda_t - lambda_last,  h = lambda_next - lambda_t,  r = h_last/h
//   D_t    = (1 + 1/(2r)) * x0_t - (1/(2r)) * x0_prev
//   x_prev = (sigma_next/sigma_t) * x_t - (exp(-h) - 1) * D_t
// →  c_xt     = sigma_next / sigma_t
//    c_x0t    = -(exp(-h) - 1) * (1 + 1/(2r))
//    c_x0prev = -(exp(-h) - 1) * (-1/(2r))
//
// First step (no x0_prev cached): use euler_step instead.
// All tensors FP16, same shape; x_prev and x0_out resized to match.
void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred,
                   const Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   Tensor& x_prev, Tensor& x0_out);

// ─── Sinusoidal timestep embedding (FP32) ──────────────────────────────────
//
// Matches diffusers' get_timestep_embedding with flip_sin_to_cos=True,
// downscale_freq_shift=0 — the SD / SDXL default.
//   half      = dim / 2
//   freqs[j]  = exp(-log(max_period) * j / half)        for j in [0, half)
//   Y[i, 0:half]      = cos(timesteps[i] * freqs[:])
//   Y[i, half:2*half] = sin(timesteps[i] * freqs[:])
//   if dim is odd: Y[i, dim-1] = 0
// Used for the diffusion timestep itself and for SDXL's added-cond
// micro-conditioning vector (original_size / crop_top_left / target_size).
//
//   timesteps: (N, 1) FP32
//   Y:         (N, dim) FP32 — resized as needed.
void timestep_embedding(const Tensor& timesteps,
                        int dim, float max_period,
                        Tensor& Y);

// ─── INT8 weight-only quantisation (W8A16) ─────────────────────────────────

// Host helper: quantise an FP16 weight matrix to per-output-row symmetric
// INT8.
//   W_fp16:      (out, in) FP16 host buffer (uint16_t bit pattern)
//   out, in:     dimensions
//   W_int8_out:  filled with out*in int8_t values, row-major (out, in)
//   scales_out:  filled with `out` FP32 scales (one per output row).
// Scale per row = max(|w|) / 127 (or 0 if the row is all zero); quantised
// value = clamp(round(w / scale), -127, 127).
//
// Host-only helper, not dispatched per device — operates on plain host
// buffers passed by the caller. Backend-independent.
void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out);

// W8A16 matmul: Y = dequant(W_int8, scales) @ X.
//   W_int8: (out, in) Dtype::INT8 — quantised weights, per-row scales.
//   scales: (out, 1) FP32 — per-row dequant scales.
//   X:      (in, B)  FP16 — activations.
//   Y:      (out, B) FP16 — resized as needed.
// Matches the shape convention of matmul (plain (M,K)@(K,N) without
// transpose); callsites can substitute one for the other.
void matmul_int8w_fp16(const Tensor& W_int8,
                       const Tensor& scales,
                       const Tensor& X,
                       Tensor& Y);

// W8A16 conv2d forward. Mirrors conv2d_forward's signature; only the
// weight dtype differs.
//   W_int8: (C_out, C_in/groups * kH * kW) Dtype::INT8 — OIHW filter,
//           quantised per output channel.
//   scales: (C_out, 1) FP32 — per-output-channel dequant scales.
//   bias:   FP16 (C_out, 1) or nullptr.
//   X, Y:   FP16, same layout as conv2d_forward.
void conv2d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y);

// W8A16 batched linear: Y(B, out) = X(B, in) @ dequant(W_int8(out, in))^T + b.
// Mirrors linear_forward_batched_fp16's shape contract — the (B, in) →
// (B, out) layout used by the fused flash-attention projection wrappers and
// any transformer FFN. Per-output-row symmetric scales; FP16 activations
// and bias. FP32 accumulation inside the kernel.
//
//   W_int8: (out, in)  Dtype::INT8
//   scales: (out, 1)   FP32 — per-output-row dequant scales
//   bias:   FP16 (out, 1) or (1, out), optional (nullptr to skip)
//   X_BD:   (B, in)    FP16
//   Y_BD:   (B, out)   FP16 — resized as needed
void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD,
                                       Tensor& Y_BD);

// ─── W8A16 variants of the three fused flash-attention ops ─────────────────
//
// Same composition as flash_attention_project_kv /
// flash_attention_q_with_kv_cached_forward / flash_attention_qkvo_forward,
// but every linear projection consumes an INT8 weight + per-output-row FP32
// scale tensor instead of an FP16 weight. The attention core itself stays
// FP16 — activations are never quantised. Each W*_int8 has shape (D, in_dim)
// with its own scales tensor of shape (D, 1); biases remain FP16 (D, 1) and
// optional. Semantics, masks, causal flag, and num_heads are identical to the
// FP16 versions.
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

} // namespace brotensor
