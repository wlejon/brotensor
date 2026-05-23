#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace brotensor {

// ─── brotensor ops (declarations only) ─────────────────────────────────────
//
// Every op is declared once and dispatched at runtime to the backend (CPU /
// CUDA / Metal) where its operand tensors live. Tensors are row-major
// (rows, cols) and carry a Dtype and a Device.
//
// Contract conventions used throughout:
//   - Output tensors are resized (and dtype-set, where noted) to the expected
//     shape if mis-shaped — except accumulation outputs (dW, dB, ...), which
//     the caller must pre-size and zero; the op adds into them.
//   - Backward gradient outputs are *overwritten* unless marked *accumulated*.
//   - Synchronisation is the caller's responsibility: call brotensor::sync()
//     before reading GPU results to host. CPU ops are synchronous.
//   - Backends throw std::runtime_error ("brotensor: <op>: <reason>") for
//     contract violations and for unimplemented ops.

// ─── Dense layers + elementwise activations ────────────────────────────────

// y = W*x + b.  W:(out,in)  b:(out,1)  x:(in,1)  y:(out,1) (resized).
void linear_forward(const Tensor& W, const Tensor& b,
                    const Tensor& x, Tensor& y);

// Backward of linear_forward. W:(out,in), x:(in,1), dY:(out,1) forward
// weights / input / upstream. dX:(in,1) overwritten. dW:(out,in), dB:(out,1)
// accumulated — caller zeros.
void linear_backward(const Tensor& W, const Tensor& x,
                     const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& dB);

// y = max(x, 0). Shapes match; y resized if mis-shaped. x and y may alias.
void relu_forward(const Tensor& x, Tensor& y);

// dX = dY * (x > 0). dX resized to match x; may alias dY.
void relu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// y = tanh(x). y resized to match x.
void tanh_forward(const Tensor& x, Tensor& y);

// dX = dY * (1 - y*y). `y` is the cached forward output (not raw x).
void tanh_backward(const Tensor& y, const Tensor& dY, Tensor& dX);

// y = 1 / (1 + exp(-x)).
void sigmoid_forward(const Tensor& x, Tensor& y);

// dX = dY * y * (1 - y). `y` is the cached forward output.
void sigmoid_backward(const Tensor& y, const Tensor& dY, Tensor& dX);

// y[i] += x[i]. Identical shape required.
void add_inplace(Tensor& y, const Tensor& x);

// y[i] += s. Dispatched FP32/FP16 on y.dtype.
void add_scalar_inplace(Tensor& y, float s);

// y[i] *= s. Dispatched FP32/FP16 on y.dtype.
void scale_inplace(Tensor& y, float s);

// y[i] = min(max(y[i], lo), hi), in place. Dispatched FP32/FP16 on y.dtype.
void clamp(Tensor& y, float lo, float hi);

// Build a slot-validity mask on-device. For k in [0, K):
//   mask[k] = (x[offset + k*stride] > 0.5f) ? 1.0f : 0.0f
// mask resized to (K, 1).
void build_slot_mask(const Tensor& x, int offset, int K, int stride,
                     Tensor& mask);

// ─── Softmax / LayerNorm / single-head attention ───────────────────────────

// Numerically stable softmax over a flat length-N vector.
//   logits, probs: (N,1) or (1,N), treated flat; probs resized to match.
//   mask: optional N-float device pointer (1 valid / 0 invalid); may be null.
//         Invalid positions contribute 0 to the normaliser and receive 0 in
//         probs. Caller guarantees >=1 valid entry when masking.
void softmax_forward(const Tensor& logits, Tensor& probs,
                     const float* mask = nullptr);

// Full-Jacobian softmax backward:
//   dLogits[i] = sum_j dProbs[j] * probs[j] * (delta_ij - probs[i]).
// All length-N; dLogits resized to match.
void softmax_backward(const Tensor& probs, const Tensor& dProbs,
                      Tensor& dLogits);

// LayerNorm forward over a single (N,1) vector.
//   x, gamma, beta: (N,1) input / learnable scale / learnable shift.
//   y:    (N,1) output.
//   xhat: (N,1) cached normalised x = (x - mean) * rstd.
//   mean_out, rstd_out: scalar caches written by the op (rstd = 1/sqrt(var+eps)).
//   eps:  variance epsilon, typically 1e-5f.
// y and xhat resized if mis-shaped. Backward consumes (xhat, gamma, rstd).
void layernorm_forward(const Tensor& x,
                       const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat,
                       float& mean_out, float& rstd_out,
                       float eps);

// LayerNorm backward. Dtype-dispatched (FP32/FP16); all tensors share dtype.
//   dY, xhat, gamma: (N,1) upstream / forward cache / forward scale.
//   rstd: scalar from forward.
//   dX: (N,1) overwritten (resized + dtype-set to match dY).
//   dGamma, dBeta: (N,1) accumulated — caller zeros.
void layernorm_backward(const Tensor& dY, const Tensor& xhat,
                        const Tensor& gamma, float rstd,
                        Tensor& dX,
                        Tensor& dGamma, Tensor& dBeta);

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
//   d_mask: optional length-K device mask (1 valid / 0 invalid); may be null.
//           Same semantics as single-head attention.
//   O: (K,D) output, resized if mis-shaped.
//   Backward caches (out-params, resized if mis-shaped): Qh, Kh, Vh
//   (num_heads*K, head_dim) with head h in rows [h*K, (h+1)*K); Attnh
//   (num_heads*K, K) per-head softmax weights; Yconcat (K,D) pre-Wo concat.
void mha_forward(const Tensor& X,
                 const Tensor& Wq, const Tensor& Wk,
                 const Tensor& Wv, const Tensor& Wo,
                 const float* d_mask,
                 int num_heads,
                 Tensor& Qh, Tensor& Kh, Tensor& Vh,
                 Tensor& Attnh, Tensor& Yconcat,
                 Tensor& O);

// Backward of mha_forward.
//   dO: (K,D) upstream.  X, Qh, Kh, Vh, Attnh, Yconcat: forward caches.
//   Wq, Wk, Wv, Wo: (D,D) forward weights.  d_mask: as forward (or null).
//   num_heads must match forward.
//   dX: (K,D) overwritten.  dWq, dWk, dWv, dWo: (D,D) accumulated — caller zeros.
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

// ─── Pooling / losses / embeddings / concat ────────────────────────────────

// Masked mean-pool over the rows of a (K,D) matrix.
//   X: (K,D).  d_mask: K-float device mask (1 valid / 0 invalid), or null
//   (all rows valid).  y: (D,1) output, resized if mis-shaped.
//   y[j] = (1/num_valid) * sum_{k : mask[k]==1} X[k,j];
//   all-zero output if num_valid == 0.
void masked_mean_pool_forward(const Tensor& X, const float* d_mask,
                              Tensor& y);

// Backward of masked_mean_pool.
//   dY: (D,1) upstream.  mask: as forward (or null).  K: original row count.
//   dX: (K,D) overwritten — valid rows get dY/num_valid, invalid rows get 0;
//   all-zero if num_valid == 0.
void masked_mean_pool_backward(const Tensor& dY, const float* d_mask,
                               int K, Tensor& dX);

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

// Embedding lookup: out[b,:] = table[d_idx[b],:].
//   table: (V,D) FP32 or FP16.  d_idx: B int32 indices, each in [0,V).
//   out: (B,D), resized AND dtype-set to match table if mis-shaped/-typed.
void embedding_lookup_forward(const Tensor& table,
                              const int32_t* d_idx, int B,
                              Tensor& out);

// Scatter-accumulate backward of embedding_lookup. Dtype-dispatched (FP32/FP16);
// dOut and dTable share dtype.
//   dOut: (B,D) upstream.  d_idx: the B forward indices.
//   dTable: (V,D) accumulated — caller zeros; repeated indices sum.
void embedding_lookup_backward(const Tensor& dOut,
                               const int32_t* d_idx, int B,
                               Tensor& dTable);

// Concatenate flat tensors end-to-end. Each part is treated as a flat buffer.
// out resized to (total,1), total = sum of part sizes; parts laid in order.
void concat_rows(const std::vector<const Tensor*>& parts,
                 Tensor& out);

// Inverse of concat_rows: copy disjoint segments of `in` into the flat buffers
// of `parts` (each *overwritten*, not accumulated). Part sizes must match the
// concat call; segments are laid end-to-end from offset 0 in `in`.
void split_rows(const Tensor& in,
                const std::vector<Tensor*>& parts);

// Batched column-block concat. Each part is (B, d_i) for a shared B; out
// becomes (B, sum_i d_i) with parts as per-row column blocks:
//   out[b, off_i + j] = parts[i][b, j].
void concat_batched_rows(const std::vector<const Tensor*>& parts,
                         Tensor& out);

// Channel-axis concat of NCHW tensors. Part i is (N, C_i*H*W) flat NCHW; out
// becomes (N, sum_i C_i*H*W) with channel blocks regrouped per sample:
//   out[n, (off_i+c)*H*W + h*W + w] = parts[i][n, c*H*W + h*W + w],
//   off_i = sum_{j<i} C_j.
// The correct U-Net skip-merge concat for N >= 1 (a flat concat_rows would
// interleave samples for N > 1). Dtype-dispatched (FP16/FP32); all parts share
// dtype. C_per_part.size() must equal parts.size().
void concat_nchw_channels(const std::vector<const Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          Tensor& out);

// Inverse of concat_nchw_channels: copy disjoint channel-axis slices of dY
// into per-source buffers — each parts[i] *overwritten* with channels
// [off_i, off_i+C_per_part[i]) of dY, off_i = sum_{j<i} C_per_part[j].
// Dtype-dispatched (FP32/FP16); parts resized AND dtype-set to match dY.
// C_per_part.size() must equal parts.size(); dY.cols must be
// N * sum(C_per_part) * H * W.
void concat_nchw_channels_backward(const Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts);

// Device-to-device chunk copy: copies `n` floats from src.data+src_off into
// dst.data+dst_off. Both treated as flat float buffers regardless of (rows,
// cols). Async on the default stream.
void copy_d2d(const Tensor& src, int src_off,
              Tensor& dst,       int dst_off,
              int n);

// Dtype cast: dst = src converted to out_dtype. dst resized to (src.rows,
// src.cols, out_dtype) on src's device. Supports FP32<->FP16, FP32<->BF16, and
// a same-dtype passthrough copy; other pairs throw. The standard
// mixed-precision primitive (low-precision weight <-> FP32 master copy).
void cast(const Tensor& src, Tensor& dst, Dtype out_dtype);

// Inference-only batched LayerNorm: R independent rows of length D, no caches,
// no host syncs. Use layernorm_forward when backward is needed.
//   X_RD: (R,D).  gamma, beta: (D,).  Y_RD: (R,D), resized if mis-shaped.
void layernorm_forward_inference_batched(const Tensor& X_RD,
                                         const Tensor& gamma,
                                         const Tensor& beta,
                                         Tensor& Y_RD,
                                         float eps);

// SGD with momentum, in place:
//   velocity = momentum*velocity + grad;  param -= lr*velocity.
// All three tensors share shape; caller zeros grad between batches.
void sgd_step(Tensor& param, Tensor& grad, Tensor& velocity,
              float lr, float momentum);

// Adam step, in place. `step` is the 1-based bias-correction counter.
//   m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g^2
//   param -= lr * (m/(1-b1^step)) / (sqrt(v/(1-b2^step)) + eps)
// All four tensors share shape.
void adam_step(Tensor& param, const Tensor& grad,
               Tensor& m, Tensor& v,
               float lr, float beta1, float beta2, float eps, int step);

// Deterministic xavier-uniform init of a Linear weight; rng_state is a
// splitmix64 state advanced in place. CPU-only (weights init on the host).
void xavier_init(Tensor& W, uint64_t& rng_state);

// ─── Batched inference-only variants ───────────────────────────────────────
//
// B independent forward passes in a single launch, forward-only. Tensors
// carrying B rows are (B, D) row-major: row b holds the b'th sample.

// Y[b,:] = W*X[b,:] + bias for b in [0,B).
//   W: (out,in).  bias: (out,1).  X_BD: (B,in).  Y_BD: (B,out) resized.
void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD);

// Elementwise ReLU / Tanh over (B,D). Y resized to match X; X and Y may alias.
void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD);
void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD);

// Y[i] += X[i] over (B,D). Identical shape required.
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD);

// ─── Batched training backward variants ────────────────────────────────────

// Linear backward over a B-row minibatch. Dtype-dispatched (FP32/FP16); all
// tensors share dtype.
//   dX[b] = W^T*dY[b];  dW += sum_b dY[b]*X[b]^T;  dB += sum_b dY[b].
//   W: (out,in) forward weights.  X_BD: (B,in) forward input.
//   dY_BD: (B,out) upstream.  dX_BD: (B,in) overwritten (resized + dtype-set).
//   dW: (out,in), dB: (out,1) accumulated — caller zeros.
void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD,
                             Tensor& dW, Tensor& dB);

// Elementwise activation backward over (B,D), same shapes throughout.
//   relu: dX = dY*(X>0), reads X_BD (forward input).
//   tanh: dX = dY*(1-Y*Y), reads Y_BD (forward output).
void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD,
                           Tensor& dX_BD);
void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD,
                           Tensor& dX_BD);

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

// ─── Conv2d / diffusion-vision ops ─────────────────────────────────────────
//
// NCHW tensors are carried as flat (rows, cols) buffers; the (N,C,H,W) dims
// are passed as int args. Unless noted, these ops are dtype-dispatched on the
// primary input (FP32 or FP16) with FP32 internal accumulation, and resize +
// dtype-set their outputs to match.

// 2D convolution, NCHW. Dispatched on X.dtype (FP32/FP16); Wt, bias, Y share
// it. FP32 accumulation.
//   X:    (N, C_in*H*W).
//   Wt:   (C_out, (C_in/groups)*kH*kW)  OIHW filter layout.
//   bias: (C_out,1) or null.
//   Y:    (N, C_out*H_out*W_out), resized + dtype-set to match X.
//   groups: divides C_in and C_out. Output channel c_out belongs to group
//           g = c_out/(C_out/groups) and reads only input channels
//           [g*(C_in/groups), (g+1)*(C_in/groups)). groups=1 is standard conv;
//           groups==C_in==C_out is depthwise (Wt becomes (C_out, kH*kW)).
//   H_out = (H + 2*pad_h - dil_h*(kH-1) - 1)/stride_h + 1   (W_out analogous).
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

// 2D convolution backward w.r.t. input. Dtype-dispatched (FP32/FP16); Wt and
// dY share dtype, dX matches. All conv hyperparams match the forward call.
//   dX[n,c_in,i,j] = sum over kernel taps (kh,kw) and the group's output
//   channels of dY[n,c_out,i_out,j_out]*Wt[c_out,c_in_local,kh,kw], where
//     i_out = (i + pad_h - dil_h*kh)/stride_h   (j_out analogous),
//   counted only when divisible by stride and in [0,H_out)x[0,W_out).
//   Wt: (C_out, (C_in/groups)*kH*kW)  forward filter, OIHW.
//   dY: (N, C_out*H_out*W_out)        upstream gradient.
//   dX: (N, C_in*H*W)                 overwritten, resized + dtype-set to dY.
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

// 2D convolution backward w.r.t. weights. Dtype-dispatched (FP32/FP16); X, dY,
// dWt share dtype. All conv hyperparams match the forward call.
//   dWt[c_out,c_in_local,kh,kw] += sum over (n,i_out,j_out) of
//     dY[n,c_out,i_out,j_out] *
//     X[n,c_in, stride_h*i_out-pad_h+dil_h*kh, stride_w*j_out-pad_w+dil_w*kw]
//   (OOB input reads treated as zero), with c_in = g*(C_in/groups)+c_in_local,
//   g = c_out/(C_out/groups).
//   X:   (N, C_in*H*W)               forward input.
//   dY:  (N, C_out*H_out*W_out)      upstream gradient.
//   dWt: (C_out, (C_in/groups)*kH*kW)  accumulated — caller zeros.
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

// 2D convolution backward w.r.t. bias. Dtype-dispatched (FP32/FP16); dY and dB
// share dtype.
//   dB[c_out] += sum over (n,i_out,j_out) of dY[n,c_out,i_out,j_out].
//   dY: (N, C_out*H_out*W_out)  upstream gradient.
//   dB: (C_out,1)               accumulated — caller zeros.
void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB);

// GroupNorm forward, NCHW. Dispatched on X.dtype (FP32/FP16); gamma, beta, Y
// share it. FP32 accumulation.
//   X, Y: (N, C*H*W).  gamma, beta: (C,1) per-channel scale / shift.
//   num_groups divides C; mean/var computed over (C/num_groups, H, W) within
//   each (n, group) tile. eps typically 1e-5f. Y resized + dtype-set to X.
void group_norm_forward(const Tensor& X,
                        const Tensor& gamma,
                        const Tensor& beta,
                        int N, int C, int H, int W,
                        int num_groups,
                        float eps,
                        Tensor& Y);

// GroupNorm backward, NCHW. Dispatched on X.dtype (FP32/FP16); gamma and dY
// share it. FP32 accumulation. Mean/rstd are recomputed per (n, group) tile
// from X (no forward cache needed).
//   Per tile of M = (C/num_groups)*H*W elements:
//     xhat = (x-mean)*rstd;  dxhat = dY*gamma_c
//     dX = rstd*(dxhat - (sum dxhat + xhat*sum(dxhat*xhat))/M)
//   dGamma_c += sum dY*xhat;  dBeta_c += sum dY.
//   X, dY, dX: (N, C*H*W); dX overwritten (resized + dtype-set to X).
//   gamma: (C,1) forward scale.  dGamma, dBeta: (C,1) accumulated — caller zeros.
void group_norm_backward(const Tensor& X,
                         const Tensor& gamma,
                         const Tensor& dY,
                         int N, int C, int H, int W,
                         int num_groups,
                         float eps,
                         Tensor& dX,
                         Tensor& dGamma,
                         Tensor& dBeta);

// SiLU / Swish: y = x*sigmoid(x). Dispatched FP32/FP16 on x.dtype; y resized +
// dtype-set to match x. x and y may alias.
void silu_forward(const Tensor& x, Tensor& y);

// SiLU backward, reads the raw forward input x:
//   dX = dY * sigmoid(x) * (1 + x*(1-sigmoid(x))).
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void silu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// GELU, tanh approximation (PyTorch approximate="tanh"):
//   y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
// Dispatched FP32/FP16 on x.dtype.
void gelu_forward(const Tensor& x, Tensor& y);

// GELU (tanh-approx) backward, reads x. With k=sqrt(2/pi),
// u=k*(x+0.044715*x^3), t=tanh(u):
//   dX = dY * [0.5*(1+t) + 0.5*x*(1-t^2)*k*(1+3*0.044715*x^2)].
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void gelu_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// Exact GELU (erf form, PyTorch approximate="none", diffusers default):
//   y = 0.5*x*(1 + erf(x/sqrt(2))).
// Distinct from the tanh-approx gelu_forward. Dispatched FP32/FP16 on x.dtype;
// y resized + dtype-set to match x. x and y may alias.
void gelu_exact_forward(const Tensor& x, Tensor& y);

// Exact-GELU backward, reads x:
//   dX = dY * [0.5*(1+erf(x/sqrt(2))) + x*phi(x)],  phi = standard normal pdf.
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void gelu_exact_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX);

// QuickGELU: y = x*sigmoid(1.702*x). OpenAI CLIP's activation (SD1.5 text
// encoder). Dispatched FP32/FP16 on x.dtype; y resized + dtype-set to match x.
// x and y may alias.
void quick_gelu_forward(const Tensor& x, Tensor& y);

// QuickGELU backward, reads x. With s = sigmoid(1.702*x):
//   dX = dY * (s + x*1.702*s*(1-s)).
// Dispatched FP32/FP16 on x.dtype; dX resized + dtype-set to match x; may alias dY.
void quick_gelu_backward(const Tensor& x, const Tensor& dY,
                         Tensor& dX);

// 2x nearest-neighbour spatial upsample, NCHW. Output pixel (i,j) reads X at
// (i/2, j/2). Dispatched FP32/FP16 on X.dtype; Y resized + dtype-set to match X.
//   X: (N, C*H*W).  Y: (N, C*2H*2W).
void upsample_nearest_2x(const Tensor& X,
                         int N, int C, int H, int W,
                         Tensor& Y);

// 2x bilinear spatial upsample, NCHW, align_corners=False (PyTorch default for
// interpolate(scale_factor=2)). Dispatched FP32/FP16 on X.dtype; FP32 math;
// Y resized + dtype-set to match X.
//   X: (N, C*H*W).  Y: (N, C*2H*2W).
void upsample_bilinear_2x(const Tensor& X,
                          int N, int C, int H, int W,
                          Tensor& Y);

// 2x average-pool downsample, NCHW, stride 2 / kernel 2 / no padding; H and W
// must be even. Dispatched FP32/FP16 on X.dtype; FP32 math; Y resized + dtype-set.
//   X: (N, C*H*W).  Y: (N, C*H/2*W/2).
void downsample_avg_2x(const Tensor& X,
                       int N, int C, int H, int W,
                       Tensor& Y);

// Backward of upsample_nearest_2x:
//   dX[n,c,i,j] = sum_{a,b in {0,1}} dY[n,c,2i+a,2j+b].
// N,C,H,W are the INPUT (pre-upsample) dims. Dispatched FP32/FP16 on dY.dtype;
// FP32 accumulation; dX (N,C*H*W) overwritten, resized + dtype-set to dY.
void upsample_nearest_2x_backward(const Tensor& dY,
                                  int N, int C, int H, int W,
                                  Tensor& dX);

// Backward of upsample_bilinear_2x (align_corners=False): scatters each
// dY pixel into its 4 source pixels with the forward's bilinear weights.
// N,C,H,W are the INPUT dims. Dispatched FP32/FP16 on dY.dtype; FP32
// accumulation; dX (N,C*H*W) overwritten, resized + dtype-set to dY.
void upsample_bilinear_2x_backward(const Tensor& dY,
                                   int N, int C, int H, int W,
                                   Tensor& dX);

// Backward of downsample_avg_2x:
//   dX[n,c,2*i+a,2*j+b] = (1/4) * dY[n,c,i,j].
// N,C,H,W are the INPUT dims, H and W even. Dispatched FP32/FP16 on dY.dtype;
// FP32 accumulation; dX (N,C*H*W) overwritten, resized + dtype-set to dY.
void downsample_avg_2x_backward(const Tensor& dY,
                                int N, int C, int H, int W,
                                Tensor& dX);

// FP16 batched linear forward, inference-only. Like linear_forward_batched but
// FP16 storage throughout.
//   W: (out,in).  bias: (out,1) or null.  X_BD: (B,in).  Y_BD: (B,out) resized.
void linear_forward_batched_fp16(const Tensor& W, const Tensor* bias,
                                 const Tensor& X_BD, Tensor& Y_BD);

// y[i] *= x[i]. Identical shape and dtype; dispatched FP32/FP16 on y.dtype.
void mul_inplace(Tensor& y, const Tensor& x);

// ─── AdaLN modulation + GEGLU (DiT / SD3 / Flux) ───────────────────────────

// AdaLN modulation: Y = X*(1+scale) + shift, with scale/shift broadcast across
// every token row — the affine step every DiT block applies after norm().
//   X, Y: (L,D) token activations.
//   scale, shift: length-D vectors ((1,D) or (D,1)), same dtype/device as X.
//   Y resized + dtype-set to match X. Dispatched on X.dtype (FP32/FP16/BF16);
//   FP32 math.
void modulate(const Tensor& X, const Tensor& scale, const Tensor& shift,
              Tensor& Y);

// Broadcast channel-wise multiply: Y[l,d] = X[l,d]*v[d], v broadcast across
// every token row — the DiT residual gate and any per-channel rescale.
//   X, Y: (L,D).  v: length-D vector ((1,D) or (D,1)), same dtype/device as X.
//   Y resized + dtype-set to match X. Dispatched on X.dtype (FP32/FP16/BF16);
//   FP32 math.
void broadcast_mul(const Tensor& X, const Tensor& v, Tensor& Y);

// GEGLU: input (B,2*D) split along the last dim into A=(B,D) and B_half=(B,D);
// output (B,D) = A * gelu(B_half) (tanh-approx). Dispatched FP32/FP16 on
// X.dtype; Y resized + dtype-set to match X.
void geglu_forward(const Tensor& X, Tensor& Y);

// GEGLU backward. With g = gelu(B_half) (tanh-approx):
//   dA = dY*g;   dB_half = dY*A*gelu'(B_half).
// dX = concat(dA, dB_half) along the last dim (A then B_half). Dispatched
// FP32/FP16 on X.dtype; dX resized + dtype-set to match X.
void geglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX);

// Exact-GELU GEGLU: same split as geglu_forward but output = A*gelu_exact(B_half),
// using the exact erf-based GELU. Matches diffusers' default GEGLU. Dispatched
// FP32/FP16 on X.dtype; Y resized + dtype-set to match X.
void geglu_exact_forward(const Tensor& X, Tensor& Y);

// Exact-GELU GEGLU backward. With g = gelu_exact(B_half):
//   dA = dY*g;   dB_half = dY*A*gelu_exact'(B_half).
// dX = concat(dA, dB_half) along the last dim (A then B_half). Dispatched
// FP32/FP16 on X.dtype; dX resized + dtype-set to match X.
void geglu_exact_backward(const Tensor& X, const Tensor& dY,
                          Tensor& dX);

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

// FP16 inference-only batched LayerNorm: R rows of length D, FP32
// accumulation, no caches.
//   X_RD, Y_RD: (R,D) FP16.  gamma, beta: (D,) FP16.  Y_RD resized as needed.
void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              const Tensor& beta,
                                              Tensor& Y_RD,
                                              float eps);

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

// ─── NCHW <-> sequence transposes ──────────────────────────────────────────

// NCHW <-> sequence (token) layout transpose — lets (L,D)-token ops consume
// tensors from NCHW primitives (conv2d, group_norm, resblock) and back. Pure
// gather/scatter, no math. Dispatched on X.dtype (FP32/FP16); Y resized +
// dtype-set to match X. X and Y must not alias.
//   nchw_to_sequence: X (N,C*H*W) -> Y (N*H*W, C); Y[n*H*W+h*W+w,c]=X[n,c,h,w].
//   sequence_to_nchw: the inverse.
void nchw_to_sequence(const Tensor& X,
                      int N, int C, int H, int W,
                      Tensor& Y);

void sequence_to_nchw(const Tensor& X,
                      int N, int C, int H, int W,
                      Tensor& Y);

// ─── Diffusion ResBlock ────────────────────────────────────────────────────

// Fused diffusion ResBlock (FP16, inference-only). Computes the SD U-Net
// residual block in one op:
//   h = silu(group_norm(X, gamma1, beta1))
//   h = conv2d_3x3_same(h, W1, b1)
//   if t_emb_shift: h += broadcast(t_emb_shift)          // (N,C_out) or (C_out,)
//   h = silu(group_norm(h, gamma2, beta2))
//   h = conv2d_3x3_same(h, W2, b2)
//   Y = h + (C_in==C_out && !Wskip ? X : conv2d_1x1(X, Wskip, bskip))
// All tensors FP16; OIHW conv filter layout.
//   X: (N,C_in*H*W).  gamma1, beta1: (C_in,1).  W1: (C_out,C_in*9); b1: (C_out,1)/null.
//   t_emb_shift: (N,C_out) or (C_out,1) or null.
//   gamma2, beta2: (C_out,1).  W2: (C_out,C_out*9); b2: (C_out,1)/null.
//   Wskip: (C_out,C_in*1) 1x1, or null when C_in==C_out;  bskip: (C_out,1)/null.
//   Y: (N,C_out*H*W), resized as needed.
//   num_groups divides C_in and C_out (typically 32); eps default 1e-5.
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

// W8A16 variant of resblock_forward (inference-only). Same math, but conv1,
// conv2, and the optional 1x1 skip conv take INT8 weights with per-output-row
// FP32 scales (the conv2d_int8w_fp16_forward W8A16 contract). Activations, GN
// params, biases, and t_emb_shift stay FP16. No backward.
//   As resblock_forward, with each W*_int8 (INT8) paired with s* (C_out,1) FP32
//   dequant scales; sskip is required iff Wskip_int8 != null.
//   num_groups divides C_in and C_out (typically 32); eps default 1e-5.
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

// Backward of resblock_forward (FP16). Composed from the public group_norm,
// silu, conv2d, and add ops. Forward intermediates are not cached, so they are
// recomputed from X and the parameters, then dY is routed back through
// SiLU/GN/conv1/conv2/t_emb_shift and summed with the residual (identity- or
// conv1x1-skip) gradient. num_groups/eps must match the forward.
//   X, gamma1, beta1, W1, b1, t_emb_shift, gamma2, beta2, W2, b2, Wskip,
//     bskip: forward inputs (read-only).
//   dY: (N,C_out*H*W) upstream.   dX: (N,C_in*H*W) overwritten.
//   dGamma1, dBeta1: (C_in,1);  dW1: (C_out,C_in*9);  dGamma2, dBeta2: (C_out,1);
//   dW2: (C_out,C_out*9);  dWskip: (C_out,C_in*1) — all accumulated, caller zeros.
//   db1, db2, dbskip, dt_emb_shift: accumulated iff non-null and the matching
//     forward input was used; pass null otherwise.
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

// ─── Llama-style transformer ops ───────────────────────────────────────────

// Row-major matrix multiply, no bias: C(M,N) = A(M,K) @ B(K,N).
// Dispatched on A.dtype; B and C share it (C resized + dtype-set to match A).
// FP32 accumulation for both the FP32 and FP16 paths.
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Backward of matmul. For C = A @ B:
//   dA(M,K) += dC(M,N) @ B^T;   dB(K,N) += A^T @ dC(M,N).
// Dtype-dispatched (FP32/FP16); all five tensors share dtype. dC is read-only;
// dA, dB are accumulated — caller pre-sizes and zeros (mirrors linear_backward).
//   A: (M,K) forward input.  B: (K,N) forward weight.  dC: (M,N) upstream.
void matmul_backward(const Tensor& A,
                     const Tensor& B,
                     const Tensor& dC,
                     Tensor& dA,
                     Tensor& dB);

// RoPE (rotary position embedding) forward, per head and dimension pair i:
//   x_{2i}   <- x_{2i}*cos(t) - x_{2i+1}*sin(t)
//   x_{2i+1} <- x_{2i}*sin(t) + x_{2i+1}*cos(t)
//   t = pos * theta_base^{-2i/head_dim};  pos = row index + seq_offset.
//   X, Y: (L, num_heads*head_dim); head_dim even. Y resized + dtype-set to X.
// Dispatched on X.dtype (FP32/FP16).
void rope_forward(const Tensor& X, int head_dim, int num_heads,
                 int seq_offset, float theta_base, Tensor& Y);

// RoPE backward — the inverse (transpose) rotation of rope_forward:
//   dX_{2i}   <-  dY_{2i}*cos(t) + dY_{2i+1}*sin(t)
//   dX_{2i+1} <- -dY_{2i}*sin(t) + dY_{2i+1}*cos(t)
//   dX, dY: (L, num_heads*head_dim). dX resized + dtype-set to match dY.
// Dispatched on dY.dtype.
void rope_backward(const Tensor& dY, int head_dim, int num_heads,
                  int seq_offset, float theta_base, Tensor& dX);

// RoPE with explicit caller-supplied cos/sin tables — the caller owns all
// position semantics (arbitrary position ids, 2D axial RoPE for Flux/SD3).
//   x_{2i}   <- x_{2i}*cos_tbl[row,i] - x_{2i+1}*sin_tbl[row,i]
//   x_{2i+1} <- x_{2i}*sin_tbl[row,i] + x_{2i+1}*cos_tbl[row,i]
//   X, Y: (L, num_heads*head_dim); head_dim even.
//   cos_tbl, sin_tbl: (L, head_dim/2) FP32 (any backend), shared across heads.
//   Y resized + dtype-set to X. Dispatched on X.dtype (FP32/FP16/BF16); FP32 math.
void rope_apply(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                int head_dim, int num_heads, Tensor& Y);

// Backward of rope_apply — the inverse (transpose) rotation:
//   dX_{2i}   <-  dY_{2i}*cos_tbl[row,i] + dY_{2i+1}*sin_tbl[row,i]
//   dX_{2i+1} <- -dY_{2i}*sin_tbl[row,i] + dY_{2i+1}*cos_tbl[row,i]
//   dX, dY: (L, num_heads*head_dim). cos_tbl/sin_tbl as in rope_apply.
// Dispatched on dY.dtype.
void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl,
                         const Tensor& sin_tbl, int head_dim, int num_heads,
                         Tensor& dX);

// RMSNorm forward, per row:
//   rms[b] = sqrt(mean_j x[b,j]^2 + eps);  y[b,j] = x[b,j]*gamma[j]/rms[b].
//   X, Y: (B,D).  gamma: (D,1), same dtype as X.  Y resized + dtype-set to X.
// Dispatched on X.dtype (FP32/FP16); FP32 accumulation.
void rms_norm_forward(const Tensor& X, const Tensor& gamma,
                     float eps, Tensor& Y);

// RMSNorm backward.
//   X: (B,D) forward input.  gamma: (D,1) forward scale.  dY: (B,D) upstream.
//   dX: (B,D) overwritten (resized + dtype-set to X).
//   dGamma: (D,1) accumulated — caller zeros. Dtype matches X.
void rms_norm_backward(const Tensor& X, const Tensor& gamma,
                      const Tensor& dY, float eps,
                      Tensor& dX, Tensor& dGamma);

// SwiGLU (Llama FFN gate): input (B,2*D) split along the last dim into A=(B,D)
// and B_half=(B,D); output (B,D) = silu(A) * B_half. Dispatched FP32/FP16 on
// X.dtype; Y resized + dtype-set to match X.
void swiglu_forward(const Tensor& X, Tensor& Y);

// SwiGLU backward. With s = silu(A):
//   dA = dY*B_half*silu'(A);   dB_half = dY*s.
// dX = concat(dA, dB_half) along the last dim (A then B_half). Dispatched on
// X.dtype; dX resized + dtype-set to match X.
void swiglu_backward(const Tensor& X, const Tensor& dY,
                    Tensor& dX);

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

// ─── Public reductions ─────────────────────────────────────────────────────

// Row-wise sum: Y[m,0] = sum_n X[m,n]. X:(M,N), Y:(M,1) — same dtype
// (FP32/FP16), resized as needed.
void sum_rows(const Tensor& X, Tensor& Y);

// Column-wise sum: Y[0,n] = sum_m X[m,n]. X:(M,N), Y:(1,N) — same dtype
// (FP32/FP16), resized as needed.
void sum_cols(const Tensor& X, Tensor& Y);

// Row-wise argmax: Idx[m,0] = argmax_n X[m,n], stored as the integer index
// cast to float. X:(M,N) FP32/FP16, Idx:(M,1) FP32, resized. Ties keep the
// lowest index.
void argmax_rows(const Tensor& X, Tensor& Idx);

// ─── Diffusion sampler steps + timestep embedding ──────────────────────────

// Fused DDIM step (FP16), elementwise:
//   x0_pred = (x_t - sqrt(1-alpha_t)*eps_pred) / sqrt(alpha_t)
//   x_prev  = sqrt(alpha_prev)*x0_pred + sqrt(1-alpha_prev-sigma_t^2)*eps_pred
// sigma_t = 0 gives deterministic DDIM. FP16 in/out, FP32 math. x_t and
// eps_pred share shape; x_prev resized to match.
void ddim_step(const Tensor& x_t, const Tensor& eps_pred,
               float alpha_t, float alpha_prev, float sigma_t,
               Tensor& x_prev);

// Fused first-order Euler sampler step (FP16), elementwise:
//   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred
// The kernel never interprets eps_pred — it covers both the eps/k-diffusion
// derivative form and flow-matching velocity (Flux/SD3); pass the model term
// as eps_pred with the corresponding sigma schedule. FP16 in/out, FP32 math.
// x_t and eps_pred share shape; x_prev resized to match.
void euler_step(const Tensor& x_t, const Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                Tensor& x_prev);

// Fused DPM-Solver++ 2M sampler step (FP16), multistep, eps-prediction. The
// caller maintains a running x0 cache and computes the three coefficients
// host-side. The kernel reconstructs:
//   x0_t   = x_t - sigma_t*eps_pred
//   x_prev = c_xt*x_t + c_x0t*x0_t + c_x0prev*x0_prev
//   x0_out = x0_t            (caller copies into x0_prev for the next step)
// Coefficients (k-diffusion DPM++ 2M, eps-prediction, alpha==1; h = log-SNR
// step, h_last the previous step, r = h_last/h):
//   c_xt     = sigma_next / sigma_t
//   c_x0t    = -(exp(-h)-1) * (1 + 1/(2r))
//   c_x0prev = -(exp(-h)-1) * (-1/(2r))
// First step (no x0_prev cached): use euler_step. All tensors FP16, same
// shape; x_prev and x0_out resized to match.
void dpmpp_2m_step(const Tensor& x_t, const Tensor& eps_pred,
                   const Tensor& x0_prev,
                   float sigma_t,
                   float c_xt, float c_x0t, float c_x0prev,
                   Tensor& x_prev, Tensor& x0_out);

// Sinusoidal timestep embedding (FP32). Matches diffusers'
// get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0
// (the SD/SDXL default). With half = dim/2:
//   freqs[j]         = exp(-log(max_period) * j / half),  j in [0, half)
//   Y[i, 0:half]     = cos(timesteps[i] * freqs)
//   Y[i, half:2half] = sin(timesteps[i] * freqs)
//   Y[i, dim-1]      = 0   if dim is odd
//   timesteps: (N,1) FP32.  Y: (N,dim) FP32, resized as needed.
void timestep_embedding(const Tensor& timesteps,
                        int dim, float max_period,
                        Tensor& Y);

// ─── INT8 weight-only quantisation (W8A16) ─────────────────────────────────

// Host helper: quantise an FP16 weight matrix to per-output-row symmetric INT8.
// Operates on plain host buffers — not device-dispatched.
//   W_fp16: (out,in) FP16 bit patterns.  W_int8_out: out*in int8, row-major.
//   scales_out: `out` FP32 scales.
//   scale[row] = max(|w|)/127 (0 if the row is all zero);
//   quantised w = clamp(round(w/scale), -127, 127).
void quantize_int8_per_row_host(const uint16_t* W_fp16,
                                int out, int in,
                                int8_t* W_int8_out,
                                float* scales_out);

// W8A16 matmul: Y = dequant(W_int8, scales) @ X.
//   W_int8: (out,in) INT8.  scales: (out,1) FP32 per-row dequant scales.
//   X: (in,B) FP16.  Y: (out,B) FP16, resized as needed.
// Same (M,K)@(K,N) shape convention as matmul.
void matmul_int8w_fp16(const Tensor& W_int8,
                       const Tensor& scales,
                       const Tensor& X,
                       Tensor& Y);

// W8A16 conv2d forward. Mirrors conv2d_forward; only the weight dtype differs.
//   W_int8: (C_out, C_in/groups*kH*kW) INT8 OIHW, quantised per output channel.
//   scales: (C_out,1) FP32 per-output-channel dequant scales.
//   bias: (C_out,1) FP16 or null.  X, Y: FP16, layout as conv2d_forward.
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

// W8A16 batched linear: Y(B,out) = X(B,in) @ dequant(W_int8)^T + bias. Same
// (B,in)->(B,out) layout as linear_forward_batched_fp16. FP32 accumulation.
//   W_int8: (out,in) INT8.  scales: (out,1) FP32 per-output-row dequant scales.
//   bias: (out,1) or (1,out) FP16, or null.  X_BD: (B,in) FP16.
//   Y_BD: (B,out) FP16, resized as needed.
void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD,
                                       Tensor& Y_BD);

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

// ─── Spectral / FFT core (audio) ───────────────────────────────────────────
//
// Audio primitives for STT / TTS / neural-codec models. Implemented on all
// three backends (CPU / CUDA / Metal), FP32 on every backend; ops throw
// "brotensor: <op>: <reason>" for a non-FP32 or wrong-device tensor.
//
// Complex layout: there is no complex Dtype. A complex tensor is an FP32
// Tensor with the bin axis interleaved [re,im,re,im,...]; a C-bin spectrum
// over R rows is an (R, 2*C) FP32 tensor (column 2*c real, 2*c+1 imaginary).
// Real tensors keep the natural (R, C) shape. All sizes >= 1 are supported
// (mixed-radix core with a Bluestein fallback for large/prime factors).
//
// Normalisation: numpy's "backward" convention — the forward transform
// (fft/rfft) is unscaled, the inverse (ifft/irfft) is scaled by 1/N.
//
// Gradients: all four transforms are linear. fft/ifft have NO backward op —
// the adjoint is the other transform plus a scalar:
//   grad of y=fft(x):  grad_x = ifft(grad_y); scale_inplace(grad_x, N)
//   grad of y=ifft(x): grad_x = fft(grad_y);  scale_inplace(grad_x, 1/N)
// rfft/irfft DO have explicit backward ops: they fold the Hermitian half and
// carry bin weighting, so they are not mutual transposes up to a scalar.
// Outputs are resized to the documented shape when mis-shaped, except
// complex_mul_backward's dA/dB, which accumulate (caller pre-sizes and zeros).

// Complex elementwise multiply: y = a*b per bin.
//   a, b, y: interleaved-complex (R, 2*C); a and b share shape; y resized to match.
void complex_mul(const Tensor& a, const Tensor& b, Tensor& y);

// Backward of complex_mul. For y = a*b: dA = dY*conj(b), dB = dY*conj(a).
//   a, b, dY, dA, dB: interleaved-complex (R, 2*C), all the same shape.
//   dA, dB accumulated — caller pre-sizes and zeros.
void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY,
                          Tensor& dA, Tensor& dB);

// Complex magnitude: y[r,c] = sqrt(z.re^2 + z.im^2).
//   z: interleaved-complex (R, 2*C).  y: REAL (R, C), resized if mis-shaped.
void complex_abs(const Tensor& z, Tensor& y);

// Backward of complex_abs. With r = |z|: dZ.re = dY*z.re/r, dZ.im = dY*z.im/r
// (gradient set to 0 at r == 0).
//   z: interleaved-complex (R, 2*C).  dY: REAL (R, C).
//   dZ: interleaved-complex (R, 2*C), overwritten (resized if mis-shaped).
void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ);

// Complex phase: y = atan2(z.im, z.re) per bin, radians (-pi, pi].
//   z: interleaved-complex (R, 2*C).  y: REAL (R, C), resized if mis-shaped.
// No backward (non-differentiable at the origin).
void complex_angle(const Tensor& z, Tensor& y);

// Build a complex tensor from polar form: y = mag*exp(i*phase), i.e.
// y.re = mag*cos(phase), y.im = mag*sin(phase).
//   mag, phase: REAL (R, C), same shape.
//   y: interleaved-complex (R, 2*C), resized if mis-shaped.
void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y);

// Forward FFT (complex->complex), one signal per row.
//   x, y: interleaved-complex (R, 2*N); y resized to match x.
// "backward" normalisation (unscaled). No fft_backward — see the section note.
void fft(const Tensor& x, Tensor& y);

// Inverse FFT (complex->complex), one signal per row.
//   x, y: interleaved-complex (R, 2*N); y resized to match x.
// "backward" normalisation (scaled by 1/N). No ifft_backward — see section note.
void ifft(const Tensor& x, Tensor& y);

// Real-input FFT: real signal -> non-redundant half-spectrum.
//   x: REAL (R, L), one length-L signal per row.
//   y: interleaved-complex (R, 2*(L/2+1)) — bins 0..L/2, resized if mis-shaped.
// Unscaled. Backward is rfft_backward.
void rfft(const Tensor& x, Tensor& y);

// Inverse real FFT: half-spectrum -> real signal.
//   x: interleaved-complex (R, 2*(L/2+1)) half-spectrum.
//   L: output signal length — required (a C-bin half-spectrum is ambiguous
//      between L=2*(C-1) and 2*C-1); throws unless C == L/2+1.
//   y: REAL (R, L), resized if mis-shaped.
// Scaled by 1/L. Backward is irfft_backward.
void irfft(const Tensor& x, int L, Tensor& y);

// Backward (adjoint) of rfft: half-spectrum gradient -> real-signal gradient.
//   dY: interleaved-complex (R, 2*(L/2+1)).  L: original signal length
//       (dY.cols/2 must equal L/2+1).  dX: REAL (R, L), overwritten.
// Interior bins (all but DC, and Nyquist when L is even) are weighted by 2.
void rfft_backward(const Tensor& dY, int L, Tensor& dX);

// Backward (adjoint) of irfft: real-signal gradient -> half-spectrum gradient.
//   dY: REAL (R, L).  dX: interleaved-complex (R, 2*(L/2+1)), overwritten
//       (L inferred from dY.cols).
// Carries the 1/L scaling and irfft's bin weighting; transpose of rfft_backward.
void irfft_backward(const Tensor& dY, Tensor& dX);

// ─── STFT / iSTFT (audio) ──────────────────────────────────────────────────
//
// Short-time Fourier transform and its inverse. CPU / CUDA / Metal, FP32-only.
//
// Shapes: a length-L real signal is one row of an (N, L) real tensor (N
// batched signals, N passed as an int). The complex spectrogram is
// (N*frames, 2*bins) interleaved-complex — one frame per row, each signal's
// frame block stacked in order; bins = n_fft/2+1.
//
// Framing: each frame takes win_length samples, multiplies by the caller's
// real (1, win_length) `window`, centres them in an n_fft buffer
// (win_length <= n_fft), and runs rfft. Frame f starts at sample
// f*hop_length - (center ? n_fft/2 : 0).
//   center == false: frames = 1 + (L - n_fft)/hop_length (requires L >= n_fft).
//   center == true:  signal is reflect-padded by n_fft/2 each side
//                    (torch.stft(center=True)); frames = 1 + L/hop_length.
//
// Normalisation: FFT "backward" convention; `normalized == true` additionally
// scales the forward transform by 1/sqrt(n_fft) (istft by the reciprocal).
//
// istft is windowed overlap-add divided per sample by the overlap-added
// squared window (the COLA envelope); with a COLA-satisfying window+hop,
// istft(stft(x)) == x. signal_len is passed explicitly so the output length is
// unambiguous. stft and istft are linear but NOT mutual adjoints once window +
// COLA are folded in, so stft_backward / istft_backward are explicit ops —
// each the exact transpose of its forward map.

// Short-time Fourier transform: real signal -> complex spectrogram.
//   signal: REAL (N, signal_len).  window: REAL (1, win_length).
//   spec: interleaved-complex (N*frames, 2*(n_fft/2+1)), resized if mis-shaped.
// win_length <= n_fft. See the section note for framing / normalisation.
void stft(const Tensor& signal, const Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, Tensor& spec);

// Backward (adjoint) of stft: spectrogram gradient -> signal gradient.
//   dSpec: interleaved-complex (N*frames, 2*(n_fft/2+1)).
//   window: the forward analysis window.  dSignal: REAL (N, signal_len),
//   overwritten. All frame params must match the forward call.
void stft_backward(const Tensor& dSpec, const Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   Tensor& dSignal);

// Inverse STFT: complex spectrogram -> real signal, windowed overlap-add with
// COLA normalisation.
//   spec: interleaved-complex (N*frames, 2*(n_fft/2+1)).
//   window: REAL (1, win_length) (use the forward window for a clean round trip).
//   signal: REAL (N, signal_len), resized if mis-shaped.
void istft(const Tensor& spec, const Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, Tensor& signal);

// Backward (adjoint) of istft: signal gradient -> spectrogram gradient.
// Transposes the COLA division as well as the overlap-add and irfft.
//   dSignal: REAL (N, signal_len).  window: the forward synthesis window.
//   dSpec: interleaved-complex (N*frames, 2*(n_fft/2+1)), overwritten.
void istft_backward(const Tensor& dSignal, const Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    Tensor& dSpec);

// ─── 1D convolution family (audio) ─────────────────────────────────────────
//
// The audio counterpart of the conv2d family — the building block of
// WaveNet / Conformer / vocoder stacks.
//
// Layout (NCL): a 1D-conv activation is (N, C*L) — N signals folded into rows,
// each row a flat C-major / L-minor buffer X[(n*C+c)*L + l] (the NCHW
// convention with the height axis dropped). Weights are OIL:
// Wt[(c_out*(C_in/groups)+c_in_local)*kL + kl].
//
// conv1d, its three backward halves, and conv1d_int8w_fp16 are header-only
// inline wrappers over the conv2d ops (a 1D conv is a 2D conv with H=kH=1), so
// every backend that implements conv2d gets conv1d for free; conv1d_int8w_fp16
// therefore throws on the CPU backend (conv2d_int8w is a null CPU slot).
// causal_conv1d is likewise a wrapper (left-pad, then a valid conv1d).
// pad1d, conv_transpose1d, and causal_conv1d_update are genuinely new ops with
// their own vtable rows, implemented on all three backends (CPU / CUDA / Metal).

// Pad the length axis of an NCL tensor by pad_left / pad_right samples — the
// temporal analogue of an image pad (causal-conv left padding, "same" padding,
// reflect padding). `mode`: 0 zero, 1 reflect (mirror without repeating the
// edge sample; requires pad < L), 2 replicate (clamp to the edge sample).
//   X: (N, C*L).  Y: (N, C*(L+pad_left+pad_right)), resized + dtype-set to X.
void pad1d_forward(const Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode, Tensor& Y);

// Backward (adjoint) of pad1d: each input sample sums the gradients of the
// output samples that read it. dX overwritten (resized + dtype-set to dY).
//   dY: (N, C*(L+pad_left+pad_right)).  dX: (N, C*L).
// N, C, L and the pad / mode args match the forward call.
void pad1d_backward(const Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode, Tensor& dX);

// 1D convolution, NCL. Header-only wrapper over conv2d_forward (H=kH=1).
//   X: (N,C_in*L).  Wt: (C_out, (C_in/groups)*kL) OIL.  bias: (C_out,1) or null.
//   Y: (N,C_out*L_out);  L_out = (L+2*padding-dilation*(kL-1)-1)/stride + 1.
inline void conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                   int N, int C_in, int L, int C_out, int kL,
                   int stride, int padding, int dilation, int groups,
                   Tensor& Y) {
    conv2d_forward(X, Wt, bias, N, C_in, /*H=*/1, /*W=*/L, C_out,
                   /*kH=*/1, /*kW=*/kL, /*stride_h=*/1, /*stride_w=*/stride,
                   /*pad_h=*/0, /*pad_w=*/padding, /*dil_h=*/1, /*dil_w=*/dilation,
                   groups, Y);
}
// Convenience overload: groups defaults to 1.
inline void conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                   int N, int C_in, int L, int C_out, int kL,
                   int stride, int padding, int dilation, Tensor& Y) {
    conv1d(X, Wt, bias, N, C_in, L, C_out, kL, stride, padding, dilation,
           /*groups=*/1, Y);
}

// conv1d backward w.r.t. input — wrapper over conv2d_backward_input (H=kH=1).
// dX overwritten.
inline void conv1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                  int N, int C_in, int L, int C_out, int kL,
                                  int stride, int padding, int dilation,
                                  int groups, Tensor& dX) {
    conv2d_backward_input(Wt, dY, N, C_in, /*H=*/1, /*W=*/L, C_out,
                          /*kH=*/1, /*kW=*/kL, 1, stride, 0, padding,
                          1, dilation, groups, dX);
}
inline void conv1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                  int N, int C_in, int L, int C_out, int kL,
                                  int stride, int padding, int dilation,
                                  Tensor& dX) {
    conv1d_backward_input(Wt, dY, N, C_in, L, C_out, kL, stride, padding,
                          dilation, /*groups=*/1, dX);
}

// conv1d backward w.r.t. weight — wrapper over conv2d_backward_weight.
// dWt accumulated — caller zeros.
inline void conv1d_backward_weight(const Tensor& X, const Tensor& dY,
                                   int N, int C_in, int L, int C_out, int kL,
                                   int stride, int padding, int dilation,
                                   int groups, Tensor& dWt) {
    conv2d_backward_weight(X, dY, N, C_in, /*H=*/1, /*W=*/L, C_out,
                           /*kH=*/1, /*kW=*/kL, 1, stride, 0, padding,
                           1, dilation, groups, dWt);
}
inline void conv1d_backward_weight(const Tensor& X, const Tensor& dY,
                                   int N, int C_in, int L, int C_out, int kL,
                                   int stride, int padding, int dilation,
                                   Tensor& dWt) {
    conv1d_backward_weight(X, dY, N, C_in, L, C_out, kL, stride, padding,
                           dilation, /*groups=*/1, dWt);
}

// conv1d backward w.r.t. bias — wrapper over conv2d_backward_bias.
// dB accumulated — caller zeros.
inline void conv1d_backward_bias(const Tensor& dY, int N, int C_out, int L_out,
                                 Tensor& dB) {
    conv2d_backward_bias(dY, N, C_out, /*H_out=*/1, /*W_out=*/L_out, dB);
}

// W8A16 1D convolution. Header-only wrapper over conv2d_int8w_fp16_forward
// (H=kH=1). FP16 activations, INT8 per-output-row weights. Throws on the CPU
// backend (no CPU W8A16 conv slot).
//   X: (N,C_in*L) FP16.  W_int8: (C_out, (C_in/groups)*kL) INT8 OIL.
//   scales: (C_out,1) FP32.  bias: (C_out,1) FP16 or null.  Y: (N,C_out*L_out) FP16.
inline void conv1d_int8w_fp16(const Tensor& X, const Tensor& W_int8,
                              const Tensor& scales, const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int dilation, int groups,
                              Tensor& Y) {
    conv2d_int8w_fp16_forward(X, W_int8, scales, bias, N, C_in, /*H=*/1,
                              /*W=*/L, C_out, /*kH=*/1, /*kW=*/kL,
                              /*stride_h=*/1, /*stride_w=*/stride,
                              /*pad_h=*/0, /*pad_w=*/padding,
                              /*dil_h=*/1, /*dil_w=*/dilation, groups, Y);
}
inline void conv1d_int8w_fp16(const Tensor& X, const Tensor& W_int8,
                              const Tensor& scales, const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int dilation, Tensor& Y) {
    conv1d_int8w_fp16(X, W_int8, scales, bias, N, C_in, L, C_out, kL, stride,
                      padding, dilation, /*groups=*/1, Y);
}

// Causal 1D convolution. Header-only wrapper: left-pad the length axis by
// dilation*(kL-1) (zero), then run a valid (padding=0) conv1d. Output length
// equals L when stride==1; every output sample depends only on inputs at or
// before its position.
//   X: (N,C_in*L).  Wt: (C_out, (C_in/groups)*kL) OIL.  bias: (C_out,1) or null.
//   scratch: caller-owned, resized to (N, C_in*(L+dilation*(kL-1))) and
//            overwritten — keeps the wrapper allocation-free across calls.
//   Y: (N,C_out*L_out), resized by conv1d.
inline void causal_conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C_in, int L, int C_out, int kL,
                          int stride, int dilation, int groups,
                          Tensor& scratch, Tensor& Y) {
    const int pad_left = dilation * (kL - 1);
    pad1d_forward(X, N, C_in, L, pad_left, /*pad_right=*/0, /*mode=*/0,
                  scratch);
    conv1d(scratch, Wt, bias, N, C_in, L + pad_left, C_out, kL, stride,
           /*padding=*/0, dilation, groups, Y);
}
inline void causal_conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C_in, int L, int C_out, int kL,
                          int stride, int dilation, Tensor& scratch, Tensor& Y) {
    causal_conv1d(X, Wt, bias, N, C_in, L, C_out, kL, stride, dilation,
                  /*groups=*/1, scratch, Y);
}

// 1D transposed convolution, NCL — the upsampling primitive of neural vocoders
// (HiFi-GAN, EnCodec/DAC decoders). A genuinely new kernel, CPU FP32-only.
//   L_out = (L-1)*stride - 2*padding + dilation*(kL-1) + output_padding + 1.
// output_padding (< stride) disambiguates the L_out values that map to one L
// under a strided forward conv (torch's ConvTranspose1d arg).
// Weight layout is input-channel-major: Wt (C_in, (C_out/groups)*kL),
// Wt[(c_in*(C_out/groups)+c_out_local)*kL + kl]. groups divides C_in and C_out;
// groups==C_in==C_out is depthwise transposed conv.
// Forward (scatter): each X[n,c_in,l] is scattered, per kernel tap kl, into
// output position l_out = l*stride - padding + kl*dilation.
//   X: (N,C_in*L).  Wt: (C_in,(C_out/groups)*kL).  bias: (C_out,1) or null.
//   Y: (N,C_out*L_out), resized + dtype-set to match X.
void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, Tensor& Y);
// Convenience overload: groups defaults to 1.
inline void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt,
                                     const Tensor* bias,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     Tensor& Y) {
    conv_transpose1d_forward(X, Wt, bias, N, C_in, L, C_out, kL, stride,
                             padding, output_padding, dilation, /*groups=*/1, Y);
}

// conv_transpose1d backward w.r.t. input — the adjoint is a plain gather conv:
// dX[n,c_in,l] gathers dY over every tap and the group's output channels at
// l_out = l*stride - padding + kl*dilation. dX overwritten (resized +
// dtype-set to dY). All hyperparams match the forward call.
void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, Tensor& dX);
inline void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                            int N, int C_in, int L, int C_out,
                                            int kL, int stride, int padding,
                                            int output_padding, int dilation,
                                            Tensor& dX) {
    conv_transpose1d_backward_input(Wt, dY, N, C_in, L, C_out, kL, stride,
                                    padding, output_padding, dilation,
                                    /*groups=*/1, dX);
}

// conv_transpose1d backward w.r.t. weight:
//   dWt[c_in,c_out_local,kl] += sum_{n,l} X[n,c_in,l]*dY[n,c_out,l_out],
//   l_out = l*stride - padding + kl*dilation (skipped when OOB).
// dWt accumulated — caller zeros. All hyperparams match the forward call.
void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, Tensor& dWt);
inline void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY,
                                             int N, int C_in, int L, int C_out,
                                             int kL, int stride, int padding,
                                             int output_padding, int dilation,
                                             Tensor& dWt) {
    conv_transpose1d_backward_weight(X, dY, N, C_in, L, C_out, kL, stride,
                                     padding, output_padding, dilation,
                                     /*groups=*/1, dWt);
}

// conv_transpose1d backward w.r.t. bias:
//   dB[c_out] += sum_{n,l_out} dY[n,c_out,l_out].
// dB accumulated — caller zeros.
void conv_transpose1d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int L_out, Tensor& dB);

// One streaming step of a causal depthwise 1D conv against a rolling state
// cache (in the spirit of kv_cache_append) — for autoregressive / streaming
// decoders. Forward-only, new vtable row, CPU FP32-only.
// Depthwise: C channels in and out, one length-kL filter per channel. With
// L_step new samples it produces L_step outputs:
//   Y[n,c,t] = bias[c] + sum_{kl} W[c,kl] * buf[n,c,t+kl*dilation],
//   buf = state[n,c,:] ++ X[n,c,:].
// `state` is updated in place to the last (kL-1)*dilation samples of buf, so a
// sequence of calls reproduces one full causal_conv1d over the concatenated
// input (caller zero-initialises state before the first step).
//   X: (N,C*L_step) new samples.  Wt: (C,kL) depthwise filter.
//   bias: (C,1) or null.  state: (N,C*(kL-1)*dilation) — read AND overwritten.
//   Y: (N,C*L_step), resized + dtype-set to match X.
void causal_conv1d_update(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          Tensor& state, Tensor& Y);

// ─── Vocoder / codec activations (audio) ───────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal). NCL
// layout — the (N,C,L) dims are passed as int args; element (n,c,l) is at flat
// index (n*C+c)*L + l.

// Snake activation (BigVGAN / DAC vocoder), per-channel learnable alpha (and
// optional beta):
//   plain snake (beta == null): y = x + (1/alpha_c)*sin^2(alpha_c*x)
//   snakebeta   (beta != null): y = x + (1/beta_c) *sin^2(alpha_c*x)
// alpha/beta are per-channel (broadcast across the (n,l) plane). The reciprocal
// denominator is sign-preserved-floored at magnitude 1e-9 to avoid NaN/Inf.
//   X, Y: (N,C*L).  alpha: (C,1) or (1,C).  beta: (C,1)/(1,C) or null.
//   Y resized + dtype-set to match X; X and Y may alias. FP32, CPU-resident.
void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y);

// Snake backward, reads the raw forward input X. With s=sin(a*x), c=cos(a*x),
// a=alpha_c, denom=(beta?beta_c:a), r=1/denom (sign-guarded as in the forward):
//   dy/dx     = 1 + 2*a*r*s*c
//   dy/dalpha = 2*r*x*s*c          (plain snake also adds the -r^2*s^2 term,
//                                   since denom==alpha there)
//   dy/dbeta  = -r^2*s^2           (snakebeta only)
//   dX: (N,C*L) overwritten (resized + dtype-set to X).
//   dAlpha: (C,1) accumulated — caller zeros.
//   dBeta:  (C,1) accumulated — caller zeros; non-null exactly when beta is.
void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                    const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta);

// ELU (EnCodec activation), elementwise:
//   y = x                    if x > 0
//   y = alpha*(exp(x) - 1)   otherwise
// y resized to match x; x and y may alias. CPU FP32-only.
void elu_forward(const Tensor& x, float alpha, Tensor& y);
inline void elu_forward(const Tensor& x, Tensor& y) {
    elu_forward(x, /*alpha=*/1.0f, y);
}

// ELU backward, reads the raw forward input x:
//   dX = dY * (x > 0 ? 1 : alpha*exp(x)).
// dX overwritten (resized to match x); may alias dY. CPU FP32-only.
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX);
inline void elu_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    elu_backward(x, dY, /*alpha=*/1.0f, dX);
}

// Leaky ReLU (HiFi-GAN activation), elementwise:
//   y = x > 0 ? x : negative_slope*x.
// y resized to match x; x and y may alias. CPU FP32-only.
void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y);

// Leaky ReLU backward, reads the raw forward input x:
//   dX = dY * (x > 0 ? 1 : negative_slope).
// dX overwritten (resized to match x); may alias dY. CPU FP32-only.
void leaky_relu_backward(const Tensor& x, const Tensor& dY,
                         float negative_slope, Tensor& dX);

// ─── Codec quantization (audio) ────────────────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal). The
// quantization bottlenecks of neural audio codecs (EnCodec/DAC residual-VQ,
// NanoCodec FSQ).

// Vector-quantization encode. For each row x[n], picks the codeword k
// minimising ||x[n] - codebook[k]||^2, emits the index, and copies that
// codeword out as the quantized vector. Ties keep the lowest index.
//   x: (N,D) FP32.  codebook: (K,D) FP32.
//   indices: (N,1) INT32 output — resized + dtype-set to INT32.
//   quantized: (N,D) FP32 output — codebook[indices[n],:], resized + dtype-set.
// Decode indices with embedding_lookup_forward (indices.data is the d_idx
// buffer). RVQ is composed caller-side; there is no rvq op.
void vq_encode_forward(const Tensor& x, const Tensor& codebook,
                       Tensor& indices, Tensor& quantized);

// Vector-quantization encode backward — straight-through estimator: the argmin
// is non-differentiable, so the upstream gradient is copied through.
//   dX = dQuantized      (overwritten — NOT accumulated)
// Encoder STE path only; the codebook/commitment losses are separate
// caller-side MSE terms. dX, dQuantized: (N,D) FP32; dX resized + dtype-set to
// match dQuantized; may alias it.
void vq_encode_backward(const Tensor& dQuantized, Tensor& dX);

// Finite Scalar Quantization (NanoCodec quantizer). Each coordinate is snapped
// independently to one of L_d evenly spaced levels. Input x is assumed already
// bounded into [-1,1] by a caller-side tanh. For dimension d with L_d levels
// and half-width h = (L_d-1)/2:
//   v = clamp(x, -1, 1)
//   i = round((v+1)/2 * (L_d-1))            in [0, L_d-1]
//   quantized = i/h - 1                      back into [-1,1]
// The per-dim indices are packed mixed-radix (dimension 0 least-significant):
//   packed = i_0 + L_0*(i_1 + L_1*(i_2 + ...)).
//   x: (N,D) FP32, pre-bounded.  levels: (D,1) INT32 per-dim level count (>=2).
//   quantized: (N,D) FP32 output — resized + dtype-set to FP32.
//   packed_indices: (N,1) INT32 output — resized + dtype-set to INT32.
void fsq_quantize_forward(const Tensor& x, const Tensor& levels,
                          Tensor& quantized, Tensor& packed_indices);

// FSQ backward — straight-through estimator (the round is non-differentiable):
//   dX = dQuantized      (overwritten — NOT accumulated)
// dX, dQuantized: (N,D) FP32; dX resized + dtype-set to match dQuantized;
// may alias it.
void fsq_quantize_backward(const Tensor& dQuantized, Tensor& dX);

// ─── 1D resampling (audio) ─────────────────────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal).
// Arbitrary-scale resampling along the length axis of an NCL tensor — for
// sample-rate conversion in STT / TTS / codec front-ends.

// 1D resample along the length axis: N and C pass through, the length axis is
// rescaled from L_in to any positive L_out. PyTorch align_corners=False
// convention — for output position dst the source coordinate is
//   src = (dst + 0.5)*(L_in/L_out) - 0.5
//   mode == 0  nearest — Y[dst] = X[clamp(round_half_to_even(src), 0, L_in-1)].
//   mode == 1  linear  — s=clamp(src,0,L_in-1), x0=floor(s), x1=min(x0+1,L_in-1),
//              f=s-x0;  Y[dst] = (1-f)*X[x0] + f*X[x1].
//   X: (N,C*L_in).  Y: (N,C*L_out), resized + dtype-set to FP32.
// L_out == L_in is the identity. mode other than 0/1 throws.
void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out,
                        int mode, Tensor& Y);

// Backward (adjoint) of resample1d_forward: each output gradient is scattered
// back onto the input position(s) it sampled, with the forward's weights:
//   nearest: dX[round(src)] += dY[dst]
//   linear:  dX[x0] += (1-f)*dY[dst];  dX[x1] += f*dY[dst]
//   dY: (N,C*L_out).  dX: (N,C*L_in), overwritten (resized + dtype-set to FP32).
// N, C, L_in, L_out, mode match the forward call.
void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out,
                         int mode, Tensor& dX);

// ─── log / exp / round elementwise (audio) ─────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal).
// Elementwise scalar maps; outputs resized + dtype-set to match the input; x/y
// and dX/dY may alias. None has learnable parameters, so every backward
// overwrites dX.

// Natural logarithm: y = log(x), elementwise (log-mel spectrograms, log-domain
// losses). The caller owns the x > 0 precondition — this op does NOT guard the
// input, so a mis-clamped pipeline fails loudly (log(0) = -inf, log(neg) = NaN).
//   x, y: (R,C) FP32.
void log_forward(const Tensor& x, Tensor& y);

// Natural-log backward, reads the raw forward input x: dX = dY / x. The caller
// owns the x > 0 precondition (no guard). dX overwritten; may alias dY.
//   x, dY, dX: (R,C) FP32.
void log_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// Natural exponential: y = exp(x), elementwise — the inverse of log_forward.
//   x, y: (R,C) FP32.
void exp_forward(const Tensor& x, Tensor& y);

// Exponential backward, reads the raw forward input x: dX = dY * exp(x).
// dX overwritten; may alias dY.  x, dY, dX: (R,C) FP32.
void exp_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// Round-half-to-even: y = nearest integer to x, ties to the even integer
// (banker's rounding — matches torch.round / numpy.round / std::nearbyint
// under FE_TONEAREST: 0.5->0, 1.5->2, 2.5->2, -2.5->-2).
//   x, y: (R,C) FP32.
void round_forward(const Tensor& x, Tensor& y);

// Round backward — straight-through estimator (round has zero derivative
// almost everywhere): dX = dY. Needs only dY. dX overwritten; may alias dY.
//   dY, dX: (R,C) FP32.
void round_backward(const Tensor& dY, Tensor& dX);

// ─── Autoregressive logit sampling ─────────────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal). The
// next-token sampler for autoregressive generation loops — a general LLM /
// codec-LM sampler, not audio-specific.

// Draw one token id per row of an (N, V) logit matrix (N independent streams,
// V = vocabulary size), applying, in order:
//   1. temperature   logit /= temperature
//   2. softmax       p = softmax(logit)
//   3. top-k         (top_k > 0) keep the top_k highest-p tokens, rest -> 0
//   4. top-p         (top_p < 1) keep the smallest highest-p set with cumulative
//                    probability >= top_p, rest -> 0 (applied after top-k)
//   5. renormalize   kept probabilities rescaled to sum to 1
//   6. draw          inverse-CDF lookup of a uniform u in [0,1)
// Greedy: temperature == 0 is deterministic argmax — steps 2-6 skipped, no RNG
// consumed, ties keep the lowest index. top_k == 1 is likewise deterministic.
//
// RNG: counter-based Philox 4x32-10, seeded by two scalar args (so dispatch
// resolves on `logits`):
//   key     — 64-bit run seed.
//   counter — 64-bit base counter offset.
// Row n draws from the Philox counter block for the 64-bit value (counter + n),
// converting the first output word to a uniform via its top 24 bits / 2^24, so
// row n's draw depends only on (key, counter + n) — reproducible, independent
// of N and row order. To get fresh draws across decode steps, advance `counter`
// by the rows sampled so far and keep `key` fixed.
//
// Metal's FP32 reductions are not bit-identical to the CPU op's FP64
// accumulators, so a draw landing within a few ulp of a CDF-bucket boundary may
// pick a different token; with well-separated logits the backends agree.
//
//   logits:  (N,V) FP32 input.
//   indices: (N,1) INT32 output — resized + dtype-set to INT32.
// Throws ("brotensor: sample_logits: <reason>") for temperature < 0, top_k < 0,
// top_p < 0, or V == 0 while N > 0. No backward.
void sample_logits(const Tensor& logits, float temperature, int top_k,
                   float top_p, uint64_t key, uint64_t counter,
                   Tensor& indices);

// ─── L2 norm + Gated Delta Rule (brolm Qwen3-Next text path) ───────────────
//
// Building blocks for the Gated DeltaNet text layers of Qwen3-Next / Qwen3.5.
// The recurrence acts on (L, num_heads*d) sequence-major tensors using the
// same head-contiguous layout as rope_forward / rms_norm.

// L2-normalize each head row over its head_dim slice, with epsilon:
//   y[r, h*head_dim + d] = x[r, h*head_dim + d] /
//                          sqrt(sum_d x[r, h*head_dim + d]^2 + eps)
// Distinct from rms_norm — sum (not mean) of squares, no learnable gamma, no
// sqrt(d) factor; used to normalise q/k per head in Gated DeltaNet.
//   X, Y: (L, num_heads*head_dim).  head_dim and num_heads partition the cols
//         exactly as rope_forward; head_dim must be positive.
//   Y resized + dtype-set to match X.  X and Y may alias.
// Dispatched on X.dtype (FP32/FP16); FP32 accumulation. CPU is FP32-only.
void l2_norm_forward(const Tensor& X, int head_dim, int num_heads,
                     float eps, Tensor& Y);

// L2-norm backward. With s_r = sum_d x_d^2 + eps, n_r = 1/sqrt(s_r):
//   dX_d = n_r * (dY_d - x_d * n_r^2 * sum_{d'} (x_{d'} * dY_{d'}))
//   X, dY, dX: (L, num_heads*head_dim), same dtype.
//   dX overwritten (resized + dtype-set to match X). FP32 accumulation.
void l2_norm_backward(const Tensor& X, int head_dim, int num_heads,
                      float eps, const Tensor& dY, Tensor& dX);

// Gated Delta Rule — chunked prefill. Runs the matrix-valued recurrence
//   alpha_t = exp(-softplus(a_raw_t) * exp(log_A))      (per token, per head)
//   beta_t  = sigmoid(beta_raw_t)
//   S_t     = alpha_t * S_{t-1}
//           + beta_t * (v_t - S_{t-1} k_t) k_t^T
//   o_t     = S_t q_t                                  (per head)
// over L tokens, sequentially within each head. The chunked WY/UT-transform
// is an internal optimisation — the contract is exactly the per-token rule.
//   Q, K: (L, num_heads*d_k).      V: (L, num_heads*d_v).
//         Heads contiguous within each row, exactly as rope_forward / rms_norm.
//   a_raw, beta: (L, num_heads) FP32 — per-token gate / write inputs (raw).
//                softplus / sigmoid are applied inside the op.
//   log_A: (num_heads, 1) FP32 — per-head learnable decay scale.
//   state: (num_heads, d_v*d_k) FP32 — initial S per head (caller zero-fills
//          for a fresh sequence; row h is S_h[v, k] = state[h, v*d_k + k]).
//          Read AND updated in place; on return holds S after token L-1.
//   O: (L, num_heads*d_v) — output, resized + dtype-set to match Q.
// Q/K/V/O are dispatched on Q.dtype (FP32 on CPU; FP16 or FP32 on GPU). state,
// a_raw, beta, log_A are FP32 on every backend (accumulator + gate precision).
// num_heads*d_k must equal Q.cols and K.cols; num_heads*d_v must equal V.cols
// and O.cols. d_k and d_v may differ. FP32 accumulation. Forward-only.
void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& a_raw, const Tensor& beta,
                              const Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              Tensor& state, Tensor& O);

// Gated Delta Rule — streaming step. Same math as gated_delta_rule_chunked
// but for L_step new tokens against an existing state. With L_step == 1 this
// is the per-step recurrence; with L_step > 1 it's a plain non-chunked scan
// (correct, just without the WY/UT speedup).
//   Q, K: (L_step, num_heads*d_k).    V: (L_step, num_heads*d_v).
//   a_raw, beta: (L_step, num_heads) FP32.   log_A: (num_heads, 1) FP32.
//   state: (num_heads, d_v*d_k) FP32 — read AND overwritten with S after the
//          last new token, ready for the next call.
//   O: (L_step, num_heads*d_v), resized + dtype-set to match Q.
// Dtype rules identical to gated_delta_rule_chunked.
void gated_delta_rule_step(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& a_raw, const Tensor& beta,
                           const Tensor& log_A,
                           int num_heads, int d_k, int d_v,
                           Tensor& state, Tensor& O);

} // namespace brotensor
