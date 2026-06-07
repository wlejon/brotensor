#pragma once

// brotensor ops/norm.h — Normalizations: layernorm, rms_norm, group_norm, batch_norm, l2_norm.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


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


// Inference-only batched LayerNorm: R independent rows of length D, no caches,
// no host syncs. Use layernorm_forward when backward is needed.
//   X_RD: (R,D).  gamma, beta: (D,).  Y_RD: (R,D), resized if mis-shaped.
// Training-mode batched LayerNorm: R rows of length D, with caches needed by
// the backward.
//   X_RD: (R,D).  gamma, beta: (D,).
//   Y_RD, Xhat_RD: (R,D), resized/dtype-set to match X_RD.
//   Mean_R, Rstd_R: (R,1) FP32 regardless of X_RD.dtype — resized/dtyped here.
//   eps: variance epsilon.
// GPU backends accept FP32 / FP16 / BF16 for X (gamma/beta/Y/Xhat share dtype).
// CPU is FP32-only. Caches are always FP32 so backward can dispatch identically.
void layernorm_forward_batched_with_caches(const Tensor& X_RD,
                                           const Tensor& gamma,
                                           const Tensor& beta,
                                           Tensor& Y_RD, Tensor& Xhat_RD,
                                           Tensor& Mean_R, Tensor& Rstd_R,
                                           float eps);


// Training-mode batched LayerNorm backward, consuming the forward caches.
//   dY_RD, Xhat_RD: (R,D).  gamma: (D,).  Rstd_R: (R,1) FP32 from forward.
//   dX_RD: (R,D) overwritten (resized + dtype-set to match dY_RD).
//   dGamma, dBeta: (D,) accumulated — caller zeros (project convention).
// Dtype rules mirror the forward: GPU FP32/FP16/BF16, CPU FP32 only.
void layernorm_backward_batched_with_caches(const Tensor& dY_RD,
                                            const Tensor& Xhat_RD,
                                            const Tensor& gamma,
                                            const Tensor& Rstd_R,
                                            Tensor& dX_RD,
                                            Tensor& dGamma, Tensor& dBeta);


void layernorm_forward_inference_batched(const Tensor& X_RD,
                                         const Tensor& gamma,
                                         const Tensor& beta,
                                         Tensor& Y_RD,
                                         float eps);


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


// L2 normalization over the channel axis, NCHW. For every spatial position
// (n, h, w), rescales the length-C channel vector to unit L2 norm with an
// epsilon floor on the divisor:
//   Y[n,c,h,w] = X[n,c,h,w] / max(sqrt(sum_c X[n,c,h,w]^2), eps)
// The per-pixel direction normalize used by surface-normal / direction-field
// models (DSINE normal normalize), feature-map L2 norm, and cosine-sim prep.
// Distinct from l2_norm_forward (gated-deltanet per-head, last dim of an
// (L, H*D) layout) — here the unit axis is channels in an NCHW grid.
//   X, Y: (N, C*H*W). Y resized + dtype-set to X; X and Y may alias.
// Reduction in double. Dispatched FP32/FP16/BF16 on X.dtype (CPU is FP32-only).
// Inference-only: there is no backward.
void l2_normalize_nchw_forward(const Tensor& X,
                               int N, int C, int H, int W,
                               float eps,
                               Tensor& Y);


// FP16 inference-only batched LayerNorm: R rows of length D, FP32
// accumulation, no caches.
//   X_RD, Y_RD: (R,D) FP16.  gamma, beta: (D,) FP16.  Y_RD resized as needed.
void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              const Tensor& beta,
                                              Tensor& Y_RD,
                                              float eps);


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


// ─── L2 norm (per-head; Gated DeltaNet q/k) ────────────────────────────────
//
// Building blocks for Gated DeltaNet linear-attention text layers (the
// recurrence-based half of hybrid linear/standard attention decoders). Acts on
// (L, num_heads*d) sequence-major tensors using the same head-contiguous
// layout as rope_forward / rms_norm.

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


// ─── BatchNorm (NCHW) ──────────────────────────────────────────────────────
//
// Standard BatchNorm: statistics are reduced over (N, H, W) per channel —
// the variant pretrained ResNet / DETR-ResNet50 / classic Mask2Former
// backbones use, and distinct from GroupNorm (which reduces within a single
// sample). Three slots: training forward, inference forward, and backward.
// CPU is FP32-only; GPU may add FP16 paths later (slots dispatched on X.dtype).

// BatchNorm training forward.
//   X, Y:                (N, C*H*W).
//   gamma, beta:         (C,1) per-channel scale / shift.
//   running_mean,
//   running_var:         (C,1) — read AND updated in place via
//                          running = (1 - momentum) * running + momentum * batch
//                        (PyTorch nn.BatchNorm2d convention). running_var is
//                        updated using the *unbiased* batch variance estimator.
//   saved_mean,
//   saved_rstd:          (C,1) — written, fed to batch_norm_backward.
//                        rstd = 1 / sqrt(biased_var + eps).
//   eps:                 typically 1e-5f.
//   momentum:            PyTorch convention, typically 0.1f.
void batch_norm_forward(const Tensor& X,
                        const Tensor& gamma, const Tensor& beta,
                        Tensor& running_mean, Tensor& running_var,
                        int N, int C, int H, int W,
                        float eps, float momentum,
                        Tensor& Y,
                        Tensor& saved_mean, Tensor& saved_rstd);


// BatchNorm inference forward. Uses running_mean / running_var; no state
// mutation, no saved tensors. This is the path loaded pretrained
// checkpoints want at inference time.
//   X, Y:               (N, C*H*W).
//   gamma, beta, running_mean, running_var: (C,1).
void batch_norm_inference(const Tensor& X,
                          const Tensor& gamma, const Tensor& beta,
                          const Tensor& running_mean,
                          const Tensor& running_var,
                          int N, int C, int H, int W,
                          float eps,
                          Tensor& Y);


// BatchNorm backward, given saved batch mean/rstd from the training forward.
//   X, dY, dX: (N, C*H*W). dX overwritten (resized + dtype-set to X).
//   gamma:               (C,1) forward scale.
//   saved_mean,
//   saved_rstd:          (C,1) from forward.
//   dGamma, dBeta:       (C,1) accumulated — caller zeros.
// Math per channel (M = N*H*W):
//   xhat = (x - mean) * rstd
//   dxhat = dY * gamma
//   dX = rstd * (dxhat - (sum dxhat + xhat * sum(dxhat*xhat)) / M)
//   dGamma_c += sum dY*xhat ; dBeta_c += sum dY.
void batch_norm_backward(const Tensor& X,
                         const Tensor& gamma,
                         const Tensor& saved_mean,
                         const Tensor& saved_rstd,
                         const Tensor& dY,
                         int N, int C, int H, int W,
                         Tensor& dX,
                         Tensor& dGamma, Tensor& dBeta);


// ─── Pixel norm (StyleGAN mapping network) ──────────────────────────────────
//
// RMS-style normalisation over the feature/channel axis (the trailing `cols`
// dim), per row: Y = X * rsqrt(mean_c(X^2) + eps). This is NOT l2_normalize —
// it divides by the root-MEAN-square (includes the 1/C factor), matching
// StyleGAN's `normalize_2nd_moment`. No learnable parameters.
//   X, Y: (N, C) FP32. Y resized + dtype-set to match X.
//   eps:  added to the mean-square before rsqrt (StyleGAN uses 1e-8f).
void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y);

// Pixel-norm backward. With r = rsqrt(mean(x^2)+eps) and s = Σ_c dY_c·X_c
// (per row), dX_c = r·dY_c − (r^3·X_c / C)·s. Reads the raw forward input X.
//   X, dY, dX: (N, C) FP32. dX overwritten (resized + dtype-set to X).
void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps,
                         Tensor& dX);

}  // namespace brotensor
