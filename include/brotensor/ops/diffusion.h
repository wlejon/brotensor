#pragma once

// brotensor ops/diffusion.h — Diffusion: ResBlock, AdaLN modulate, sampler steps (DDIM/Euler/DPM++), timestep embed.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// ─── AdaLN modulation (DiT / SD3 / Flux) ───────────────────────────────────

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

}  // namespace brotensor
