#pragma once

// brotensor ops/stylegan.h — StyleGAN3-R generator primitives:
//   modulated_conv2d, upfirdn2d, bias_act (backend ops, fwd+bwd) and
//   filtered_lrelu (a composite over bias_act + upfirdn2d, no backend kernel).
//
// CPU FP32 reference path; mirrors the pure-PyTorch `_ref` implementations in
// NVlabs/stylegan3 (modulated_conv2d / SynthesisLayer, _upfirdn2d_ref,
// _bias_act_ref, _filtered_lrelu_ref) so numerics match bit-for-the-math.
// CUDA / Metal slots are left null until the CPU path is validated.

#include "../tensor.h"

namespace brotensor {


// ─── modulated_conv2d ───────────────────────────────────────────────────────
//
// The StyleGAN synthesis-layer core: a per-sample style modulation of the conv
// weights, optional demodulation, then a standard conv2d (stride 1, dilation 1)
// per sample. Config-R uses kH=kW=1 but the op is general.
//
//   X:     (N, C_in*H*W)   NCHW input.
//   W:     (C_out, C_in*kH*kW)   OIHW shared weights (un-modulated).
//   s:     (N, C_in)        per-sample, per-in-channel style (output of the
//                           layer's affine FC — a separate `linear`).
//   dcoef: (N, C_out)       OUT — saved demod coefficients (1.0 if !demodulate),
//                           needed by the backward.
//   Y:     (N, C_out*H_out*W_out)  resized + dtype-set to match X.
//
// Forward (per sample n):
//   w'[o,i,kh,kw]  = W[o,i,kh,kw] * s[n,i]
//   dcoef[n,o]     = demodulate ? rsqrt(Σ_{i,kh,kw} w'^2 + eps) : 1
//   w''            = w' * dcoef[n,o]
//   Y[n]           = conv2d(X[n], w'', pad=(pad_h,pad_w), stride=1)
// H_out = H + 2*pad_h - (kH-1);  W_out = Wd + 2*pad_w - (kW-1).
void modulated_conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW,
                              int pad_h, int pad_w,
                              bool demodulate, float eps,
                              Tensor& dcoef, Tensor& Y);

// modulated_conv2d backward. Reconstructs w' from (W,s) and uses the saved
// `dcoef`. dX is overwritten; dW ACCUMULATES (caller zeros); ds is overwritten.
//   dX: (N, C_in*H*W).  dW: (C_out, C_in*kH*kW).  ds: (N, C_in).
// dW is OPTIONAL: pass an uncommitted (default-constructed / empty) Tensor to
// skip the weight gradient entirely — the op then drops the dW GEMM and its
// scratch. Inversion uses this (it freezes the weights and discards dW). A
// committed dW must match X's dtype and (C_out, C_in*kH*kW) shape.
// Math (per sample n; dw'' = conv2d_backward_weight(X[n],dY[n])):
//   g[n,o]   = Σ_{i,kh,kw} dw''[n,o,..] * w'[n,o,..]
//   dw'[n,o] = demodulate ? dw''*dcoef - g*dcoef^3*w' : dw''
//   dW[o,..] += Σ_n dw'[n,o,..] * s[n,i]
//   ds[n,i]   = Σ_{o,kh,kw} dw'[n,o,i,kh,kw] * W[o,i,kh,kw]
//   dX[n]     = conv2d_backward_input(w''[n], dY[n])
void modulated_conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& s,
                               const Tensor& dcoef, const Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW,
                               int pad_h, int pad_w, bool demodulate, float eps,
                               Tensor& dX, Tensor& dW, Tensor& ds);


// ─── upfirdn2d ──────────────────────────────────────────────────────────────
//
// Upsample (zero-insert) → pad/crop → 2D FIR correlation → downsample → gain.
// The general (non-separable) 2D path required by config-R's radial filters.
// The filter `f` is a CONSTANT shared across channels (depthwise) — no gradient
// to `f`. Mirrors `_upfirdn2d_ref`.
//
//   X: (N, C*H*W).  f: (fH, fW).  Y: (N, C*H_out*W_out) resized to match X.
//   up_*/down_*: per-axis up/down factors.
//   pad_{x,y}{0,1}: pre-FIR padding; NEGATIVE values crop.
//   flip_filter: false ⇒ true convolution (filter flipped before correlate);
//                true  ⇒ plain correlation.
//   gain: output scale (for a 2D filter this equals scaling f by `gain`).
// H_out = (H*up_y + pad_y0 + pad_y1 - fH)/down_y + 1  (W_out analogous).
void upfirdn2d_forward(const Tensor& X, const Tensor& f,
                       int N, int C, int H, int Wd, int fH, int fW,
                       int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, Tensor& Y);

// upfirdn2d backward. upfirdn2d is linear in X, so dX is itself an upfirdn2d of
// dY with up/down swapped, the filter flip inverted, and padding recomputed
// (mirrors `_upfirdn2d_cuda`'s backward). Pass the SAME forward params (H,Wd
// are the forward INPUT spatial dims); dX is overwritten, sized to (N,C*H*Wd).
//   p = [ fW - pad_x0 - 1,
//         Wd*up_x - W_out*down_x + pad_x0 - up_x + 1,
//         fH - pad_y0 - 1,
//         H *up_y - H_out*down_y + pad_y0 - up_y + 1 ]
//   dX = upfirdn2d(dY, f, up=(down), down=(up), pad=p, flip=!flip, gain=gain)
void upfirdn2d_backward(const Tensor& dY, const Tensor& f,
                        int N, int C, int H, int Wd, int fH, int fW,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, Tensor& dX);


// ─── bias_act ───────────────────────────────────────────────────────────────
//
// Fused per-channel bias + activation + gain + clamp. Mirrors `_bias_act_ref`.
//   X: (N, C*HW)  with HW the flattened spatial size.  b: (C,1) or null.
//   act: 0 = linear, 1 = lrelu.  StyleGAN uses alpha=0.2 and, for lrelu,
//        a default gain=sqrt(2).  clamp < 0 ⇒ no clamp.
//   Y: (N, C*HW) resized + dtype-set to match X.
// Forward:  t = X + b[c];  y = act(t);  y *= gain;  if clamp>=0: clip(y,±clamp).
void bias_act_forward(const Tensor& X, const Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, Tensor& Y);

// bias_act backward. Recomputes t = X + b[c] for the activation / clamp masks.
//   dt = dY * gain * act'(t) * (clamp active ? 0 : 1)
//   dX = dt (overwritten);  dB[c] += Σ_{n,hw} dt  (accumulated; caller zeros).
// dB may be null (skip the bias gradient). The clamp mask uses the pre-clamp
// value y_preclamp = gain*act(t): the gradient is killed where |y_preclamp|>clamp.
void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp, Tensor& dX, Tensor* dB);


// ─── filtered_lrelu (composite — no backend kernel) ─────────────────────────
//
// The alias-free nonlinearity: bias → upsample → lrelu (+bias-free) → downsample.
// Implemented over bias_act + upfirdn2d so correctness is inherited and the
// backward chains the sub-backwards. Mirrors `_filtered_lrelu_ref` EXACTLY,
// including that the channel bias `b` is applied BEFORE the upsample (a linear
// bias_act) and the post-upsample bias_act applies only the lrelu.
//
//   X:  (N, C*H*W).  fu, fd: (fuH,fuW)/(fdH,fdW) constant filters.  b: (C,1)|null.
//   up/down: integer scale factors.  pad_*: padding fed to the UP-stage upfirdn2d.
//   gain: lrelu gain (default sqrt(2)).  slope: lrelu alpha (default 0.2).
//   clamp: post-activation clamp (<0 ⇒ none).
//   up_buf:  OUT cache — input to the lrelu bias_act (post-upsample tensor).
//   act_buf: OUT cache — output of the lrelu (input to the down-stage).
//   Y: (N, C*out_h*out_w) resized to match.
// out_w = (W*up + (px0+px1) - (fuW-1) - (fdW-1) + (down-1)) / down  (out_h analog).
void filtered_lrelu_forward(const Tensor& X, const Tensor& fu, const Tensor& fd,
                            const Tensor* b, int N, int C, int H, int W,
                            int up, int down, int pad_x0, int pad_x1,
                            int pad_y0, int pad_y1, float gain, float slope,
                            float clamp, Tensor& up_buf, Tensor& act_buf,
                            Tensor& Y);

// filtered_lrelu backward — reverses the forward chain:
//   upfirdn2d_backward(down) → bias_act_backward(lrelu) → upfirdn2d_backward(up)
//   → bias_act_backward(linear, accumulates dB).
// Needs the forward caches (up_buf) and the original X/b for the bias-grad pass.
// dX overwritten; dB (if non-null) ACCUMULATES — caller zeros.
void filtered_lrelu_backward(const Tensor& dY, const Tensor& X,
                             const Tensor& fu, const Tensor& fd,
                             const Tensor* b, int N, int C, int H, int W,
                             int up, int down, int pad_x0, int pad_x1,
                             int pad_y0, int pad_y1, float gain, float slope,
                             float clamp, const Tensor& up_buf,
                             Tensor& dX, Tensor* dB);

}  // namespace brotensor
