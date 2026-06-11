#pragma once

// brotensor ops/elementwise.h — Elementwise tensor ops: add/scale/clamp/mul/cast + log/exp/round.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// y[i] += x[i]. Identical shape required.
void add_inplace(Tensor& y, const Tensor& x);


// y[i] = a*y[i] + b*x[i], elementwise, with the arithmetic carried out in FP32
// regardless of the storage dtype. Identical shape and dtype required;
// dispatched FP32/FP16/BF16 on y.dtype. Motivation: classifier-free guidance
// blends v = s*v_cond - (s-1)*v_uncond where both terms are large and nearly
// cancel — FP16 storage is fine but the combine must accumulate in FP32 (a
// half-precision combine catastrophically cancels); only the final store
// rounds back to the storage dtype.
void axpby_inplace(Tensor& y, const Tensor& x, float a, float b);


// y[i] += s. Dispatched FP32/FP16 on y.dtype.
void add_scalar_inplace(Tensor& y, float s);


// y[i] *= s. Dispatched FP32/FP16 on y.dtype.
void scale_inplace(Tensor& y, float s);


// y[i] = min(max(y[i], lo), hi), in place. Dispatched FP32/FP16 on y.dtype.
void clamp(Tensor& y, float lo, float hi);


// Dtype cast: dst = src converted to out_dtype. dst resized to (src.rows,
// src.cols, out_dtype) on src's device. Supports FP32<->FP16, FP32<->BF16,
// FP16<->BF16 (via FP32 — exact for FP16→BF16's overlapping mantissa range),
// and a same-dtype passthrough copy; other pairs throw. The standard
// mixed-precision primitive (low-precision weight <-> FP32 master copy).
void cast(const Tensor& src, Tensor& dst, Dtype out_dtype);


// Y[i] += X[i] over (B,D). Identical shape required.
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD);


// y[i] *= x[i]. Identical shape and dtype; dispatched FP32/FP16 on y.dtype.
void mul_inplace(Tensor& y, const Tensor& x);


// Binary threshold to a byte mask: Y[i] = X[i] > t ? 1 : 0 (strict > —
// elements exactly at t map to 0). Backs SAM AMG's device-side mask
// binarization (logits -> 0/1 mask without a host round-trip).
//   X: (R, C) FP32 or FP16.
//   Y: (R, C) INT8, resized + dtype-set. Not differentiable.
void threshold_u8(const Tensor& X, float t, Tensor& Y);


// ─── log / exp / round elementwise ─────────────────────────────────────────
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


// ─── sin / cos / rsqrt elementwise (StyleGAN3 Fourier features + demod) ─────
//
// FP32-only elementwise scalar maps. Outputs resized + dtype-set to match the
// input; x/y and dX/dY may alias. None has learnable parameters, so every
// backward overwrites dX. sin/cos back the SynthesisInput Fourier features
// (sin(2π·…)); rsqrt backs the modulation-demod / pixel-norm reciprocal-sqrt.

// y = sin(x), elementwise.  x, y: (R,C) FP32.
void sin_forward(const Tensor& x, Tensor& y);

// Sine backward, reads the raw forward input x: dX = dY * cos(x).
// dX overwritten; may alias dY.  x, dY, dX: (R,C) FP32.
void sin_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// y = cos(x), elementwise.  x, y: (R,C) FP32.
void cos_forward(const Tensor& x, Tensor& y);

// Cosine backward, reads the raw forward input x: dX = -dY * sin(x).
// dX overwritten; may alias dY.  x, dY, dX: (R,C) FP32.
void cos_backward(const Tensor& x, const Tensor& dY, Tensor& dX);

// y = 1/sqrt(x), elementwise reciprocal square root. The caller owns the
// x > 0 precondition — no guard (rsqrt(0) = +inf, rsqrt(<0) = NaN).
//   x, y: (R,C) FP32.
void rsqrt_forward(const Tensor& x, Tensor& y);

// Rsqrt backward, reads the forward OUTPUT y (= 1/sqrt(x)): dX = -0.5*dY*y^3.
// dX overwritten; may alias dY.  y, dY, dX: (R,C) FP32.
void rsqrt_backward(const Tensor& y, const Tensor& dY, Tensor& dX);

}  // namespace brotensor
