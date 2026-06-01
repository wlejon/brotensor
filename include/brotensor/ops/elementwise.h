#pragma once

// brotensor ops/elementwise.h — Elementwise tensor ops: add/scale/clamp/mul/cast + log/exp/round.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// y[i] += x[i]. Identical shape required.
void add_inplace(Tensor& y, const Tensor& x);


// y[i] += s. Dispatched FP32/FP16 on y.dtype.
void add_scalar_inplace(Tensor& y, float s);


// y[i] *= s. Dispatched FP32/FP16 on y.dtype.
void scale_inplace(Tensor& y, float s);


// y[i] = min(max(y[i], lo), hi), in place. Dispatched FP32/FP16 on y.dtype.
void clamp(Tensor& y, float lo, float hi);


// Dtype cast: dst = src converted to out_dtype. dst resized to (src.rows,
// src.cols, out_dtype) on src's device. Supports FP32<->FP16, FP32<->BF16, and
// a same-dtype passthrough copy; other pairs throw. The standard
// mixed-precision primitive (low-precision weight <-> FP32 master copy).
void cast(const Tensor& src, Tensor& dst, Dtype out_dtype);


// Y[i] += X[i] over (B,D). Identical shape required.
void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD);


// y[i] *= x[i]. Identical shape and dtype; dispatched FP32/FP16 on y.dtype.
void mul_inplace(Tensor& y, const Tensor& x);


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

}  // namespace brotensor
