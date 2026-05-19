#pragma once

#include "tensor.h"

#include <cstdint>

namespace brotensor {

// ─── CPU primitive ops ─────────────────────────────────────────────────────
//
// Plain C-style functions over `brotensor::Tensor`. Hand-written, scalar or
// simple SIMD-friendly loops; compilers autovectorize what matters and any
// AVX2 hot paths can be dropped in per-call site later. Shape contracts are
// documented per-op and enforced by assert in debug.
//
// Mirror of the GPU surface in <brotensor/ops.h> with `_cpu` suffix and host
// `Tensor` arguments. The contracts are bitwise-spec-identical to the GPU
// ops; differences are limited to dtype (CPU is FP32-only for now) and
// dispatch type (Tensor vs GpuTensor).

// y = W * x + b   (W: out x in, x: in, b: out, y: out)
void linear_forward_cpu(const Tensor& W, const Tensor& b, const Tensor& x, Tensor& y);

// Given dY (out-dim), produce dX (in-dim), accumulate dW (out x in), dB (out).
// dW and dB are accumulated into — caller zeros them once per batch.
void linear_backward_cpu(const Tensor& W, const Tensor& x, const Tensor& dY,
                         Tensor& dX, Tensor& dW, Tensor& dB);

// Elementwise ReLU. y = max(x, 0). In-place safe if &x == &y.
void relu_forward_cpu(const Tensor& x, Tensor& y);
// dX = dY * (x > 0).
void relu_backward_cpu(const Tensor& x, const Tensor& dY, Tensor& dX);

// Elementwise tanh.
void tanh_forward_cpu(const Tensor& x, Tensor& y);
// dX = dY * (1 - y^2)  — cached `y` avoids recomputing tanh.
void tanh_backward_cpu(const Tensor& y, const Tensor& dY, Tensor& dX);

// Elementwise sigmoid. y = 1/(1+exp(-x)).
void sigmoid_forward_cpu(const Tensor& x, Tensor& y);
// dX = dY * y * (1 - y)  — cached `y` avoids recomputing sigmoid.
void sigmoid_backward_cpu(const Tensor& y, const Tensor& dY, Tensor& dX);

// Softmax over a flat vector. Stable (subtract max). Optional mask: if
// non-null, elements with mask[i] == 0 contribute 0 to normaliser and
// receive output 0. At least one mask entry must be >0 when masking.
void softmax_forward_cpu(const Tensor& logits, Tensor& probs, const float* mask = nullptr);

// Given probs and upstream dL/dprobs, produce dL/dlogits.
// Full Jacobian: dL/dz_i = sum_j dL/dp_j * p_j * (δ_ij - p_i).
void softmax_backward_cpu(const Tensor& probs, const Tensor& dProbs, Tensor& dLogits);

// Combined softmax + cross-entropy backward for a one-hot or soft target.
// Convenient because the gradient collapses to (p - target). Mask is the
// same legal-action mask — illegal entries are set to 0 in `probs` and the
// gradient ignores them.
//
// Returns scalar loss = -sum_i target_i * log(p_i) (illegal ignored).
float softmax_xent_cpu(const Tensor& logits, const Tensor& target,
                       Tensor& probs, Tensor& dLogits,
                       const float* mask = nullptr);

// Pointer/length form of softmax_xent_cpu. Operates on n contiguous floats
// starting at the supplied pointers. Used by callers that want to apply
// xent to a segment of a larger logit/target buffer (e.g. the per-head
// policy loss in GenericExItTrainer) without copying through temporary
// Tensors. Same return value semantics as softmax_xent_cpu.
float softmax_xent_segment_cpu(const float* logits, const float* target,
                               float* probs, float* dLogits,
                               int n, const float* mask = nullptr);

// Mean-squared error for scalar value head. pred and target are both size 1.
// Returns 0.5 * (pred - target)^2; dPred = (pred - target).
float mse_scalar_cpu(float pred, float target, float& dPred);

// y += x  (elementwise, same shape).
void add_inplace_cpu(Tensor& y, const Tensor& x);

// y[i] += s  (broadcast scalar).
void add_scalar_inplace_cpu(Tensor& y, float s);

// Deterministic xavier-uniform init for a Linear weight matrix.
// rng_state is a 64-bit splitmix state advanced in place.
void xavier_init_cpu(Tensor& W, uint64_t& rng_state);

} // namespace brotensor
