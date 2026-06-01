#pragma once

// brotensor ops/codec.h — Neural-codec quantizers: vector quant (VQ) + finite scalar quant (FSQ).

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


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

}  // namespace brotensor
