#pragma once

// brotensor ops/quant.h — Standalone quantization: W8A16 matmul + host quantizer, GGUF Q4_K/Q6_K/Q8_0.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


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


// ─── Per-row weight-only quantisation (INT8 / FP8 E4M3) ────────────────────
//
// Symmetric per-output-row quantisation of a weight W (out, in): scale[r] =
// max|W[r, :]| / qmax (qmax 127 for INT8, 448 for FP8_E4M3; 0 for an all-zero
// row) and Wq[r, k] = W[r, k] / scale[r] rounded to the format (INT8: round
// half to even, clamped to +-127; E4M3: round to nearest even, saturating).
//   W: (out, in) FP16 / BF16 / FP32 (CUDA), FP32 (CPU).  fmt: INT8 or FP8_E4M3.
//   Wq: (out, in) fmt, resized.  scales: (out, 1) FP32, resized.
void quantize_rows(const Tensor& W, Dtype fmt, Tensor& Wq, Tensor& scales);

// W = scale * Wq, row by row: quantize_rows's inverse, for a prefill that
// runs a GEMM over the quantised weights without keeping the originals (the
// caller dequantises a layer into one shared scratch, uses it, moves on).
//   Wq: (out, in) INT8 or FP8_E4M3.  scales: (out, 1) FP32.
//   out: FP16 / BF16 / FP32 (CUDA), FP32 (CPU).  W: (out, in) out; a W that
//   already has that shape and dtype (a view into a scratch) is written in
//   place, otherwise it is resized.
void dequantize_rows(const Tensor& Wq, const Tensor& scales, Dtype out, Tensor& W);

// Y = X @ (scale * Wq)^T + bias, with linear_forward_batched_ex's activation
// and epilogues (kLinearEpiStore / Accumulate / Geglu / Swiglu; the gated
// ones read rows as that op does). FP32 accumulation; the row scale multiplies
// the dot product before bias and activation. Decode-shaped: CUDA reads each
// weight row once for up to eight input rows (larger batches run in slices of
// eight), at one byte per weight.
//   Wq: (out, in) INT8 or FP8_E4M3 (from quantize_rows).  scales: (out, 1) FP32.
//   bias: (out, 1) in X's dtype, or null.  X_BD: (B, in) FP16 / BF16 (CUDA),
//   FP32 (CPU).  Y_BD: (B, out) or (B, out/2) gated, X's dtype; accumulate
//   needs it sized.
void linear_forward_batched_wq_ex(const Tensor& Wq, const Tensor& scales, const Tensor* bias,
                                  const Tensor& X_BD, int act, int epilogue, Tensor& Y_BD);


// ─── GGUF Q4_K (W4A16) ─────────────────────────────────────────────────────
//
// Q4_K is a 256-element block quantization: 144 bytes per block, with
// per-block FP16 scale + min and eight 6-bit sub-block scales / mins, each
// element a 4-bit nibble. Cols must be a multiple of 256 (the block runs
// along the inner / contiguous axis).

// W_q4k: (out, in) Dtype::Q4_K. W_fp16: (out, in) Dtype::FP16, resized.
// Pure dequantization — useful for tests and for callers who want a one-shot
// dequant before reusing the FP16 weight across many matmuls.
void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16);


// GEMV: y(out, 1) = W_q4k(out, in) @ x(in, 1) + bias(out, 1)?
// x and y are FP16 (Dtype::FP16). bias is optional. FP32 accumulation. The
// kernel fuses the Q4_K dequant into the matmul — no temporary FP16 weight
// is materialized.
void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                             const Tensor& x, Tensor& y);


// Batched form: Y(B, out) = X(B, in) @ W_q4k(out, in)^T + bias(out)? Same
// (B, in) -> (B, out) row layout as linear_forward_batched_fp16. Small B is
// GEMV-optimized (one decode of the weight per output token); large B (prefill)
// takes a tensor-core GEMM path — CUDA a fused-dequant WMMA kernel, Metal a
// dequant-once + simdgroup matmul — so the weight decode is amortized across B.
void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD);


// ─── GGUF Q8_0 (W8A16-style) ───────────────────────────────────────────────
//
// Q8_0 is a 32-element block quantization: 34 bytes per block, with one FP16
// scale `d` and 32 int8 quants per block. Decoded value is just d * qs[i] —
// no offset, no nested sub-scales. Cols must be a multiple of 32 (the block
// runs along the inner / contiguous axis).

// W_q8: (out, in) Dtype::Q8_0. W_fp16: (out, in) Dtype::FP16, resized.
void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16);


// GEMV: y(out, 1) = W_q8(out, in) @ x(in, 1) + bias(out, 1)?
// x and y are FP16. bias is optional. FP32 accumulation. The kernel fuses the
// Q8_0 dequant into the matmul — no temporary FP16 weight is materialized.
void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y);


// Batched form: Y(B, out) = X(B, in) @ W_q8(out, in)^T + bias(out)?. Large B
// (prefill) uses a tensor-core GEMM — a fused-dequant WMMA kernel on CUDA, a
// dequant-once + simdgroup matmul on Metal; small B falls back to a per-token
// GEMV that re-decodes the weight.
void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD);


// ─── GGUF Q6_K (W6A16) ─────────────────────────────────────────────────────
//
// Q6_K is a 256-element block quantization: 210 bytes per block, with one
// FP16 super-block scale `d`, sixteen signed int8 sub-block scales, and each
// element a 6-bit signed value packed across ql[128] (low 4 bits) and qh[64]
// (high 2 bits). Cols must be a multiple of 256 (the block runs along the
// inner / contiguous axis).

// W_q6k: (out, in) Dtype::Q6_K. W_fp16: (out, in) Dtype::FP16, resized.
void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16);


// GEMV: y(out, 1) = W_q6k(out, in) @ x(in, 1) + bias(out, 1)?
// x and y are FP16. bias is optional. FP32 accumulation, fused dequant.
void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                             const Tensor& x, Tensor& y);


// Batched form. WMMA path when B >= 4 and K is aligned; GEMV-loop fallback
// otherwise. Same (B, in) -> (B, out) layout as the FP16 batched linear.
void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD);

}  // namespace brotensor
