#pragma once

// brotensor ops/rope.h — Rotary position embedding: rope forward/backward/apply/mrope.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


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


// RoPE with explicit PER-HEAD cos/sin tables — like rope_apply, but each head
// carries its own rotation angles (content-dependent / per-head positional
// schemes, e.g. TripoSplat's RePo3D rotary where every attention head predicts
// its own 3-D delta position). Pairing is the same adjacent-pair (interleaved)
// convention as rope_apply.
//   x_{2i}   <- x_{2i}*cos_tbl[(row*num_heads+h),i] - x_{2i+1}*sin_tbl[...]
//   x_{2i+1} <- x_{2i}*sin_tbl[(row*num_heads+h),i] + x_{2i+1}*cos_tbl[...]
//   X, Y: (L, num_heads*head_dim); head_dim even.
//   cos_tbl, sin_tbl: (L*num_heads, head_dim/2) FP32 — one angle per
//     (row, head, pair), rows ordered head-minor within each token.
//   Y resized + dtype-set to X. Dispatched on X.dtype (FP32/FP16/BF16); FP32
//   math. Inference-only (no backward).
void rope_apply_perhead(const Tensor& X, const Tensor& cos_tbl,
                        const Tensor& sin_tbl, int head_dim, int num_heads,
                        Tensor& Y);


// Backward of rope_apply — the inverse (transpose) rotation:
//   dX_{2i}   <-  dY_{2i}*cos_tbl[row,i] + dY_{2i+1}*sin_tbl[row,i]
//   dX_{2i+1} <- -dY_{2i}*sin_tbl[row,i] + dY_{2i+1}*cos_tbl[row,i]
//   dX, dY: (L, num_heads*head_dim). cos_tbl/sin_tbl as in rope_apply.
// Dispatched on dY.dtype.
void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl,
                         const Tensor& sin_tbl, int head_dim, int num_heads,
                         Tensor& dX);


// M-RoPE (Qwen2.5-VL / Qwen3-VL): three independent per-axis position streams
// (t, h, w) rotating disjoint sub-ranges of each head_dim. head_dim is split
// into three contiguous scalar sub-ranges of widths 2*d_t, 2*d_h, 2*d_w (in
// order t, h, w) with 2*(d_t + d_h + d_w) == head_dim. Within sub-range a the
// op rotates pairs (x[2*i], x[2*i+1]) by angle
//   theta = cos_a[pos_a[row], i_local], sin_a[pos_a[row], i_local]
// where i_local is the pair index within sub-range a (0..d_a-1).
//   X, Y: (L, num_heads*head_dim).
//   cos_a, sin_a: (max_pos_a, d_a) FP32, shared across heads.
//   pos_t / pos_h / pos_w: length-L INT32 position-ID streams.
//     CPU backend: host pointers. CUDA/Metal: device pointers (mirrors
//     flash_attention_varlen_forward's cu_seqlens convention).
//   Y resized + dtype-set to X. Dispatched on X.dtype (FP32/FP16/BF16); FP32
//   math. Inference-only (no backward).
//
// Degenerate case: with d_h == d_w == 0 and pos_t = {0,1,2,...,L-1}, this
// reproduces rope_apply exactly.
void rope_apply_mrope(const Tensor& X,
                      const Tensor& cos_t, const Tensor& sin_t,
                      const Tensor& cos_h, const Tensor& sin_h,
                      const Tensor& cos_w, const Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h,
                      const int32_t* pos_w,
                      int head_dim, int num_heads,
                      int d_t, int d_h, int d_w,
                      Tensor& Y);

}  // namespace brotensor
