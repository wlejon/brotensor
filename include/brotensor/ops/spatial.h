#pragma once

// brotensor ops/spatial.h — NCHW spatial layout: pad2d, slice2d, unfold2d, window partition, patch merge, transpose.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// 2D pad on the H and W axes of an NCHW tensor (image padding — same role as
// torch.nn.functional.pad with a 4-element pad). `mode`: 0 zero, 1 reflect
// (mirror without repeating the edge sample; requires pad_top/pad_bottom < H
// and pad_left/pad_right < W), 2 replicate (clamp to the edge sample). The
// 2D counterpart to pad1d, with H/W independently padded.
//   X: (N, C*H*W).  Y: (N, C*(H+pt+pb)*(W+pl+pr)), resized + dtype-set to X.
// CPU is FP32-only. GPU dispatch follows the registered backend's dtype
// support (CUDA + Metal TBD — null slots until those follow-ups land).
void pad2d_forward(const Tensor& X, int N, int C, int H, int W,
                   int pad_top, int pad_bottom, int pad_left, int pad_right,
                   int mode, Tensor& Y);


// Backward (adjoint) of pad2d_forward: each input pixel sums the gradients
// of the output pixels that read it (reflect / replicate may collapse
// several output positions into one input — those gradients are summed).
// dX overwritten, resized + dtype-set to dY.
//   dY: (N, C*(H+pt+pb)*(W+pl+pr)).  dX: (N, C*H*W).
// All shape / pad / mode args match the forward call.
void pad2d_backward(const Tensor& dY, int N, int C, int H, int W,
                    int pad_top, int pad_bottom, int pad_left, int pad_right,
                    int mode, Tensor& dX);


// 2D spatial slice / crop on NCHW. Extracts the (H_out, W_out) sub-region
// starting at (h0, w0); N and C pass through unchanged. Used for RoI
// extraction, prompt-region cropping, window-attention partitioning, and
// generally as the inverse of pad2d's zero mode.
// Preconditions (throws on violation):
//   h0 >= 0,  w0 >= 0,  H_out >= 0,  W_out >= 0,
//   h0 + H_out <= H,    w0 + W_out <= W.
// h0 == 0 && w0 == 0 && H_out == H && W_out == W is the identity copy.
//   X: (N, C*H*W).  Y: (N, C*H_out*W_out), resized + dtype-set to X.
void slice2d_forward(const Tensor& X, int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out, Tensor& Y);


// Backward (adjoint) of slice2d_forward: dX is zeroed, then dY is copied
// into the same (h0, w0)+(H_out, W_out) sub-region. Pixels outside the
// slice contribute zero.
//   dY: (N, C*H_out*W_out).  dX: (N, C*H*W), resized + dtype-set to dY.
// All shape / offset args match the forward call.
void slice2d_backward(const Tensor& dY, int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out, Tensor& dX);


// 2D neighborhood unfold (spatial-preserving im2col), NCHW. For every output
// pixel, gathers the kH×kW window around the corresponding input position into
// its own channel block — the "keep the spatial grid, add a neighbor axis"
// flavour of im2col (neighborhood attention, guided/bilateral filtering, DSINE
// NRN propagation), distinct from torch.nn.Unfold's column-collapse form.
//   X: (N, C*H*W).
//   Y: (N, C*kK*H_out*W_out), kK = kH*kW, resized + dtype-set to X.
//   Y[n, c, k, oy, ox] = X[n, c, oy*stride_h - pad_top + ky,
//                                ox*stride_w - pad_left + kx]
//   with k = ky*kW + kx and out-of-range source resolved by `mode`:
//     0 = zero, 1 = reflect (no edge repeat), 2 = replicate (clamp to edge).
//   H_out = (H + pad_top + pad_bottom - kH)/stride_h + 1   (W_out analogous).
// stride 1 + pad (k-1)/2 gives the same-size neighborhood unfold (H_out==H).
// Dispatched FP32/FP16/BF16 on X.dtype (CPU is FP32-only). Inference-only:
// there is no unfold2d backward.
void unfold2d_forward(const Tensor& X,
                      int N, int C, int H, int W,
                      int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_top, int pad_bottom,
                      int pad_left, int pad_right,
                      int mode,
                      Tensor& Y);


// SAM-style window partition. Splits the (H, W) spatial plane of an NCHW
// tensor into (H/window) * (W/window) non-overlapping window-sized tiles
// and stacks them as a longer batch dimension. H and W must be multiples
// of `window` (caller handles any padding via pad2d).
//   X: (N, C*H*W).
//   Y: (N * nw_h * nw_w, C * window * window), with nw_h = H/window,
//      nw_w = W/window. Row index = n*nw_h*nw_w + nh*nw_w + nw, within-row
//      layout is (C, window, window) flat.
// No backward op — window_reverse_forward IS the adjoint (and exactly the
// inverse). Use it to map gradients back from the windowed batch to NCHW.
void window_partition_forward(const Tensor& X, int N, int C, int H, int W,
                              int window, Tensor& Y);


// Inverse of window_partition_forward. Takes the windowed batch
// (N * nw_h * nw_w, C * window * window) and reassembles it into
// (N, C * H * W). H, W must be multiples of window (and consistent with
// the input's first-axis row count).
//   X: (N * nw_h * nw_w, C * window * window).
//   Y: (N, C*H*W).
// No backward op — window_partition_forward IS the adjoint.
void window_reverse_forward(const Tensor& X, int N, int C, int H, int W,
                            int window, Tensor& Y);


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


// ─── Spatial 2x2 patch merger (Qwen2.5-VL / Qwen3-VL) ──────────────────────
//
// Patch merger used before the vision -> LLM projector. Stacks each 2x2 spatial
// block of X into the channel axis, producing 4x as many channels and half the
// spatial extent in each direction:
//   X: (N, C*H*W) — NCHW; H and W must both be even.
//   Y: (N, 4*C*(H/2)*(W/2)) — NCHW with (C_out, H_out, W_out) =
//                              (4*C, H/2, W/2). Resized + dtype-set to X.
// Layout:
//   c_out = (dh*2 + dw)*C + c_in,    dh in {0,1}, dw in {0,1}, c_in in [0,C).
//   (h_out, w_out) maps to (h_in = 2*h_out + dh, w_in = 2*w_out + dw).
// Pure gather — no arithmetic. Inference-only (no backward).
// Dispatched on X.dtype (FP32/FP16/BF16 on GPU; FP32 only on CPU).
void spatial_merge_2x2_forward(const Tensor& X,
                               int N, int C, int H, int W,
                               Tensor& Y);

}  // namespace brotensor
