#pragma once

// brotensor ops/resize.h — Resampling: upsample/downsample 2x, interp2d, resample1d, convex upsample.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// 2x nearest-neighbour spatial upsample, NCHW. Output pixel (i,j) reads X at
// (i/2, j/2). Dispatched FP32/FP16 on X.dtype; Y resized + dtype-set to match X.
//   X: (N, C*H*W).  Y: (N, C*2H*2W).
void upsample_nearest_2x(const Tensor& X,
                         int N, int C, int H, int W,
                         Tensor& Y);


// 2x bilinear spatial upsample, NCHW, align_corners=False (PyTorch default for
// interpolate(scale_factor=2)). Dispatched FP32/FP16 on X.dtype; FP32 math;
// Y resized + dtype-set to match X.
//   X: (N, C*H*W).  Y: (N, C*2H*2W).
void upsample_bilinear_2x(const Tensor& X,
                          int N, int C, int H, int W,
                          Tensor& Y);


// Backward of upsample_nearest_2x:
//   dX[n,c,i,j] = sum_{a,b in {0,1}} dY[n,c,2i+a,2j+b].
// N,C,H,W are the INPUT (pre-upsample) dims. Dispatched FP32/FP16 on dY.dtype;
// FP32 accumulation; dX (N,C*H*W) overwritten, resized + dtype-set to dY.
void upsample_nearest_2x_backward(const Tensor& dY,
                                  int N, int C, int H, int W,
                                  Tensor& dX);


// Backward of upsample_bilinear_2x (align_corners=False): scatters each
// dY pixel into its 4 source pixels with the forward's bilinear weights.
// N,C,H,W are the INPUT dims. Dispatched FP32/FP16 on dY.dtype; FP32
// accumulation; dX (N,C*H*W) overwritten, resized + dtype-set to dY.
void upsample_bilinear_2x_backward(const Tensor& dY,
                                   int N, int C, int H, int W,
                                   Tensor& dX);


// Arbitrary-scale 2D spatial resample, NCHW, align_corners=False (half-pixel).
// The general counterpart to the fixed-2x upsample_*_2x family — use those
// when scale is exactly 2 (they're cheaper); use this for anything else
// (SAM 1024->64, DPT 4x upsample, depth-anything head, etc.).
//
// For output position (oh, ow) the source coordinate is
//   src_y = (oh + 0.5) * (H_in / H_out) - 0.5
//   src_x = (ow + 0.5) * (W_in / W_out) - 0.5
//   mode == 0  nearest  — round_half_to_even then clamp to border.
//   mode == 1  bilinear — 2x2 tap weighted blend with border-clamped indices.
//   mode == 2  bicubic  — 4x4 cubic-convolution, a = -0.5 (Catmull-Rom, matches
//                         PIL/Pillow BICUBIC), border-clamped.
//   mode == 3  bicubic  — same, a = -0.75 (matches torch.nn.functional.
//                         interpolate mode="bicubic" and OpenCV).
//
// H_in / W_in / H_out / W_out may be any non-negative ints; H_out==H_in and
// W_out==W_in is the identity. Modes other than 0/1/2/3 throw; bicubic (2/3) is
// FP32-only. Dispatched FP32/FP16 on X.dtype where the backend supports it (CPU
// is FP32-only); Y resized + dtype-set to match X.
//   X: (N, C*H_in*W_in).  Y: (N, C*H_out*W_out).
void interp2d_forward(const Tensor& X,
                      int N, int C, int H_in, int W_in, int H_out, int W_out,
                      int mode, Tensor& Y);


// Backward (adjoint) of interp2d_forward. Scatters each dY pixel onto the
// input position(s) it sampled, with the forward's weights:
//   nearest:  dX[round_src] += dY[dst]
//   bilinear: 4 taps with the 2x2 bilinear weights
// dX is OVERWRITTEN (zero-then-scatter — resampling has no learnable params).
// mode == 2 (bicubic) is not supported for backward and throws "not
// implemented" — bicubic upsamplers are inference-only in practice.
// N, C, H_in, W_in, H_out, W_out, mode match the forward call.
//   dY: (N, C*H_out*W_out).  dX: (N, C*H_in*W_in), resized + dtype-set to dY.
void interp2d_backward(const Tensor& dY,
                       int N, int C, int H_in, int W_in, int H_out, int W_out,
                       int mode, Tensor& dX);


// align_corners=True counterpart of interp2d_forward (forward only). The source
// coordinate uses the corner-aligned mapping
//   src = o * (in - 1) / (out - 1)     (src = 0 when out == 1)
// instead of the half-pixel mapping, matching torch.nn.functional.interpolate(
// ..., align_corners=True). This is the convention DPT-style depth /
// segmentation heads (Depth-Anything, DPT) use for their fusion and final
// upsamples, so a faithful inference port needs it. nearest / bilinear / bicubic
// all honoured (mode 0/1/2), same NCHW layout and FP32/FP16 dtype rules as
// interp2d_forward. Inference-only: there is no align-corners backward.
// Registered on CPU and CUDA; the Metal slot is intentionally left null.
//   X: (N, C*H_in*W_in).  Y: (N, C*H_out*W_out).
void interp2d_align_corners_forward(
    const Tensor& X,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int mode, Tensor& Y);


// Convex (mask-based) upsample, NCHW — the RAFT-style learned-upsampler used by
// optical-flow, stereo, and surface-normal refinement (DSINE up_prob_head).
// Each low-res pixel expands to a scale×scale block; every fine pixel is a
// softmax-weighted blend of the 3×3 low-res neighborhood around its source:
//   Y[n,c,k*y+sy,k*x+sx] = sum_{m=0..8} W[n,m,sy,sx,y,x] * X[n,c,ny,nx]
//   W = softmax over the 9 neighbors m of Mask[n,m,sy,sx,y,x]
//   neighbor m: ny = clamp(y-1+m/3), nx = clamp(x-1+m%3)  (replicate pad)
// Mask layout matches torch view (N,9,k,k,H,W): flat channel = (m*k*k + sy*k + sx).
//   X:    (N, C*H*W).
//   Mask: (N, 9*scale*scale*H*W) (shares X.dtype).
//   Y:    (N, C*(scale*H)*(scale*W)), resized + dtype-set to X.
// Softmax in double. Dispatched FP32/FP16/BF16 on X.dtype (CPU is FP32-only).
// Inference-only: there is no backward.
void convex_upsample_forward(const Tensor& X, const Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             Tensor& Y);


// ─── 1D resampling (audio) ─────────────────────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal).
// Arbitrary-scale resampling along the length axis of an NCL tensor — for
// sample-rate conversion in STT / TTS / codec front-ends.

// 1D resample along the length axis: N and C pass through, the length axis is
// rescaled from L_in to any positive L_out. PyTorch align_corners=False
// convention — for output position dst the source coordinate is
//   src = (dst + 0.5)*(L_in/L_out) - 0.5
//   mode == 0  nearest — Y[dst] = X[clamp(round_half_to_even(src), 0, L_in-1)].
//   mode == 1  linear  — s=clamp(src,0,L_in-1), x0=floor(s), x1=min(x0+1,L_in-1),
//              f=s-x0;  Y[dst] = (1-f)*X[x0] + f*X[x1].
//   X: (N,C*L_in).  Y: (N,C*L_out), resized + dtype-set to FP32.
// L_out == L_in is the identity. mode other than 0/1 throws.
void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out,
                        int mode, Tensor& Y);


// Backward (adjoint) of resample1d_forward: each output gradient is scattered
// back onto the input position(s) it sampled, with the forward's weights:
//   nearest: dX[round(src)] += dY[dst]
//   linear:  dX[x0] += (1-f)*dY[dst];  dX[x1] += f*dY[dst]
//   dY: (N,C*L_out).  dX: (N,C*L_in), overwritten (resized + dtype-set to FP32).
// N, C, L_in, L_out, mode match the forward call.
void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out,
                         int mode, Tensor& dX);

}  // namespace brotensor
