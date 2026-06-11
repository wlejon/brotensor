#pragma once

// brotensor ops/conv.h — Convolution: conv2d / conv3d / conv_transpose2d (+ int8 weight variants).

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// ─── Convolution: conv2d / conv3d / conv_transpose2d ───────────────────────
//
// NCHW tensors are carried as flat (rows, cols) buffers; the (N,C,H,W) dims
// are passed as int args. Unless noted, these ops are dtype-dispatched on the
// primary input (FP32 or FP16) with FP32 internal accumulation, and resize +
// dtype-set their outputs to match.

// 2D convolution, NCHW. Dispatched on X.dtype (FP32/FP16); Wt, bias, Y share
// it. FP32 accumulation.
//   X:    (N, C_in*H*W).
//   Wt:   (C_out, (C_in/groups)*kH*kW)  OIHW filter layout.
//   bias: (C_out,1) or null.
//   Y:    (N, C_out*H_out*W_out), resized + dtype-set to match X.
//   groups: divides C_in and C_out. Output channel c_out belongs to group
//           g = c_out/(C_out/groups) and reads only input channels
//           [g*(C_in/groups), (g+1)*(C_in/groups)). groups=1 is standard conv;
//           groups==C_in==C_out is depthwise (Wt becomes (C_out, kH*kW)).
//   H_out = (H + 2*pad_h - dil_h*(kH-1) - 1)/stride_h + 1   (W_out analogous).
void conv2d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    Tensor& Y);

// Convenience overload: groups defaults to 1 (full convolution).
inline void conv2d_forward(const Tensor& X,
                           const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           Tensor& Y) {
    conv2d_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                   /*groups=*/1, Y);
}


// 2D convolution backward w.r.t. input. Dtype-dispatched (FP32/FP16); Wt and
// dY share dtype, dX matches. All conv hyperparams match the forward call.
//   dX[n,c_in,i,j] = sum over kernel taps (kh,kw) and the group's output
//   channels of dY[n,c_out,i_out,j_out]*Wt[c_out,c_in_local,kh,kw], where
//     i_out = (i + pad_h - dil_h*kh)/stride_h   (j_out analogous),
//   counted only when divisible by stride and in [0,H_out)x[0,W_out).
//   Wt: (C_out, (C_in/groups)*kH*kW)  forward filter, OIHW.
//   dY: (N, C_out*H_out*W_out)        upstream gradient.
//   dX: (N, C_in*H*W)                 overwritten, resized + dtype-set to dY.
void conv2d_backward_input(const Tensor& Wt,
                           const Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           Tensor& dX);

// Convenience overload: groups defaults to 1.
inline void conv2d_backward_input(const Tensor& Wt,
                                  const Tensor& dY,
                                  int N, int C_in, int H, int W,
                                  int C_out, int kH, int kW,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dil_h, int dil_w,
                                  Tensor& dX) {
    conv2d_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW,
                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                          /*groups=*/1, dX);
}


// 2D convolution backward w.r.t. weights. Dtype-dispatched (FP32/FP16); X, dY,
// dWt share dtype. All conv hyperparams match the forward call.
//   dWt[c_out,c_in_local,kh,kw] += sum over (n,i_out,j_out) of
//     dY[n,c_out,i_out,j_out] *
//     X[n,c_in, stride_h*i_out-pad_h+dil_h*kh, stride_w*j_out-pad_w+dil_w*kw]
//   (OOB input reads treated as zero), with c_in = g*(C_in/groups)+c_in_local,
//   g = c_out/(C_out/groups).
//   X:   (N, C_in*H*W)               forward input.
//   dY:  (N, C_out*H_out*W_out)      upstream gradient.
//   dWt: (C_out, (C_in/groups)*kH*kW)  accumulated — caller zeros.
void conv2d_backward_weight(const Tensor& X,
                            const Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            Tensor& dWt);

// Convenience overload: groups defaults to 1.
inline void conv2d_backward_weight(const Tensor& X,
                                   const Tensor& dY,
                                   int N, int C_in, int H, int W,
                                   int C_out, int kH, int kW,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dil_h, int dil_w,
                                   Tensor& dWt) {
    conv2d_backward_weight(X, dY, N, C_in, H, W, C_out, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                           /*groups=*/1, dWt);
}


// 2D convolution backward w.r.t. bias. Dtype-dispatched (FP32/FP16); dY and dB
// share dtype.
//   dB[c_out] += sum over (n,i_out,j_out) of dY[n,c_out,i_out,j_out].
//   dY: (N, C_out*H_out*W_out)  upstream gradient.
//   dB: (C_out,1)               accumulated — caller zeros.
void conv2d_backward_bias(const Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          Tensor& dB);


// Modulated deformable 2D convolution forward — torchvision `deform_conv2d`
// (Deformable ConvNets v2), NCHW, forward/inference only. Each output pixel's
// kH×kW sampling grid is shifted per-tap by a learned `offset` field and
// (optionally) reweighted by a learned `mask` modulator; taps are bilinearly
// sampled from X with ZERO padding outside the input (torchvision convention).
//   X:      (N, C_in*H*W).
//   offset: (N, deform_groups*2*kH*kW * H_out*W_out). Channel-major within a
//           batch row: channel = grp*(2*kH*kW) + 2*(kh*kW+kw) [+1 for the col
//           axis], then (H_out,W_out). Channel 2*(kh*kW+kw) is the ROW (y)
//           offset, +1 the COL (x) offset — matches an offset_conv whose output
//           channels are laid out (deform_groups, 2, kH, kW).
//   mask:   (N, deform_groups*kH*kW * H_out*W_out) or null. null == plain
//           deformable conv (all modulators 1). Channel grp*(kH*kW)+(kh*kW+kw).
//   Wt:     (C_out, (C_in/groups)*kH*kW)  OIHW (same layout as conv2d_forward).
//   bias:   (C_out,1) or null.
//   Y:      (N, C_out*H_out*W_out), resized + dtype-set to match X.
//   groups divides C_in and C_out (regular conv grouping); deform_groups
//   divides C_in (offset/mask grouping). H_out/W_out follow the conv2d formula.
// Dispatched FP32/FP16 on X.dtype (CPU is FP32-only); FP32 accumulation. No
// backward (an inference op — BiRefNet's ASPP-deformable decoder).
void deform_conv2d_forward(const Tensor& X,
                           const Tensor& offset,
                           const Tensor* mask,
                           const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups, int deform_groups,
                           Tensor& Y);


// 3D convolution, NCTHW (forward only). Dispatched on X.dtype (FP32/FP16/BF16
// on GPU; CPU is FP32-only). FP32 accumulation.
//   X:    (N, C_in*T*H*W).
//   Wt:   (C_out, (C_in/groups)*kT*kH*kW)  OICTHW filter layout (grouped).
//   bias: (C_out,1) or null.
//   Y:    (N, C_out*T_out*H_out*W_out), resized + dtype-set to match X.
//   Per-axis stride/pad/dilation: (_t, _h, _w). `groups` divides C_in and C_out
//   exactly as in conv2d (output channel c_out belongs to group
//   c_out/(C_out/groups); reads input channels of that group only).
//   T_out = (T + 2*pad_t - dil_t*(kT-1) - 1)/stride_t + 1   (H_out, W_out analogous).
// Y is OVERWRITTEN (the kernel stores acc directly), matching conv2d_forward.
void conv3d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                    int N, int C_in, int T, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups,
                    Tensor& Y);

// Convenience overload: groups defaults to 1 (full convolution).
inline void conv3d_forward(const Tensor& X,
                           const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int T, int H, int W,
                           int C_out, int kT, int kH, int kW,
                           int stride_t, int stride_h, int stride_w,
                           int pad_t, int pad_h, int pad_w,
                           int dil_t, int dil_h, int dil_w,
                           Tensor& Y) {
    conv3d_forward(X, Wt, bias, N, C_in, T, H, W, C_out, kT, kH, kW,
                   stride_t, stride_h, stride_w, pad_t, pad_h, pad_w,
                   dil_t, dil_h, dil_w, /*groups=*/1, Y);
}


// W8A16 3D convolution forward — the Qwen3-VL patch-embed variant of
// conv3d_forward. Same NCTHW / OICTHW layout, same per-axis stride/pad/dilation
// and groups semantics; X / bias FP16, W_int8 INT8 with per-output-row FP32
// dequant scales (analogous to conv2d_int8w_fp16_forward). GPU-only — the CPU
// vtable slot is left null and the dispatcher throws "not implemented on CPU".
//   X:        (N, C_in*T*H*W) FP16.
//   W_int8:   (C_out, (C_in/groups)*kT*kH*kW) INT8.
//   scales:   (C_out, 1) FP32.
//   bias:     (C_out, 1) FP16 or null.
//   Y:        (N, C_out*T_out*H_out*W_out) FP16, resized + dtype-set as needed.
void conv3d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int T, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w,
                               int groups,
                               Tensor& Y);

// Convenience overload: groups defaults to 1.
inline void conv3d_int8w_fp16_forward(const Tensor& X,
                                      const Tensor& W_int8,
                                      const Tensor& scales,
                                      const Tensor* bias,
                                      int N, int C_in, int T, int H, int W,
                                      int C_out, int kT, int kH, int kW,
                                      int stride_t, int stride_h, int stride_w,
                                      int pad_t, int pad_h, int pad_w,
                                      int dil_t, int dil_h, int dil_w,
                                      Tensor& Y) {
    conv3d_int8w_fp16_forward(X, W_int8, scales, bias,
                              N, C_in, T, H, W, C_out, kT, kH, kW,
                              stride_t, stride_h, stride_w,
                              pad_t, pad_h, pad_w,
                              dil_t, dil_h, dil_w, /*groups=*/1, Y);
}


// 2D transposed convolution, NCHW. The 2D counterpart of conv_transpose1d,
// generalising independently across H and W. Used by SAM's mask decoder
// (4x upsampler), DPT depth heads, and any segmentation decoder that
// learns the upsample. Output spatial dims (torch ConvTranspose2d formula):
//   H_out = (H - 1)*stride_h - 2*pad_h + dilation_h*(kH-1) + output_padding_h + 1
//   W_out = (W - 1)*stride_w - 2*pad_w + dilation_w*(kW-1) + output_padding_w + 1
// Weight layout is input-channel-major (transposed-conv convention):
//   Wt: (C_in, (C_out/groups)*kH*kW) with index
//       (c_in*(Cg_out*kH*kW) + (oc_local*kH + kh)*kW + kw)
// bias may be null. groups must divide both C_in and C_out (default 1).
// output_padding must be < stride or < dilation on each axis (matches torch).
//   X: (N, C_in*H*W).  Y: (N, C_out*H_out*W_out), resized + dtype-set.
// CPU backend is FP32-only; the CUDA forward is dtype-dispatched on X
// (FP32/FP16/BF16 — Wt and bias must match, FP32 accumulation). The three
// backward ops are FP32-only on both backends.
void conv_transpose2d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              Tensor& Y);


// Backward to the input. dX OVERWRITTEN — the adjoint of the forward
// scatter is a plain gather conv (cross-correlation in disguise).
//   Wt: (C_in, (C_out/groups)*kH*kW).  dY: (N, C_out*H_out*W_out).
//   dX: (N, C_in*H*W), resized + dtype-set.
void conv_transpose2d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     Tensor& dX);


// Backward to the weights. dWt ACCUMULATES (+=) — caller zeros it first
// (matches the conv2d contract).
//   X: (N, C_in*H*W).  dY: (N, C_out*H_out*W_out).
//   dWt: (C_in, (C_out/groups)*kH*kW), pre-zeroed by caller.
void conv_transpose2d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      Tensor& dWt);


// Backward to the bias. dB ACCUMULATES (+=) — caller zeros first.
//   dY: (N, C_out*H_out*W_out).  dB: (C_out, 1).
void conv_transpose2d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int H_out, int W_out, Tensor& dB);


// W8A16 conv2d forward. Mirrors conv2d_forward; only the weight dtype differs.
//   W_int8: (C_out, C_in/groups*kH*kW) INT8 OIHW, quantised per output channel.
//   scales: (C_out,1) FP32 per-output-channel dequant scales.
//   bias: (C_out,1) FP16 or null.  X, Y: FP16, layout as conv2d_forward.
void conv2d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y);

}  // namespace brotensor
