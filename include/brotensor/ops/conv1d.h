#pragma once

// brotensor ops/conv1d.h — 1D convolution family (audio): conv1d, pad1d, conv_transpose1d, causal_conv1d.

#include "../tensor.h"
#include "conv.h"
#include <cstdint>

namespace brotensor {


// ─── 1D convolution family (audio) ─────────────────────────────────────────
//
// The audio counterpart of the conv2d family — the building block of
// WaveNet / Conformer / vocoder stacks.
//
// Layout (NCL): a 1D-conv activation is (N, C*L) — N signals folded into rows,
// each row a flat C-major / L-minor buffer X[(n*C+c)*L + l] (the NCHW
// convention with the height axis dropped). Weights are OIL:
// Wt[(c_out*(C_in/groups)+c_in_local)*kL + kl].
//
// conv1d, its three backward halves, and conv1d_int8w_fp16 are header-only
// inline wrappers over the conv2d ops (a 1D conv is a 2D conv with H=kH=1), so
// every backend that implements conv2d gets conv1d for free; conv1d_int8w_fp16
// therefore throws on the CPU backend (conv2d_int8w is a null CPU slot).
// causal_conv1d is likewise a wrapper (left-pad, then a valid conv1d).
// pad1d, conv_transpose1d, and causal_conv1d_update are genuinely new ops with
// their own vtable rows, implemented on all three backends (CPU / CUDA / Metal).

// Pad the length axis of an NCL tensor by pad_left / pad_right samples — the
// temporal analogue of an image pad (causal-conv left padding, "same" padding,
// reflect padding). `mode`: 0 zero, 1 reflect (mirror without repeating the
// edge sample; requires pad < L), 2 replicate (clamp to the edge sample).
//   X: (N, C*L).  Y: (N, C*(L+pad_left+pad_right)), resized + dtype-set to X.
void pad1d_forward(const Tensor& X, int N, int C, int L,
                   int pad_left, int pad_right, int mode, Tensor& Y);


// Backward (adjoint) of pad1d: each input sample sums the gradients of the
// output samples that read it. dX overwritten (resized + dtype-set to dY).
//   dY: (N, C*(L+pad_left+pad_right)).  dX: (N, C*L).
// N, C, L and the pad / mode args match the forward call.
void pad1d_backward(const Tensor& dY, int N, int C, int L,
                    int pad_left, int pad_right, int mode, Tensor& dX);


// 1D convolution, NCL. Header-only wrapper over conv2d_forward (H=kH=1).
//   X: (N,C_in*L).  Wt: (C_out, (C_in/groups)*kL) OIL.  bias: (C_out,1) or null.
//   Y: (N,C_out*L_out);  L_out = (L+2*padding-dilation*(kL-1)-1)/stride + 1.
inline void conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                   int N, int C_in, int L, int C_out, int kL,
                   int stride, int padding, int dilation, int groups,
                   Tensor& Y) {
    conv2d_forward(X, Wt, bias, N, C_in, /*H=*/1, /*W=*/L, C_out,
                   /*kH=*/1, /*kW=*/kL, /*stride_h=*/1, /*stride_w=*/stride,
                   /*pad_h=*/0, /*pad_w=*/padding, /*dil_h=*/1, /*dil_w=*/dilation,
                   groups, Y);
}

// Convenience overload: groups defaults to 1.
inline void conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                   int N, int C_in, int L, int C_out, int kL,
                   int stride, int padding, int dilation, Tensor& Y) {
    conv1d(X, Wt, bias, N, C_in, L, C_out, kL, stride, padding, dilation,
           /*groups=*/1, Y);
}


// conv1d backward w.r.t. input — wrapper over conv2d_backward_input (H=kH=1).
// dX overwritten.
inline void conv1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                  int N, int C_in, int L, int C_out, int kL,
                                  int stride, int padding, int dilation,
                                  int groups, Tensor& dX) {
    conv2d_backward_input(Wt, dY, N, C_in, /*H=*/1, /*W=*/L, C_out,
                          /*kH=*/1, /*kW=*/kL, 1, stride, 0, padding,
                          1, dilation, groups, dX);
}

inline void conv1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                  int N, int C_in, int L, int C_out, int kL,
                                  int stride, int padding, int dilation,
                                  Tensor& dX) {
    conv1d_backward_input(Wt, dY, N, C_in, L, C_out, kL, stride, padding,
                          dilation, /*groups=*/1, dX);
}


// conv1d backward w.r.t. weight — wrapper over conv2d_backward_weight.
// dWt accumulated — caller zeros.
inline void conv1d_backward_weight(const Tensor& X, const Tensor& dY,
                                   int N, int C_in, int L, int C_out, int kL,
                                   int stride, int padding, int dilation,
                                   int groups, Tensor& dWt) {
    conv2d_backward_weight(X, dY, N, C_in, /*H=*/1, /*W=*/L, C_out,
                           /*kH=*/1, /*kW=*/kL, 1, stride, 0, padding,
                           1, dilation, groups, dWt);
}

inline void conv1d_backward_weight(const Tensor& X, const Tensor& dY,
                                   int N, int C_in, int L, int C_out, int kL,
                                   int stride, int padding, int dilation,
                                   Tensor& dWt) {
    conv1d_backward_weight(X, dY, N, C_in, L, C_out, kL, stride, padding,
                           dilation, /*groups=*/1, dWt);
}


// conv1d backward w.r.t. bias — wrapper over conv2d_backward_bias.
// dB accumulated — caller zeros.
inline void conv1d_backward_bias(const Tensor& dY, int N, int C_out, int L_out,
                                 Tensor& dB) {
    conv2d_backward_bias(dY, N, C_out, /*H_out=*/1, /*W_out=*/L_out, dB);
}


// W8A16 1D convolution. Header-only wrapper over conv2d_int8w_fp16_forward
// (H=kH=1). FP16 activations, INT8 per-output-row weights. Throws on the CPU
// backend (no CPU W8A16 conv slot).
//   X: (N,C_in*L) FP16.  W_int8: (C_out, (C_in/groups)*kL) INT8 OIL.
//   scales: (C_out,1) FP32.  bias: (C_out,1) FP16 or null.  Y: (N,C_out*L_out) FP16.
inline void conv1d_int8w_fp16(const Tensor& X, const Tensor& W_int8,
                              const Tensor& scales, const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int dilation, int groups,
                              Tensor& Y) {
    conv2d_int8w_fp16_forward(X, W_int8, scales, bias, N, C_in, /*H=*/1,
                              /*W=*/L, C_out, /*kH=*/1, /*kW=*/kL,
                              /*stride_h=*/1, /*stride_w=*/stride,
                              /*pad_h=*/0, /*pad_w=*/padding,
                              /*dil_h=*/1, /*dil_w=*/dilation, groups, Y);
}

inline void conv1d_int8w_fp16(const Tensor& X, const Tensor& W_int8,
                              const Tensor& scales, const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int dilation, Tensor& Y) {
    conv1d_int8w_fp16(X, W_int8, scales, bias, N, C_in, L, C_out, kL, stride,
                      padding, dilation, /*groups=*/1, Y);
}


// Causal 1D convolution. Header-only wrapper: left-pad the length axis by
// dilation*(kL-1) (zero), then run a valid (padding=0) conv1d. Output length
// equals L when stride==1; every output sample depends only on inputs at or
// before its position.
//   X: (N,C_in*L).  Wt: (C_out, (C_in/groups)*kL) OIL.  bias: (C_out,1) or null.
//   scratch: caller-owned, resized to (N, C_in*(L+dilation*(kL-1))) and
//            overwritten — keeps the wrapper allocation-free across calls.
//   Y: (N,C_out*L_out), resized by conv1d.
inline void causal_conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C_in, int L, int C_out, int kL,
                          int stride, int dilation, int groups,
                          Tensor& scratch, Tensor& Y) {
    const int pad_left = dilation * (kL - 1);
    pad1d_forward(X, N, C_in, L, pad_left, /*pad_right=*/0, /*mode=*/0,
                  scratch);
    conv1d(scratch, Wt, bias, N, C_in, L + pad_left, C_out, kL, stride,
           /*padding=*/0, dilation, groups, Y);
}

inline void causal_conv1d(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C_in, int L, int C_out, int kL,
                          int stride, int dilation, Tensor& scratch, Tensor& Y) {
    causal_conv1d(X, Wt, bias, N, C_in, L, C_out, kL, stride, dilation,
                  /*groups=*/1, scratch, Y);
}


// 1D transposed convolution, NCL — the upsampling primitive of neural vocoders
// (HiFi-GAN, EnCodec/DAC decoders). A genuinely new kernel, CPU FP32-only.
//   L_out = (L-1)*stride - 2*padding + dilation*(kL-1) + output_padding + 1.
// output_padding (< stride) disambiguates the L_out values that map to one L
// under a strided forward conv (torch's ConvTranspose1d arg).
// Weight layout is input-channel-major: Wt (C_in, (C_out/groups)*kL),
// Wt[(c_in*(C_out/groups)+c_out_local)*kL + kl]. groups divides C_in and C_out;
// groups==C_in==C_out is depthwise transposed conv.
// Forward (scatter): each X[n,c_in,l] is scattered, per kernel tap kl, into
// output position l_out = l*stride - padding + kl*dilation.
//   X: (N,C_in*L).  Wt: (C_in,(C_out/groups)*kL).  bias: (C_out,1) or null.
//   Y: (N,C_out*L_out), resized + dtype-set to match X.
void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int L, int C_out, int kL,
                              int stride, int padding, int output_padding,
                              int dilation, int groups, Tensor& Y);

// Convenience overload: groups defaults to 1.
inline void conv_transpose1d_forward(const Tensor& X, const Tensor& Wt,
                                     const Tensor* bias,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     Tensor& Y) {
    conv_transpose1d_forward(X, Wt, bias, N, C_in, L, C_out, kL, stride,
                             padding, output_padding, dilation, /*groups=*/1, Y);
}


// conv_transpose1d backward w.r.t. input — the adjoint is a plain gather conv:
// dX[n,c_in,l] gathers dY over every tap and the group's output channels at
// l_out = l*stride - padding + kl*dilation. dX overwritten (resized +
// dtype-set to dY). All hyperparams match the forward call.
void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int L, int C_out, int kL,
                                     int stride, int padding,
                                     int output_padding, int dilation,
                                     int groups, Tensor& dX);

inline void conv_transpose1d_backward_input(const Tensor& Wt, const Tensor& dY,
                                            int N, int C_in, int L, int C_out,
                                            int kL, int stride, int padding,
                                            int output_padding, int dilation,
                                            Tensor& dX) {
    conv_transpose1d_backward_input(Wt, dY, N, C_in, L, C_out, kL, stride,
                                    padding, output_padding, dilation,
                                    /*groups=*/1, dX);
}


// conv_transpose1d backward w.r.t. weight:
//   dWt[c_in,c_out_local,kl] += sum_{n,l} X[n,c_in,l]*dY[n,c_out,l_out],
//   l_out = l*stride - padding + kl*dilation (skipped when OOB).
// dWt accumulated — caller zeros. All hyperparams match the forward call.
void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int L, int C_out, int kL,
                                      int stride, int padding,
                                      int output_padding, int dilation,
                                      int groups, Tensor& dWt);

inline void conv_transpose1d_backward_weight(const Tensor& X, const Tensor& dY,
                                             int N, int C_in, int L, int C_out,
                                             int kL, int stride, int padding,
                                             int output_padding, int dilation,
                                             Tensor& dWt) {
    conv_transpose1d_backward_weight(X, dY, N, C_in, L, C_out, kL, stride,
                                     padding, output_padding, dilation,
                                     /*groups=*/1, dWt);
}


// conv_transpose1d backward w.r.t. bias:
//   dB[c_out] += sum_{n,l_out} dY[n,c_out,l_out].
// dB accumulated — caller zeros.
void conv_transpose1d_backward_bias(const Tensor& dY, int N, int C_out,
                                    int L_out, Tensor& dB);


// One streaming step of a causal depthwise 1D conv against a rolling state
// cache (in the spirit of kv_cache_append) — for autoregressive / streaming
// decoders. Forward-only, new vtable row, CPU FP32-only.
// Depthwise: C channels in and out, one length-kL filter per channel. With
// L_step new samples it produces L_step outputs:
//   Y[n,c,t] = bias[c] + sum_{kl} W[c,kl] * buf[n,c,t+kl*dilation],
//   buf = state[n,c,:] ++ X[n,c,:].
// `state` is updated in place to the last (kL-1)*dilation samples of buf, so a
// sequence of calls reproduces one full causal_conv1d over the concatenated
// input (caller zero-initialises state before the first step).
//   X: (N,C*L_step) new samples.  Wt: (C,kL) depthwise filter.
//   bias: (C,1) or null.  state: (N,C*(kL-1)*dilation) — read AND overwritten.
//   Y: (N,C*L_step), resized + dtype-set to match X.
void causal_conv1d_update(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                          int N, int C, int L_step, int kL, int dilation,
                          Tensor& state, Tensor& Y);

}  // namespace brotensor
