#pragma once

// brotensor ops/pooling.h — Pooling: max_pool2d, adaptive_avg_pool2d, avg downsample, masked mean-pool.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {

// Masked mean-pool over the rows of a (K,D) matrix.
//   X: (K,D).  d_mask: K-float device mask (1 valid / 0 invalid), or null
//   (all rows valid).  y: (D,1) output, resized if mis-shaped.
//   y[j] = (1/num_valid) * sum_{k : mask[k]==1} X[k,j];
//   all-zero output if num_valid == 0.
void masked_mean_pool_forward(const Tensor& X, const float* d_mask,
                              Tensor& y);


// Backward of masked_mean_pool.
//   dY: (D,1) upstream.  mask: as forward (or null).  K: original row count.
//   dX: (K,D) overwritten — valid rows get dY/num_valid, invalid rows get 0;
//   all-zero if num_valid == 0.
void masked_mean_pool_backward(const Tensor& dY, const float* d_mask,
                               int K, Tensor& dX);


// 2x average-pool downsample, NCHW, stride 2 / kernel 2 / no padding; H and W
// must be even. Dispatched FP32/FP16 on X.dtype; FP32 math; Y resized + dtype-set.
//   X: (N, C*H*W).  Y: (N, C*H/2*W/2).
void downsample_avg_2x(const Tensor& X,
                       int N, int C, int H, int W,
                       Tensor& Y);


// Backward of downsample_avg_2x:
//   dX[n,c,2*i+a,2*j+b] = (1/4) * dY[n,c,i,j].
// N,C,H,W are the INPUT dims, H and W even. Dispatched FP32/FP16 on dY.dtype;
// FP32 accumulation; dX (N,C*H*W) overwritten, resized + dtype-set to dY.
void downsample_avg_2x_backward(const Tensor& dY,
                                int N, int C, int H, int W,
                                Tensor& dX);


// Adaptive 2D average pool, NCHW. Each output pixel averages the input
// region defined by PyTorch's adaptive-pool formula:
//   start_h(oh) = floor(oh     * H / H_out)
//   end_h(oh)   = ceil ((oh+1) * H / H_out)
// (and same for W). The region size varies across output pixels when the
// spatial dims don't divide evenly. Used by SegFormer / Mask2Former
// decoder-side aggregation and detection-head global pooling.
//   X: (N, C*H*W).  Y: (N, C*H_out*W_out), resized + dtype-set to X.
// FP32-only, on both the CPU and CUDA backends.
void adaptive_avg_pool2d_forward(const Tensor& X, int N, int C, int H, int W,
                                 int H_out, int W_out, Tensor& Y);


// Backward (adjoint) of adaptive_avg_pool2d: each input pixel accumulates
// (dY[oh, ow] / region_size(oh, ow)) for every output region that contained
// it. dX OVERWRITTEN (zeroed first, then scatter-added).
//   dY: (N, C*H_out*W_out).  dX: (N, C*H*W).
void adaptive_avg_pool2d_backward(const Tensor& dY, int N, int C, int H, int W,
                                  int H_out, int W_out, Tensor& dX);


// 2D max pool, NCHW. Standard CV pooling with kernel, stride, and padding.
// Padding pixels are treated as -inf so they never win the max (no special
// case downstream). Output spatial size:
//   H_out = (H + 2*pad_h - kH) / stride_h + 1
//   W_out = (W + 2*pad_w - kW) / stride_w + 1
// Returns Y AND a per-output INT32 index into the per-channel flat input
// spatial plane (Idx[n, c, oh, ow] == in_h * W + in_w of the winning pixel),
// so max_pool2d_backward can scatter dY without re-scanning the kernel.
//   X:   (N, C*H*W) FP32.
//   Y:   (N, C*H_out*W_out) FP32.
//   Idx: (N, C*H_out*W_out) INT32. -1 means "all kernel positions were
//        padding" (shouldn't happen with kH/kW <= H/W + 2*pad, but signals
//        a degenerate case cleanly).
void max_pool2d_forward(const Tensor& X, int N, int C, int H, int W,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w, Tensor& Y, Tensor& Idx);


// Backward of max_pool2d: each input pixel accumulates dY from every output
// pixel that selected it (the per-output Idx returned by the forward).
// dX OVERWRITTEN (zeroed, then scatter-added — overlapping kernels with
// stride < kernel size collide on the same input pixel and sum).
//   dY:  (N, C*H_out*W_out) FP32.
//   Idx: (N, C*H_out*W_out) INT32 from the forward call.
//   dX:  (N, C*H*W) FP32, resized + dtype-set.
// H_out and W_out are the post-pool spatial dims (caller passes them back —
// they're cheaper to forward than to re-derive).
void max_pool2d_backward(const Tensor& dY, const Tensor& Idx,
                         int N, int C, int H, int W, int H_out, int W_out,
                         Tensor& dX);

}  // namespace brotensor
