#pragma once

// Tiled simdgroup-matrix FP16 implicit-GEMM conv2d forward (Metal mirror of
// src/cuda/conv2d_wmma.cu). Internal — used by src/metal/conv2d.mm to attempt
// the fast path before falling back to the naive direct conv.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace brotensor {
namespace conv2d_wmma_internal {

// Returns true if the call was consumed by the fast path; returns false for
// shapes outside the SD set (caller falls back to naive direct conv).
bool launch_conv2d_implicit_gemm_simdgroup(
        id<MTLBuffer> X, NSUInteger ofs_X,
        id<MTLBuffer> Wt, NSUInteger ofs_Wt,
        id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
        id<MTLBuffer> Y, NSUInteger ofs_Y,
        int N, int C_in, int H, int W,
        int C_out, int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int H_out, int W_out);

} // namespace conv2d_wmma_internal
} // namespace brotensor
