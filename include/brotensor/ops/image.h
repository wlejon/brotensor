#pragma once

// brotensor ops/image.h — Image preprocessing: per-channel normalize, uint8 HWC -> FP32 NCHW.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// ─── Image preprocessing helpers ───────────────────────────────────────────
//
// Tiny ops every vision model wants. Live in brotensor so brogameagent /
// brodiffusion / future vision projects share one canonical path instead of
// re-implementing them in each caller.

// Per-channel (X - mean[c]) / std[c] on NCHW. The ImageNet / CLIP / SAM
// preprocess. X, Y: (N, C*H*W). mean, std: (C,1). std[c] must be non-zero.
void image_normalize(const Tensor& X,
                     const Tensor& mean, const Tensor& std_,
                     int N, int C, int H, int W,
                     Tensor& Y);


// Convert a packed uint8 HWC image buffer (decoder output) into a FP32 NCHW
// tensor, applying a single scale+bias pass:
//     Y[n,c,h,w] = src[n*H*W*C + (h*W+w)*C + c] * scale + bias.
// Covers the standard scaling conventions:
//     [0,255] -> [0,1]   : scale = 1.0f / 255.0f,        bias = 0.0f
//     [0,255] -> [-1,1]  : scale = 2.0f / 255.0f,        bias = -1.0f
// Y resized to (N, C*H*W) FP32. src is a raw host pointer of length
// N*H*W*C bytes (CPU backend) — image bytes essentially always originate
// host-side, and there's no UINT8 dtype to wrap them in. GPU backends are
// expected to either take a device pointer or upload internally; the
// signature here matches `embedding_lookup_forward(const int32_t*)`.
void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  Tensor& Y);

}  // namespace brotensor
