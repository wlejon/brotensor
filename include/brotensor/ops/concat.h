#pragma once

// brotensor ops/concat.h — Concat / split (rows, batched, NCHW channels) + device copy.

#include "../tensor.h"
#include <vector>
#include <cstdint>

namespace brotensor {


// Concatenate flat tensors end-to-end. Each part is treated as a flat buffer.
// out resized to (total,1), total = sum of part sizes; parts laid in order.
void concat_rows(const std::vector<const Tensor*>& parts,
                 Tensor& out);


// Inverse of concat_rows: copy disjoint segments of `in` into the flat buffers
// of `parts` (each *overwritten*, not accumulated). Part sizes must match the
// concat call; segments are laid end-to-end from offset 0 in `in`.
void split_rows(const Tensor& in,
                const std::vector<Tensor*>& parts);


// Batched column-block concat. Each part is (B, d_i) for a shared B; out
// becomes (B, sum_i d_i) with parts as per-row column blocks:
//   out[b, off_i + j] = parts[i][b, j].
void concat_batched_rows(const std::vector<const Tensor*>& parts,
                         Tensor& out);


// Channel-axis concat of NCHW tensors. Part i is (N, C_i*H*W) flat NCHW; out
// becomes (N, sum_i C_i*H*W) with channel blocks regrouped per sample:
//   out[n, (off_i+c)*H*W + h*W + w] = parts[i][n, c*H*W + h*W + w],
//   off_i = sum_{j<i} C_j.
// The correct U-Net skip-merge concat for N >= 1 (a flat concat_rows would
// interleave samples for N > 1). Dtype-dispatched (FP16/FP32); all parts share
// dtype. C_per_part.size() must equal parts.size().
void concat_nchw_channels(const std::vector<const Tensor*>& parts,
                          int N, int H, int W,
                          const std::vector<int>& C_per_part,
                          Tensor& out);


// Inverse of concat_nchw_channels: copy disjoint channel-axis slices of dY
// into per-source buffers — each parts[i] *overwritten* with channels
// [off_i, off_i+C_per_part[i]) of dY, off_i = sum_{j<i} C_per_part[j].
// Dtype-dispatched (FP32/FP16); parts resized AND dtype-set to match dY.
// C_per_part.size() must equal parts.size(); dY.cols must be
// N * sum(C_per_part) * H * W.
void concat_nchw_channels_backward(const Tensor& dY,
                                   int N, int H, int W,
                                   const std::vector<int>& C_per_part,
                                   const std::vector<Tensor*>& parts);


// Device-to-device chunk copy: copies `n` floats from src.data+src_off into
// dst.data+dst_off. Both treated as flat float buffers regardless of (rows,
// cols). Async on the default stream.
void copy_d2d(const Tensor& src, int src_off,
              Tensor& dst,       int dst_off,
              int n);


// Strided device-to-device row-block copy: copies `height` rows of `width`
// floats; row r reads src.data + src_off + r*src_pitch and writes
// dst.data + dst_off + r*dst_pitch (offsets/pitches in elements, pitch >=
// width). Both tensors treated as flat float buffers regardless of (rows,
// cols). One call replaces a host loop of `height` copy_d2d calls — e.g.
// padding/unpadding the W dim of an NCHW activation across all (n, c, h)
// rows. Async on the default stream.
void copy_d2d_strided(const Tensor& src, int src_off, int src_pitch,
                      Tensor& dst,       int dst_off, int dst_pitch,
                      int width, int height);

}  // namespace brotensor
