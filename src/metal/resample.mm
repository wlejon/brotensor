#include <brotensor/runtime.h>

#include <cstring>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// ─── Forward kernels ───────────────────────────────────────────────────────

kernel void k_upsample_nearest_2x_fp16(device const half* X [[buffer(0)]],
                                       device half*       Y [[buffer(1)]],
                                       constant uint& N      [[buffer(2)]],
                                       constant uint& C      [[buffer(3)]],
                                       constant uint& H      [[buffer(4)]],
                                       constant uint& W      [[buffer(5)]],
                                       constant uint& H_out  [[buffer(6)]],
                                       constant uint& W_out  [[buffer(7)]],
                                       constant uint& total  [[buffer(8)]],
                                       uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    uint ih = oh / 2;
    uint iw = ow / 2;
    uint in_idx = ((n * C + c) * H + ih) * W + iw;
    Y[idx] = X[in_idx];
}

kernel void k_upsample_nearest_2x_fp32(device const float* X [[buffer(0)]],
                                       device float*       Y [[buffer(1)]],
                                       constant uint& N      [[buffer(2)]],
                                       constant uint& C      [[buffer(3)]],
                                       constant uint& H      [[buffer(4)]],
                                       constant uint& W      [[buffer(5)]],
                                       constant uint& H_out  [[buffer(6)]],
                                       constant uint& W_out  [[buffer(7)]],
                                       constant uint& total  [[buffer(8)]],
                                       uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    uint ih = oh / 2;
    uint iw = ow / 2;
    uint in_idx = ((n * C + c) * H + ih) * W + iw;
    Y[idx] = X[in_idx];
}

kernel void k_upsample_bilinear_2x_fp16(device const half* X [[buffer(0)]],
                                        device half*       Y [[buffer(1)]],
                                        constant uint& N      [[buffer(2)]],
                                        constant uint& C      [[buffer(3)]],
                                        constant uint& H      [[buffer(4)]],
                                        constant uint& W      [[buffer(5)]],
                                        constant uint& H_out  [[buffer(6)]],
                                        constant uint& W_out  [[buffer(7)]],
                                        constant uint& total  [[buffer(8)]],
                                        uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    float src_y = (float(oh) + 0.5f) * 0.5f - 0.5f;
    float src_x = (float(ow) + 0.5f) * 0.5f - 0.5f;
    int y0 = int(floor(src_y));
    int x0 = int(floor(src_x));
    float fy = src_y - float(y0);
    float fx = src_x - float(x0);
    int Hi = int(H); int Wi = int(W);
    int y0c = clamp(y0,     0, Hi - 1);
    int x0c = clamp(x0,     0, Wi - 1);
    int y1c = clamp(y0 + 1, 0, Hi - 1);
    int x1c = clamp(x0 + 1, 0, Wi - 1);
    uint base = (n * C + c) * H;
    float v00 = float(X[(base + uint(y0c)) * W + uint(x0c)]);
    float v01 = float(X[(base + uint(y0c)) * W + uint(x1c)]);
    float v10 = float(X[(base + uint(y1c)) * W + uint(x0c)]);
    float v11 = float(X[(base + uint(y1c)) * W + uint(x1c)]);
    float top = v00 + (v01 - v00) * fx;
    float bot = v10 + (v11 - v10) * fx;
    float v   = top + (bot - top) * fy;
    Y[idx] = half(v);
}

kernel void k_upsample_bilinear_2x_fp32(device const float* X [[buffer(0)]],
                                        device float*       Y [[buffer(1)]],
                                        constant uint& N      [[buffer(2)]],
                                        constant uint& C      [[buffer(3)]],
                                        constant uint& H      [[buffer(4)]],
                                        constant uint& W      [[buffer(5)]],
                                        constant uint& H_out  [[buffer(6)]],
                                        constant uint& W_out  [[buffer(7)]],
                                        constant uint& total  [[buffer(8)]],
                                        uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    float src_y = (float(oh) + 0.5f) * 0.5f - 0.5f;
    float src_x = (float(ow) + 0.5f) * 0.5f - 0.5f;
    int y0 = int(floor(src_y));
    int x0 = int(floor(src_x));
    float fy = src_y - float(y0);
    float fx = src_x - float(x0);
    int Hi = int(H); int Wi = int(W);
    int y0c = clamp(y0,     0, Hi - 1);
    int x0c = clamp(x0,     0, Wi - 1);
    int y1c = clamp(y0 + 1, 0, Hi - 1);
    int x1c = clamp(x0 + 1, 0, Wi - 1);
    uint base = (n * C + c) * H;
    float v00 = X[(base + uint(y0c)) * W + uint(x0c)];
    float v01 = X[(base + uint(y0c)) * W + uint(x1c)];
    float v10 = X[(base + uint(y1c)) * W + uint(x0c)];
    float v11 = X[(base + uint(y1c)) * W + uint(x1c)];
    float top = v00 + (v01 - v00) * fx;
    float bot = v10 + (v11 - v10) * fx;
    Y[idx] = top + (bot - top) * fy;
}

kernel void k_downsample_avg_2x_fp16(device const half* X [[buffer(0)]],
                                     device half*       Y [[buffer(1)]],
                                     constant uint& N      [[buffer(2)]],
                                     constant uint& C      [[buffer(3)]],
                                     constant uint& H      [[buffer(4)]],
                                     constant uint& W      [[buffer(5)]],
                                     constant uint& H_out  [[buffer(6)]],
                                     constant uint& W_out  [[buffer(7)]],
                                     constant uint& total  [[buffer(8)]],
                                     uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    uint ih = oh * 2;
    uint iw = ow * 2;
    uint base = ((n * C + c) * H + ih) * W + iw;
    float v00 = float(X[base]);
    float v01 = float(X[base + 1]);
    float v10 = float(X[base + W]);
    float v11 = float(X[base + W + 1]);
    Y[idx] = half(0.25f * (v00 + v01 + v10 + v11));
}

kernel void k_downsample_avg_2x_fp32(device const float* X [[buffer(0)]],
                                     device float*       Y [[buffer(1)]],
                                     constant uint& N      [[buffer(2)]],
                                     constant uint& C      [[buffer(3)]],
                                     constant uint& H      [[buffer(4)]],
                                     constant uint& W      [[buffer(5)]],
                                     constant uint& H_out  [[buffer(6)]],
                                     constant uint& W_out  [[buffer(7)]],
                                     constant uint& total  [[buffer(8)]],
                                     uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    uint ih = oh * 2;
    uint iw = ow * 2;
    uint base = ((n * C + c) * H + ih) * W + iw;
    float v00 = X[base];
    float v01 = X[base + 1];
    float v10 = X[base + W];
    float v11 = X[base + W + 1];
    Y[idx] = 0.25f * (v00 + v01 + v10 + v11);
}

// ─── Backward kernels ──────────────────────────────────────────────────────

kernel void k_upsample_nearest_2x_backward_fp16(
        device const half* dY [[buffer(0)]],
        device half*       dX [[buffer(1)]],
        constant uint& N       [[buffer(2)]],
        constant uint& C       [[buffer(3)]],
        constant uint& H       [[buffer(4)]],
        constant uint& W       [[buffer(5)]],
        constant uint& H_out   [[buffer(6)]],
        constant uint& W_out   [[buffer(7)]],
        constant uint& total_in [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_in) return;
    uint iw = idx % W;
    uint t  = idx / W;
    uint ih = t % H;
    t /= H;
    uint c  = t % C;
    uint n  = t / C;
    uint base = (n * C + c) * H_out;
    uint oh0 = 2 * ih;
    uint ow0 = 2 * iw;
    float v00 = float(dY[(base + oh0    ) * W_out + ow0    ]);
    float v01 = float(dY[(base + oh0    ) * W_out + ow0 + 1]);
    float v10 = float(dY[(base + oh0 + 1) * W_out + ow0    ]);
    float v11 = float(dY[(base + oh0 + 1) * W_out + ow0 + 1]);
    dX[idx] = half(v00 + v01 + v10 + v11);
}

kernel void k_upsample_nearest_2x_backward_fp32(
        device const float* dY [[buffer(0)]],
        device float*       dX [[buffer(1)]],
        constant uint& N       [[buffer(2)]],
        constant uint& C       [[buffer(3)]],
        constant uint& H       [[buffer(4)]],
        constant uint& W       [[buffer(5)]],
        constant uint& H_out   [[buffer(6)]],
        constant uint& W_out   [[buffer(7)]],
        constant uint& total_in [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_in) return;
    uint iw = idx % W;
    uint t  = idx / W;
    uint ih = t % H;
    t /= H;
    uint c  = t % C;
    uint n  = t / C;
    uint base = (n * C + c) * H_out;
    uint oh0 = 2 * ih;
    uint ow0 = 2 * iw;
    float v00 = dY[(base + oh0    ) * W_out + ow0    ];
    float v01 = dY[(base + oh0    ) * W_out + ow0 + 1];
    float v10 = dY[(base + oh0 + 1) * W_out + ow0    ];
    float v11 = dY[(base + oh0 + 1) * W_out + ow0 + 1];
    dX[idx] = v00 + v01 + v10 + v11;
}

// Bilinear backward: one thread per OUTPUT pixel, atomicAdd into FP32 dX
// scratch. FP32 path passes the destination dX buffer directly; FP16 path
// passes an FP32 scratch and folds back via copy_fp32_to_fp16.
kernel void k_upsample_bilinear_2x_backward_scatter_fp32(
        device const float* dY              [[buffer(0)]],
        device atomic_float* dX_acc         [[buffer(1)]],
        constant uint& N        [[buffer(2)]],
        constant uint& C        [[buffer(3)]],
        constant uint& H        [[buffer(4)]],
        constant uint& W        [[buffer(5)]],
        constant uint& H_out    [[buffer(6)]],
        constant uint& W_out    [[buffer(7)]],
        constant uint& total_out [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_out) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    float src_y = (float(oh) + 0.5f) * 0.5f - 0.5f;
    float src_x = (float(ow) + 0.5f) * 0.5f - 0.5f;
    int y0 = int(floor(src_y));
    int x0 = int(floor(src_x));
    float fy = src_y - float(y0);
    float fx = src_x - float(x0);
    int Hi = int(H); int Wi = int(W);
    int y0c = clamp(y0,     0, Hi - 1);
    int x0c = clamp(x0,     0, Wi - 1);
    int y1c = clamp(y0 + 1, 0, Hi - 1);
    int x1c = clamp(x0 + 1, 0, Wi - 1);
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w01 = (1.0f - fy) * fx;
    float w10 = fy * (1.0f - fx);
    float w11 = fy * fx;
    float g = dY[idx];
    uint base = (n * C + c) * H;
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y0c)) * W + uint(x0c)], w00 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y0c)) * W + uint(x1c)], w01 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y1c)) * W + uint(x0c)], w10 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y1c)) * W + uint(x1c)], w11 * g, memory_order_relaxed);
}

kernel void k_upsample_bilinear_2x_backward_scatter_fp16(
        device const half* dY               [[buffer(0)]],
        device atomic_float* dX_acc         [[buffer(1)]],
        constant uint& N        [[buffer(2)]],
        constant uint& C        [[buffer(3)]],
        constant uint& H        [[buffer(4)]],
        constant uint& W        [[buffer(5)]],
        constant uint& H_out    [[buffer(6)]],
        constant uint& W_out    [[buffer(7)]],
        constant uint& total_out [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_out) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    float src_y = (float(oh) + 0.5f) * 0.5f - 0.5f;
    float src_x = (float(ow) + 0.5f) * 0.5f - 0.5f;
    int y0 = int(floor(src_y));
    int x0 = int(floor(src_x));
    float fy = src_y - float(y0);
    float fx = src_x - float(x0);
    int Hi = int(H); int Wi = int(W);
    int y0c = clamp(y0,     0, Hi - 1);
    int x0c = clamp(x0,     0, Wi - 1);
    int y1c = clamp(y0 + 1, 0, Hi - 1);
    int x1c = clamp(x0 + 1, 0, Wi - 1);
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w01 = (1.0f - fy) * fx;
    float w10 = fy * (1.0f - fx);
    float w11 = fy * fx;
    float g = float(dY[idx]);
    uint base = (n * C + c) * H;
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y0c)) * W + uint(x0c)], w00 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y0c)) * W + uint(x1c)], w01 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y1c)) * W + uint(x0c)], w10 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y1c)) * W + uint(x1c)], w11 * g, memory_order_relaxed);
}

kernel void k_copy_fp32_to_fp16(device const float* src [[buffer(0)]],
                                device half*        dst [[buffer(1)]],
                                constant uint& n        [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = half(src[gid]);
}

kernel void k_downsample_avg_2x_backward_fp16(
        device const half* dY [[buffer(0)]],
        device half*       dX [[buffer(1)]],
        constant uint& N       [[buffer(2)]],
        constant uint& C       [[buffer(3)]],
        constant uint& H       [[buffer(4)]],
        constant uint& W       [[buffer(5)]],
        constant uint& H_out   [[buffer(6)]],
        constant uint& W_out   [[buffer(7)]],
        constant uint& total_in [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_in) return;
    uint iw = idx % W;
    uint t  = idx / W;
    uint ih = t % H;
    t /= H;
    uint c  = t % C;
    uint n  = t / C;
    uint oh = ih / 2;
    uint ow = iw / 2;
    uint out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
    dX[idx] = half(0.25f * float(dY[out_idx]));
}

kernel void k_downsample_avg_2x_backward_fp32(
        device const float* dY [[buffer(0)]],
        device float*       dX [[buffer(1)]],
        constant uint& N       [[buffer(2)]],
        constant uint& C       [[buffer(3)]],
        constant uint& H       [[buffer(4)]],
        constant uint& W       [[buffer(5)]],
        constant uint& H_out   [[buffer(6)]],
        constant uint& W_out   [[buffer(7)]],
        constant uint& total_in [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_in) return;
    uint iw = idx % W;
    uint t  = idx / W;
    uint ih = t % H;
    t /= H;
    uint c  = t % C;
    uint n  = t / C;
    uint oh = ih / 2;
    uint ow = iw / 2;
    uint out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
    dX[idx] = 0.25f * dY[out_idx];
}

// ─── BF16 variants ────────────────────────────────────────────────────────

kernel void k_upsample_nearest_2x_bf16(device const bfloat* X [[buffer(0)]],
                                       device bfloat*       Y [[buffer(1)]],
                                       constant uint& N        [[buffer(2)]],
                                       constant uint& C        [[buffer(3)]],
                                       constant uint& H        [[buffer(4)]],
                                       constant uint& W        [[buffer(5)]],
                                       constant uint& H_out    [[buffer(6)]],
                                       constant uint& W_out    [[buffer(7)]],
                                       constant uint& total    [[buffer(8)]],
                                       uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    uint ih = oh / 2;
    uint iw = ow / 2;
    uint in_idx = ((n * C + c) * H + ih) * W + iw;
    Y[idx] = X[in_idx];
}

kernel void k_upsample_bilinear_2x_bf16(device const bfloat* X [[buffer(0)]],
                                        device bfloat*       Y [[buffer(1)]],
                                        constant uint& N        [[buffer(2)]],
                                        constant uint& C        [[buffer(3)]],
                                        constant uint& H        [[buffer(4)]],
                                        constant uint& W        [[buffer(5)]],
                                        constant uint& H_out    [[buffer(6)]],
                                        constant uint& W_out    [[buffer(7)]],
                                        constant uint& total    [[buffer(8)]],
                                        uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    float src_y = (float(oh) + 0.5f) * 0.5f - 0.5f;
    float src_x = (float(ow) + 0.5f) * 0.5f - 0.5f;
    int y0 = int(floor(src_y));
    int x0 = int(floor(src_x));
    float fy = src_y - float(y0);
    float fx = src_x - float(x0);
    int Hi = int(H); int Wi = int(W);
    int y0c = clamp(y0,     0, Hi - 1);
    int x0c = clamp(x0,     0, Wi - 1);
    int y1c = clamp(y0 + 1, 0, Hi - 1);
    int x1c = clamp(x0 + 1, 0, Wi - 1);
    uint base = (n * C + c) * H;
    float v00 = float(X[(base + uint(y0c)) * W + uint(x0c)]);
    float v01 = float(X[(base + uint(y0c)) * W + uint(x1c)]);
    float v10 = float(X[(base + uint(y1c)) * W + uint(x0c)]);
    float v11 = float(X[(base + uint(y1c)) * W + uint(x1c)]);
    float top = v00 + (v01 - v00) * fx;
    float bot = v10 + (v11 - v10) * fx;
    float v   = top + (bot - top) * fy;
    Y[idx] = bfloat(v);
}

kernel void k_downsample_avg_2x_bf16(device const bfloat* X [[buffer(0)]],
                                     device bfloat*       Y [[buffer(1)]],
                                     constant uint& N        [[buffer(2)]],
                                     constant uint& C        [[buffer(3)]],
                                     constant uint& H        [[buffer(4)]],
                                     constant uint& W        [[buffer(5)]],
                                     constant uint& H_out    [[buffer(6)]],
                                     constant uint& W_out    [[buffer(7)]],
                                     constant uint& total    [[buffer(8)]],
                                     uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    uint ih = oh * 2;
    uint iw = ow * 2;
    uint base = ((n * C + c) * H + ih) * W + iw;
    float v00 = float(X[base]);
    float v01 = float(X[base + 1]);
    float v10 = float(X[base + W]);
    float v11 = float(X[base + W + 1]);
    Y[idx] = bfloat(0.25f * (v00 + v01 + v10 + v11));
}

kernel void k_upsample_nearest_2x_backward_bf16(
        device const bfloat* dY [[buffer(0)]],
        device bfloat*       dX [[buffer(1)]],
        constant uint& N         [[buffer(2)]],
        constant uint& C         [[buffer(3)]],
        constant uint& H         [[buffer(4)]],
        constant uint& W         [[buffer(5)]],
        constant uint& H_out     [[buffer(6)]],
        constant uint& W_out     [[buffer(7)]],
        constant uint& total_in  [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_in) return;
    uint iw = idx % W;
    uint t  = idx / W;
    uint ih = t % H;
    t /= H;
    uint c  = t % C;
    uint n  = t / C;
    uint base = (n * C + c) * H_out;
    uint oh0 = 2 * ih;
    uint ow0 = 2 * iw;
    float v00 = float(dY[(base + oh0    ) * W_out + ow0    ]);
    float v01 = float(dY[(base + oh0    ) * W_out + ow0 + 1]);
    float v10 = float(dY[(base + oh0 + 1) * W_out + ow0    ]);
    float v11 = float(dY[(base + oh0 + 1) * W_out + ow0 + 1]);
    dX[idx] = bfloat(v00 + v01 + v10 + v11);
}

kernel void k_upsample_bilinear_2x_backward_scatter_bf16(
        device const bfloat* dY             [[buffer(0)]],
        device atomic_float* dX_acc         [[buffer(1)]],
        constant uint& N        [[buffer(2)]],
        constant uint& C        [[buffer(3)]],
        constant uint& H        [[buffer(4)]],
        constant uint& W        [[buffer(5)]],
        constant uint& H_out    [[buffer(6)]],
        constant uint& W_out    [[buffer(7)]],
        constant uint& total_out [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_out) return;
    uint ow = idx % W_out;
    uint t  = idx / W_out;
    uint oh = t % H_out;
    t /= H_out;
    uint c  = t % C;
    uint n  = t / C;
    float src_y = (float(oh) + 0.5f) * 0.5f - 0.5f;
    float src_x = (float(ow) + 0.5f) * 0.5f - 0.5f;
    int y0 = int(floor(src_y));
    int x0 = int(floor(src_x));
    float fy = src_y - float(y0);
    float fx = src_x - float(x0);
    int Hi = int(H); int Wi = int(W);
    int y0c = clamp(y0,     0, Hi - 1);
    int x0c = clamp(x0,     0, Wi - 1);
    int y1c = clamp(y0 + 1, 0, Hi - 1);
    int x1c = clamp(x0 + 1, 0, Wi - 1);
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w01 = (1.0f - fy) * fx;
    float w10 = fy * (1.0f - fx);
    float w11 = fy * fx;
    float g = float(dY[idx]);
    uint base = (n * C + c) * H;
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y0c)) * W + uint(x0c)], w00 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y0c)) * W + uint(x1c)], w01 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y1c)) * W + uint(x0c)], w10 * g, memory_order_relaxed);
    atomic_fetch_add_explicit(&dX_acc[(base + uint(y1c)) * W + uint(x1c)], w11 * g, memory_order_relaxed);
}

kernel void k_copy_fp32_to_bf16(device const float* src [[buffer(0)]],
                                device bfloat*      dst [[buffer(1)]],
                                constant uint& n        [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = bfloat(src[gid]);
}

kernel void k_downsample_avg_2x_backward_bf16(
        device const bfloat* dY [[buffer(0)]],
        device bfloat*       dX [[buffer(1)]],
        constant uint& N         [[buffer(2)]],
        constant uint& C         [[buffer(3)]],
        constant uint& H         [[buffer(4)]],
        constant uint& W         [[buffer(5)]],
        constant uint& H_out     [[buffer(6)]],
        constant uint& W_out     [[buffer(7)]],
        constant uint& total_in  [[buffer(8)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= total_in) return;
    uint iw = idx % W;
    uint t  = idx / W;
    uint ih = t % H;
    t /= H;
    uint c  = t % C;
    uint n  = t / C;
    uint oh = ih / 2;
    uint ow = iw / 2;
    uint out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
    dX[idx] = bfloat(0.25f * float(dY[out_idx]));
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_up_nearest_fp16,    @"k_upsample_nearest_2x_fp16")
DEF_PSO(pso_up_nearest_fp32,    @"k_upsample_nearest_2x_fp32")
DEF_PSO(pso_up_nearest_bf16,    @"k_upsample_nearest_2x_bf16")
DEF_PSO(pso_up_bilinear_fp16,   @"k_upsample_bilinear_2x_fp16")
DEF_PSO(pso_up_bilinear_fp32,   @"k_upsample_bilinear_2x_fp32")
DEF_PSO(pso_up_bilinear_bf16,   @"k_upsample_bilinear_2x_bf16")
DEF_PSO(pso_down_avg_fp16,      @"k_downsample_avg_2x_fp16")
DEF_PSO(pso_down_avg_fp32,      @"k_downsample_avg_2x_fp32")
DEF_PSO(pso_down_avg_bf16,      @"k_downsample_avg_2x_bf16")
DEF_PSO(pso_up_nearest_bwd_fp16,  @"k_upsample_nearest_2x_backward_fp16")
DEF_PSO(pso_up_nearest_bwd_fp32,  @"k_upsample_nearest_2x_backward_fp32")
DEF_PSO(pso_up_nearest_bwd_bf16,  @"k_upsample_nearest_2x_backward_bf16")
DEF_PSO(pso_up_bilinear_bwd_fp32, @"k_upsample_bilinear_2x_backward_scatter_fp32")
DEF_PSO(pso_up_bilinear_bwd_fp16, @"k_upsample_bilinear_2x_backward_scatter_fp16")
DEF_PSO(pso_up_bilinear_bwd_bf16, @"k_upsample_bilinear_2x_backward_scatter_bf16")
DEF_PSO(pso_copy_fp32_to_fp16,    @"k_copy_fp32_to_fp16")
DEF_PSO(pso_copy_fp32_to_bf16,    @"k_copy_fp32_to_bf16")
DEF_PSO(pso_down_avg_bwd_fp16,    @"k_downsample_avg_2x_backward_fp16")
DEF_PSO(pso_down_avg_bwd_fp32,    @"k_downsample_avg_2x_backward_fp32")
DEF_PSO(pso_down_avg_bwd_bf16,    @"k_downsample_avg_2x_backward_bf16")
#undef DEF_PSO

inline void check_dtype_fp(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != Dtype::FP16 && t.dtype != Dtype::FP32 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": " + name + " must be FP16, BF16, or FP32");
    }
}

void launch_resample_fwd(id<MTLComputePipelineState> pso_fp16,
                         id<MTLComputePipelineState> pso_fp32,
                         id<MTLComputePipelineState> pso_bf16,
                         const Tensor& X, Tensor& Y,
                         int N, int C, int H, int W,
                         int H_out, int W_out) {
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);
    if (total == 0) return;
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Hu = static_cast<uint32_t>(H);
    const uint32_t Wu = static_cast<uint32_t>(W);
    const uint32_t Ho = static_cast<uint32_t>(H_out);
    const uint32_t Wo = static_cast<uint32_t>(W_out);
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_fp16
      : (X.dtype == Dtype::BF16) ? pso_bf16
      : pso_fp32;
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Wu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Ho length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Wo length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:8];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// Launch a simple one-thread-per-input-pixel backward kernel (nearest, avg).
void launch_resample_bwd_simple(id<MTLComputePipelineState> pso_fp16,
                                id<MTLComputePipelineState> pso_fp32,
                                id<MTLComputePipelineState> pso_bf16,
                                const Tensor& dY, Tensor& dX,
                                int N, int C, int H, int W,
                                int H_out, int W_out) {
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    const uint32_t total_in = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);
    if (total_in == 0) return;
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Hu = static_cast<uint32_t>(H);
    const uint32_t Wu = static_cast<uint32_t>(W);
    const uint32_t Ho = static_cast<uint32_t>(H_out);
    const uint32_t Wo = static_cast<uint32_t>(W_out);
    id<MTLComputePipelineState> pso =
        (dY.dtype == Dtype::FP16) ? pso_fp16
      : (dY.dtype == Dtype::BF16) ? pso_bf16
      : pso_fp32;
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:bdx offset:odx atIndex:1];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Wu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Ho length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Wo length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&total_in length:sizeof(uint32_t) atIndex:8];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total_in, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

// ─── Forward ───────────────────────────────────────────────────────────────

void upsample_nearest_2x(const Tensor& X, int N, int C, int H, int W,
                         Tensor& Y) {
    check_dtype_fp(X, "upsample_nearest_2x", "X");
    launch_resample_fwd(pso_up_nearest_fp16(), pso_up_nearest_fp32(),
                        pso_up_nearest_bf16(),
                        X, Y, N, C, H, W, 2 * H, 2 * W);
}

void upsample_bilinear_2x(const Tensor& X, int N, int C, int H, int W,
                          Tensor& Y) {
    check_dtype_fp(X, "upsample_bilinear_2x", "X");
    launch_resample_fwd(pso_up_bilinear_fp16(), pso_up_bilinear_fp32(),
                        pso_up_bilinear_bf16(),
                        X, Y, N, C, H, W, 2 * H, 2 * W);
}

void downsample_avg_2x(const Tensor& X, int N, int C, int H, int W,
                       Tensor& Y) {
    check_dtype_fp(X, "downsample_avg_2x", "X");
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x: H and W must be even");
    }
    launch_resample_fwd(pso_down_avg_fp16(), pso_down_avg_fp32(),
                        pso_down_avg_bf16(),
                        X, Y, N, C, H, W, H / 2, W / 2);
}

// ─── Backward ──────────────────────────────────────────────────────────────

void upsample_nearest_2x_backward(const Tensor& dY,
                                  int N, int C, int H, int W,
                                  Tensor& dX) {
    check_dtype_fp(dY, "upsample_nearest_2x_backward", "dY");
    launch_resample_bwd_simple(pso_up_nearest_bwd_fp16(),
                               pso_up_nearest_bwd_fp32(),
                               pso_up_nearest_bwd_bf16(),
                               dY, dX, N, C, H, W, 2 * H, 2 * W);
}

void downsample_avg_2x_backward(const Tensor& dY,
                                int N, int C, int H, int W,
                                Tensor& dX) {
    check_dtype_fp(dY, "downsample_avg_2x_backward", "dY");
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x_backward: H and W must be even");
    }
    launch_resample_bwd_simple(pso_down_avg_bwd_fp16(),
                               pso_down_avg_bwd_fp32(),
                               pso_down_avg_bwd_bf16(),
                               dY, dX, N, C, H, W, H / 2, W / 2);
}

void upsample_bilinear_2x_backward(const Tensor& dY,
                                   int N, int C, int H, int W,
                                   Tensor& dX) {
    check_dtype_fp(dY, "upsample_bilinear_2x_backward", "dY");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols_in = C * H * W;
    const int cols_out = C * H_out * W_out;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    const uint32_t total_in = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);
    const uint32_t total_out = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_out);
    if (total_out == 0) return;
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Hu = static_cast<uint32_t>(H);
    const uint32_t Wu = static_cast<uint32_t>(W);
    const uint32_t Ho = static_cast<uint32_t>(H_out);
    const uint32_t Wo = static_cast<uint32_t>(W_out);

    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);

    @autoreleasepool {
        id<MTLBuffer> scratch = nil;
        id<MTLBuffer> scatter_dst = nil;
        NSUInteger scatter_off = 0;
        if (dY.dtype == Dtype::FP32) {
            // Scatter directly into dX (must be zero-init).
            scatter_dst = bdx;
            scatter_off = odx;
            std::memset(static_cast<char*>([bdx contents]) + odx, 0,
                        total_in * sizeof(float));
        } else {
            scratch = [metal_impl::device()
                newBufferWithLength:total_in * sizeof(float)
                            options:MTLResourceStorageModeShared];
            std::memset([scratch contents], 0, total_in * sizeof(float));
            scatter_dst = scratch;
            scatter_off = 0;
        }

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso_scatter =
            (dY.dtype == Dtype::FP16) ? pso_up_bilinear_bwd_fp16()
          : (dY.dtype == Dtype::BF16) ? pso_up_bilinear_bwd_bf16()
          : pso_up_bilinear_bwd_fp32();
        [enc setComputePipelineState:pso_scatter];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:scatter_dst offset:scatter_off atIndex:1];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Wu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Ho length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Wo length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&total_out length:sizeof(uint32_t) atIndex:8];
        NSUInteger tg = [pso_scatter maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total_out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        if (dY.dtype == Dtype::FP16 || dY.dtype == Dtype::BF16) {
            // Fold FP32 scratch back into low-precision dX (overwrite).
            id<MTLComputePipelineState> pso_copy = (dY.dtype == Dtype::BF16)
                ? pso_copy_fp32_to_bf16() : pso_copy_fp32_to_fp16();
            [enc setComputePipelineState:pso_copy];
            [enc setBuffer:scratch offset:0 atIndex:0];
            [enc setBuffer:bdx offset:odx atIndex:1];
            [enc setBytes:&total_in length:sizeof(uint32_t) atIndex:2];
            NSUInteger tg2 = [pso_copy maxTotalThreadsPerThreadgroup];
            if (tg2 > 256) tg2 = 256;
            [enc dispatchThreads:MTLSizeMake(total_in, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        }

        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
