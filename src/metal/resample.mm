#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_upsample_nearest_2x(device const half* X [[buffer(0)]],
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

kernel void k_upsample_bilinear_2x(device const half* X [[buffer(0)]],
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

    // align_corners=False, scale=2: src = (dst + 0.5)/2 - 0.5
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

kernel void k_downsample_avg_2x(device const half* X [[buffer(0)]],
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
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_up_nearest,  @"k_upsample_nearest_2x")
DEF_PSO(pso_up_bilinear, @"k_upsample_bilinear_2x")
DEF_PSO(pso_down_avg,    @"k_downsample_avg_2x")
#undef DEF_PSO

void launch_resample(id<MTLComputePipelineState> pso,
                     const GpuTensor& X, GpuTensor& Y,
                     int N, int C, int H, int W,
                     int H_out, int W_out) {
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, cols, Dtype::FP16);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);
    if (total == 0) return;
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Hu = static_cast<uint32_t>(H);
    const uint32_t Wu = static_cast<uint32_t>(W);
    const uint32_t Ho = static_cast<uint32_t>(H_out);
    const uint32_t Wo = static_cast<uint32_t>(W_out);
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
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void upsample_nearest_2x_gpu(const GpuTensor& X, int N, int C, int H, int W,
                             GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("upsample_nearest_2x_gpu: X must be FP16");
    }
    launch_resample(pso_up_nearest(), X, Y, N, C, H, W, 2 * H, 2 * W);
}

void upsample_bilinear_2x_gpu(const GpuTensor& X, int N, int C, int H, int W,
                              GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("upsample_bilinear_2x_gpu: X must be FP16");
    }
    launch_resample(pso_up_bilinear(), X, Y, N, C, H, W, 2 * H, 2 * W);
}

void downsample_avg_2x_gpu(const GpuTensor& X, int N, int C, int H, int W,
                           GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("downsample_avg_2x_gpu: X must be FP16");
    }
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x_gpu: H and W must be even");
    }
    launch_resample(pso_down_avg(), X, Y, N, C, H, W, H / 2, W / 2);
}

} // namespace brotensor
