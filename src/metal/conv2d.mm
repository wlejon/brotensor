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

struct ConvParams {
    uint N, C_in, H, W;
    uint C_out, kH, kW;
    uint H_out, W_out;
    int  stride_h, stride_w;
    int  pad_h, pad_w;
    int  dil_h, dil_w;
    uint has_bias;
    uint total;
};

// One thread per output element. Direct conv, FP32 accumulator, FP16 IO.
// Matches CUDA naive implementation 1:1.
kernel void k_conv2d_forward(device const half* X    [[buffer(0)]],
                             device const half* Wt   [[buffer(1)]],
                             device const half* bias [[buffer(2)]],
                             device half*       Y    [[buffer(3)]],
                             constant ConvParams& p  [[buffer(4)]],
                             uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float acc = 0.0f;
    uint w_oc_base = oc * p.C_in * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.H * p.W;

    for (uint ic = 0; ic < p.C_in; ++ic) {
        uint w_ic_base = w_oc_base + ic * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.H * p.W;
        for (uint kh = 0; kh < p.kH; ++kh) {
            int in_h = in_h_origin + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint kw = 0; kw < p.kW; ++kw) {
                int in_w = in_w_origin + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                float x_v = float(X[x_ic_base + uint(in_h) * p.W + uint(in_w)]);
                float w_v = float(Wt[w_ic_base + kh * p.kW + kw]);
                acc += x_v * w_v;
            }
        }
    }
    if (p.has_bias != 0u) {
        acc += float(bias[oc]);
    }
    Y[idx] = half(acc);
}
)msl";

id<MTLComputePipelineState> pso_conv() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_forward"); });
    return pso;
}

struct ConvParams {
    uint32_t N, C_in, H, W;
    uint32_t C_out, kH, kW;
    uint32_t H_out, W_out;
    int32_t  stride_h, stride_w;
    int32_t  pad_h, pad_w;
    int32_t  dil_h, dil_w;
    uint32_t has_bias;
    uint32_t total;
};

} // namespace

void conv2d_forward_gpu(const GpuTensor& X,
                        const GpuTensor& Wt,
                        const GpuTensor* bias,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        int dil_h, int dil_w,
                        GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || Wt.dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_forward_gpu: X and Wt must be FP16");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_forward_gpu: bias must be FP16");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward_gpu: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = bias ? 1u : 0u;
    p.total = total;

    id<MTLComputePipelineState> pso = pso_conv();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(Wt);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx; // dummy bind if no bias
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(Wt);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor
