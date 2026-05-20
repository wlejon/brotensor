// Sinusoidal timestep embedding (Metal, FP32). Diffusers default for SD/SDXL:
// flip_sin_to_cos=True, downscale_freq_shift=0 → output [cos, sin].

#include <brotensor/runtime.h>

#include <cmath>
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

kernel void k_timestep_embedding(device const float* ts [[buffer(0)]],
                                 device float*       Y  [[buffer(1)]],
                                 constant uint&  N      [[buffer(2)]],
                                 constant uint&  dim    [[buffer(3)]],
                                 constant uint&  half_d [[buffer(4)]],
                                 constant float& log_mp [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    uint total = N * dim;
    if (gid >= total) return;
    uint i = gid / dim;
    uint j = gid - i * dim;
    if (j >= 2u * half_d) { Y[gid] = 0.0f; return; }
    uint k = j < half_d ? j : (j - half_d);
    float freq = precise::exp(-log_mp * float(k) / float(half_d));
    float arg  = ts[i] * freq;
    Y[gid] = j < half_d ? precise::cos(arg) : precise::sin(arg);
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_timestep_embedding"); });
    return p;
}

} // namespace

void timestep_embedding(const Tensor& timesteps,
                        int dim, float max_period,
                        Tensor& Y) {
    if (timesteps.dtype != Dtype::FP32) {
        throw std::runtime_error("timestep_embedding_gpu: timesteps must be FP32");
    }
    if (timesteps.cols != 1) {
        throw std::runtime_error("timestep_embedding_gpu: timesteps must be (N,1)");
    }
    if (dim <= 0) {
        throw std::runtime_error("timestep_embedding_gpu: dim must be positive");
    }
    const int N = timesteps.rows;
    if (Y.rows != N || Y.cols != dim || Y.dtype != Dtype::FP32) {
        Y.resize(N, dim, Dtype::FP32);
    }
    if (N == 0) return;

    const uint32_t Nu     = static_cast<uint32_t>(N);
    const uint32_t dimu   = static_cast<uint32_t>(dim);
    const uint32_t halfu  = static_cast<uint32_t>(dim / 2);
    const float log_mp    = std::log(max_period);
    const NSUInteger total = static_cast<NSUInteger>(N) * static_cast<NSUInteger>(dim);

    id<MTLComputePipelineState> p = pso();
    id<MTLBuffer> bT = buffer_for(timesteps);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oT = buffer_offset_for(timesteps);
    const NSUInteger oY = buffer_offset_for(Y);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:bT offset:oT atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBytes:&Nu     length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&dimu   length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&halfu  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&log_mp length:sizeof(float)    atIndex:5];
        NSUInteger tpt = [p maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor::detail::metal
