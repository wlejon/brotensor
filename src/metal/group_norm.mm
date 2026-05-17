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

constexpr NSUInteger GN_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint GN_BLOCK = 256;

// One threadgroup per (sample, group) tile. Cooperative reduction over the
// tile (channels_per_group * spatial elements). FP32 accumulation, FP16 IO.
kernel void k_group_norm_forward(
        device const half*  X     [[buffer(0)]],
        device const half*  gamma [[buffer(1)]],
        device const half*  beta  [[buffer(2)]],
        device half*        Y     [[buffer(3)]],
        constant uint& C                  [[buffer(4)]],
        constant uint& spatial            [[buffer(5)]],
        constant uint& channels_per_group [[buffer(6)]],
        constant float& eps               [[buffer(7)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float s_sum[GN_BLOCK];
    threadgroup float s_sumsq[GN_BLOCK];
    threadgroup float s_mean;
    threadgroup float s_rstd;

    const uint g = gid.x;
    const uint n = gid.y;
    const uint tile_size = channels_per_group * spatial;
    const uint chan_base = g * channels_per_group;
    const uint sample_stride = C * spatial;

    device const half* x_tile = X + n * sample_stride + chan_base * spatial;
    device       half* y_tile = Y + n * sample_stride + chan_base * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        float v = float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }
    s_sum[tid]   = sum;
    s_sumsq[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]   += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float inv_n = 1.0f / float(tile_size);
        float mean = s_sum[0] * inv_n;
        float var  = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_mean;
    float rstd = s_rstd;

    for (uint i = tid; i < tile_size; i += tg_size) {
        uint local_c = i / spatial;
        uint channel = chan_base + local_c;
        float gv = float(gamma[channel]);
        float bv = float(beta[channel]);
        float v  = float(x_tile[i]);
        float yn = (v - mean) * rstd;
        y_tile[i] = half(yn * gv + bv);
    }
}
)msl";

id<MTLComputePipelineState> pso_gn() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_group_norm_forward"); });
    return pso;
}

} // namespace

void group_norm_forward_gpu(const GpuTensor& X,
                            const GpuTensor& gamma,
                            const GpuTensor& beta,
                            int N, int C, int H, int W,
                            int num_groups,
                            float eps,
                            GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma.dtype != Dtype::FP16 ||
        beta.dtype != Dtype::FP16) {
        throw std::runtime_error("group_norm_forward_gpu: all tensors must be FP16");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward_gpu: num_groups must divide C");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, cols, Dtype::FP16);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t channels_per_group = static_cast<uint32_t>(C / num_groups);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);

    id<MTLComputePipelineState> pso = pso_gn();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bg = buffer_for(gamma);
    id<MTLBuffer> bb = buffer_for(beta);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger og = buffer_offset_for(gamma);
    const NSUInteger ob = buffer_offset_for(beta);
    const NSUInteger oy = buffer_offset_for(Y);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&channels_per_group length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&eps length:sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, N, 1)
            threadsPerThreadgroup:MTLSizeMake(GN_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor
