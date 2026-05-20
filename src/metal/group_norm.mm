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

constexpr NSUInteger GN_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint GN_BLOCK = 256;

// One threadgroup per (sample, group) tile. Cooperative reduction over the
// tile (channels_per_group * spatial elements). FP32 accumulation, FP16 IO.
kernel void k_group_norm_forward_fp16(
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

kernel void k_group_norm_forward_fp32(
        device const float* X     [[buffer(0)]],
        device const float* gamma [[buffer(1)]],
        device const float* beta  [[buffer(2)]],
        device float*       Y     [[buffer(3)]],
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

    device const float* x_tile = X + n * sample_stride + chan_base * spatial;
    device       float* y_tile = Y + n * sample_stride + chan_base * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        float v = x_tile[i];
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
        float gv = gamma[channel];
        float bv = beta[channel];
        float v  = x_tile[i];
        float yn = (v - mean) * rstd;
        y_tile[i] = yn * gv + bv;
    }
}

// Backward: 3 passes per (n, g) tile. Per-channel dGamma/dBeta accumulated
// into FP32 atomic scratch buffers; a follow-up kernel adds into the
// caller-owned (FP16 or FP32) accumulators.
kernel void k_group_norm_backward_fp16(
        device const half*  X     [[buffer(0)]],
        device const half*  gamma [[buffer(1)]],
        device const half*  dY    [[buffer(2)]],
        device half*        dX    [[buffer(3)]],
        device atomic_float* dGamma_acc [[buffer(4)]],
        device atomic_float* dBeta_acc  [[buffer(5)]],
        constant uint& C                  [[buffer(6)]],
        constant uint& spatial            [[buffer(7)]],
        constant uint& channels_per_group [[buffer(8)]],
        constant float& eps               [[buffer(9)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float s_a[GN_BLOCK];
    threadgroup float s_b[GN_BLOCK];
    threadgroup float s_mean;
    threadgroup float s_rstd;
    threadgroup float s_sum1;
    threadgroup float s_sum2;

    const uint g = gid.x;
    const uint n = gid.y;
    const uint tile_size = channels_per_group * spatial;
    const uint chan_base = g * channels_per_group;
    const uint sample_stride = C * spatial;

    device const half* x_tile  = X  + n * sample_stride + chan_base * spatial;
    device const half* dy_tile = dY + n * sample_stride + chan_base * spatial;
    device       half* dx_tile = dX + n * sample_stride + chan_base * spatial;

    // Pass 1: mean, var.
    float sum = 0.0f, sumsq = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        float v = float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }
    s_a[tid] = sum; s_b[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_a[tid] += s_a[tid + s];
            s_b[tid] += s_b[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float inv_n = 1.0f / float(tile_size);
        float mean = s_a[0] * inv_n;
        float var  = s_b[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_mean;
    float rstd = s_rstd;

    // Pass 2: sum1, sum2; also atomic dGamma/dBeta per channel.
    float sum1 = 0.0f, sum2 = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        uint local_c = i / spatial;
        uint channel = chan_base + local_c;
        float gv  = float(gamma[channel]);
        float dyv = float(dy_tile[i]);
        float xv  = float(x_tile[i]);
        float xh  = (xv - mean) * rstd;
        float dxh = dyv * gv;
        sum1 += dxh;
        sum2 += dxh * xh;
        atomic_fetch_add_explicit(&dGamma_acc[channel], dyv * xh,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&dBeta_acc[channel],  dyv,
                                  memory_order_relaxed);
    }
    s_a[tid] = sum1; s_b[tid] = sum2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_a[tid] += s_a[tid + s];
            s_b[tid] += s_b[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { s_sum1 = s_a[0]; s_sum2 = s_b[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum1_t = s_sum1;
    float sum2_t = s_sum2;
    float inv_M  = 1.0f / float(tile_size);

    // Pass 3.
    for (uint i = tid; i < tile_size; i += tg_size) {
        uint local_c = i / spatial;
        uint channel = chan_base + local_c;
        float gv  = float(gamma[channel]);
        float dyv = float(dy_tile[i]);
        float xv  = float(x_tile[i]);
        float xh  = (xv - mean) * rstd;
        float dxh = dyv * gv;
        float dx  = rstd * (dxh - (sum1_t + xh * sum2_t) * inv_M);
        dx_tile[i] = half(dx);
    }
}

kernel void k_group_norm_backward_fp32(
        device const float* X     [[buffer(0)]],
        device const float* gamma [[buffer(1)]],
        device const float* dY    [[buffer(2)]],
        device float*       dX    [[buffer(3)]],
        device atomic_float* dGamma_acc [[buffer(4)]],
        device atomic_float* dBeta_acc  [[buffer(5)]],
        constant uint& C                  [[buffer(6)]],
        constant uint& spatial            [[buffer(7)]],
        constant uint& channels_per_group [[buffer(8)]],
        constant float& eps               [[buffer(9)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float s_a[GN_BLOCK];
    threadgroup float s_b[GN_BLOCK];
    threadgroup float s_mean;
    threadgroup float s_rstd;
    threadgroup float s_sum1;
    threadgroup float s_sum2;

    const uint g = gid.x;
    const uint n = gid.y;
    const uint tile_size = channels_per_group * spatial;
    const uint chan_base = g * channels_per_group;
    const uint sample_stride = C * spatial;

    device const float* x_tile  = X  + n * sample_stride + chan_base * spatial;
    device const float* dy_tile = dY + n * sample_stride + chan_base * spatial;
    device       float* dx_tile = dX + n * sample_stride + chan_base * spatial;

    float sum = 0.0f, sumsq = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        float v = x_tile[i];
        sum   += v;
        sumsq += v * v;
    }
    s_a[tid] = sum; s_b[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_a[tid] += s_a[tid + s];
            s_b[tid] += s_b[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float inv_n = 1.0f / float(tile_size);
        float mean = s_a[0] * inv_n;
        float var  = s_b[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_mean;
    float rstd = s_rstd;

    float sum1 = 0.0f, sum2 = 0.0f;
    for (uint i = tid; i < tile_size; i += tg_size) {
        uint local_c = i / spatial;
        uint channel = chan_base + local_c;
        float gv  = gamma[channel];
        float dyv = dy_tile[i];
        float xv  = x_tile[i];
        float xh  = (xv - mean) * rstd;
        float dxh = dyv * gv;
        sum1 += dxh;
        sum2 += dxh * xh;
        atomic_fetch_add_explicit(&dGamma_acc[channel], dyv * xh,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&dBeta_acc[channel],  dyv,
                                  memory_order_relaxed);
    }
    s_a[tid] = sum1; s_b[tid] = sum2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_a[tid] += s_a[tid + s];
            s_b[tid] += s_b[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { s_sum1 = s_a[0]; s_sum2 = s_b[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum1_t = s_sum1;
    float sum2_t = s_sum2;
    float inv_M  = 1.0f / float(tile_size);

    for (uint i = tid; i < tile_size; i += tg_size) {
        uint local_c = i / spatial;
        uint channel = chan_base + local_c;
        float gv  = gamma[channel];
        float dyv = dy_tile[i];
        float xv  = x_tile[i];
        float xh  = (xv - mean) * rstd;
        float dxh = dyv * gv;
        dx_tile[i] = rstd * (dxh - (sum1_t + xh * sum2_t) * inv_M);
    }
}

// Add FP32 scratch into FP16/FP32 dGamma/dBeta accumulators.
kernel void k_add_fp32_into_fp16(device const float* src [[buffer(0)]],
                                 device half*        dst [[buffer(1)]],
                                 constant uint& n        [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float prev = float(dst[gid]);
    dst[gid] = half(prev + src[gid]);
}

kernel void k_add_fp32_into_fp32(device const float* src [[buffer(0)]],
                                 device float*       dst [[buffer(1)]],
                                 constant uint& n        [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] += src[gid];
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_gn_fwd_fp16, @"k_group_norm_forward_fp16")
DEF_PSO(pso_gn_fwd_fp32, @"k_group_norm_forward_fp32")
DEF_PSO(pso_gn_bwd_fp16, @"k_group_norm_backward_fp16")
DEF_PSO(pso_gn_bwd_fp32, @"k_group_norm_backward_fp32")
DEF_PSO(pso_add_fp16,    @"k_add_fp32_into_fp16")
DEF_PSO(pso_add_fp32,    @"k_add_fp32_into_fp32")
#undef DEF_PSO

} // namespace

void group_norm_forward(const Tensor& X,
                        const Tensor& gamma,
                        const Tensor& beta,
                        int N, int C, int H, int W,
                        int num_groups,
                        float eps,
                        Tensor& Y) {
    if (gamma.dtype != X.dtype || beta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_forward: gamma/beta dtype must match X");
    }
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("group_norm_forward: X must be FP16 or FP32");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward: num_groups must divide C");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t channels_per_group = static_cast<uint32_t>(C / num_groups);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);

    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16)
        ? pso_gn_fwd_fp16() : pso_gn_fwd_fp32();
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

void group_norm_backward(const Tensor& X,
                         const Tensor& gamma,
                         const Tensor& dY,
                         int N, int C, int H, int W,
                         int num_groups,
                         float eps,
                         Tensor& dX,
                         Tensor& dGamma,
                         Tensor& dBeta) {
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: gamma/dY dtype must match X");
    }
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("group_norm_backward: X must be FP16 or FP32");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_backward: num_groups must divide C");
    }
    if (dGamma.dtype != X.dtype || dBeta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: dGamma/dBeta dtype must match X");
    }
    if (dGamma.rows != C || dGamma.cols != 1 ||
        dBeta.rows  != C || dBeta.cols  != 1) {
        throw std::runtime_error("group_norm_backward: dGamma/dBeta must be (C,1)");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype) {
        dX.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t channels_per_group = static_cast<uint32_t>(C / num_groups);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);

    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16)
        ? pso_gn_bwd_fp16() : pso_gn_bwd_fp32();

    id<MTLBuffer> bx  = buffer_for(X);
    id<MTLBuffer> bg  = buffer_for(gamma);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    id<MTLBuffer> bdg = buffer_for(dGamma);
    id<MTLBuffer> bdb = buffer_for(dBeta);
    const NSUInteger ox  = buffer_offset_for(X);
    const NSUInteger og  = buffer_offset_for(gamma);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odx = buffer_offset_for(dX);
    const NSUInteger odg = buffer_offset_for(dGamma);
    const NSUInteger odb = buffer_offset_for(dBeta);

    @autoreleasepool {
        // FP32 scratch for per-channel grads (atomic adds in kernel).
        id<MTLBuffer> scratch_g = [metal_impl::device()
            newBufferWithLength:C * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> scratch_b = [metal_impl::device()
            newBufferWithLength:C * sizeof(float)
                        options:MTLResourceStorageModeShared];
        std::memset([scratch_g contents], 0, C * sizeof(float));
        std::memset([scratch_b contents], 0, C * sizeof(float));

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx  offset:ox  atIndex:0];
        [enc setBuffer:bg  offset:og  atIndex:1];
        [enc setBuffer:bdy offset:ody atIndex:2];
        [enc setBuffer:bdx offset:odx atIndex:3];
        [enc setBuffer:scratch_g offset:0 atIndex:4];
        [enc setBuffer:scratch_b offset:0 atIndex:5];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&channels_per_group length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&eps length:sizeof(float) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, N, 1)
            threadsPerThreadgroup:MTLSizeMake(GN_BLOCK, 1, 1)];

        // Fold FP32 scratch into caller-owned accumulators.
        id<MTLComputePipelineState> add_pso = (X.dtype == Dtype::FP16)
            ? pso_add_fp16() : pso_add_fp32();
        const uint32_t Cn = static_cast<uint32_t>(C);
        const NSUInteger tpt = 64;
        const NSUInteger ngroups = (C + tpt - 1) / tpt;
        [enc setComputePipelineState:add_pso];
        [enc setBuffer:scratch_g offset:0 atIndex:0];
        [enc setBuffer:bdg offset:odg atIndex:1];
        [enc setBytes:&Cn length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(ngroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];

        [enc setBuffer:scratch_b offset:0 atIndex:0];
        [enc setBuffer:bdb offset:odb atIndex:1];
        [enc setBytes:&Cn length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(ngroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor::detail::metal
