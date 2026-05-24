// Metal BatchNorm (NCHW, FP32-only — matches CPU / CUDA contract).
//
// Three slots, all reducing over (N, H, W) per channel:
//
//   batch_norm_forward    — training; computes per-channel batch mean / var,
//                           writes Y, updates running_mean / running_var
//                           in place (PyTorch convention; running_var uses
//                           the unbiased estimator). Saves batch mean and
//                           rstd for the backward pass.
//   batch_norm_inference  — applies the affine using running stats; no state
//                           mutation.
//   batch_norm_backward   — caller zeros dX / dGamma / dBeta; op overwrites
//                           dX and accumulates into dGamma / dBeta.
//
// Kernel layout: one threadgroup per channel (BN_BLOCK threads), threads
// cooperate over the M = N*spatial elements of that channel scattered
// through the NCHW tensor. Single writer per channel, so dGamma / dBeta
// accumulation in backward does not need atomics — mirrors the CUDA port.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger BN_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint BN_BLOCK = 256;

// One threadgroup per channel `c`. Threads iterate over M = N*spatial
// elements of channel `c`, addressed as X[(n*C + c)*spatial + s].
kernel void k_bn_forward(
        device const float* X            [[buffer(0)]],
        device const float* gamma        [[buffer(1)]],
        device const float* beta         [[buffer(2)]],
        device float*       running_mean [[buffer(3)]],
        device float*       running_var  [[buffer(4)]],
        device float*       Y            [[buffer(5)]],
        device float*       saved_mean   [[buffer(6)]],
        device float*       saved_rstd   [[buffer(7)]],
        constant uint& N        [[buffer(8)]],
        constant uint& C        [[buffer(9)]],
        constant uint& spatial  [[buffer(10)]],
        constant float& eps     [[buffer(11)]],
        constant float& momentum [[buffer(12)]],
        uint3 gid   [[threadgroup_position_in_grid]],
        uint3 tid3  [[thread_position_in_threadgroup]],
        uint3 tgs3  [[threads_per_threadgroup]]) {
    const uint c = gid.x;
    const uint tid = tid3.x;
    const uint tg_size = tgs3.x;
    const uint M = N * spatial;

    threadgroup float s_sum[BN_BLOCK];
    threadgroup float s_sumsq[BN_BLOCK];
    threadgroup float s_mean, s_rstd, s_gv, s_bv;

    // Pass 1: sum + sumsq.
    float lsum = 0.0f, lsumsq = 0.0f;
    for (uint n = 0; n < N; ++n) {
        device const float* x_chan = X + (n * C + c) * spatial;
        for (uint s = tid; s < spatial; s += tg_size) {
            float v = x_chan[s];
            lsum   += v;
            lsumsq += v * v;
        }
    }
    s_sum[tid]   = lsum;
    s_sumsq[tid] = lsumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float inv_M = 1.0f / float(M);
        float mean  = s_sum[0]   * inv_M;
        float var_b = s_sumsq[0] * inv_M - mean * mean;   // biased
        float rstd  = rsqrt(var_b + eps);
        float bessel = (M > 1) ? float(M) / float(M - 1) : 1.0f;
        float var_unb = var_b * bessel;
        s_mean = mean;
        s_rstd = rstd;
        s_gv   = gamma[c];
        s_bv   = beta[c];
        saved_mean[c] = mean;
        saved_rstd[c] = rstd;
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var_unb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_mean;
    float rstd = s_rstd;
    float gv   = s_gv;
    float bv   = s_bv;

    // Pass 2: Y = (x - mean) * rstd * gamma + beta.
    for (uint n = 0; n < N; ++n) {
        device const float* x_chan = X + (n * C + c) * spatial;
        device       float* y_chan = Y + (n * C + c) * spatial;
        for (uint s = tid; s < spatial; s += tg_size) {
            y_chan[s] = (x_chan[s] - mean) * rstd * gv + bv;
        }
    }
}

kernel void k_bn_inference(
        device const float* X            [[buffer(0)]],
        device const float* gamma        [[buffer(1)]],
        device const float* beta         [[buffer(2)]],
        device const float* running_mean [[buffer(3)]],
        device const float* running_var  [[buffer(4)]],
        device float*       Y            [[buffer(5)]],
        constant uint& N        [[buffer(6)]],
        constant uint& C        [[buffer(7)]],
        constant uint& spatial  [[buffer(8)]],
        constant float& eps     [[buffer(9)]],
        uint3 gid   [[threadgroup_position_in_grid]],
        uint3 tid3  [[thread_position_in_threadgroup]],
        uint3 tgs3  [[threads_per_threadgroup]]) {
    const uint c = gid.x;
    const uint tid = tid3.x;
    const uint tg_size = tgs3.x;
    float inv = rsqrt(running_var[c] + eps);
    float a = gamma[c] * inv;
    float b = beta[c] - running_mean[c] * a;
    for (uint n = 0; n < N; ++n) {
        device const float* x_chan = X + (n * C + c) * spatial;
        device       float* y_chan = Y + (n * C + c) * spatial;
        for (uint s = tid; s < spatial; s += tg_size) {
            y_chan[s] = x_chan[s] * a + b;
        }
    }
}

// One threadgroup per channel `c`. Two reduction passes (sum_dY +
// sum_dY_xh), then the dX write pass. dGamma / dBeta are accumulated
// (caller-zeroed) — single writer per channel, so no atomics needed.
kernel void k_bn_backward(
        device const float* X          [[buffer(0)]],
        device const float* gamma      [[buffer(1)]],
        device const float* saved_mean [[buffer(2)]],
        device const float* saved_rstd [[buffer(3)]],
        device const float* dY         [[buffer(4)]],
        device float*       dX         [[buffer(5)]],
        device float*       dGamma     [[buffer(6)]],
        device float*       dBeta      [[buffer(7)]],
        constant uint& N        [[buffer(8)]],
        constant uint& C        [[buffer(9)]],
        constant uint& spatial  [[buffer(10)]],
        uint3 gid   [[threadgroup_position_in_grid]],
        uint3 tid3  [[thread_position_in_threadgroup]],
        uint3 tgs3  [[threads_per_threadgroup]]) {
    const uint c = gid.x;
    const uint tid = tid3.x;
    const uint tg_size = tgs3.x;
    const uint M = N * spatial;

    threadgroup float s_a[BN_BLOCK];
    threadgroup float s_b[BN_BLOCK];
    threadgroup float s_sum_dY, s_sum_dY_xh;

    float mean = saved_mean[c];
    float rstd = saved_rstd[c];
    float gv   = gamma[c];

    float lsum_dY = 0.0f, lsum_dY_xh = 0.0f;
    for (uint n = 0; n < N; ++n) {
        device const float* x_chan  = X  + (n * C + c) * spatial;
        device const float* dy_chan = dY + (n * C + c) * spatial;
        for (uint s = tid; s < spatial; s += tg_size) {
            float xh = (x_chan[s] - mean) * rstd;
            lsum_dY    += dy_chan[s];
            lsum_dY_xh += dy_chan[s] * xh;
        }
    }
    s_a[tid] = lsum_dY;
    s_b[tid] = lsum_dY_xh;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
            s_b[tid] += s_b[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        s_sum_dY    = s_a[0];
        s_sum_dY_xh = s_b[0];
        dGamma[c] += s_b[0];   // accumulate
        dBeta[c]  += s_a[0];   // accumulate
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_dY    = s_sum_dY;
    float sum_dY_xh = s_sum_dY_xh;
    float sum1 = gv * sum_dY;
    float sum2 = gv * sum_dY_xh;
    float inv_M = 1.0f / float(M);

    for (uint n = 0; n < N; ++n) {
        device const float* x_chan  = X  + (n * C + c) * spatial;
        device const float* dy_chan = dY + (n * C + c) * spatial;
        device       float* dx_chan = dX + (n * C + c) * spatial;
        for (uint s = tid; s < spatial; s += tg_size) {
            float xh  = (x_chan[s] - mean) * rstd;
            float dxh = dy_chan[s] * gv;
            dx_chan[s] = rstd * (dxh - (sum1 + xh * sum2) * inv_M);
        }
    }
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_bn_forward,   @"k_bn_forward")
DEF_PSO(pso_bn_inference, @"k_bn_inference")
DEF_PSO(pso_bn_backward,  @"k_bn_backward")
#undef DEF_PSO

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32");
    }
}

inline void check_per_channel(const ::brotensor::Tensor& t,
                              int C, const char* op, const char* name) {
    if (t.size() != C) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must have C elements");
    }
}

} // namespace

void batch_norm_forward(const Tensor& X,
                        const Tensor& gamma,
                        const Tensor& beta,
                        Tensor& running_mean,
                        Tensor& running_var,
                        int N, int C, int H, int W,
                        float eps, float momentum,
                        Tensor& Y,
                        Tensor& saved_mean,
                        Tensor& saved_rstd) {
    check_fp32(X,            "batch_norm_forward", "X");
    check_fp32(gamma,        "batch_norm_forward", "gamma");
    check_fp32(beta,         "batch_norm_forward", "beta");
    check_fp32(running_mean, "batch_norm_forward", "running_mean");
    check_fp32(running_var,  "batch_norm_forward", "running_var");
    check_per_channel(gamma,        C, "batch_norm_forward", "gamma");
    check_per_channel(beta,         C, "batch_norm_forward", "beta");
    check_per_channel(running_mean, C, "batch_norm_forward", "running_mean");
    check_per_channel(running_var,  C, "batch_norm_forward", "running_var");

    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (saved_mean.rows != C || saved_mean.cols != 1 ||
        saved_mean.dtype != Dtype::FP32) {
        saved_mean.resize(C, 1, Dtype::FP32);
    }
    if (saved_rstd.rows != C || saved_rstd.cols != 1 ||
        saved_rstd.dtype != Dtype::FP32) {
        saved_rstd.resize(C, 1, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);

    id<MTLBuffer> bX  = buffer_for(X);
    id<MTLBuffer> bG  = buffer_for(gamma);
    id<MTLBuffer> bB  = buffer_for(beta);
    id<MTLBuffer> bRM = buffer_for(running_mean);
    id<MTLBuffer> bRV = buffer_for(running_var);
    id<MTLBuffer> bY  = buffer_for(Y);
    id<MTLBuffer> bSM = buffer_for(saved_mean);
    id<MTLBuffer> bSR = buffer_for(saved_rstd);
    const NSUInteger oX  = buffer_offset_for(X);
    const NSUInteger oG  = buffer_offset_for(gamma);
    const NSUInteger oB  = buffer_offset_for(beta);
    const NSUInteger oRM = buffer_offset_for(running_mean);
    const NSUInteger oRV = buffer_offset_for(running_var);
    const NSUInteger oY  = buffer_offset_for(Y);
    const NSUInteger oSM = buffer_offset_for(saved_mean);
    const NSUInteger oSR = buffer_offset_for(saved_rstd);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_bn_forward()];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bG  offset:oG  atIndex:1];
        [enc setBuffer:bB  offset:oB  atIndex:2];
        [enc setBuffer:bRM offset:oRM atIndex:3];
        [enc setBuffer:bRV offset:oRV atIndex:4];
        [enc setBuffer:bY  offset:oY  atIndex:5];
        [enc setBuffer:bSM offset:oSM atIndex:6];
        [enc setBuffer:bSR offset:oSR atIndex:7];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:10];
        [enc setBytes:&eps      length:sizeof(float) atIndex:11];
        [enc setBytes:&momentum length:sizeof(float) atIndex:12];
        [enc dispatchThreadgroups:MTLSizeMake(C, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(BN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void batch_norm_inference(const Tensor& X,
                          const Tensor& gamma,
                          const Tensor& beta,
                          const Tensor& running_mean,
                          const Tensor& running_var,
                          int N, int C, int H, int W,
                          float eps,
                          Tensor& Y) {
    check_fp32(X,            "batch_norm_inference", "X");
    check_fp32(gamma,        "batch_norm_inference", "gamma");
    check_fp32(beta,         "batch_norm_inference", "beta");
    check_fp32(running_mean, "batch_norm_inference", "running_mean");
    check_fp32(running_var,  "batch_norm_inference", "running_var");
    check_per_channel(gamma,        C, "batch_norm_inference", "gamma");
    check_per_channel(beta,         C, "batch_norm_inference", "beta");
    check_per_channel(running_mean, C, "batch_norm_inference", "running_mean");
    check_per_channel(running_var,  C, "batch_norm_inference", "running_var");

    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);

    id<MTLBuffer> bX  = buffer_for(X);
    id<MTLBuffer> bG  = buffer_for(gamma);
    id<MTLBuffer> bB  = buffer_for(beta);
    id<MTLBuffer> bRM = buffer_for(running_mean);
    id<MTLBuffer> bRV = buffer_for(running_var);
    id<MTLBuffer> bY  = buffer_for(Y);
    const NSUInteger oX  = buffer_offset_for(X);
    const NSUInteger oG  = buffer_offset_for(gamma);
    const NSUInteger oB  = buffer_offset_for(beta);
    const NSUInteger oRM = buffer_offset_for(running_mean);
    const NSUInteger oRV = buffer_offset_for(running_var);
    const NSUInteger oY  = buffer_offset_for(Y);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_bn_inference()];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bG  offset:oG  atIndex:1];
        [enc setBuffer:bB  offset:oB  atIndex:2];
        [enc setBuffer:bRM offset:oRM atIndex:3];
        [enc setBuffer:bRV offset:oRV atIndex:4];
        [enc setBuffer:bY  offset:oY  atIndex:5];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&eps length:sizeof(float)   atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(C, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(BN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void batch_norm_backward(const Tensor& X,
                         const Tensor& gamma,
                         const Tensor& saved_mean,
                         const Tensor& saved_rstd,
                         const Tensor& dY,
                         int N, int C, int H, int W,
                         Tensor& dX,
                         Tensor& dGamma,
                         Tensor& dBeta) {
    check_fp32(X,          "batch_norm_backward", "X");
    check_fp32(gamma,      "batch_norm_backward", "gamma");
    check_fp32(saved_mean, "batch_norm_backward", "saved_mean");
    check_fp32(saved_rstd, "batch_norm_backward", "saved_rstd");
    check_fp32(dY,         "batch_norm_backward", "dY");
    check_fp32(dGamma,     "batch_norm_backward", "dGamma");
    check_fp32(dBeta,      "batch_norm_backward", "dBeta");
    check_per_channel(gamma,      C, "batch_norm_backward", "gamma");
    check_per_channel(saved_mean, C, "batch_norm_backward", "saved_mean");
    check_per_channel(saved_rstd, C, "batch_norm_backward", "saved_rstd");
    if (dGamma.rows != C || dGamma.cols != 1 ||
        dBeta.rows  != C || dBeta.cols  != 1) {
        throw std::runtime_error(
            "brotensor: batch_norm_backward: dGamma/dBeta must be (C,1)");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (dY.rows != N || dY.cols != cols) {
        throw std::runtime_error("brotensor: batch_norm_backward: dY shape mismatch");
    }
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("brotensor: batch_norm_backward: X shape mismatch");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);

    id<MTLBuffer> bX  = buffer_for(X);
    id<MTLBuffer> bG  = buffer_for(gamma);
    id<MTLBuffer> bSM = buffer_for(saved_mean);
    id<MTLBuffer> bSR = buffer_for(saved_rstd);
    id<MTLBuffer> bDY = buffer_for(dY);
    id<MTLBuffer> bDX = buffer_for(dX);
    id<MTLBuffer> bDG = buffer_for(dGamma);
    id<MTLBuffer> bDB = buffer_for(dBeta);
    const NSUInteger oX  = buffer_offset_for(X);
    const NSUInteger oG  = buffer_offset_for(gamma);
    const NSUInteger oSM = buffer_offset_for(saved_mean);
    const NSUInteger oSR = buffer_offset_for(saved_rstd);
    const NSUInteger oDY = buffer_offset_for(dY);
    const NSUInteger oDX = buffer_offset_for(dX);
    const NSUInteger oDG = buffer_offset_for(dGamma);
    const NSUInteger oDB = buffer_offset_for(dBeta);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_bn_backward()];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bG  offset:oG  atIndex:1];
        [enc setBuffer:bSM offset:oSM atIndex:2];
        [enc setBuffer:bSR offset:oSR atIndex:3];
        [enc setBuffer:bDY offset:oDY atIndex:4];
        [enc setBuffer:bDX offset:oDX atIndex:5];
        [enc setBuffer:bDG offset:oDG atIndex:6];
        [enc setBuffer:bDB offset:oDB atIndex:7];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&Su length:sizeof(uint32_t) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(C, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(BN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
