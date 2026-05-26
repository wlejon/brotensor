#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger LN_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

kernel void k_ln_forward(device const float* x       [[buffer(0)]],
                         device const float* gamma   [[buffer(1)]],
                         device const float* beta    [[buffer(2)]],
                         device float*       y       [[buffer(3)]],
                         device float*       xhat    [[buffer(4)]],
                         device float*       scratch [[buffer(5)]],
                         constant uint& n            [[buffer(6)]],
                         constant float& eps         [[buffer(7)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) local += x[i];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(n);

    float local_v = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = x[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(n);
    float rstd = rsqrt(var + eps);
    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }
    for (uint i = tid; i < n; i += tg_size) {
        float xh = (x[i] - mean) * rstd;
        xhat[i] = xh;
        y[i] = gamma[i] * xh + beta[i];
    }
}

kernel void k_ln_backward(device const float* dY     [[buffer(0)]],
                          device const float* xhat   [[buffer(1)]],
                          device const float* gamma  [[buffer(2)]],
                          constant float& rstd       [[buffer(3)]],
                          device float*       dX     [[buffer(4)]],
                          device float*       dGamma [[buffer(5)]],
                          device float*       dBeta  [[buffer(6)]],
                          constant uint& n           [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g = dY[i];
        dGamma[i] += g * xhat[i];
        dBeta[i]  += g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += dY[i] * gamma[i];
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += dY[i] * gamma[i] * xhat[i];
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = dY[i] * gamma[i];
        dX[i] = scale * (nf * dxh - sum_dxh - xhat[i] * sum_dxh_xhat);
    }
}

// FP16 backward. dGamma/dBeta written to FP32 scratch; host folds into
// caller-owned FP16 accumulators.
kernel void k_ln_backward_fp16(device const half*  dY     [[buffer(0)]],
                               device const half*  xhat   [[buffer(1)]],
                               device const half*  gamma  [[buffer(2)]],
                               constant float& rstd       [[buffer(3)]],
                               device half*        dX     [[buffer(4)]],
                               device float*       dGamma_scratch [[buffer(5)]],
                               device float*       dBeta_scratch  [[buffer(6)]],
                               constant uint& n           [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g  = float(dY[i]);
        float xh = float(xhat[i]);
        dGamma_scratch[i] = g * xh;
        dBeta_scratch[i]  = g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += float(dY[i]) * float(gamma[i]);
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += float(dY[i]) * float(gamma[i]) * float(xhat[i]);
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = float(dY[i]) * float(gamma[i]);
        float xh  = float(xhat[i]);
        dX[i] = half(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
    }
}

kernel void k_ln_add_fp32_into_fp16(device const float* src [[buffer(0)]],
                                    device half*        dst [[buffer(1)]],
                                    constant uint& n        [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}

// BF16 backward — verbatim copy of k_ln_backward_fp16 with half→bfloat.
// FP32 scratch/fold pattern mirrored exactly.
kernel void k_ln_backward_bf16(device const bfloat* dY     [[buffer(0)]],
                               device const bfloat* xhat   [[buffer(1)]],
                               device const bfloat* gamma  [[buffer(2)]],
                               constant float& rstd         [[buffer(3)]],
                               device bfloat*       dX     [[buffer(4)]],
                               device float*       dGamma_scratch [[buffer(5)]],
                               device float*       dBeta_scratch  [[buffer(6)]],
                               constant uint& n             [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g  = float(dY[i]);
        float xh = float(xhat[i]);
        dGamma_scratch[i] = g * xh;
        dBeta_scratch[i]  = g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += float(dY[i]) * float(gamma[i]);
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += float(dY[i]) * float(gamma[i]) * float(xhat[i]);
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = float(dY[i]) * float(gamma[i]);
        float xh  = float(xhat[i]);
        dX[i] = bfloat(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
    }
}

kernel void k_ln_add_fp32_into_bf16(device const float* src [[buffer(0)]],
                                    device bfloat*      dst [[buffer(1)]],
                                    constant uint& n        [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}

kernel void k_ln_forward_inference_batched(device const float* x      [[buffer(0)]],
                                           device const float* gamma  [[buffer(1)]],
                                           device const float* beta   [[buffer(2)]],
                                           device float*       y      [[buffer(3)]],
                                           constant uint& R           [[buffer(4)]],
                                           constant uint& D           [[buffer(5)]],
                                           constant float& eps        [[buffer(6)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    if (row >= R) return;
    device const float* xrow = x + row * D;
    device float*       yrow = y + row * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tg_size) local += xrow[i];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(D);

    float local_v = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float d = xrow[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(D);
    float rstd = rsqrt(var + eps);
    for (uint i = tid; i < D; i += tg_size) {
        float xh = (xrow[i] - mean) * rstd;
        yrow[i] = xh * gamma[i] + beta[i];
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
DEF_PSO(pso_fw, @"k_ln_forward")
DEF_PSO(pso_bw, @"k_ln_backward")
DEF_PSO(pso_bw_fp16, @"k_ln_backward_fp16")
DEF_PSO(pso_ln_add_fp16, @"k_ln_add_fp32_into_fp16")
DEF_PSO(pso_bw_bf16, @"k_ln_backward_bf16")
DEF_PSO(pso_ln_add_bf16, @"k_ln_add_fp32_into_bf16")
DEF_PSO(pso_fw_inf, @"k_ln_forward_inference_batched")
#undef DEF_PSO

} // namespace

void layernorm_forward(const Tensor& x,
                       const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat,
                       float& mean_out, float& rstd_out,
                       float eps) {
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (xhat.rows != x.rows || xhat.cols != x.cols) xhat.resize(x.rows, x.cols);
    if (n == 0) { mean_out = 0.0f; rstd_out = 0.0f; return; }
    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:2 * sizeof(float)
                        options:MTLResourceStorageModeShared];
        float* sptr = static_cast<float*>([scratch contents]);
        sptr[0] = 0.0f; sptr[1] = 0.0f;
        id<MTLComputePipelineState> pso = pso_fw();
        id<MTLBuffer> bx = buffer_for(x);
    NSUInteger ox = buffer_offset_for(x);
        id<MTLBuffer> bg = buffer_for(gamma);
    NSUInteger og = buffer_offset_for(gamma);
        id<MTLBuffer> bb = buffer_for(beta);
    NSUInteger ob = buffer_offset_for(beta);
        id<MTLBuffer> by = buffer_for(y);
    NSUInteger oy = buffer_offset_for(y);
        id<MTLBuffer> bh = buffer_for(xhat);
    NSUInteger oh = buffer_offset_for(xhat);
        const uint32_t nu = static_cast<uint32_t>(n);
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBuffer:bh offset:oh atIndex:4];
        [enc setBuffer:scratch offset:0 atIndex:5];
        [enc setBytes:&nu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&eps length:sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        // `cmd` is the shared batched buffer; flush() commits it (after the
        // prior batch) and drains the GPU so the scratch readback is valid.
        metal_impl::flush();
        mean_out = sptr[0];
        rstd_out = sptr[1];
    }
}

void layernorm_backward(const Tensor& dY, const Tensor& xhat,
                        const Tensor& gamma, float rstd,
                        Tensor& dX,
                        Tensor& dGamma, Tensor& dBeta) {
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("layernorm_backward: dY must be FP16, BF16, or FP32");
    }
    if (xhat.dtype != dY.dtype || gamma.dtype != dY.dtype ||
        dGamma.dtype != dY.dtype || dBeta.dtype != dY.dtype) {
        throw std::runtime_error("layernorm_backward: all tensors must share dtype");
    }
    const int n = dY.size();
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    if (n == 0) return;

    id<MTLBuffer> bdy = buffer_for(dY);
    NSUInteger ody = buffer_offset_for(dY);
    id<MTLBuffer> bxh = buffer_for(xhat);
    NSUInteger oxh = buffer_offset_for(xhat);
    id<MTLBuffer> bg  = buffer_for(gamma);
    NSUInteger og = buffer_offset_for(gamma);
    id<MTLBuffer> bdx = buffer_for(dX);
    NSUInteger odx = buffer_offset_for(dX);
    id<MTLBuffer> bdg = buffer_for(dGamma);
    NSUInteger odg = buffer_offset_for(dGamma);
    id<MTLBuffer> bdb = buffer_for(dBeta);
    NSUInteger odb = buffer_offset_for(dBeta);
    const uint32_t nu = static_cast<uint32_t>(n);

    if (dY.dtype == Dtype::FP32) {
        id<MTLComputePipelineState> pso = pso_bw();
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bdy offset:ody atIndex:0];
            [enc setBuffer:bxh offset:oxh atIndex:1];
            [enc setBuffer:bg offset:og atIndex:2];
            [enc setBytes:&rstd length:sizeof(float) atIndex:3];
            [enc setBuffer:bdx offset:odx atIndex:4];
            [enc setBuffer:bdg offset:odg atIndex:5];
            [enc setBuffer:bdb offset:odb atIndex:6];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
            [enc endEncoding];
            // `cmd` is the shared batched buffer — flush() commits + drains it.
            metal_impl::flush();
        }
    } else if (dY.dtype == Dtype::BF16) {
        @autoreleasepool {
            id<MTLBuffer> scratch_g = [metal_impl::device()
                newBufferWithLength:n * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> scratch_b = [metal_impl::device()
                newBufferWithLength:n * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pso = pso_bw_bf16();
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bdy offset:ody atIndex:0];
            [enc setBuffer:bxh offset:oxh atIndex:1];
            [enc setBuffer:bg offset:og atIndex:2];
            [enc setBytes:&rstd length:sizeof(float) atIndex:3];
            [enc setBuffer:bdx offset:odx atIndex:4];
            [enc setBuffer:scratch_g offset:0 atIndex:5];
            [enc setBuffer:scratch_b offset:0 atIndex:6];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];

            id<MTLComputePipelineState> add_pso = pso_ln_add_bf16();
            [enc setComputePipelineState:add_pso];
            [enc setBuffer:scratch_g offset:0 atIndex:0];
            [enc setBuffer:bdg offset:odg atIndex:1];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:2];
            NSUInteger tg = [add_pso maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(nu, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

            [enc setBuffer:scratch_b offset:0 atIndex:0];
            [enc setBuffer:bdb offset:odb atIndex:1];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(nu, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

            [enc endEncoding];
            // `cmd` is the shared batched buffer — flush() commits + drains it.
            metal_impl::flush();
        }
    } else {
        @autoreleasepool {
            id<MTLBuffer> scratch_g = [metal_impl::device()
                newBufferWithLength:n * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> scratch_b = [metal_impl::device()
                newBufferWithLength:n * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pso = pso_bw_fp16();
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bdy offset:ody atIndex:0];
            [enc setBuffer:bxh offset:oxh atIndex:1];
            [enc setBuffer:bg offset:og atIndex:2];
            [enc setBytes:&rstd length:sizeof(float) atIndex:3];
            [enc setBuffer:bdx offset:odx atIndex:4];
            [enc setBuffer:scratch_g offset:0 atIndex:5];
            [enc setBuffer:scratch_b offset:0 atIndex:6];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];

            id<MTLComputePipelineState> add_pso = pso_ln_add_fp16();
            [enc setComputePipelineState:add_pso];
            [enc setBuffer:scratch_g offset:0 atIndex:0];
            [enc setBuffer:bdg offset:odg atIndex:1];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:2];
            NSUInteger tg = [add_pso maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:MTLSizeMake(nu, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

            [enc setBuffer:scratch_b offset:0 atIndex:0];
            [enc setBuffer:bdb offset:odb atIndex:1];
            [enc setBytes:&nu length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(nu, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

            [enc endEncoding];
            // `cmd` is the shared batched buffer — flush() commits + drains it.
            metal_impl::flush();
        }
    }
}

void layernorm_forward_inference_batched(const Tensor& X_RD,
                                         const Tensor& gamma,
                                         const Tensor& beta,
                                         Tensor& Y_RD,
                                         float eps) {
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D) Y_RD.resize(R, D);
    if (R == 0 || D == 0) return;
    id<MTLComputePipelineState> pso = pso_fw_inf();
    id<MTLBuffer> bx = buffer_for(X_RD);
    NSUInteger ox = buffer_offset_for(X_RD);
    id<MTLBuffer> bg = buffer_for(gamma);
    NSUInteger og = buffer_offset_for(gamma);
    id<MTLBuffer> bb = buffer_for(beta);
    NSUInteger ob = buffer_offset_for(beta);
    id<MTLBuffer> by = buffer_for(Y_RD);
    NSUInteger oy = buffer_offset_for(Y_RD);
    const uint32_t Ru = static_cast<uint32_t>(R);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&Ru length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&eps length:sizeof(float) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

namespace {

NSString* const kFp16Src = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

kernel void k_ln_forward_inference_batched_fp16(
        device const half* x      [[buffer(0)]],
        device const half* gamma  [[buffer(1)]],
        device const half* beta   [[buffer(2)]],
        device half*       y      [[buffer(3)]],
        constant uint& R          [[buffer(4)]],
        constant uint& D          [[buffer(5)]],
        constant float& eps       [[buffer(6)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    if (row >= R) return;
    device const half* xrow = x + (ulong)row * D;
    device       half* yrow = y + (ulong)row * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tg_size) local += float(xrow[i]);
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(D);

    float local_v = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float d = float(xrow[i]) - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(D);
    float rstd = rsqrt(var + eps);
    for (uint i = tid; i < D; i += tg_size) {
        float xh = (float(xrow[i]) - mean) * rstd;
        float g  = float(gamma[i]);
        float b  = float(beta[i]);
        yrow[i] = half(xh * g + b);
    }
}
)msl";

id<MTLComputePipelineState> pso_fw_inf_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kFp16Src, @"k_ln_forward_inference_batched_fp16");
    });
    return pso;
}

} // namespace

void layernorm_forward_inference_batched_fp16(const Tensor& X_RD,
                                              const Tensor& gamma,
                                              const Tensor& beta,
                                              Tensor& Y_RD,
                                              float eps) {
    if (X_RD.dtype != Dtype::FP16 || gamma.dtype != Dtype::FP16 ||
        beta.dtype != Dtype::FP16) {
        throw std::runtime_error("layernorm_forward_inference_batched_fp16: all tensors must be FP16");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != Dtype::FP16) {
        Y_RD.resize(R, D, Dtype::FP16);
    }
    if (R == 0 || D == 0) return;
    id<MTLComputePipelineState> pso = pso_fw_inf_fp16();
    id<MTLBuffer> bx = buffer_for(X_RD);
    id<MTLBuffer> bg = buffer_for(gamma);
    id<MTLBuffer> bb = buffer_for(beta);
    id<MTLBuffer> by = buffer_for(Y_RD);
    const NSUInteger ox = buffer_offset_for(X_RD);
    const NSUInteger og = buffer_offset_for(gamma);
    const NSUInteger ob = buffer_offset_for(beta);
    const NSUInteger oy = buffer_offset_for(Y_RD);
    const uint32_t Ru = static_cast<uint32_t>(R);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&Ru length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&eps length:sizeof(float) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// ─── Batched (training) LayerNorm with caches ─────────────────────────────
//
// Two-pass backward (avoids cross-threadgroup atomics): pass 1 computes dX
// row-by-row (R threadgroups); pass 2 reduces dGamma/dBeta column-by-column
// across rows (D threadgroups). FP16/BF16 paths go through an FP32 scratch
// buffer + fold to preserve "caller zeros, op accumulates" semantics.
namespace {

NSString* const kBatchedCachesSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

// ─── Forward (one threadgroup per row, FP32/FP16/BF16) ───────────────────
#define LN_FWD_BATCHED(SUFFIX, T)                                                  \
kernel void k_ln_fwd_batched_caches_##SUFFIX(                                       \
        device const T*     x     [[buffer(0)]],                                    \
        device const T*     gamma [[buffer(1)]],                                    \
        device const T*     beta  [[buffer(2)]],                                    \
        device T*           y     [[buffer(3)]],                                    \
        device T*           xhat  [[buffer(4)]],                                    \
        device float*       mean_out [[buffer(5)]],                                 \
        device float*       rstd_out [[buffer(6)]],                                 \
        constant uint& R          [[buffer(7)]],                                    \
        constant uint& D          [[buffer(8)]],                                    \
        constant float& eps       [[buffer(9)]],                                    \
        uint row [[threadgroup_position_in_grid]],                                  \
        uint tid [[thread_position_in_threadgroup]],                                \
        uint tg_size [[threads_per_threadgroup]]) {                                 \
    threadgroup float sdata[LN_BLOCK];                                              \
    if (row >= R) return;                                                           \
    device const T* xrow = x    + (ulong)row * D;                                   \
    device       T* yrow = y    + (ulong)row * D;                                   \
    device       T* hrow = xhat + (ulong)row * D;                                   \
    float local = 0.0f;                                                             \
    for (uint i = tid; i < D; i += tg_size) local += float(xrow[i]);                \
    sdata[tid] = local;                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                    \
        if (tid < s) sdata[tid] += sdata[tid + s];                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float mean = sdata[0] / float(D);                                               \
    float local_v = 0.0f;                                                           \
    for (uint i = tid; i < D; i += tg_size) {                                       \
        float d = float(xrow[i]) - mean;                                            \
        local_v += d * d;                                                           \
    }                                                                               \
    sdata[tid] = local_v;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                    \
        if (tid < s) sdata[tid] += sdata[tid + s];                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float var  = sdata[0] / float(D);                                               \
    float rstd = rsqrt(var + eps);                                                  \
    if (tid == 0) {                                                                 \
        mean_out[row] = mean;                                                       \
        rstd_out[row] = rstd;                                                       \
    }                                                                               \
    for (uint i = tid; i < D; i += tg_size) {                                       \
        float xh = (float(xrow[i]) - mean) * rstd;                                  \
        hrow[i]  = T(xh);                                                           \
        float g  = float(gamma[i]);                                                 \
        float b  = float(beta[i]);                                                  \
        yrow[i]  = T(xh * g + b);                                                   \
    }                                                                               \
}
LN_FWD_BATCHED(f32,  float)
LN_FWD_BATCHED(fp16, half)
LN_FWD_BATCHED(bf16, bfloat)
#undef LN_FWD_BATCHED

// ─── Backward pass 1 — per-row dX (R threadgroups) ──────────────────────
// Computes sum_dxh, sum_dxh_xhat per row in shmem, then writes dX.
#define LN_BWD_DX_BATCHED(SUFFIX, T)                                                \
kernel void k_ln_bwd_dx_batched_##SUFFIX(                                          \
        device const T*     dY    [[buffer(0)]],                                   \
        device const T*     xhat  [[buffer(1)]],                                   \
        device const T*     gamma [[buffer(2)]],                                   \
        device const float* rstd_R[[buffer(3)]],                                   \
        device T*           dX    [[buffer(4)]],                                   \
        constant uint& R          [[buffer(5)]],                                   \
        constant uint& D          [[buffer(6)]],                                   \
        uint row [[threadgroup_position_in_grid]],                                 \
        uint tid [[thread_position_in_threadgroup]],                               \
        uint tg_size [[threads_per_threadgroup]]) {                                \
    threadgroup float sdata[LN_BLOCK];                                             \
    if (row >= R) return;                                                          \
    device const T* dyr = dY   + (ulong)row * D;                                   \
    device const T* hr  = xhat + (ulong)row * D;                                   \
    device       T* dxr = dX   + (ulong)row * D;                                   \
    float rstd = rstd_R[row];                                                      \
    float local = 0.0f;                                                            \
    for (uint i = tid; i < D; i += tg_size) {                                      \
        local += float(dyr[i]) * float(gamma[i]);                                  \
    }                                                                              \
    sdata[tid] = local;                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) sdata[tid] += sdata[tid + s];                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    }                                                                              \
    float sum_dxh = sdata[0];                                                      \
    float local2 = 0.0f;                                                           \
    for (uint i = tid; i < D; i += tg_size) {                                      \
        local2 += float(dyr[i]) * float(gamma[i]) * float(hr[i]);                  \
    }                                                                              \
    sdata[tid] = local2;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) sdata[tid] += sdata[tid + s];                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    }                                                                              \
    float sum_dxh_xhat = sdata[0];                                                 \
    float nf = float(D);                                                           \
    float scale = rstd / nf;                                                       \
    for (uint i = tid; i < D; i += tg_size) {                                      \
        float dxh = float(dyr[i]) * float(gamma[i]);                               \
        float xh  = float(hr[i]);                                                  \
        dxr[i] = T(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));              \
    }                                                                              \
}
LN_BWD_DX_BATCHED(f32,  float)
LN_BWD_DX_BATCHED(fp16, half)
LN_BWD_DX_BATCHED(bf16, bfloat)
#undef LN_BWD_DX_BATCHED

// ─── Backward pass 2 — per-feature dGamma/dBeta (D threadgroups) ────────
// Each threadgroup d sums over all R rows: dGamma[d] = sum_r dY[r,d]*xhat[r,d];
// dBeta[d] = sum_r dY[r,d]. Writes FP32 scratch; host fold-adds into the
// caller's accumulator. Single threadgroup per column avoids races.
#define LN_BWD_DGB_BATCHED(SUFFIX, T)                                              \
kernel void k_ln_bwd_dgb_batched_##SUFFIX(                                         \
        device const T*     dY    [[buffer(0)]],                                   \
        device const T*     xhat  [[buffer(1)]],                                   \
        device float*       dGamma_f32 [[buffer(2)]],                              \
        device float*       dBeta_f32  [[buffer(3)]],                              \
        constant uint& R          [[buffer(4)]],                                   \
        constant uint& D          [[buffer(5)]],                                   \
        uint d   [[threadgroup_position_in_grid]],                                 \
        uint tid [[thread_position_in_threadgroup]],                               \
        uint tg_size [[threads_per_threadgroup]]) {                                \
    threadgroup float sg[LN_BLOCK];                                                \
    threadgroup float sb[LN_BLOCK];                                                \
    if (d >= D) return;                                                            \
    float lg = 0.0f, lb = 0.0f;                                                    \
    for (uint r = tid; r < R; r += tg_size) {                                      \
        float g  = float(dY[(ulong)r * D + d]);                                    \
        float xh = float(xhat[(ulong)r * D + d]);                                  \
        lg += g * xh;                                                              \
        lb += g;                                                                   \
    }                                                                              \
    sg[tid] = lg;                                                                  \
    sb[tid] = lb;                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) { sg[tid] += sg[tid + s]; sb[tid] += sb[tid + s]; }           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    }                                                                              \
    if (tid == 0) {                                                                \
        dGamma_f32[d] = sg[0];                                                     \
        dBeta_f32[d]  = sb[0];                                                     \
    }                                                                              \
}
LN_BWD_DGB_BATCHED(f32,  float)
LN_BWD_DGB_BATCHED(fp16, half)
LN_BWD_DGB_BATCHED(bf16, bfloat)
#undef LN_BWD_DGB_BATCHED

// ─── Folds: dGamma += scratch (typed) ──────────────────────────────────
kernel void k_ln_fold_add_f32(device const float* src [[buffer(0)]],
                              device float*       dst [[buffer(1)]],
                              constant uint& n        [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = dst[i] + src[i];
}
kernel void k_ln_fold_add_fp16(device const float* src [[buffer(0)]],
                               device half*        dst [[buffer(1)]],
                               constant uint& n        [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}
kernel void k_ln_fold_add_bf16(device const float* src [[buffer(0)]],
                               device bfloat*      dst [[buffer(1)]],
                               constant uint& n        [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}
)msl";

#define DEF_BC_PSO(NAME, FN)                                                       \
    id<MTLComputePipelineState> NAME() {                                            \
        static dispatch_once_t once;                                                \
        static id<MTLComputePipelineState> pso;                                     \
        dispatch_once(&once, ^{ pso = compile_pipeline(kBatchedCachesSrc, FN); });  \
        return pso;                                                                 \
    }
DEF_BC_PSO(pso_fwbc_f32,  @"k_ln_fwd_batched_caches_f32")
DEF_BC_PSO(pso_fwbc_fp16, @"k_ln_fwd_batched_caches_fp16")
DEF_BC_PSO(pso_fwbc_bf16, @"k_ln_fwd_batched_caches_bf16")
DEF_BC_PSO(pso_bwdx_f32,  @"k_ln_bwd_dx_batched_f32")
DEF_BC_PSO(pso_bwdx_fp16, @"k_ln_bwd_dx_batched_fp16")
DEF_BC_PSO(pso_bwdx_bf16, @"k_ln_bwd_dx_batched_bf16")
DEF_BC_PSO(pso_bwdgb_f32,  @"k_ln_bwd_dgb_batched_f32")
DEF_BC_PSO(pso_bwdgb_fp16, @"k_ln_bwd_dgb_batched_fp16")
DEF_BC_PSO(pso_bwdgb_bf16, @"k_ln_bwd_dgb_batched_bf16")
DEF_BC_PSO(pso_fold_f32,   @"k_ln_fold_add_f32")
DEF_BC_PSO(pso_fold_fp16,  @"k_ln_fold_add_fp16")
DEF_BC_PSO(pso_fold_bf16,  @"k_ln_fold_add_bf16")
#undef DEF_BC_PSO

} // namespace

void layernorm_forward_batched_with_caches(const Tensor& X_RD,
                                           const Tensor& gamma,
                                           const Tensor& beta,
                                           Tensor& Y_RD, Tensor& Xhat_RD,
                                           Tensor& Mean_R, Tensor& Rstd_R,
                                           float eps) {
    if (X_RD.dtype != Dtype::FP16 && X_RD.dtype != Dtype::BF16 &&
        X_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward_batched_with_caches: X must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != X_RD.dtype || beta.dtype != X_RD.dtype) {
        throw std::runtime_error("layernorm_forward_batched_with_caches: gamma/beta dtype must match X.dtype");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != X_RD.dtype) {
        Y_RD.resize(R, D, X_RD.dtype);
    }
    if (Xhat_RD.rows != R || Xhat_RD.cols != D || Xhat_RD.dtype != X_RD.dtype) {
        Xhat_RD.resize(R, D, X_RD.dtype);
    }
    if (Mean_R.rows != R || Mean_R.cols != 1 || Mean_R.dtype != Dtype::FP32) {
        Mean_R.resize(R, 1, Dtype::FP32);
    }
    if (Rstd_R.rows != R || Rstd_R.cols != 1 || Rstd_R.dtype != Dtype::FP32) {
        Rstd_R.resize(R, 1, Dtype::FP32);
    }
    if (R == 0 || D == 0) return;

    id<MTLComputePipelineState> pso = (X_RD.dtype == Dtype::FP16) ? pso_fwbc_fp16()
                                    : (X_RD.dtype == Dtype::BF16) ? pso_fwbc_bf16()
                                                                  : pso_fwbc_f32();
    id<MTLBuffer> bx = buffer_for(X_RD);    NSUInteger ox = buffer_offset_for(X_RD);
    id<MTLBuffer> bg = buffer_for(gamma);   NSUInteger og = buffer_offset_for(gamma);
    id<MTLBuffer> bb = buffer_for(beta);    NSUInteger ob = buffer_offset_for(beta);
    id<MTLBuffer> by = buffer_for(Y_RD);    NSUInteger oy = buffer_offset_for(Y_RD);
    id<MTLBuffer> bh = buffer_for(Xhat_RD); NSUInteger oh = buffer_offset_for(Xhat_RD);
    id<MTLBuffer> bm = buffer_for(Mean_R);  NSUInteger om = buffer_offset_for(Mean_R);
    id<MTLBuffer> bs = buffer_for(Rstd_R);  NSUInteger os = buffer_offset_for(Rstd_R);
    const uint32_t Ru = static_cast<uint32_t>(R);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bg offset:og atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBuffer:bh offset:oh atIndex:4];
        [enc setBuffer:bm offset:om atIndex:5];
        [enc setBuffer:bs offset:os atIndex:6];
        [enc setBytes:&Ru length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&eps length:sizeof(float) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void layernorm_backward_batched_with_caches(const Tensor& dY_RD,
                                            const Tensor& Xhat_RD,
                                            const Tensor& gamma,
                                            const Tensor& Rstd_R,
                                            Tensor& dX_RD,
                                            Tensor& dGamma, Tensor& dBeta) {
    if (dY_RD.dtype != Dtype::FP16 && dY_RD.dtype != Dtype::BF16 &&
        dY_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY must be FP16, BF16, or FP32");
    }
    if (Xhat_RD.dtype != dY_RD.dtype || gamma.dtype != dY_RD.dtype ||
        dGamma.dtype != dY_RD.dtype || dBeta.dtype != dY_RD.dtype) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY/Xhat/gamma/dGamma/dBeta must share dtype");
    }
    if (Rstd_R.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: Rstd_R must be FP32");
    }
    const int R = dY_RD.rows;
    const int D = dY_RD.cols;
    if (dX_RD.rows != R || dX_RD.cols != D || dX_RD.dtype != dY_RD.dtype) {
        dX_RD.resize(R, D, dY_RD.dtype);
    }
    if (R == 0 || D == 0) return;

    id<MTLComputePipelineState> pso_dx  = (dY_RD.dtype == Dtype::FP16) ? pso_bwdx_fp16()
                                        : (dY_RD.dtype == Dtype::BF16) ? pso_bwdx_bf16()
                                                                       : pso_bwdx_f32();
    id<MTLComputePipelineState> pso_dgb = (dY_RD.dtype == Dtype::FP16) ? pso_bwdgb_fp16()
                                        : (dY_RD.dtype == Dtype::BF16) ? pso_bwdgb_bf16()
                                                                       : pso_bwdgb_f32();
    id<MTLComputePipelineState> pso_fold= (dY_RD.dtype == Dtype::FP16) ? pso_fold_fp16()
                                        : (dY_RD.dtype == Dtype::BF16) ? pso_fold_bf16()
                                                                       : pso_fold_f32();

    id<MTLBuffer> bdy = buffer_for(dY_RD);    NSUInteger ody = buffer_offset_for(dY_RD);
    id<MTLBuffer> bxh = buffer_for(Xhat_RD);  NSUInteger oxh = buffer_offset_for(Xhat_RD);
    id<MTLBuffer> bg  = buffer_for(gamma);    NSUInteger og  = buffer_offset_for(gamma);
    id<MTLBuffer> bs  = buffer_for(Rstd_R);   NSUInteger os  = buffer_offset_for(Rstd_R);
    id<MTLBuffer> bdx = buffer_for(dX_RD);    NSUInteger odx = buffer_offset_for(dX_RD);
    id<MTLBuffer> bdg = buffer_for(dGamma);   NSUInteger odg = buffer_offset_for(dGamma);
    id<MTLBuffer> bdb = buffer_for(dBeta);    NSUInteger odb = buffer_offset_for(dBeta);
    const uint32_t Ru = static_cast<uint32_t>(R);
    const uint32_t Du = static_cast<uint32_t>(D);

    @autoreleasepool {
        id<MTLBuffer> scratch_g = [metal_impl::device()
            newBufferWithLength:D * sizeof(float)
                        options:MTLResourceStorageModePrivate];
        id<MTLBuffer> scratch_b = [metal_impl::device()
            newBufferWithLength:D * sizeof(float)
                        options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Pass 1: dX, one threadgroup per row.
        [enc setComputePipelineState:pso_dx];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:bxh offset:oxh atIndex:1];
        [enc setBuffer:bg  offset:og  atIndex:2];
        [enc setBuffer:bs  offset:os  atIndex:3];
        [enc setBuffer:bdx offset:odx atIndex:4];
        [enc setBytes:&Ru length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];

        // Pass 2: dGamma/dBeta into FP32 scratch, one threadgroup per column.
        [enc setComputePipelineState:pso_dgb];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:bxh offset:oxh atIndex:1];
        [enc setBuffer:scratch_g offset:0 atIndex:2];
        [enc setBuffer:scratch_b offset:0 atIndex:3];
        [enc setBytes:&Ru length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(D, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LN_BLOCK, 1, 1)];

        // Fold: dGamma += scratch_g; dBeta += scratch_b (typed widen/narrow).
        NSUInteger tg = [pso_fold maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc setComputePipelineState:pso_fold];
        [enc setBuffer:scratch_g offset:0 atIndex:0];
        [enc setBuffer:bdg offset:odg atIndex:1];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Du, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc setBuffer:scratch_b offset:0 atIndex:0];
        [enc setBuffer:bdb offset:odb atIndex:1];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Du, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        [enc endEncoding];
        metal_impl::flush();
    }
}

} // namespace brotensor::detail::metal
