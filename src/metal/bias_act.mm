// ─── Metal bias_act (StyleGAN3) ─────────────────────────────────────────────
//
// Metal port of src/cuda/bias_act.cu. Fused per-channel bias + activation +
// gain + clamp, mirroring NVlabs `_bias_act_ref`.
//
//   X: (N, C*HW) — channel c owns the contiguous block [c*HW, (c+1)*HW)
//                  within each row.
//   b: (C,1) or null.   act: 0 = linear, 1 = lrelu.   clamp < 0 ⇒ no clamp.
//
// Forward:  t = X + b[c];  y = act(t);  y *= gain;  if clamp>=0: clip(±clamp).
// Backward: dt = dY*gain*act'(t)*(clamp active ? 0 : 1);
//           dX = dt (overwrite);  dB[c] += Σ dt (accumulate — caller zeros).
//
// Since dX[i] == dt exactly, dB[c] = Σ_{i in channel c} dX[i]. The backward
// therefore computes dX first, then reduces it per channel into dB (one
// threadgroup per channel, FP32 accumulation) — no float atomics needed.
//
// Dispatched on X.dtype (FP32 / FP16 / BF16); all arithmetic runs in FP32.

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

constexpr NSUInteger BA_BLOCK = 256;
constexpr int ACT_LINEAR = 0;
constexpr int ACT_LRELU  = 1;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint BA_BLOCK = 256;

inline float apply_act(float t, int act, float alpha) {
    if (act == 1) return t > 0.0f ? t : alpha * t;   // lrelu
    return t;
}
inline float act_grad(float t, int act, float alpha) {
    if (act == 1) return t > 0.0f ? 1.0f : alpha;
    return 1.0f;
}

struct BAParams {
    uint total; uint C; uint HW;
    int act; float alpha; float gain; float clamp;
    uint has_bias;
};

#define BIAS_ACT_FWD(NAME, T)                                                 \
kernel void NAME(device const T* X    [[buffer(0)]],                          \
                 device const T* b    [[buffer(1)]],                          \
                 device T*       Y    [[buffer(2)]],                          \
                 constant BAParams& p [[buffer(3)]],                          \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= p.total) return;                                                 \
    uint c = (i / p.HW) % p.C;                                                \
    float bias_v = p.has_bias != 0u ? float(b[c]) : 0.0f;                     \
    float y = apply_act(float(X[i]) + bias_v, p.act, p.alpha) * p.gain;       \
    if (p.clamp >= 0.0f) y = clamp(y, -p.clamp, p.clamp);                     \
    Y[i] = T(y);                                                              \
}

#define BIAS_ACT_BWD_DX(NAME, T)                                              \
kernel void NAME(device const T* dY   [[buffer(0)]],                          \
                 device const T* X    [[buffer(1)]],                          \
                 device const T* b    [[buffer(2)]],                          \
                 device T*       dX   [[buffer(3)]],                          \
                 constant BAParams& p [[buffer(4)]],                          \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= p.total) return;                                                 \
    uint c = (i / p.HW) % p.C;                                                \
    float bias_v = p.has_bias != 0u ? float(b[c]) : 0.0f;                     \
    float t = float(X[i]) + bias_v;                                           \
    float dt = float(dY[i]) * p.gain * act_grad(t, p.act, p.alpha);           \
    if (p.clamp >= 0.0f) {                                                    \
        float y_pre = p.gain * apply_act(t, p.act, p.alpha);                  \
        if (y_pre < -p.clamp || y_pre > p.clamp) dt = 0.0f;                   \
    }                                                                         \
    dX[i] = T(dt);                                                            \
}

// dB[c] += Σ_{n,hw} dX[n*C*HW + c*HW + hw]. One threadgroup per channel.
#define BIAS_ACT_DB(NAME, T)                                                  \
kernel void NAME(device const T* dX   [[buffer(0)]],                          \
                 device T*       dB   [[buffer(1)]],                          \
                 constant uint& N     [[buffer(2)]],                          \
                 constant uint& C     [[buffer(3)]],                          \
                 constant uint& HW    [[buffer(4)]],                          \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    uint tid = tid3.x; uint tg = tgs3.x;                                      \
    threadgroup float s[BA_BLOCK];                                            \
    const uint c = gid.x;                                                     \
    const uint cols = C * HW;                                                 \
    const uint per_c = N * HW;                                                \
    float local = 0.0f;                                                       \
    for (uint k = tid; k < per_c; k += tg) {                                  \
        uint n = k / HW; uint hw = k % HW;                                    \
        local += float(dX[(ulong)n * cols + (ulong)c * HW + hw]);             \
    }                                                                         \
    s[tid] = local;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint d = tg / 2; d > 0; d >>= 1) {                                   \
        if (tid < d) s[tid] += s[tid + d];                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) dB[c] = T(float(dB[c]) + s[0]);                             \
}

BIAS_ACT_FWD(k_bias_act_fwd_fp32, float)
BIAS_ACT_FWD(k_bias_act_fwd_fp16, half)
BIAS_ACT_FWD(k_bias_act_fwd_bf16, bfloat)
BIAS_ACT_BWD_DX(k_bias_act_bwd_dx_fp32, float)
BIAS_ACT_BWD_DX(k_bias_act_bwd_dx_fp16, half)
BIAS_ACT_BWD_DX(k_bias_act_bwd_dx_bf16, bfloat)
BIAS_ACT_DB(k_bias_act_db_fp32, float)
BIAS_ACT_DB(k_bias_act_db_fp16, half)
BIAS_ACT_DB(k_bias_act_db_bf16, bfloat)
)msl";

struct BAParams {
    uint32_t total; uint32_t C; uint32_t HW;
    int32_t act; float alpha; float gain; float clamp;
    uint32_t has_bias;
};

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_fwd_fp32, @"k_bias_act_fwd_fp32") DEF_PSO(pso_fwd_fp16, @"k_bias_act_fwd_fp16") DEF_PSO(pso_fwd_bf16, @"k_bias_act_fwd_bf16")
DEF_PSO(pso_dx_fp32, @"k_bias_act_bwd_dx_fp32") DEF_PSO(pso_dx_fp16, @"k_bias_act_bwd_dx_fp16") DEF_PSO(pso_dx_bf16, @"k_bias_act_bwd_dx_bf16")
DEF_PSO(pso_db_fp32, @"k_bias_act_db_fp32") DEF_PSO(pso_db_fp16, @"k_bias_act_db_fp16") DEF_PSO(pso_db_bf16, @"k_bias_act_db_bf16")
#undef DEF_PSO

void require_fp(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

void check_act(int act, const char* op) {
    if (act != ACT_LINEAR && act != ACT_LRELU)
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": act must be 0 (linear) or 1 (lrelu)");
}

void check_shapes(const char* op, const Tensor& X, const Tensor* b,
                  int N, int C, int HW) {
    const int cols = C * HW;
    if (X.rows != N || X.cols != cols)
        throw std::runtime_error(std::string("brotensor: ") + op + ": X shape mismatch");
    if (b && b->size() != C)
        throw std::runtime_error(std::string("brotensor: ") + op + ": b must have C elements");
}

id<MTLComputePipelineState> pick(const Tensor& t,
                                 id<MTLComputePipelineState> fp32,
                                 id<MTLComputePipelineState> fp16,
                                 id<MTLComputePipelineState> bf16) {
    return (t.dtype == Dtype::FP16) ? fp16
         : (t.dtype == Dtype::BF16) ? bf16 : fp32;
}

} // namespace

void bias_act_forward(const Tensor& X, const Tensor* b,
                      int N, int C, int HW, int act, float alpha,
                      float gain, float clamp, Tensor& Y) {
    require_fp("bias_act_forward", X, "X");
    if (b) {
        require_fp("bias_act_forward", *b, "b");
        if (b->dtype != X.dtype)
            throw std::runtime_error("brotensor: bias_act_forward: b.dtype must match X.dtype");
    }
    check_act(act, "bias_act_forward");
    check_shapes("bias_act_forward", X, b, N, C, HW);
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype)
        Y.resize(N, cols, X.dtype);
    const NSUInteger total = static_cast<NSUInteger>(N) * cols;
    if (total == 0) return;

    BAParams p{static_cast<uint32_t>(total), static_cast<uint32_t>(C),
               static_cast<uint32_t>(HW), act, alpha, gain, clamp,
               b ? 1u : 0u};
    id<MTLComputePipelineState> pso = pick(X, pso_fwd_fp32(), pso_fwd_fp16(), pso_fwd_bf16());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        id<MTLBuffer> bb = b ? buffer_for(*b) : buffer_for(X);  // dummy bind if no bias
        NSUInteger ob = b ? buffer_offset_for(*b) : buffer_offset_for(X);
        [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:2];
        [enc setBytes:&p length:sizeof(BAParams) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b,
                       int N, int C, int HW, int act, float alpha,
                       float gain, float clamp, Tensor& dX, Tensor* dB) {
    require_fp("bias_act_backward", dY, "dY");
    require_fp("bias_act_backward", X, "X");
    if (dY.dtype != X.dtype)
        throw std::runtime_error("brotensor: bias_act_backward: dY.dtype must match X.dtype");
    if (b) {
        require_fp("bias_act_backward", *b, "b");
        if (b->dtype != X.dtype)
            throw std::runtime_error("brotensor: bias_act_backward: b.dtype must match X.dtype");
    }
    if (dB) {
        require_fp("bias_act_backward", *dB, "dB");
        if (dB->dtype != X.dtype)
            throw std::runtime_error("brotensor: bias_act_backward: dB.dtype must match X.dtype");
        if (dB->size() != C)
            throw std::runtime_error("brotensor: bias_act_backward: dB must have C elements");
    }
    check_act(act, "bias_act_backward");
    check_shapes("bias_act_backward", X, b, N, C, HW);
    const int cols = C * HW;
    if (dY.rows != N || dY.cols != cols)
        throw std::runtime_error("brotensor: bias_act_backward: dY shape mismatch");
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype)
        dX.resize(N, cols, X.dtype);
    const NSUInteger total = static_cast<NSUInteger>(N) * cols;
    if (total == 0) return;

    BAParams p{static_cast<uint32_t>(total), static_cast<uint32_t>(C),
               static_cast<uint32_t>(HW), act, alpha, gain, clamp,
               b ? 1u : 0u};
    id<MTLComputePipelineState> pso_dx = pick(X, pso_dx_fp32(), pso_dx_fp16(), pso_dx_bf16());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_dx];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(X)  offset:buffer_offset_for(X)  atIndex:1];
        id<MTLBuffer> bb = b ? buffer_for(*b) : buffer_for(X);
        NSUInteger ob = b ? buffer_offset_for(*b) : buffer_offset_for(X);
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:3];
        [enc setBytes:&p length:sizeof(BAParams) atIndex:4];
        NSUInteger tpt = [pso_dx maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }

    if (!dB) return;
    // dB[c] += Σ dX over channel c — one threadgroup per channel, FP32 reduce.
    id<MTLComputePipelineState> pso_db = pick(X, pso_db_fp32(), pso_db_fp16(), pso_db_bf16());
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t HWu = static_cast<uint32_t>(HW);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_db];
        [enc setBuffer:buffer_for(dX)  offset:buffer_offset_for(dX)  atIndex:0];
        [enc setBuffer:buffer_for(*dB) offset:buffer_offset_for(*dB) atIndex:1];
        [enc setBytes:&Nu  length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Cu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&HWu length:sizeof(uint32_t) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(C, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(BA_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
