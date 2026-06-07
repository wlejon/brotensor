// ─── Metal StyleGAN3 synthesis-input primitives ────────────────────────────
//
// Metal port of src/cuda/stylegan_elementwise.cu (and the CPU reference
// src/cpu/stylegan_elementwise.cpp):
//   sin / cos     y = sin/cos(x);   dX = dY*cos(x) / -dY*sin(x)
//   rsqrt         y = 1/sqrt(x);    dX = -0.5*dY*y^3   (backward reads y)
//   pixel_norm    per-row RMS-over-channel normalise; backward vs the same
//                 closed form as the CPU/CUDA reference.
//
// sin/cos/rsqrt are elementwise; their backward OVERWRITES dX (no learnable
// parameters, x/y and dX/dY may alias). pixel_norm operates per row over the
// trailing (cols) axis — one threadgroup per row, threadgroup-reduction over
// the columns.
//
// rsqrt: the caller owns the x > 0 precondition (no guard — matching the CPU
// and CUDA backends and log/exp), so the IEEE result for x<=0 surfaces loudly.
//
// Dispatched on the input dtype (FP32 / FP16 / BF16). All arithmetic runs in
// FP32; only the storage loads / stores change type.

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

constexpr NSUInteger SG_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint SG_BLOCK = 256;

// ── elementwise unary fwd: one thread per element ──
#define UNARY_FWD(NAME, T, EXPR)                                              \
kernel void NAME(device const T* x [[buffer(0)]],                             \
                 device T*       y [[buffer(1)]],                             \
                 constant uint&  n [[buffer(2)]],                             \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= n) return;                                                       \
    float xv = float(x[i]);                                                   \
    y[i] = T(EXPR);                                                           \
}

// ── elementwise unary bwd: reads x (or y) and dY, overwrites dX ──
#define UNARY_BWD(NAME, T, EXPR)                                              \
kernel void NAME(device const T* x  [[buffer(0)]],                            \
                 device const T* dY [[buffer(1)]],                            \
                 device T*       dX [[buffer(2)]],                            \
                 constant uint&  n  [[buffer(3)]],                            \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= n) return;                                                       \
    float xv = float(x[i]);                                                   \
    float gv = float(dY[i]);                                                  \
    dX[i] = T(EXPR);                                                          \
}

UNARY_FWD(k_sin_fwd_fp32, float,  sin(xv))
UNARY_FWD(k_sin_fwd_fp16, half,   sin(xv))
UNARY_FWD(k_sin_fwd_bf16, bfloat, sin(xv))
UNARY_BWD(k_sin_bwd_fp32, float,  gv * cos(xv))
UNARY_BWD(k_sin_bwd_fp16, half,   gv * cos(xv))
UNARY_BWD(k_sin_bwd_bf16, bfloat, gv * cos(xv))

UNARY_FWD(k_cos_fwd_fp32, float,  cos(xv))
UNARY_FWD(k_cos_fwd_fp16, half,   cos(xv))
UNARY_FWD(k_cos_fwd_bf16, bfloat, cos(xv))
UNARY_BWD(k_cos_bwd_fp32, float,  -gv * sin(xv))
UNARY_BWD(k_cos_bwd_fp16, half,   -gv * sin(xv))
UNARY_BWD(k_cos_bwd_bf16, bfloat, -gv * sin(xv))

UNARY_FWD(k_rsqrt_fwd_fp32, float,  rsqrt(xv))
UNARY_FWD(k_rsqrt_fwd_fp16, half,   rsqrt(xv))
UNARY_FWD(k_rsqrt_fwd_bf16, bfloat, rsqrt(xv))
// backward reads the OUTPUT y (bound at buffer 0): dy/dx = -1/2 y^3.
UNARY_BWD(k_rsqrt_bwd_fp32, float,  -0.5f * gv * xv * xv * xv)
UNARY_BWD(k_rsqrt_bwd_fp16, half,   -0.5f * gv * xv * xv * xv)
UNARY_BWD(k_rsqrt_bwd_bf16, bfloat, -0.5f * gv * xv * xv * xv)

// ── pixel_norm: one threadgroup per row, reduce over columns ──
#define PIXEL_NORM_FWD(NAME, T)                                               \
kernel void NAME(device const T* X    [[buffer(0)]],                          \
                 device T*       Y    [[buffer(1)]],                          \
                 constant uint&  C    [[buffer(2)]],                          \
                 constant float& eps  [[buffer(3)]],                          \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    uint tid = tid3.x; uint tg = tgs3.x;                                      \
    threadgroup float s_ss[SG_BLOCK];                                         \
    threadgroup float s_rinv;                                                 \
    const uint r = gid.x;                                                     \
    device const T* xr = X + (ulong)r * C;                                    \
    device       T* yr = Y + (ulong)r * C;                                    \
    float local = 0.0f;                                                       \
    for (uint c = tid; c < C; c += tg) { float v = float(xr[c]); local += v * v; } \
    s_ss[tid] = local;                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tg / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) s_ss[tid] += s_ss[tid + s];                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) s_rinv = rsqrt(s_ss[0] / float(C) + eps);                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    float rinv = s_rinv;                                                      \
    for (uint c = tid; c < C; c += tg) yr[c] = T(float(xr[c]) * rinv);        \
}

#define PIXEL_NORM_BWD(NAME, T)                                               \
kernel void NAME(device const T* X    [[buffer(0)]],                          \
                 device const T* dY   [[buffer(1)]],                          \
                 device T*       dX   [[buffer(2)]],                          \
                 constant uint&  C    [[buffer(3)]],                          \
                 constant float& eps  [[buffer(4)]],                          \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    uint tid = tid3.x; uint tg = tgs3.x;                                      \
    threadgroup float s_ss[SG_BLOCK];                                         \
    threadgroup float s_s[SG_BLOCK];                                          \
    threadgroup float s_rinv; threadgroup float s_r3s;                        \
    const uint r = gid.x;                                                     \
    device const T* xr  = X  + (ulong)r * C;                                  \
    device const T* dyr = dY + (ulong)r * C;                                  \
    device       T* dxr = dX + (ulong)r * C;                                  \
    const float invC = 1.0f / float(C);                                       \
    float l_ss = 0.0f, l_s = 0.0f;                                            \
    for (uint c = tid; c < C; c += tg) {                                      \
        float xv = float(xr[c]); float dv = float(dyr[c]);                    \
        l_ss += xv * xv; l_s += dv * xv;                                      \
    }                                                                         \
    s_ss[tid] = l_ss; s_s[tid] = l_s;                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tg / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) { s_ss[tid] += s_ss[tid + s]; s_s[tid] += s_s[tid + s]; } \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) {                                                           \
        float rinv = rsqrt(s_ss[0] * invC + eps);                             \
        s_rinv = rinv;                                                        \
        s_r3s = rinv * rinv * rinv * s_s[0] * invC;                           \
    }                                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    float rinv = s_rinv; float r3s = s_r3s;                                   \
    for (uint c = tid; c < C; c += tg)                                        \
        dxr[c] = T(rinv * float(dyr[c]) - r3s * float(xr[c]));                \
}

PIXEL_NORM_FWD(k_pixel_norm_fwd_fp32, float)
PIXEL_NORM_FWD(k_pixel_norm_fwd_fp16, half)
PIXEL_NORM_FWD(k_pixel_norm_fwd_bf16, bfloat)
PIXEL_NORM_BWD(k_pixel_norm_bwd_fp32, float)
PIXEL_NORM_BWD(k_pixel_norm_bwd_fp16, half)
PIXEL_NORM_BWD(k_pixel_norm_bwd_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_sin_fwd_fp32, @"k_sin_fwd_fp32") DEF_PSO(pso_sin_fwd_fp16, @"k_sin_fwd_fp16") DEF_PSO(pso_sin_fwd_bf16, @"k_sin_fwd_bf16")
DEF_PSO(pso_sin_bwd_fp32, @"k_sin_bwd_fp32") DEF_PSO(pso_sin_bwd_fp16, @"k_sin_bwd_fp16") DEF_PSO(pso_sin_bwd_bf16, @"k_sin_bwd_bf16")
DEF_PSO(pso_cos_fwd_fp32, @"k_cos_fwd_fp32") DEF_PSO(pso_cos_fwd_fp16, @"k_cos_fwd_fp16") DEF_PSO(pso_cos_fwd_bf16, @"k_cos_fwd_bf16")
DEF_PSO(pso_cos_bwd_fp32, @"k_cos_bwd_fp32") DEF_PSO(pso_cos_bwd_fp16, @"k_cos_bwd_fp16") DEF_PSO(pso_cos_bwd_bf16, @"k_cos_bwd_bf16")
DEF_PSO(pso_rsqrt_fwd_fp32, @"k_rsqrt_fwd_fp32") DEF_PSO(pso_rsqrt_fwd_fp16, @"k_rsqrt_fwd_fp16") DEF_PSO(pso_rsqrt_fwd_bf16, @"k_rsqrt_fwd_bf16")
DEF_PSO(pso_rsqrt_bwd_fp32, @"k_rsqrt_bwd_fp32") DEF_PSO(pso_rsqrt_bwd_fp16, @"k_rsqrt_bwd_fp16") DEF_PSO(pso_rsqrt_bwd_bf16, @"k_rsqrt_bwd_bf16")
DEF_PSO(pso_pn_fwd_fp32, @"k_pixel_norm_fwd_fp32") DEF_PSO(pso_pn_fwd_fp16, @"k_pixel_norm_fwd_fp16") DEF_PSO(pso_pn_fwd_bf16, @"k_pixel_norm_fwd_bf16")
DEF_PSO(pso_pn_bwd_fp32, @"k_pixel_norm_bwd_fp32") DEF_PSO(pso_pn_bwd_fp16, @"k_pixel_norm_bwd_fp16") DEF_PSO(pso_pn_bwd_bf16, @"k_pixel_norm_bwd_bf16")
#undef DEF_PSO

void require_fp(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

// Run an elementwise unary: arg0=input, arg1=output, n at index 2.
void run_unary(id<MTLComputePipelineState> pso, const Tensor& in, Tensor& out) {
    const uint32_t n = static_cast<uint32_t>(in.size());
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(in)  offset:buffer_offset_for(in)  atIndex:0];
        [enc setBuffer:buffer_for(out) offset:buffer_offset_for(out) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// Run an elementwise unary backward: arg0=x|y, arg1=dY, arg2=dX, n at index 3.
void run_unary_bwd(id<MTLComputePipelineState> pso, const Tensor& a,
                   const Tensor& dY, Tensor& dX) {
    const uint32_t n = static_cast<uint32_t>(a.size());
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(a)  offset:buffer_offset_for(a)  atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

id<MTLComputePipelineState> pick(const Tensor& t,
                                 id<MTLComputePipelineState> fp32,
                                 id<MTLComputePipelineState> fp16,
                                 id<MTLComputePipelineState> bf16) {
    return (t.dtype == Dtype::FP16) ? fp16
         : (t.dtype == Dtype::BF16) ? bf16
         : fp32;
}

} // namespace

// ─── sin ────────────────────────────────────────────────────────────────────

void sin_forward(const Tensor& x, Tensor& y) {
    require_fp("sin_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)
        y.resize(x.rows, x.cols, x.dtype);
    run_unary(pick(x, pso_sin_fwd_fp32(), pso_sin_fwd_fp16(), pso_sin_fwd_bf16()), x, y);
}

void sin_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    require_fp("sin_backward", x, "x");
    require_fp("sin_backward", dY, "dY");
    if (dY.dtype != x.dtype)
        throw std::runtime_error("brotensor: sin_backward: dY.dtype must match x.dtype");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype)
        dX.resize(x.rows, x.cols, x.dtype);
    run_unary_bwd(pick(x, pso_sin_bwd_fp32(), pso_sin_bwd_fp16(), pso_sin_bwd_bf16()), x, dY, dX);
}

// ─── cos ────────────────────────────────────────────────────────────────────

void cos_forward(const Tensor& x, Tensor& y) {
    require_fp("cos_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)
        y.resize(x.rows, x.cols, x.dtype);
    run_unary(pick(x, pso_cos_fwd_fp32(), pso_cos_fwd_fp16(), pso_cos_fwd_bf16()), x, y);
}

void cos_backward(const Tensor& x, const Tensor& dY, Tensor& dX) {
    require_fp("cos_backward", x, "x");
    require_fp("cos_backward", dY, "dY");
    if (dY.dtype != x.dtype)
        throw std::runtime_error("brotensor: cos_backward: dY.dtype must match x.dtype");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype)
        dX.resize(x.rows, x.cols, x.dtype);
    run_unary_bwd(pick(x, pso_cos_bwd_fp32(), pso_cos_bwd_fp16(), pso_cos_bwd_bf16()), x, dY, dX);
}

// ─── rsqrt ──────────────────────────────────────────────────────────────────

void rsqrt_forward(const Tensor& x, Tensor& y) {
    require_fp("rsqrt_forward", x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)
        y.resize(x.rows, x.cols, x.dtype);
    run_unary(pick(x, pso_rsqrt_fwd_fp32(), pso_rsqrt_fwd_fp16(), pso_rsqrt_fwd_bf16()), x, y);
}

void rsqrt_backward(const Tensor& y, const Tensor& dY, Tensor& dX) {
    require_fp("rsqrt_backward", y, "y");
    require_fp("rsqrt_backward", dY, "dY");
    if (dY.dtype != y.dtype)
        throw std::runtime_error("brotensor: rsqrt_backward: dY.dtype must match y.dtype");
    if (dX.rows != y.rows || dX.cols != y.cols || dX.dtype != y.dtype)
        dX.resize(y.rows, y.cols, y.dtype);
    run_unary_bwd(pick(y, pso_rsqrt_bwd_fp32(), pso_rsqrt_bwd_fp16(), pso_rsqrt_bwd_bf16()), y, dY, dX);
}

// ─── pixel_norm ───────────────────────────────────────────────────────────────

void pixel_norm_forward(const Tensor& X, float eps, Tensor& Y) {
    require_fp("pixel_norm_forward", X, "X");
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype)
        Y.resize(X.rows, X.cols, X.dtype);
    const int R = X.rows, C = X.cols;
    if (R == 0 || C == 0) return;
    const uint32_t Cu = static_cast<uint32_t>(C);
    id<MTLComputePipelineState> pso =
        pick(X, pso_pn_fwd_fp32(), pso_pn_fwd_fp16(), pso_pn_fwd_bf16());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&Cu  length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&eps length:sizeof(float)    atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SG_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void pixel_norm_backward(const Tensor& X, const Tensor& dY, float eps, Tensor& dX) {
    require_fp("pixel_norm_backward", X, "X");
    require_fp("pixel_norm_backward", dY, "dY");
    if (dY.dtype != X.dtype)
        throw std::runtime_error("brotensor: pixel_norm_backward: dY.dtype must match X.dtype");
    if (dX.rows != X.rows || dX.cols != X.cols || dX.dtype != X.dtype)
        dX.resize(X.rows, X.cols, X.dtype);
    const int R = X.rows, C = X.cols;
    if (R == 0 || C == 0) return;
    const uint32_t Cu = static_cast<uint32_t>(C);
    id<MTLComputePipelineState> pso =
        pick(X, pso_pn_bwd_fp32(), pso_pn_bwd_fp16(), pso_pn_bwd_bf16());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X)  offset:buffer_offset_for(X)  atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&Cu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float)    atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SG_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
