// ─── Metal modulated_conv2d (StyleGAN3) ─────────────────────────────────────
//
// Metal port of src/cuda/modulated_conv2d.cu. The StyleGAN synthesis-layer
// core: per-sample style modulation of the conv weights, optional
// demodulation, then a standard stride-1 conv per sample.
//
// Realized by looping the batch and reusing the validated Metal conv2d kernels
// (groups=1) on a per-sample weight — only the weight construction + demod are
// new (small fused reduction kernels). Layouts: X (N,C_in*H*W) NCHW;
// W (C_out, C_in*kH*kW) OIHW; s (N,C_in).
//
//   w'[o,i,kh,kw] = W[o,i,kh,kw] * s[n,i]
//   dcoef[n,o]    = demodulate ? rsqrt(Σ_{i,kh,kw} w'^2 + eps) : 1
//   w''           = w' * dcoef[n,o]
//   Y[n]          = conv2d(X[n], w'', pad, stride=1)
//
// Backward (per n; dw'' = conv2d_backward_weight(X[n],dY[n])):
//   g[o]    = Σ dw''[o,..] * w'[o,..]
//   dw'[o]  = demodulate ? dw''[o]*dcoef - g[o]*dcoef^3*w'[o] : dw''[o]
//   dW[o]  += Σ_n dw'[n,o] * s[n,i]      (accumulate — caller zeros dW)
//   ds[n,i] = Σ_{o,kh,kw} dw'[o,i,kh,kw] * W[o,i,kh,kw]   (overwrite)
//   dX[n]   = conv2d_backward_input(w''[n], dY[n])         (overwrite)
//
// dtype: FP32/FP16/BF16. The modulated weights/gradients carry the storage
// dtype (so the reused conv2d kernels dispatch correctly); all reductions and
// the dW accumulation run in FP32. The demod coefficient cache `dcoef` is
// always FP32.

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

// Reused Metal conv2d kernels (defined in conv2d.mm, same namespace).
void conv2d_forward(const Tensor& X, const Tensor& Wt, const Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, Tensor& Y);
void conv2d_backward_input(const Tensor& Wt, const Tensor& dY,
                           int N, int C_in, int H, int W, int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups, Tensor& dX);
void conv2d_backward_weight(const Tensor& X, const Tensor& dY,
                            int N, int C_in, int H, int W, int C_out, int kH, int kW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dil_h, int dil_w, int groups, Tensor& dWt);

namespace {

constexpr NSUInteger MC_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint MC_BLOCK = 256;

// Forward: one threadgroup per output channel o. Build w' into Wn, reduce its
// sum of squares (FP32), write dcoef[o], then scale by the demod coefficient.
#define MODULATE_BUILD(NAME, T)                                               \
kernel void NAME(device const T* W      [[buffer(0)]],                        \
                 device const T* sn     [[buffer(1)]],                        \
                 device T*       Wn     [[buffer(2)]],                        \
                 device float*   dcoef  [[buffer(3)]],                        \
                 constant uint& C_out   [[buffer(4)]],                        \
                 constant uint& khw     [[buffer(5)]],                        \
                 constant uint& wk      [[buffer(6)]],                        \
                 constant int&  demod   [[buffer(7)]],                        \
                 constant float& eps    [[buffer(8)]],                        \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    uint tid = tid3.x; uint tg = tgs3.x;                                      \
    threadgroup float s[MC_BLOCK]; threadgroup float s_d;                     \
    uint o = gid.x;                                                           \
    device const T* Wo = W + (ulong)o * wk;                                   \
    device T* Wno = Wn + (ulong)o * wk;                                       \
    float local = 0.0f;                                                       \
    for (uint col = tid; col < wk; col += tg) {                               \
        uint i = col / khw;                                                   \
        float wp = float(Wo[col]) * float(sn[i]);                             \
        Wno[col] = T(wp); local += wp * wp;                                   \
    }                                                                         \
    s[tid] = local;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint d = tg / 2; d > 0; d >>= 1) {                                   \
        if (tid < d) s[tid] += s[tid + d];                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) s_d = demod ? rsqrt(s[0] + eps) : 1.0f;                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    float dv = s_d;                                                           \
    if (tid == 0) dcoef[o] = dv;                                              \
    if (demod) for (uint col = tid; col < wk; col += tg)                      \
        Wno[col] = T(float(Wno[col]) * dv);                                   \
}

// Backward: rebuild w' (Wpr) and w'' (Wpp) from W, s, and the cached dcoef.
#define BUILD_WPR_WPP(NAME, T)                                                \
kernel void NAME(device const T* W      [[buffer(0)]],                        \
                 device const T* sn     [[buffer(1)]],                        \
                 device const float* dcn[[buffer(2)]],                        \
                 device T*       Wpr    [[buffer(3)]],                        \
                 device T*       Wpp    [[buffer(4)]],                        \
                 constant uint& khw     [[buffer(5)]],                        \
                 constant uint& wk      [[buffer(6)]],                        \
                 constant uint& total   [[buffer(7)]],                        \
                 uint idx [[thread_position_in_grid]]) {                      \
    if (idx >= total) return;                                                 \
    uint o = idx / wk; uint col = idx % wk; uint i = col / khw;               \
    float wp = float(W[idx]) * float(sn[i]);                                  \
    Wpr[idx] = T(wp); Wpp[idx] = T(wp * dcn[o]);                              \
}

// Backward: dw'' (=dWpp) through demod → dw' (dWpr). One threadgroup per o.
#define DEMOD_DWPR(NAME, T)                                                   \
kernel void NAME(device const T* dWpp   [[buffer(0)]],                        \
                 device const T* Wpr    [[buffer(1)]],                        \
                 device const float* dcn[[buffer(2)]],                        \
                 device T*       dWpr   [[buffer(3)]],                        \
                 constant uint& C_out   [[buffer(4)]],                        \
                 constant uint& wk      [[buffer(5)]],                        \
                 constant int&  demod   [[buffer(6)]],                        \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    uint tid = tid3.x; uint tg = tgs3.x;                                      \
    threadgroup float s[MC_BLOCK]; threadgroup float s_g;                     \
    uint o = gid.x; ulong ob = (ulong)o * wk;                                 \
    if (!demod) {                                                             \
        for (uint col = tid; col < wk; col += tg) dWpr[ob + col] = dWpp[ob + col]; \
        return;                                                               \
    }                                                                         \
    float local = 0.0f;                                                       \
    for (uint col = tid; col < wk; col += tg)                                 \
        local += float(dWpp[ob + col]) * float(Wpr[ob + col]);               \
    s[tid] = local;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint d = tg / 2; d > 0; d >>= 1) {                                   \
        if (tid < d) s[tid] += s[tid + d];                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) s_g = s[0];                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    float d = dcn[o]; float gd3 = s_g * d * d * d;                            \
    for (uint col = tid; col < wk; col += tg)                                 \
        dWpr[ob + col] = T(float(dWpp[ob + col]) * d - gd3 * float(Wpr[ob + col])); \
}

// Backward: dW_f32[idx] += dw'[idx] * s[i]   (FP32 accumulate across the batch).
#define ACCUM_DW(NAME, T)                                                     \
kernel void NAME(device const T* dWpr   [[buffer(0)]],                        \
                 device const T* sn     [[buffer(1)]],                        \
                 device float*   dW_f32 [[buffer(2)]],                        \
                 constant uint& khw     [[buffer(3)]],                        \
                 constant uint& wk      [[buffer(4)]],                        \
                 constant uint& total   [[buffer(5)]],                        \
                 uint idx [[thread_position_in_grid]]) {                      \
    if (idx >= total) return;                                                 \
    uint col = idx % wk; uint i = col / khw;                                  \
    dW_f32[idx] += float(dWpr[idx]) * float(sn[i]);                           \
}

// Backward: ds[i] = Σ_{o,t} dw'[o,i*khw+t] * W[o,i*khw+t]. One threadgroup per i.
#define DS_KERNEL(NAME, T)                                                    \
kernel void NAME(device const T* dWpr   [[buffer(0)]],                        \
                 device const T* W      [[buffer(1)]],                        \
                 device T*       dsn    [[buffer(2)]],                        \
                 constant uint& C_out   [[buffer(3)]],                        \
                 constant uint& C_in    [[buffer(4)]],                        \
                 constant uint& khw     [[buffer(5)]],                        \
                 constant uint& wk      [[buffer(6)]],                        \
                 uint3 gid  [[threadgroup_position_in_grid]],                 \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    uint tid = tid3.x; uint tg = tgs3.x;                                      \
    threadgroup float s[MC_BLOCK];                                            \
    uint i = gid.x;                                                           \
    uint inner = C_out * khw;                                                 \
    float local = 0.0f;                                                       \
    for (uint k = tid; k < inner; k += tg) {                                  \
        uint o = k / khw; uint t = k % khw;                                   \
        ulong col = (ulong)o * wk + (ulong)i * khw + t;                       \
        local += float(dWpr[col]) * float(W[col]);                           \
    }                                                                         \
    s[tid] = local;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint d = tg / 2; d > 0; d >>= 1) {                                   \
        if (tid < d) s[tid] += s[tid + d];                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    if (tid == 0) dsn[i] = T(s[0]);                                           \
}

// Merge FP32 dW accumulator into the caller's dW (accumulate — caller zeros).
#define MERGE_DW(NAME, T)                                                     \
kernel void NAME(device const float* dW_f32 [[buffer(0)]],                    \
                 device T*       dW    [[buffer(1)]],                         \
                 constant uint& total  [[buffer(2)]],                         \
                 uint idx [[thread_position_in_grid]]) {                      \
    if (idx >= total) return;                                                 \
    dW[idx] = T(float(dW[idx]) + dW_f32[idx]);                                \
}

MODULATE_BUILD(k_mc_build_fp32, float) MODULATE_BUILD(k_mc_build_fp16, half) MODULATE_BUILD(k_mc_build_bf16, bfloat)
BUILD_WPR_WPP(k_mc_wprwpp_fp32, float) BUILD_WPR_WPP(k_mc_wprwpp_fp16, half) BUILD_WPR_WPP(k_mc_wprwpp_bf16, bfloat)
DEMOD_DWPR(k_mc_demod_fp32, float) DEMOD_DWPR(k_mc_demod_fp16, half) DEMOD_DWPR(k_mc_demod_bf16, bfloat)
ACCUM_DW(k_mc_accdw_fp32, float) ACCUM_DW(k_mc_accdw_fp16, half) ACCUM_DW(k_mc_accdw_bf16, bfloat)
DS_KERNEL(k_mc_ds_fp32, float) DS_KERNEL(k_mc_ds_fp16, half) DS_KERNEL(k_mc_ds_bf16, bfloat)
MERGE_DW(k_mc_merge_fp32, float) MERGE_DW(k_mc_merge_fp16, half) MERGE_DW(k_mc_merge_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_build_fp32, @"k_mc_build_fp32") DEF_PSO(pso_build_fp16, @"k_mc_build_fp16") DEF_PSO(pso_build_bf16, @"k_mc_build_bf16")
DEF_PSO(pso_wprwpp_fp32, @"k_mc_wprwpp_fp32") DEF_PSO(pso_wprwpp_fp16, @"k_mc_wprwpp_fp16") DEF_PSO(pso_wprwpp_bf16, @"k_mc_wprwpp_bf16")
DEF_PSO(pso_demod_fp32, @"k_mc_demod_fp32") DEF_PSO(pso_demod_fp16, @"k_mc_demod_fp16") DEF_PSO(pso_demod_bf16, @"k_mc_demod_bf16")
DEF_PSO(pso_accdw_fp32, @"k_mc_accdw_fp32") DEF_PSO(pso_accdw_fp16, @"k_mc_accdw_fp16") DEF_PSO(pso_accdw_bf16, @"k_mc_accdw_bf16")
DEF_PSO(pso_ds_fp32, @"k_mc_ds_fp32") DEF_PSO(pso_ds_fp16, @"k_mc_ds_fp16") DEF_PSO(pso_ds_bf16, @"k_mc_ds_bf16")
DEF_PSO(pso_merge_fp32, @"k_mc_merge_fp32") DEF_PSO(pso_merge_fp16, @"k_mc_merge_fp16") DEF_PSO(pso_merge_bf16, @"k_mc_merge_bf16")
#undef DEF_PSO

void require_fp(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16)
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
}

id<MTLComputePipelineState> pick(Dtype dt, id<MTLComputePipelineState> f32,
                                 id<MTLComputePipelineState> f16,
                                 id<MTLComputePipelineState> bf16) {
    return (dt == Dtype::FP16) ? f16 : (dt == Dtype::BF16) ? bf16 : f32;
}

// Byte offset of element-`n*cols` into a tensor's backing buffer.
NSUInteger row_off(const Tensor& t, int n, int cols) {
    return buffer_offset_for(t) +
           static_cast<NSUInteger>(static_cast<size_t>(n) * cols *
                                   dtype_size_bytes(t.dtype));
}

// Byte-correct row view honouring the tensor's element size (for conv2d reuse).
Tensor row_view(const Tensor& T, int n, int cols) {
    const size_t elt = static_cast<size_t>(dtype_size_bytes(T.dtype));
    char* base = static_cast<char*>(T.data) + static_cast<size_t>(n) * cols * elt;
    return Tensor::view(Device::Metal, base, 1, cols, T.dtype);
}

} // namespace

void modulated_conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW, int pad_h, int pad_w,
                              bool demodulate, float eps,
                              Tensor& dcoef, Tensor& Y) {
    require_fp(X, "modulated_conv2d_forward", "X");
    require_fp(W, "modulated_conv2d_forward", "W");
    require_fp(s, "modulated_conv2d_forward", "s");
    if (W.dtype != X.dtype || s.dtype != X.dtype)
        throw std::runtime_error("modulated_conv2d_forward: W/s dtype must match X");
    const int wk = C_in * kH * kW;
    const int khw = kH * kW;
    if (X.rows != N || X.cols != C_in * H * Wd)
        throw std::runtime_error("modulated_conv2d_forward: X shape mismatch");
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_forward: W shape mismatch");
    if (s.rows != N || s.cols != C_in)
        throw std::runtime_error("modulated_conv2d_forward: s shape mismatch");
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("modulated_conv2d_forward: non-positive output shape");
    if (dcoef.rows != N || dcoef.cols != C_out || dcoef.dtype != Dtype::FP32)
        dcoef.resize(N, C_out, Dtype::FP32);
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype)
        Y.resize(N, out_cols, X.dtype);
    if (N == 0 || out_cols == 0) return;

    Tensor Wn = Tensor::zeros_on(Device::Metal, C_out, wk, X.dtype);
    id<MTLComputePipelineState> pso = pick(X.dtype, pso_build_fp32(), pso_build_fp16(), pso_build_bf16());
    const uint32_t Cou = C_out, khwu = khw, wku = wk;
    const int demodi = demodulate ? 1 : 0;
    const size_t se = dtype_size_bytes(s.dtype);

    for (int n = 0; n < N; ++n) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:buffer_for(W) offset:buffer_offset_for(W) atIndex:0];
            [enc setBuffer:buffer_for(s)
                    offset:buffer_offset_for(s) + (NSUInteger)((size_t)n * C_in * se)
                   atIndex:1];
            [enc setBuffer:buffer_for(Wn) offset:buffer_offset_for(Wn) atIndex:2];
            [enc setBuffer:buffer_for(dcoef)
                    offset:buffer_offset_for(dcoef) + (NSUInteger)((size_t)n * C_out * sizeof(float))
                   atIndex:3];
            [enc setBytes:&Cou  length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&khwu length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&wku  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&demodi length:sizeof(int) atIndex:7];
            [enc setBytes:&eps  length:sizeof(float) atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(C_out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MC_BLOCK, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        Tensor Xn = row_view(X, n, C_in * H * Wd);
        Tensor Yn = row_view(Y, n, out_cols);
        conv2d_forward(Xn, Wn, nullptr, 1, C_in, H, Wd, C_out, kH, kW,
                       1, 1, pad_h, pad_w, 1, 1, 1, Yn);
    }
}

void modulated_conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& s,
                               const Tensor& dcoef, const Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW, int pad_h, int pad_w,
                               bool demodulate, float eps,
                               Tensor& dX, Tensor& dW, Tensor& ds) {
    require_fp(X, "modulated_conv2d_backward", "X");
    require_fp(W, "modulated_conv2d_backward", "W");
    require_fp(s, "modulated_conv2d_backward", "s");
    require_fp(dY, "modulated_conv2d_backward", "dY");
    if (W.dtype != X.dtype || s.dtype != X.dtype || dY.dtype != X.dtype || dW.dtype != X.dtype)
        throw std::runtime_error("modulated_conv2d_backward: W/s/dY/dW dtype must match X");
    if (dcoef.dtype != Dtype::FP32)
        throw std::runtime_error("modulated_conv2d_backward: dcoef must be FP32");
    (void)eps;  // demod coefficient is precomputed (passed in as dcoef)
    const int wk = C_in * kH * kW;
    const int khw = kH * kW;
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: W shape mismatch");
    if (dW.rows != C_out || dW.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: dW shape mismatch");
    if (dX.rows != N || dX.cols != C_in * H * Wd || dX.dtype != X.dtype)
        dX.resize(N, C_in * H * Wd, X.dtype);
    if (ds.rows != N || ds.cols != C_in || ds.dtype != X.dtype)
        ds.resize(N, C_in, X.dtype);
    if (N == 0) return;
    const int out_cols = C_out * H_out * W_out;

    Tensor Wpr  = Tensor::zeros_on(Device::Metal, C_out, wk, X.dtype);
    Tensor Wpp  = Tensor::zeros_on(Device::Metal, C_out, wk, X.dtype);
    Tensor dWpp = Tensor::zeros_on(Device::Metal, C_out, wk, X.dtype);
    Tensor dWpr = Tensor::zeros_on(Device::Metal, C_out, wk, X.dtype);
    Tensor dW_f32 = Tensor::zeros_on(Device::Metal, C_out, wk, Dtype::FP32);
    const uint32_t wtotal = static_cast<uint32_t>(C_out) * wk;
    const uint32_t Cou = C_out, Cinu = C_in, khwu = khw, wku = wk;
    const int demodi = demodulate ? 1 : 0;
    const size_t se = dtype_size_bytes(s.dtype);

    id<MTLComputePipelineState> pso_wprwpp = pick(X.dtype, pso_wprwpp_fp32(), pso_wprwpp_fp16(), pso_wprwpp_bf16());
    id<MTLComputePipelineState> pso_demod  = pick(X.dtype, pso_demod_fp32(), pso_demod_fp16(), pso_demod_bf16());
    id<MTLComputePipelineState> pso_accdw  = pick(X.dtype, pso_accdw_fp32(), pso_accdw_fp16(), pso_accdw_bf16());
    id<MTLComputePipelineState> pso_ds     = pick(X.dtype, pso_ds_fp32(), pso_ds_fp16(), pso_ds_bf16());

    for (int n = 0; n < N; ++n) {
        const NSUInteger s_off  = buffer_offset_for(s) + (NSUInteger)((size_t)n * C_in * se);
        const NSUInteger dc_off = buffer_offset_for(dcoef) + (NSUInteger)((size_t)n * C_out * sizeof(float));

        // Rebuild w' (Wpr) and w'' (Wpp) from W, s, cached dcoef.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_wprwpp];
            [enc setBuffer:buffer_for(W) offset:buffer_offset_for(W) atIndex:0];
            [enc setBuffer:buffer_for(s) offset:s_off atIndex:1];
            [enc setBuffer:buffer_for(dcoef) offset:dc_off atIndex:2];
            [enc setBuffer:buffer_for(Wpr) offset:buffer_offset_for(Wpr) atIndex:3];
            [enc setBuffer:buffer_for(Wpp) offset:buffer_offset_for(Wpp) atIndex:4];
            [enc setBytes:&khwu length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&wku  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&wtotal length:sizeof(uint32_t) atIndex:7];
            NSUInteger tpt = [pso_wprwpp maxTotalThreadsPerThreadgroup];
            if (tpt > 256) tpt = 256;
            [enc dispatchThreads:MTLSizeMake(wtotal, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }

        Tensor Xn  = row_view(X, n, C_in * H * Wd);
        Tensor dYn = row_view(dY, n, out_cols);

        dWpp.zero();  // conv2d_backward_weight accumulates
        conv2d_backward_weight(Xn, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                               1, 1, pad_h, pad_w, 1, 1, 1, dWpp);
        Tensor dXn = row_view(dX, n, C_in * H * Wd);
        conv2d_backward_input(Wpp, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                              1, 1, pad_h, pad_w, 1, 1, 1, dXn);

        // dw'' → dw' through demod (one threadgroup per output channel o).
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_demod];
            [enc setBuffer:buffer_for(dWpp) offset:buffer_offset_for(dWpp) atIndex:0];
            [enc setBuffer:buffer_for(Wpr)  offset:buffer_offset_for(Wpr)  atIndex:1];
            [enc setBuffer:buffer_for(dcoef) offset:dc_off atIndex:2];
            [enc setBuffer:buffer_for(dWpr) offset:buffer_offset_for(dWpr) atIndex:3];
            [enc setBytes:&Cou length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&wku length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&demodi length:sizeof(int) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(C_out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MC_BLOCK, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }

        // dW_f32 += dw' * s   (FP32 accumulate across the batch).
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_accdw];
            [enc setBuffer:buffer_for(dWpr) offset:buffer_offset_for(dWpr) atIndex:0];
            [enc setBuffer:buffer_for(s) offset:s_off atIndex:1];
            [enc setBuffer:buffer_for(dW_f32) offset:buffer_offset_for(dW_f32) atIndex:2];
            [enc setBytes:&khwu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&wku  length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&wtotal length:sizeof(uint32_t) atIndex:5];
            NSUInteger tpt = [pso_accdw maxTotalThreadsPerThreadgroup];
            if (tpt > 256) tpt = 256;
            [enc dispatchThreads:MTLSizeMake(wtotal, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }

        // ds[i] = Σ_{o,t} dw'[o,i,t] * W[o,i,t]  (one threadgroup per input channel i).
        const NSUInteger ds_off = buffer_offset_for(ds) + (NSUInteger)((size_t)n * C_in * se);
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_ds];
            [enc setBuffer:buffer_for(dWpr) offset:buffer_offset_for(dWpr) atIndex:0];
            [enc setBuffer:buffer_for(W) offset:buffer_offset_for(W) atIndex:1];
            [enc setBuffer:buffer_for(ds) offset:ds_off atIndex:2];
            [enc setBytes:&Cou  length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Cinu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&khwu length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&wku  length:sizeof(uint32_t) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(C_in, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MC_BLOCK, 1, 1)];
            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
    }

    // Merge the FP32 dW accumulator into the caller's dW (accumulate).
    id<MTLComputePipelineState> pso_merge = pick(X.dtype, pso_merge_fp32(), pso_merge_fp16(), pso_merge_bf16());
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_merge];
        [enc setBuffer:buffer_for(dW_f32) offset:buffer_offset_for(dW_f32) atIndex:0];
        [enc setBuffer:buffer_for(dW) offset:buffer_offset_for(dW) atIndex:1];
        [enc setBytes:&wtotal length:sizeof(uint32_t) atIndex:2];
        NSUInteger tpt = [pso_merge maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(wtotal, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
