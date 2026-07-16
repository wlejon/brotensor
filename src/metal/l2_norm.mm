// Per-head L2 normalisation (Metal). Mirrors src/cuda/l2_norm.cu.
//
// Layout: (L, num_heads * head_dim), row-major. Head h occupies columns
// [h*head_dim, (h+1)*head_dim) on each row, same convention as rope/rms_norm.
//
// Supports FP32 / FP16 / BF16 — math in FP32, loads/stores cast at the
// boundary, matching CUDA (CPU stays FP32-only).
//
// One threadgroup per (row, head). 256 threads per group with a power-of-two
// shared-memory reduction; head_dim < 256 is fine because the over-range
// threads contribute zero to the sumsq / dot.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger L2_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint L2_BLOCK = 256;

#define L2_NORM_FW(NAME, T)                                                   \
kernel void NAME(device const T*     X     [[buffer(0)]],                     \
                 device T*           Y     [[buffer(1)]],                     \
                 constant uint& num_heads  [[buffer(2)]],                     \
                 constant uint& head_dim   [[buffer(3)]],                     \
                 constant float& eps       [[buffer(4)]],                     \
                 uint3 gid [[threadgroup_position_in_grid]],                  \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    threadgroup float sdata[L2_BLOCK];                                        \
    uint h = gid.x;                                                           \
    uint r = gid.y;                                                           \
    uint tid = tid3.x;                                                        \
    uint tgs = tgs3.x;                                                        \
    uint D = num_heads * head_dim;                                            \
    uint off = r * D + h * head_dim;                                          \
    float local = 0.0f;                                                       \
    for (uint d = tid; d < head_dim; d += tgs) {                              \
        float v = float(X[off + d]);                                          \
        local += v * v;                                                       \
    }                                                                         \
    sdata[tid] = local;                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tgs / 2; s > 0; s >>= 1) {                                  \
        if (tid < s) sdata[tid] += sdata[tid + s];                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    float inv = rsqrt(sdata[0] + eps);                                        \
    for (uint d = tid; d < head_dim; d += tgs) {                              \
        Y[off + d] = T(float(X[off + d]) * inv);                              \
    }                                                                         \
}

#define L2_NORM_BW(NAME, T)                                                   \
kernel void NAME(device const T*     X     [[buffer(0)]],                     \
                 device const T*     dY    [[buffer(1)]],                     \
                 device T*           dX    [[buffer(2)]],                     \
                 constant uint& num_heads  [[buffer(3)]],                     \
                 constant uint& head_dim   [[buffer(4)]],                     \
                 constant float& eps       [[buffer(5)]],                     \
                 uint3 gid [[threadgroup_position_in_grid]],                  \
                 uint3 tid3 [[thread_position_in_threadgroup]],               \
                 uint3 tgs3 [[threads_per_threadgroup]]) {                    \
    threadgroup float ssum[L2_BLOCK];                                         \
    threadgroup float sdot[L2_BLOCK];                                         \
    uint h = gid.x;                                                           \
    uint r = gid.y;                                                           \
    uint tid = tid3.x;                                                        \
    uint tgs = tgs3.x;                                                        \
    uint D = num_heads * head_dim;                                            \
    uint off = r * D + h * head_dim;                                          \
    float ls = 0.0f, ld = 0.0f;                                               \
    for (uint d = tid; d < head_dim; d += tgs) {                              \
        float v = float(X[off + d]);                                          \
        float g = float(dY[off + d]);                                         \
        ls += v * v;                                                          \
        ld += v * g;                                                          \
    }                                                                         \
    ssum[tid] = ls;                                                           \
    sdot[tid] = ld;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tgs / 2; s > 0; s >>= 1) {                                  \
        if (tid < s) {                                                        \
            ssum[tid] += ssum[tid + s];                                       \
            sdot[tid] += sdot[tid + s];                                       \
        }                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    float n2 = 1.0f / (ssum[0] + eps);                                        \
    float n  = sqrt(n2);                                                      \
    float c  = sdot[0] * n2;                                                  \
    for (uint d = tid; d < head_dim; d += tgs) {                              \
        dX[off + d] = T(n * (float(dY[off + d]) - float(X[off + d]) * c));    \
    }                                                                         \
}

L2_NORM_FW(k_l2_norm_fw_fp32, float)
L2_NORM_FW(k_l2_norm_fw_fp16, half)
L2_NORM_FW(k_l2_norm_fw_bf16, bfloat)
L2_NORM_BW(k_l2_norm_bw_fp32, float)
L2_NORM_BW(k_l2_norm_bw_fp16, half)
L2_NORM_BW(k_l2_norm_bw_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_fw_fp32, @"k_l2_norm_fw_fp32") DEF_PSO(pso_fw_fp16, @"k_l2_norm_fw_fp16") DEF_PSO(pso_fw_bf16, @"k_l2_norm_fw_bf16")
DEF_PSO(pso_bw_fp32, @"k_l2_norm_bw_fp32") DEF_PSO(pso_bw_fp16, @"k_l2_norm_bw_fp16") DEF_PSO(pso_bw_bf16, @"k_l2_norm_bw_bf16")
#undef DEF_PSO

id<MTLComputePipelineState> pick(const Tensor& t,
                                 id<MTLComputePipelineState> fp32,
                                 id<MTLComputePipelineState> fp16,
                                 id<MTLComputePipelineState> bf16) {
    return (t.dtype == Dtype::FP16) ? fp16
         : (t.dtype == Dtype::BF16) ? bf16 : fp32;
}

void check_fp(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != Dtype::FP32 && t.dtype != Dtype::FP16 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

void check_shape(const Tensor& t, int head_dim, int num_heads,
                 const char* op, const char* name) {
    if (head_dim <= 0)  throw std::runtime_error(std::string(op) + ": head_dim must be positive");
    if (num_heads <= 0) throw std::runtime_error(std::string(op) + ": num_heads must be positive");
    if (t.cols != num_heads * head_dim) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 ".cols != num_heads * head_dim");
    }
}

} // namespace

void l2_norm_forward(const Tensor& X, int head_dim, int num_heads, float eps,
                     Tensor& Y) {
    check_fp(X, "l2_norm_forward", "X");
    check_shape(X, head_dim, num_heads, "l2_norm_forward", "X");
    const int L = X.rows;
    const int D = X.cols;
    if (Y.rows != L || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oY = buffer_offset_for(Y);
    const uint32_t Hu = static_cast<uint32_t>(num_heads);
    const uint32_t Du = static_cast<uint32_t>(head_dim);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pick(X, pso_fw_fp32(), pso_fw_fp16(), pso_fw_bf16())];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float)   atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads, L, 1)
            threadsPerThreadgroup:MTLSizeMake(L2_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void l2_norm_backward(const Tensor& X, int head_dim, int num_heads, float eps,
                      const Tensor& dY, Tensor& dX) {
    check_fp(X,  "l2_norm_backward", "X");
    check_fp(dY, "l2_norm_backward", "dY");
    if (dY.dtype != X.dtype) {
        throw std::runtime_error("l2_norm_backward: dY.dtype must match X.dtype");
    }
    check_shape(X,  head_dim, num_heads, "l2_norm_backward", "X");
    check_shape(dY, head_dim, num_heads, "l2_norm_backward", "dY");
    if (dY.rows != X.rows) {
        throw std::runtime_error("l2_norm_backward: dY.rows != X.rows");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (dX.rows != L || dX.cols != D || dX.dtype != X.dtype) {
        dX.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    id<MTLBuffer> bX  = buffer_for(X);
    id<MTLBuffer> bdY = buffer_for(dY);
    id<MTLBuffer> bdX = buffer_for(dX);
    const NSUInteger oX  = buffer_offset_for(X);
    const NSUInteger odY = buffer_offset_for(dY);
    const NSUInteger odX = buffer_offset_for(dX);
    const uint32_t Hu = static_cast<uint32_t>(num_heads);
    const uint32_t Du = static_cast<uint32_t>(head_dim);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pick(X, pso_bw_fp32(), pso_bw_fp16(), pso_bw_bf16())];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bdY offset:odY atIndex:1];
        [enc setBuffer:bdX offset:odX atIndex:2];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float)   atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads, L, 1)
            threadsPerThreadgroup:MTLSizeMake(L2_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
