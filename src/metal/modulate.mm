// ─── Metal AdaLN modulation ops ────────────────────────────────────────────
//
// DiT / SD3 / Flux broadcast-affine primitives — the Metal counterpart of
// src/cuda/modulate.cu.
//   modulate:      Y[l, d] = X[l, d] * (1 + scale[d]) + shift[d]
//   broadcast_mul: Y[l, d] = X[l, d] * v[d]
// scale / shift / v are length-D vectors broadcast across every token row.
//
// Dispatched on X.dtype (FP32 / FP16 / BF16). The scale / shift / v operands
// must share X's dtype. All arithmetic is performed in FP32; only the storage
// loads / stores change type. Both ops fully overwrite their output.

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

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

#define MODULATE_KERNEL(NAME, T)                                              \
kernel void NAME(device const T* X     [[buffer(0)]],                         \
                 device const T* scale [[buffer(1)]],                         \
                 device const T* shift [[buffer(2)]],                         \
                 device T*       Y     [[buffer(3)]],                         \
                 constant uint&  total [[buffer(4)]],                         \
                 constant uint&  D     [[buffer(5)]],                         \
                 uint gid [[thread_position_in_grid]]) {                      \
    if (gid >= total) return;                                                 \
    uint d = gid % D;                                                         \
    Y[gid] = T(float(X[gid]) * (1.0f + float(scale[d])) + float(shift[d]));    \
}

#define BCAST_MUL_KERNEL(NAME, T)                                             \
kernel void NAME(device const T* X     [[buffer(0)]],                         \
                 device const T* v     [[buffer(1)]],                         \
                 device T*       Y     [[buffer(2)]],                         \
                 constant uint&  total [[buffer(3)]],                         \
                 constant uint&  D     [[buffer(4)]],                         \
                 uint gid [[thread_position_in_grid]]) {                      \
    if (gid >= total) return;                                                 \
    uint d = gid % D;                                                         \
    Y[gid] = T(float(X[gid]) * float(v[d]));                                  \
}

MODULATE_KERNEL(k_modulate_fp32, float)
MODULATE_KERNEL(k_modulate_fp16, half)
MODULATE_KERNEL(k_modulate_bf16, bfloat)
BCAST_MUL_KERNEL(k_broadcast_mul_fp32, float)
BCAST_MUL_KERNEL(k_broadcast_mul_fp16, half)
BCAST_MUL_KERNEL(k_broadcast_mul_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_modulate_fp32, @"k_modulate_fp32")
DEF_PSO(pso_modulate_fp16, @"k_modulate_fp16")
DEF_PSO(pso_modulate_bf16, @"k_modulate_bf16")
DEF_PSO(pso_bcast_mul_fp32, @"k_broadcast_mul_fp32")
DEF_PSO(pso_bcast_mul_fp16, @"k_broadcast_mul_fp16")
DEF_PSO(pso_bcast_mul_bf16, @"k_broadcast_mul_bf16")
#undef DEF_PSO

void check_same_dtype(const Tensor& X, const Tensor& t,
                      const char* op, const char* name) {
    if (t.dtype != X.dtype) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " dtype must match X");
    }
}

void check_dtype(const Tensor& X, const char* op) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string(op) +
                                 ": X must be FP32, FP16, or BF16");
    }
}

} // namespace

void modulate(const Tensor& X, const Tensor& scale,
              const Tensor& shift, Tensor& Y) {
    check_dtype(X, "modulate");
    check_same_dtype(X, scale, "modulate", "scale");
    check_same_dtype(X, shift, "modulate", "shift");
    const int L = X.rows;
    const int D = X.cols;
    if (scale.size() != D || shift.size() != D) {
        throw std::runtime_error("modulate: scale and shift must each have X.cols elements");
    }
    if (Y.rows != L || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(L, D, X.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(L) * D;
    if (total == 0) return;

    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_modulate_fp16()
      : (X.dtype == Dtype::BF16) ? pso_modulate_bf16()
      : pso_modulate_fp32();
    const uint32_t totalu = static_cast<uint32_t>(total);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X)     offset:buffer_offset_for(X)     atIndex:0];
        [enc setBuffer:buffer_for(scale) offset:buffer_offset_for(scale) atIndex:1];
        [enc setBuffer:buffer_for(shift) offset:buffer_offset_for(shift) atIndex:2];
        [enc setBuffer:buffer_for(Y)     offset:buffer_offset_for(Y)     atIndex:3];
        [enc setBytes:&totalu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:5];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void broadcast_mul(const Tensor& X, const Tensor& v, Tensor& Y) {
    check_dtype(X, "broadcast_mul");
    check_same_dtype(X, v, "broadcast_mul", "v");
    const int L = X.rows;
    const int D = X.cols;
    if (v.size() != D) {
        throw std::runtime_error("broadcast_mul: v must have X.cols elements");
    }
    if (Y.rows != L || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(L, D, X.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(L) * D;
    if (total == 0) return;

    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_bcast_mul_fp16()
      : (X.dtype == Dtype::BF16) ? pso_bcast_mul_bf16()
      : pso_bcast_mul_fp32();
    const uint32_t totalu = static_cast<uint32_t>(total);
    const uint32_t Du = static_cast<uint32_t>(D);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(v) offset:buffer_offset_for(v) atIndex:1];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:2];
        [enc setBytes:&totalu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:4];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
