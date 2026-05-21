// Public reductions (Metal): sum_rows, sum_cols, argmax_rows.
// FP32 + FP16 input. argmax always writes FP32 indices.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger RED_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint RED_BLOCK = 256;

kernel void k_sum_rows_fp32(device const float* X [[buffer(0)]],
                            device float*       Y [[buffer(1)]],
                            constant uint& M [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint m   [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sm[RED_BLOCK];
    if (m >= M) return;
    float acc = 0.0f;
    for (uint n = tid; n < N; n += tgs) acc += X[m * N + n];
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) Y[m] = sm[0];
}

kernel void k_sum_rows_fp16(device const half* X [[buffer(0)]],
                            device half*       Y [[buffer(1)]],
                            constant uint& M [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint m   [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sm[RED_BLOCK];
    if (m >= M) return;
    float acc = 0.0f;
    for (uint n = tid; n < N; n += tgs) acc += float(X[m * N + n]);
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) Y[m] = half(sm[0]);
}

kernel void k_sum_cols_fp32(device const float* X [[buffer(0)]],
                            device float*       Y [[buffer(1)]],
                            constant uint& M [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float acc = 0.0f;
    for (uint m = 0; m < M; ++m) acc += X[m * N + gid];
    Y[gid] = acc;
}

kernel void k_sum_cols_fp16(device const half* X [[buffer(0)]],
                            device half*       Y [[buffer(1)]],
                            constant uint& M [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float acc = 0.0f;
    for (uint m = 0; m < M; ++m) acc += float(X[m * N + gid]);
    Y[gid] = half(acc);
}

kernel void k_argmax_rows_fp32(device const float* X [[buffer(0)]],
                               device float*       Idx [[buffer(1)]],
                               constant uint& M [[buffer(2)]],
                               constant uint& N [[buffer(3)]],
                               uint m   [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sm_val[RED_BLOCK];
    threadgroup int   sm_idx[RED_BLOCK];
    if (m >= M) return;
    float best_v = -3.4028235e38f;
    int   best_i = 0;
    for (uint n = tid; n < N; n += tgs) {
        float v = X[m * N + n];
        if (v > best_v) { best_v = v; best_i = int(n); }
    }
    sm_val[tid] = best_v;
    sm_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm_val[tid + s] > sm_val[tid]) {
                sm_val[tid] = sm_val[tid + s];
                sm_idx[tid] = sm_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) Idx[m] = float(sm_idx[0]);
}

kernel void k_argmax_rows_fp16(device const half* X [[buffer(0)]],
                               device float*      Idx [[buffer(1)]],
                               constant uint& M [[buffer(2)]],
                               constant uint& N [[buffer(3)]],
                               uint m   [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sm_val[RED_BLOCK];
    threadgroup int   sm_idx[RED_BLOCK];
    if (m >= M) return;
    float best_v = -3.4028235e38f;
    int   best_i = 0;
    for (uint n = tid; n < N; n += tgs) {
        float v = float(X[m * N + n]);
        if (v > best_v) { best_v = v; best_i = int(n); }
    }
    sm_val[tid] = best_v;
    sm_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm_val[tid + s] > sm_val[tid]) {
                sm_val[tid] = sm_val[tid + s];
                sm_idx[tid] = sm_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) Idx[m] = float(sm_idx[0]);
}

kernel void k_sum_rows_bf16(device const bfloat* X [[buffer(0)]],
                            device bfloat*       Y [[buffer(1)]],
                            constant uint& M [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint m   [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sm[RED_BLOCK];
    if (m >= M) return;
    float acc = 0.0f;
    for (uint n = tid; n < N; n += tgs) acc += float(X[m * N + n]);
    sm[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) Y[m] = bfloat(sm[0]);
}

kernel void k_sum_cols_bf16(device const bfloat* X [[buffer(0)]],
                            device bfloat*       Y [[buffer(1)]],
                            constant uint& M [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    float acc = 0.0f;
    for (uint m = 0; m < M; ++m) acc += float(X[m * N + gid]);
    Y[gid] = bfloat(acc);
}

kernel void k_argmax_rows_bf16(device const bfloat* X [[buffer(0)]],
                               device float*        Idx [[buffer(1)]],
                               constant uint& M [[buffer(2)]],
                               constant uint& N [[buffer(3)]],
                               uint m   [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tgs [[threads_per_threadgroup]]) {
    threadgroup float sm_val[RED_BLOCK];
    threadgroup int   sm_idx[RED_BLOCK];
    if (m >= M) return;
    float best_v = -3.4028235e38f;
    int   best_i = 0;
    for (uint n = tid; n < N; n += tgs) {
        float v = float(X[m * N + n]);
        if (v > best_v) { best_v = v; best_i = int(n); }
    }
    sm_val[tid] = best_v;
    sm_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm_val[tid + s] > sm_val[tid]) {
                sm_val[tid] = sm_val[tid + s];
                sm_idx[tid] = sm_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) Idx[m] = float(sm_idx[0]);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_sum_rows_fp32, @"k_sum_rows_fp32")
DEF_PSO(pso_sum_rows_fp16, @"k_sum_rows_fp16")
DEF_PSO(pso_sum_rows_bf16, @"k_sum_rows_bf16")
DEF_PSO(pso_sum_cols_fp32, @"k_sum_cols_fp32")
DEF_PSO(pso_sum_cols_fp16, @"k_sum_cols_fp16")
DEF_PSO(pso_sum_cols_bf16, @"k_sum_cols_bf16")
DEF_PSO(pso_argmax_fp32,   @"k_argmax_rows_fp32")
DEF_PSO(pso_argmax_fp16,   @"k_argmax_rows_fp16")
DEF_PSO(pso_argmax_bf16,   @"k_argmax_rows_bf16")
#undef DEF_PSO

void run_per_row(id<MTLComputePipelineState> pso, uint32_t M, uint32_t N,
                 id<MTLBuffer> bX, NSUInteger oX,
                 id<MTLBuffer> bY, NSUInteger oY) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(RED_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void run_per_col(id<MTLComputePipelineState> pso, uint32_t M, uint32_t N,
                 id<MTLBuffer> bX, NSUInteger oX,
                 id<MTLBuffer> bY, NSUInteger oY) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void sum_rows(const Tensor& X, Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("sum_rows_gpu: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != X.dtype) {
        Y.resize(M, 1, X.dtype);
    }
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_sum_rows_fp16()
      : (X.dtype == Dtype::BF16) ? pso_sum_rows_bf16()
      : pso_sum_rows_fp32();
    run_per_row(pso, M, N,
                buffer_for(X), buffer_offset_for(X),
                buffer_for(Y), buffer_offset_for(Y));
}

void sum_cols(const Tensor& X, Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("sum_cols_gpu: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != X.dtype) {
        Y.resize(1, N, X.dtype);
    }
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_sum_cols_fp16()
      : (X.dtype == Dtype::BF16) ? pso_sum_cols_bf16()
      : pso_sum_cols_fp32();
    run_per_col(pso, M, N,
                buffer_for(X), buffer_offset_for(X),
                buffer_for(Y), buffer_offset_for(Y));
}

void argmax_rows(const Tensor& X, Tensor& Idx) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("argmax_rows_gpu: X must be FP16, BF16, or FP32");
    }
    const int M = X.rows;
    const int N = X.cols;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != Dtype::FP32) {
        Idx.resize(M, 1, Dtype::FP32);
    }
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_argmax_fp16()
      : (X.dtype == Dtype::BF16) ? pso_argmax_bf16()
      : pso_argmax_fp32();
    run_per_row(pso, M, N,
                buffer_for(X), buffer_offset_for(X),
                buffer_for(Idx), buffer_offset_for(Idx));
}

} // namespace brotensor::detail::metal
