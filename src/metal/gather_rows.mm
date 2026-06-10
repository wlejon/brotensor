// ─── Metal row gather / scatter-add ─────────────────────────────────────────
//
// Metal counterpart of src/cpu/gather_rows.cpp. FP32-only. Same shape contract:
//   X        : (R, C) FP32
//   Idx      : (M, 1) INT32
//   Y/dY     : (M, C) FP32
//   dX       : (R, C) FP32
//
//   gather_rows       — Y OVERWRITTEN (one thread per output element).
//   scatter_rows_add  — dX OVERWRITTEN (zero kernel then atomic scatter-add).
//                       Duplicate Idx values accumulate atomically.

#include <brotensor/runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32");
    }
}

void req_idx(const char* op, const Tensor& Idx) {
    if (Idx.dtype != Dtype::INT32) fail(op, "Idx must be INT32");
    if (Idx.cols != 1)             fail(op, "Idx must be shaped (M, 1)");
}

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_gather_rows(device const float* X   [[buffer(0)]],
                          device const int*   Idx [[buffer(1)]],
                          device float*       Y   [[buffer(2)]],
                          constant uint& M        [[buffer(3)]],
                          constant uint& C        [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    uint total = M * C;
    if (gid >= total) return;
    uint m = gid / C;
    uint j = gid - m * C;
    int r = Idx[m];
    Y[gid] = X[uint(r) * C + j];
}

kernel void k_zero_f32(device float* p [[buffer(0)]],
                       constant uint& n [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    p[gid] = 0.0f;
}

kernel void k_scatter_rows(device const float* Y   [[buffer(0)]],
                           device const int*   Idx [[buffer(1)]],
                           device float*       X   [[buffer(2)]],
                           constant uint& M        [[buffer(3)]],
                           constant uint& C        [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    uint total = M * C;
    if (gid >= total) return;
    uint m = gid / C;
    uint j = gid - m * C;
    int r = Idx[m];
    X[uint(r) * C + j] = Y[gid];
}

kernel void k_scatter_rows_add(device const float*  dY  [[buffer(0)]],
                               device const int*    Idx [[buffer(1)]],
                               device atomic_float* dX  [[buffer(2)]],
                               constant uint& M         [[buffer(3)]],
                               constant uint& C         [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    uint total = M * C;
    if (gid >= total) return;
    uint m = gid / C;
    uint j = gid - m * C;
    int r = Idx[m];
    atomic_fetch_add_explicit(&dX[uint(r) * C + j], dY[gid],
                              memory_order_relaxed);
}
)msl";

id<MTLComputePipelineState> pso_gather() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_gather_rows"); });
    return pso;
}
id<MTLComputePipelineState> pso_zero() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_zero_f32"); });
    return pso;
}
id<MTLComputePipelineState> pso_scatter() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_scatter_rows_add"); });
    return pso;
}
id<MTLComputePipelineState> pso_scatter_overwrite() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_scatter_rows"); });
    return pso;
}

void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger total,
                void (^binders)(id<MTLComputeCommandEncoder>)) {
    if (total == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        binders(enc);
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void gather_rows(const Tensor& X,
                 const Tensor& Idx,
                 Tensor& Y) {
    const char* op = "gather_rows";
    req_fp32(op, X, "X");
    req_idx(op, Idx);
    const int M = Idx.rows;
    const int C = X.cols;
    if (Y.rows != M || Y.cols != C || Y.dtype != Dtype::FP32) {
        Y.resize(M, C, Dtype::FP32);
    }
    if (M == 0 || C == 0) return;

    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const NSUInteger total = static_cast<NSUInteger>(M) * static_cast<NSUInteger>(C);

    dispatch1d(pso_gather(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)   offset:buffer_offset_for(X)   atIndex:0];
        [enc setBuffer:buffer_for(Idx) offset:buffer_offset_for(Idx) atIndex:1];
        [enc setBuffer:buffer_for(Y)   offset:buffer_offset_for(Y)   atIndex:2];
        [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
    });
}

void scatter_rows_add(const Tensor& dY,
                      const Tensor& Idx, int R,
                      Tensor& dX) {
    const char* op = "scatter_rows_add";
    req_fp32(op, dY, "dY");
    req_idx(op, Idx);
    if (R < 0) fail(op, "R must be >= 0");
    const int M = Idx.rows;
    if (dY.rows != M) fail(op, "dY.rows must equal Idx.rows");
    const int C = dY.cols;
    if (dX.rows != R || dX.cols != C || dX.dtype != Dtype::FP32) {
        dX.resize(R, C, Dtype::FP32);
    }
    if (R == 0 || C == 0) return;

    const uint32_t total_dx = static_cast<uint32_t>(R) * static_cast<uint32_t>(C);
    // Zero dX first.
    dispatch1d(pso_zero(), total_dx, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:0];
        [enc setBytes:&total_dx length:sizeof(uint32_t) atIndex:1];
    });

    if (M == 0) return;

    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const NSUInteger total = static_cast<NSUInteger>(M) * static_cast<NSUInteger>(C);

    dispatch1d(pso_scatter(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY)  offset:buffer_offset_for(dY)  atIndex:0];
        [enc setBuffer:buffer_for(Idx) offset:buffer_offset_for(Idx) atIndex:1];
        [enc setBuffer:buffer_for(dX)  offset:buffer_offset_for(dX)  atIndex:2];
        [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
    });
}

void scatter_rows(const Tensor& Y,
                  const Tensor& Idx,
                  Tensor& X) {
    const char* op = "scatter_rows";
    req_fp32(op, Y, "Y");
    req_idx(op, Idx);
    const int M = Idx.rows;
    if (Y.rows != M) fail(op, "Y.rows must equal Idx.rows");
    if (X.dtype != Dtype::FP32) fail(op, "X must be FP32");
    if (X.cols != Y.cols) fail(op, "X.cols must equal Y.cols");
    const int C = Y.cols;
    if (M == 0 || C == 0) return;

    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const NSUInteger total = static_cast<NSUInteger>(M) * static_cast<NSUInteger>(C);

    dispatch1d(pso_scatter_overwrite(), total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(Y)   offset:buffer_offset_for(Y)   atIndex:0];
        [enc setBuffer:buffer_for(Idx) offset:buffer_offset_for(Idx) atIndex:1];
        [enc setBuffer:buffer_for(X)   offset:buffer_offset_for(X)   atIndex:2];
        [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
    });
}

} // namespace brotensor::detail::metal
