#include <brotensor/runtime.h>

#include <cstring>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_emb_fw(device const float* table [[buffer(0)]],
                     device const int*   idx   [[buffer(1)]],
                     device float*       out   [[buffer(2)]],
                     constant uint& B          [[buffer(3)]],
                     constant uint& D          [[buffer(4)]],
                     uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    out[t] = table[uint(row) * D + j];
}

kernel void k_emb_fw_fp16(device const half* table [[buffer(0)]],
                          device const int*  idx   [[buffer(1)]],
                          device half*       out   [[buffer(2)]],
                          constant uint& B         [[buffer(3)]],
                          constant uint& D         [[buffer(4)]],
                          uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    out[t] = table[uint(row) * D + j];
}

kernel void k_emb_bw(device const float* dOut          [[buffer(0)]],
                     device const int*   idx           [[buffer(1)]],
                     device atomic_float* dTable        [[buffer(2)]],
                     constant uint& B                  [[buffer(3)]],
                     constant uint& D                  [[buffer(4)]],
                     uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    atomic_fetch_add_explicit(&dTable[uint(row) * D + j],
                              dOut[t], memory_order_relaxed);
}

// FP16 input: atomic-add into FP32 scratch (atomic on half is not portable).
kernel void k_emb_bw_fp16(device const half* dOut          [[buffer(0)]],
                          device const int*  idx           [[buffer(1)]],
                          device atomic_float* dTable_scratch [[buffer(2)]],
                          constant uint& B                 [[buffer(3)]],
                          constant uint& D                 [[buffer(4)]],
                          uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    atomic_fetch_add_explicit(&dTable_scratch[uint(row) * D + j],
                              float(dOut[t]), memory_order_relaxed);
}

kernel void k_emb_add_fp32_into_fp16(device const float* src [[buffer(0)]],
                                     device half*        dst [[buffer(1)]],
                                     constant uint& n        [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}

kernel void k_emb_fw_bf16(device const bfloat* table [[buffer(0)]],
                          device const int*    idx   [[buffer(1)]],
                          device bfloat*       out   [[buffer(2)]],
                          constant uint& B            [[buffer(3)]],
                          constant uint& D            [[buffer(4)]],
                          uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    out[t] = table[uint(row) * D + j];
}

// BF16 input: atomic-add into FP32 scratch (atomic on bfloat is not portable).
kernel void k_emb_bw_bf16(device const bfloat* dOut            [[buffer(0)]],
                          device const int*    idx             [[buffer(1)]],
                          device atomic_float* dTable_scratch  [[buffer(2)]],
                          constant uint& B                     [[buffer(3)]],
                          constant uint& D                     [[buffer(4)]],
                          uint t [[thread_position_in_grid]]) {
    uint total = B * D;
    if (t >= total) return;
    uint b = t / D;
    uint j = t - b * D;
    int row = idx[b];
    atomic_fetch_add_explicit(&dTable_scratch[uint(row) * D + j],
                              float(dOut[t]), memory_order_relaxed);
}

kernel void k_emb_add_fp32_into_bf16(device const float* src [[buffer(0)]],
                                     device bfloat*      dst [[buffer(1)]],
                                     constant uint& n        [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}
)msl";

id<MTLComputePipelineState> fw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_fw"); });
    return pso;
}
id<MTLComputePipelineState> fw_pso_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_fw_fp16"); });
    return pso;
}
id<MTLComputePipelineState> bw_pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_bw"); });
    return pso;
}
id<MTLComputePipelineState> bw_pso_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_bw_fp16"); });
    return pso;
}
id<MTLComputePipelineState> add_pso_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_add_fp32_into_fp16"); });
    return pso;
}
id<MTLComputePipelineState> fw_pso_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_fw_bf16"); });
    return pso;
}
id<MTLComputePipelineState> bw_pso_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_bw_bf16"); });
    return pso;
}
id<MTLComputePipelineState> add_pso_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_emb_add_fp32_into_bf16"); });
    return pso;
}

void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger n,
                void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void embedding_lookup_forward(const Tensor& table,
                              const int32_t* d_idx, int B,
                              Tensor& out) {
    const int D = table.cols;
    if (out.rows != B || out.cols != D || out.dtype != table.dtype) {
        out.resize(B, D, table.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    id<MTLBuffer> bT = buffer_for(table);
    NSUInteger oT = buffer_offset_for(table);
    id<MTLBuffer> bO = buffer_for(out);
    NSUInteger oO = buffer_offset_for(out);
    id<MTLBuffer> bI = pool_lookup(d_idx);
    NSUInteger oI = pool_lookup_offset(d_idx);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);
    id<MTLComputePipelineState> pso =
        (table.dtype == Dtype::FP16) ? fw_pso_fp16()
      : (table.dtype == Dtype::BF16) ? fw_pso_bf16()
      : fw_pso();
    dispatch1d(pso, total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bT offset:oT atIndex:0];
        [enc setBuffer:bI offset:oI atIndex:1];
        [enc setBuffer:bO offset:oO atIndex:2];
        [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
    });
}

void embedding_lookup_backward(const Tensor& dOut,
                               const int32_t* d_idx, int B,
                               Tensor& dTable) {
    if (dTable.dtype != Dtype::FP16 && dTable.dtype != Dtype::FP32 &&
        dTable.dtype != Dtype::BF16) {
        throw std::runtime_error("embedding_lookup_backward_gpu: dTable must be FP16, BF16, or FP32");
    }
    if (dOut.dtype != dTable.dtype) {
        throw std::runtime_error("embedding_lookup_backward_gpu: dOut/dTable dtype must match");
    }
    const int D = dTable.cols;
    const int total = B * D;
    if (total == 0) return;
    id<MTLBuffer> bdO = buffer_for(dOut);
    NSUInteger odO = buffer_offset_for(dOut);
    id<MTLBuffer> bdT = buffer_for(dTable);
    NSUInteger odT = buffer_offset_for(dTable);
    id<MTLBuffer> bI  = pool_lookup(d_idx);
    NSUInteger oI = pool_lookup_offset(d_idx);
    const uint32_t Bu = static_cast<uint32_t>(B);
    const uint32_t Du = static_cast<uint32_t>(D);

    if (dTable.dtype == Dtype::FP32) {
        dispatch1d(bw_pso(), total, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bdO offset:odO atIndex:0];
            [enc setBuffer:bI  offset:oI atIndex:1];
            [enc setBuffer:bdT offset:odT atIndex:2];
            [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        });
    } else {
        const bool is_bf16 = (dTable.dtype == Dtype::BF16);
        const NSUInteger table_n = static_cast<NSUInteger>(dTable.rows) * D;
        @autoreleasepool {
            id<MTLBuffer> scratch = [metal_impl::device()
                newBufferWithLength:table_n * sizeof(float)
                            options:MTLResourceStorageModeShared];
            std::memset([scratch contents], 0, table_n * sizeof(float));
            id<MTLComputePipelineState> scatter_pso = is_bf16 ? bw_pso_bf16() : bw_pso_fp16();
            dispatch1d(scatter_pso, total, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:bdO offset:odO atIndex:0];
                [enc setBuffer:bI  offset:oI atIndex:1];
                [enc setBuffer:scratch offset:0 atIndex:2];
                [enc setBytes:&Bu length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
            });
            const uint32_t n = static_cast<uint32_t>(table_n);
            id<MTLComputePipelineState> add_pso = is_bf16 ? add_pso_bf16() : add_pso_fp16();
            dispatch1d(add_pso, n, ^(id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:scratch offset:0 atIndex:0];
                [enc setBuffer:bdT offset:odT atIndex:1];
                [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
            });
        }
    }
}

} // namespace brotensor::detail::metal
