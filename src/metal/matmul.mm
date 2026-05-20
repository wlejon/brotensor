// Public matmul_gpu: row-major C(M,N) = A(M,K) @ B(K,N), no bias.
// FP32 + FP16 dispatch; FP32 accumulation throughout. Naive tiled GEMM,
// mirroring src/cuda/matmul.cu.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger MM_TILE = 16;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint MM_TILE = 16;

kernel void k_matmul_fp32(device const float* A [[buffer(0)]],
                          device const float* B [[buffer(1)]],
                          device float*       C [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[MM_TILE][MM_TILE];
    threadgroup float Bs[MM_TILE][MM_TILE];

    uint row = tg.y * MM_TILE + lid.y;
    uint col = tg.x * MM_TILE + lid.x;

    float acc = 0.0f;
    uint n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint a_col = t * MM_TILE + lid.x;
        uint b_row = t * MM_TILE + lid.y;
        As[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[lid.y][lid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MM_TILE; ++k) {
            acc += As[lid.y][k] * Bs[k][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

kernel void k_matmul_fp16(device const half* A [[buffer(0)]],
                          device const half* B [[buffer(1)]],
                          device half*       C [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[MM_TILE][MM_TILE];
    threadgroup float Bs[MM_TILE][MM_TILE];

    uint row = tg.y * MM_TILE + lid.y;
    uint col = tg.x * MM_TILE + lid.x;

    float acc = 0.0f;
    uint n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint a_col = t * MM_TILE + lid.x;
        uint b_row = t * MM_TILE + lid.y;
        As[lid.y][lid.x] = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;
        Bs[lid.y][lid.x] = (b_row < K && col < N) ? float(B[b_row * N + col]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < MM_TILE; ++k) {
            acc += As[lid.y][k] * Bs[k][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = half(acc);
}
)msl";

id<MTLComputePipelineState> pso_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_fp32"); });
    return pso;
}
id<MTLComputePipelineState> pso_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_fp16"); });
    return pso;
}

} // namespace

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("matmul_gpu: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_gpu: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) {
        C.resize(M, N, A.dtype);
    }
    if (M == 0 || N == 0) return;
    if (K == 0) {
        C.zero();
        return;
    }
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16) {
        throw std::runtime_error("matmul_gpu: only FP32/FP16 supported");
    }

    id<MTLComputePipelineState> pso = (A.dtype == Dtype::FP16) ? pso_fp16() : pso_fp32();
    id<MTLBuffer> bA = buffer_for(A);
    id<MTLBuffer> bB = buffer_for(B);
    id<MTLBuffer> bC = buffer_for(C);
    const NSUInteger oA = buffer_offset_for(A);
    const NSUInteger oB = buffer_offset_for(B);
    const NSUInteger oC = buffer_offset_for(C);
    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Ku = static_cast<uint32_t>(K);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bA offset:oA atIndex:0];
        [enc setBuffer:bB offset:oB atIndex:1];
        [enc setBuffer:bC offset:oC atIndex:2];
        [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        MTLSize grid = MTLSizeMake((N + MM_TILE - 1) / MM_TILE,
                                   (M + MM_TILE - 1) / MM_TILE, 1);
        MTLSize tg = MTLSizeMake(MM_TILE, MM_TILE, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
