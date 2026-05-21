// Backward of matmul_gpu (Metal). FP32 path uses atomic_float accumulation
// directly into dA / dB. FP16 path accumulates into FP32 scratch then folds
// into the caller-owned FP16 dA / dB (accumulating semantics).

#include <brotensor/runtime.h>

#include <cstring>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger MMB_TILE = 16;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint MMB_TILE = 16;

kernel void k_mmb_dA_fp32(device const float* dC [[buffer(0)]],
                          device const float* B  [[buffer(1)]],
                          device atomic_float* dA [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float dCs[MMB_TILE][MMB_TILE];
    threadgroup float Bts[MMB_TILE][MMB_TILE];
    uint row = tg.y * MMB_TILE + lid.y;
    uint col = tg.x * MMB_TILE + lid.x;
    float acc = 0.0f;
    uint n_tiles = (N + MMB_TILE - 1) / MMB_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint dc_col = t * MMB_TILE + lid.x;
        uint bt_row = t * MMB_TILE + lid.y;
        dCs[lid.y][lid.x] = (row < M && dc_col < N) ? dC[row * N + dc_col] : 0.0f;
        Bts[lid.y][lid.x] = (col < K && bt_row < N) ? B[col * N + bt_row] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < MMB_TILE; ++n) acc += dCs[lid.y][n] * Bts[n][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) {
        atomic_fetch_add_explicit(&dA[row * K + col], acc, memory_order_relaxed);
    }
}

kernel void k_mmb_dB_fp32(device const float* A  [[buffer(0)]],
                          device const float* dC [[buffer(1)]],
                          device atomic_float* dB [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float Ats[MMB_TILE][MMB_TILE];
    threadgroup float dCs[MMB_TILE][MMB_TILE];
    uint row = tg.y * MMB_TILE + lid.y;
    uint col = tg.x * MMB_TILE + lid.x;
    float acc = 0.0f;
    uint n_tiles = (M + MMB_TILE - 1) / MMB_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint a_row  = t * MMB_TILE + lid.x;
        uint dc_row = t * MMB_TILE + lid.y;
        Ats[lid.y][lid.x] = (row < K && a_row < M) ? A[a_row * K + row] : 0.0f;
        dCs[lid.y][lid.x] = (dc_row < M && col < N) ? dC[dc_row * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < MMB_TILE; ++m) acc += Ats[lid.y][m] * dCs[m][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        atomic_fetch_add_explicit(&dB[row * N + col], acc, memory_order_relaxed);
    }
}

kernel void k_mmb_dA_fp16(device const half* dC [[buffer(0)]],
                          device const half* B  [[buffer(1)]],
                          device atomic_float* dA [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float dCs[MMB_TILE][MMB_TILE];
    threadgroup float Bts[MMB_TILE][MMB_TILE];
    uint row = tg.y * MMB_TILE + lid.y;
    uint col = tg.x * MMB_TILE + lid.x;
    float acc = 0.0f;
    uint n_tiles = (N + MMB_TILE - 1) / MMB_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint dc_col = t * MMB_TILE + lid.x;
        uint bt_row = t * MMB_TILE + lid.y;
        dCs[lid.y][lid.x] = (row < M && dc_col < N) ? float(dC[row * N + dc_col]) : 0.0f;
        Bts[lid.y][lid.x] = (col < K && bt_row < N) ? float(B[col * N + bt_row]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < MMB_TILE; ++n) acc += dCs[lid.y][n] * Bts[n][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) {
        atomic_fetch_add_explicit(&dA[row * K + col], acc, memory_order_relaxed);
    }
}

kernel void k_mmb_dB_fp16(device const half* A  [[buffer(0)]],
                          device const half* dC [[buffer(1)]],
                          device atomic_float* dB [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tg  [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float Ats[MMB_TILE][MMB_TILE];
    threadgroup float dCs[MMB_TILE][MMB_TILE];
    uint row = tg.y * MMB_TILE + lid.y;
    uint col = tg.x * MMB_TILE + lid.x;
    float acc = 0.0f;
    uint n_tiles = (M + MMB_TILE - 1) / MMB_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint a_row  = t * MMB_TILE + lid.x;
        uint dc_row = t * MMB_TILE + lid.y;
        Ats[lid.y][lid.x] = (row < K && a_row < M) ? float(A[a_row * K + row]) : 0.0f;
        dCs[lid.y][lid.x] = (dc_row < M && col < N) ? float(dC[dc_row * N + col]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < MMB_TILE; ++m) acc += Ats[lid.y][m] * dCs[m][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        atomic_fetch_add_explicit(&dB[row * N + col], acc, memory_order_relaxed);
    }
}

kernel void k_mmb_fold_fp16(device half*        dst [[buffer(0)]],
                            device const float* src [[buffer(1)]],
                            constant uint& n        [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float cur = float(dst[gid]);
    dst[gid] = half(cur + src[gid]);
}

kernel void k_mmb_dA_bf16(device const bfloat* dC [[buffer(0)]],
                           device const bfloat* B  [[buffer(1)]],
                           device atomic_float* dA [[buffer(2)]],
                           constant uint& M [[buffer(3)]],
                           constant uint& N [[buffer(4)]],
                           constant uint& K [[buffer(5)]],
                           uint2 tg  [[threadgroup_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float dCs[MMB_TILE][MMB_TILE];
    threadgroup float Bts[MMB_TILE][MMB_TILE];
    uint row = tg.y * MMB_TILE + lid.y;
    uint col = tg.x * MMB_TILE + lid.x;
    float acc = 0.0f;
    uint n_tiles = (N + MMB_TILE - 1) / MMB_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint dc_col = t * MMB_TILE + lid.x;
        uint bt_row = t * MMB_TILE + lid.y;
        dCs[lid.y][lid.x] = (row < M && dc_col < N) ? float(dC[row * N + dc_col]) : 0.0f;
        Bts[lid.y][lid.x] = (col < K && bt_row < N) ? float(B[col * N + bt_row]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n = 0; n < MMB_TILE; ++n) acc += dCs[lid.y][n] * Bts[n][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) {
        atomic_fetch_add_explicit(&dA[row * K + col], acc, memory_order_relaxed);
    }
}

kernel void k_mmb_dB_bf16(device const bfloat* A  [[buffer(0)]],
                           device const bfloat* dC [[buffer(1)]],
                           device atomic_float* dB [[buffer(2)]],
                           constant uint& M [[buffer(3)]],
                           constant uint& N [[buffer(4)]],
                           constant uint& K [[buffer(5)]],
                           uint2 tg  [[threadgroup_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float Ats[MMB_TILE][MMB_TILE];
    threadgroup float dCs[MMB_TILE][MMB_TILE];
    uint row = tg.y * MMB_TILE + lid.y;
    uint col = tg.x * MMB_TILE + lid.x;
    float acc = 0.0f;
    uint n_tiles = (M + MMB_TILE - 1) / MMB_TILE;
    for (uint t = 0; t < n_tiles; ++t) {
        uint a_row  = t * MMB_TILE + lid.x;
        uint dc_row = t * MMB_TILE + lid.y;
        Ats[lid.y][lid.x] = (row < K && a_row < M) ? float(A[a_row * K + row]) : 0.0f;
        dCs[lid.y][lid.x] = (dc_row < M && col < N) ? float(dC[dc_row * N + col]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < MMB_TILE; ++m) acc += Ats[lid.y][m] * dCs[m][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) {
        atomic_fetch_add_explicit(&dB[row * N + col], acc, memory_order_relaxed);
    }
}

kernel void k_mmb_fold_bf16(device bfloat*      dst [[buffer(0)]],
                             device const float* src [[buffer(1)]],
                             constant uint& n        [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float cur = float(dst[gid]);
    dst[gid] = bfloat(cur + src[gid]);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_dA_fp32,  @"k_mmb_dA_fp32")
DEF_PSO(pso_dB_fp32,  @"k_mmb_dB_fp32")
DEF_PSO(pso_dA_fp16,  @"k_mmb_dA_fp16")
DEF_PSO(pso_dB_fp16,  @"k_mmb_dB_fp16")
DEF_PSO(pso_fold,     @"k_mmb_fold_fp16")
DEF_PSO(pso_dA_bf16,  @"k_mmb_dA_bf16")
DEF_PSO(pso_dB_bf16,  @"k_mmb_dB_bf16")
DEF_PSO(pso_fold_bf16, @"k_mmb_fold_bf16")
#undef DEF_PSO

} // namespace

void matmul_backward(const Tensor& A,
                     const Tensor& B,
                     const Tensor& dC,
                     Tensor& dA,
                     Tensor& dB) {
    if (A.dtype != B.dtype || A.dtype != dC.dtype ||
        A.dtype != dA.dtype || A.dtype != dB.dtype) {
        throw std::runtime_error("matmul_backward_gpu: dtype mismatch");
    }
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_backward_gpu: only FP32/FP16/BF16 supported");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_backward_gpu: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (dC.rows != M || dC.cols != N) {
        throw std::runtime_error("matmul_backward_gpu: dC shape mismatch");
    }
    if (dA.rows != M || dA.cols != K) {
        throw std::runtime_error("matmul_backward_gpu: dA must be pre-sized to (M, K)");
    }
    if (dB.rows != K || dB.cols != N) {
        throw std::runtime_error("matmul_backward_gpu: dB must be pre-sized to (K, N)");
    }
    if (M == 0 || N == 0 || K == 0) return;

    id<MTLBuffer> bA  = buffer_for(A);
    id<MTLBuffer> bB  = buffer_for(B);
    id<MTLBuffer> bdC = buffer_for(dC);
    const NSUInteger oA  = buffer_offset_for(A);
    const NSUInteger oB  = buffer_offset_for(B);
    const NSUInteger odC = buffer_offset_for(dC);
    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Ku = static_cast<uint32_t>(K);

    MTLSize tg = MTLSizeMake(MMB_TILE, MMB_TILE, 1);
    MTLSize gridA = MTLSizeMake((K + MMB_TILE - 1) / MMB_TILE,
                                (M + MMB_TILE - 1) / MMB_TILE, 1);
    MTLSize gridB = MTLSizeMake((N + MMB_TILE - 1) / MMB_TILE,
                                (K + MMB_TILE - 1) / MMB_TILE, 1);

    if (A.dtype == Dtype::FP32) {
        id<MTLBuffer> bdA = buffer_for(dA);
        id<MTLBuffer> bdB = buffer_for(dB);
        const NSUInteger odA = buffer_offset_for(dA);
        const NSUInteger odB = buffer_offset_for(dB);
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dA_fp32()];
            [enc setBuffer:bdC offset:odC atIndex:0];
            [enc setBuffer:bB  offset:oB  atIndex:1];
            [enc setBuffer:bdA offset:odA atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:gridA threadsPerThreadgroup:tg];

            [enc setComputePipelineState:pso_dB_fp32()];
            [enc setBuffer:bA  offset:oA  atIndex:0];
            [enc setBuffer:bdC offset:odC atIndex:1];
            [enc setBuffer:bdB offset:odB atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:gridB threadsPerThreadgroup:tg];

            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    // BF16 path: FP32 scratch + fold (identical structure to FP16 path).
    if (A.dtype == Dtype::BF16) {
        const NSUInteger nA = static_cast<NSUInteger>(M) * K;
        const NSUInteger nB = static_cast<NSUInteger>(K) * N;
        @autoreleasepool {
            id<MTLBuffer> scratchA = [metal_impl::device()
                newBufferWithLength:nA * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> scratchB = [metal_impl::device()
                newBufferWithLength:nB * sizeof(float)
                            options:MTLResourceStorageModeShared];
            std::memset([scratchA contents], 0, nA * sizeof(float));
            std::memset([scratchB contents], 0, nB * sizeof(float));

            id<MTLBuffer> bdA = buffer_for(dA);
            id<MTLBuffer> bdB = buffer_for(dB);
            const NSUInteger odA = buffer_offset_for(dA);
            const NSUInteger odB = buffer_offset_for(dB);

            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:pso_dA_bf16()];
            [enc setBuffer:bdC offset:odC atIndex:0];
            [enc setBuffer:bB  offset:oB  atIndex:1];
            [enc setBuffer:scratchA offset:0 atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:gridA threadsPerThreadgroup:tg];

            [enc setComputePipelineState:pso_dB_bf16()];
            [enc setBuffer:bA  offset:oA  atIndex:0];
            [enc setBuffer:bdC offset:odC atIndex:1];
            [enc setBuffer:scratchB offset:0 atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:gridB threadsPerThreadgroup:tg];

            // Fold scratch into BF16 dst.
            const NSUInteger tpt = 256;
            id<MTLComputePipelineState> fold = pso_fold_bf16();
            [enc setComputePipelineState:fold];
            const uint32_t nAu = static_cast<uint32_t>(nA);
            [enc setBuffer:bdA offset:odA atIndex:0];
            [enc setBuffer:scratchA offset:0 atIndex:1];
            [enc setBytes:&nAu length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((nA + tpt - 1) / tpt, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];

            const uint32_t nBu = static_cast<uint32_t>(nB);
            [enc setBuffer:bdB offset:odB atIndex:0];
            [enc setBuffer:scratchB offset:0 atIndex:1];
            [enc setBytes:&nBu length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((nB + tpt - 1) / tpt, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];

            [enc endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    // FP16 path: FP32 scratch + fold.
    const NSUInteger nA = static_cast<NSUInteger>(M) * K;
    const NSUInteger nB = static_cast<NSUInteger>(K) * N;
    @autoreleasepool {
        id<MTLBuffer> scratchA = [metal_impl::device()
            newBufferWithLength:nA * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> scratchB = [metal_impl::device()
            newBufferWithLength:nB * sizeof(float)
                        options:MTLResourceStorageModeShared];
        std::memset([scratchA contents], 0, nA * sizeof(float));
        std::memset([scratchB contents], 0, nB * sizeof(float));

        id<MTLBuffer> bdA = buffer_for(dA);
        id<MTLBuffer> bdB = buffer_for(dB);
        const NSUInteger odA = buffer_offset_for(dA);
        const NSUInteger odB = buffer_offset_for(dB);

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso_dA_fp16()];
        [enc setBuffer:bdC offset:odC atIndex:0];
        [enc setBuffer:bB  offset:oB  atIndex:1];
        [enc setBuffer:scratchA offset:0 atIndex:2];
        [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:gridA threadsPerThreadgroup:tg];

        [enc setComputePipelineState:pso_dB_fp16()];
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bdC offset:odC atIndex:1];
        [enc setBuffer:scratchB offset:0 atIndex:2];
        [enc setBytes:&Mu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Ku length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreadgroups:gridB threadsPerThreadgroup:tg];

        // Fold scratch into FP16 dst.
        const NSUInteger tpt = 256;
        id<MTLComputePipelineState> fold = pso_fold();
        [enc setComputePipelineState:fold];
        const uint32_t nAu = static_cast<uint32_t>(nA);
        [enc setBuffer:bdA offset:odA atIndex:0];
        [enc setBuffer:scratchA offset:0 atIndex:1];
        [enc setBytes:&nAu length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((nA + tpt - 1) / tpt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];

        const uint32_t nBu = static_cast<uint32_t>(nB);
        [enc setBuffer:bdB offset:odB atIndex:0];
        [enc setBuffer:scratchB offset:0 atIndex:1];
        [enc setBytes:&nBu length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((nB + tpt - 1) / tpt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];

        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
