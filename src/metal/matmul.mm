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

kernel void k_matmul_bf16(device const bfloat* A [[buffer(0)]],
                          device const bfloat* B [[buffer(1)]],
                          device bfloat*       C [[buffer(2)]],
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
    if (row < M && col < N) C[row * N + col] = bfloat(acc);
}
)msl";

// C(batch, M, N) = A(batch, M, K) @ B(batch, N, K)^T, with an optional
// per-N bias (shared across the batch) and an epilogue activation fused into
// the store. Mirrors src/cuda/fp16_matmul.cu's matmul_ABT_naive_kernel —
// tiled per-batch strides, no WMMA fast path (CUDA's tensor-core ABT kernel
// has no Metal analog yet; this is a correctness-first naive GEMM, one
// thread per output element).
NSString* const kAbtSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// MSL has no built-in erf; Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).
inline float abt_erf_approx(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    float sign_x = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t  = 1.0f / (1.0f + p * ax);
    float y  = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign_x * y;
}

inline float abt_apply_act(int act, float v) {
    switch (act) {
        case 1:  return max(v, 0.0f);
        case 2: {
            constexpr float kSqrt2OverPi = 0.7978845608f;
            float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanh(clamp(u, -9.0f, 9.0f)));
        }
        case 3: {
            constexpr float kInvSqrt2 = 0.70710678118654752440f;
            return 0.5f * v * (1.0f + abt_erf_approx(v * kInvSqrt2));
        }
        case 4:  return v / (1.0f + exp(-v));
        case 5:  return v / (1.0f + exp(-1.702f * v));
        default: return v;
    }
}

kernel void k_matmul_abt_fp16(device const half* A [[buffer(0)]],
                              device const half* B [[buffer(1)]],
                              device half*       C [[buffer(2)]],
                              constant uint& M       [[buffer(3)]],
                              constant uint& N       [[buffer(4)]],
                              constant uint& K       [[buffer(5)]],
                              constant ulong& strideA [[buffer(6)]],
                              constant ulong& strideB [[buffer(7)]],
                              constant ulong& strideC [[buffer(8)]],
                              device const half* bias [[buffer(9)]],
                              constant uint& has_bias [[buffer(10)]],
                              constant int& act        [[buffer(11)]],
                              uint2 idx3 [[thread_position_in_grid]]) {
    uint idx = idx3.x;
    uint b   = idx3.y;
    uint total = M * N;
    if (idx >= total) return;
    device const half* Ab = A + (ulong)b * strideA;
    device const half* Bb = B + (ulong)b * strideB;
    device half*       Cb = C + (ulong)b * strideC;
    uint m = idx / N;
    uint n = idx % N;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(Ab[m * K + k]) * float(Bb[n * K + k]);
    }
    if (has_bias) acc += float(bias[n]);
    acc = abt_apply_act(act, acc);
    Cb[idx] = half(acc);
}

kernel void k_matmul_abt_bf16(device const bfloat* A [[buffer(0)]],
                              device const bfloat* B [[buffer(1)]],
                              device bfloat*       C [[buffer(2)]],
                              constant uint& M       [[buffer(3)]],
                              constant uint& N       [[buffer(4)]],
                              constant uint& K       [[buffer(5)]],
                              constant ulong& strideA [[buffer(6)]],
                              constant ulong& strideB [[buffer(7)]],
                              constant ulong& strideC [[buffer(8)]],
                              device const bfloat* bias [[buffer(9)]],
                              constant uint& has_bias   [[buffer(10)]],
                              constant int& act          [[buffer(11)]],
                              uint2 idx3 [[thread_position_in_grid]]) {
    uint idx = idx3.x;
    uint b   = idx3.y;
    uint total = M * N;
    if (idx >= total) return;
    device const bfloat* Ab = A + (ulong)b * strideA;
    device const bfloat* Bb = B + (ulong)b * strideB;
    device bfloat*       Cb = C + (ulong)b * strideC;
    uint m = idx / N;
    uint n = idx % N;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(Ab[m * K + k]) * float(Bb[n * K + k]);
    }
    if (has_bias) acc += float(bias[n]);
    acc = abt_apply_act(act, acc);
    Cb[idx] = bfloat(acc);
}
)msl";

id<MTLComputePipelineState> pso_abt_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kAbtSrc, @"k_matmul_abt_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_abt_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kAbtSrc, @"k_matmul_abt_bf16"); });
    return pso;
}

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
id<MTLComputePipelineState> pso_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_bf16"); });
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
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_gpu: only FP32/FP16/BF16 supported");
    }

    id<MTLComputePipelineState> pso = (A.dtype == Dtype::FP16) ? pso_fp16()
                                    : (A.dtype == Dtype::BF16) ? pso_bf16()
                                    : pso_fp32();
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

void matmul_abt(const Tensor& A, const Tensor& B, Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const Tensor* bias, int act) {
    if (A.dtype != B.dtype || A.dtype != C.dtype) {
        throw std::runtime_error("matmul_abt: A, B, C must share dtype");
    }
    if (A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_abt: dtype must be FP16 or BF16");
    }
    if (bias && bias->dtype != A.dtype) {
        throw std::runtime_error("matmul_abt: bias dtype must match operands");
    }
    if (batch <= 0 || M == 0 || N == 0) return;

    id<MTLBuffer> bA = buffer_for(A);
    id<MTLBuffer> bB = buffer_for(B);
    id<MTLBuffer> bC = buffer_for(C);
    const NSUInteger oA = buffer_offset_for(A);
    const NSUInteger oB = buffer_offset_for(B);
    const NSUInteger oC = buffer_offset_for(C);

    if (K == 0) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            const NSUInteger elem = 2;  // FP16/BF16 are both 2-byte carriers here
            for (int b = 0; b < batch; ++b) {
                [blit fillBuffer:bC
                           range:NSMakeRange(oC + (NSUInteger)b * (NSUInteger)strideC * elem,
                                            (NSUInteger)M * (NSUInteger)N * elem)
                           value:0];
            }
            [blit endEncoding];
            ::brotensor::metal_impl::submit(cmd);
        }
        return;
    }

    id<MTLComputePipelineState> pso =
        (A.dtype == Dtype::FP16) ? pso_abt_fp16() : pso_abt_bf16();
    id<MTLBuffer> bBias = bias ? buffer_for(*bias) : bA;
    const NSUInteger oBias = bias ? buffer_offset_for(*bias) : oA;
    const uint32_t Mu = static_cast<uint32_t>(M);
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Ku = static_cast<uint32_t>(K);
    const uint64_t sA = static_cast<uint64_t>(strideA);
    const uint64_t sB = static_cast<uint64_t>(strideB);
    const uint64_t sC = static_cast<uint64_t>(strideC);
    const uint32_t hasBias = bias ? 1u : 0u;
    const int32_t  actI = static_cast<int32_t>(act);

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
        [enc setBytes:&sA length:sizeof(uint64_t) atIndex:6];
        [enc setBytes:&sB length:sizeof(uint64_t) atIndex:7];
        [enc setBytes:&sC length:sizeof(uint64_t) atIndex:8];
        [enc setBuffer:bBias offset:oBias atIndex:9];
        [enc setBytes:&hasBias length:sizeof(uint32_t) atIndex:10];
        [enc setBytes:&actI length:sizeof(int32_t) atIndex:11];
        const uint32_t total = Mu * Nu;
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, (NSUInteger)batch, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
