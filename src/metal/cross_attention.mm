#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

constexpr NSUInteger CA_BLOCK = 128;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint CA_BLOCK = 128;

// C(M, N) = A(M, K) @ B(N, K)^T, FP16 IO, FP32 accumulator. One thread per
// output element. Mirrors the CUDA matmul_ABT_fp16_kernel.
kernel void k_matmul_abt_fp16(device const half* A [[buffer(0)]],
                              device const half* B [[buffer(1)]],
                              device half*       C [[buffer(2)]],
                              constant uint& M     [[buffer(3)]],
                              constant uint& N     [[buffer(4)]],
                              constant uint& K     [[buffer(5)]],
                              uint idx [[thread_position_in_grid]]) {
    uint total = M * N;
    if (idx >= total) return;
    uint m = idx / N;
    uint n = idx % N;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(A[m * K + k]) * float(B[n * K + k]);
    }
    C[idx] = half(acc);
}

// One threadgroup per (query, head) tile. Threadgroup memory:
//   scores[0..Lk)     – per-key score / softmax weight
//   s_red[0..CA_BLOCK) – reduction scratch
kernel void k_cross_attention_core(
        device const half*  Q   [[buffer(0)]],   // (Lq, D)
        device const half*  Kk  [[buffer(1)]],   // (Lk, D)
        device const half*  V   [[buffer(2)]],   // (Lk, D)
        device const float* mask[[buffer(3)]],   // (Lk,) may be dummy
        device half*        Out [[buffer(4)]],   // (Lq, D)
        constant uint& Lq        [[buffer(5)]],
        constant uint& Lk        [[buffer(6)]],
        constant uint& D         [[buffer(7)]],
        constant uint& head_dim  [[buffer(8)]],
        constant uint& has_mask  [[buffer(9)]],
        threadgroup float* scratch [[threadgroup(0)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float* scores = scratch;
    threadgroup float* s_red  = scratch + Lk;

    uint q = gid.x;
    uint h = gid.y;
    uint head_off = h * head_dim;
    float inv_sqrt = rsqrt(float(head_dim));

    // 1. scores[k] = Q[q, head] · K[k, head] * inv_sqrt (− ∞ if masked)
    for (uint k = tid; k < Lk; k += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += float(Q[q * D + head_off + d]) *
                   float(Kk[k * D + head_off + d]);
        }
        float s = dot * inv_sqrt;
        if (has_mask != 0u && mask[k] <= 0.5f) {
            s = -1e30f;
        }
        scores[k] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. max-reduce
    float local_max = -1e30f;
    for (uint k = tid; k < Lk; k += tg_size) {
        if (scores[k] > local_max) local_max = scores[k];
    }
    s_red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_red[tid + s];
            if (other > s_red[tid]) s_red[tid] = other;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_v = s_red[0];

    // 3. exp + sum
    float local_sum = 0.0f;
    for (uint k = tid; k < Lk; k += tg_size) {
        float e = exp(scores[k] - max_v);
        scores[k] = e;
        local_sum += e;
    }
    s_red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = s_red[0];
    float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

    // 4. Out[q, head + d] = Σ_k softmax[k] * V[k, head + d]
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint k = 0; k < Lk; ++k) {
            acc += scores[k] * inv_denom *
                   float(V[k * D + head_off + d]);
        }
        Out[q * D + head_off + d] = half(acc);
    }
}
)msl";

id<MTLComputePipelineState> pso_matmul_abt() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_abt_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_core() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cross_attention_core"); });
    return pso;
}

void launch_matmul_abt(const GpuTensor& A, const GpuTensor& B, GpuTensor& C,
                       int M, int N, int K) {
    const uint32_t total = static_cast<uint32_t>(M) * static_cast<uint32_t>(N);
    if (total == 0) return;
    id<MTLComputePipelineState> pso = pso_matmul_abt();
    id<MTLBuffer> bA = buffer_for(A);
    id<MTLBuffer> bB = buffer_for(B);
    id<MTLBuffer> bC = buffer_for(C);
    const NSUInteger oA = buffer_offset_for(A);
    const NSUInteger oB = buffer_offset_for(B);
    const NSUInteger oC = buffer_offset_for(C);
    const uint32_t Mu = M, Nu = N, Ku = K;
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
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void cross_attention_forward_gpu(const GpuTensor& X,
                                 const GpuTensor& Ctx,
                                 const GpuTensor& Wq, const GpuTensor& Wk,
                                 const GpuTensor& Wv, const GpuTensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 GpuTensor& O) {
    if (X.dtype != Dtype::FP16 || Ctx.dtype != Dtype::FP16 ||
        Wq.dtype != Dtype::FP16 || Wk.dtype != Dtype::FP16 ||
        Wv.dtype != Dtype::FP16 || Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("cross_attention_forward_gpu: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int Lk = Ctx.rows;
    const int D  = X.cols;
    if (Ctx.cols != D || Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("cross_attention_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("cross_attention_forward_gpu: num_heads must divide D");
    }
    const int head_dim = D / num_heads;

    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    GpuTensor Qp(Lq, D, Dtype::FP16);
    GpuTensor Kp(Lk, D, Dtype::FP16);
    GpuTensor Vp(Lk, D, Dtype::FP16);
    GpuTensor Op(Lq, D, Dtype::FP16);

    launch_matmul_abt(X,   Wq, Qp, Lq, D, D);
    launch_matmul_abt(Ctx, Wk, Kp, Lk, D, D);
    launch_matmul_abt(Ctx, Wv, Vp, Lk, D, D);

    id<MTLComputePipelineState> pso = pso_core();
    id<MTLBuffer> bQ = buffer_for(Qp);
    id<MTLBuffer> bK = buffer_for(Kp);
    id<MTLBuffer> bV = buffer_for(Vp);
    id<MTLBuffer> bO = buffer_for(Op);
    const NSUInteger oQ = buffer_offset_for(Qp);
    const NSUInteger oK = buffer_offset_for(Kp);
    const NSUInteger oV = buffer_offset_for(Vp);
    const NSUInteger oO = buffer_offset_for(Op);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bQ; // dummy bind when mask is null
    NSUInteger oM_arg = bM ? oM : oQ;

    const uint32_t Lqu = Lq, Lku = Lk, Du = D, hdU = head_dim;
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const NSUInteger shmem = (static_cast<NSUInteger>(Lk) + CA_BLOCK) * sizeof(float);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:oQ atIndex:0];
        [enc setBuffer:bK offset:oK atIndex:1];
        [enc setBuffer:bV offset:oV atIndex:2];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:3];
        [enc setBuffer:bO offset:oO atIndex:4];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&hdU length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:9];
        [enc setThreadgroupMemoryLength:shmem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(Lq, num_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(CA_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    launch_matmul_abt(Op, Wo, O, Lq, D, D);
}

} // namespace brotensor
