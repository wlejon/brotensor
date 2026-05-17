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

constexpr NSUInteger FA_BLOCK = 128;
constexpr NSUInteger FA_KTILE = 64;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint FA_BLOCK = 128;
constant uint FA_KTILE = 64;
constant uint MAX_HD_PER_THREAD = 8;

// Naive FP16 matmul A·Bᵀ. Local to this MSL library; mirrored from
// fp16_internal.cuh.
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

// Flash-attention online-softmax kernel. One threadgroup per (q, head).
// Tiles K/V along Lk (FA_KTILE = 64). Per-thread partial output kept in
// registers (`partial[]`), strided over head_dim with up to MAX_HD_PER_THREAD
// slots/thread (FA_BLOCK * MAX_HD_PER_THREAD ≥ head_dim required).
//
// Threadgroup memory layout (dynamic):
//   scratch[0..FA_KTILE)    – tile scores
//   scratch[FA_KTILE..FA_KTILE+FA_BLOCK) – reduction scratch
kernel void k_flash_attention(
        device const half*  Q    [[buffer(0)]],   // (Lq, D)
        device const half*  Kk   [[buffer(1)]],   // (Lk, D)
        device const half*  V    [[buffer(2)]],   // (Lk, D)
        device const float* mask [[buffer(3)]],   // (Lk,) may be dummy
        device half*        Out  [[buffer(4)]],   // (Lq, D)
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
    threadgroup float* red    = scratch + FA_KTILE;

    uint q = gid.x;
    uint h = gid.y;
    uint head_off = h * head_dim;
    float inv_sqrt = rsqrt(float(head_dim));

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (uint k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        uint klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;

        // 1. scores[t] = Q[q] · K[k0+t] * inv_sqrt
        for (uint t = tid; t < klen; t += tg_size) {
            uint kg = k0 + t;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q[q * D + head_off + d]) *
                       float(Kk[kg * D + head_off + d]);
            }
            float s = dot * inv_sqrt;
            if (has_mask != 0u && mask[kg] <= 0.5f) s = -1e30f;
            scores[t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. tile max
        float local_max = -1e30f;
        for (uint t = tid; t < klen; t += tg_size) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = red[tid + s];
                if (other > red[tid]) red[tid] = other;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_max = red[0];
        float m_new = (tile_max > run_max) ? tile_max : run_max;

        // 3. exponentiate against m_new, sum
        bool tile_empty = (m_new <= -1e29f);
        for (uint t = tid; t < klen; t += tg_size) {
            float e = tile_empty ? 0.0f : exp(scores[t] - m_new);
            scores[t] = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local_sum = 0.0f;
        for (uint t = tid; t < klen; t += tg_size) local_sum += scores[t];
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float tile_sum = red[0];

        // 4. rescale running state
        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = exp(run_max - m_new);
        }

        // 5. update partial output for this thread's d-slots
        uint slot = 0;
        for (uint d = tid; d < head_dim; d += tg_size) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (uint t = 0; t < klen; ++t) {
                acc += scores[t] * float(V[(k0 + t) * D + head_off + d]);
            }
            partial[slot] = acc;
            ++slot;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 6. normalize + write
    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < head_dim; d += tg_size) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q * D + head_off + d] = half(partial[slot] * inv);
        ++slot;
    }
}
)msl";

id<MTLComputePipelineState> pso_flash() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_flash_attention"); });
    return pso;
}
id<MTLComputePipelineState> pso_matmul_abt() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_matmul_abt_fp16"); });
    return pso;
}

void launch_matmul_abt(const GpuTensor& A, const GpuTensor& B, GpuTensor& C,
                       int M, int N, int Kdim) {
    const uint32_t total = static_cast<uint32_t>(M) * static_cast<uint32_t>(N);
    if (total == 0) return;
    id<MTLComputePipelineState> pso = pso_matmul_abt();
    id<MTLBuffer> bA = buffer_for(A);
    id<MTLBuffer> bB = buffer_for(B);
    id<MTLBuffer> bC = buffer_for(C);
    const NSUInteger oA = buffer_offset_for(A);
    const NSUInteger oB = buffer_offset_for(B);
    const NSUInteger oC = buffer_offset_for(C);
    const uint32_t Mu = M, Nu = N, Ku = Kdim;
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

void flash_attention_forward_gpu(const GpuTensor& Q,
                                 const GpuTensor& K,
                                 const GpuTensor& V,
                                 const float* d_mask,
                                 int num_heads,
                                 GpuTensor& O) {
    if (Q.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_forward_gpu: Q, K, V must be FP16");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward_gpu: num_heads must divide D");
    }
    const int head_dim = D / num_heads;
    if ((head_dim + (int)FA_BLOCK - 1) / (int)FA_BLOCK > 8) {
        throw std::runtime_error("flash_attention_forward_gpu: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    id<MTLComputePipelineState> pso = pso_flash();
    id<MTLBuffer> bQ = buffer_for(Q);
    id<MTLBuffer> bK = buffer_for(K);
    id<MTLBuffer> bV = buffer_for(V);
    id<MTLBuffer> bO = buffer_for(O);
    const NSUInteger oQ = buffer_offset_for(Q);
    const NSUInteger oK = buffer_offset_for(K);
    const NSUInteger oV = buffer_offset_for(V);
    const NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bQ;
    NSUInteger oM_arg = bM ? oM : oQ;

    const uint32_t Lqu = Lq, Lku = Lk, Du = D, hdU = head_dim;
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const NSUInteger shmem = (FA_KTILE + FA_BLOCK) * sizeof(float);

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
            threadsPerThreadgroup:MTLSizeMake(FA_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void flash_attention_qkvo_forward_gpu(const GpuTensor& X,
                                      const GpuTensor* Ctx,
                                      const GpuTensor& Wq, const GpuTensor& Wk,
                                      const GpuTensor& Wv, const GpuTensor& Wo,
                                      const float* d_mask,
                                      int num_heads,
                                      GpuTensor& O) {
    if (X.dtype != Dtype::FP16 || Wq.dtype != Dtype::FP16 ||
        Wk.dtype != Dtype::FP16 || Wv.dtype != Dtype::FP16 ||
        Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const GpuTensor& kv_src = Ctx ? *Ctx : X;
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: Ctx must be FP16");
    }
    const int Lk = kv_src.rows;
    if (kv_src.cols != D ||
        Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_forward_gpu: num_heads must divide D");
    }

    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    GpuTensor Qp(Lq, D, Dtype::FP16);
    GpuTensor Kp(Lk, D, Dtype::FP16);
    GpuTensor Vp(Lk, D, Dtype::FP16);
    GpuTensor Op(Lq, D, Dtype::FP16);

    launch_matmul_abt(X,      Wq, Qp, Lq, D, D);
    launch_matmul_abt(kv_src, Wk, Kp, Lk, D, D);
    launch_matmul_abt(kv_src, Wv, Vp, Lk, D, D);

    flash_attention_forward_gpu(Qp, Kp, Vp, d_mask, num_heads, Op);

    launch_matmul_abt(Op, Wo, O, Lq, D, D);
}

} // namespace brotensor
