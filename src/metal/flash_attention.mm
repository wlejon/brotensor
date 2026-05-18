#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;
using metal_impl::launch_matmul_abt_fp16;

namespace {

constexpr NSUInteger FA_BLOCK = 128;
constexpr NSUInteger FA_KTILE = 64;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint FA_BLOCK = 128;
constant uint FA_KTILE = 64;
constant uint MAX_HD_PER_THREAD = 8;

// Flash-attention online-softmax kernel. One threadgroup per (q, head).
// Tiles K/V along Lk (FA_KTILE = 64). Per-thread partial output kept in
// registers (`partial[]`), strided over head_dim with up to MAX_HD_PER_THREAD
// slots/thread (FA_BLOCK * MAX_HD_PER_THREAD >= head_dim required).
//
// Threadgroup memory layout (dynamic):
//   scratch[0..FA_KTILE)    - tile scores
//   scratch[FA_KTILE..FA_KTILE+FA_BLOCK) - reduction scratch
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
        constant uint& causal    [[buffer(10)]],
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
        if (causal != 0u && k0 > q) break;
        uint klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal != 0u && k0 + klen - 1u > q) klen = q - k0 + 1u;

        // 1. scores[t] = Q[q] . K[k0+t] * inv_sqrt
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

// ---- Per-head extract / pack-back kernels (mirror cuda flash_attention) ----
//
// X is (L, D) with D = num_heads * head_dim. The matmul fast path wants a
// contiguous (L, head_dim) view per head.

kernel void k_extract_head_LD(
        device const half* X   [[buffer(0)]],     // (L, D)
        device half*       Y   [[buffer(1)]],     // (L, head_dim)
        constant uint& L         [[buffer(2)]],
        constant uint& D         [[buffer(3)]],
        constant uint& head_off  [[buffer(4)]],
        constant uint& head_dim  [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    uint total = L * head_dim;
    if (gid >= total) return;
    uint l = gid / head_dim;
    uint d = gid % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

// Extract a single head and TRANSPOSE: Y is (head_dim, L) with element
// (d, l) = X[l, head_off + d]. Feeds the second matmul as B-operand.
kernel void k_extract_head_DL(
        device const half* X   [[buffer(0)]],     // (L, D)
        device half*       Y   [[buffer(1)]],     // (head_dim, L)
        constant uint& L         [[buffer(2)]],
        constant uint& D         [[buffer(3)]],
        constant uint& head_off  [[buffer(4)]],
        constant uint& head_dim  [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    uint total = L * head_dim;
    if (gid >= total) return;
    uint l = gid / head_dim;
    uint d = gid % head_dim;
    Y[d * L + l] = X[l * D + head_off + d];
}

// Inverse of extract_head_LD: write per-head (Lq, head_dim) back into the
// (Lq, D) output at column slot [head_off, head_off+head_dim).
kernel void k_pack_head_LD(
        device const half* Yh  [[buffer(0)]],     // (L, head_dim)
        device half*       Out [[buffer(1)]],     // (L, D)
        constant uint& L         [[buffer(2)]],
        constant uint& D         [[buffer(3)]],
        constant uint& head_off  [[buffer(4)]],
        constant uint& head_dim  [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    uint total = L * head_dim;
    if (gid >= total) return;
    uint l = gid / head_dim;
    uint d = gid % head_dim;
    Out[l * D + head_off + d] = Yh[l * head_dim + d];
}

// Row-wise softmax over S(Lq, Lk) with scalar scale (1/sqrt(head_dim)) and
// optional Lk-shaped float mask (positions with mask[k] <= 0.5 -> -inf).
// One threadgroup per query row; threadgroup size chosen by host.
kernel void k_scale_mask_softmax_rows(
        device half*       S    [[buffer(0)]],    // (Lq, Lk)
        device const float* mask [[buffer(1)]],    // (Lk,) may be dummy
        constant uint& Lq        [[buffer(2)]],
        constant uint& Lk        [[buffer(3)]],
        constant float& scale    [[buffer(4)]],
        constant uint& has_mask  [[buffer(5)]],
        threadgroup float* ssm   [[threadgroup(0)]],   // size = tg_size
        uint3 gid  [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 tgs3 [[threads_per_threadgroup]]) {
    uint q = gid.x;
    uint tid = tid3.x;
    uint tg = tgs3.x;
    device half* row = S + (ulong)q * (ulong)Lk;

    // 1. row max with scale and mask applied.
    float local_max = -1e30f;
    for (uint k = tid; k < Lk; k += tg) {
        float v = float(row[k]) * scale;
        if (has_mask != 0u && mask[k] <= 0.5f) v = -1e30f;
        if (v > local_max) local_max = v;
    }
    ssm[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float o = ssm[tid + s];
            if (o > ssm[tid]) ssm[tid] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rmax = ssm[0];
    bool empty = (rmax <= -1e29f);

    // 2. exponentiate, accumulate sum, write back exp(v - rmax).
    float local_sum = 0.0f;
    for (uint k = tid; k < Lk; k += tg) {
        float v = float(row[k]) * scale;
        if (has_mask != 0u && mask[k] <= 0.5f) v = -1e30f;
        float e = empty ? 0.0f : exp(v - rmax);
        row[k] = half(e);
        local_sum += e;
    }
    ssm[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rsum = ssm[0];
    float inv = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;

    // 3. normalise.
    for (uint k = tid; k < Lk; k += tg) {
        float e = float(row[k]);
        row[k] = half(e * inv);
    }
}
)msl";

id<MTLComputePipelineState> pso_flash() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_flash_attention"); });
    return pso;
}
id<MTLComputePipelineState> pso_extract_LD() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_extract_head_LD"); });
    return pso;
}
id<MTLComputePipelineState> pso_extract_DL() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_extract_head_DL"); });
    return pso;
}
id<MTLComputePipelineState> pso_pack_LD() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_pack_head_LD"); });
    return pso;
}
id<MTLComputePipelineState> pso_softmax_rows() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_scale_mask_softmax_rows"); });
    return pso;
}

void run_causal_flash(const GpuTensor& Q,
                      const GpuTensor& K,
                      const GpuTensor& V,
                      const float* d_mask,
                      int num_heads,
                      GpuTensor& O,
                      int Lq, int Lk, int D, int head_dim) {
    if ((head_dim + (int)FA_BLOCK - 1) / (int)FA_BLOCK > 8) {
        throw std::runtime_error("flash_attention_forward_gpu: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
    }
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
    const uint32_t causal_u = 1u;
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
        [enc setBytes:&causal_u length:sizeof(uint32_t) atIndex:10];
        [enc setThreadgroupMemoryLength:shmem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(Lq, num_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(FA_BLOCK, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// Encode an extract / pack kernel onto the given encoder. n_elems = L*head_dim.
void encode_per_elem(id<MTLComputeCommandEncoder> enc,
                     id<MTLComputePipelineState> pso,
                     id<MTLBuffer> bIn,  NSUInteger oIn,
                     id<MTLBuffer> bOut, NSUInteger oOut,
                     uint32_t L, uint32_t D, uint32_t head_off, uint32_t head_dim) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:bIn  offset:oIn  atIndex:0];
    [enc setBuffer:bOut offset:oOut atIndex:1];
    [enc setBytes:&L        length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&D        length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&head_off length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:5];
    NSUInteger total = (NSUInteger)L * (NSUInteger)head_dim;
    NSUInteger tg = 256;
    NSUInteger grid = ((total + tg - 1) / tg) * tg;
    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

} // namespace

void flash_attention_forward_gpu(const GpuTensor& Q,
                                 const GpuTensor& K,
                                 const GpuTensor& V,
                                 const float* d_mask,
                                 int num_heads,
                                 bool causal,
                                 GpuTensor& O) {
    if (Q.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_forward_gpu: Q, K, V must be FP16");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward_gpu: causal requires Lq == Lk");
    }
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward_gpu: num_heads must divide D");
    }
    const int head_dim = D / num_heads;
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    // Causal path: keep existing online-softmax kernel. SD1.5 does not use
    // causal here, so the per-head matmul fast path covers production.
    if (causal) {
        run_causal_flash(Q, K, V, d_mask, num_heads, O, Lq, Lk, D, head_dim);
        return;
    }

    // ---- Per-head matmul pipeline (mirror cuda 35f72b0) ----
    // Reuse scratch tensors across calls to keep allocator pressure flat.
    // SD1.5 worst-case S buffer is 32 MB (Lq=Lk=4096, fp16).
    thread_local static GpuTensor Qh;
    thread_local static GpuTensor Kh;
    thread_local static GpuTensor Vth;
    thread_local static GpuTensor S;
    thread_local static GpuTensor Oh;
    if (Qh.rows != Lq  || Qh.cols != head_dim || Qh.dtype != Dtype::FP16) Qh.resize(Lq, head_dim, Dtype::FP16);
    if (Kh.rows != Lk  || Kh.cols != head_dim || Kh.dtype != Dtype::FP16) Kh.resize(Lk, head_dim, Dtype::FP16);
    if (Vth.rows != head_dim || Vth.cols != Lk || Vth.dtype != Dtype::FP16) Vth.resize(head_dim, Lk, Dtype::FP16);
    if (S.rows != Lq   || S.cols != Lk        || S.dtype != Dtype::FP16) S.resize(Lq, Lk, Dtype::FP16);
    if (Oh.rows != Lq  || Oh.cols != head_dim || Oh.dtype != Dtype::FP16) Oh.resize(Lq, head_dim, Dtype::FP16);

    id<MTLComputePipelineState> p_ext_LD = pso_extract_LD();
    id<MTLComputePipelineState> p_ext_DL = pso_extract_DL();
    id<MTLComputePipelineState> p_pack   = pso_pack_LD();
    id<MTLComputePipelineState> p_sm     = pso_softmax_rows();

    id<MTLBuffer> bQ  = buffer_for(Q);   NSUInteger oQ  = buffer_offset_for(Q);
    id<MTLBuffer> bK  = buffer_for(K);   NSUInteger oK  = buffer_offset_for(K);
    id<MTLBuffer> bV  = buffer_for(V);   NSUInteger oV  = buffer_offset_for(V);
    id<MTLBuffer> bO  = buffer_for(O);   NSUInteger oO  = buffer_offset_for(O);
    id<MTLBuffer> bQh = buffer_for(Qh);  NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh);  NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVt = buffer_for(Vth); NSUInteger oVt = buffer_offset_for(Vth);
    id<MTLBuffer> bS  = buffer_for(S);   NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bOh = buffer_for(Oh);  NSUInteger oOh = buffer_offset_for(Oh);

    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bS;
    NSUInteger oM_arg = bM ? oM : oS;
    const uint32_t has_mask = d_mask ? 1u : 0u;

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Softmax threadgroup size: start at 32, double until >= Lk, cap 1024.
    NSUInteger sm_tg = 32;
    while ((int)sm_tg < Lk && sm_tg < 1024) sm_tg *= 2;
    if (sm_tg > 1024) sm_tg = 1024;
    const NSUInteger sm_shmem = sm_tg * sizeof(float);

    const uint32_t Lqu = Lq, Lku = Lk, Du = D, hdU = head_dim;

    for (int h = 0; h < num_heads; ++h) {
        const uint32_t head_off = (uint32_t)h * (uint32_t)head_dim;

        // 1. Extract Qh, Kh, Vth on one command buffer.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_per_elem(enc, p_ext_LD, bQ, oQ, bQh, oQh, Lqu, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_LD, bK, oK, bKh, oKh, Lku, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_DL, bV, oV, bVt, oVt, Lku, Du, head_off, hdU);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // 2. S(Lq, Lk) = Qh(Lq, hd) @ Kh(Lk, hd)^T  — A @ B^T.
        launch_matmul_abt_fp16(bQh, oQh, bKh, oKh, bS, oS, Lq, Lk, head_dim);

        // 3. Row-wise scaled+masked softmax over S.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:p_sm];
            [enc setBuffer:bS offset:oS atIndex:0];
            [enc setBuffer:bM_arg offset:oM_arg atIndex:1];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&scale length:sizeof(float)    atIndex:4];
            [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:5];
            [enc setThreadgroupMemoryLength:sm_shmem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(Lq, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(sm_tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // 4. Oh(Lq, hd) = S(Lq, Lk) @ Vth(hd, Lk)^T  — Vth as B (N=hd, K=Lk).
        launch_matmul_abt_fp16(bS, oS, bVt, oVt, bOh, oOh, Lq, head_dim, Lk);

        // 5. Pack Oh back into O at slot [head_off, head_off+head_dim).
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_per_elem(enc, p_pack, bOh, oOh, bO, oO, Lqu, Du, head_off, hdU);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
}

void flash_attention_qkvo_forward_gpu(const GpuTensor& X,
                                      const GpuTensor* Ctx,
                                      const GpuTensor& Wq, const GpuTensor* bq,
                                      const GpuTensor& Wk, const GpuTensor* bk,
                                      const GpuTensor& Wv, const GpuTensor* bv,
                                      const GpuTensor& Wo, const GpuTensor* bo,
                                      const float* d_mask,
                                      int num_heads,
                                      bool causal,
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
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
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

    linear_forward_batched_fp16_gpu(Wq, bq, X,      Qp);
    linear_forward_batched_fp16_gpu(Wk, bk, kv_src, Kp);
    linear_forward_batched_fp16_gpu(Wv, bv, kv_src, Vp);

    flash_attention_forward_gpu(Qp, Kp, Vp, d_mask, num_heads, causal, Op);

    linear_forward_batched_fp16_gpu(Wo, bo, Op, O);
}

// Metal backward stub. The CUDA implementation composes ~7 backward kernels
// (per-head dP / dV / dS / dQ / dK plus a causal-aware softmax recompute and
// FP16 in-place add). Porting them to MSL is mechanical but several hundred
// LOC; brodiffusion training currently targets CUDA only, so this is deferred.
// We declare the symbol so the link surface matches across backends and throw
// at call time. When Metal training lands, replace the body with the MSL port
// of the CUDA path in src/cuda/flash_attention.cu.
void flash_attention_qkvo_backward_gpu(
    const GpuTensor&, const GpuTensor*,
    const GpuTensor&, const GpuTensor*,
    const GpuTensor&, const GpuTensor*,
    const GpuTensor&, const GpuTensor*,
    const GpuTensor&, const GpuTensor*,
    const float*,
    int,
    bool,
    const GpuTensor&,
    GpuTensor&, GpuTensor*,
    GpuTensor&, GpuTensor*,
    GpuTensor&, GpuTensor*,
    GpuTensor&, GpuTensor*,
    GpuTensor&, GpuTensor*) {
    throw std::runtime_error("flash_attention_qkvo_backward_gpu: Metal backend not yet implemented");
}

} // namespace brotensor
