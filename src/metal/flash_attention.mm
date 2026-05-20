#include <brotensor/runtime.h>
#include <brotensor/detail/op_table.h>

#include <stdexcept>
#include <vector>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

// Forward declarations of sibling Metal ops implemented in other TUs (and of
// ops defined later in this file). Generated from the canonical op table.
#define BROTENSOR_METAL_DECL(name, ret, params) ret name params;
BROTENSOR_FOR_EACH_OP(BROTENSOR_METAL_DECL)
#undef BROTENSOR_METAL_DECL

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

// Causal-aware row-wise softmax: same as k_scale_mask_softmax_rows but with
// an extra "k > q -> -inf" clause when causal != 0. Used by the bwd recompute.
kernel void k_scale_mask_causal_softmax_rows(
        device half*       S    [[buffer(0)]],    // (Lq, Lk)
        device const float* mask [[buffer(1)]],    // (Lk,) may be dummy
        constant uint& Lq        [[buffer(2)]],
        constant uint& Lk        [[buffer(3)]],
        constant float& scale    [[buffer(4)]],
        constant uint& has_mask  [[buffer(5)]],
        constant uint& causal    [[buffer(6)]],
        threadgroup float* ssm   [[threadgroup(0)]],
        uint3 gid  [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 tgs3 [[threads_per_threadgroup]]) {
    uint q = gid.x;
    uint tid = tid3.x;
    uint tg = tgs3.x;
    device half* row = S + (ulong)q * (ulong)Lk;

    float local_max = -1e30f;
    for (uint k = tid; k < Lk; k += tg) {
        float v = float(row[k]) * scale;
        if (has_mask != 0u && mask[k] <= 0.5f) v = -1e30f;
        if (causal != 0u && k > q) v = -1e30f;
        if (v > local_max) local_max = v;
    }
    ssm[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s) { float o = ssm[tid + s]; if (o > ssm[tid]) ssm[tid] = o; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rmax = ssm[0];
    bool empty = (rmax <= -1e29f);

    float local_sum = 0.0f;
    for (uint k = tid; k < Lk; k += tg) {
        float v = float(row[k]) * scale;
        if (has_mask != 0u && mask[k] <= 0.5f) v = -1e30f;
        if (causal != 0u && k > q) v = -1e30f;
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
    for (uint k = tid; k < Lk; k += tg) {
        float e = float(row[k]);
        row[k] = half(e * inv);
    }
}

// dP[q,k] = sum_d dOh[q,d] * Vh[k,d]    (Lq, Lk)
kernel void k_fa_dP(
        device const half* dOh [[buffer(0)]],   // (Lq, hd)
        device const half* Vh  [[buffer(1)]],   // (Lk, hd)
        device half*       dP  [[buffer(2)]],   // (Lq, Lk)
        constant uint& Lq      [[buffer(3)]],
        constant uint& Lk      [[buffer(4)]],
        constant uint& hd      [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x;
    uint q = gid.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (uint d = 0; d < hd; ++d) {
        acc += float(dOh[q * hd + d]) * float(Vh[k * hd + d]);
    }
    dP[q * Lk + k] = half(acc);
}

// In-place dS over P: dS[q,k] = P[q,k] * (dP[q,k] - D_q) * scale
// where D_q = sum_k P[q,k]*dP[q,k]. One threadgroup per query row.
kernel void k_fa_dS_from_P_dP(
        device half*       P_dS [[buffer(0)]],   // (Lq, Lk) in-place
        device const half* dP   [[buffer(1)]],   // (Lq, Lk)
        constant uint& Lq       [[buffer(2)]],
        constant uint& Lk       [[buffer(3)]],
        constant float& scale   [[buffer(4)]],
        threadgroup float* ssm  [[threadgroup(0)]],
        uint3 gid  [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 tgs3 [[threads_per_threadgroup]]) {
    uint q = gid.x;
    uint tid = tid3.x;
    uint tg = tgs3.x;
    device half* prow = P_dS + (ulong)q * (ulong)Lk;
    device const half* dprow = dP + (ulong)q * (ulong)Lk;

    float local = 0.0f;
    for (uint k = tid; k < Lk; k += tg) {
        local += float(prow[k]) * float(dprow[k]);
    }
    ssm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float Dq = ssm[0];
    for (uint k = tid; k < Lk; k += tg) {
        float p  = float(prow[k]);
        float dp = float(dprow[k]);
        prow[k] = half(p * (dp - Dq) * scale);
    }
}

// dVh[k,d] = sum_q P[q,k] * dOh[q,d]   (Lk, hd) (overwrite)
kernel void k_fa_dVh(
        device const half* P   [[buffer(0)]],   // (Lq, Lk)
        device const half* dOh [[buffer(1)]],   // (Lq, hd)
        device half*       dVh [[buffer(2)]],   // (Lk, hd)
        constant uint& Lq      [[buffer(3)]],
        constant uint& Lk      [[buffer(4)]],
        constant uint& hd      [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint k = gid.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (uint q = 0; q < Lq; ++q) {
        acc += float(P[q * Lk + k]) * float(dOh[q * hd + d]);
    }
    dVh[k * hd + d] = half(acc);
}

// dQh[q,d] = sum_k dS[q,k] * Kh[k,d]
kernel void k_fa_dQh(
        device const half* dS  [[buffer(0)]],   // (Lq, Lk)
        device const half* Kh  [[buffer(1)]],   // (Lk, hd)
        device half*       dQh [[buffer(2)]],   // (Lq, hd)
        constant uint& Lq      [[buffer(3)]],
        constant uint& Lk      [[buffer(4)]],
        constant uint& hd      [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint q = gid.y;
    if (q >= Lq || d >= hd) return;
    float acc = 0.0f;
    for (uint k = 0; k < Lk; ++k) {
        acc += float(dS[q * Lk + k]) * float(Kh[k * hd + d]);
    }
    dQh[q * hd + d] = half(acc);
}

// dKh[k,d] = sum_q dS[q,k] * Qh[q,d]
kernel void k_fa_dKh(
        device const half* dS  [[buffer(0)]],   // (Lq, Lk)
        device const half* Qh  [[buffer(1)]],   // (Lq, hd)
        device half*       dKh [[buffer(2)]],   // (Lk, hd)
        constant uint& Lq      [[buffer(3)]],
        constant uint& Lk      [[buffer(4)]],
        constant uint& hd      [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x;
    uint k = gid.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (uint q = 0; q < Lq; ++q) {
        acc += float(dS[q * Lk + k]) * float(Qh[q * hd + d]);
    }
    dKh[k * hd + d] = half(acc);
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
id<MTLComputePipelineState> pso_softmax_rows_causal() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_scale_mask_causal_softmax_rows"); });
    return pso;
}
id<MTLComputePipelineState> pso_fa_dP() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_fa_dP"); });
    return pso;
}
id<MTLComputePipelineState> pso_fa_dS() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_fa_dS_from_P_dP"); });
    return pso;
}
id<MTLComputePipelineState> pso_fa_dVh() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_fa_dVh"); });
    return pso;
}
id<MTLComputePipelineState> pso_fa_dQh() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_fa_dQh"); });
    return pso;
}
id<MTLComputePipelineState> pso_fa_dKh() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_fa_dKh"); });
    return pso;
}

void run_causal_flash(const Tensor& Q,
                      const Tensor& K,
                      const Tensor& V,
                      const float* d_mask,
                      int num_heads,
                      Tensor& O,
                      int Lq, int Lk, int D, int head_dim) {
    if ((head_dim + (int)FA_BLOCK - 1) / (int)FA_BLOCK > 8) {
        throw std::runtime_error("flash_attention_forward: head_dim too large for register tile (max 8 * FA_BLOCK = 1024)");
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

void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O) {
    if (Q.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_forward: Q, K, V must be FP16");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward: causal requires Lq == Lk");
    }
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward: num_heads must divide D");
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
    thread_local static Tensor Qh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Kh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Vth = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor S = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Oh = Tensor::empty_on(Device::Metal, 0, 0);
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

void flash_attention_qkvo_forward(const Tensor& X,
                                  const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask,
                                  int num_heads,
                                  bool causal,
                                  Tensor& O) {
    if (X.dtype != Dtype::FP16 || Wq.dtype != Dtype::FP16 ||
        Wk.dtype != Dtype::FP16 || Wv.dtype != Dtype::FP16 ||
        Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_forward: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_forward: Ctx must be FP16");
    }
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_forward: num_heads must divide D");
    }

    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Kp = Tensor::empty_on(Device::Metal, Lk, D, Dtype::FP16);
    Tensor Vp = Tensor::empty_on(Device::Metal, Lk, D, Dtype::FP16);

    flash_attention_project_kv(kv_src, Wk, bk, Wv, bv, Kp, Vp);
    flash_attention_q_with_kv_cached_forward(
        X, Kp, Vp, Wq, bq, Wo, bo, d_mask, num_heads, causal, O);
}

void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out,
                                Tensor& V_out) {
    if (ctx.dtype != Dtype::FP16 || Wk.dtype != Dtype::FP16 ||
        Wv.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_project_kv: all tensors must be FP16");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk.rows;
    if (Wk.cols != D_ctx || Wv.rows != D || Wv.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != Dtype::FP16) {
        K_out.resize(Lk, D, Dtype::FP16);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != Dtype::FP16) {
        V_out.resize(Lk, D, Dtype::FP16);
    }
    if (Lk == 0 || D == 0) return;
    linear_forward_batched_fp16(Wk, bk, ctx, K_out);
    linear_forward_batched_fp16(Wv, bv, ctx, V_out);
}

void flash_attention_q_with_kv_cached_forward(const Tensor& X,
                                              const Tensor& K,
                                              const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask,
                                              int num_heads,
                                              bool causal,
                                              Tensor& O) {
    if (X.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16 ||
        Wq.dtype != Dtype::FP16 || Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: K/V shape mismatch");
    }
    if (Wq.rows != D || Wq.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_forward: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Qp = Tensor::empty_on(Device::Metal, Lq, D, Dtype::FP16);
    Tensor Op = Tensor::empty_on(Device::Metal, Lq, D, Dtype::FP16);

    linear_forward_batched_fp16(Wq, bq, X, Qp);
    flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_fp16(Wo, bo, Op, O);
}

// ─── W8A16 variants of the three fused flash-attention ops ─────────────────
// See src/cuda/flash_attention.cu for documentation.

void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8,
                                           const Tensor& sk,
                                           const Tensor* bk,
                                           const Tensor& Wv_int8,
                                           const Tensor& sv,
                                           const Tensor* bv,
                                           Tensor& K_out,
                                           Tensor& V_out) {
    if (ctx.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: ctx must be FP16");
    }
    if (Wk_int8.dtype != Dtype::INT8 || Wv_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: Wk/Wv must be INT8");
    }
    const int Lk = ctx.rows;
    const int D_ctx = ctx.cols;
    const int D = Wk_int8.rows;
    if (Wk_int8.cols != D_ctx || Wv_int8.rows != D || Wv_int8.cols != D_ctx) {
        throw std::runtime_error("flash_attention_project_kv_int8w_fp16: Wk/Wv shape mismatch");
    }
    if (K_out.rows != Lk || K_out.cols != D || K_out.dtype != Dtype::FP16) {
        K_out.resize(Lk, D, Dtype::FP16);
    }
    if (V_out.rows != Lk || V_out.cols != D || V_out.dtype != Dtype::FP16) {
        V_out.resize(Lk, D, Dtype::FP16);
    }
    if (Lk == 0 || D == 0) return;
    linear_forward_batched_int8w_fp16(Wk_int8, sk, bk, ctx, K_out);
    linear_forward_batched_int8w_fp16(Wv_int8, sv, bv, ctx, V_out);
}

void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X,
                                                 const Tensor& K,
                                                 const Tensor& V,
                                                 const Tensor& Wq_int8,
                                                 const Tensor& sq,
                                                 const Tensor* bq,
                                                 const Tensor& Wo_int8,
                                                 const Tensor& so,
                                                 const Tensor* bo,
                                                 const float* d_mask,
                                                 int num_heads,
                                                 bool causal,
                                                 Tensor& O) {
    if (X.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 || V.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: X/K/V must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: Wq/Wo must be INT8");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const int Lk = K.rows;
    if (K.cols != D || V.rows != Lk || V.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: K/V shape mismatch");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: Wq/Wo shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_q_with_kv_cached_int8w_fp16: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Qp = Tensor::empty_on(Device::Metal, Lq, D, Dtype::FP16);
    Tensor Op = Tensor::empty_on(Device::Metal, Lq, D, Dtype::FP16);

    linear_forward_batched_int8w_fp16(Wq_int8, sq, bq, X, Qp);
    flash_attention_forward(Qp, K, V, d_mask, num_heads, causal, Op);
    linear_forward_batched_int8w_fp16(Wo_int8, so, bo, Op, O);
}

void flash_attention_qkvo_int8w_fp16(const Tensor& X,
                                     const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask,
                                     int num_heads,
                                     bool causal,
                                     Tensor& O) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: X must be FP16");
    }
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: Ctx must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: all weights must be INT8");
    }
    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D_ctx ||
        Wv_int8.rows != D || Wv_int8.cols != D_ctx ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_int8w_fp16: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    Tensor Kp = Tensor::empty_on(Device::Metal, Lk, D, Dtype::FP16);
    Tensor Vp = Tensor::empty_on(Device::Metal, Lk, D, Dtype::FP16);
    flash_attention_project_kv_int8w_fp16(kv_src,
                                          Wk_int8, sk, bk,
                                          Wv_int8, sv, bv,
                                          Kp, Vp);
    flash_attention_q_with_kv_cached_int8w_fp16(
        X, Kp, Vp,
        Wq_int8, sq, bq,
        Wo_int8, so, bo,
        d_mask, num_heads, causal, O);
}

// Recompute-style FP16 backward. Mirrors the CUDA implementation in
// src/cuda/flash_attention.cu (flash_attention_qkvo_backward_gpu, lines 724+):
//   1. Re-project X→Q, kv_src→K, kv_src→V.
//   2. Per-head: extract, S = Qh·Kh^T, softmax → P, Oh = P·Vh; pack into O_attn.
//   3. Wo backward via linear_backward_batched_gpu → dO_attn.
//   4. Per-head: extract dOh; recompute P; dVh = P^T·dOh; dP = dOh·Vh^T;
//                dS = P*(dP-D_q)*inv_sqrt (in-place over P); dQh = dS·Kh;
//                dKh = dS^T·Qh; pack dQ/dK/dV.
//   5. Linear backward for Q/K/V projections, accumulate into dX (and dCtx).
void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask,
    int num_heads,
    bool causal,
    const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo) {

    // ── Argument validation ──────────────────────────────────────────────
    if (X.dtype != Dtype::FP16 || dO.dtype != Dtype::FP16 ||
        Wq.dtype != Dtype::FP16 || Wk.dtype != Dtype::FP16 ||
        Wv.dtype != Dtype::FP16 || Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_backward: all tensors must be FP16");
    }
    if (Ctx && Ctx->dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_qkvo_backward: Ctx must be FP16");
    }
    const bool self_attn = (Ctx == nullptr);
    if (self_attn) {
        if (dCtx != nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be null when Ctx is null");
        }
    } else {
        if (dCtx == nullptr) {
            throw std::runtime_error("flash_attention_qkvo_backward: dCtx must be non-null when Ctx is non-null");
        }
    }
    auto bias_pair_ok = [](const Tensor* b, const Tensor* db) {
        return static_cast<bool>(b) == static_cast<bool>(db);
    };
    if (!bias_pair_ok(bq, dbq) || !bias_pair_ok(bk, dbk) ||
        !bias_pair_ok(bv, dbv) || !bias_pair_ok(bo, dbo)) {
        throw std::runtime_error("flash_attention_qkvo_backward: bias/grad-bias presence mismatch");
    }

    const int Lq = X.rows;
    const int D  = X.cols;
    const Tensor& kv_src = Ctx ? *Ctx : X;
    const int Lk = kv_src.rows;
    const int D_ctx = kv_src.cols;
    if (Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D_ctx ||
        Wv.rows != D || Wv.cols != D_ctx ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward: shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_qkvo_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_qkvo_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_qkvo_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dX.rows != Lq || dX.cols != D || dX.dtype != Dtype::FP16) {
        dX.resize(Lq, D, Dtype::FP16);
    }
    if (!self_attn) {
        if (dCtx->rows != Lk || dCtx->cols != D_ctx || dCtx->dtype != Dtype::FP16) {
            dCtx->resize(Lk, D_ctx, Dtype::FP16);
        }
        dCtx->zero();
    }
    dX.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    // ── Scratch tensors (reused across calls). ────────────────────────────
    thread_local static Tensor Qp = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Kp = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Vp = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Qh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Kh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Vh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Vth = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dOh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor S = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor P_main = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dP = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor Oh_scratch = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor O_attn = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dO_attn = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dQh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dKh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dVh = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dQ = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dK = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dV = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dX_from_Q = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dX_from_K = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor dX_from_V = Tensor::empty_on(Device::Metal, 0, 0);
    thread_local static Tensor scratch_db = Tensor::empty_on(Device::Metal, 0, 0);

    auto ensure = [](Tensor& t, int r, int c) {
        if (t.rows != r || t.cols != c || t.dtype != Dtype::FP16) {
            t.resize(r, c, Dtype::FP16);
        }
    };

    ensure(Qp, Lq, D);
    ensure(Kp, Lk, D);
    ensure(Vp, Lk, D);
    ensure(Qh, Lq, hd);
    ensure(Kh, Lk, hd);
    ensure(Vh, Lk, hd);
    ensure(Vth, hd, Lk);
    ensure(dOh, Lq, hd);
    ensure(S, Lq, Lk);
    ensure(P_main, Lq, Lk);
    ensure(dP, Lq, Lk);
    ensure(Oh_scratch, Lq, hd);
    ensure(O_attn, Lq, D);
    ensure(dO_attn, Lq, D);
    ensure(dQh, Lq, hd);
    ensure(dKh, Lk, hd);
    ensure(dVh, Lk, hd);
    ensure(dQ, Lq, D);
    ensure(dK, Lk, D);
    ensure(dV, Lk, D);
    ensure(dX_from_Q, Lq, D);
    ensure(dX_from_K, Lk, D_ctx);
    ensure(dX_from_V, Lk, D_ctx);

    // dQ/dK/dV must be zeroed before per-head pack-back (we use overwrite-style
    // per-head packs which only write the head's column slot; other heads'
    // slots are not touched in that step, but earlier-iteration data lingers
    // across calls because the buffer is reused).
    dQ.zero();
    dK.zero();
    dV.zero();

    // ── 1. Recompute forward projections. ─────────────────────────────────
    linear_forward_batched_fp16(Wq, bq, X,      Qp);
    linear_forward_batched_fp16(Wk, bk, kv_src, Kp);
    linear_forward_batched_fp16(Wv, bv, kv_src, Vp);

    // Common encode plumbing.
    id<MTLComputePipelineState> p_ext_LD = pso_extract_LD();
    id<MTLComputePipelineState> p_ext_DL = pso_extract_DL();
    id<MTLComputePipelineState> p_pack   = pso_pack_LD();
    id<MTLComputePipelineState> p_sm_c   = pso_softmax_rows_causal();
    id<MTLComputePipelineState> p_dP     = pso_fa_dP();
    id<MTLComputePipelineState> p_dS     = pso_fa_dS();
    id<MTLComputePipelineState> p_dVh    = pso_fa_dVh();
    id<MTLComputePipelineState> p_dQh    = pso_fa_dQh();
    id<MTLComputePipelineState> p_dKh    = pso_fa_dKh();

    id<MTLBuffer> bQp = buffer_for(Qp);     NSUInteger oQp = buffer_offset_for(Qp);
    id<MTLBuffer> bKp = buffer_for(Kp);     NSUInteger oKp = buffer_offset_for(Kp);
    id<MTLBuffer> bVp = buffer_for(Vp);     NSUInteger oVp = buffer_offset_for(Vp);
    id<MTLBuffer> bQh = buffer_for(Qh);     NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh);     NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh);     NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bVt = buffer_for(Vth);    NSUInteger oVt = buffer_offset_for(Vth);
    id<MTLBuffer> bdOh= buffer_for(dOh);    NSUInteger odOh= buffer_offset_for(dOh);
    id<MTLBuffer> bS  = buffer_for(S);      NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bP  = buffer_for(P_main); NSUInteger oP  = buffer_offset_for(P_main);
    id<MTLBuffer> bdP = buffer_for(dP);     NSUInteger odP = buffer_offset_for(dP);
    id<MTLBuffer> bOh = buffer_for(Oh_scratch); NSUInteger oOh = buffer_offset_for(Oh_scratch);
    id<MTLBuffer> bOA = buffer_for(O_attn); NSUInteger oOA = buffer_offset_for(O_attn);
    id<MTLBuffer> bdOA= buffer_for(dO_attn);NSUInteger odOA= buffer_offset_for(dO_attn);
    id<MTLBuffer> bdQh= buffer_for(dQh);    NSUInteger odQh= buffer_offset_for(dQh);
    id<MTLBuffer> bdKh= buffer_for(dKh);    NSUInteger odKh= buffer_offset_for(dKh);
    id<MTLBuffer> bdVh= buffer_for(dVh);    NSUInteger odVh= buffer_offset_for(dVh);
    id<MTLBuffer> bdQ = buffer_for(dQ);     NSUInteger odQ = buffer_offset_for(dQ);
    id<MTLBuffer> bdK = buffer_for(dK);     NSUInteger odK = buffer_offset_for(dK);
    id<MTLBuffer> bdV = buffer_for(dV);     NSUInteger odV = buffer_offset_for(dV);

    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bS;
    NSUInteger oM_arg = bM ? oM : oS;
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const uint32_t causal_u = causal ? 1u : 0u;

    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));
    const uint32_t Lqu = Lq, Lku = Lk, Du = D, hdU = hd;

    NSUInteger sm_tg = 32;
    while ((int)sm_tg < Lk && sm_tg < 1024) sm_tg *= 2;
    if (sm_tg > 1024) sm_tg = 1024;
    const NSUInteger sm_shmem = sm_tg * sizeof(float);

    // ── 2. Recompute O_attn (Lq, D) by running fwd per head. ─────────────
    for (int h = 0; h < num_heads; ++h) {
        const uint32_t head_off = (uint32_t)h * (uint32_t)hd;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_per_elem(enc, p_ext_LD, bQp, oQp, bQh, oQh, Lqu, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_LD, bKp, oKp, bKh, oKh, Lku, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_DL, bVp, oVp, bVt, oVt, Lku, Du, head_off, hdU);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // S = Qh @ Kh^T
        launch_matmul_abt_fp16(bQh, oQh, bKh, oKh, bS, oS, Lq, Lk, hd);

        // S = softmax(scale * S, mask, causal)
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:p_sm_c];
            [enc setBuffer:bS offset:oS atIndex:0];
            [enc setBuffer:bM_arg offset:oM_arg atIndex:1];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&inv_sqrt length:sizeof(float) atIndex:4];
            [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&causal_u length:sizeof(uint32_t) atIndex:6];
            [enc setThreadgroupMemoryLength:sm_shmem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(Lq, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(sm_tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Oh = S @ Vth^T  (N = hd, K = Lk)
        launch_matmul_abt_fp16(bS, oS, bVt, oVt, bOh, oOh, Lq, hd, Lk);

        // Pack Oh back into O_attn slot.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_per_elem(enc, p_pack, bOh, oOh, bOA, oOA, Lqu, Du, head_off, hdU);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    // ── 3. Wo + bo backward. dO_attn(Lq, D) = dO @ Wo. ────────────────────
    {
        const bool has_bo = (bo != nullptr);
        if (!has_bo) {
            if (scratch_db.rows != D || scratch_db.cols != 1 || scratch_db.dtype != Dtype::FP16) {
                scratch_db.resize(D, 1, Dtype::FP16);
            }
            scratch_db.zero();
        }
        linear_backward_batched(Wo, O_attn, dO, dO_attn, dWo,
                                has_bo ? *dbo : scratch_db);
    }

    // ── 4. Per-head backward sweep. ───────────────────────────────────────
    for (int h = 0; h < num_heads; ++h) {
        const uint32_t head_off = (uint32_t)h * (uint32_t)hd;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_per_elem(enc, p_ext_LD, bQp, oQp, bQh, oQh, Lqu, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_LD, bKp, oKp, bKh, oKh, Lku, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_LD, bVp, oVp, bVh, oVh, Lku, Du, head_off, hdU);
            encode_per_elem(enc, p_ext_LD, bdOA, odOA, bdOh, odOh, Lqu, Du, head_off, hdU);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Recompute P = softmax(scale * Qh@Kh^T, mask, causal).
        launch_matmul_abt_fp16(bQh, oQh, bKh, oKh, bP, oP, Lq, Lk, hd);
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:p_sm_c];
            [enc setBuffer:bP offset:oP atIndex:0];
            [enc setBuffer:bM_arg offset:oM_arg atIndex:1];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&inv_sqrt length:sizeof(float) atIndex:4];
            [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&causal_u length:sizeof(uint32_t) atIndex:6];
            [enc setThreadgroupMemoryLength:sm_shmem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(Lq, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(sm_tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // dVh, dP — independent, can share one command buffer.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // dVh(Lk, hd): grid (hd, Lk).
            [enc setComputePipelineState:p_dVh];
            [enc setBuffer:bP offset:oP atIndex:0];
            [enc setBuffer:bdOh offset:odOh atIndex:1];
            [enc setBuffer:bdVh offset:odVh atIndex:2];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&hdU length:sizeof(uint32_t) atIndex:5];
            {
                NSUInteger gx = ((hd + 15) / 16) * 16;
                NSUInteger gy = ((Lk + 15) / 16) * 16;
                [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            }

            // dP(Lq, Lk): grid (Lk, Lq).
            [enc setComputePipelineState:p_dP];
            [enc setBuffer:bdOh offset:odOh atIndex:0];
            [enc setBuffer:bVh offset:oVh atIndex:1];
            [enc setBuffer:bdP offset:odP atIndex:2];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&hdU length:sizeof(uint32_t) atIndex:5];
            {
                NSUInteger gx = ((Lk + 15) / 16) * 16;
                NSUInteger gy = ((Lq + 15) / 16) * 16;
                [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // dS = P * (dP - D_q) * inv_sqrt  (in-place over P).
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:p_dS];
            [enc setBuffer:bP offset:oP atIndex:0];
            [enc setBuffer:bdP offset:odP atIndex:1];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&inv_sqrt length:sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:sm_shmem atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(Lq, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(sm_tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // dQh, dKh.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:p_dQh];
            [enc setBuffer:bP offset:oP atIndex:0];
            [enc setBuffer:bKh offset:oKh atIndex:1];
            [enc setBuffer:bdQh offset:odQh atIndex:2];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&hdU length:sizeof(uint32_t) atIndex:5];
            {
                NSUInteger gx = ((hd + 15) / 16) * 16;
                NSUInteger gy = ((Lq + 15) / 16) * 16;
                [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            }

            [enc setComputePipelineState:p_dKh];
            [enc setBuffer:bP offset:oP atIndex:0];
            [enc setBuffer:bQh offset:oQh atIndex:1];
            [enc setBuffer:bdKh offset:odKh atIndex:2];
            [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&hdU length:sizeof(uint32_t) atIndex:5];
            {
                NSUInteger gx = ((hd + 15) / 16) * 16;
                NSUInteger gy = ((Lk + 15) / 16) * 16;
                [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
                  threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Pack dQh / dKh / dVh back into per-batch dQ / dK / dV.
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_per_elem(enc, p_pack, bdQh, odQh, bdQ, odQ, Lqu, Du, head_off, hdU);
            encode_per_elem(enc, p_pack, bdKh, odKh, bdK, odK, Lku, Du, head_off, hdU);
            encode_per_elem(enc, p_pack, bdVh, odVh, bdV, odV, Lku, Du, head_off, hdU);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    // ── 5. Q / K / V projection backward. ─────────────────────────────────
    auto run_proj_back = [&](const Tensor& W, const Tensor& In,
                             const Tensor& dOut, Tensor& dIn_out,
                             Tensor& dW_acc, const Tensor* b_fwd,
                             Tensor* db_acc) {
        const bool has_b = (b_fwd != nullptr);
        if (!has_b) {
            if (scratch_db.rows != W.rows || scratch_db.cols != 1 ||
                scratch_db.dtype != Dtype::FP16) {
                scratch_db.resize(W.rows, 1, Dtype::FP16);
            }
            scratch_db.zero();
        }
        linear_backward_batched(W, In, dOut, dIn_out, dW_acc,
                                has_b ? *db_acc : scratch_db);
    };

    run_proj_back(Wq, X,      dQ, dX_from_Q, dWq, bq, dbq);
    run_proj_back(Wk, kv_src, dK, dX_from_K, dWk, bk, dbk);
    run_proj_back(Wv, kv_src, dV, dX_from_V, dWv, bv, dbv);

    // ── 6. Accumulate into dX / dCtx. ─────────────────────────────────────
    add_inplace(dX, dX_from_Q);
    if (self_attn) {
        add_inplace(dX, dX_from_K);
        add_inplace(dX, dX_from_V);
    } else {
        add_inplace(*dCtx, dX_from_K);
        add_inplace(*dCtx, dX_from_V);
    }
}

} // namespace brotensor::detail::metal
