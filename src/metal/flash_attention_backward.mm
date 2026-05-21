// Metal backend for flash_attention_backward_gpu. Recompute-based bare-core
// FlashAttention backward, mirroring src/cuda/flash_attention_backward.cu and
// the per-head helper kernels already present in src/metal/flash_attention.mm
// (which are file-local there, so we re-declare equivalents here).
//
// Per head:
//   1) Extract per-head Qh, Kh, Vh, dOh from the (L, D = nh*hd) tensors.
//   2) Recompute S = Qh · Kh^T via the FP16 matmul (M=Lq, N=Lk, K=hd, ABT).
//   3) Apply scale + optional mask + optional causal mask, then row-softmax,
//      producing P in (Lq, Lk).
//   4) dVh = P^T · dOh                              (Lk, hd)
//   5) dP  = dOh · Vh^T                             (Lq, Lk)
//   6) dS = P * (dP - D_q) * inv_sqrt   (in-place over P)
//   7) dQh = dS  · Kh                               (Lq, hd)
//   8) dKh = dS^T · Qh                              (Lk, hd)
//   9) Pack dQh / dKh / dVh back into the per-head slot of dQ / dK / dV.
//
// dQ / dK / dV are overwritten (zero-initialized before the head loop, then
// each head writes into its column slot via the pack kernel).

#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;
using metal_impl::launch_matmul_abt_fp16;

namespace {

NSString* const kBwdSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// Extract per-head (L, hd) view from (L, D) at column slot [head_off, head_off+hd).
kernel void k_fab_extract_head_LD(
        device const half* X   [[buffer(0)]],
        device half*       Y   [[buffer(1)]],
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

// Pack per-head (L, hd) buffer back into (L, D) at column slot.
kernel void k_fab_pack_head_LD(
        device const half* Yh  [[buffer(0)]],
        device half*       Out [[buffer(1)]],
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

// Row-wise scale + optional mask + optional causal softmax. One threadgroup
// per query row. Operates in-place over S (Lq, Lk).
kernel void k_fab_scale_mask_causal_softmax_rows(
        device half*       S    [[buffer(0)]],
        device const float* mask [[buffer(1)]],
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

// dP[q,k] = sum_d dOh[q,d] * Vh[k,d]
kernel void k_fab_dP(
        device const half* dOh [[buffer(0)]],
        device const half* Vh  [[buffer(1)]],
        device half*       dP  [[buffer(2)]],
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

// dS = P * (dP - D_q) * scale, in-place over P. One threadgroup per query row.
kernel void k_fab_dS_from_P_dP(
        device half*       P_dS [[buffer(0)]],
        device const half* dP   [[buffer(1)]],
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

// dVh[k,d] = sum_q P[q,k] * dOh[q,d]
kernel void k_fab_dVh(
        device const half* P   [[buffer(0)]],
        device const half* dOh [[buffer(1)]],
        device half*       dVh [[buffer(2)]],
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
kernel void k_fab_dQh(
        device const half* dS  [[buffer(0)]],
        device const half* Kh  [[buffer(1)]],
        device half*       dQh [[buffer(2)]],
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
kernel void k_fab_dKh(
        device const half* dS  [[buffer(0)]],
        device const half* Qh  [[buffer(1)]],
        device half*       dKh [[buffer(2)]],
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

// ── BF16 twins ──────────────────────────────────────────────────────────────

kernel void k_fab_extract_head_LD_bf16(
        device const bfloat* X   [[buffer(0)]],
        device bfloat*       Y   [[buffer(1)]],
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

kernel void k_fab_pack_head_LD_bf16(
        device const bfloat* Yh  [[buffer(0)]],
        device bfloat*       Out [[buffer(1)]],
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

// Naive BF16 ABT matmul. C[m,n] = sum_k A[m,k] * B[n,k]. FP32 accumulation.
kernel void k_fab_matmul_ABT_bf16(
        device const bfloat* A [[buffer(0)]],
        device const bfloat* B [[buffer(1)]],
        device bfloat*       C [[buffer(2)]],
        constant uint& M         [[buffer(3)]],
        constant uint& N         [[buffer(4)]],
        constant uint& K         [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    uint total = M * N;
    if (gid >= total) return;
    uint m = gid / N;
    uint n = gid % N;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(A[m * K + k]) * float(B[n * K + k]);
    }
    C[gid] = bfloat(acc);
}

kernel void k_fab_scale_mask_causal_softmax_rows_bf16(
        device bfloat*       S    [[buffer(0)]],
        device const float* mask [[buffer(1)]],
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
    device bfloat* row = S + (ulong)q * (ulong)Lk;

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
        row[k] = bfloat(e);
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
        row[k] = bfloat(e * inv);
    }
}

// dP[q,k] = sum_d dOh[q,d] * Vh[k,d]
kernel void k_fab_dP_bf16(
        device const bfloat* dOh [[buffer(0)]],
        device const bfloat* Vh  [[buffer(1)]],
        device bfloat*       dP  [[buffer(2)]],
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
    dP[q * Lk + k] = bfloat(acc);
}

// dS = P * (dP - D_q) * scale, in-place over P. One threadgroup per query row.
kernel void k_fab_dS_from_P_dP_bf16(
        device bfloat*       P_dS [[buffer(0)]],
        device const bfloat* dP   [[buffer(1)]],
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
    device bfloat* prow = P_dS + (ulong)q * (ulong)Lk;
    device const bfloat* dprow = dP + (ulong)q * (ulong)Lk;

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
        prow[k] = bfloat(p * (dp - Dq) * scale);
    }
}

// dVh[k,d] = sum_q P[q,k] * dOh[q,d]
kernel void k_fab_dVh_bf16(
        device const bfloat* P   [[buffer(0)]],
        device const bfloat* dOh [[buffer(1)]],
        device bfloat*       dVh [[buffer(2)]],
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
    dVh[k * hd + d] = bfloat(acc);
}

// dQh[q,d] = sum_k dS[q,k] * Kh[k,d]
kernel void k_fab_dQh_bf16(
        device const bfloat* dS  [[buffer(0)]],
        device const bfloat* Kh  [[buffer(1)]],
        device bfloat*       dQh [[buffer(2)]],
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
    dQh[q * hd + d] = bfloat(acc);
}

// dKh[k,d] = sum_q dS[q,k] * Qh[q,d]
kernel void k_fab_dKh_bf16(
        device const bfloat* dS  [[buffer(0)]],
        device const bfloat* Qh  [[buffer(1)]],
        device bfloat*       dKh [[buffer(2)]],
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
    dKh[k * hd + d] = bfloat(acc);
}
)msl";

id<MTLComputePipelineState> pso_extract() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_extract_head_LD"); });
    return pso;
}
id<MTLComputePipelineState> pso_pack() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_pack_head_LD"); });
    return pso;
}
id<MTLComputePipelineState> pso_softmax() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_scale_mask_causal_softmax_rows"); });
    return pso;
}
id<MTLComputePipelineState> pso_dP() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dP"); });
    return pso;
}
id<MTLComputePipelineState> pso_dS() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dS_from_P_dP"); });
    return pso;
}
id<MTLComputePipelineState> pso_dVh() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dVh"); });
    return pso;
}
id<MTLComputePipelineState> pso_dQh() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dQh"); });
    return pso;
}
id<MTLComputePipelineState> pso_dKh() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dKh"); });
    return pso;
}

// ── BF16 PSO accessors ───────────────────────────────────────────────────────
id<MTLComputePipelineState> pso_extract_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_extract_head_LD_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_pack_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_pack_head_LD_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_matmul_abt_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_matmul_ABT_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_softmax_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_scale_mask_causal_softmax_rows_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_dP_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dP_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_dS_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dS_from_P_dP_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_dVh_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dVh_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_dQh_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dQh_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_dKh_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kBwdSrc, @"k_fab_dKh_bf16"); });
    return pso;
}

// Self-contained naive BF16 ABT matmul launch helper (mirrors the one in
// flash_attention.mm; this TU can't reach that file-local symbol).
void launch_matmul_abt_bf16_bwd(id<MTLBuffer> bA, NSUInteger oA,
                                 id<MTLBuffer> bB, NSUInteger oB,
                                 id<MTLBuffer> bC, NSUInteger oC,
                                 int M, int N, int K) {
    if (M == 0 || N == 0) return;
    id<MTLComputePipelineState> pso = pso_matmul_abt_bf16();
    const uint32_t Mu = M, Nu = N, Ku = K;
    const NSUInteger total = (NSUInteger)M * (NSUInteger)N;
    const NSUInteger tg = 128;
    const NSUInteger grid = ((total + tg - 1) / tg) * tg;
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
        [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// Dispatch a per-element extract/pack kernel.
void run_pack_or_extract(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> bIn,  NSUInteger oIn,
                         id<MTLBuffer> bOut, NSUInteger oOut,
                         uint32_t L, uint32_t D, uint32_t head_off, uint32_t hd) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bIn  offset:oIn  atIndex:0];
        [enc setBuffer:bOut offset:oOut atIndex:1];
        [enc setBytes:&L        length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&D        length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&head_off length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&hd       length:sizeof(uint32_t) atIndex:5];
        NSUInteger total = (NSUInteger)L * (NSUInteger)hd;
        NSUInteger tg = 256;
        if (total < tg) tg = total;
        if (tg == 0) tg = 1;
        NSUInteger grid = ((total + tg - 1) / tg) * tg;
        [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

NSUInteger next_pow2_for_rows(int Lk) {
    NSUInteger tg = 32;
    while ((int)tg < Lk && tg < 1024) tg *= 2;
    if (tg > 1024) tg = 1024;
    return tg;
}

void run_softmax_rows(id<MTLComputePipelineState> pso,
                      id<MTLBuffer> bS, NSUInteger oS,
                      id<MTLBuffer> bMask, NSUInteger oMask, bool has_mask,
                      int Lq, int Lk, float scale, bool causal) {
    const uint32_t Lqu = Lq, Lku = Lk;
    const uint32_t hm = has_mask ? 1u : 0u;
    const uint32_t cu = causal ? 1u : 0u;
    NSUInteger tg = next_pow2_for_rows(Lk);
    NSUInteger shmem = tg * sizeof(float);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bS    offset:oS    atIndex:0];
        [enc setBuffer:bMask offset:oMask atIndex:1];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&scale length:sizeof(float) atIndex:4];
        [enc setBytes:&hm length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&cu length:sizeof(uint32_t) atIndex:6];
        [enc setThreadgroupMemoryLength:shmem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(Lq, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void run_2d_kernel(id<MTLComputePipelineState> pso,
                   id<MTLBuffer> bA, NSUInteger oA,
                   id<MTLBuffer> bB, NSUInteger oB,
                   id<MTLBuffer> bC, NSUInteger oC,
                   int Lq, int Lk, int hd,
                   int grid_x, int grid_y) {
    const uint32_t Lqu = Lq, Lku = Lk, hdu = hd;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bA offset:oA atIndex:0];
        [enc setBuffer:bB offset:oB atIndex:1];
        [enc setBuffer:bC offset:oC atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&hdu length:sizeof(uint32_t) atIndex:5];
        NSUInteger tgx = 16, tgy = 16;
        NSUInteger gx = ((grid_x + tgx - 1) / tgx) * tgx;
        NSUInteger gy = ((grid_y + tgy - 1) / tgy) * tgy;
        [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void run_dS(id<MTLComputePipelineState> pso,
            id<MTLBuffer> bP, NSUInteger oP,
            id<MTLBuffer> bdP, NSUInteger odP,
            int Lq, int Lk, float scale) {
    const uint32_t Lqu = Lq, Lku = Lk;
    NSUInteger tg = next_pow2_for_rows(Lk);
    NSUInteger shmem = tg * sizeof(float);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bP  offset:oP  atIndex:0];
        [enc setBuffer:bdP offset:odP atIndex:1];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&scale length:sizeof(float) atIndex:4];
        [enc setThreadgroupMemoryLength:shmem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(Lq, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void flash_attention_backward(const Tensor& Q,
                              const Tensor& K,
                              const Tensor& V,
                              const Tensor& O,
                              const Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              Tensor& dQ,
                              Tensor& dK,
                              Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry with CUDA.

    const Dtype dt = Q.dtype;
    if ((dt != Dtype::FP16 && dt != Dtype::BF16) ||
        K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must be FP16 or BF16");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dQ.rows != Lq || dQ.cols != D || dQ.dtype != dt) {
        dQ.resize(Lq, D, dt);
    }
    if (dK.rows != Lk || dK.cols != D || dK.dtype != dt) {
        dK.resize(Lk, D, dt);
    }
    if (dV.rows != Lk || dV.cols != D || dV.dtype != dt) {
        dV.resize(Lk, D, dt);
    }
    dQ.zero();
    dK.zero();
    dV.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    Tensor Qh  = Tensor::empty_on(Device::Metal, Lq, hd, dt);
    Tensor Kh  = Tensor::empty_on(Device::Metal, Lk, hd, dt);
    Tensor Vh  = Tensor::empty_on(Device::Metal, Lk, hd, dt);
    Tensor dOh = Tensor::empty_on(Device::Metal, Lq, hd, dt);
    Tensor P   = Tensor::empty_on(Device::Metal, Lq, Lk, dt);
    Tensor dP  = Tensor::empty_on(Device::Metal, Lq, Lk, dt);
    Tensor dQh = Tensor::empty_on(Device::Metal, Lq, hd, dt);
    Tensor dKh = Tensor::empty_on(Device::Metal, Lk, hd, dt);
    Tensor dVh = Tensor::empty_on(Device::Metal, Lk, hd, dt);

    id<MTLBuffer> bQ  = buffer_for(Q);   NSUInteger oQ_  = buffer_offset_for(Q);
    id<MTLBuffer> bK  = buffer_for(K);   NSUInteger oK_  = buffer_offset_for(K);
    id<MTLBuffer> bV  = buffer_for(V);   NSUInteger oV_  = buffer_offset_for(V);
    id<MTLBuffer> bdO = buffer_for(dO);  NSUInteger odO_ = buffer_offset_for(dO);
    id<MTLBuffer> bdQ = buffer_for(dQ);  NSUInteger odQ_ = buffer_offset_for(dQ);
    id<MTLBuffer> bdK = buffer_for(dK);  NSUInteger odK_ = buffer_offset_for(dK);
    id<MTLBuffer> bdV = buffer_for(dV);  NSUInteger odV_ = buffer_offset_for(dV);

    id<MTLBuffer> bQh  = buffer_for(Qh);   NSUInteger oQh_  = buffer_offset_for(Qh);
    id<MTLBuffer> bKh  = buffer_for(Kh);   NSUInteger oKh_  = buffer_offset_for(Kh);
    id<MTLBuffer> bVh  = buffer_for(Vh);   NSUInteger oVh_  = buffer_offset_for(Vh);
    id<MTLBuffer> bdOh = buffer_for(dOh);  NSUInteger odOh_ = buffer_offset_for(dOh);
    id<MTLBuffer> bP   = buffer_for(P);    NSUInteger oP_   = buffer_offset_for(P);
    id<MTLBuffer> bdP  = buffer_for(dP);   NSUInteger odP_  = buffer_offset_for(dP);
    id<MTLBuffer> bdQh = buffer_for(dQh);  NSUInteger odQh_ = buffer_offset_for(dQh);
    id<MTLBuffer> bdKh = buffer_for(dKh);  NSUInteger odKh_ = buffer_offset_for(dKh);
    id<MTLBuffer> bdVh = buffer_for(dVh);  NSUInteger odVh_ = buffer_offset_for(dVh);

    id<MTLBuffer> bMask = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger    oMask = d_mask ? pool_lookup_offset(d_mask) : 0;
    bool has_mask = (d_mask != nullptr);
    if (!bMask) { bMask = bQ; oMask = oQ_; }  // dummy bind

    // Select PSOs once — BF16 paths use the BF16 twin kernels.
    id<MTLComputePipelineState> p_ext    = bf16 ? pso_extract_bf16() : pso_extract();
    id<MTLComputePipelineState> p_pack   = bf16 ? pso_pack_bf16()    : pso_pack();
    id<MTLComputePipelineState> p_sm     = bf16 ? pso_softmax_bf16() : pso_softmax();
    id<MTLComputePipelineState> p_dVh    = bf16 ? pso_dVh_bf16()     : pso_dVh();
    id<MTLComputePipelineState> p_dP     = bf16 ? pso_dP_bf16()      : pso_dP();
    id<MTLComputePipelineState> p_dS_sel = bf16 ? pso_dS_bf16()      : pso_dS();
    id<MTLComputePipelineState> p_dQh    = bf16 ? pso_dQh_bf16()     : pso_dQh();
    id<MTLComputePipelineState> p_dKh    = bf16 ? pso_dKh_bf16()     : pso_dKh();

    for (int h = 0; h < num_heads; ++h) {
        const uint32_t head_off = static_cast<uint32_t>(h * hd);

        // 1) Extract Qh, Kh, Vh, dOh.
        run_pack_or_extract(p_ext, bQ,  oQ_,  bQh,  oQh_,  Lq, D, head_off, hd);
        run_pack_or_extract(p_ext, bK,  oK_,  bKh,  oKh_,  Lk, D, head_off, hd);
        run_pack_or_extract(p_ext, bV,  oV_,  bVh,  oVh_,  Lk, D, head_off, hd);
        run_pack_or_extract(p_ext, bdO, odO_, bdOh, odOh_, Lq, D, head_off, hd);

        // 2) S = Qh · Kh^T  →  P (Lq, Lk). FP32 accumulation.
        if (bf16) {
            launch_matmul_abt_bf16_bwd(bQh, oQh_, bKh, oKh_, bP, oP_, Lq, Lk, hd);
        } else {
            launch_matmul_abt_fp16(bQh, oQh_, bKh, oKh_, bP, oP_, Lq, Lk, hd);
        }

        // 3) Row-softmax with scale + mask + causal, in-place over P.
        run_softmax_rows(p_sm, bP, oP_, bMask, oMask, has_mask, Lq, Lk, inv_sqrt, causal);

        // 4) dVh = P^T · dOh   (Lk, hd)
        run_2d_kernel(p_dVh, bP, oP_, bdOh, odOh_, bdVh, odVh_,
                      Lq, Lk, hd, hd, Lk);

        // 5) dP = dOh · Vh^T   (Lq, Lk)
        run_2d_kernel(p_dP, bdOh, odOh_, bVh, oVh_, bdP, odP_,
                      Lq, Lk, hd, Lk, Lq);

        // 6) dS = P * (dP - D_q) * inv_sqrt   (in-place over P)
        run_dS(p_dS_sel, bP, oP_, bdP, odP_, Lq, Lk, inv_sqrt);

        // 7) dQh = dS · Kh   (Lq, hd)
        run_2d_kernel(p_dQh, bP, oP_, bKh, oKh_, bdQh, odQh_,
                      Lq, Lk, hd, hd, Lq);

        // 8) dKh = dS^T · Qh   (Lk, hd)
        run_2d_kernel(p_dKh, bP, oP_, bQh, oQh_, bdKh, odKh_,
                      Lq, Lk, hd, hd, Lk);

        // 9) Pack per-head grads into the (L, D) slot.
        run_pack_or_extract(p_pack, bdQh, odQh_, bdQ, odQ_, Lq, D, head_off, hd);
        run_pack_or_extract(p_pack, bdKh, odKh_, bdK, odK_, Lk, D, head_off, hd);
        run_pack_or_extract(p_pack, bdVh, odVh_, bdV, odV_, Lk, D, head_off, hd);
    }
}

} // namespace brotensor::detail::metal
