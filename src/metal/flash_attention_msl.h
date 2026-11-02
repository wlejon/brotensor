#pragma once

#import <Foundation/Foundation.h>

namespace brotensor::detail::metal {
namespace {

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

// ---- Per-head V transpose for the fallback's P @ V GEMM ----
//
// X is (L, D) with D = num_heads * head_dim. The GEMMs read Q / K / O's head
// columns in place; only V is copied, transposed, so P @ V is an A @ B^T.

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

kernel void k_extract_head_DL_bf16(
        device const bfloat* X   [[buffer(0)]],   // (L, D)
        device bfloat*       Y   [[buffer(1)]],   // (head_dim, L)
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

// BF16 online-softmax flash-attention kernel. Verbatim copy of
// k_flash_attention with half -> bfloat.
kernel void k_flash_attention_bf16(
        device const bfloat* Q    [[buffer(0)]],   // (Lq, D)
        device const bfloat* Kk   [[buffer(1)]],   // (Lk, D)
        device const bfloat* V    [[buffer(2)]],   // (Lk, D)
        device const float* mask [[buffer(3)]],   // (Lk,) may be dummy
        device bfloat*       Out  [[buffer(4)]],   // (Lq, D)
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
        Out[q * D + head_off + d] = bfloat(partial[slot] * inv);
        ++slot;
    }
}

// Packed variable-length flash attention (Qwen3-VL window attention). One
// threadgroup per (q_global, head). Each block scans cu_q[] to find its
// sequence, then runs the same online-softmax loop as k_flash_attention with
// K bounds [cu_k[b], cu_k[b+1]). Causal diagonal is per-sequence (q_local vs
// k_local).
kernel void k_flash_attention_varlen(
        device const half*  Q    [[buffer(0)]],   // (total_q, D)
        device const half*  Kk   [[buffer(1)]],   // (total_k, D)
        device const half*  V    [[buffer(2)]],   // (total_k, D)
        device const int*   cu_q [[buffer(3)]],   // (B+1)
        device const int*   cu_k [[buffer(4)]],   // (B+1)
        device half*        Out  [[buffer(5)]],   // (total_q, D)
        constant uint& B         [[buffer(6)]],
        constant uint& D         [[buffer(7)]],
        constant uint& head_dim  [[buffer(8)]],
        constant uint& causal    [[buffer(9)]],
        threadgroup float* scratch [[threadgroup(0)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float* scores = scratch;
    threadgroup float* red    = scratch + FA_KTILE;

    uint q_global = gid.x;
    uint h        = gid.y;
    uint head_off = h * head_dim;

    // Locate sequence b: cu_q[b] <= q_global < cu_q[b+1].
    uint b = 0;
    while (b < B && uint(cu_q[b + 1]) <= q_global) ++b;
    if (b >= B) return;
    uint q_beg = uint(cu_q[b]);
    uint k_beg = uint(cu_k[b]);
    uint k_end = uint(cu_k[b + 1]);
    if (k_end <= k_beg) {
        for (uint d = tid; d < head_dim; d += tg_size) {
            Out[q_global * D + head_off + d] = half(0.0f);
        }
        return;
    }
    uint Lk = k_end - k_beg;
    uint q_local = q_global - q_beg;
    float inv_sqrt = rsqrt(float(head_dim));

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (uint k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        if (causal != 0u && k0 > q_local) break;
        uint klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal != 0u && k0 + klen - 1u > q_local) klen = q_local - k0 + 1u;

        for (uint t = tid; t < klen; t += tg_size) {
            uint kg = k_beg + k0 + t;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q[q_global * D + head_off + d]) *
                       float(Kk[kg * D + head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = exp(run_max - m_new);
        }

        uint slot = 0;
        for (uint d = tid; d < head_dim; d += tg_size) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (uint t = 0; t < klen; ++t) {
                acc += scores[t] * float(V[(k_beg + k0 + t) * D + head_off + d]);
            }
            partial[slot] = acc;
            ++slot;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < head_dim; d += tg_size) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q_global * D + head_off + d] = half(partial[slot] * inv);
        ++slot;
    }
}

// BF16 twin of k_flash_attention_varlen.
kernel void k_flash_attention_varlen_bf16(
        device const bfloat* Q    [[buffer(0)]],
        device const bfloat* Kk   [[buffer(1)]],
        device const bfloat* V    [[buffer(2)]],
        device const int*    cu_q [[buffer(3)]],
        device const int*    cu_k [[buffer(4)]],
        device bfloat*       Out  [[buffer(5)]],
        constant uint& B          [[buffer(6)]],
        constant uint& D          [[buffer(7)]],
        constant uint& head_dim   [[buffer(8)]],
        constant uint& causal     [[buffer(9)]],
        threadgroup float* scratch [[threadgroup(0)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float* scores = scratch;
    threadgroup float* red    = scratch + FA_KTILE;

    uint q_global = gid.x;
    uint h        = gid.y;
    uint head_off = h * head_dim;

    uint b = 0;
    while (b < B && uint(cu_q[b + 1]) <= q_global) ++b;
    if (b >= B) return;
    uint q_beg = uint(cu_q[b]);
    uint k_beg = uint(cu_k[b]);
    uint k_end = uint(cu_k[b + 1]);
    if (k_end <= k_beg) {
        for (uint d = tid; d < head_dim; d += tg_size) {
            Out[q_global * D + head_off + d] = bfloat(0.0f);
        }
        return;
    }
    uint Lk = k_end - k_beg;
    uint q_local = q_global - q_beg;
    float inv_sqrt = rsqrt(float(head_dim));

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (uint k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        if (causal != 0u && k0 > q_local) break;
        uint klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal != 0u && k0 + klen - 1u > q_local) klen = q_local - k0 + 1u;

        for (uint t = tid; t < klen; t += tg_size) {
            uint kg = k_beg + k0 + t;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += float(Q[q_global * D + head_off + d]) *
                       float(Kk[kg * D + head_off + d]);
            }
            scores[t] = dot * inv_sqrt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = exp(run_max - m_new);
        }

        uint slot = 0;
        for (uint d = tid; d < head_dim; d += tg_size) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (uint t = 0; t < klen; ++t) {
                acc += scores[t] * float(V[(k_beg + k0 + t) * D + head_off + d]);
            }
            partial[slot] = acc;
            ++slot;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < head_dim; d += tg_size) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q_global * D + head_off + d] = bfloat(partial[slot] * inv);
        ++slot;
    }
}

// FP32 twin of k_flash_attention_varlen.
kernel void k_flash_attention_varlen_fp32(
        device const float* Q    [[buffer(0)]],
        device const float* Kk   [[buffer(1)]],
        device const float* V    [[buffer(2)]],
        device const int*   cu_q [[buffer(3)]],
        device const int*   cu_k [[buffer(4)]],
        device float*       Out  [[buffer(5)]],
        constant uint& B         [[buffer(6)]],
        constant uint& D         [[buffer(7)]],
        constant uint& head_dim  [[buffer(8)]],
        constant uint& causal    [[buffer(9)]],
        threadgroup float* scratch [[threadgroup(0)]],
        uint3 gid    [[threadgroup_position_in_grid]],
        uint3 tid3   [[thread_position_in_threadgroup]],
        uint3 tgs3   [[threads_per_threadgroup]]) {
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    threadgroup float* scores = scratch;
    threadgroup float* red    = scratch + FA_KTILE;

    uint q_global = gid.x;
    uint h        = gid.y;
    uint head_off = h * head_dim;

    uint b = 0;
    while (b < B && uint(cu_q[b + 1]) <= q_global) ++b;
    if (b >= B) return;
    uint q_beg = uint(cu_q[b]);
    uint k_beg = uint(cu_k[b]);
    uint k_end = uint(cu_k[b + 1]);
    if (k_end <= k_beg) {
        for (uint d = tid; d < head_dim; d += tg_size) {
            Out[q_global * D + head_off + d] = 0.0f;
        }
        return;
    }
    uint Lk = k_end - k_beg;
    uint q_local = q_global - q_beg;
    float inv_sqrt = rsqrt(float(head_dim));

    float run_max = -1e30f;
    float run_sum = 0.0f;
    float partial[MAX_HD_PER_THREAD];
    for (uint i = 0; i < MAX_HD_PER_THREAD; ++i) partial[i] = 0.0f;

    for (uint k0 = 0; k0 < Lk; k0 += FA_KTILE) {
        if (causal != 0u && k0 > q_local) break;
        uint klen = (Lk - k0) < FA_KTILE ? (Lk - k0) : FA_KTILE;
        if (causal != 0u && k0 + klen - 1u > q_local) klen = q_local - k0 + 1u;

        for (uint t = tid; t < klen; t += tg_size) {
            uint kg = k_beg + k0 + t;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += Q[q_global * D + head_off + d] *
                       Kk[kg * D + head_off + d];
            }
            scores[t] = dot * inv_sqrt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        float alpha;
        if (run_max <= -1e29f) {
            alpha = 0.0f;
        } else {
            alpha = exp(run_max - m_new);
        }

        uint slot = 0;
        for (uint d = tid; d < head_dim; d += tg_size) {
            if (slot >= MAX_HD_PER_THREAD) break;
            float acc = alpha * partial[slot];
            for (uint t = 0; t < klen; ++t) {
                acc += scores[t] * V[(k_beg + k0 + t) * D + head_off + d];
            }
            partial[slot] = acc;
            ++slot;
        }

        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    uint slot = 0;
    for (uint d = tid; d < head_dim; d += tg_size) {
        if (slot >= MAX_HD_PER_THREAD) break;
        Out[q_global * D + head_off + d] = partial[slot] * inv;
        ++slot;
    }
}

// Row softmax over FP32 scores S (rows, Lk) — the per-head fallback's scores
// come out of the GEMM in FP32, so every exp() sees an exact score and only
// the normalised probability is narrowed, once, for the P @ V GEMM. Scores
// are scaled by `scale`; with a mask, key k is dropped where mask[k] <= 0.5.
// One threadgroup per row.
template <typename TO>
kernel void k_fa_softmax_f32(device const float* S    [[buffer(0)]],
                             device TO*          P    [[buffer(1)]],
                             device const float* mask [[buffer(2)]],
                             constant uint& Lk        [[buffer(3)]],
                             constant float& scale    [[buffer(4)]],
                             constant uint& has_mask  [[buffer(5)]],
                             threadgroup float* red   [[threadgroup(0)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint tg   [[threads_per_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    device const float* s = S + (ulong)row * Lk;
    device TO* pr = P + (ulong)row * Lk;
    const uint nsg = (tg + 31) / 32;
    float mx = -1e30f;
    for (uint k = tid; k < Lk; k += tg) {
        const float v = (has_mask != 0u && mask[k] <= 0.5f) ? -1e30f : s[k] * scale;
        mx = max(mx, v);
    }
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = lane < nsg ? red[lane] : -1e30f;
    mx = simd_max(mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool empty = mx <= -1e29f;
    float sum = 0.0f;
    for (uint k = tid; k < Lk; k += tg) {
        const bool drop = empty || (has_mask != 0u && mask[k] <= 0.5f);
        sum += drop ? 0.0f : exp(s[k] * scale - mx);
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = lane < nsg ? red[lane] : 0.0f;
    sum = simd_sum(sum);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint k = tid; k < Lk; k += tg) {
        const bool drop = empty || (has_mask != 0u && mask[k] <= 0.5f);
        pr[k] = TO(drop ? 0.0f : exp(s[k] * scale - mx) * inv);
    }
}
template [[host_name("k_fa_softmax_f32_half")]] kernel void k_fa_softmax_f32<half>(
    device const float*, device half*, device const float*, constant uint&, constant float&,
    constant uint&, threadgroup float*, uint, uint, uint, uint, uint);
template [[host_name("k_fa_softmax_f32_bf16")]] kernel void k_fa_softmax_f32<bfloat>(
    device const float*, device bfloat*, device const float*, constant uint&, constant float&,
    constant uint&, threadgroup float*, uint, uint, uint, uint, uint);
)msl";

} // namespace
} // namespace brotensor::detail::metal
