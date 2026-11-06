#pragma once

#import <Foundation/Foundation.h>

namespace brotensor::detail::metal {
namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

#define SAB_SM_BLOCK 256u
#define RPB_XL_MAX_HEAD_DIM 512u

// Per-head projection: Out[(hh*L+i), j] = sum_k In[i,k] * W[hh*dh+j, k].
// In: (L, Din) typed, W: (D, Din) typed, Out: (H*L, dh) FP32.
#define SAB_PROJ_KERNEL(NAME, T)                                              \
kernel void NAME(device const T*     In       [[buffer(0)]],                  \
                 device const T*     W        [[buffer(1)]],                  \
                 device const T*     bias     [[buffer(2)]],                  \
                 constant uint&      has_bias [[buffer(3)]],                  \
                 device float*       Out      [[buffer(4)]],                  \
                 constant uint& L   [[buffer(5)]],                            \
                 constant uint& Din [[buffer(6)]],                            \
                 constant uint& dh  [[buffer(7)]],                            \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint j = gid.x, i = gid.y, hh = gid.z;                                    \
    if (i >= L || j >= dh) return;                                            \
    uint o = hh * dh + j;                                                     \
    device const T* xr = In + (ulong)i * Din;                                 \
    device const T* wr = W  + (ulong)o * Din;                                 \
    float acc = has_bias != 0u ? float(bias[o]) : 0.0f;                       \
    for (uint k = 0; k < Din; ++k) acc += float(xr[k]) * float(wr[k]);        \
    Out[((ulong)hh * L + i) * dh + j] = acc;                                  \
}

SAB_PROJ_KERNEL(k_sab_proj_fp32, float)
SAB_PROJ_KERNEL(k_sab_proj_fp16, half)
SAB_PROJ_KERNEL(k_sab_proj_bf16, bfloat)

// S[(hh*L+i), j] = scale * (Q_h[i] . K_h[j]) + bias[(hh*L+i), j].
kernel void k_sab_scores(device const float* Qh   [[buffer(0)]],
                         device const float* Kh   [[buffer(1)]],
                         device const float* bias [[buffer(2)]],
                         constant uint& has_bias  [[buffer(3)]],
                         device float*       S    [[buffer(4)]],
                         constant uint& L     [[buffer(5)]],
                         constant uint& dh    [[buffer(6)]],
                         constant float& scale [[buffer(7)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x, i = gid.y, hh = gid.z;
    if (i >= L || j >= L) return;
    ulong qrow = ((ulong)hh * L + i) * dh;
    ulong krow = ((ulong)hh * L + j) * dh;
    float s = 0.0f;
    for (uint k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    ulong srow = ((ulong)hh * L + i) * L;
    s *= scale;
    if (has_bias) s += bias[srow + j];
    S[srow + j] = s;
}

// Per-row masked softmax over (H*L, L). One threadgroup per (head, query row).
kernel void k_sab_softmax(device const float* scores [[buffer(0)]],
                          device float*       Attn   [[buffer(1)]],
                          device const float* mask   [[buffer(2)]],
                          constant uint& has_mask    [[buffer(3)]],
                          constant uint& L           [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[SAB_SM_BLOCK];
    uint i_within = row % L;
    device const float* srow = scores + (ulong)row * L;
    device float*       arow = Attn   + (ulong)row * L;

    if (has_mask && mask[i_within] < 0.5f) {
        for (uint j = tid; j < L; j += tg_size) arow[j] = 0.0f;
        return;
    }
    float local_max = -1e30f;
    for (uint j = tid; j < L; j += tg_size) {
        if (has_mask && mask[j] < 0.5f) continue;
        float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint j = tid; j < L; j += tg_size) {
        if (has_mask && mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
        float e = exp(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = sdata[0];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint j = tid; j < L; j += tg_size) arow[j] *= inv;
}

// Yconcat[i, hh*dh+k] = sum_j Attn[(hh*L+i), j] * Vh[(hh*L+j), k].
kernel void k_sab_apply_v(device const float* Attn    [[buffer(0)]],
                          device const float* Vh      [[buffer(1)]],
                          device float*       Yconcat [[buffer(2)]],
                          constant uint& L  [[buffer(3)]],
                          constant uint& dh [[buffer(4)]],
                          constant uint& D  [[buffer(5)]],
                          uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x, i = gid.y, hh = gid.z;
    if (i >= L || k >= dh) return;
    ulong arow = ((ulong)hh * L + i) * L;
    float acc = 0.0f;
    for (uint j = 0; j < L; ++j) {
        ulong vrow = ((ulong)hh * L + j) * dh;
        acc += Attn[arow + j] * Vh[vrow + k];
    }
    Yconcat[(ulong)i * D + (hh * dh + k)] = acc;
}

// O[i, c] = mask[i] ? sum_k Yconcat[i,k] * Wo[c,k] : 0.
#define SAB_OUTPUT_KERNEL(NAME, T)                                            \
kernel void NAME(device const float* Y        [[buffer(0)]],                  \
                 device const T*     Wo       [[buffer(1)]],                  \
                 device const T*     bias     [[buffer(2)]],                  \
                 constant uint&      has_bias [[buffer(3)]],                  \
                 device const float* mask     [[buffer(4)]],                  \
                 constant uint& has_mask      [[buffer(5)]],                  \
                 device T*           O        [[buffer(6)]],                  \
                 constant uint& L [[buffer(7)]],                              \
                 constant uint& D [[buffer(8)]],                              \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint c = gid.x, i = gid.y;                                                \
    if (i >= L || c >= D) return;                                             \
    if (has_mask && mask[i] < 0.5f) { O[(ulong)i * D + c] = T(0.0f); return; }\
    device const float* yr = Y  + (ulong)i * D;                               \
    device const T*     wr = Wo + (ulong)c * D;                               \
    float acc = has_bias != 0u ? float(bias[c]) : 0.0f;                       \
    for (uint k = 0; k < D; ++k) acc += yr[k] * float(wr[k]);                 \
    O[(ulong)i * D + c] = T(acc);                                             \
}

SAB_OUTPUT_KERNEL(k_sab_output_fp32, float)
SAB_OUTPUT_KERNEL(k_sab_output_fp16, half)
SAB_OUTPUT_KERNEL(k_sab_output_bf16, bfloat)

// ── W8A16 (INT8 weight, FP16 activation) projection / output ───────────────
//
// Same dot products as k_sab_proj_fp16 / k_sab_output_fp16, but the weight is
// an INT8 (D, Din) matrix paired with an FP32 per-output-row dequant scale.
// The row accumulates against int8 weights, then the whole sum is multiplied
// by scales[wrow] — one scale covers the entire row, so this equals
// dequantising the row first.

// Out[(hh*L+i), j] = scales[wrow] * sum_k In[i,k] * W[wrow,k];  wrow = hh*dh+j.
kernel void k_sab_proj_int8(device const half*  In     [[buffer(0)]],
                            device const char*  W      [[buffer(1)]],
                            device const float* scales [[buffer(2)]],
                            device float*       Out    [[buffer(3)]],
                            constant uint& L   [[buffer(4)]],
                            constant uint& Din [[buffer(5)]],
                            constant uint& dh  [[buffer(6)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x, i = gid.y, hh = gid.z;
    if (i >= L || j >= dh) return;
    uint wrow = hh * dh + j;
    device const half*   xr = In + (ulong)i * Din;
    device const int8_t* wr = (device const int8_t*)W + (ulong)wrow * Din;
    float acc = 0.0f;
    for (uint k = 0; k < Din; ++k) acc += float(xr[k]) * float(wr[k]);
    Out[((ulong)hh * L + i) * dh + j] = acc * scales[wrow];
}

// O[i, c] = mask[i] ? scales[c] * sum_k Yconcat[i,k] * Wo[c,k] : 0.
kernel void k_sab_output_int8(device const float* Y      [[buffer(0)]],
                              device const char*  Wo     [[buffer(1)]],
                              device const float* scales [[buffer(2)]],
                              device const float* mask   [[buffer(3)]],
                              constant uint& has_mask    [[buffer(4)]],
                              device half*        O      [[buffer(5)]],
                              constant uint& L [[buffer(6)]],
                              constant uint& D [[buffer(7)]],
                              uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x, i = gid.y;
    if (i >= L || c >= D) return;
    if (has_mask && mask[i] < 0.5f) { O[(ulong)i * D + c] = half(0.0f); return; }
    device const float*  yr = Y + (ulong)i * D;
    device const int8_t* wr = (device const int8_t*)Wo + (ulong)c * D;
    float acc = 0.0f;
    for (uint k = 0; k < D; ++k) acc += yr[k] * float(wr[k]);
    O[(ulong)i * D + c] = half(acc * scales[c]);
}

// ── Decomposed 2D relative-position attention (SAM / ViTDet) ────────────────
// Same scratch/softmax/apply_v pipeline as the bias path, but the qkv/output
// projections carry optional biases and the pre-softmax bias is the decomposed
// rel-pos term computed from Q inline in the scores kernel.

// Out[(hh*L+i), j] = (has_bias ? bias[hh*dh+j] : 0) + sum_k In[i,k]*W[hh*dh+j,k].
#define SARDP_PROJ_KERNEL(NAME, T)                                            \
kernel void NAME(device const T*     In       [[buffer(0)]],                  \
                 device const T*     W        [[buffer(1)]],                  \
                 device const T*     bias     [[buffer(2)]],                  \
                 constant uint&      has_bias [[buffer(3)]],                  \
                 device float*       Out      [[buffer(4)]],                  \
                 constant uint& L   [[buffer(5)]],                            \
                 constant uint& Din [[buffer(6)]],                            \
                 constant uint& dh  [[buffer(7)]],                            \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint j = gid.x, i = gid.y, hh = gid.z;                                    \
    if (i >= L || j >= dh) return;                                            \
    uint o = hh * dh + j;                                                     \
    device const T* xr = In + (ulong)i * Din;                                 \
    device const T* wr = W  + (ulong)o * Din;                                 \
    float acc = has_bias != 0u ? float(bias[o]) : 0.0f;                       \
    for (uint k = 0; k < Din; ++k) acc += float(xr[k]) * float(wr[k]);        \
    Out[((ulong)hh * L + i) * dh + j] = acc;                                  \
}

SARDP_PROJ_KERNEL(k_sardp_proj_fp32, float)
SARDP_PROJ_KERNEL(k_sardp_proj_fp16, half)
SARDP_PROJ_KERNEL(k_sardp_proj_bf16, bfloat)

// S[(hh*L+i), j] = scale*(Q_h[i].K_h[j]) + q.rel_h[(qh-kh)+gh-1] + q.rel_w[(qw-kw)+gw-1].
#define SARDP_SCORES_KERNEL(NAME, T)                                          \
kernel void NAME(device const float* Qh    [[buffer(0)]],                     \
                 device const float* Kh    [[buffer(1)]],                     \
                 device const T*     rel_h [[buffer(2)]],                     \
                 device const T*     rel_w [[buffer(3)]],                     \
                 device float*       S     [[buffer(4)]],                     \
                 constant uint& L     [[buffer(5)]],                          \
                 constant uint& dh    [[buffer(6)]],                          \
                 constant float& scale[[buffer(7)]],                          \
                 constant uint& gh    [[buffer(8)]],                          \
                 constant uint& gw    [[buffer(9)]],                          \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint j = gid.x, i = gid.y, hh = gid.z;                                    \
    if (i >= L || j >= L) return;                                             \
    ulong qrow = ((ulong)hh * L + i) * dh;                                    \
    ulong krow = ((ulong)hh * L + j) * dh;                                    \
    int qh = int(i) / int(gw), qw = int(i) % int(gw);                         \
    int kh = int(j) / int(gw), kw = int(j) % int(gw);                         \
    device const T* rhr = rel_h + (ulong)(qh - kh + int(gh) - 1) * dh;        \
    device const T* rwr = rel_w + (ulong)(qw - kw + int(gw) - 1) * dh;        \
    float s = 0.0f, bh = 0.0f, bw = 0.0f;                                     \
    for (uint k = 0; k < dh; ++k) {                                           \
        float q = Qh[qrow + k];                                               \
        s  += q * Kh[krow + k];                                               \
        bh += q * float(rhr[k]);                                              \
        bw += q * float(rwr[k]);                                              \
    }                                                                         \
    S[((ulong)hh * L + i) * L + j] = s * scale + bh + bw;                     \
}

SARDP_SCORES_KERNEL(k_sardp_scores_fp32, float)
SARDP_SCORES_KERNEL(k_sardp_scores_fp16, half)
SARDP_SCORES_KERNEL(k_sardp_scores_bf16, bfloat)

// O[i, c] = (has_bias ? bias[c] : 0) + sum_k Yconcat[i,k] * Wo[c,k].
#define SARDP_OUTPUT_KERNEL(NAME, T)                                          \
kernel void NAME(device const float* Y        [[buffer(0)]],                  \
                 device const T*     Wo       [[buffer(1)]],                  \
                 device const T*     bias     [[buffer(2)]],                  \
                 constant uint&      has_bias [[buffer(3)]],                  \
                 device T*           O        [[buffer(4)]],                  \
                 constant uint& L [[buffer(5)]],                              \
                 constant uint& D [[buffer(6)]],                              \
                 uint3 gid [[thread_position_in_grid]]) {                     \
    uint c = gid.x, i = gid.y;                                                \
    if (i >= L || c >= D) return;                                             \
    device const float* yr = Y  + (ulong)i * D;                               \
    device const T*     wr = Wo + (ulong)c * D;                               \
    float acc = has_bias != 0u ? float(bias[c]) : 0.0f;                       \
    for (uint k = 0; k < D; ++k) acc += yr[k] * float(wr[k]);                 \
    O[(ulong)i * D + c] = T(acc);                                             \
}

SARDP_OUTPUT_KERNEL(k_sardp_output_fp32, float)
SARDP_OUTPUT_KERNEL(k_sardp_output_fp16, half)
SARDP_OUTPUT_KERNEL(k_sardp_output_bf16, bfloat)

// Windowed partition gather / scatter (zero-pad bottom/right to a window
// multiple). Partition row r = wi*window*window + lh*window + lw maps to grid
// token (h, w) = (nh*window+lh, nw*window+lw), (nh, nw) = (wi/nw_w, wi%nw_w).
#define WIN_GATHER_KERNEL(NAME, T)                                            \
kernel void NAME(device const T* X   [[buffer(0)]],                           \
                 device T*       P   [[buffer(1)]],                           \
                 constant uint& grid_h [[buffer(2)]],                         \
                 constant uint& grid_w [[buffer(3)]],                         \
                 constant uint& window [[buffer(4)]],                         \
                 constant uint& nw_w   [[buffer(5)]],                         \
                 constant uint& D      [[buffer(6)]],                         \
                 constant uint& nrows  [[buffer(7)]],                         \
                 uint2 gid [[thread_position_in_grid]]) {                     \
    uint col = gid.x, row = gid.y;                                            \
    if (col >= D || row >= nrows) return;                                     \
    uint ww  = window * window;                                               \
    uint loc = row % ww, wi = row / ww;                                       \
    uint h = (wi / nw_w) * window + loc / window;                             \
    uint w = (wi % nw_w) * window + loc % window;                             \
    device T* dst = P + (ulong)row * D + col;                                 \
    if (h < grid_h && w < grid_w) *dst = X[(ulong)(h * grid_w + w) * D + col];\
    else *dst = T(0.0f);                                                      \
}

WIN_GATHER_KERNEL(k_sardp_win_gather_fp32, float)
WIN_GATHER_KERNEL(k_sardp_win_gather_fp16, half)
WIN_GATHER_KERNEL(k_sardp_win_gather_bf16, bfloat)

#define WIN_SCATTER_KERNEL(NAME, T)                                           \
kernel void NAME(device const T* P   [[buffer(0)]],                           \
                 device T*       O   [[buffer(1)]],                           \
                 constant uint& grid_h [[buffer(2)]],                         \
                 constant uint& grid_w [[buffer(3)]],                         \
                 constant uint& window [[buffer(4)]],                         \
                 constant uint& nw_w   [[buffer(5)]],                         \
                 constant uint& D      [[buffer(6)]],                         \
                 constant uint& nrows  [[buffer(7)]],                         \
                 uint2 gid [[thread_position_in_grid]]) {                     \
    uint col = gid.x, row = gid.y;                                            \
    if (col >= D || row >= nrows) return;                                     \
    uint ww  = window * window;                                               \
    uint loc = row % ww, wi = row / ww;                                       \
    uint h = (wi / nw_w) * window + loc / window;                             \
    uint w = (wi % nw_w) * window + loc % window;                             \
    if (h < grid_h && w < grid_w)                                             \
        O[(ulong)(h * grid_w + w) * D + col] = P[(ulong)row * D + col];       \
}

WIN_SCATTER_KERNEL(k_sardp_win_scatter_fp32, float)
WIN_SCATTER_KERNEL(k_sardp_win_scatter_fp16, half)
WIN_SCATTER_KERNEL(k_sardp_win_scatter_bf16, bfloat)

// ── Transformer-XL relative-position bias ──────────────────────────────────
//
// out[(h*T + q), k] = sum_d qv[q, h*head_dim + d] * pk[(T-1-q) + k, h*head_dim + d]
//
// The rel_shift is the column offset (T-1-q) on the pk read, so no (T, 2T-1)
// matrix_bd intermediate is built. One threadgroup per (head, query) row: the
// row's query slice is staged in threadgroup memory once and reused across all
// T keys.
kernel void k_rel_pos_bias_xl(device const float* qv  [[buffer(0)]],
                              device const float* pk  [[buffer(1)]],
                              device float*       out [[buffer(2)]],
                              constant uint& T        [[buffer(3)]],
                              constant uint& D        [[buffer(4)]],
                              constant uint& head_dim [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float qs[RPB_XL_MAX_HEAD_DIM];
    uint h  = row / T;
    uint q  = row - h * T;
    uint co = h * head_dim;

    device const float* qrow = qv + (ulong)q * D + co;
    for (uint d = tid; d < head_dim; d += tg_size) qs[d] = qrow[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = T - 1u - q;
    for (uint k = tid; k < T; k += tg_size) {
        device const float* prow = pk + (ulong)(base + k) * D + co;
        float s = 0.0f;
        for (uint d = 0; d < head_dim; ++d) s += qs[d] * prow[d];
        out[(ulong)row * T + k] = s;
    }
}
)msl";

} // namespace
} // namespace brotensor::detail::metal
