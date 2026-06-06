// ─── Metal self-attention with additive pre-softmax bias ───────────────────
//
// Metal counterpart of src/cuda/self_attention_bias.cu — multi-head
// self-attention that adds an optional per-head (L, L) bias to the attention
// logits before softmax (the primitive behind T5 relative-position bias and
// ALiBi-style biases).
//
//   S[h,q,k] = scale * (Q_h[q] . K_h[k]) + attn_bias[h*L+q, k]
//   O        = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// Scores are materialised (L, L) per head — intended for encoder-length
// sequences, not long-context decoding. Dispatched on X.dtype (FP32 / FP16 /
// BF16): the projection inputs/outputs are typed, every intermediate
// (Q/K/V/scores/softmax) is FP32 scratch, math is FP32. attn_bias is FP32.

#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

constexpr NSUInteger kSabSoftmaxBlock = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

#define SAB_SM_BLOCK 256u

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
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_proj_fp32,   @"k_sab_proj_fp32")
DEF_PSO(pso_proj_fp16,   @"k_sab_proj_fp16")
DEF_PSO(pso_proj_bf16,   @"k_sab_proj_bf16")
DEF_PSO(pso_scores,      @"k_sab_scores")
DEF_PSO(pso_softmax,     @"k_sab_softmax")
DEF_PSO(pso_apply_v,     @"k_sab_apply_v")
DEF_PSO(pso_output_fp32, @"k_sab_output_fp32")
DEF_PSO(pso_output_fp16, @"k_sab_output_fp16")
DEF_PSO(pso_output_bf16, @"k_sab_output_bf16")
DEF_PSO(pso_proj_int8,   @"k_sab_proj_int8")
DEF_PSO(pso_output_int8, @"k_sab_output_int8")
DEF_PSO(pso_sardp_proj_fp32,    @"k_sardp_proj_fp32")
DEF_PSO(pso_sardp_proj_fp16,    @"k_sardp_proj_fp16")
DEF_PSO(pso_sardp_proj_bf16,    @"k_sardp_proj_bf16")
DEF_PSO(pso_sardp_scores_fp32,  @"k_sardp_scores_fp32")
DEF_PSO(pso_sardp_scores_fp16,  @"k_sardp_scores_fp16")
DEF_PSO(pso_sardp_scores_bf16,  @"k_sardp_scores_bf16")
DEF_PSO(pso_sardp_output_fp32,  @"k_sardp_output_fp32")
DEF_PSO(pso_sardp_output_fp16,  @"k_sardp_output_fp16")
DEF_PSO(pso_sardp_output_bf16,  @"k_sardp_output_bf16")
DEF_PSO(pso_win_gather_fp32,    @"k_sardp_win_gather_fp32")
DEF_PSO(pso_win_gather_fp16,    @"k_sardp_win_gather_fp16")
DEF_PSO(pso_win_gather_bf16,    @"k_sardp_win_gather_bf16")
DEF_PSO(pso_win_scatter_fp32,   @"k_sardp_win_scatter_fp32")
DEF_PSO(pso_win_scatter_fp16,   @"k_sardp_win_scatter_fp16")
DEF_PSO(pso_win_scatter_bf16,   @"k_sardp_win_scatter_bf16")
#undef DEF_PSO

// Forward declarations — run3d / run_rows are defined just below.
void run3d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           NSUInteger nz, void (^bind)(id<MTLComputeCommandEncoder>));
void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>));

// 2-D thread grid (nx, ny), threadgroup capped at 16x16. For win gather/scatter.
void run2d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tx = nx < 16 ? nx : 16;
        NSUInteger ty = ny < 16 ? ny : 16;
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// Core decomposed-rel-pos attention over one token grid (gh x gw = L rows).
// X / O are passed as buffer + byte offset so the windowed path can drive
// per-window slices of a partition buffer. Wq/Wk/Wv/Wo and rel_h/rel_w are
// whole tensors; biases are optional. dt selects the typed proj/scores/output
// kernels; every intermediate is FP32 scratch.
void run_sardp(Dtype dt,
               id<MTLBuffer> bX, NSUInteger oX,
               const Tensor& Wq, const Tensor* bq,
               const Tensor& Wk, const Tensor* bk,
               const Tensor& Wv, const Tensor* bv,
               const Tensor& Wo, const Tensor* bo,
               const Tensor& rel_h, const Tensor& rel_w,
               int H, int L, int D, int dh, int gh, int gw, float scale,
               id<MTLBuffer> bO, NSUInteger oO) {
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    id<MTLComputePipelineState> proj_pso =
        (dt == Dtype::FP16) ? pso_sardp_proj_fp16()
      : (dt == Dtype::BF16) ? pso_sardp_proj_bf16()
      : pso_sardp_proj_fp32();
    id<MTLComputePipelineState> scores_pso =
        (dt == Dtype::FP16) ? pso_sardp_scores_fp16()
      : (dt == Dtype::BF16) ? pso_sardp_scores_bf16()
      : pso_sardp_scores_fp32();
    id<MTLComputePipelineState> out_pso =
        (dt == Dtype::FP16) ? pso_sardp_output_fp16()
      : (dt == Dtype::BF16) ? pso_sardp_output_bf16()
      : pso_sardp_output_fp32();

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);
    const uint32_t ghu = static_cast<uint32_t>(gh);
    const uint32_t gwu = static_cast<uint32_t>(gw);

    // Per-head projection with optional bias.
    auto proj = [&](const Tensor& W, const Tensor* b,
                    id<MTLBuffer> bOut, NSUInteger oOut) {
        id<MTLBuffer> bW = buffer_for(W); NSUInteger oW = buffer_offset_for(W);
        const bool hb = (b && b->data);
        id<MTLBuffer> bB = hb ? buffer_for(*b) : bW;
        NSUInteger oB = hb ? buffer_offset_for(*b) : oW;
        const uint32_t has_bias = hb ? 1u : 0u;
        run3d(proj_pso, dhu, Lu, static_cast<NSUInteger>(H),
              ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bB   offset:oB   atIndex:2];
            [enc setBytes:&has_bias length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:bOut offset:oOut atIndex:4];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:7];
        });
    };
    proj(Wq, bq, bQh, oQh);
    proj(Wk, bk, bKh, oKh);
    proj(Wv, bv, bVh, oVh);

    // Scores with inline decomposed rel-pos bias.
    id<MTLBuffer> bRh = buffer_for(rel_h); NSUInteger oRh = buffer_offset_for(rel_h);
    id<MTLBuffer> bRw = buffer_for(rel_w); NSUInteger oRw = buffer_offset_for(rel_w);
    run3d(scores_pso, Lu, Lu, static_cast<NSUInteger>(H),
          ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh offset:oQh atIndex:0];
        [enc setBuffer:bKh offset:oKh atIndex:1];
        [enc setBuffer:bRh offset:oRh atIndex:2];
        [enc setBuffer:bRw offset:oRw atIndex:3];
        [enc setBuffer:bS  offset:oS  atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
        [enc setBytes:&ghu   length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&gwu   length:sizeof(uint32_t) atIndex:9];
    });

    // Softmax (no mask) — reuse the bias-path row softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS offset:oS atIndex:0];
        [enc setBuffer:bA offset:oA atIndex:1];
        [enc setBuffer:bS offset:oS atIndex:2];  // dummy mask
        const uint32_t has_mask = 0u;
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat — reuse the bias-path apply-V kernel.
    run3d(pso_apply_v(), dhu, Lu, static_cast<NSUInteger>(H),
          ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // Output projection with optional bias.
    id<MTLBuffer> bWo = buffer_for(Wo); NSUInteger oWo = buffer_offset_for(Wo);
    const bool hbo = (bo && bo->data);
    id<MTLBuffer> bBo = hbo ? buffer_for(*bo) : bWo;
    NSUInteger oBo = hbo ? buffer_offset_for(*bo) : oWo;
    const uint32_t has_bo = hbo ? 1u : 0u;
    run3d(out_pso, Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc offset:oYc atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bBo offset:oBo atIndex:2];
        [enc setBytes:&has_bo length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bO  offset:oO  atIndex:4];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:6];
    });
}

// Dispatch a 3-D thread grid (nx, ny, nz), threadgroup capped at 256 threads.
void run3d(id<MTLComputePipelineState> pso,
           NSUInteger nx, NSUInteger ny, NSUInteger nz,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tx = nx < 16 ? nx : 16;
        NSUInteger ty = ny < 16 ? ny : 16;
        [enc dispatchThreads:MTLSizeMake(nx, ny, nz)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// One threadgroup per row; `kSabSoftmaxBlock` threads each.
void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (rows == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kSabSoftmaxBlock, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void self_attention_bias_forward(const Tensor& X, const Tensor& Wq,
                                 const Tensor& Wk, const Tensor& Wv,
                                 const Tensor& Wo,
                                 const Tensor* bq, const Tensor* bk,
                                 const Tensor* bv, const Tensor* bo,
                                 const float* d_mask,
                                 const Tensor* attn_bias, int num_heads,
                                 float scale, Tensor& O) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("self_attention_bias_forward: X must be FP32, FP16, or BF16");
    }
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo dtype must match X");
    }
    const int L = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    if (H <= 0 || D % H != 0) {
        throw std::runtime_error("self_attention_bias_forward: num_heads must divide D");
    }
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo must be (D, D)");
    }
    const int dh = D / H;

    bool has_bias = false;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be FP32");
        }
        if (attn_bias->size() != H * L * L) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be (num_heads*L, L)");
        }
        has_bias = true;
    }
    if (O.rows != L || O.cols != D || O.dtype != X.dtype) {
        O.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    // FP32 scratch for every intermediate.
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bO = buffer_for(O);   NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    const uint32_t has_mask = (bM != nil) ? 1u : 0u;

    id<MTLBuffer> bB = has_bias ? buffer_for(*attn_bias) : nil;
    NSUInteger oB = has_bias ? buffer_offset_for(*attn_bias) : 0;
    id<MTLBuffer> bB_arg = bB ? bB : bX;
    NSUInteger oB_arg = bB ? oB : oX;
    const uint32_t has_bias_u = has_bias ? 1u : 0u;

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);

    id<MTLComputePipelineState> proj_pso =
        (X.dtype == Dtype::FP16) ? pso_proj_fp16()
      : (X.dtype == Dtype::BF16) ? pso_proj_bf16()
      : pso_proj_fp32();

    // Q / K / V per-head projections, each with an optional bias.
    auto proj = [&](const Tensor& W, const Tensor* b,
                    id<MTLBuffer> bOut, NSUInteger oOut) {
        id<MTLBuffer> bW = buffer_for(W); NSUInteger oW = buffer_offset_for(W);
        const bool hb = (b && b->data);
        id<MTLBuffer> bB = hb ? buffer_for(*b) : bW;
        NSUInteger oB = hb ? buffer_offset_for(*b) : oW;
        const uint32_t hb_u = hb ? 1u : 0u;
        run3d(proj_pso, dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bB   offset:oB   atIndex:2];
            [enc setBytes:&hb_u length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:bOut offset:oOut atIndex:4];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:7];
        });
    };
    proj(Wq, bq, bQh, oQh);
    proj(Wk, bk, bKh, oKh);
    proj(Wv, bv, bVh, oVh);

    // Scores: scale * (Q.K) + bias.
    run3d(pso_scores(), Lu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh   offset:oQh   atIndex:0];
        [enc setBuffer:bKh   offset:oKh   atIndex:1];
        [enc setBuffer:bB_arg offset:oB_arg atIndex:2];
        [enc setBytes:&has_bias_u length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bS    offset:oS    atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
    });

    // Row-wise masked softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS     offset:oS     atIndex:0];
        [enc setBuffer:bA     offset:oA     atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat.
    run3d(pso_apply_v(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // Output projection by Wo.
    id<MTLComputePipelineState> out_pso =
        (X.dtype == Dtype::FP16) ? pso_output_fp16()
      : (X.dtype == Dtype::BF16) ? pso_output_bf16()
      : pso_output_fp32();
    id<MTLBuffer> bWo = buffer_for(Wo);
    NSUInteger oWo = buffer_offset_for(Wo);
    const bool hbo = (bo && bo->data);
    id<MTLBuffer> bBo = hbo ? buffer_for(*bo) : bWo;
    NSUInteger oBo = hbo ? buffer_offset_for(*bo) : oWo;
    const uint32_t has_bo = hbo ? 1u : 0u;
    run3d(out_pso, Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc    offset:oYc    atIndex:0];
        [enc setBuffer:bWo    offset:oWo    atIndex:1];
        [enc setBuffer:bBo    offset:oBo    atIndex:2];
        [enc setBytes:&has_bo length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:4];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:5];
        [enc setBuffer:bO     offset:oO     atIndex:6];
        [enc setBytes:&Lu     length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:8];
    });
}

// W8A16 INT8 weight-only variant. The four projection weights are INT8 (D, D)
// matrices paired with FP32 (D, 1) per-output-row dequant scales; activations
// stay FP16. The attention core (scores / softmax / Attn@V) is byte-identical
// to the FP16 path above — only the projection and output matmuls differ.
void self_attention_bias_int8w_fp16(const Tensor& X,
                                    const Tensor& Wq_int8, const Tensor& sq,
                                    const Tensor& Wk_int8, const Tensor& sk,
                                    const Tensor& Wv_int8, const Tensor& sv,
                                    const Tensor& Wo_int8, const Tensor& so,
                                    const float* d_mask,
                                    const Tensor* attn_bias, int num_heads,
                                    float scale, Tensor& O) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("self_attention_bias_int8w_fp16: X must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be INT8");
    }
    if (sq.dtype != Dtype::FP32 || sk.dtype != Dtype::FP32 ||
        sv.dtype != Dtype::FP32 || so.dtype != Dtype::FP32) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: scales must be FP32");
    }
    const int L = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    if (H <= 0 || D % H != 0) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: num_heads must divide D");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D ||
        Wv_int8.rows != D || Wv_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be (D, D)");
    }
    if (sq.size() != D || sk.size() != D || sv.size() != D || so.size() != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: each scale tensor must have D entries");
    }
    const int dh = D / H;

    bool has_bias = false;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be FP32");
        }
        if (attn_bias->size() != H * L * L) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be (num_heads*L, L)");
        }
        has_bias = true;
    }
    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(L, D, Dtype::FP16);
    }
    if (L == 0 || D == 0) return;

    // FP32 scratch for every intermediate.
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bO = buffer_for(O);   NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    const uint32_t has_mask = (bM != nil) ? 1u : 0u;

    id<MTLBuffer> bB = has_bias ? buffer_for(*attn_bias) : nil;
    NSUInteger oB = has_bias ? buffer_offset_for(*attn_bias) : 0;
    id<MTLBuffer> bB_arg = bB ? bB : bX;
    NSUInteger oB_arg = bB ? oB : oX;
    const uint32_t has_bias_u = has_bias ? 1u : 0u;

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);

    // Q / K / V per-head INT8 projections.
    auto proj = ^(id<MTLBuffer> bW, NSUInteger oW,
                  id<MTLBuffer> bSc, NSUInteger oSc,
                  id<MTLBuffer> bOut, NSUInteger oOut) {
        run3d(pso_proj_int8(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bSc  offset:oSc  atIndex:2];
            [enc setBuffer:bOut offset:oOut atIndex:3];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:6];
        });
    };
    proj(buffer_for(Wq_int8), buffer_offset_for(Wq_int8),
         buffer_for(sq), buffer_offset_for(sq), bQh, oQh);
    proj(buffer_for(Wk_int8), buffer_offset_for(Wk_int8),
         buffer_for(sk), buffer_offset_for(sk), bKh, oKh);
    proj(buffer_for(Wv_int8), buffer_offset_for(Wv_int8),
         buffer_for(sv), buffer_offset_for(sv), bVh, oVh);

    // Scores: scale * (Q.K) + bias.
    run3d(pso_scores(), Lu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh   offset:oQh   atIndex:0];
        [enc setBuffer:bKh   offset:oKh   atIndex:1];
        [enc setBuffer:bB_arg offset:oB_arg atIndex:2];
        [enc setBytes:&has_bias_u length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bS    offset:oS    atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
    });

    // Row-wise masked softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS     offset:oS     atIndex:0];
        [enc setBuffer:bA     offset:oA     atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat.
    run3d(pso_apply_v(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // INT8 output projection by Wo.
    id<MTLBuffer> bWo = buffer_for(Wo_int8);
    NSUInteger oWo = buffer_offset_for(Wo_int8);
    id<MTLBuffer> bSo = buffer_for(so);
    NSUInteger oSo = buffer_offset_for(so);
    run3d(pso_output_int8(), Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc    offset:oYc    atIndex:0];
        [enc setBuffer:bWo    offset:oWo    atIndex:1];
        [enc setBuffer:bSo    offset:oSo    atIndex:2];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:3];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:4];
        [enc setBuffer:bO     offset:oO     atIndex:5];
        [enc setBytes:&Lu     length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:7];
    });
}

// ── Decomposed 2D relative-position attention (public entry points) ─────────

namespace {

void check_rdp_common(const char* fn, const Tensor& X,
                      const Tensor& Wq, const Tensor& Wk,
                      const Tensor& Wv, const Tensor& Wo,
                      const Tensor& rel_h, const Tensor& rel_w,
                      const Tensor* bq, const Tensor* bk,
                      const Tensor* bv, const Tensor* bo,
                      int num_heads, int D) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error(std::string(fn) + ": X must be FP32, FP16, or BF16");
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype ||
        rel_h.dtype != X.dtype || rel_w.dtype != X.dtype)
        throw std::runtime_error(std::string(fn) +
            ": Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(std::string(fn) + ": num_heads must divide D");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(std::string(fn) + ": Wq/Wk/Wv/Wo must be (D, D)");
    auto check_bias = [&](const Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string(fn) + ": " + name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string(fn) + ": " + name + " must have D entries");
        }
    };
    check_bias(bq, "bq"); check_bias(bk, "bk");
    check_bias(bv, "bv"); check_bias(bo, "bo");
}

} // namespace

void self_attention_decomposed_rel_pos_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale, Tensor& O) {
    const char* fn = "self_attention_decomposed_rel_pos_forward";
    const int L = X.rows;
    const int D = X.cols;
    check_rdp_common(fn, X, Wq, Wk, Wv, Wo, rel_pos_h, rel_pos_w,
                     bq, bk, bv, bo, num_heads, D);
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * grid_h - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*grid_h-1, head_dim)");
    if (rel_pos_w.rows != 2 * grid_w - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*grid_w-1, head_dim)");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    run_sardp(X.dtype, buffer_for(X), buffer_offset_for(X),
              Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
              num_heads, L, D, dh, grid_h, grid_w, scale,
              buffer_for(O), buffer_offset_for(O));
}

void self_attention_decomposed_rel_pos_windowed_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        Tensor& O) {
    const char* fn = "self_attention_decomposed_rel_pos_windowed_forward";
    const int L = X.rows;
    const int D = X.cols;
    check_rdp_common(fn, X, Wq, Wk, Wv, Wo, rel_pos_h, rel_pos_w,
                     bq, bk, bv, bo, num_heads, D);
    if (window <= 0)
        throw std::runtime_error(std::string(fn) + ": window must be >= 1");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * window - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*window-1, head_dim)");
    if (rel_pos_w.rows != 2 * window - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*window-1, head_dim)");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    const Dtype dt = X.dtype;
    const int pad_h = (window - grid_h % window) % window;
    const int pad_w = (window - grid_w % window) % window;
    const int nw_h  = (grid_h + pad_h) / window;
    const int nw_w  = (grid_w + pad_w) / window;
    const int nW    = nw_h * nw_w;
    const int ww    = window * window;
    const int nrows = nW * ww;

    Tensor Pin  = Tensor::empty_on(Device::Metal, nrows, D, dt);
    Tensor Pout = Tensor::empty_on(Device::Metal, nrows, D, dt);
    id<MTLBuffer> bPin  = buffer_for(Pin);  NSUInteger oPin  = buffer_offset_for(Pin);
    id<MTLBuffer> bPout = buffer_for(Pout); NSUInteger oPout = buffer_offset_for(Pout);

    id<MTLComputePipelineState> gather_pso =
        (dt == Dtype::FP16) ? pso_win_gather_fp16()
      : (dt == Dtype::BF16) ? pso_win_gather_bf16()
      : pso_win_gather_fp32();
    id<MTLComputePipelineState> scatter_pso =
        (dt == Dtype::FP16) ? pso_win_scatter_fp16()
      : (dt == Dtype::BF16) ? pso_win_scatter_bf16()
      : pso_win_scatter_fp32();

    const uint32_t ghu = static_cast<uint32_t>(grid_h);
    const uint32_t gwu = static_cast<uint32_t>(grid_w);
    const uint32_t winu = static_cast<uint32_t>(window);
    const uint32_t nwwu = static_cast<uint32_t>(nw_w);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t nrowsu = static_cast<uint32_t>(nrows);

    auto win_args = ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&ghu    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&gwu    length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&winu   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&nwwu   length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&nrowsu length:sizeof(uint32_t) atIndex:7];
    };

    // Gather grid → padded per-window partition.
    run2d(gather_pso, Du, nrowsu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:bPin offset:oPin atIndex:1];
        win_args(enc);
    });

    // Per-window attention over contiguous row-slices of the partition buffers.
    const NSUInteger row_stride =
        static_cast<NSUInteger>(ww) * D * ::brotensor::dtype_size_bytes(dt);
    for (int w = 0; w < nW; ++w) {
        const NSUInteger inOff  = oPin  + static_cast<NSUInteger>(w) * row_stride;
        const NSUInteger outOff = oPout + static_cast<NSUInteger>(w) * row_stride;
        run_sardp(dt, bPin, inOff, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                  rel_pos_h, rel_pos_w, num_heads, ww, D, dh, window, window,
                  scale, bPout, outOff);
    }

    // Scatter per-window results back to the grid (dropping padded tokens).
    run2d(scatter_pso, Du, nrowsu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bPout offset:oPout atIndex:0];
        [enc setBuffer:buffer_for(O) offset:buffer_offset_for(O) atIndex:1];
        win_args(enc);
    });
}

} // namespace brotensor::detail::metal
