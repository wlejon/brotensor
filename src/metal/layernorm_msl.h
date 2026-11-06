#pragma once

#import <Foundation/Foundation.h>

namespace brotensor::detail::metal {
namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

kernel void k_ln_forward(device const float* x       [[buffer(0)]],
                         device const float* gamma   [[buffer(1)]],
                         device const float* beta    [[buffer(2)]],
                         device float*       y       [[buffer(3)]],
                         device float*       xhat    [[buffer(4)]],
                         device float*       scratch [[buffer(5)]],
                         constant uint& n            [[buffer(6)]],
                         constant float& eps         [[buffer(7)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) local += x[i];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_v = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = x[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(n);
    float rstd = rsqrt(var + eps);
    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }
    for (uint i = tid; i < n; i += tg_size) {
        float xh = (x[i] - mean) * rstd;
        xhat[i] = xh;
        y[i] = gamma[i] * xh + beta[i];
    }
}

kernel void k_ln_backward(device const float* dY     [[buffer(0)]],
                          device const float* xhat   [[buffer(1)]],
                          device const float* gamma  [[buffer(2)]],
                          constant float& rstd       [[buffer(3)]],
                          device float*       dX     [[buffer(4)]],
                          device float*       dGamma [[buffer(5)]],
                          device float*       dBeta  [[buffer(6)]],
                          constant uint& n           [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g = dY[i];
        dGamma[i] += g * xhat[i];
        dBeta[i]  += g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += dY[i] * gamma[i];
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += dY[i] * gamma[i] * xhat[i];
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = dY[i] * gamma[i];
        dX[i] = scale * (nf * dxh - sum_dxh - xhat[i] * sum_dxh_xhat);
    }
}

// FP16 backward. dGamma/dBeta written to FP32 scratch; host folds into
// caller-owned FP16 accumulators.
kernel void k_ln_backward_fp16(device const half*  dY     [[buffer(0)]],
                               device const half*  xhat   [[buffer(1)]],
                               device const half*  gamma  [[buffer(2)]],
                               constant float& rstd       [[buffer(3)]],
                               device half*        dX     [[buffer(4)]],
                               device float*       dGamma_scratch [[buffer(5)]],
                               device float*       dBeta_scratch  [[buffer(6)]],
                               constant uint& n           [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g  = float(dY[i]);
        float xh = float(xhat[i]);
        dGamma_scratch[i] = g * xh;
        dBeta_scratch[i]  = g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += float(dY[i]) * float(gamma[i]);
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += float(dY[i]) * float(gamma[i]) * float(xhat[i]);
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = float(dY[i]) * float(gamma[i]);
        float xh  = float(xhat[i]);
        dX[i] = half(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
    }
}

kernel void k_ln_add_fp32_into_fp16(device const float* src [[buffer(0)]],
                                    device half*        dst [[buffer(1)]],
                                    constant uint& n        [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}

// BF16 backward — verbatim copy of k_ln_backward_fp16 with half→bfloat.
// FP32 scratch/fold pattern mirrored exactly.
kernel void k_ln_backward_bf16(device const bfloat* dY     [[buffer(0)]],
                               device const bfloat* xhat   [[buffer(1)]],
                               device const bfloat* gamma  [[buffer(2)]],
                               constant float& rstd         [[buffer(3)]],
                               device bfloat*       dX     [[buffer(4)]],
                               device float*       dGamma_scratch [[buffer(5)]],
                               device float*       dBeta_scratch  [[buffer(6)]],
                               constant uint& n             [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    for (uint i = tid; i < n; i += tg_size) {
        float g  = float(dY[i]);
        float xh = float(xhat[i]);
        dGamma_scratch[i] = g * xh;
        dBeta_scratch[i]  = g;
    }
    float local = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local += float(dY[i]) * float(gamma[i]);
    }
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local2 = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local2 += float(dY[i]) * float(gamma[i]) * float(xhat[i]);
    }
    sdata[tid] = local2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dxh_xhat = sdata[0];

    float nf = float(n);
    float scale = rstd / nf;
    for (uint i = tid; i < n; i += tg_size) {
        float dxh = float(dY[i]) * float(gamma[i]);
        float xh  = float(xhat[i]);
        dX[i] = bfloat(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
    }
}

kernel void k_ln_add_fp32_into_bf16(device const float* src [[buffer(0)]],
                                    device bfloat*      dst [[buffer(1)]],
                                    constant uint& n        [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}

// FP32 X; Y FP32, or FP16 (a mixed-precision model feeding half GEMMs
// straight from an FP32 residual stream: one rounding, at the store).
template <typename OutT>
void ln_forward_inference_row(device const float* x, device const float* gamma, device const float* beta,
                              device OutT* y, uint R, uint D, float eps, uint has_beta, uint row, uint tid,
                              uint tg_size, threadgroup float* sdata) {
    if (row >= R) return;
    device const float* xrow = x + row * D;
    device OutT*        yrow = y + row * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tg_size) local += xrow[i];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(D);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_v = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float d = xrow[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(D);
    float rstd = rsqrt(var + eps);
    for (uint i = tid; i < D; i += tg_size) {
        float xh = (xrow[i] - mean) * rstd;
        float b = (has_beta != 0u) ? beta[i] : 0.0f;
        yrow[i] = OutT(xh * gamma[i] + b);
    }
}

kernel void k_ln_forward_inference_batched(device const float* x      [[buffer(0)]],
                                           device const float* gamma  [[buffer(1)]],
                                           device const float* beta   [[buffer(2)]],
                                           device float*       y      [[buffer(3)]],
                                           constant uint& R           [[buffer(4)]],
                                           constant uint& D           [[buffer(5)]],
                                           constant float& eps        [[buffer(6)]],
                                           constant uint& has_beta    [[buffer(7)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    ln_forward_inference_row<float>(x, gamma, beta, y, R, D, eps, has_beta, row, tid, tg_size, sdata);
}

kernel void k_ln_forward_inference_batched_to_fp16(device const float* x      [[buffer(0)]],
                                                   device const float* gamma  [[buffer(1)]],
                                                   device const float* beta   [[buffer(2)]],
                                                   device half*        y      [[buffer(3)]],
                                                   constant uint& R           [[buffer(4)]],
                                                   constant uint& D           [[buffer(5)]],
                                                   constant float& eps        [[buffer(6)]],
                                                   constant uint& has_beta    [[buffer(7)]],
                                                   uint row [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_position_in_threadgroup]],
                                                   uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    ln_forward_inference_row<half>(x, gamma, beta, y, R, D, eps, has_beta, row, tid, tg_size, sdata);
}
)msl";

NSString* const kFp16Src = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

kernel void k_ln_forward_inference_batched_fp16(
        device const half* x      [[buffer(0)]],
        device const half* gamma  [[buffer(1)]],
        device const half* beta   [[buffer(2)]],
        device half*       y      [[buffer(3)]],
        constant uint& R          [[buffer(4)]],
        constant uint& D          [[buffer(5)]],
        constant float& eps       [[buffer(6)]],
        constant uint& has_beta   [[buffer(7)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[LN_BLOCK];
    if (row >= R) return;
    device const half* xrow = x + (ulong)row * D;
    device       half* yrow = y + (ulong)row * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tg_size) local += float(xrow[i]);
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(D);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_v = 0.0f;
    for (uint i = tid; i < D; i += tg_size) {
        float d = float(xrow[i]) - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = sdata[0] / float(D);
    float rstd = rsqrt(var + eps);
    for (uint i = tid; i < D; i += tg_size) {
        float xh = (float(xrow[i]) - mean) * rstd;
        float g  = float(gamma[i]);
        float b  = (has_beta != 0u) ? float(beta[i]) : 0.0f;
        yrow[i] = half(xh * g + b);
    }
}
)msl";

NSString* const kBatchedCachesSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint LN_BLOCK = 256;

// ─── Forward (one threadgroup per row, FP32/FP16/BF16) ───────────────────
#define LN_FWD_BATCHED(SUFFIX, T)                                                  \
kernel void k_ln_fwd_batched_caches_##SUFFIX(                                       \
        device const T*     x     [[buffer(0)]],                                    \
        device const T*     gamma [[buffer(1)]],                                    \
        device const T*     beta  [[buffer(2)]],                                    \
        device T*           y     [[buffer(3)]],                                    \
        device T*           xhat  [[buffer(4)]],                                    \
        device float*       mean_out [[buffer(5)]],                                 \
        device float*       rstd_out [[buffer(6)]],                                 \
        constant uint& R          [[buffer(7)]],                                    \
        constant uint& D          [[buffer(8)]],                                    \
        constant float& eps       [[buffer(9)]],                                    \
        uint row [[threadgroup_position_in_grid]],                                  \
        uint tid [[thread_position_in_threadgroup]],                                \
        uint tg_size [[threads_per_threadgroup]]) {                                 \
    threadgroup float sdata[LN_BLOCK];                                              \
    if (row >= R) return;                                                           \
    device const T* xrow = x    + (ulong)row * D;                                   \
    device       T* yrow = y    + (ulong)row * D;                                   \
    device       T* hrow = xhat + (ulong)row * D;                                   \
    float local = 0.0f;                                                             \
    for (uint i = tid; i < D; i += tg_size) local += float(xrow[i]);                \
    sdata[tid] = local;                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                    \
        if (tid < s) sdata[tid] += sdata[tid + s];                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float mean = sdata[0] / float(D);                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    float local_v = 0.0f;                                                           \
    for (uint i = tid; i < D; i += tg_size) {                                       \
        float d = float(xrow[i]) - mean;                                            \
        local_v += d * d;                                                           \
    }                                                                               \
    sdata[tid] = local_v;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                    \
        if (tid < s) sdata[tid] += sdata[tid + s];                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float var  = sdata[0] / float(D);                                               \
    float rstd = rsqrt(var + eps);                                                  \
    if (tid == 0) {                                                                 \
        mean_out[row] = mean;                                                       \
        rstd_out[row] = rstd;                                                       \
    }                                                                               \
    for (uint i = tid; i < D; i += tg_size) {                                       \
        float xh = (float(xrow[i]) - mean) * rstd;                                  \
        hrow[i]  = T(xh);                                                           \
        float g  = float(gamma[i]);                                                 \
        float b  = float(beta[i]);                                                  \
        yrow[i]  = T(xh * g + b);                                                   \
    }                                                                               \
}
LN_FWD_BATCHED(f32,  float)
LN_FWD_BATCHED(fp16, half)
LN_FWD_BATCHED(bf16, bfloat)
#undef LN_FWD_BATCHED

// ─── Backward pass 1 — per-row dX (R threadgroups) ──────────────────────
// Computes sum_dxh, sum_dxh_xhat per row in shmem, then writes dX.
#define LN_BWD_DX_BATCHED(SUFFIX, T)                                                \
kernel void k_ln_bwd_dx_batched_##SUFFIX(                                          \
        device const T*     dY    [[buffer(0)]],                                   \
        device const T*     xhat  [[buffer(1)]],                                   \
        device const T*     gamma [[buffer(2)]],                                   \
        device const float* rstd_R[[buffer(3)]],                                   \
        device T*           dX    [[buffer(4)]],                                   \
        constant uint& R          [[buffer(5)]],                                   \
        constant uint& D          [[buffer(6)]],                                   \
        uint row [[threadgroup_position_in_grid]],                                 \
        uint tid [[thread_position_in_threadgroup]],                               \
        uint tg_size [[threads_per_threadgroup]]) {                                \
    threadgroup float sdata[LN_BLOCK];                                             \
    if (row >= R) return;                                                          \
    device const T* dyr = dY   + (ulong)row * D;                                   \
    device const T* hr  = xhat + (ulong)row * D;                                   \
    device       T* dxr = dX   + (ulong)row * D;                                   \
    float rstd = rstd_R[row];                                                      \
    float local = 0.0f;                                                            \
    for (uint i = tid; i < D; i += tg_size) {                                      \
        local += float(dyr[i]) * float(gamma[i]);                                  \
    }                                                                              \
    sdata[tid] = local;                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) sdata[tid] += sdata[tid + s];                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    }                                                                              \
    float sum_dxh = sdata[0];                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    float local2 = 0.0f;                                                           \
    for (uint i = tid; i < D; i += tg_size) {                                      \
        local2 += float(dyr[i]) * float(gamma[i]) * float(hr[i]);                  \
    }                                                                              \
    sdata[tid] = local2;                                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) sdata[tid] += sdata[tid + s];                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    }                                                                              \
    float sum_dxh_xhat = sdata[0];                                                 \
    float nf = float(D);                                                           \
    float scale = rstd / nf;                                                       \
    for (uint i = tid; i < D; i += tg_size) {                                      \
        float dxh = float(dyr[i]) * float(gamma[i]);                               \
        float xh  = float(hr[i]);                                                  \
        dxr[i] = T(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));              \
    }                                                                              \
}
LN_BWD_DX_BATCHED(f32,  float)
LN_BWD_DX_BATCHED(fp16, half)
LN_BWD_DX_BATCHED(bf16, bfloat)
#undef LN_BWD_DX_BATCHED

// ─── Backward pass 2 — per-feature dGamma/dBeta (D threadgroups) ────────
// Each threadgroup d sums over all R rows: dGamma[d] = sum_r dY[r,d]*xhat[r,d];
// dBeta[d] = sum_r dY[r,d]. Writes FP32 scratch; host fold-adds into the
// caller's accumulator. Single threadgroup per column avoids races.
#define LN_BWD_DGB_BATCHED(SUFFIX, T)                                              \
kernel void k_ln_bwd_dgb_batched_##SUFFIX(                                         \
        device const T*     dY    [[buffer(0)]],                                   \
        device const T*     xhat  [[buffer(1)]],                                   \
        device float*       dGamma_f32 [[buffer(2)]],                              \
        device float*       dBeta_f32  [[buffer(3)]],                              \
        constant uint& R          [[buffer(4)]],                                   \
        constant uint& D          [[buffer(5)]],                                   \
        uint d   [[threadgroup_position_in_grid]],                                 \
        uint tid [[thread_position_in_threadgroup]],                               \
        uint tg_size [[threads_per_threadgroup]]) {                                \
    threadgroup float sg[LN_BLOCK];                                                \
    threadgroup float sb[LN_BLOCK];                                                \
    if (d >= D) return;                                                            \
    float lg = 0.0f, lb = 0.0f;                                                    \
    for (uint r = tid; r < R; r += tg_size) {                                      \
        float g  = float(dY[(ulong)r * D + d]);                                    \
        float xh = float(xhat[(ulong)r * D + d]);                                  \
        lg += g * xh;                                                              \
        lb += g;                                                                   \
    }                                                                              \
    sg[tid] = lg;                                                                  \
    sb[tid] = lb;                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                               \
    for (uint s = tg_size / 2; s > 0; s >>= 1) {                                   \
        if (tid < s) { sg[tid] += sg[tid + s]; sb[tid] += sb[tid + s]; }           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    }                                                                              \
    if (tid == 0) {                                                                \
        dGamma_f32[d] = sg[0];                                                     \
        dBeta_f32[d]  = sb[0];                                                     \
    }                                                                              \
}
LN_BWD_DGB_BATCHED(f32,  float)
LN_BWD_DGB_BATCHED(fp16, half)
LN_BWD_DGB_BATCHED(bf16, bfloat)
#undef LN_BWD_DGB_BATCHED

// ─── Folds: dGamma += scratch (typed) ──────────────────────────────────
kernel void k_ln_fold_add_f32(device const float* src [[buffer(0)]],
                              device float*       dst [[buffer(1)]],
                              constant uint& n        [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = dst[i] + src[i];
}
kernel void k_ln_fold_add_fp16(device const float* src [[buffer(0)]],
                               device half*        dst [[buffer(1)]],
                               constant uint& n        [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}
kernel void k_ln_fold_add_bf16(device const float* src [[buffer(0)]],
                               device bfloat*      dst [[buffer(1)]],
                               constant uint& n        [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}
)msl";

} // namespace
} // namespace brotensor::detail::metal
