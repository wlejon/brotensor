// FP16 cross-attention forward that exposes a head-averaged attention map
// and accepts an optional pre-softmax logit bias. Materialised per-head FP32
// score matrix; structurally mirrors src/cuda/cross_attention_with_attn.cu.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger ROW_SM_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint ROW_SM_BLOCK = 256;

struct ProjParams {
    uint L;
    uint Din;
    uint dh;
};

// Per-head projection FP16 -> FP32. In: (L, Din) FP16, W: (D, Din) FP16,
// Out: (h*L, dh) FP32. gid = (j in dh, i in L, hh in H).
kernel void k_cxa_proj(device const half*  In   [[buffer(0)]],
                       device const half*  W    [[buffer(1)]],
                       device float*       Out  [[buffer(2)]],
                       constant ProjParams& p   [[buffer(3)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint j  = gid.x;
    uint i  = gid.y;
    uint hh = gid.z;
    if (i >= p.L || j >= p.dh) return;
    uint row_off = hh * p.dh;
    device const half* xr = In + uint(i) * p.Din;
    device const half* wr = W  + uint(row_off + j) * p.Din;
    float acc = 0.0;
    for (uint k = 0; k < p.Din; ++k) {
        acc += float(xr[k]) * float(wr[k]);
    }
    uint out_row = hh * p.L + i;
    Out[out_row * p.dh + j] = acc;
}

struct ScoresParams {
    uint Lq;
    uint Lk;
    uint dh;
    uint has_bias;
    float inv_sqrtdh;
};

// S(hh, i, j) = (Qh(hh, i, :) . Kh(hh, j, :)) * inv_sqrtdh + bias(i, j).
kernel void k_cxa_scores(device const float* Qh    [[buffer(0)]],
                         device const float* Kh    [[buffer(1)]],
                         device const float* bias  [[buffer(2)]],
                         device float*       S     [[buffer(3)]],
                         constant ScoresParams& p  [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint j  = gid.x;
    uint i  = gid.y;
    uint hh = gid.z;
    if (i >= p.Lq || j >= p.Lk) return;
    uint qrow = (hh * p.Lq + i) * p.dh;
    uint krow = (hh * p.Lk + j) * p.dh;
    float s = 0.0;
    for (uint k = 0; k < p.dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    s *= p.inv_sqrtdh;
    if (p.has_bias != 0u) s += bias[uint(i) * p.Lk + j];
    uint srow = (hh * p.Lq + i) * p.Lk;
    S[srow + j] = s;
}

struct SoftmaxParams {
    uint Lk;
    uint has_mask;
};

// Per-row masked softmax: one threadgroup per (head, row).
kernel void k_cxa_row_softmax(device const float* scores [[buffer(0)]],
                              device float*       Attn   [[buffer(1)]],
                              device const float* mask   [[buffer(2)]],
                              constant SoftmaxParams& p  [[buffer(3)]],
                              uint3 gid  [[threadgroup_position_in_grid]],
                              uint3 tid3 [[thread_position_in_threadgroup]],
                              uint3 tgs3 [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    uint row = gid.x;
    uint tid = tid3.x;
    uint tg_size = tgs3.x;
    device const float* srow = scores + row * p.Lk;
    device float*       arow = Attn   + row * p.Lk;

    float local_max = -1e30;
    for (uint j = tid; j < p.Lk; j += tg_size) {
        if (p.has_mask != 0u && mask[j] < 0.5) continue;
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

    float local_sum = 0.0;
    for (uint j = tid; j < p.Lk; j += tg_size) {
        if (p.has_mask != 0u && mask[j] < 0.5) {
            arow[j] = 0.0;
            continue;
        }
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
    float inv = sum > 0.0 ? 1.0 / sum : 0.0;
    for (uint j = tid; j < p.Lk; j += tg_size) arow[j] = arow[j] * inv;
}

struct HeadAvgParams {
    uint Lq;
    uint Lk;
    uint H;
};

// AttnAvg(i, j) = (1/H) sum_h Attnh(h, i, j) — FP32 -> FP16.
kernel void k_cxa_head_average(device const float* Attnh   [[buffer(0)]],
                               device half*        AttnAvg [[buffer(1)]],
                               constant HeadAvgParams& p   [[buffer(2)]],
                               uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint i = gid.y;
    if (i >= p.Lq || j >= p.Lk) return;
    float acc = 0.0;
    for (uint hh = 0; hh < p.H; ++hh) {
        acc += Attnh[(hh * p.Lq + i) * p.Lk + j];
    }
    AttnAvg[uint(i) * p.Lk + j] = half(acc / float(p.H));
}

struct ApplyVParams {
    uint Lq;
    uint Lk;
    uint dh;
    uint D;
};

// Y_h(i, k) = sum_j Attnh(h, i, j) * Vh(h, j, k); writes Yconcat(i, hh*dh+k).
kernel void k_cxa_attn_apply_v(device const float* Attnh   [[buffer(0)]],
                               device const float* Vh      [[buffer(1)]],
                               device float*       Yconcat [[buffer(2)]],
                               constant ApplyVParams& p    [[buffer(3)]],
                               uint3 gid [[thread_position_in_grid]]) {
    uint k  = gid.x;
    uint i  = gid.y;
    uint hh = gid.z;
    if (i >= p.Lq || k >= p.dh) return;
    uint arow = (hh * p.Lq + i) * p.Lk;
    float acc = 0.0;
    for (uint j = 0; j < p.Lk; ++j) {
        uint vrow = (hh * p.Lk + j) * p.dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[uint(i) * p.D + (hh * p.dh + k)] = acc;
}

struct OutProjParams {
    uint Lq;
    uint D;
};

// O = Yconcat @ Wo^T. Y: (Lq, D) FP32, Wo: (D, D) FP16, O: (Lq, D) FP16.
kernel void k_cxa_output_proj(device const float* Y   [[buffer(0)]],
                              device const half*  Wo  [[buffer(1)]],
                              device half*        O   [[buffer(2)]],
                              constant OutProjParams& p [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint i = gid.y;
    if (i >= p.Lq || c >= p.D) return;
    device const float* yr = Y  + uint(i) * p.D;
    device const half*  wr = Wo + uint(c) * p.D;
    float acc = 0.0;
    for (uint k = 0; k < p.D; ++k) acc += yr[k] * float(wr[k]);
    O[uint(i) * p.D + c] = half(acc);
}
)msl";

struct ProjParams    { uint32_t L, Din, dh; };
struct ScoresParams  { uint32_t Lq, Lk, dh, has_bias; float inv_sqrtdh; };
struct SoftmaxParams { uint32_t Lk, has_mask; };
struct HeadAvgParams { uint32_t Lq, Lk, H; };
struct ApplyVParams  { uint32_t Lq, Lk, dh, D; };
struct OutProjParams { uint32_t Lq, D; };

id<MTLComputePipelineState> pso_proj() {
    static dispatch_once_t once; static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cxa_proj"); });
    return pso;
}
id<MTLComputePipelineState> pso_scores() {
    static dispatch_once_t once; static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cxa_scores"); });
    return pso;
}
id<MTLComputePipelineState> pso_row_softmax() {
    static dispatch_once_t once; static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cxa_row_softmax"); });
    return pso;
}
id<MTLComputePipelineState> pso_head_avg() {
    static dispatch_once_t once; static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cxa_head_average"); });
    return pso;
}
id<MTLComputePipelineState> pso_apply_v() {
    static dispatch_once_t once; static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cxa_attn_apply_v"); });
    return pso;
}
id<MTLComputePipelineState> pso_out_proj() {
    static dispatch_once_t once; static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_cxa_output_proj"); });
    return pso;
}

inline void check_fp16(const GpuTensor& t, const char* name) {
    if (t.dtype != Dtype::FP16) {
        throw std::runtime_error(
            std::string("cross_attention_forward_with_attn_gpu requires FP16 ") + name);
    }
}

void dispatch3d(id<MTLComputePipelineState> pso,
                NSUInteger gx, NSUInteger gy, NSUInteger gz,
                void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger w = 16, h = 16;
        NSUInteger maxT = [pso maxTotalThreadsPerThreadgroup];
        if (w * h > maxT) { w = 8; h = 8; }
        if (gz > 1) { /* 3D grid: use small per-axis TG */ }
        MTLSize grid = MTLSizeMake(gx, gy, gz);
        MTLSize tg   = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void cross_attention_forward_with_attn_gpu(const GpuTensor& X,
                                           const GpuTensor& Ctx,
                                           const GpuTensor& Wq, const GpuTensor& Wk,
                                           const GpuTensor& Wv, const GpuTensor& Wo,
                                           const float* d_mask,
                                           const GpuTensor* attn_logit_bias,
                                           int num_heads,
                                           GpuTensor& O,
                                           GpuTensor& AttnAvg) {
    check_fp16(X, "X");
    check_fp16(Ctx, "Ctx");
    check_fp16(Wq, "Wq"); check_fp16(Wk, "Wk");
    check_fp16(Wv, "Wv"); check_fp16(Wo, "Wo");

    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;
    if (H <= 0 || dh * H != D) {
        throw std::runtime_error("cross_attention_forward_with_attn_gpu: num_heads must divide D");
    }
    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (AttnAvg.rows != Lq || AttnAvg.cols != Lk || AttnAvg.dtype != Dtype::FP16) {
        AttnAvg.resize(Lq, Lk, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    if (attn_logit_bias) {
        if (attn_logit_bias->dtype != Dtype::FP32) {
            throw std::runtime_error(
                "cross_attention_forward_with_attn_gpu: attn_logit_bias must be FP32");
        }
        if (attn_logit_bias->rows != Lq || attn_logit_bias->cols != Lk) {
            throw std::runtime_error(
                "cross_attention_forward_with_attn_gpu: attn_logit_bias must be (Lq, Lk)");
        }
    }

    GpuTensor Qh(H * Lq, dh, Dtype::FP32);
    GpuTensor Kh(H * Lk, dh, Dtype::FP32);
    GpuTensor Vh(H * Lk, dh, Dtype::FP32);
    GpuTensor scores(H * Lq, Lk, Dtype::FP32);
    GpuTensor Attnh(H * Lq, Lk, Dtype::FP32);
    GpuTensor Yconcat(Lq, D, Dtype::FP32);

    id<MTLBuffer> bX   = buffer_for(X);    const NSUInteger oX   = buffer_offset_for(X);
    id<MTLBuffer> bCtx = buffer_for(Ctx);  const NSUInteger oCtx = buffer_offset_for(Ctx);
    id<MTLBuffer> bWq  = buffer_for(Wq);   const NSUInteger oWq  = buffer_offset_for(Wq);
    id<MTLBuffer> bWk  = buffer_for(Wk);   const NSUInteger oWk  = buffer_offset_for(Wk);
    id<MTLBuffer> bWv  = buffer_for(Wv);   const NSUInteger oWv  = buffer_offset_for(Wv);
    id<MTLBuffer> bWo  = buffer_for(Wo);   const NSUInteger oWo  = buffer_offset_for(Wo);
    id<MTLBuffer> bO   = buffer_for(O);    const NSUInteger oO   = buffer_offset_for(O);
    id<MTLBuffer> bAA  = buffer_for(AttnAvg); const NSUInteger oAA = buffer_offset_for(AttnAvg);
    id<MTLBuffer> bQh  = buffer_for(Qh);   const NSUInteger oQh  = buffer_offset_for(Qh);
    id<MTLBuffer> bKh  = buffer_for(Kh);   const NSUInteger oKh  = buffer_offset_for(Kh);
    id<MTLBuffer> bVh  = buffer_for(Vh);   const NSUInteger oVh  = buffer_offset_for(Vh);
    id<MTLBuffer> bS   = buffer_for(scores); const NSUInteger oS = buffer_offset_for(scores);
    id<MTLBuffer> bAn  = buffer_for(Attnh); const NSUInteger oAn = buffer_offset_for(Attnh);
    id<MTLBuffer> bY   = buffer_for(Yconcat); const NSUInteger oY = buffer_offset_for(Yconcat);

    // --- Projections: Q, K, V ---
    {
        ProjParams p{};
        p.L = Lq; p.Din = D; p.dh = dh;
        dispatch3d(pso_proj(), dh, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX  offset:oX  atIndex:0];
            [enc setBuffer:bWq offset:oWq atIndex:1];
            [enc setBuffer:bQh offset:oQh atIndex:2];
            [enc setBytes:&p length:sizeof(ProjParams) atIndex:3];
        });
    }
    {
        ProjParams p{};
        p.L = Lk; p.Din = Dctx; p.dh = dh;
        dispatch3d(pso_proj(), dh, Lk, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bCtx offset:oCtx atIndex:0];
            [enc setBuffer:bWk  offset:oWk  atIndex:1];
            [enc setBuffer:bKh  offset:oKh  atIndex:2];
            [enc setBytes:&p length:sizeof(ProjParams) atIndex:3];
        });
        dispatch3d(pso_proj(), dh, Lk, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bCtx offset:oCtx atIndex:0];
            [enc setBuffer:bWv  offset:oWv  atIndex:1];
            [enc setBuffer:bVh  offset:oVh  atIndex:2];
            [enc setBytes:&p length:sizeof(ProjParams) atIndex:3];
        });
    }

    // --- Scores: S = Q · K^T / sqrt(dh) + bias ---
    {
        ScoresParams p{};
        p.Lq = Lq; p.Lk = Lk; p.dh = dh;
        p.has_bias = attn_logit_bias ? 1u : 0u;
        p.inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
        id<MTLBuffer> bBias = attn_logit_bias ? buffer_for(*attn_logit_bias) : bQh;
        const NSUInteger oBias = attn_logit_bias ? buffer_offset_for(*attn_logit_bias) : 0;
        dispatch3d(pso_scores(), Lk, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bQh   offset:oQh  atIndex:0];
            [enc setBuffer:bKh   offset:oKh  atIndex:1];
            [enc setBuffer:bBias offset:oBias atIndex:2];
            [enc setBuffer:bS    offset:oS   atIndex:3];
            [enc setBytes:&p length:sizeof(ScoresParams) atIndex:4];
        });
    }

    // --- Row-masked softmax over (H * Lq) rows ---
    {
        SoftmaxParams p{};
        p.Lk = static_cast<uint32_t>(Lk);
        p.has_mask = d_mask ? 1u : 0u;
        id<MTLBuffer> bMask = d_mask ? metal_impl::pool_lookup(d_mask) : bQh;
        const NSUInteger oMask = d_mask ? metal_impl::pool_lookup_offset(d_mask) : 0;
        id<MTLComputePipelineState> pso = pso_row_softmax();
        const uint32_t rows = static_cast<uint32_t>(H) * static_cast<uint32_t>(Lq);
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = new_command_buffer();
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bS    offset:oS   atIndex:0];
            [enc setBuffer:bAn   offset:oAn  atIndex:1];
            [enc setBuffer:bMask offset:oMask atIndex:2];
            [enc setBytes:&p length:sizeof(SoftmaxParams) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(ROW_SM_BLOCK, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    // --- Head average -> AttnAvg ---
    {
        HeadAvgParams p{};
        p.Lq = Lq; p.Lk = Lk; p.H = H;
        dispatch3d(pso_head_avg(), Lk, Lq, 1, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bAn offset:oAn atIndex:0];
            [enc setBuffer:bAA offset:oAA atIndex:1];
            [enc setBytes:&p length:sizeof(HeadAvgParams) atIndex:2];
        });
    }

    // --- Attn @ V into Yconcat ---
    {
        ApplyVParams p{};
        p.Lq = Lq; p.Lk = Lk; p.dh = dh; p.D = D;
        dispatch3d(pso_apply_v(), dh, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bAn offset:oAn atIndex:0];
            [enc setBuffer:bVh offset:oVh atIndex:1];
            [enc setBuffer:bY  offset:oY  atIndex:2];
            [enc setBytes:&p length:sizeof(ApplyVParams) atIndex:3];
        });
    }

    // --- Output projection: O = Yconcat @ Wo^T ---
    {
        OutProjParams p{};
        p.Lq = Lq; p.D = D;
        dispatch3d(pso_out_proj(), D, Lq, 1, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bY  offset:oY  atIndex:0];
            [enc setBuffer:bWo offset:oWo atIndex:1];
            [enc setBuffer:bO  offset:oO  atIndex:2];
            [enc setBytes:&p length:sizeof(OutProjParams) atIndex:3];
        });
    }
}

} // namespace brotensor
