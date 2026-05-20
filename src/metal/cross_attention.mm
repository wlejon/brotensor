#include <brotensor/runtime.h>
#include <brotensor/detail/op_table.h>

#import "internal.h"

#include <cmath>
#include <stdexcept>
#include <vector>

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

namespace {

constexpr NSUInteger ROW_SM_BLOCK = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint ROW_SM_BLOCK = 256;

// In: (L, Din), W: (D, Din), Out: (h*L, dh).
kernel void k_cx_proj(device const float* In  [[buffer(0)]],
                      device const float* W   [[buffer(1)]],
                      device float*       Out [[buffer(2)]],
                      constant uint& L        [[buffer(3)]],
                      constant uint& Din      [[buffer(4)]],
                      constant uint& dh       [[buffer(5)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= L || j >= dh) return;
    uint row_off = hh * dh;
    device const float* xr = In + i * Din;
    device const float* wr = W  + (row_off + j) * Din;
    float acc = 0.0f;
    for (uint k = 0; k < Din; ++k) acc += xr[k] * wr[k];
    uint out_row = hh * L + i;
    Out[out_row * dh + j] = acc;
}

kernel void k_cx_scores(device const float* Qh [[buffer(0)]],
                        device const float* Kh [[buffer(1)]],
                        device float*       S  [[buffer(2)]],
                        constant uint& Lq      [[buffer(3)]],
                        constant uint& Lk      [[buffer(4)]],
                        constant uint& dh      [[buffer(5)]],
                        constant float& inv_sqrtdh [[buffer(6)]],
                        uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= Lq || j >= Lk) return;
    uint qrow = (hh * Lq + i) * dh;
    uint krow = (hh * Lk + j) * dh;
    float s = 0.0f;
    for (uint k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    uint srow = (hh * Lq + i) * Lk;
    S[srow + j] = s * inv_sqrtdh;
}

kernel void k_cx_row_softmax(device const float* scores [[buffer(0)]],
                             device float*       Attn   [[buffer(1)]],
                             device const float* mask   [[buffer(2)]],
                             constant uint& has_mask    [[buffer(3)]],
                             constant uint& gate_query  [[buffer(4)]],
                             constant uint& Lq          [[buffer(5)]],
                             constant uint& Lk          [[buffer(6)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    uint i_within = row % Lq;
    device const float* srow = scores + row * Lk;
    device float*       arow = Attn + row * Lk;
    if (gate_query && has_mask && mask[i_within] < 0.5f) {
        for (uint j = tid; j < Lk; j += tg_size) arow[j] = 0.0f;
        return;
    }
    float local_max = -1e30f;
    for (uint j = tid; j < Lk; j += tg_size) {
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
    for (uint j = tid; j < Lk; j += tg_size) {
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
    for (uint j = tid; j < Lk; j += tg_size) arow[j] = arow[j] * inv;
}

kernel void k_cx_attn_apply_v(device const float* Attnh   [[buffer(0)]],
                              device const float* Vh      [[buffer(1)]],
                              device float*       Yconcat [[buffer(2)]],
                              constant uint& Lq           [[buffer(3)]],
                              constant uint& Lk           [[buffer(4)]],
                              constant uint& dh           [[buffer(5)]],
                              constant uint& D            [[buffer(6)]],
                              uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= Lq || k >= dh) return;
    uint arow = (hh * Lq + i) * Lk;
    float acc = 0.0f;
    for (uint j = 0; j < Lk; ++j) {
        uint vrow = (hh * Lk + j) * dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[i * D + (hh * dh + k)] = acc;
}

kernel void k_cx_output_proj(device const float* Y    [[buffer(0)]],
                             device const float* Wo   [[buffer(1)]],
                             device const float* mask [[buffer(2)]],
                             constant uint& has_mask  [[buffer(3)]],
                             constant uint& gate_query[[buffer(4)]],
                             device float*       O    [[buffer(5)]],
                             constant uint& Lq        [[buffer(6)]],
                             constant uint& D         [[buffer(7)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x; uint i = gid.y;
    if (i >= Lq || c >= D) return;
    if (gate_query && has_mask && mask[i] < 0.5f) {
        O[i * D + c] = 0.0f;
        return;
    }
    device const float* yr = Y + i * D;
    device const float* wr = Wo + c * D;
    float acc = 0.0f;
    for (uint k = 0; k < D; ++k) acc += yr[k] * wr[k];
    O[i * D + c] = acc;
}

kernel void k_cx_wo_back_dW(device const float* dO   [[buffer(0)]],
                            device const float* Y    [[buffer(1)]],
                            device const float* mask [[buffer(2)]],
                            constant uint& has_mask  [[buffer(3)]],
                            constant uint& gate_query[[buffer(4)]],
                            device float*       dWo  [[buffer(5)]],
                            constant uint& Lq        [[buffer(6)]],
                            constant uint& D         [[buffer(7)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint c = gid.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < Lq; ++i) {
        if (gate_query && has_mask && mask[i] < 0.5f) continue;
        acc += dO[i * D + c] * Y[i * D + k];
    }
    dWo[c * D + k] += acc;
}

kernel void k_cx_wo_back_dY(device const float* dO  [[buffer(0)]],
                            device const float* Wo  [[buffer(1)]],
                            device const float* mask[[buffer(2)]],
                            constant uint& has_mask [[buffer(3)]],
                            constant uint& gate_query[[buffer(4)]],
                            device float*       dY  [[buffer(5)]],
                            constant uint& Lq       [[buffer(6)]],
                            constant uint& D        [[buffer(7)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= Lq || k >= D) return;
    if (gate_query && has_mask && mask[i] < 0.5f) {
        dY[i * D + k] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (uint c = 0; c < D; ++c) {
        acc += Wo[c * D + k] * dO[i * D + c];
    }
    dY[i * D + k] = acc;
}

kernel void k_cx_dAttn(device const float* dYconcat [[buffer(0)]],
                       device const float* Vh       [[buffer(1)]],
                       device float*       dAttn    [[buffer(2)]],
                       constant uint& Lq            [[buffer(3)]],
                       constant uint& Lk            [[buffer(4)]],
                       constant uint& dh            [[buffer(5)]],
                       constant uint& D             [[buffer(6)]],
                       uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= Lq || j >= Lk) return;
    float acc = 0.0f;
    for (uint k = 0; k < dh; ++k) {
        float dy = dYconcat[i * D + (hh * dh + k)];
        float vv = Vh[(hh * Lk + j) * dh + k];
        acc += dy * vv;
    }
    dAttn[(hh * Lq + i) * Lk + j] = acc;
}

kernel void k_cx_dV(device const float* Attnh    [[buffer(0)]],
                    device const float* dYconcat [[buffer(1)]],
                    device float*       dVh      [[buffer(2)]],
                    constant uint& Lq            [[buffer(3)]],
                    constant uint& Lk            [[buffer(4)]],
                    constant uint& dh            [[buffer(5)]],
                    constant uint& D             [[buffer(6)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y; uint hh = gid.z;
    if (j >= Lk || k >= dh) return;
    float acc = 0.0f;
    for (uint i = 0; i < Lq; ++i) {
        float a  = Attnh[(hh * Lq + i) * Lk + j];
        float dy = dYconcat[i * D + (hh * dh + k)];
        acc += a * dy;
    }
    dVh[(hh * Lk + j) * dh + k] = acc;
}

kernel void k_cx_row_softmax_back(device const float* Attn    [[buffer(0)]],
                                  device const float* dAttn   [[buffer(1)]],
                                  device const float* mask    [[buffer(2)]],
                                  constant uint& has_mask     [[buffer(3)]],
                                  constant uint& gate_query   [[buffer(4)]],
                                  device float*       dScores [[buffer(5)]],
                                  constant uint& Lq           [[buffer(6)]],
                                  constant uint& Lk           [[buffer(7)]],
                                  constant float& inv_sqrtdh  [[buffer(8)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float sdata[ROW_SM_BLOCK];
    uint i_within = row % Lq;
    device const float* prow  = Attn  + row * Lk;
    device const float* dprow = dAttn + row * Lk;
    device float*       drow  = dScores + row * Lk;
    if (gate_query && has_mask && mask[i_within] < 0.5f) {
        for (uint j = tid; j < Lk; j += tg_size) drow[j] = 0.0f;
        return;
    }
    float local = 0.0f;
    for (uint j = tid; j < Lk; j += tg_size) local += dprow[j] * prow[j];
    sdata[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = sdata[0];
    for (uint j = tid; j < Lk; j += tg_size) {
        if (has_mask && mask[j] < 0.5f) drow[j] = 0.0f;
        else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
    }
}

kernel void k_cx_dQ(device const float* dScores [[buffer(0)]],
                    device const float* Kh      [[buffer(1)]],
                    device float*       dQh     [[buffer(2)]],
                    constant uint& Lq           [[buffer(3)]],
                    constant uint& Lk           [[buffer(4)]],
                    constant uint& dh           [[buffer(5)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y; uint hh = gid.z;
    if (i >= Lq || k >= dh) return;
    float acc = 0.0f;
    for (uint j = 0; j < Lk; ++j) {
        float ds = dScores[(hh * Lq + i) * Lk + j];
        float kk = Kh[(hh * Lk + j) * dh + k];
        acc += ds * kk;
    }
    dQh[(hh * Lq + i) * dh + k] = acc;
}

kernel void k_cx_dK(device const float* dScores [[buffer(0)]],
                    device const float* Qh      [[buffer(1)]],
                    device float*       dKh     [[buffer(2)]],
                    constant uint& Lq           [[buffer(3)]],
                    constant uint& Lk           [[buffer(4)]],
                    constant uint& dh           [[buffer(5)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y; uint hh = gid.z;
    if (j >= Lk || k >= dh) return;
    float acc = 0.0f;
    for (uint i = 0; i < Lq; ++i) {
        float ds = dScores[(hh * Lq + i) * Lk + j];
        float qq = Qh[(hh * Lq + i) * dh + k];
        acc += ds * qq;
    }
    dKh[(hh * Lk + j) * dh + k] = acc;
}

// dW(wrow, k_col) += sum_i dHh(hh, i, j) * In(i, k_col); wrow = hh*dh + j.
kernel void k_cx_dW_proj(device const float* dHh [[buffer(0)]],
                         device const float* In  [[buffer(1)]],
                         device float*       dW  [[buffer(2)]],
                         constant uint& L        [[buffer(3)]],
                         constant uint& D        [[buffer(4)]],
                         constant uint& Din      [[buffer(5)]],
                         constant uint& dh       [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint k_col = gid.x; uint wrow = gid.y;
    if (wrow >= D || k_col >= Din) return;
    uint hh = wrow / dh;
    uint j  = wrow % dh;
    float acc = 0.0f;
    for (uint i = 0; i < L; ++i) {
        float xv = In[i * Din + k_col];
        acc += dHh[(hh * L + i) * dh + j] * xv;
    }
    dW[wrow * Din + k_col] += acc;
}

kernel void k_cx_dX(device const float* dQh [[buffer(0)]],
                    device const float* Wq  [[buffer(1)]],
                    device float*       dX  [[buffer(2)]],
                    constant uint& Lq       [[buffer(3)]],
                    constant uint& D        [[buffer(4)]],
                    constant uint& dh       [[buffer(5)]],
                    constant uint& H        [[buffer(6)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint i = gid.y;
    if (i >= Lq || k >= D) return;
    float acc = 0.0f;
    for (uint hh = 0; hh < H; ++hh) {
        for (uint j = 0; j < dh; ++j) {
            uint wrow = hh * dh + j;
            float gq = dQh[(hh * Lq + i) * dh + j];
            acc += gq * Wq[wrow * D + k];
        }
    }
    dX[i * D + k] = acc;
}

kernel void k_cx_dCtx(device const float* dKh [[buffer(0)]],
                      device const float* dVh [[buffer(1)]],
                      device const float* Wk  [[buffer(2)]],
                      device const float* Wv  [[buffer(3)]],
                      device float*       dCtx[[buffer(4)]],
                      constant uint& Lk       [[buffer(5)]],
                      constant uint& D        [[buffer(6)]],
                      constant uint& Dctx     [[buffer(7)]],
                      constant uint& dh       [[buffer(8)]],
                      constant uint& H        [[buffer(9)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x; uint j = gid.y;
    if (j >= Lk || k >= Dctx) return;
    float acc = 0.0f;
    for (uint hh = 0; hh < H; ++hh) {
        for (uint m = 0; m < dh; ++m) {
            uint wrow = hh * dh + m;
            uint widx = wrow * Dctx + k;
            float gk = dKh[(hh * Lk + j) * dh + m];
            float gv = dVh[(hh * Lk + j) * dh + m];
            acc += gk * Wk[widx] + gv * Wv[widx];
        }
    }
    dCtx[j * Dctx + k] = acc;
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_cx_proj,    @"k_cx_proj")
DEF_PSO(pso_cx_scores,  @"k_cx_scores")
DEF_PSO(pso_cx_rsm,     @"k_cx_row_softmax")
DEF_PSO(pso_cx_av,      @"k_cx_attn_apply_v")
DEF_PSO(pso_cx_op,      @"k_cx_output_proj")
DEF_PSO(pso_cx_wodW,    @"k_cx_wo_back_dW")
DEF_PSO(pso_cx_wodY,    @"k_cx_wo_back_dY")
DEF_PSO(pso_cx_dAttn,   @"k_cx_dAttn")
DEF_PSO(pso_cx_dV,      @"k_cx_dV")
DEF_PSO(pso_cx_rsmb,    @"k_cx_row_softmax_back")
DEF_PSO(pso_cx_dQ,      @"k_cx_dQ")
DEF_PSO(pso_cx_dK,      @"k_cx_dK")
DEF_PSO(pso_cx_dWp,     @"k_cx_dW_proj")
DEF_PSO(pso_cx_dX,      @"k_cx_dX")
DEF_PSO(pso_cx_dCtx,    @"k_cx_dCtx")
#undef DEF_PSO

void run3d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny, NSUInteger nz,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger w = 8, h = 8;
        if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
        [enc dispatchThreads:MTLSizeMake(nx, ny, nz)
        threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void run2d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger w = 16, h = 16;
        if (w > [pso threadExecutionWidth]) w = [pso threadExecutionWidth];
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1)
        threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (rows == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(ROW_SM_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

inline void check_fp32(const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("cross_attention training path requires FP32 ") + name);
    }
}

void cross_attention_forward_train_core(const Tensor& X,
                                        const Tensor& Ctx,
                                        const Tensor& Wq, const Tensor& Wk,
                                        const Tensor& Wv, const Tensor& Wo,
                                        const float* d_mask,
                                        int num_heads,
                                        Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                        Tensor& Attnh, Tensor& Yconcat,
                                        Tensor& O) {
    check_fp32(X, "X"); check_fp32(Ctx, "Ctx");
    check_fp32(Wq, "Wq"); check_fp32(Wk, "Wk");
    check_fp32(Wv, "Wv"); check_fp32(Wo, "Wo");

    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;
    if (Qh.rows != H * Lq || Qh.cols != dh) Qh.resize(H * Lq, dh);
    if (Kh.rows != H * Lk || Kh.cols != dh) Kh.resize(H * Lk, dh);
    if (Vh.rows != H * Lk || Vh.cols != dh) Vh.resize(H * Lk, dh);
    if (Attnh.rows != H * Lq || Attnh.cols != Lk) Attnh.resize(H * Lq, Lk);
    if (Yconcat.rows != Lq || Yconcat.cols != D) Yconcat.resize(Lq, D);
    if (O.rows != Lq || O.cols != D) O.resize(Lq, D);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const uint32_t Lqu = static_cast<uint32_t>(Lq);
    const uint32_t Lku = static_cast<uint32_t>(Lk);
    const uint32_t Du  = static_cast<uint32_t>(D);
    const uint32_t Dctxu = static_cast<uint32_t>(Dctx);
    const uint32_t dhU = static_cast<uint32_t>(dh);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const uint32_t gate_query = (Lq == Lk) ? 1u : 0u;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bC = buffer_for(Ctx); NSUInteger oC = buffer_offset_for(Ctx);
    id<MTLBuffer> bWq = buffer_for(Wq); NSUInteger oWq = buffer_offset_for(Wq);
    id<MTLBuffer> bWk = buffer_for(Wk); NSUInteger oWk = buffer_offset_for(Wk);
    id<MTLBuffer> bWv = buffer_for(Wv); NSUInteger oWv = buffer_offset_for(Wv);
    id<MTLBuffer> bWo = buffer_for(Wo); NSUInteger oWo = buffer_offset_for(Wo);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bAh = buffer_for(Attnh); NSUInteger oAh = buffer_offset_for(Attnh);
    id<MTLBuffer> bYc = buffer_for(Yconcat); NSUInteger oYc = buffer_offset_for(Yconcat);
    id<MTLBuffer> bO  = buffer_for(O); NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bM  = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? metal_impl::pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;

    // Q projection from X (Din = D).
    run3d(pso_cx_proj(), dh, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bWq offset:oWq atIndex:1];
        [enc setBuffer:bQh offset:oQh atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
    });
    // K, V from Ctx (Din = Dctx).
    auto proj_ctx = ^(id<MTLBuffer> bW, NSUInteger oWp,
                      id<MTLBuffer> bOut, NSUInteger oOut) {
        run3d(pso_cx_proj(), dh, Lk, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bC offset:oC atIndex:0];
            [enc setBuffer:bW offset:oWp atIndex:1];
            [enc setBuffer:bOut offset:oOut atIndex:2];
            [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&Dctxu length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        });
    };
    proj_ctx(bWk, oWk, bKh, oKh);
    proj_ctx(bWv, oWv, bVh, oVh);

    Tensor scores = Tensor::empty_on(Device::Metal, H * Lq, Lk);
    id<MTLBuffer> bS = buffer_for(scores); NSUInteger oS = buffer_offset_for(scores);
    run3d(pso_cx_scores(), Lk, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh offset:oQh atIndex:0];
        [enc setBuffer:bKh offset:oKh atIndex:1];
        [enc setBuffer:bS offset:oS atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&inv_sqrtdh length:sizeof(float) atIndex:6];
    });

    run_rows(pso_cx_rsm(), H * Lq, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS offset:oS atIndex:0];
        [enc setBuffer:bAh offset:oAh atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&gate_query length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:6];
    });

    run3d(pso_cx_av(), dh, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bAh offset:oAh atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });

    run2d(pso_cx_op(), D, Lq, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc offset:oYc atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&gate_query length:sizeof(uint32_t) atIndex:4];
        [enc setBuffer:bO offset:oO atIndex:5];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:7];
    });
}

} // namespace

void self_attention_forward_train(const Tensor& X,
                                      const Tensor& Wq, const Tensor& Wk,
                                      const Tensor& Wv, const Tensor& Wo,
                                      const float* d_mask,
                                      int num_heads,
                                      Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                      Tensor& Attnh, Tensor& Yconcat,
                                      Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo, d_mask, num_heads,
                    Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_backward(const Tensor& dO,
                                 const Tensor& X,
                                 const Tensor& Qh, const Tensor& Kh,
                                 const Tensor& Vh, const Tensor& Attnh,
                                 const Tensor& Yconcat,
                                 const Tensor& Wq, const Tensor& Wk,
                                 const Tensor& Wv, const Tensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 Tensor& dX,
                                 Tensor& dWq, Tensor& dWk,
                                 Tensor& dWv, Tensor& dWo) {
    mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                     Wq, Wk, Wv, Wo, d_mask, num_heads,
                     dX, dWq, dWk, dWv, dWo);
}

void cross_attention_forward_train(const Tensor& X,
                                       const Tensor& Ctx,
                                       const Tensor& Wq, const Tensor& Wk,
                                       const Tensor& Wv, const Tensor& Wo,
                                       const float* d_mask,
                                       int num_heads,
                                       Tensor& Qh, Tensor& Kh, Tensor& Vh,
                                       Tensor& Attnh, Tensor& Yconcat,
                                       Tensor& O) {
    cross_attention_forward_train_core(X, Ctx, Wq, Wk, Wv, Wo, d_mask,
                                       num_heads, Qh, Kh, Vh, Attnh,
                                       Yconcat, O);
}

void cross_attention_backward(const Tensor& dO,
                                  const Tensor& X,
                                  const Tensor& Ctx,
                                  const Tensor& Qh, const Tensor& Kh,
                                  const Tensor& Vh, const Tensor& Attnh,
                                  const Tensor& Yconcat,
                                  const Tensor& Wq, const Tensor& Wk,
                                  const Tensor& Wv, const Tensor& Wo,
                                  const float* d_mask,
                                  int num_heads,
                                  Tensor& dX,
                                  Tensor& dCtx,
                                  Tensor& dWq, Tensor& dWk,
                                  Tensor& dWv, Tensor& dWo) {
    check_fp32(dO, "dO"); check_fp32(X, "X"); check_fp32(Ctx, "Ctx");

    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;
    if (dX.rows != Lq || dX.cols != D) dX.resize(Lq, D);
    if (dCtx.rows != Lk || dCtx.cols != Dctx) dCtx.resize(Lk, Dctx);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const uint32_t Lqu = static_cast<uint32_t>(Lq);
    const uint32_t Lku = static_cast<uint32_t>(Lk);
    const uint32_t Du  = static_cast<uint32_t>(D);
    const uint32_t Dctxu = static_cast<uint32_t>(Dctx);
    const uint32_t dhU = static_cast<uint32_t>(dh);
    const uint32_t Hu  = static_cast<uint32_t>(H);
    const uint32_t has_mask = d_mask ? 1u : 0u;
    const uint32_t gate_query = (Lq == Lk) ? 1u : 0u;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    id<MTLBuffer> bdO = buffer_for(dO); NSUInteger odO = buffer_offset_for(dO);
    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bC = buffer_for(Ctx); NSUInteger oC = buffer_offset_for(Ctx);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bAh = buffer_for(Attnh); NSUInteger oAh = buffer_offset_for(Attnh);
    id<MTLBuffer> bYc = buffer_for(Yconcat); NSUInteger oYc = buffer_offset_for(Yconcat);
    id<MTLBuffer> bWq = buffer_for(Wq); NSUInteger oWq = buffer_offset_for(Wq);
    id<MTLBuffer> bWk = buffer_for(Wk); NSUInteger oWk = buffer_offset_for(Wk);
    id<MTLBuffer> bWv = buffer_for(Wv); NSUInteger oWv = buffer_offset_for(Wv);
    id<MTLBuffer> bWo = buffer_for(Wo); NSUInteger oWo = buffer_offset_for(Wo);
    id<MTLBuffer> bdX = buffer_for(dX); NSUInteger odX = buffer_offset_for(dX);
    id<MTLBuffer> bdCtx = buffer_for(dCtx); NSUInteger odCtx = buffer_offset_for(dCtx);
    id<MTLBuffer> bdWq = buffer_for(dWq); NSUInteger odWq = buffer_offset_for(dWq);
    id<MTLBuffer> bdWk = buffer_for(dWk); NSUInteger odWk = buffer_offset_for(dWk);
    id<MTLBuffer> bdWv = buffer_for(dWv); NSUInteger odWv = buffer_offset_for(dWv);
    id<MTLBuffer> bdWo = buffer_for(dWo); NSUInteger odWo = buffer_offset_for(dWo);
    id<MTLBuffer> bM = d_mask ? pool_lookup(d_mask) : nil;
    NSUInteger oM = d_mask ? metal_impl::pool_lookup_offset(d_mask) : 0;
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;

    Tensor dYconcat = Tensor::empty_on(Device::Metal, Lq, D);
    id<MTLBuffer> bdY = buffer_for(dYconcat); NSUInteger odY = buffer_offset_for(dYconcat);
    run2d(pso_cx_wodY(), D, Lq, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&gate_query length:sizeof(uint32_t) atIndex:4];
        [enc setBuffer:bdY offset:odY atIndex:5];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:7];
    });
    run2d(pso_cx_wodW(), D, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdO offset:odO atIndex:0];
        [enc setBuffer:bYc offset:oYc atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&gate_query length:sizeof(uint32_t) atIndex:4];
        [enc setBuffer:bdWo offset:odWo atIndex:5];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:7];
    });

    Tensor dAttn = Tensor::empty_on(Device::Metal, H * Lq, Lk);
    Tensor dVh = Tensor::empty_on(Device::Metal, H * Lk, dh);
    id<MTLBuffer> bdA = buffer_for(dAttn); NSUInteger odA = buffer_offset_for(dAttn);
    id<MTLBuffer> bdVh = buffer_for(dVh); NSUInteger odVh = buffer_offset_for(dVh);
    run3d(pso_cx_dAttn(), Lk, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdY offset:odY atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bdA offset:odA atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });
    run3d(pso_cx_dV(), dh, Lk, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bAh offset:oAh atIndex:0];
        [enc setBuffer:bdY offset:odY atIndex:1];
        [enc setBuffer:bdVh offset:odVh atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
    });

    Tensor dScores = Tensor::empty_on(Device::Metal, H * Lq, Lk);
    id<MTLBuffer> bdS = buffer_for(dScores); NSUInteger odS = buffer_offset_for(dScores);
    run_rows(pso_cx_rsmb(), H * Lq, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bAh offset:oAh atIndex:0];
        [enc setBuffer:bdA offset:odA atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&gate_query length:sizeof(uint32_t) atIndex:4];
        [enc setBuffer:bdS offset:odS atIndex:5];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&inv_sqrtdh length:sizeof(float) atIndex:8];
    });

    Tensor dQh = Tensor::empty_on(Device::Metal, H * Lq, dh);
    Tensor dKh = Tensor::empty_on(Device::Metal, H * Lk, dh);
    id<MTLBuffer> bdQh = buffer_for(dQh); NSUInteger odQh = buffer_offset_for(dQh);
    id<MTLBuffer> bdKh = buffer_for(dKh); NSUInteger odKh = buffer_offset_for(dKh);
    run3d(pso_cx_dQ(), dh, Lq, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdS offset:odS atIndex:0];
        [enc setBuffer:bKh offset:oKh atIndex:1];
        [enc setBuffer:bdQh offset:odQh atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
    });
    run3d(pso_cx_dK(), dh, Lk, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdS offset:odS atIndex:0];
        [enc setBuffer:bQh offset:oQh atIndex:1];
        [enc setBuffer:bdKh offset:odKh atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
    });

    // dWq (D, D) from dQh @ X.
    run2d(pso_cx_dWp(), D, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdQh offset:odQh atIndex:0];
        [enc setBuffer:bX offset:oX atIndex:1];
        [enc setBuffer:bdWq offset:odWq atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:6];
    });
    // dWk, dWv (D, Dctx) from dKh, dVh @ Ctx.
    run2d(pso_cx_dWp(), Dctx, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdKh offset:odKh atIndex:0];
        [enc setBuffer:bC offset:oC atIndex:1];
        [enc setBuffer:bdWk offset:odWk atIndex:2];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Dctxu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:6];
    });
    run2d(pso_cx_dWp(), Dctx, D, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdVh offset:odVh atIndex:0];
        [enc setBuffer:bC offset:oC atIndex:1];
        [enc setBuffer:bdWv offset:odWv atIndex:2];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Dctxu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:6];
    });

    run2d(pso_cx_dX(), D, Lq, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdQh offset:odQh atIndex:0];
        [enc setBuffer:bWq offset:oWq atIndex:1];
        [enc setBuffer:bdX offset:odX atIndex:2];
        [enc setBytes:&Lqu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:6];
    });
    run2d(pso_cx_dCtx(), Dctx, Lk, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bdKh offset:odKh atIndex:0];
        [enc setBuffer:bdVh offset:odVh atIndex:1];
        [enc setBuffer:bWk offset:oWk atIndex:2];
        [enc setBuffer:bWv offset:oWv atIndex:3];
        [enc setBuffer:bdCtx offset:odCtx atIndex:4];
        [enc setBytes:&Lku length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Dctxu length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:9];
    });
}

void cross_attention_forward(const Tensor& X,
                                 const Tensor& Ctx,
                                 const Tensor& Wq, const Tensor& Wk,
                                 const Tensor& Wv, const Tensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 Tensor& O) {
    if (X.dtype == Dtype::FP16) {
        if (Ctx.dtype != Dtype::FP16) {
            throw std::runtime_error("cross_attention_forward_gpu: Ctx dtype must match X dtype");
        }
        flash_attention_qkvo_forward(X, &Ctx,
                                         Wq, nullptr, Wk, nullptr,
                                         Wv, nullptr, Wo, nullptr,
                                         d_mask, num_heads, /*causal=*/false, O);
        return;
    }
    if (Ctx.dtype != Dtype::FP32) {
        throw std::runtime_error("cross_attention_forward_gpu: Ctx dtype must match X dtype");
    }
    Tensor Qh, Kh, Vh, Attnh, Yconcat;
    cross_attention_forward_train_core(X, Ctx, Wq, Wk, Wv, Wo, d_mask,
                                       num_heads, Qh, Kh, Vh, Attnh,
                                       Yconcat, O);
}

void self_attention_forward(const Tensor& X,
                                const Tensor& Wq, const Tensor& Wk,
                                const Tensor& Wv, const Tensor& Wo,
                                const float* d_mask,
                                int num_heads,
                                Tensor& O) {
    if (X.dtype == Dtype::FP16) {
        flash_attention_qkvo_forward(X, nullptr,
                                         Wq, nullptr, Wk, nullptr,
                                         Wv, nullptr, Wo, nullptr,
                                         d_mask, num_heads, /*causal=*/false, O);
        return;
    }
    Tensor Qh, Kh, Vh, Attnh, Yconcat;
    mha_forward(X, Wq, Wk, Wv, Wo, d_mask, num_heads,
                    Qh, Kh, Vh, Attnh, Yconcat, O);
}

} // namespace brotensor::detail::metal
