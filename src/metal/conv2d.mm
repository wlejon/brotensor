#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct ConvParams {
    uint N, C_in, H, W;
    uint C_out, kH, kW;
    uint H_out, W_out;
    int  stride_h, stride_w;
    int  pad_h, pad_w;
    int  dil_h, dil_w;
    uint has_bias;
    uint total;
    uint groups;
    uint Cg_in;
    uint Cg_out;
};

// One thread per output element. Direct conv, FP32 accumulator, FP16 IO.
kernel void k_conv2d_forward_fp16(device const half* X    [[buffer(0)]],
                                  device const half* Wt   [[buffer(1)]],
                                  device const half* bias [[buffer(2)]],
                                  device half*       Y    [[buffer(3)]],
                                  constant ConvParams& p  [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float acc = 0.0f;
    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.H * p.W;

    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.H * p.W;
        for (uint kh = 0; kh < p.kH; ++kh) {
            int in_h = in_h_origin + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint kw = 0; kw < p.kW; ++kw) {
                int in_w = in_w_origin + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                float x_v = float(X[x_ic_base + uint(in_h) * p.W + uint(in_w)]);
                float w_v = float(Wt[w_ic_base + kh * p.kW + kw]);
                acc += x_v * w_v;
            }
        }
    }
    if (p.has_bias != 0u) {
        acc += float(bias[oc]);
    }
    Y[idx] = half(acc);
}

// FP32 variant of forward.
kernel void k_conv2d_forward_fp32(device const float* X    [[buffer(0)]],
                                  device const float* Wt   [[buffer(1)]],
                                  device const float* bias [[buffer(2)]],
                                  device float*       Y    [[buffer(3)]],
                                  constant ConvParams& p   [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float acc = 0.0f;
    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.H * p.W;

    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.H * p.W;
        for (uint kh = 0; kh < p.kH; ++kh) {
            int in_h = in_h_origin + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint kw = 0; kw < p.kW; ++kw) {
                int in_w = in_w_origin + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                float x_v = X[x_ic_base + uint(in_h) * p.W + uint(in_w)];
                float w_v = Wt[w_ic_base + kh * p.kW + kw];
                acc += x_v * w_v;
            }
        }
    }
    if (p.has_bias != 0u) {
        acc += bias[oc];
    }
    Y[idx] = acc;
}

// FP16 backward-w.r.t.-input. Same shape as FP32 but reads/writes half;
// FP32 accumulator.
kernel void k_conv2d_backward_input_fp16(device const half* Wt [[buffer(0)]],
                                         device const half* dY [[buffer(1)]],
                                         device half*       dX [[buffer(2)]],
                                         constant ConvParams& p [[buffer(3)]],
                                         uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint j = idx % p.W;
    uint t = idx / p.W;
    uint i = t % p.H;
    t /= p.H;
    uint c_in = t % p.C_in;
    uint n    = t / p.C_in;

    uint g = c_in / p.Cg_in;
    uint c_in_local = c_in - g * p.Cg_in;
    uint oc_lo = g * p.Cg_out;
    uint oc_hi = oc_lo + p.Cg_out;

    float acc = 0.0f;
    for (uint kh = 0; kh < p.kH; ++kh) {
        int num_h = int(i) + p.pad_h - p.dil_h * int(kh);
        if (num_h < 0) continue;
        if (num_h % p.stride_h != 0) continue;
        int i_out = num_h / p.stride_h;
        if (i_out < 0 || i_out >= int(p.H_out)) continue;
        for (uint kw = 0; kw < p.kW; ++kw) {
            int num_w = int(j) + p.pad_w - p.dil_w * int(kw);
            if (num_w < 0) continue;
            if (num_w % p.stride_w != 0) continue;
            int j_out = num_w / p.stride_w;
            if (j_out < 0 || j_out >= int(p.W_out)) continue;

            for (uint c_out = oc_lo; c_out < oc_hi; ++c_out) {
                uint dy_idx = ((n * p.C_out + c_out) * p.H_out + uint(i_out)) * p.W_out + uint(j_out);
                uint w_idx  = ((c_out * p.Cg_in + c_in_local) * p.kH + kh) * p.kW + kw;
                acc += float(dY[dy_idx]) * float(Wt[w_idx]);
            }
        }
    }
    dX[idx] = half(acc);
}

// FP16 backward-w.r.t.-weights. Writes FP32 scratch; host folds into dWt.
kernel void k_conv2d_backward_weight_fp16(device const half*  X   [[buffer(0)]],
                                          device const half*  dY  [[buffer(1)]],
                                          device float*       dWt_scratch [[buffer(2)]],
                                          constant ConvParams& p  [[buffer(3)]],
                                          uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint kw = idx % p.kW;
    uint t  = idx / p.kW;
    uint kh = t % p.kH;
    t /= p.kH;
    uint c_in_local = t % p.Cg_in;
    uint c_out      = t / p.Cg_in;

    uint g = c_out / p.Cg_out;
    uint c_in = g * p.Cg_in + c_in_local;

    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        for (uint i_out = 0; i_out < p.H_out; ++i_out) {
            int in_h = int(i_out) * p.stride_h - p.pad_h + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint j_out = 0; j_out < p.W_out; ++j_out) {
                int in_w = int(j_out) * p.stride_w - p.pad_w + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                uint x_idx  = ((n * p.C_in  + c_in)  * p.H     + uint(in_h))  * p.W     + uint(in_w);
                uint dy_idx = ((n * p.C_out + c_out) * p.H_out + i_out) * p.W_out + j_out;
                acc += float(dY[dy_idx]) * float(X[x_idx]);
            }
        }
    }
    dWt_scratch[idx] = acc;
}

// FP32 backward-weight variant that writes to scratch (parity with FP16 path).
kernel void k_conv2d_backward_weight_fp32_to_scratch(
        device const float* X   [[buffer(0)]],
        device const float* dY  [[buffer(1)]],
        device float*       dWt_scratch [[buffer(2)]],
        constant ConvParams& p  [[buffer(3)]],
        uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint kw = idx % p.kW;
    uint t  = idx / p.kW;
    uint kh = t % p.kH;
    t /= p.kH;
    uint c_in_local = t % p.Cg_in;
    uint c_out      = t / p.Cg_in;

    uint g = c_out / p.Cg_out;
    uint c_in = g * p.Cg_in + c_in_local;

    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        for (uint i_out = 0; i_out < p.H_out; ++i_out) {
            int in_h = int(i_out) * p.stride_h - p.pad_h + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint j_out = 0; j_out < p.W_out; ++j_out) {
                int in_w = int(j_out) * p.stride_w - p.pad_w + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                uint x_idx  = ((n * p.C_in  + c_in)  * p.H     + uint(in_h))  * p.W     + uint(in_w);
                uint dy_idx = ((n * p.C_out + c_out) * p.H_out + i_out) * p.W_out + j_out;
                acc += dY[dy_idx] * X[x_idx];
            }
        }
    }
    dWt_scratch[idx] = acc;
}

// FP16 backward-w.r.t.-bias. Writes per-c_out FP32 partial sum into scratch.
kernel void k_conv2d_backward_bias_fp16(device const half*  dY [[buffer(0)]],
                                        device float*       dB_scratch [[buffer(1)]],
                                        constant ConvParams& p [[buffer(2)]],
                                        uint  tid    [[thread_position_in_threadgroup]],
                                        uint  tg_sz  [[threads_per_threadgroup]],
                                        uint  gid    [[threadgroup_position_in_grid]]) {
    threadgroup float s_acc[256];
    const uint c_out = gid;
    const uint spatial = p.H_out * p.W_out;
    const uint total_per_chan = p.N * spatial;

    float acc = 0.0f;
    for (uint i = tid; i < total_per_chan; i += tg_sz) {
        uint n  = i / spatial;
        uint sp = i - n * spatial;
        uint dy_idx = (n * p.C_out + c_out) * spatial + sp;
        acc += float(dY[dy_idx]);
    }
    s_acc[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dB_scratch[c_out] = s_acc[0];
}

// FP32 backward-bias to scratch variant.
kernel void k_conv2d_backward_bias_fp32_to_scratch(
        device const float* dY [[buffer(0)]],
        device float*       dB_scratch [[buffer(1)]],
        constant ConvParams& p [[buffer(2)]],
        uint  tid    [[thread_position_in_threadgroup]],
        uint  tg_sz  [[threads_per_threadgroup]],
        uint  gid    [[threadgroup_position_in_grid]]) {
    threadgroup float s_acc[256];
    const uint c_out = gid;
    const uint spatial = p.H_out * p.W_out;
    const uint total_per_chan = p.N * spatial;

    float acc = 0.0f;
    for (uint i = tid; i < total_per_chan; i += tg_sz) {
        uint n  = i / spatial;
        uint sp = i - n * spatial;
        uint dy_idx = (n * p.C_out + c_out) * spatial + sp;
        acc += dY[dy_idx];
    }
    s_acc[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dB_scratch[c_out] = s_acc[0];
}

// Fold FP32 scratch into FP16/FP32 destination (add).
kernel void k_conv2d_add_fp32_into_fp16(device const float* src [[buffer(0)]],
                                        device half*        dst [[buffer(1)]],
                                        constant uint& n        [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = half(float(dst[i]) + src[i]);
}
kernel void k_conv2d_add_fp32_into_fp32(device const float* src [[buffer(0)]],
                                        device float*       dst [[buffer(1)]],
                                        constant uint& n        [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] += src[i];
}

// Backward-w.r.t.-input. One thread per input pixel; gather form, no atomics.
// FP32 only.
kernel void k_conv2d_backward_input_fp32(device const float* Wt [[buffer(0)]],
                                         device const float* dY [[buffer(1)]],
                                         device float*       dX [[buffer(2)]],
                                         constant ConvParams& p [[buffer(3)]],
                                         uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint j = idx % p.W;
    uint t = idx / p.W;
    uint i = t % p.H;
    t /= p.H;
    uint c_in = t % p.C_in;
    uint n    = t / p.C_in;

    uint g = c_in / p.Cg_in;
    uint c_in_local = c_in - g * p.Cg_in;
    uint oc_lo = g * p.Cg_out;
    uint oc_hi = oc_lo + p.Cg_out;

    float acc = 0.0f;
    for (uint kh = 0; kh < p.kH; ++kh) {
        int num_h = int(i) + p.pad_h - p.dil_h * int(kh);
        if (num_h < 0) continue;
        if (num_h % p.stride_h != 0) continue;
        int i_out = num_h / p.stride_h;
        if (i_out < 0 || i_out >= int(p.H_out)) continue;
        for (uint kw = 0; kw < p.kW; ++kw) {
            int num_w = int(j) + p.pad_w - p.dil_w * int(kw);
            if (num_w < 0) continue;
            if (num_w % p.stride_w != 0) continue;
            int j_out = num_w / p.stride_w;
            if (j_out < 0 || j_out >= int(p.W_out)) continue;

            for (uint c_out = oc_lo; c_out < oc_hi; ++c_out) {
                uint dy_idx = ((n * p.C_out + c_out) * p.H_out + uint(i_out)) * p.W_out + uint(j_out);
                uint w_idx  = ((c_out * p.Cg_in + c_in_local) * p.kH + kh) * p.kW + kw;
                acc += dY[dy_idx] * Wt[w_idx];
            }
        }
    }
    dX[idx] = acc;
}

// Backward-w.r.t.-weights. One thread per (c_out, c_in, kh, kw) element of
// dWt. Iterates (n, i_out, j_out) and accumulates into a single dWt slot.
// FP32 only. No atomics. *Accumulates* into dWt (caller zeros).
kernel void k_conv2d_backward_weight_fp32(device const float* X   [[buffer(0)]],
                                          device const float* dY  [[buffer(1)]],
                                          device float*       dWt [[buffer(2)]],
                                          constant ConvParams& p  [[buffer(3)]],
                                          uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    // Unflatten idx → (c_out, c_in_local, kh, kw) in OIHW (I-dim sized as Cg_in
    // for grouped conv).
    uint kw = idx % p.kW;
    uint t  = idx / p.kW;
    uint kh = t % p.kH;
    t /= p.kH;
    uint c_in_local = t % p.Cg_in;
    uint c_out      = t / p.Cg_in;

    uint g = c_out / p.Cg_out;
    uint c_in = g * p.Cg_in + c_in_local;

    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        for (uint i_out = 0; i_out < p.H_out; ++i_out) {
            int in_h = int(i_out) * p.stride_h - p.pad_h + int(kh) * p.dil_h;
            if (in_h < 0 || in_h >= int(p.H)) continue;
            for (uint j_out = 0; j_out < p.W_out; ++j_out) {
                int in_w = int(j_out) * p.stride_w - p.pad_w + int(kw) * p.dil_w;
                if (in_w < 0 || in_w >= int(p.W)) continue;
                uint x_idx  = ((n * p.C_in  + c_in)  * p.H     + uint(in_h))  * p.W     + uint(in_w);
                uint dy_idx = ((n * p.C_out + c_out) * p.H_out + i_out) * p.W_out + j_out;
                acc += dY[dy_idx] * X[x_idx];
            }
        }
    }
    dWt[idx] += acc;
}

// Backward-w.r.t.-bias. One threadgroup per c_out; threadgroup-wide reduce
// over (n, i_out, j_out); thread 0 accumulates into dB[c_out]. FP32 only.
kernel void k_conv2d_backward_bias_fp32(device const float* dY [[buffer(0)]],
                                        device float*       dB [[buffer(1)]],
                                        constant ConvParams& p [[buffer(2)]],
                                        uint  tid    [[thread_position_in_threadgroup]],
                                        uint  tg_sz  [[threads_per_threadgroup]],
                                        uint  gid    [[threadgroup_position_in_grid]]) {
    threadgroup float s_acc[256];
    const uint c_out = gid;
    const uint spatial = p.H_out * p.W_out;
    const uint total_per_chan = p.N * spatial;

    float acc = 0.0f;
    for (uint i = tid; i < total_per_chan; i += tg_sz) {
        uint n  = i / spatial;
        uint sp = i - n * spatial;
        uint dy_idx = (n * p.C_out + c_out) * spatial + sp;
        acc += dY[dy_idx];
    }
    s_acc[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dB[c_out] += s_acc[0];
}
)msl";

id<MTLComputePipelineState> pso_conv_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_forward_fp16"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_forward_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_input_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_input_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_weight_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_bias_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_fp32"); });
    return pso;
}

id<MTLComputePipelineState> pso_conv_bwd_input_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_input_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_weight_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_weight_fp32_scratch() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_weight_fp32_to_scratch"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_bias_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_bwd_bias_fp32_scratch() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_backward_bias_fp32_to_scratch"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_add_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_add_fp32_into_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv_add_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv2d_add_fp32_into_fp32"); });
    return pso;
}

struct ConvParams {
    uint32_t N, C_in, H, W;
    uint32_t C_out, kH, kW;
    uint32_t H_out, W_out;
    int32_t  stride_h, stride_w;
    int32_t  pad_h, pad_w;
    int32_t  dil_h, dil_w;
    uint32_t has_bias;
    uint32_t total;
    uint32_t groups;
    uint32_t Cg_in;
    uint32_t Cg_out;
};

} // namespace

void conv2d_forward_gpu(const GpuTensor& X,
                        const GpuTensor& Wt,
                        const GpuTensor* bias,
                        int N, int C_in, int H, int W,
                        int C_out, int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        int dil_h, int dil_w,
                        int groups,
                        GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_forward_gpu: X must be FP16 or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward_gpu: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv2d_forward_gpu: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_forward_gpu: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_forward_gpu: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = bias ? 1u : 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_conv_fp16()
                                                               : pso_conv_fp32();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(Wt);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx; // dummy bind if no bias
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(Wt);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void conv2d_backward_input_gpu(const GpuTensor& Wt,
                               const GpuTensor& dY,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w,
                               int groups,
                               GpuTensor& dX) {
    if (Wt.dtype != Dtype::FP16 && Wt.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_backward_input_gpu: Wt must be FP16 or FP32");
    }
    if (dY.dtype != Wt.dtype) {
        throw std::runtime_error("conv2d_backward_input_gpu: dY dtype must match Wt");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_input_gpu: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_input_gpu: non-positive output shape");
    }
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(in_cols);
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    id<MTLComputePipelineState> pso = (Wt.dtype == Dtype::FP16)
        ? pso_conv_bwd_input_fp16() : pso_conv_bwd_input_fp32();
    id<MTLBuffer> bw  = buffer_for(Wt);
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdx = buffer_for(dX);
    const NSUInteger ow_  = buffer_offset_for(Wt);
    const NSUInteger ody  = buffer_offset_for(dY);
    const NSUInteger odx  = buffer_offset_for(dX);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bw  offset:ow_ atIndex:0];
        [enc setBuffer:bdy offset:ody atIndex:1];
        [enc setBuffer:bdx offset:odx atIndex:2];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void conv2d_backward_weight_gpu(const GpuTensor& X,
                                const GpuTensor& dY,
                                int N, int C_in, int H, int W,
                                int C_out, int kH, int kW,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dil_h, int dil_w,
                                int groups,
                                GpuTensor& dWt) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_backward_weight_gpu: X must be FP16 or FP32");
    }
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_backward_weight_gpu: X, dY, dWt dtype must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_backward_weight_gpu: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_backward_weight_gpu: non-positive output shape");
    }
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight_gpu: dWt shape mismatch");
    }
    const uint32_t total = static_cast<uint32_t>(C_out) * Cg_in * kH * kW;
    if (total == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    const bool is_fp16 = (X.dtype == Dtype::FP16);
    id<MTLComputePipelineState> pso = is_fp16
        ? pso_conv_bwd_weight_fp16() : pso_conv_bwd_weight_fp32_scratch();
    id<MTLBuffer> bx   = buffer_for(X);
    id<MTLBuffer> bdy  = buffer_for(dY);
    id<MTLBuffer> bdwt = buffer_for(dWt);
    const NSUInteger ox   = buffer_offset_for(X);
    const NSUInteger ody  = buffer_offset_for(dY);
    const NSUInteger odwt = buffer_offset_for(dWt);

    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:total * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx   offset:ox   atIndex:0];
        [enc setBuffer:bdy  offset:ody  atIndex:1];
        [enc setBuffer:scratch offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:3];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        // Fold scratch into dWt.
        id<MTLComputePipelineState> add_pso = is_fp16
            ? pso_conv_add_fp16() : pso_conv_add_fp32();
        [enc setComputePipelineState:add_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:bdwt offset:odwt atIndex:1];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg2 = [add_pso maxTotalThreadsPerThreadgroup];
        if (tg2 > 256) tg2 = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void conv2d_backward_bias_gpu(const GpuTensor& dY,
                              int N, int C_out, int H_out, int W_out,
                              GpuTensor& dB) {
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_backward_bias_gpu: dY must be FP16 or FP32");
    }
    if (dB.dtype != dY.dtype) {
        throw std::runtime_error("conv2d_backward_bias_gpu: dB dtype must match dY");
    }
    if (dB.rows != C_out || dB.cols != 1) {
        throw std::runtime_error("conv2d_backward_bias_gpu: dB shape mismatch");
    }
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    ConvParams p{};
    p.N = N; p.C_in = 0; p.H = 0; p.W = 0;
    p.C_out = C_out; p.kH = 0; p.kW = 0;
    p.H_out = H_out; p.W_out = W_out;
    p.stride_h = 0; p.stride_w = 0;
    p.pad_h = 0; p.pad_w = 0;
    p.dil_h = 0; p.dil_w = 0;
    p.has_bias = 0u;
    p.total = 0u;

    const bool is_fp16 = (dY.dtype == Dtype::FP16);
    id<MTLComputePipelineState> pso = is_fp16
        ? pso_conv_bwd_bias_fp16() : pso_conv_bwd_bias_fp32_scratch();
    id<MTLBuffer> bdy = buffer_for(dY);
    id<MTLBuffer> bdb = buffer_for(dB);
    const NSUInteger ody = buffer_offset_for(dY);
    const NSUInteger odb = buffer_offset_for(dB);

    @autoreleasepool {
        id<MTLBuffer> scratch = [metal_impl::device()
            newBufferWithLength:C_out * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bdy offset:ody atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        // One threadgroup per c_out; tg threads per group.
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(C_out), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

        id<MTLComputePipelineState> add_pso = is_fp16
            ? pso_conv_add_fp16() : pso_conv_add_fp32();
        const uint32_t Cn = static_cast<uint32_t>(C_out);
        [enc setComputePipelineState:add_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:bdb offset:odb atIndex:1];
        [enc setBytes:&Cn length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg2 = [add_pso maxTotalThreadsPerThreadgroup];
        if (tg2 > 256) tg2 = 256;
        [enc dispatchThreads:MTLSizeMake(Cn, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brotensor
