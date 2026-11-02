#pragma once

#import <Foundation/Foundation.h>

namespace brotensor::detail::metal {
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

// Fold FP32 scratch into FP16/FP32/BF16 destination (add).
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
kernel void k_conv2d_add_fp32_into_bf16(device const float* src [[buffer(0)]],
                                        device bfloat*      dst [[buffer(1)]],
                                        constant uint& n        [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[i] = bfloat(float(dst[i]) + src[i]);
}

// Split-K fold: dst[i] += sum_s part[s * n + i] (FP32 partials, one rounding).
#define SPLITK_FOLD(NAME, T)                                                  \
kernel void NAME(device const float* part [[buffer(0)]],                      \
                 device T*           dst  [[buffer(1)]],                      \
                 constant uint& n         [[buffer(2)]],                      \
                 constant uint& S         [[buffer(3)]],                      \
                 uint i [[thread_position_in_grid]]) {                        \
    if (i >= n) return;                                                       \
    float acc = 0.0f;                                                         \
    for (uint s = 0; s < S; ++s) acc += part[(ulong)s * n + i];               \
    dst[i] = T(float(dst[i]) + acc);                                          \
}
SPLITK_FOLD(k_conv2d_splitk_fold_fp32, float)
SPLITK_FOLD(k_conv2d_splitk_fold_fp16, half)
SPLITK_FOLD(k_conv2d_splitk_fold_bf16, bfloat)

// ─── BF16 variants (verbatim copies of FP16 kernels with half→bfloat) ───────

// One thread per output element. Direct conv, FP32 accumulator, BF16 IO.
kernel void k_conv2d_forward_bf16(device const bfloat* X    [[buffer(0)]],
                                  device const bfloat* Wt   [[buffer(1)]],
                                  device const bfloat* bias [[buffer(2)]],
                                  device bfloat*       Y    [[buffer(3)]],
                                  constant ConvParams& p    [[buffer(4)]],
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
    Y[idx] = bfloat(acc);
}

kernel void k_conv2d_backward_input_bf16(device const bfloat* Wt [[buffer(0)]],
                                         device const bfloat* dY [[buffer(1)]],
                                         device bfloat*       dX [[buffer(2)]],
                                         constant ConvParams& p  [[buffer(3)]],
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
    dX[idx] = bfloat(acc);
}

kernel void k_conv2d_backward_weight_bf16(device const bfloat* X   [[buffer(0)]],
                                          device const bfloat* dY  [[buffer(1)]],
                                          device float*        dWt_scratch [[buffer(2)]],
                                          constant ConvParams& p   [[buffer(3)]],
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

kernel void k_conv2d_backward_bias_bf16(device const bfloat* dY [[buffer(0)]],
                                        device float*        dB_scratch [[buffer(1)]],
                                        constant ConvParams& p  [[buffer(2)]],
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

} // namespace
} // namespace brotensor::detail::metal
