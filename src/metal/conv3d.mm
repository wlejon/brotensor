// Metal conv3d_forward + conv3d_int8w_fp16_forward (Qwen3-VL patch embed).
//
// Mirrors src/metal/conv2d.mm's naive direct-conv kernel with an added T
// (depth) axis. No simdgroup-matrix fast path — the patch embedder is small.
//
// Layouts (match the CPU/CUDA ports): NCTHW activations + OICTHW (grouped)
// filter. FP32 accumulator; storage half / bfloat / float for the generic
// forward, and FP16 / INT8 / FP32 for the W8A16 path. Y is OVERWRITTEN.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct Conv3dParams {
    uint N, C_in, T_in, H, W;
    uint C_out, kT, kH, kW;
    uint T_out, H_out, W_out;
    int  stride_t, stride_h, stride_w;
    int  pad_t, pad_h, pad_w;
    int  dil_t, dil_h, dil_w;
    uint has_bias;
    uint total;
    uint groups;
    uint Cg_in;
    uint Cg_out;
};

// One thread per output element. Direct conv, FP32 accumulator, FP16 IO.
kernel void k_conv3d_forward_fp16(device const half* X    [[buffer(0)]],
                                  device const half* Wt   [[buffer(1)]],
                                  device const half* bias [[buffer(2)]],
                                  device half*       Y    [[buffer(3)]],
                                  constant Conv3dParams& p [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint ot = t % p.T_out;
    t /= p.T_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_t_origin = int(ot) * p.stride_t - p.pad_t;
    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float acc = 0.0f;
    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kT * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.T_in * p.H * p.W;

    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kT * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.T_in * p.H * p.W;
        for (uint kt = 0; kt < p.kT; ++kt) {
            int in_t = in_t_origin + int(kt) * p.dil_t;
            if (in_t < 0 || in_t >= int(p.T_in)) continue;
            for (uint kh = 0; kh < p.kH; ++kh) {
                int in_h = in_h_origin + int(kh) * p.dil_h;
                if (in_h < 0 || in_h >= int(p.H)) continue;
                for (uint kw = 0; kw < p.kW; ++kw) {
                    int in_w = in_w_origin + int(kw) * p.dil_w;
                    if (in_w < 0 || in_w >= int(p.W)) continue;
                    uint x_off = x_ic_base +
                                 ((uint(in_t) * p.H + uint(in_h)) * p.W) +
                                 uint(in_w);
                    uint w_off = w_ic_base +
                                 ((kt * p.kH + kh) * p.kW) + kw;
                    float x_v = float(X[x_off]);
                    float w_v = float(Wt[w_off]);
                    acc += x_v * w_v;
                }
            }
        }
    }
    if (p.has_bias != 0u) acc += float(bias[oc]);
    Y[idx] = half(acc);
}

kernel void k_conv3d_forward_fp32(device const float* X    [[buffer(0)]],
                                  device const float* Wt   [[buffer(1)]],
                                  device const float* bias [[buffer(2)]],
                                  device float*       Y    [[buffer(3)]],
                                  constant Conv3dParams& p [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint ot = t % p.T_out;
    t /= p.T_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_t_origin = int(ot) * p.stride_t - p.pad_t;
    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float acc = 0.0f;
    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kT * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.T_in * p.H * p.W;

    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kT * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.T_in * p.H * p.W;
        for (uint kt = 0; kt < p.kT; ++kt) {
            int in_t = in_t_origin + int(kt) * p.dil_t;
            if (in_t < 0 || in_t >= int(p.T_in)) continue;
            for (uint kh = 0; kh < p.kH; ++kh) {
                int in_h = in_h_origin + int(kh) * p.dil_h;
                if (in_h < 0 || in_h >= int(p.H)) continue;
                for (uint kw = 0; kw < p.kW; ++kw) {
                    int in_w = in_w_origin + int(kw) * p.dil_w;
                    if (in_w < 0 || in_w >= int(p.W)) continue;
                    uint x_off = x_ic_base +
                                 ((uint(in_t) * p.H + uint(in_h)) * p.W) +
                                 uint(in_w);
                    uint w_off = w_ic_base +
                                 ((kt * p.kH + kh) * p.kW) + kw;
                    acc += X[x_off] * Wt[w_off];
                }
            }
        }
    }
    if (p.has_bias != 0u) acc += bias[oc];
    Y[idx] = acc;
}

kernel void k_conv3d_forward_bf16(device const bfloat* X    [[buffer(0)]],
                                  device const bfloat* Wt   [[buffer(1)]],
                                  device const bfloat* bias [[buffer(2)]],
                                  device bfloat*       Y    [[buffer(3)]],
                                  constant Conv3dParams& p  [[buffer(4)]],
                                  uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint ot = t % p.T_out;
    t /= p.T_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_t_origin = int(ot) * p.stride_t - p.pad_t;
    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float acc = 0.0f;
    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kT * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.T_in * p.H * p.W;

    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kT * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.T_in * p.H * p.W;
        for (uint kt = 0; kt < p.kT; ++kt) {
            int in_t = in_t_origin + int(kt) * p.dil_t;
            if (in_t < 0 || in_t >= int(p.T_in)) continue;
            for (uint kh = 0; kh < p.kH; ++kh) {
                int in_h = in_h_origin + int(kh) * p.dil_h;
                if (in_h < 0 || in_h >= int(p.H)) continue;
                for (uint kw = 0; kw < p.kW; ++kw) {
                    int in_w = in_w_origin + int(kw) * p.dil_w;
                    if (in_w < 0 || in_w >= int(p.W)) continue;
                    uint x_off = x_ic_base +
                                 ((uint(in_t) * p.H + uint(in_h)) * p.W) +
                                 uint(in_w);
                    uint w_off = w_ic_base +
                                 ((kt * p.kH + kh) * p.kW) + kw;
                    acc += float(X[x_off]) * float(Wt[w_off]);
                }
            }
        }
    }
    if (p.has_bias != 0u) acc += float(bias[oc]);
    Y[idx] = bfloat(acc);
}

struct Conv3dI8Params {
    uint N, C_in, T_in, H, W;
    uint C_out, kT, kH, kW;
    uint T_out, H_out, W_out;
    int  stride_t, stride_h, stride_w;
    int  pad_t, pad_h, pad_w;
    int  dil_t, dil_h, dil_w;
    uint has_bias;
    uint total;
    uint groups;
    uint Cg_in;
    uint Cg_out;
};

// W8A16: X / bias FP16, W INT8 with per-c_out FP32 scale.
kernel void k_conv3d_int8w_fp16_forward(device const half*  X      [[buffer(0)]],
                                        device const char*  W      [[buffer(1)]],
                                        device const float* scales [[buffer(2)]],
                                        device const half*  bias   [[buffer(3)]],
                                        device half*        Y      [[buffer(4)]],
                                        constant Conv3dI8Params& p [[buffer(5)]],
                                        uint idx [[thread_position_in_grid]]) {
    if (idx >= p.total) return;
    uint ow = idx % p.W_out;
    uint t  = idx / p.W_out;
    uint oh = t % p.H_out;
    t /= p.H_out;
    uint ot = t % p.T_out;
    t /= p.T_out;
    uint oc = t % p.C_out;
    uint n  = t / p.C_out;

    int in_t_origin = int(ot) * p.stride_t - p.pad_t;
    int in_h_origin = int(oh) * p.stride_h - p.pad_h;
    int in_w_origin = int(ow) * p.stride_w - p.pad_w;

    float scale = scales[oc];

    uint g_out = oc / p.Cg_out;
    uint ic_abs_base = g_out * p.Cg_in;
    uint w_oc_base = oc * p.Cg_in * p.kT * p.kH * p.kW;
    uint x_n_base  = n * p.C_in * p.T_in * p.H * p.W;

    float acc = 0.0f;
    device const int8_t* W_i8 = (device const int8_t*)W;
    for (uint ic_local = 0; ic_local < p.Cg_in; ++ic_local) {
        uint ic = ic_abs_base + ic_local;
        uint w_ic_base = w_oc_base + ic_local * p.kT * p.kH * p.kW;
        uint x_ic_base = x_n_base  + ic * p.T_in * p.H * p.W;
        for (uint kt = 0; kt < p.kT; ++kt) {
            int in_t = in_t_origin + int(kt) * p.dil_t;
            if (in_t < 0 || in_t >= int(p.T_in)) continue;
            for (uint kh = 0; kh < p.kH; ++kh) {
                int in_h = in_h_origin + int(kh) * p.dil_h;
                if (in_h < 0 || in_h >= int(p.H)) continue;
                for (uint kw = 0; kw < p.kW; ++kw) {
                    int in_w = in_w_origin + int(kw) * p.dil_w;
                    if (in_w < 0 || in_w >= int(p.W)) continue;
                    uint x_off = x_ic_base +
                                 ((uint(in_t) * p.H + uint(in_h)) * p.W) +
                                 uint(in_w);
                    uint w_off = w_ic_base +
                                 ((kt * p.kH + kh) * p.kW) + kw;
                    float xv = float(X[x_off]);
                    float wv = float(W_i8[w_off]) * scale;
                    acc += xv * wv;
                }
            }
        }
    }
    if (p.has_bias != 0u) acc += float(bias[oc]);
    Y[idx] = half(acc);
}
)msl";

id<MTLComputePipelineState> pso_conv3d_fp16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv3d_forward_fp16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv3d_fp32() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv3d_forward_fp32"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv3d_bf16() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv3d_forward_bf16"); });
    return pso;
}
id<MTLComputePipelineState> pso_conv3d_int8w() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_conv3d_int8w_fp16_forward"); });
    return pso;
}

struct Conv3dParams {
    uint32_t N, C_in, T_in, H, W;
    uint32_t C_out, kT, kH, kW;
    uint32_t T_out, H_out, W_out;
    int32_t  stride_t, stride_h, stride_w;
    int32_t  pad_t, pad_h, pad_w;
    int32_t  dil_t, dil_h, dil_w;
    uint32_t has_bias;
    uint32_t total;
    uint32_t groups;
    uint32_t Cg_in;
    uint32_t Cg_out;
};

struct Conv3dI8Params {
    uint32_t N, C_in, T_in, H, W;
    uint32_t C_out, kT, kH, kW;
    uint32_t T_out, H_out, W_out;
    int32_t  stride_t, stride_h, stride_w;
    int32_t  pad_t, pad_h, pad_w;
    int32_t  dil_t, dil_h, dil_w;
    uint32_t has_bias;
    uint32_t total;
    uint32_t groups;
    uint32_t Cg_in;
    uint32_t Cg_out;
};

} // namespace

void conv3d_forward(const Tensor& X,
                    const Tensor& Wt,
                    const Tensor* bias,
                    int N, int C_in, int T_in, int H, int W,
                    int C_out, int kT, int kH, int kW,
                    int stride_t, int stride_h, int stride_w,
                    int pad_t, int pad_h, int pad_w,
                    int dil_t, int dil_h, int dil_w,
                    int groups,
                    Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("conv3d_forward: X must be FP16, BF16, or FP32");
    }
    if (Wt.dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: Wt dtype must match X");
    }
    if (bias && bias->dtype != X.dtype) {
        throw std::runtime_error("conv3d_forward: bias dtype must match X");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv3d_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);
    if (total == 0) return;

    Conv3dParams p{};
    p.N = N; p.C_in = C_in; p.T_in = T_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kT = kT; p.kH = kH; p.kW = kW;
    p.T_out = T_out; p.H_out = H_out; p.W_out = W_out;
    p.stride_t = stride_t; p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_t = pad_t; p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_t = dil_t; p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = bias ? 1u : 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(Wt);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx;
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(Wt);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_conv3d_fp16()
                                   : (X.dtype == Dtype::BF16) ? pso_conv3d_bf16()
                                                               : pso_conv3d_fp32();

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bb offset:ob atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&p length:sizeof(Conv3dParams) atIndex:4];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void conv3d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int T_in, int H, int W,
                               int C_out, int kT, int kH, int kW,
                               int stride_t, int stride_h, int stride_w,
                               int pad_t, int pad_h, int pad_w,
                               int dil_t, int dil_h, int dil_w, int groups,
                               Tensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv3d_int8w_fp16_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kT * kH * kW) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: scales shape mismatch");
    }
    const int T_out = (T_in + 2 * pad_t - dil_t * (kT - 1) - 1) / stride_t + 1;
    const int H_out = (H    + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W    + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv3d_int8w_fp16_forward: non-positive output shape");
    }
    const int out_cols = C_out * T_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);
    if (total == 0) return;

    Conv3dI8Params p{};
    p.N = N; p.C_in = C_in; p.T_in = T_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kT = kT; p.kH = kH; p.kW = kW;
    p.T_out = T_out; p.H_out = H_out; p.W_out = W_out;
    p.stride_t = stride_t; p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_t = pad_t; p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_t = dil_t; p.dil_h = dil_h; p.dil_w = dil_w;
    p.has_bias = bias ? 1u : 0u;
    p.total = total;
    p.groups = static_cast<uint32_t>(groups);
    p.Cg_in = static_cast<uint32_t>(Cg_in);
    p.Cg_out = static_cast<uint32_t>(Cg_out);

    id<MTLComputePipelineState> pso = pso_conv3d_int8w();
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> bw = buffer_for(W_int8);
    id<MTLBuffer> bs = buffer_for(scales);
    id<MTLBuffer> by = buffer_for(Y);
    id<MTLBuffer> bb = bias ? buffer_for(*bias) : bx;
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger ow_ = buffer_offset_for(W_int8);
    const NSUInteger os = buffer_offset_for(scales);
    const NSUInteger oy = buffer_offset_for(Y);
    const NSUInteger ob = bias ? buffer_offset_for(*bias) : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bw offset:ow_ atIndex:1];
        [enc setBuffer:bs offset:os atIndex:2];
        [enc setBuffer:bb offset:ob atIndex:3];
        [enc setBuffer:by offset:oy atIndex:4];
        [enc setBytes:&p length:sizeof(Conv3dI8Params) atIndex:5];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
