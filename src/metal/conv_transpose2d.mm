// ─── Metal 2D transposed convolution ────────────────────────────────────────
//
// Metal counterpart of src/cpu/conv_transpose2d.cpp. FP32-only on this
// backend. Direct conv-transpose with no im2col / matmul reformulation —
// each kernel is a per-output (or per-input / per-weight / per-channel)
// thread that gathers what it needs.
//
//   conv_transpose2d_forward          — Y OVERWRITTEN. Scatter inverted to a
//                                       gather: per output (n, oc, ho, wo),
//                                       sum contributions from every (c_in,
//                                       h, w, kh, kw) that scatters into it.
//   conv_transpose2d_backward_input   — dX OVERWRITTEN. Adjoint is a plain
//                                       gather conv (same as conv2d_forward
//                                       on dY with the weight indexing flipped).
//   conv_transpose2d_backward_weight  — dWt ACCUMULATES (+=).
//   conv_transpose2d_backward_bias    — dB  ACCUMULATES (+=).
//
// Memory layout — NCHW for X/Y; weights are (C_in, (C_out/groups)*kH*kW)
// input-channel-major (same as conv_transpose2d.cpp).

#include <brotensor/runtime.h>

#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32");
    }
}

void check_groups(const char* op, int C_in, int C_out, int groups) {
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        fail(op, "groups must be >=1 and divide both C_in and C_out");
    }
}

void check_geometry(const char* op, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int output_padding_h, int output_padding_w,
                    int dil_h, int dil_w) {
    if (kH < 1 || kW < 1 || stride_h < 1 || stride_w < 1
        || dil_h < 1 || dil_w < 1 || pad_h < 0 || pad_w < 0
        || output_padding_h < 0 || output_padding_w < 0) {
        fail(op, "kH/kW/stride/dilation >=1 and pad/output_padding >=0");
    }
    if (output_padding_h >= stride_h && output_padding_h >= dil_h) {
        fail(op, "output_padding_h must be < stride_h or < dil_h");
    }
    if (output_padding_w >= stride_w && output_padding_w >= dil_w) {
        fail(op, "output_padding_w must be < stride_w or < dil_w");
    }
}

inline int convt2d_out(int L, int stride, int padding, int output_padding,
                       int dilation, int kL) {
    return (L - 1) * stride - 2 * padding + dilation * (kL - 1)
           + output_padding + 1;
}

struct CT2dParams {
    uint32_t N, C_in, H, W, C_out, kH, kW, H_out, W_out;
    int32_t  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
    uint32_t Cg_in, Cg_out;
    uint32_t has_bias;
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct CT2dParams {
    uint N, C_in, H, W, C_out, kH, kW, H_out, W_out;
    int  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
    uint Cg_in, Cg_out;
    uint has_bias;
    uint total;
};

// ── forward: one thread per output element (n, oc, ho, wo) ──────────────────
// Input (n, c_in, h, w) reaches output ho = h*stride - pad + kh*dil, so
//   h = (ho + pad - kh*dil) / stride  must be a non-negative integer in
//   range. Same for w.
kernel void k_ct2d_forward(device const float* X    [[buffer(0)]],
                           device const float* Wt   [[buffer(1)]],
                           device const float* bias [[buffer(2)]],
                           device float*       Y    [[buffer(3)]],
                           constant CT2dParams& P   [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint wo  = gid % P.W_out;
    uint t1  = gid / P.W_out;
    uint ho  = t1 % P.H_out;
    uint t2  = t1 / P.H_out;
    uint oc  = t2 % P.C_out;
    uint n   = t2 / P.C_out;
    uint g        = oc / P.Cg_out;
    uint oc_local = oc - g * P.Cg_out;
    uint ic_base  = g * P.Cg_in;
    float acc = (P.has_bias != 0u) ? bias[oc] : 0.0f;
    for (uint ci_local = 0u; ci_local < P.Cg_in; ++ci_local) {
        uint c_in = ic_base + ci_local;
        for (uint kh = 0u; kh < P.kH; ++kh) {
            int num_h = int(ho) + P.pad_h - int(kh) * P.dil_h;
            if (num_h < 0 || (num_h % P.stride_h) != 0) continue;
            int h = num_h / P.stride_h;
            if (h >= int(P.H)) continue;
            for (uint kw = 0u; kw < P.kW; ++kw) {
                int num_w = int(wo) + P.pad_w - int(kw) * P.dil_w;
                if (num_w < 0 || (num_w % P.stride_w) != 0) continue;
                int w = num_w / P.stride_w;
                if (w >= int(P.W)) continue;
                uint w_idx = (c_in * P.Cg_out + oc_local) * (P.kH * P.kW)
                           + kh * P.kW + kw;
                uint x_idx = (n * P.C_in + c_in) * P.H * P.W
                           + uint(h) * P.W + uint(w);
                acc += X[x_idx] * Wt[w_idx];
            }
        }
    }
    Y[gid] = acc;
}

// ── backward_input: one thread per input element (n, c_in, h, w) ────────────
// Plain gather conv (the adjoint of the forward scatter).
kernel void k_ct2d_backward_input(device const float* Wt  [[buffer(0)]],
                                  device const float* dY  [[buffer(1)]],
                                  device float*       dX  [[buffer(2)]],
                                  constant CT2dParams& P  [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint w   = gid % P.W;
    uint t1  = gid / P.W;
    uint h   = t1 % P.H;
    uint t2  = t1 / P.H;
    uint c_in = t2 % P.C_in;
    uint n    = t2 / P.C_in;
    uint g       = c_in / P.Cg_in;
    uint oc_base = g * P.Cg_out;
    int ho_origin = int(h) * P.stride_h - P.pad_h;
    int wo_origin = int(w) * P.stride_w - P.pad_w;
    float acc = 0.0f;
    for (uint kh = 0u; kh < P.kH; ++kh) {
        int ho = ho_origin + int(kh) * P.dil_h;
        if (ho < 0 || ho >= int(P.H_out)) continue;
        for (uint kw = 0u; kw < P.kW; ++kw) {
            int wo = wo_origin + int(kw) * P.dil_w;
            if (wo < 0 || wo >= int(P.W_out)) continue;
            for (uint oc_local = 0u; oc_local < P.Cg_out; ++oc_local) {
                uint oc = oc_base + oc_local;
                uint w_idx = (c_in * P.Cg_out + oc_local) * (P.kH * P.kW)
                           + kh * P.kW + kw;
                uint dy_idx = (n * P.C_out + oc) * P.H_out * P.W_out
                            + uint(ho) * P.W_out + uint(wo);
                acc += dY[dy_idx] * Wt[w_idx];
            }
        }
    }
    dX[gid] = acc;
}

// ── backward_weight: one thread per weight element ──────────────────────────
// gid encodes (c_in, oc_local, kh, kw). dWt ACCUMULATES (+=).
kernel void k_ct2d_backward_weight(device const float* X    [[buffer(0)]],
                                   device const float* dY   [[buffer(1)]],
                                   device float*       dWt  [[buffer(2)]],
                                   constant CT2dParams& P   [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint kHW = P.kH * P.kW;
    uint kw       = gid % P.kW;
    uint t1       = gid / P.kW;
    uint kh       = t1 % P.kH;
    uint t2       = t1 / P.kH;
    uint oc_local = t2 % P.Cg_out;
    uint c_in     = t2 / P.Cg_out;
    uint g  = c_in / P.Cg_in;
    uint oc = g * P.Cg_out + oc_local;
    float acc = 0.0f;
    for (uint n = 0u; n < P.N; ++n) {
        uint x_base  = (n * P.C_in  + c_in) * P.H * P.W;
        uint dy_base = (n * P.C_out + oc)   * P.H_out * P.W_out;
        for (uint h = 0u; h < P.H; ++h) {
            int ho = int(h) * P.stride_h - P.pad_h + int(kh) * P.dil_h;
            if (ho < 0 || ho >= int(P.H_out)) continue;
            for (uint w = 0u; w < P.W; ++w) {
                int wo = int(w) * P.stride_w - P.pad_w + int(kw) * P.dil_w;
                if (wo < 0 || wo >= int(P.W_out)) continue;
                acc += X[x_base + h * P.W + w]
                     * dY[dy_base + uint(ho) * P.W_out + uint(wo)];
            }
        }
    }
    dWt[(c_in * P.Cg_out + oc_local) * kHW + kh * P.kW + kw] += acc;
}

// ── backward_bias: one thread per output channel. dB ACCUMULATES (+=) ──────
kernel void k_ct2d_backward_bias(device const float* dY  [[buffer(0)]],
                                 device float*       dB  [[buffer(1)]],
                                 constant CT2dParams& P  [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= P.C_out) return;
    float acc = 0.0f;
    for (uint n = 0u; n < P.N; ++n) {
        uint base = (n * P.C_out + gid) * P.H_out * P.W_out;
        for (uint i = 0u; i < P.H_out * P.W_out; ++i) acc += dY[base + i];
    }
    dB[gid] += acc;
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                            \
    }
DEF_PSO(pso_fwd,           @"k_ct2d_forward")
DEF_PSO(pso_bwd_input,     @"k_ct2d_backward_input")
DEF_PSO(pso_bwd_weight,    @"k_ct2d_backward_weight")
DEF_PSO(pso_bwd_bias,      @"k_ct2d_backward_bias")
#undef DEF_PSO

void dispatch1d(id<MTLComputePipelineState> pso, NSUInteger total,
                void (^binders)(id<MTLComputeCommandEncoder>)) {
    if (total == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        binders(enc);
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

CT2dParams make_params(int N, int C_in, int H, int W, int C_out,
                       int kH, int kW, int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int output_padding_h, int output_padding_w,
                       int dil_h, int dil_w, int groups, bool has_bias) {
    CT2dParams p{};
    p.N = N; p.C_in = C_in; p.H = H; p.W = W;
    p.C_out = C_out; p.kH = kH; p.kW = kW;
    p.H_out = convt2d_out(H, stride_h, pad_h, output_padding_h, dil_h, kH);
    p.W_out = convt2d_out(W, stride_w, pad_w, output_padding_w, dil_w, kW);
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.dil_h = dil_h; p.dil_w = dil_w;
    p.Cg_in = C_in / groups;
    p.Cg_out = C_out / groups;
    p.has_bias = has_bias ? 1u : 0u;
    return p;
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_forward
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_forward(const Tensor& X, const Tensor& Wt,
                              const Tensor* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int output_padding_h, int output_padding_w,
                              int dil_h, int dil_w, int groups,
                              Tensor& Y) {
    const char* op = "conv_transpose2d_forward";
    req_fp32(op, X, "X");
    req_fp32(op, Wt, "Wt");
    if (bias) req_fp32(op, *bias, "bias");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    auto p = make_params(N, C_in, H, W, C_out, kH, kW,
                         stride_h, stride_w, pad_h, pad_w,
                         output_padding_h, output_padding_w,
                         dil_h, dil_w, groups, bias != nullptr);
    if (int(p.H_out) <= 0 || int(p.W_out) <= 0)
        fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != int(p.Cg_out) * kHW)
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    if (X.rows != N || X.cols != C_in * H * W)
        fail(op, "X shape must be (N, C_in*H*W)");
    if (bias && (bias->rows != C_out || bias->cols != 1))
        fail(op, "bias shape must be (C_out, 1)");

    const int out_cols = C_out * int(p.H_out) * int(p.W_out);
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, out_cols, Dtype::FP32);
    }
    if (N == 0 || out_cols == 0) return;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(out_cols);

    @autoreleasepool {
        id<MTLBuffer> bX = buffer_for(X);
        NSUInteger oX = buffer_offset_for(X);
        id<MTLBuffer> bW = buffer_for(Wt);
        NSUInteger oW = buffer_offset_for(Wt);
        id<MTLBuffer> bY = buffer_for(Y);
        NSUInteger oY = buffer_offset_for(Y);

        // If no bias, set a dummy buffer (any valid buffer; not read).
        id<MTLBuffer> bB;
        NSUInteger oB;
        if (bias) {
            bB = buffer_for(*bias);
            oB = buffer_offset_for(*bias);
        } else {
            bB = bX;
            oB = oX;
        }

        dispatch1d(pso_fwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX offset:oX atIndex:0];
            [enc setBuffer:bW offset:oW atIndex:1];
            [enc setBuffer:bB offset:oB atIndex:2];
            [enc setBuffer:bY offset:oY atIndex:3];
            [enc setBytes:&p length:sizeof(CT2dParams) atIndex:4];
        });
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_backward_input
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_backward_input(const Tensor& Wt, const Tensor& dY,
                                     int N, int C_in, int H, int W,
                                     int C_out, int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int output_padding_h, int output_padding_w,
                                     int dil_h, int dil_w, int groups,
                                     Tensor& dX) {
    const char* op = "conv_transpose2d_backward_input";
    req_fp32(op, Wt, "Wt");
    req_fp32(op, dY, "dY");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    auto p = make_params(N, C_in, H, W, C_out, kH, kW,
                         stride_h, stride_w, pad_h, pad_w,
                         output_padding_h, output_padding_w,
                         dil_h, dil_w, groups, false);
    if (int(p.H_out) <= 0 || int(p.W_out) <= 0)
        fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (Wt.rows != C_in || Wt.cols != int(p.Cg_out) * kHW)
        fail(op, "Wt shape must be (C_in, (C_out/groups)*kH*kW)");
    if (dY.rows != N || dY.cols != C_out * int(p.H_out) * int(p.W_out))
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");

    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, in_cols, Dtype::FP32);
    }
    if (N == 0 || in_cols == 0) return;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(in_cols);

    dispatch1d(pso_bwd_input(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(Wt) offset:buffer_offset_for(Wt) atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:2];
        [enc setBytes:&p length:sizeof(CT2dParams) atIndex:3];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_backward_weight (dWt accumulates)
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_backward_weight(const Tensor& X, const Tensor& dY,
                                      int N, int C_in, int H, int W,
                                      int C_out, int kH, int kW,
                                      int stride_h, int stride_w,
                                      int pad_h, int pad_w,
                                      int output_padding_h, int output_padding_w,
                                      int dil_h, int dil_w, int groups,
                                      Tensor& dWt) {
    const char* op = "conv_transpose2d_backward_weight";
    req_fp32(op, X, "X");
    req_fp32(op, dY, "dY");
    req_fp32(op, dWt, "dWt");
    check_groups(op, C_in, C_out, groups);
    check_geometry(op, kH, kW, stride_h, stride_w, pad_h, pad_w,
                   output_padding_h, output_padding_w, dil_h, dil_w);

    auto p = make_params(N, C_in, H, W, C_out, kH, kW,
                         stride_h, stride_w, pad_h, pad_w,
                         output_padding_h, output_padding_w,
                         dil_h, dil_w, groups, false);
    if (int(p.H_out) <= 0 || int(p.W_out) <= 0)
        fail(op, "non-positive output spatial size");
    const int kHW = kH * kW;
    if (dWt.rows != C_in || dWt.cols != int(p.Cg_out) * kHW)
        fail(op, "dWt shape must be (C_in, (C_out/groups)*kH*kW)");
    if (X.rows != N || X.cols != C_in * H * W)
        fail(op, "X shape must be (N, C_in*H*W)");
    if (dY.rows != N || dY.cols != C_out * int(p.H_out) * int(p.W_out))
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    if (C_in == 0 || p.Cg_out == 0 || kHW == 0) return;

    p.total = static_cast<uint32_t>(C_in) * p.Cg_out * static_cast<uint32_t>(kHW);

    dispatch1d(pso_bwd_weight(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)   offset:buffer_offset_for(X)   atIndex:0];
        [enc setBuffer:buffer_for(dY)  offset:buffer_offset_for(dY)  atIndex:1];
        [enc setBuffer:buffer_for(dWt) offset:buffer_offset_for(dWt) atIndex:2];
        [enc setBytes:&p length:sizeof(CT2dParams) atIndex:3];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  conv_transpose2d_backward_bias (dB accumulates)
// ════════════════════════════════════════════════════════════════════════════
void conv_transpose2d_backward_bias(const Tensor& dY,
                                    int N, int C_out, int H_out, int W_out,
                                    Tensor& dB) {
    const char* op = "conv_transpose2d_backward_bias";
    req_fp32(op, dY, "dY");
    req_fp32(op, dB, "dB");
    if (dB.rows != C_out || dB.cols != 1)
        fail(op, "dB shape must be (C_out, 1)");
    if (dY.rows != N || dY.cols != C_out * H_out * W_out)
        fail(op, "dY shape must be (N, C_out*H_out*W_out)");
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    CT2dParams p{};
    p.N = N;
    p.C_out = C_out;
    p.H_out = H_out;
    p.W_out = W_out;

    dispatch1d(pso_bwd_bias(), static_cast<NSUInteger>(C_out),
               ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dB) offset:buffer_offset_for(dB) atIndex:1];
        [enc setBytes:&p length:sizeof(CT2dParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
