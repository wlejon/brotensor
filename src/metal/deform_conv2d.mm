// ─── Metal modulated deformable conv2d (torchvision deform_conv2d v2, fwd) ───
//
// Metal port of src/cuda/deform_conv2d.cu. One thread per output element; FP32
// accumulator, storage dtype T (FP16/FP32). Direct fused form of torchvision's
// deformable_im2col + GEMM: each kH×kW tap is bilinearly sampled from X at a
// per-tap, per-pixel offset location with ZERO padding (torchvision
// convention), optionally reweighted by the mask modulator, then reduced
// against the OIHW weight.

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

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct DCParams {
    int N, C_in, H, W, C_out, kH, kW, H_out, W_out;
    int stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
    int Cg_in, Cg_out, c_per_off_grp, deform_groups, total;
    uint has_mask, has_bias;
};

// torchvision bilinear_interpolate: zero outside [0,H)×[0,W), per-corner guard.
template <typename T>
inline float dbilinear(device const T* in, int H, int W, float h, float w) {
    if (h <= -1.0f || float(H) <= h || w <= -1.0f || float(W) <= w) return 0.0f;
    int h_low = int(floor(h));
    int w_low = int(floor(w));
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    float lh = h - h_low, lw = w - w_low;
    float hh = 1.0f - lh, hw = 1.0f - lw;
    float v1 = (h_low >= 0 && w_low >= 0)            ? float(in[h_low * W + w_low])   : 0.0f;
    float v2 = (h_low >= 0 && w_high <= W - 1)       ? float(in[h_low * W + w_high])  : 0.0f;
    float v3 = (h_high <= H - 1 && w_low >= 0)       ? float(in[h_high * W + w_low])  : 0.0f;
    float v4 = (h_high <= H - 1 && w_high <= W - 1)  ? float(in[h_high * W + w_high]) : 0.0f;
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

#define DEFORM_KERNEL(NAME, T)                                                \
kernel void NAME(device const T* X      [[buffer(0)]],                        \
                 device const T* offset [[buffer(1)]],                        \
                 device const T* mask   [[buffer(2)]],                        \
                 device const T* Wt     [[buffer(3)]],                        \
                 device const T* bias   [[buffer(4)]],                        \
                 device T*       Y      [[buffer(5)]],                        \
                 constant DCParams& p   [[buffer(6)]],                        \
                 uint gidx [[thread_position_in_grid]]) {                     \
    if (int(gidx) >= p.total) return;                                         \
    int idx = int(gidx);                                                      \
    int ow = idx % p.W_out;                                                   \
    int t = idx / p.W_out;                                                    \
    int oh = t % p.H_out;                                                     \
    t /= p.H_out;                                                             \
    int oc = t % p.C_out;                                                     \
    int n  = t / p.C_out;                                                     \
    int in_h_origin = oh * p.stride_h - p.pad_h;                              \
    int in_w_origin = ow * p.stride_w - p.pad_w;                              \
    int ksz = p.kH * p.kW;                                                    \
    int g_out = oc / p.Cg_out;                                                \
    int ic_abs_base = g_out * p.Cg_in;                                        \
    int w_oc_base = oc * p.Cg_in * ksz;                                       \
    device const T* off_n = offset + (ulong)n * p.deform_groups * 2 * ksz * p.H_out * p.W_out; \
    device const T* mask_n = p.has_mask != 0u                                 \
        ? mask + (ulong)n * p.deform_groups * ksz * p.H_out * p.W_out : nullptr; \
    float acc = p.has_bias != 0u ? float(bias[oc]) : 0.0f;                    \
    for (int ic_local = 0; ic_local < p.Cg_in; ++ic_local) {                  \
        int ic = ic_abs_base + ic_local;                                      \
        int off_grp = ic / p.c_per_off_grp;                                   \
        device const T* in_ch = X + ((ulong)n * p.C_in + ic) * p.H * p.W;     \
        device const T* off_grp_base = off_n + (ulong)off_grp * 2 * ksz * p.H_out * p.W_out; \
        device const T* mask_grp_base = mask_n                                \
            ? mask_n + (ulong)off_grp * ksz * p.H_out * p.W_out : nullptr;    \
        int w_ic_base = w_oc_base + ic_local * ksz;                           \
        for (int kh = 0; kh < p.kH; ++kh) {                                   \
            for (int kw = 0; kw < p.kW; ++kw) {                               \
                int tap = kh * p.kW + kw;                                     \
                float off_y = float(off_grp_base[((2 * tap) * p.H_out + oh) * p.W_out + ow]);     \
                float off_x = float(off_grp_base[((2 * tap + 1) * p.H_out + oh) * p.W_out + ow]); \
                float m = mask_grp_base                                       \
                    ? float(mask_grp_base[(tap * p.H_out + oh) * p.W_out + ow]) : 1.0f; \
                float yy = in_h_origin + kh * p.dil_h + off_y;                \
                float xx = in_w_origin + kw * p.dil_w + off_x;                \
                float val = dbilinear<T>(in_ch, p.H, p.W, yy, xx);            \
                acc += float(Wt[w_ic_base + tap]) * (m * val);                \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    Y[idx] = T(acc);                                                          \
}

DEFORM_KERNEL(k_deform_conv2d_fp32, float)
DEFORM_KERNEL(k_deform_conv2d_fp16, half)
)msl";

struct DCParams {
    int32_t N, C_in, H, W, C_out, kH, kW, H_out, W_out;
    int32_t stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
    int32_t Cg_in, Cg_out, c_per_off_grp, deform_groups, total;
    uint32_t has_mask, has_bias;
};

#define DEF_PSO(NAME, FN)                                                     \
    id<MTLComputePipelineState> NAME() {                                      \
        static dispatch_once_t once;                                          \
        static id<MTLComputePipelineState> pso;                               \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });          \
        return pso;                                                           \
    }
DEF_PSO(pso_fp32, @"k_deform_conv2d_fp32")
DEF_PSO(pso_fp16, @"k_deform_conv2d_fp16")
#undef DEF_PSO

} // namespace

void deform_conv2d_forward(const Tensor& X, const Tensor& offset,
                           const Tensor* mask, const Tensor& Wt,
                           const Tensor* bias,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w, int dil_h, int dil_w,
                           int groups, int deform_groups, Tensor& Y) {
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32)
        throw std::runtime_error("deform_conv2d_forward: X must be FP16 or FP32");
    if (Wt.dtype != X.dtype || offset.dtype != X.dtype ||
        (mask && mask->dtype != X.dtype) || (bias && bias->dtype != X.dtype))
        throw std::runtime_error(
            "deform_conv2d_forward: X, offset, mask, Wt, bias dtype must match");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0)
        throw std::runtime_error("deform_conv2d_forward: groups must divide C_in and C_out");
    if (deform_groups < 1 || C_in % deform_groups != 0)
        throw std::runtime_error("deform_conv2d_forward: deform_groups must divide C_in");
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int c_per_off_grp = C_in / deform_groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("deform_conv2d_forward: non-positive output shape");
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype)
        Y.resize(N, out_cols, X.dtype);
    const int total = N * out_cols;
    if (total == 0) return;

    DCParams p{N, C_in, H, W, C_out, kH, kW, H_out, W_out,
               stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
               Cg_in, Cg_out, c_per_off_grp, deform_groups, total,
               mask ? 1u : 0u, bias ? 1u : 0u};
    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_fp16() : pso_fp32();
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(X)      offset:buffer_offset_for(X)      atIndex:0];
        [enc setBuffer:buffer_for(offset) offset:buffer_offset_for(offset) atIndex:1];
        id<MTLBuffer> mb = mask ? buffer_for(*mask) : buffer_for(X);  // dummy bind
        NSUInteger mo = mask ? buffer_offset_for(*mask) : buffer_offset_for(X);
        [enc setBuffer:mb offset:mo atIndex:2];
        [enc setBuffer:buffer_for(Wt) offset:buffer_offset_for(Wt) atIndex:3];
        id<MTLBuffer> bb = bias ? buffer_for(*bias) : buffer_for(X);  // dummy bind
        NSUInteger bo = bias ? buffer_offset_for(*bias) : buffer_offset_for(X);
        [enc setBuffer:bb offset:bo atIndex:4];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:5];
        [enc setBytes:&p length:sizeof(DCParams) atIndex:6];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
