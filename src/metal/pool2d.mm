// ─── Metal 2D pooling: adaptive_avg_pool2d + max_pool2d ────────────────────
//
// Metal counterpart of src/cpu/pool2d.cpp. FP32-only.
//
//   adaptive_avg_pool2d_forward  — Y OVERWRITTEN.
//   adaptive_avg_pool2d_backward — dX OVERWRITTEN. Adjoint = scatter the
//        per-output gradient evenly across its input window. Overlapping
//        windows on the same input pixel sum; CPU uses a sequential
//        zero-then-scatter for that. Metal uses a gather adjoint: per input
//        pixel, iterate the (oh, ow) outputs whose window contains it and
//        sum dY[oh, ow] / area. No atomics.
//   max_pool2d_forward           — Y + Idx OVERWRITTEN.
//   max_pool2d_backward          — dX OVERWRITTEN. Adjoint = scatter dY onto
//        the index recorded in Idx. Overlapping kernels can write the same
//        input pixel from multiple outputs — gather adjoint sums those.

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

// ── Adaptive avg pool ──────────────────────────────────────────────────────

struct AAPParams {
    uint32_t N, C, H, W, H_out, W_out;
    uint32_t total;
};

NSString* const kAAP = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct AAPParams { uint N, C, H, W, H_out, W_out, total; };

static inline void adapt(int o, int L, int L_out, thread int& start, thread int& end) {
    start = (o * L) / L_out;
    end   = ((o + 1) * L + L_out - 1) / L_out;
    if (end > L) end = L;
    if (start < 0) start = 0;
}

kernel void k_aap_forward(device const float* X [[buffer(0)]],
                          device float*       Y [[buffer(1)]],
                          constant AAPParams& P [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint ow = gid % P.W_out;
    uint t1 = gid / P.W_out;
    uint oh = t1 % P.H_out;
    uint t2 = t1 / P.H_out;
    uint c  = t2 % P.C;
    uint n  = t2 / P.C;
    int h0, h1, w0, w1;
    adapt(int(oh), int(P.H), int(P.H_out), h0, h1);
    adapt(int(ow), int(P.W), int(P.W_out), w0, w1);
    int area = (h1 - h0) * (w1 - w0);
    uint xbase = (n * P.C + c) * P.H * P.W;
    float acc = 0.0f;
    for (int h = h0; h < h1; ++h) {
        for (int w = w0; w < w1; ++w) {
            acc += X[xbase + uint(h) * P.W + uint(w)];
        }
    }
    Y[gid] = acc / float(area);
}

// backward: one thread per input pixel — gather adjoint.
kernel void k_aap_backward(device const float* dY [[buffer(0)]],
                           device float*       dX [[buffer(1)]],
                           constant AAPParams& P  [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    // Total here is the input total (caller sets P.total = N*C*H*W).
    if (gid >= P.total) return;
    uint w = gid % P.W;
    uint t1 = gid / P.W;
    uint h = t1 % P.H;
    uint t2 = t1 / P.H;
    uint c = t2 % P.C;
    uint n = t2 / P.C;

    // For each output (oh, ow), the input window covers
    //   start_h = floor(oh * H / H_out), end_h = ceil((oh+1)*H / H_out).
    // An input pixel `h` is included iff start_h <= h < end_h, which solves to
    //   oh_lo = floor(h * H_out / H)              (smallest oh with end_h > h)
    //   oh_hi = floor((h+1) * H_out / H)          (smallest oh with start_h > h)
    // and outputs oh in [oh_lo, oh_hi) include this pixel. Same for W.
    int oh_lo = int((h * P.H_out) / P.H);
    int oh_hi = int(((h + 1) * P.H_out + P.H - 1) / P.H);
    if (oh_hi > int(P.H_out)) oh_hi = int(P.H_out);
    int ow_lo = int((w * P.W_out) / P.W);
    int ow_hi = int(((w + 1) * P.W_out + P.W - 1) / P.W);
    if (ow_hi > int(P.W_out)) ow_hi = int(P.W_out);

    uint ybase = (n * P.C + c) * P.H_out * P.W_out;
    float acc = 0.0f;
    for (int oh = oh_lo; oh < oh_hi; ++oh) {
        int h0, h1, w0, w1;
        adapt(oh, int(P.H), int(P.H_out), h0, h1);
        if (int(h) < h0 || int(h) >= h1) continue;
        for (int ow = ow_lo; ow < ow_hi; ++ow) {
            adapt(ow, int(P.W), int(P.W_out), w0, w1);
            if (int(w) < w0 || int(w) >= w1) continue;
            int area = (h1 - h0) * (w1 - w0);
            acc += dY[ybase + uint(oh) * P.W_out + uint(ow)] / float(area);
        }
    }
    dX[gid] = acc;
}
)msl";

// ── Max pool ───────────────────────────────────────────────────────────────

struct MPParams {
    uint32_t N, C, H, W, H_out, W_out;
    int32_t  kH, kW, stride_h, stride_w, pad_h, pad_w;
    uint32_t total;
};

NSString* const kMP = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct MPParams {
    uint N, C, H, W, H_out, W_out;
    int kH, kW, stride_h, stride_w, pad_h, pad_w;
    uint total;
};

kernel void k_mp_forward(device const float* X   [[buffer(0)]],
                         device float*       Y   [[buffer(1)]],
                         device int*         Idx [[buffer(2)]],
                         constant MPParams&  P   [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint ow = gid % P.W_out;
    uint t1 = gid / P.W_out;
    uint oh = t1 % P.H_out;
    uint t2 = t1 / P.H_out;
    uint c  = t2 % P.C;
    uint n  = t2 / P.C;

    int h_base = int(oh) * P.stride_h - P.pad_h;
    int w_base = int(ow) * P.stride_w - P.pad_w;
    uint xbase = (n * P.C + c) * P.H * P.W;
    float best_v = -INFINITY;
    int best_i = -1;
    for (int kh = 0; kh < P.kH; ++kh) {
        int ih = h_base + kh;
        if (ih < 0 || ih >= int(P.H)) continue;
        for (int kw = 0; kw < P.kW; ++kw) {
            int iw = w_base + kw;
            if (iw < 0 || iw >= int(P.W)) continue;
            float v = X[xbase + uint(ih) * P.W + uint(iw)];
            if (v > best_v) {
                best_v = v;
                best_i = ih * int(P.W) + iw;
            }
        }
    }
    Y[gid]   = best_v;
    Idx[gid] = best_i;
}

// backward: one thread per input pixel — gather adjoint. Scan every
// (oh, ow) output (kernel geometry isn't carried into _backward) and sum
// dY where Idx points back to this input pixel. O(H_out*W_out) per input.
kernel void k_mp_backward(device const float* dY  [[buffer(0)]],
                          device const int*   Idx [[buffer(1)]],
                          device float*       dX  [[buffer(2)]],
                          constant MPParams&  P   [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint w = gid % P.W;
    uint t1 = gid / P.W;
    uint h = t1 % P.H;
    uint t2 = t1 / P.H;
    uint c  = t2 % P.C;
    uint n  = t2 / P.C;

    int my_flat = int(h) * int(P.W) + int(w);
    uint ybase = (n * P.C + c) * P.H_out * P.W_out;
    float acc = 0.0f;
    for (uint oh = 0u; oh < P.H_out; ++oh) {
        for (uint ow = 0u; ow < P.W_out; ++ow) {
            uint o = ybase + oh * P.W_out + ow;
            if (Idx[o] == my_flat) acc += dY[o];
        }
    }
    dX[gid] = acc;
}
)msl";

#define DEF_PSO(NAME, SRC, FN)                                                 \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(SRC, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_aap_fwd, kAAP, @"k_aap_forward")
DEF_PSO(pso_aap_bwd, kAAP, @"k_aap_backward")
DEF_PSO(pso_mp_fwd,  kMP,  @"k_mp_forward")
DEF_PSO(pso_mp_bwd,  kMP,  @"k_mp_backward")
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

} // namespace

// ══════════ adaptive_avg_pool2d ════════════════════════════════════════════

void adaptive_avg_pool2d_forward(const Tensor& X,
                                 int N, int C, int H, int W,
                                 int H_out, int W_out,
                                 Tensor& Y) {
    const char* op = "adaptive_avg_pool2d_forward";
    req_fp32(op, X, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 1 || W_out < 1) fail(op, "H_out and W_out must be >= 1");
    if (X.rows != N || X.cols != C * H * W) fail(op, "X shape must be (N, C*H*W)");

    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0) return;

    AAPParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W;
    p.H_out = H_out; p.W_out = W_out;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_out);

    dispatch1d(pso_aap_fwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(AAPParams) atIndex:2];
    });
}

void adaptive_avg_pool2d_backward(const Tensor& dY,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out,
                                  Tensor& dX) {
    const char* op = "adaptive_avg_pool2d_backward";
    req_fp32(op, dY, "dY");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 1 || W_out < 1) fail(op, "H_out and W_out must be >= 1");
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0) return;

    AAPParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W;
    p.H_out = H_out; p.W_out = W_out;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);

    dispatch1d(pso_aap_bwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(AAPParams) atIndex:2];
    });
}

// ══════════ max_pool2d ═════════════════════════════════════════════════════

void max_pool2d_forward(const Tensor& X,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        Tensor& Y, Tensor& Idx) {
    const char* op = "max_pool2d_forward";
    req_fp32(op, X, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH and kW must be >= 1");
    if (stride_h < 1 || stride_w < 1) fail(op, "strides must be >= 1");
    if (pad_h < 0 || pad_w < 0) fail(op, "pads must be >= 0");
    if (kH > H + 2 * pad_h || kW > W + 2 * pad_w)
        fail(op, "kernel larger than padded input");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (Idx.rows != N || Idx.cols != cols_out || Idx.dtype != Dtype::INT32) {
        Idx.resize(N, cols_out, Dtype::INT32);
    }
    if (N == 0) return;

    MPParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W;
    p.H_out = H_out; p.W_out = W_out;
    p.kH = kH; p.kW = kW;
    p.stride_h = stride_h; p.stride_w = stride_w;
    p.pad_h = pad_h; p.pad_w = pad_w;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_out);

    dispatch1d(pso_mp_fwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X)   offset:buffer_offset_for(X)   atIndex:0];
        [enc setBuffer:buffer_for(Y)   offset:buffer_offset_for(Y)   atIndex:1];
        [enc setBuffer:buffer_for(Idx) offset:buffer_offset_for(Idx) atIndex:2];
        [enc setBytes:&p length:sizeof(MPParams) atIndex:3];
    });
}

void max_pool2d_backward(const Tensor& dY,
                         const Tensor& Idx,
                         int N, int C, int H, int W,
                         int H_out, int W_out,
                         Tensor& dX) {
    const char* op = "max_pool2d_backward";
    req_fp32(op, dY, "dY");
    if (Idx.dtype != Dtype::INT32)
        fail(op, "Idx must be INT32 (as produced by max_pool2d_forward)");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 0 || W_out < 0) fail(op, "H_out and W_out must be >= 0");
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");
    if (Idx.rows != N || Idx.cols != C * H_out * W_out)
        fail(op, "Idx shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0) return;

    // Backward kernel scans all (oh, ow) outputs per input pixel — kernel
    // geometry isn't carried into the backward signature. Acceptable for
    // typical pool dims.
    MPParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W;
    p.H_out = H_out; p.W_out = W_out;
    p.kH = 0; p.kW = 0; p.stride_h = 1; p.stride_w = 1; p.pad_h = 0; p.pad_w = 0;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);

    dispatch1d(pso_mp_bwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY)  offset:buffer_offset_for(dY)  atIndex:0];
        [enc setBuffer:buffer_for(Idx) offset:buffer_offset_for(Idx) atIndex:1];
        [enc setBuffer:buffer_for(dX)  offset:buffer_offset_for(dX)  atIndex:2];
        [enc setBytes:&p length:sizeof(MPParams) atIndex:3];
    });
}

} // namespace brotensor::detail::metal
