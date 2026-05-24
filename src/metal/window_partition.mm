// ─── Metal SAM-style window partition / reverse ─────────────────────────────
//
// Metal counterpart of src/cpu/window_partition.cpp. FP32-only.
// Both ops are pure rearrangements (no math) and are exact inverses of each
// other — each has one kernel that picks the correct (src, dst) mapping.
//
// Layout (NCHW):
//   X (N, C*H*W)               -> Y (N*nw_h*nw_w, C*window*window)
//   row index in Y = n*nw_h*nw_w + nh*nw_w + nw
//   within-row    = (c*window + lh)*window + lw
//   with (h, w) = (nh*window + lh, nw*window + lw)
//
// window_reverse_forward inverts the same mapping.

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

void check_args(const char* op, int N, int C, int H, int W, int window) {
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (window < 1) fail(op, "window must be >=1");
    if (H % window != 0 || W % window != 0)
        fail(op, "H and W must be multiples of window (use pad2d first if the "
                 "input doesn't align)");
}

struct WPParams {
    uint32_t N, C, H, W, window, nw_h, nw_w;
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct WPParams { uint N, C, H, W, window, nw_h, nw_w, total; };

// One thread per element of Y (the windowed layout). Y has size
//   (N*nw_h*nw_w) * (C*window*window). Decompose gid and compute the source
// index in X.
kernel void k_window_partition(device const float* X [[buffer(0)]],
                               device float*       Y [[buffer(1)]],
                               constant WPParams&  P [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint cols_out = P.C * P.window * P.window;
    uint b_out = gid / cols_out;
    uint within = gid - b_out * cols_out;
    uint lw = within % P.window;
    uint t1 = within / P.window;
    uint lh = t1 % P.window;
    uint c  = t1 / P.window;
    uint nw = b_out % P.nw_w;
    uint t2 = b_out / P.nw_w;
    uint nh = t2 % P.nw_h;
    uint n  = t2 / P.nw_h;
    uint h = nh * P.window + lh;
    uint w = nw * P.window + lw;
    uint x_idx = (n * P.C + c) * P.H * P.W + h * P.W + w;
    Y[gid] = X[x_idx];
}

// One thread per element of Y (the NCHW layout). The inverse mapping.
kernel void k_window_reverse(device const float* X [[buffer(0)]],
                             device float*       Y [[buffer(1)]],
                             constant WPParams&  P [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint cols_in = P.C * P.window * P.window;
    // Y has shape (N, C*H*W); decompose gid as (n, c, h, w).
    uint cols_out = P.C * P.H * P.W;
    uint n = gid / cols_out;
    uint within = gid - n * cols_out;
    uint w = within % P.W;
    uint t1 = within / P.W;
    uint h = t1 % P.H;
    uint c = t1 / P.H;
    uint nh = h / P.window;
    uint lh = h - nh * P.window;
    uint nw = w / P.window;
    uint lw = w - nw * P.window;
    uint b_in = (n * P.nw_h + nh) * P.nw_w + nw;
    uint x_idx = b_in * cols_in + (c * P.window + lh) * P.window + lw;
    Y[gid] = X[x_idx];
}
)msl";

id<MTLComputePipelineState> pso_partition() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_window_partition"); });
    return pso;
}
id<MTLComputePipelineState> pso_reverse() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_window_reverse"); });
    return pso;
}

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

void window_partition_forward(const Tensor& X,
                              int N, int C, int H, int W, int window,
                              Tensor& Y) {
    const char* op = "window_partition_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, H, W, window);
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int B_out = N * nw_h * nw_w;
    const int cols_out = C * window * window;
    if (Y.rows != B_out || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(B_out, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    WPParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W;
    p.window = window; p.nw_h = nw_h; p.nw_w = nw_w;
    p.total = static_cast<uint32_t>(B_out) * static_cast<uint32_t>(cols_out);

    dispatch1d(pso_partition(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(WPParams) atIndex:2];
    });
}

void window_reverse_forward(const Tensor& X,
                            int N, int C, int H, int W, int window,
                            Tensor& Y) {
    const char* op = "window_reverse_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, H, W, window);
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int B_in = N * nw_h * nw_w;
    const int cols_in = C * window * window;
    if (X.rows != B_in || X.cols != cols_in)
        fail(op, "X shape must be (N*nw_h*nw_w, C*window*window)");
    const int cols_out = C * H * W;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    WPParams p{};
    p.N = N; p.C = C; p.H = H; p.W = W;
    p.window = window; p.nw_h = nw_h; p.nw_w = nw_w;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_out);

    dispatch1d(pso_reverse(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(WPParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
