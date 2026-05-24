// ─── Metal 2D spatial slice / crop ──────────────────────────────────────────
//
// Metal counterpart of src/cpu/slice2d.cpp. FP32-only. Forward is a gather
// copy (one thread per output element). Backward is the adjoint: zero dX then
// place dY into the slice region. We split backward into a zero kernel +
// scatter kernel — no aliasing (each input pixel receives at most one dY
// value, since slice2d is a pure read).
//
// Memory layout (NCHW flat):
//   X / dX : ((n*C + c)*H     + h)*W     + w
//   Y / dY : ((n*C + c)*H_out + h)*W_out + w

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

void check_args(const char* op,
                int N, int C, int H, int W,
                int h0, int w0, int H_out, int W_out) {
    if (N < 0 || C < 1 || H < 0 || W < 0)
        fail(op, "C must be >=1; N, H, W must be >=0");
    if (H_out < 0 || W_out < 0)
        fail(op, "H_out and W_out must be >=0");
    if (h0 < 0 || w0 < 0)
        fail(op, "h0 and w0 must be >=0");
    if (h0 + H_out > H)
        fail(op, "h0 + H_out must be <= H");
    if (w0 + W_out > W)
        fail(op, "w0 + W_out must be <= W");
}

struct Slice2dParams {
    uint32_t N, C, H, W, H_out, W_out;
    int32_t  h0, w0;
    uint32_t total_out;
    uint32_t total_in;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct Slice2dParams {
    uint N, C, H, W, H_out, W_out;
    int h0, w0;
    uint total_out;
    uint total_in;
};

kernel void k_slice2d_forward(device const float* X [[buffer(0)]],
                              device float*       Y [[buffer(1)]],
                              constant Slice2dParams& P [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total_out) return;
    uint ow  = gid % P.W_out;
    uint t1  = gid / P.W_out;
    uint oh  = t1 % P.H_out;
    uint t2  = t1 / P.H_out;
    uint c   = t2 % P.C;
    uint n   = t2 / P.C;
    uint h = uint(P.h0) + oh;
    uint w = uint(P.w0) + ow;
    uint xbase = (n * P.C + c) * P.H * P.W;
    Y[gid] = X[xbase + h * P.W + w];
}

kernel void k_slice2d_zero(device float* dX [[buffer(0)]],
                           constant uint& n [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dX[gid] = 0.0f;
}

kernel void k_slice2d_scatter(device const float* dY [[buffer(0)]],
                              device float*       dX [[buffer(1)]],
                              constant Slice2dParams& P [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total_out) return;
    uint ow  = gid % P.W_out;
    uint t1  = gid / P.W_out;
    uint oh  = t1 % P.H_out;
    uint t2  = t1 / P.H_out;
    uint c   = t2 % P.C;
    uint n   = t2 / P.C;
    uint h = uint(P.h0) + oh;
    uint w = uint(P.w0) + ow;
    uint xbase = (n * P.C + c) * P.H * P.W;
    dX[xbase + h * P.W + w] = dY[gid];
}
)msl";

id<MTLComputePipelineState> pso_fwd() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_slice2d_forward"); });
    return pso;
}
id<MTLComputePipelineState> pso_zero() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_slice2d_zero"); });
    return pso;
}
id<MTLComputePipelineState> pso_scatter() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_slice2d_scatter"); });
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

void slice2d_forward(const Tensor& X,
                     int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out,
                     Tensor& Y) {
    const char* op = "slice2d_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, H, W, h0, w0, H_out, W_out);
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must be (N, C*H*W)");
    }
    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    Slice2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H = static_cast<uint32_t>(H);
    p.W = static_cast<uint32_t>(W);
    p.H_out = static_cast<uint32_t>(H_out);
    p.W_out = static_cast<uint32_t>(W_out);
    p.h0 = h0;
    p.w0 = w0;
    p.total_out = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_out);
    p.total_in  = static_cast<uint32_t>(N) * static_cast<uint32_t>(C * H * W);

    dispatch1d(pso_fwd(), p.total_out, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(Slice2dParams) atIndex:2];
    });
}

void slice2d_backward(const Tensor& dY,
                      int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out,
                      Tensor& dX) {
    const char* op = "slice2d_backward";
    req_fp32(op, dY, "dY");
    check_args(op, N, C, H, W, h0, w0, H_out, W_out);
    if (dY.rows != N || dY.cols != C * H_out * W_out) {
        fail(op, "dY shape must be (N, C*H_out*W_out)");
    }
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    Slice2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H = static_cast<uint32_t>(H);
    p.W = static_cast<uint32_t>(W);
    p.H_out = static_cast<uint32_t>(H_out);
    p.W_out = static_cast<uint32_t>(W_out);
    p.h0 = h0;
    p.w0 = w0;
    p.total_out = static_cast<uint32_t>(N) * static_cast<uint32_t>(C * H_out * W_out);
    p.total_in  = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);

    // Zero dX.
    dispatch1d(pso_zero(), p.total_in, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:0];
        [enc setBytes:&p.total_in length:sizeof(uint32_t) atIndex:1];
    });
    // Scatter dY into the slice region.
    if (H_out == 0 || W_out == 0) return;
    dispatch1d(pso_scatter(), p.total_out, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(Slice2dParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
