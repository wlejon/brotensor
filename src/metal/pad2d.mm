// ─── Metal 2D padding ──────────────────────────────────────────────────────
//
// Metal counterpart of src/cpu/pad2d.cpp. FP32-only. Forward kernel runs one
// thread per output pixel and reads via the same pad_src axis mapping the
// CPU uses (zero / reflect / replicate). Backward inverts that: one thread
// per input pixel sums every output pixel that maps back to it — pure gather
// adjoint, no atomics.
//
// Memory layout (NCHW flat — matches pad2d.cpp / interp2d.mm):
//   X / dX : ((n*C + c)*H     + h)*W     + w
//   Y / dY : ((n*C + c)*H_pad + h)*W_pad + w
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   pad2d_forward  — Y  OVERWRITTEN.
//   pad2d_backward — dX OVERWRITTEN (gather adjoint; for reflect/replicate
//                    several output positions collapse onto the same input
//                    pixel and their contributions sum).

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
        fail(op, std::string(name) + " must be FP32 (CPU/Metal pad2d is FP32-only)");
    }
}

void check_args(const char* op,
                int N, int C, int H, int W,
                int pad_top, int pad_bottom,
                int pad_left, int pad_right, int mode) {
    if (N < 0 || C < 1 || H < 1 || W < 1) {
        fail(op, "C/H/W must be >=1 and N >=0");
    }
    if (pad_top < 0 || pad_bottom < 0 ||
        pad_left < 0 || pad_right < 0) {
        fail(op, "pad counts must be >=0");
    }
    if (mode < 0 || mode > 2) {
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    }
    if (mode == 1) {
        if (pad_top >= H || pad_bottom >= H) {
            fail(op, "reflect padding requires pad_top and pad_bottom < H");
        }
        if (pad_left >= W || pad_right >= W) {
            fail(op, "reflect padding requires pad_left and pad_right < W");
        }
    }
}

struct Pad2dParams {
    uint32_t N, C, H, W, H_pad, W_pad;
    int32_t  pad_top, pad_left;
    uint32_t mode;
    uint32_t total;
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct Pad2dParams {
    uint N, C, H, W, H_pad, W_pad;
    int pad_top, pad_left;
    uint mode;
    uint total;
};

// Map an output position p in [0, L_pad) to a source index in [0, L) for the
// given mode, or -1 for a zero-padded slot. Same as the CPU pad_src.
static inline int pad_src(int p, int L, int pad_left, uint mode) {
    int rel = p - pad_left;
    if (rel >= 0 && rel < L) return rel;
    if (mode == 0u) return -1;
    if (mode == 2u) return rel < 0 ? 0 : (L - 1);
    if (L == 1) return 0;
    int q = rel;
    int period = 2 * (L - 1);
    q = q % period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

// forward: one thread per output element.
kernel void k_pad2d_forward(device const float*  X [[buffer(0)]],
                            device float*        Y [[buffer(1)]],
                            constant Pad2dParams& P [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint q   = gid % P.W_pad;
    uint t1  = gid / P.W_pad;
    uint p   = t1 % P.H_pad;
    uint t2  = t1 / P.H_pad;
    uint c   = t2 % P.C;
    uint n   = t2 / P.C;
    int src_h = pad_src(int(p), int(P.H), P.pad_top,  P.mode);
    int src_w = pad_src(int(q), int(P.W), P.pad_left, P.mode);
    if (src_h < 0 || src_w < 0) {
        Y[gid] = 0.0f;
        return;
    }
    uint xbase = (n * P.C + c) * P.H * P.W;
    Y[gid] = X[xbase + uint(src_h) * P.W + uint(src_w)];
}

// backward: one thread per input element — gather adjoint. For each input
// pixel (iy, ix), walk every output position and sum dY where the forward
// mapping landed on this input. Mirrors CPU scatter sum order (oh asc, ow asc).
kernel void k_pad2d_backward(device const float*  dY [[buffer(0)]],
                             device float*        dX [[buffer(1)]],
                             constant Pad2dParams& P  [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= P.total) return;
    uint ix  = gid % P.W;
    uint t1  = gid / P.W;
    uint iy  = t1 % P.H;
    uint t2  = t1 / P.H;
    uint c   = t2 % P.C;
    uint n   = t2 / P.C;
    uint ybase = (n * P.C + c) * P.H_pad * P.W_pad;
    int ixi = int(ix), iyi = int(iy);
    float acc = 0.0f;
    for (uint p = 0u; p < P.H_pad; ++p) {
        int src_h = pad_src(int(p), int(P.H), P.pad_top, P.mode);
        if (src_h != iyi) continue;
        for (uint q = 0u; q < P.W_pad; ++q) {
            int src_w = pad_src(int(q), int(P.W), P.pad_left, P.mode);
            if (src_w == ixi) {
                acc += dY[ybase + p * P.W_pad + q];
            }
        }
    }
    dX[gid] = acc;
}
)msl";

id<MTLComputePipelineState> pso_fwd() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_pad2d_forward"); });
    return pso;
}
id<MTLComputePipelineState> pso_bwd() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_pad2d_backward"); });
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

void pad2d_forward(const Tensor& X,
                   int N, int C, int H, int W,
                   int pad_top, int pad_bottom,
                   int pad_left, int pad_right, int mode,
                   Tensor& Y) {
    const char* op = "pad2d_forward";
    req_fp32(op, X, "X");
    check_args(op, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode);
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must be (N, C*H*W)");
    }
    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    const int cols_out = C * H_pad * W_pad;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols_out, Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    Pad2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H = static_cast<uint32_t>(H);
    p.W = static_cast<uint32_t>(W);
    p.H_pad = static_cast<uint32_t>(H_pad);
    p.W_pad = static_cast<uint32_t>(W_pad);
    p.pad_top  = pad_top;
    p.pad_left = pad_left;
    p.mode = static_cast<uint32_t>(mode);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_out);

    dispatch1d(pso_fwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(Pad2dParams) atIndex:2];
    });
}

void pad2d_backward(const Tensor& dY,
                    int N, int C, int H, int W,
                    int pad_top, int pad_bottom,
                    int pad_left, int pad_right, int mode,
                    Tensor& dX) {
    const char* op = "pad2d_backward";
    req_fp32(op, dY, "dY");
    check_args(op, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode);
    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    if (dY.rows != N || dY.cols != C * H_pad * W_pad) {
        fail(op, "dY shape must be (N, C*(H+pt+pb)*(W+pl+pr))");
    }
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols_in, Dtype::FP32);
    }
    if (N == 0 || cols_in == 0) return;

    Pad2dParams p{};
    p.N = static_cast<uint32_t>(N);
    p.C = static_cast<uint32_t>(C);
    p.H = static_cast<uint32_t>(H);
    p.W = static_cast<uint32_t>(W);
    p.H_pad = static_cast<uint32_t>(H_pad);
    p.W_pad = static_cast<uint32_t>(W_pad);
    p.pad_top  = pad_top;
    p.pad_left = pad_left;
    p.mode = static_cast<uint32_t>(mode);
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols_in);

    dispatch1d(pso_bwd(), p.total, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&p length:sizeof(Pad2dParams) atIndex:2];
    });
}

} // namespace brotensor::detail::metal
