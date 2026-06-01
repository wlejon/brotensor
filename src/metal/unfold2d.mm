// ─── Metal 2D neighborhood unfold (im2col, spatial-preserving) ──────────────
//
// Metal counterpart of src/cpu/unfold2d.cpp (CUDA sibling: src/cuda/unfold2d.cu).
// One thread per output element. Contracts, layout, and `mode` convention are
// identical to the CPU/CUDA files:
//   X : (N, C*H*W)            NCHW flat
//   Y : (N, C*kK*H_out*W_out) with kK = kH*kW, k = ky*kW + kx
//   Y[n,c,k,oy,ox] = X[n,c, oy*sh - pad_top + ky, ox*sw - pad_left + kx]
//   out-of-range source: mode 0 zero / 1 reflect / 2 replicate.
// FP32-only on this backend (matches the interp2d / slice2d precedent). Forward
// OVERWRITES Y. Inference-only: no backward. Pure integer addressing → the
// result is bit-stable against the CPU reference.

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
        fail(op, std::string(name) + " must be FP32 (unfold2d is FP32-only on Metal)");
    }
}

// Parameter block — must match the MSL struct below.
struct UfParams {
    uint32_t N, C, H, W;
    uint32_t kH, kW;
    uint32_t stride_h, stride_w;
    uint32_t pad_top, pad_left;
    uint32_t mode;
    uint32_t H_out, W_out;
    uint32_t total;   // N * C * kK * H_out * W_out
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct UfParams {
    uint N, C, H, W;
    uint kH, kW;
    uint stride_h, stride_w;
    uint pad_top, pad_left;
    uint mode;
    uint H_out, W_out;
    uint total;
};

// Device mirror of the CPU/CUDA unf_src helper.
static inline int unf_src(int coord, int L, uint mode) {
    if (coord >= 0 && coord < L) return coord;
    if (mode == 0) return -1;
    if (mode == 2) return coord < 0 ? 0 : L - 1;
    if (L == 1) return 0;
    int q = coord;
    const int period = 2 * (L - 1);
    q %= period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

kernel void k_unfold2d_forward(device const float* X  [[buffer(0)]],
                               device float*       Y  [[buffer(1)]],
                               constant UfParams&  p  [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total) return;
    const uint kK = p.kH * p.kW;

    const uint ox = gid % p.W_out;
    uint t = gid / p.W_out;
    const uint oy = t % p.H_out;
    t /= p.H_out;
    const uint k = t % kK;
    t /= kK;
    const uint c = t % p.C;
    const uint n = t / p.C;

    const uint ky = k / p.kW;
    const uint kx = k % p.kW;
    const int sy = unf_src(int(oy * p.stride_h) - int(p.pad_top)  + int(ky), int(p.H), p.mode);
    const int sx = unf_src(int(ox * p.stride_w) - int(p.pad_left) + int(kx), int(p.W), p.mode);
    float v = 0.0f;
    if (sy >= 0 && sx >= 0) {
        const ulong xbase = ((ulong)n * p.C + c) * p.H * p.W;
        v = X[xbase + (ulong)sy * p.W + uint(sx)];
    }
    Y[gid] = v;
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_unfold2d_forward"); });
    return p;
}

} // namespace

void unfold2d_forward(const Tensor& X,
                      int N, int C, int H, int W,
                      int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_top, int pad_bottom,
                      int pad_left, int pad_right,
                      int mode,
                      Tensor& Y) {
    const char* op = "unfold2d_forward";
    req_fp32(op, X, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH/kW must be >=1");
    if (stride_h < 1 || stride_w < 1) fail(op, "stride must be >=1");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0)
        fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2)
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    const int W_out = (W + pad_left + pad_right - kW) / stride_w + 1;
    if (H_out < 1 || W_out < 1)
        fail(op, "kernel/padding/stride yield empty output");
    const int kK = kH * kW;
    const int cols_out = C * kK * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != Dtype::FP32)
        Y.resize(N, cols_out, Dtype::FP32);
    if (N == 0) return;

    UfParams p{};
    p.N        = static_cast<uint32_t>(N);
    p.C        = static_cast<uint32_t>(C);
    p.H        = static_cast<uint32_t>(H);
    p.W        = static_cast<uint32_t>(W);
    p.kH       = static_cast<uint32_t>(kH);
    p.kW       = static_cast<uint32_t>(kW);
    p.stride_h = static_cast<uint32_t>(stride_h);
    p.stride_w = static_cast<uint32_t>(stride_w);
    p.pad_top  = static_cast<uint32_t>(pad_top);
    p.pad_left = static_cast<uint32_t>(pad_left);
    p.mode     = static_cast<uint32_t>(mode);
    p.H_out    = static_cast<uint32_t>(H_out);
    p.W_out    = static_cast<uint32_t>(W_out);
    p.total    = static_cast<uint32_t>((long long)N * cols_out);
    if (p.total == 0) return;

    @autoreleasepool {
        id<MTLComputePipelineState> ps = pso();
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(UfParams) atIndex:2];
        NSUInteger tpt = [ps maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(p.total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
