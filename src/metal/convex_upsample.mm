// ─── Metal convex (mask-based) upsample, NCHW ───────────────────────────────
//
// Metal counterpart of src/cpu/convex_upsample.cpp (CUDA sibling:
// src/cuda/convex_upsample.cu). One thread per OUTPUT element (n, c, oy, ox);
// each thread derives its source low-res pixel (y, x) and sub-position
// (sy, sx), softmaxes the 9 mask logits for that (sy, sx, y, x), and blends the
// 3×3 low-res neighborhood of channel c:
//   Y[n,c,k*y+sy,k*x+sx] = sum_m softmax_m(Mask[n,m,sy,sx,y,x]) * X[n,c,ny,nx]
//   neighbor m: ny=clamp(y-1+m/3), nx=clamp(x-1+m%3)  (replicate pad)
// Mask flat channel = (m*k*k + sy*k + sx). The softmax is recomputed per output
// channel (redundant across C) for kernel simplicity — fine at the C/scale this
// targets. FP32-only on this backend (matches the interp2d / slice2d
// precedent). Y is OVERWRITTEN. Inference-only: no backward.

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
        fail(op, std::string(name) + " must be FP32 (convex_upsample is FP32-only on Metal)");
    }
}

// Parameter block — must match the MSL struct below.
struct CuParams {
    uint32_t N, C, H, W, scale;
    uint32_t total;   // N * C * (scale*H) * (scale*W)
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct CuParams {
    uint N, C, H, W, scale;
    uint total;
};

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

kernel void k_convex_upsample(device const float* X     [[buffer(0)]],
                              device const float* Mask  [[buffer(1)]],
                              device float*       Y     [[buffer(2)]],
                              constant CuParams&  p     [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total) return;
    const uint H = p.H, W = p.W, C = p.C, scale = p.scale;
    const uint HW = H * W;
    const uint kk = scale * scale;
    const uint oW = scale * W, oH = scale * H;

    const uint ox = gid % oW;
    uint t = gid / oW;
    const uint oy = t % oH;
    t /= oH;
    const uint c = t % C;
    const uint n = t / C;

    const uint y = oy / scale, sy = oy % scale;
    const uint x = ox / scale, sx = ox % scale;
    const uint sub = sy * scale + sx;
    const uint pix = y * W + x;
    device const float* m_img = Mask + (ulong)n * 9 * kk * HW;

    float mx = -3.4e38f;
    for (uint m = 0; m < 9; ++m) {
        const float v = m_img[((ulong)m * kk + sub) * HW + pix];
        if (v > mx) mx = v;
    }
    float w[9];
    float sum = 0.0f;
    for (uint m = 0; m < 9; ++m) {
        const float e = exp(m_img[((ulong)m * kk + sub) * HW + pix] - mx);
        w[m] = e; sum += e;
    }
    const float invs = 1.0f / sum;

    device const float* xc = X + ((ulong)n * C + c) * HW;
    float acc = 0.0f;
    for (uint m = 0; m < 9; ++m) {
        const int ny = clampi(int(y) - 1 + int(m / 3), 0, int(H) - 1);
        const int nx = clampi(int(x) - 1 + int(m % 3), 0, int(W) - 1);
        acc += (w[m] * invs) * xc[(ulong)ny * W + nx];
    }
    Y[gid] = acc;
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_convex_upsample"); });
    return p;
}

} // namespace

void convex_upsample_forward(const Tensor& X,
                             const Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             Tensor& Y) {
    const char* op = "convex_upsample_forward";
    req_fp32(op, X, "X");
    req_fp32(op, Mask, "Mask");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (scale < 1) fail(op, "scale must be >=1");
    const int HW = H * W;
    const int kk = scale * scale;
    if (X.rows != N || X.cols != C * HW) fail(op, "X shape must be (N, C*H*W)");
    if (Mask.rows != N || Mask.cols != 9 * kk * HW)
        fail(op, "Mask shape must be (N, 9*scale*scale*H*W)");
    const int oH = scale * H, oW = scale * W;
    const long long oHW = (long long)oH * oW;
    if (Y.rows != N || Y.cols != C * oHW || Y.dtype != Dtype::FP32)
        Y.resize(N, static_cast<int>(C * oHW), Dtype::FP32);
    if (N == 0) return;

    CuParams p{};
    p.N     = static_cast<uint32_t>(N);
    p.C     = static_cast<uint32_t>(C);
    p.H     = static_cast<uint32_t>(H);
    p.W     = static_cast<uint32_t>(W);
    p.scale = static_cast<uint32_t>(scale);
    p.total = static_cast<uint32_t>((long long)N * C * oHW);
    if (p.total == 0) return;

    @autoreleasepool {
        id<MTLComputePipelineState> ps = pso();
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_for(X)    offset:buffer_offset_for(X)    atIndex:0];
        [enc setBuffer:buffer_for(Mask) offset:buffer_offset_for(Mask) atIndex:1];
        [enc setBuffer:buffer_for(Y)    offset:buffer_offset_for(Y)    atIndex:2];
        [enc setBytes:&p length:sizeof(CuParams) atIndex:3];
        NSUInteger tpt = [ps maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(p.total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
