// ─── Metal L2 normalization over the channel axis (NCHW) ────────────────────
//
// Metal counterpart of src/cpu/l2_normalize.cpp (CUDA sibling:
// src/cuda/l2_normalize.cu). One thread per spatial position (n, h, w); each
// thread reduces over the C channels and rescales them:
//   Y[n,c,h,w] = X[n,c,h,w] / max(sqrt(sum_c X^2), eps)
// FP32-only on this backend (matches the interp2d / slice2d precedent — the
// CPU contract is FP32-only and the parity suite exercises FP32). Y is
// OVERWRITTEN; X and Y may alias. Inference-only: no backward.

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
        fail(op, std::string(name) + " must be FP32 (l2_normalize is FP32-only on Metal)");
    }
}

// Parameter block — must match the MSL struct below.
struct LnParams {
    uint32_t N, C, HW;
    float    eps;
    uint32_t total;   // N * HW
};

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

struct LnParams {
    uint N, C, HW;
    float eps;
    uint total;
};

// One thread per (n, p) where p in [0, HW). Strided gather over channels.
kernel void k_l2_normalize_nchw(device const float* X  [[buffer(0)]],
                                device float*       Y  [[buffer(1)]],
                                constant LnParams&  p  [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= p.total) return;
    const uint pix  = gid % p.HW;
    const uint n    = gid / p.HW;
    const uint base = n * p.C * p.HW + pix;
    float ss = 0.0f;
    for (uint c = 0; c < p.C; ++c) {
        const float v = X[base + c * p.HW];
        ss += v * v;
    }
    const float inv = 1.0f / max(sqrt(ss), p.eps);
    for (uint c = 0; c < p.C; ++c) {
        const uint off = base + c * p.HW;
        Y[off] = X[off] * inv;
    }
}
)msl";

id<MTLComputePipelineState> pso() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> p;
    dispatch_once(&once, ^{ p = compile_pipeline(kSrc, @"k_l2_normalize_nchw"); });
    return p;
}

} // namespace

void l2_normalize_nchw_forward(const Tensor& X,
                               int N, int C, int H, int W,
                               float eps,
                               Tensor& Y) {
    const char* op = "l2_normalize_nchw_forward";
    req_fp32(op, X, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");
    if (Y.rows != N || Y.cols != C * H * W || Y.dtype != Dtype::FP32)
        Y.resize(N, C * H * W, Dtype::FP32);
    if (N == 0) return;

    const int HW = H * W;
    LnParams p{};
    p.N     = static_cast<uint32_t>(N);
    p.C     = static_cast<uint32_t>(C);
    p.HW    = static_cast<uint32_t>(HW);
    p.eps   = eps;
    p.total = static_cast<uint32_t>(N) * static_cast<uint32_t>(HW);
    if (p.total == 0) return;

    @autoreleasepool {
        id<MTLComputePipelineState> ps = pso();
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:buffer_for(Y) offset:buffer_offset_for(Y) atIndex:1];
        [enc setBytes:&p length:sizeof(LnParams) atIndex:2];
        NSUInteger tpt = [ps maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(p.total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
