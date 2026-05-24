// Metal image preprocessing helpers (FP32).
//
//   image_normalize           — per-channel (X - mean[c]) / std[c] on NCHW.
//                               One thread per element; per-channel mean /
//                               inv_std precomputed into a C-length scratch.
//   image_u8_to_f32_nhwc_to_nchw — convert packed uint8 NHWC into FP32 NCHW
//                               with Y = src * scale + bias. The `src`
//                               pointer is a *device* pointer registered in
//                               the Metal pool (same convention as
//                               embedding_lookup_forward's `const int32_t*
//                               d_idx`).

#include <brotensor/runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_lookup;
using metal_impl::pool_lookup_offset;

namespace {

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

// One thread per element of NCHW Y. Decode (n, c, h, w) from the linear
// index — only `c` is actually needed for the per-channel lookup.
kernel void k_image_normalize(device const float* X       [[buffer(0)]],
                              device const float* mean    [[buffer(1)]],
                              device const float* inv_std [[buffer(2)]],
                              device float*       Y       [[buffer(3)]],
                              constant uint& total        [[buffer(4)]],
                              constant uint& C            [[buffer(5)]],
                              constant uint& spatial      [[buffer(6)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= total) return;
    uint c = (i / spatial) % C;
    Y[i] = (X[i] - mean[c]) * inv_std[c];
}

kernel void k_image_inv_std(device const float* std_ [[buffer(0)]],
                            device float*       inv  [[buffer(1)]],
                            constant uint& C         [[buffer(2)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    // 1/std; if std==0 the result is inf — mirrors the CUDA port (a device
    // kernel can't throw). The CPU path checks and throws beforehand.
    inv[c] = 1.0f / std_[c];
}

// One thread per output element. Maps Y[(n, c, h, w)] back to
// src[n*H*W*C + (h*W + w)*C + c].
kernel void k_image_u8_to_f32_nhwc_to_nchw(
        device const uchar* src   [[buffer(0)]],
        device float*       Y     [[buffer(1)]],
        constant uint& N          [[buffer(2)]],
        constant uint& H          [[buffer(3)]],
        constant uint& W          [[buffer(4)]],
        constant uint& C          [[buffer(5)]],
        constant float& scale     [[buffer(6)]],
        constant float& bias      [[buffer(7)]],
        uint idx [[thread_position_in_grid]]) {
    const uint spatial = H * W;
    const uint total = N * C * spatial;
    if (idx >= total) return;
    const uint n  =  idx / (C * spatial);
    const uint rm =  idx % (C * spatial);
    const uint c  =  rm / spatial;
    const uint s  =  rm % spatial;
    const uint h  =  s / W;
    const uint w  =  s % W;
    const uint src_idx = ((n * H + h) * W + w) * C + c;
    Y[idx] = float(src[src_idx]) * scale + bias;
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_normalize, @"k_image_normalize")
DEF_PSO(pso_inv_std,   @"k_image_inv_std")
DEF_PSO(pso_u8_nhwc_nchw, @"k_image_u8_to_f32_nhwc_to_nchw")
#undef DEF_PSO

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32");
    }
}

void dispatch1d(id<MTLComputeCommandEncoder> enc,
                id<MTLComputePipelineState> pso, NSUInteger n) {
    if (n == 0) return;
    [enc setComputePipelineState:pso];
    NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
    if (tg > 256) tg = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

} // namespace

void image_normalize(const Tensor& X,
                     const Tensor& mean,
                     const Tensor& std_,
                     int N, int C, int H, int W,
                     Tensor& Y) {
    check_fp32(X,    "image_normalize", "X");
    check_fp32(mean, "image_normalize", "mean");
    check_fp32(std_, "image_normalize", "std");
    if (mean.size() != C || std_.size() != C) {
        throw std::runtime_error("brotensor: image_normalize: mean/std must have C elements");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("brotensor: image_normalize: X shape mismatch");
    }
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t Su = static_cast<uint32_t>(spatial);
    const uint32_t total = static_cast<uint32_t>(N) * Cu * Su;

    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bM = buffer_for(mean);
    id<MTLBuffer> bS = buffer_for(std_);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oM = buffer_offset_for(mean);
    const NSUInteger oS = buffer_offset_for(std_);
    const NSUInteger oY = buffer_offset_for(Y);

    @autoreleasepool {
        // Per-channel 1/std scratch (cheap: C floats).
        id<MTLBuffer> inv = [metal_impl::device()
            newBufferWithLength:C * sizeof(float)
                        options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso_inv_std()];
        [enc setBuffer:bS  offset:oS  atIndex:0];
        [enc setBuffer:inv offset:0   atIndex:1];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:2];
        dispatch1d(enc, pso_inv_std(), C);

        [enc setComputePipelineState:pso_normalize()];
        [enc setBuffer:bX  offset:oX atIndex:0];
        [enc setBuffer:bM  offset:oM atIndex:1];
        [enc setBuffer:inv offset:0  atIndex:2];
        [enc setBuffer:bY  offset:oY atIndex:3];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Cu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Su    length:sizeof(uint32_t) atIndex:6];
        dispatch1d(enc, pso_normalize(), total);

        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  Tensor& Y) {
    if (src == nullptr && N > 0 && H > 0 && W > 0 && C > 0) {
        throw std::runtime_error(
            "brotensor: image_u8_to_f32_nhwc_to_nchw: src is null");
    }
    if (N < 0 || H < 0 || W < 0 || C < 0) {
        throw std::runtime_error(
            "brotensor: image_u8_to_f32_nhwc_to_nchw: negative dim");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    // Resolve the device-side src pointer through the pool (same convention
    // as embedding_lookup_forward's d_idx).
    id<MTLBuffer> bSrc = pool_lookup(src);
    const NSUInteger oSrc = pool_lookup_offset(src);
    if (bSrc == nil) {
        throw std::runtime_error(
            "brotensor: image_u8_to_f32_nhwc_to_nchw: src is not a Metal-pool pointer");
    }

    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oY = buffer_offset_for(Y);

    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Hu = static_cast<uint32_t>(H);
    const uint32_t Wu = static_cast<uint32_t>(W);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t total = Nu * Cu * static_cast<uint32_t>(spatial);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_u8_nhwc_nchw()];
        [enc setBuffer:bSrc offset:oSrc atIndex:0];
        [enc setBuffer:bY   offset:oY   atIndex:1];
        [enc setBytes:&Nu length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Hu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Wu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&scale length:sizeof(float) atIndex:6];
        [enc setBytes:&bias  length:sizeof(float) atIndex:7];
        dispatch1d(enc, pso_u8_nhwc_nchw(), total);
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
