// Metal spatial 2x2 patch merger (Qwen3-VL).
//
// Pure gather: (N, C, H, W) -> (N, 4*C, H/2, W/2). One thread per output
// element. FP32 / FP16 / BF16.

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

#define SPATIAL_MERGE(NAME, T)                                                \
kernel void NAME(device const T* X [[buffer(0)]],                             \
                 device T*       Y [[buffer(1)]],                             \
                 constant uint& N      [[buffer(2)]],                         \
                 constant uint& C      [[buffer(3)]],                         \
                 constant uint& H      [[buffer(4)]],                         \
                 constant uint& W      [[buffer(5)]],                         \
                 constant uint& H_out  [[buffer(6)]],                         \
                 constant uint& W_out  [[buffer(7)]],                         \
                 constant uint& total  [[buffer(8)]],                         \
                 uint idx [[thread_position_in_grid]]) {                      \
    if (idx >= total) return;                                                 \
    uint C_out = 4u * C;                                                      \
    uint w_out = idx % W_out;                                                 \
    uint t     = idx / W_out;                                                 \
    uint h_out = t % H_out;                                                   \
    t         /= H_out;                                                       \
    uint c_out = t % C_out;                                                   \
    uint n     = t / C_out;                                                   \
    uint block = c_out / C;                                                   \
    uint c_in  = c_out - block * C;                                           \
    uint dh    = block >> 1;                                                  \
    uint dw    = block & 1u;                                                  \
    uint h_in  = 2u * h_out + dh;                                             \
    uint w_in  = 2u * w_out + dw;                                             \
    uint HW    = H * W;                                                       \
    uint x_idx = (n * C + c_in) * HW + h_in * W + w_in;                       \
    Y[idx] = X[x_idx];                                                        \
    (void)N;                                                                  \
}

SPATIAL_MERGE(k_spatial_merge_2x2_fp32, float)
SPATIAL_MERGE(k_spatial_merge_2x2_fp16, half)
SPATIAL_MERGE(k_spatial_merge_2x2_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_sm_fp32, @"k_spatial_merge_2x2_fp32")
DEF_PSO(pso_sm_fp16, @"k_spatial_merge_2x2_fp16")
DEF_PSO(pso_sm_bf16, @"k_spatial_merge_2x2_bf16")
#undef DEF_PSO

} // namespace

void spatial_merge_2x2_forward(const Tensor& X,
                               int N, int C, int H, int W,
                               Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("spatial_merge_2x2_forward: X must be FP32, "
                                 "FP16, or BF16");
    }
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error("spatial_merge_2x2_forward: negative dimension");
    }
    if ((H & 1) != 0 || (W & 1) != 0) {
        throw std::runtime_error("spatial_merge_2x2_forward: H and W must be even");
    }
    const int H_out = H / 2;
    const int W_out = W / 2;
    const int C_out = 4 * C;
    const int cols  = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const uint32_t total = (uint32_t)(N) * (uint32_t)cols;
    if (total == 0) return;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_sm_fp16()
      : (X.dtype == Dtype::BF16) ? pso_sm_bf16()
      : pso_sm_fp32();

    const uint32_t Nu     = (uint32_t)N;
    const uint32_t Cu     = (uint32_t)C;
    const uint32_t Hu     = (uint32_t)H;
    const uint32_t Wu     = (uint32_t)W;
    const uint32_t H_outU = (uint32_t)H_out;
    const uint32_t W_outU = (uint32_t)W_out;

    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    NSUInteger ox = buffer_offset_for(X);
    NSUInteger oy = buffer_offset_for(Y);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Nu     length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Cu     length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Hu     length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Wu     length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&H_outU length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&W_outU length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&total  length:sizeof(uint32_t) atIndex:8];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
