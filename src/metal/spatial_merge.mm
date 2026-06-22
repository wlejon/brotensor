// Metal spatial 2x2 pixel-unshuffle (Qwen-VL merger / Flux.2 VAE).
//
// Pure gather: (N, C, H, W) -> (N, 4*C, H/2, W/2). One thread per output
// element. channel_major selects the output channel ordering (block-major
// Qwen-VL vs channel-major torch pixel_unshuffle). FP32 / FP16 / BF16.

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
                 constant uint& chmaj  [[buffer(9)]],                         \
                 uint idx [[thread_position_in_grid]]) {                      \
    if (idx >= total) return;                                                 \
    uint C_out = 4u * C;                                                      \
    uint w_out = idx % W_out;                                                 \
    uint t     = idx / W_out;                                                 \
    uint h_out = t % H_out;                                                   \
    t         /= H_out;                                                       \
    uint c_out = t % C_out;                                                   \
    uint n     = t / C_out;                                                   \
    uint block = chmaj ? (c_out & 3u)  : (c_out / C);                         \
    uint c_in  = chmaj ? (c_out >> 2)  : (c_out - block * C);                 \
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

// DiT unpatchify: Y[c,i*P+py,j*P+px] = tokens[i*wp+j, col], c in [0,C_keep).
#define PATCH_UNPACK(NAME, T)                                                 \
kernel void NAME(device const T* X [[buffer(0)]],                            \
                 device T*       Y [[buffer(1)]],                            \
                 constant uint& wp         [[buffer(2)]],                    \
                 constant uint& P          [[buffer(3)]],                    \
                 constant uint& C_total    [[buffer(4)]],                    \
                 constant uint& H          [[buffer(5)]],                    \
                 constant uint& W          [[buffer(6)]],                    \
                 constant uint& row_stride [[buffer(7)]],                    \
                 constant uint& total      [[buffer(8)]],                    \
                 constant uint& chmaj      [[buffer(9)]],                    \
                 uint idx [[thread_position_in_grid]]) {                     \
    if (idx >= total) return;                                                 \
    uint x = idx % W;                                                         \
    uint t = idx / W;                                                         \
    uint y = t % H;                                                           \
    uint c = t / H;                                                           \
    uint i  = y / P;                                                          \
    uint py = y % P;                                                          \
    uint j  = x / P;                                                          \
    uint px = x % P;                                                          \
    uint block = py * P + px;                                                 \
    uint col = chmaj ? (c * (P * P) + block) : (block * C_total + c);         \
    uint tok = i * wp + j;                                                    \
    Y[idx] = X[tok * row_stride + col];                                       \
}

PATCH_UNPACK(k_patch_unpack_fp32, float)
PATCH_UNPACK(k_patch_unpack_fp16, half)
PATCH_UNPACK(k_patch_unpack_bf16, bfloat)
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
DEF_PSO(pso_pu_fp32, @"k_patch_unpack_fp32")
DEF_PSO(pso_pu_fp16, @"k_patch_unpack_fp16")
DEF_PSO(pso_pu_bf16, @"k_patch_unpack_bf16")
#undef DEF_PSO

} // namespace

void spatial_merge_2x2_forward(const Tensor& X,
                               int N, int C, int H, int W,
                               bool channel_major,
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
    const uint32_t chmajU = channel_major ? 1u : 0u;

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
        [enc setBytes:&chmajU length:sizeof(uint32_t) atIndex:9];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void patch_unpack_forward(const Tensor& tokens,
                          int hp, int wp, int P, int C_total, int C_keep,
                          bool channel_major,
                          Tensor& Y) {
    if (tokens.dtype != Dtype::FP32 && tokens.dtype != Dtype::FP16 &&
        tokens.dtype != Dtype::BF16) {
        throw std::runtime_error("patch_unpack_forward: tokens must be FP32, "
                                 "FP16, or BF16");
    }
    if (hp < 0 || wp < 0 || P <= 0 || C_total <= 0 || C_keep <= 0 ||
        C_keep > C_total) {
        throw std::runtime_error("patch_unpack_forward: bad dimension");
    }
    const int PP   = P * P;
    const int N    = hp * wp;
    if (tokens.rows != N || tokens.cols != PP * C_total) {
        throw std::runtime_error("patch_unpack_forward: tokens shape mismatch");
    }
    const int H    = hp * P;
    const int W    = wp * P;
    const int cols = C_keep * H * W;
    if (Y.rows != 1 || Y.cols != cols || Y.dtype != tokens.dtype) {
        Y.resize(1, cols, tokens.dtype);
    }
    const uint32_t total = (uint32_t)cols;
    if (total == 0) return;
    id<MTLComputePipelineState> pso =
        (tokens.dtype == Dtype::FP16) ? pso_pu_fp16()
      : (tokens.dtype == Dtype::BF16) ? pso_pu_bf16()
      : pso_pu_fp32();

    const uint32_t wpU  = (uint32_t)wp;
    const uint32_t Pu   = (uint32_t)P;
    const uint32_t Ctu  = (uint32_t)C_total;
    const uint32_t Hu   = (uint32_t)H;
    const uint32_t Wu   = (uint32_t)W;
    const uint32_t rsU  = (uint32_t)(PP * C_total);
    const uint32_t chmajU = channel_major ? 1u : 0u;

    id<MTLBuffer> bx = buffer_for(tokens);
    id<MTLBuffer> by = buffer_for(Y);
    NSUInteger ox = buffer_offset_for(tokens);
    NSUInteger oy = buffer_offset_for(Y);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&wpU    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Pu     length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Ctu    length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Hu     length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Wu     length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&rsU    length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&total  length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&chmajU length:sizeof(uint32_t) atIndex:9];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
