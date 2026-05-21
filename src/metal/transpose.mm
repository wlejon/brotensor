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

kernel void k_nchw_to_seq_fp32(device const float* X [[buffer(0)]],
                               device float*       Y [[buffer(1)]],
                               constant uint& N     [[buffer(2)]],
                               constant uint& C     [[buffer(3)]],
                               constant uint& HW    [[buffer(4)]],
                               constant uint& total [[buffer(5)]],
                               uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint c = idx % C;
    uint t = idx / C;
    uint p = t % HW;
    uint n = t / HW;
    Y[idx] = X[(n * C + c) * HW + p];
}

kernel void k_nchw_to_seq_fp16(device const half*  X [[buffer(0)]],
                               device half*        Y [[buffer(1)]],
                               constant uint& N     [[buffer(2)]],
                               constant uint& C     [[buffer(3)]],
                               constant uint& HW    [[buffer(4)]],
                               constant uint& total [[buffer(5)]],
                               uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint c = idx % C;
    uint t = idx / C;
    uint p = t % HW;
    uint n = t / HW;
    Y[idx] = X[(n * C + c) * HW + p];
}

kernel void k_seq_to_nchw_fp32(device const float* X [[buffer(0)]],
                               device float*       Y [[buffer(1)]],
                               constant uint& N     [[buffer(2)]],
                               constant uint& C     [[buffer(3)]],
                               constant uint& HW    [[buffer(4)]],
                               constant uint& total [[buffer(5)]],
                               uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint p = idx % HW;
    uint t = idx / HW;
    uint c = t % C;
    uint n = t / C;
    Y[idx] = X[(n * HW + p) * C + c];
}

kernel void k_seq_to_nchw_fp16(device const half*  X [[buffer(0)]],
                               device half*        Y [[buffer(1)]],
                               constant uint& N     [[buffer(2)]],
                               constant uint& C     [[buffer(3)]],
                               constant uint& HW    [[buffer(4)]],
                               constant uint& total [[buffer(5)]],
                               uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint p = idx % HW;
    uint t = idx / HW;
    uint c = t % C;
    uint n = t / C;
    Y[idx] = X[(n * HW + p) * C + c];
}

kernel void k_nchw_to_seq_bf16(device const bfloat* X [[buffer(0)]],
                               device bfloat*        Y [[buffer(1)]],
                               constant uint& N        [[buffer(2)]],
                               constant uint& C        [[buffer(3)]],
                               constant uint& HW       [[buffer(4)]],
                               constant uint& total    [[buffer(5)]],
                               uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint c = idx % C;
    uint t = idx / C;
    uint p = t % HW;
    uint n = t / HW;
    Y[idx] = X[(n * C + c) * HW + p];
}

kernel void k_seq_to_nchw_bf16(device const bfloat* X [[buffer(0)]],
                               device bfloat*        Y [[buffer(1)]],
                               constant uint& N        [[buffer(2)]],
                               constant uint& C        [[buffer(3)]],
                               constant uint& HW       [[buffer(4)]],
                               constant uint& total    [[buffer(5)]],
                               uint idx [[thread_position_in_grid]]) {
    if (idx >= total) return;
    uint p = idx % HW;
    uint t = idx / HW;
    uint c = t % C;
    uint n = t / C;
    Y[idx] = X[(n * HW + p) * C + c];
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_nchw_to_seq_fp32, @"k_nchw_to_seq_fp32")
DEF_PSO(pso_nchw_to_seq_fp16, @"k_nchw_to_seq_fp16")
DEF_PSO(pso_nchw_to_seq_bf16, @"k_nchw_to_seq_bf16")
DEF_PSO(pso_seq_to_nchw_fp32, @"k_seq_to_nchw_fp32")
DEF_PSO(pso_seq_to_nchw_fp16, @"k_seq_to_nchw_fp16")
DEF_PSO(pso_seq_to_nchw_bf16, @"k_seq_to_nchw_bf16")
#undef DEF_PSO

void launch_transpose(id<MTLComputePipelineState> pso,
                      const Tensor& X, Tensor& Y,
                      int N, int C, int HW, uint32_t total) {
    if (total == 0) return;
    const uint32_t Nu = static_cast<uint32_t>(N);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t HWu = static_cast<uint32_t>(HW);
    id<MTLBuffer> bx = buffer_for(X);
    id<MTLBuffer> by = buffer_for(Y);
    const NSUInteger ox = buffer_offset_for(X);
    const NSUInteger oy = buffer_offset_for(Y);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBytes:&Nu    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Cu    length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&HWu   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:5];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void check_dims(const char* op, int N, int C, int H, int W) {
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error(std::string(op) + ": negative dimension");
    }
}

} // namespace

void nchw_to_sequence(const Tensor& X,
                      int N, int C, int H, int W,
                      Tensor& Y) {
    check_dims("nchw_to_sequence_gpu", N, C, H, W);
    const int HW = H * W;
    const int rows = N * HW;
    if (Y.rows != rows || Y.cols != C || Y.dtype != X.dtype) {
        Y.resize(rows, C, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(rows) * static_cast<uint32_t>(C);
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_nchw_to_seq_fp16()
      : (X.dtype == Dtype::BF16) ? pso_nchw_to_seq_bf16()
      : pso_nchw_to_seq_fp32();
    launch_transpose(pso, X, Y, N, C, HW, total);
}

void sequence_to_nchw(const Tensor& X,
                      int N, int C, int H, int W,
                      Tensor& Y) {
    check_dims("sequence_to_nchw_gpu", N, C, H, W);
    const int HW = H * W;
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const uint32_t total = static_cast<uint32_t>(N) * static_cast<uint32_t>(cols);
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_seq_to_nchw_fp16()
      : (X.dtype == Dtype::BF16) ? pso_seq_to_nchw_bf16()
      : pso_seq_to_nchw_fp32();
    launch_transpose(pso, X, Y, N, C, HW, total);
}

} // namespace brotensor::detail::metal
