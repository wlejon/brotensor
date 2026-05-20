// RoPE forward + backward (Metal). One thread per (row, head, pair).

#include <brotensor/runtime.h>

#include <stdexcept>

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

static inline float rope_theta(uint pair_i, uint head_dim, float base) {
    return exp(-float(2u * pair_i) / float(head_dim) * log(base));
}

kernel void k_rope_fw_fp32(device const float* X [[buffer(0)]],
                           device float*       Y [[buffer(1)]],
                           constant uint& L          [[buffer(2)]],
                           constant uint& num_heads  [[buffer(3)]],
                           constant uint& head_dim   [[buffer(4)]],
                           constant int&  seq_offset [[buffer(5)]],
                           constant float& base      [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    uint half_d = head_dim / 2u;
    uint total  = L * num_heads * half_d;
    if (gid >= total) return;
    uint i    = gid % half_d;
    uint rest = gid / half_d;
    uint h    = rest % num_heads;
    uint row  = rest / num_heads;
    int  pos  = int(row) + seq_offset;
    float theta = float(pos) * rope_theta(i, head_dim, base);
    float c = cos(theta), s = sin(theta);
    uint D = num_heads * head_dim;
    uint base_off = row * D + h * head_dim;
    float x0 = X[base_off + 2u * i];
    float x1 = X[base_off + 2u * i + 1u];
    Y[base_off + 2u * i]      = x0 * c - x1 * s;
    Y[base_off + 2u * i + 1u] = x0 * s + x1 * c;
}

kernel void k_rope_fw_fp16(device const half* X [[buffer(0)]],
                           device half*       Y [[buffer(1)]],
                           constant uint& L          [[buffer(2)]],
                           constant uint& num_heads  [[buffer(3)]],
                           constant uint& head_dim   [[buffer(4)]],
                           constant int&  seq_offset [[buffer(5)]],
                           constant float& base      [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    uint half_d = head_dim / 2u;
    uint total  = L * num_heads * half_d;
    if (gid >= total) return;
    uint i    = gid % half_d;
    uint rest = gid / half_d;
    uint h    = rest % num_heads;
    uint row  = rest / num_heads;
    int  pos  = int(row) + seq_offset;
    float theta = float(pos) * rope_theta(i, head_dim, base);
    float c = cos(theta), s = sin(theta);
    uint D = num_heads * head_dim;
    uint base_off = row * D + h * head_dim;
    float x0 = float(X[base_off + 2u * i]);
    float x1 = float(X[base_off + 2u * i + 1u]);
    Y[base_off + 2u * i]      = half(x0 * c - x1 * s);
    Y[base_off + 2u * i + 1u] = half(x0 * s + x1 * c);
}

kernel void k_rope_bw_fp32(device const float* dY [[buffer(0)]],
                           device float*       dX [[buffer(1)]],
                           constant uint& L          [[buffer(2)]],
                           constant uint& num_heads  [[buffer(3)]],
                           constant uint& head_dim   [[buffer(4)]],
                           constant int&  seq_offset [[buffer(5)]],
                           constant float& base      [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    uint half_d = head_dim / 2u;
    uint total  = L * num_heads * half_d;
    if (gid >= total) return;
    uint i    = gid % half_d;
    uint rest = gid / half_d;
    uint h    = rest % num_heads;
    uint row  = rest / num_heads;
    int  pos  = int(row) + seq_offset;
    float theta = float(pos) * rope_theta(i, head_dim, base);
    float c = cos(theta), s = sin(theta);
    uint D = num_heads * head_dim;
    uint base_off = row * D + h * head_dim;
    float dy0 = dY[base_off + 2u * i];
    float dy1 = dY[base_off + 2u * i + 1u];
    dX[base_off + 2u * i]      =  dy0 * c + dy1 * s;
    dX[base_off + 2u * i + 1u] = -dy0 * s + dy1 * c;
}

kernel void k_rope_bw_fp16(device const half* dY [[buffer(0)]],
                           device half*       dX [[buffer(1)]],
                           constant uint& L          [[buffer(2)]],
                           constant uint& num_heads  [[buffer(3)]],
                           constant uint& head_dim   [[buffer(4)]],
                           constant int&  seq_offset [[buffer(5)]],
                           constant float& base      [[buffer(6)]],
                           uint gid [[thread_position_in_grid]]) {
    uint half_d = head_dim / 2u;
    uint total  = L * num_heads * half_d;
    if (gid >= total) return;
    uint i    = gid % half_d;
    uint rest = gid / half_d;
    uint h    = rest % num_heads;
    uint row  = rest / num_heads;
    int  pos  = int(row) + seq_offset;
    float theta = float(pos) * rope_theta(i, head_dim, base);
    float c = cos(theta), s = sin(theta);
    uint D = num_heads * head_dim;
    uint base_off = row * D + h * head_dim;
    float dy0 = float(dY[base_off + 2u * i]);
    float dy1 = float(dY[base_off + 2u * i + 1u]);
    dX[base_off + 2u * i]      = half( dy0 * c + dy1 * s);
    dX[base_off + 2u * i + 1u] = half(-dy0 * s + dy1 * c);
}
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_fw_fp32, @"k_rope_fw_fp32")
DEF_PSO(pso_fw_fp16, @"k_rope_fw_fp16")
DEF_PSO(pso_bw_fp32, @"k_rope_bw_fp32")
DEF_PSO(pso_bw_fp16, @"k_rope_bw_fp16")
#undef DEF_PSO

void launch(id<MTLComputePipelineState> pso, NSUInteger total,
            id<MTLBuffer> bin, NSUInteger oin,
            id<MTLBuffer> bout, NSUInteger oout,
            uint32_t L, uint32_t num_heads, uint32_t head_dim,
            int32_t seq_offset, float base) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bin  offset:oin  atIndex:0];
        [enc setBuffer:bout offset:oout atIndex:1];
        [enc setBytes:&L          length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&num_heads  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&head_dim   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&seq_offset length:sizeof(int32_t)  atIndex:5];
        [enc setBytes:&base       length:sizeof(float)    atIndex:6];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace

void rope_forward(const Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_forward_gpu: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_forward_gpu: num_heads must be positive");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_forward_gpu: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(L) * num_heads * (head_dim / 2);
    if (total == 0) return;
    id<MTLComputePipelineState> pso = (X.dtype == Dtype::FP16) ? pso_fw_fp16() : pso_fw_fp32();
    launch(pso, total, buffer_for(X), buffer_offset_for(X),
           buffer_for(Y), buffer_offset_for(Y),
           (uint32_t)L, (uint32_t)num_heads, (uint32_t)head_dim,
           (int32_t)seq_offset, theta_base);
}

void rope_backward(const Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, Tensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_backward_gpu: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_backward_gpu: num_heads must be positive");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_backward_gpu: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(L) * num_heads * (head_dim / 2);
    if (total == 0) return;
    id<MTLComputePipelineState> pso = (dY.dtype == Dtype::FP16) ? pso_bw_fp16() : pso_bw_fp32();
    launch(pso, total, buffer_for(dY), buffer_offset_for(dY),
           buffer_for(dX), buffer_offset_for(dX),
           (uint32_t)L, (uint32_t)num_heads, (uint32_t)head_dim,
           (int32_t)seq_offset, theta_base);
}

} // namespace brotensor::detail::metal
