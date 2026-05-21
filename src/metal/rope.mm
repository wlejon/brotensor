// RoPE forward + backward (Metal). One thread per (row, head, pair).

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

kernel void k_rope_fw_bf16(device const bfloat* X [[buffer(0)]],
                           device bfloat*       Y [[buffer(1)]],
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
    Y[base_off + 2u * i]      = bfloat(x0 * c - x1 * s);
    Y[base_off + 2u * i + 1u] = bfloat(x0 * s + x1 * c);
}

kernel void k_rope_bw_bf16(device const bfloat* dY [[buffer(0)]],
                           device bfloat*       dX [[buffer(1)]],
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
    dX[base_off + 2u * i]      = bfloat( dy0 * c + dy1 * s);
    dX[base_off + 2u * i + 1u] = bfloat(-dy0 * s + dy1 * c);
}

// ─── rope_apply: RoPE with explicit cos/sin tables ─────────────────────────
//
// cos_tbl / sin_tbl are (L, head_dim/2) FP32: one angle per (row, pair),
// shared across heads. X / Y are typed (FP32 / FP16 / BF16); math in FP32.

#define ROPE_APPLY_FW(NAME, T)                                                \
kernel void NAME(device const T*     X       [[buffer(0)]],                   \
                 device const float* cos_tbl [[buffer(1)]],                   \
                 device const float* sin_tbl [[buffer(2)]],                   \
                 device T*           Y       [[buffer(3)]],                   \
                 constant uint& L         [[buffer(4)]],                      \
                 constant uint& num_heads [[buffer(5)]],                      \
                 constant uint& head_dim  [[buffer(6)]],                      \
                 uint gid [[thread_position_in_grid]]) {                      \
    uint half_d = head_dim / 2u;                                              \
    uint total  = L * num_heads * half_d;                                     \
    if (gid >= total) return;                                                 \
    uint i    = gid % half_d;                                                 \
    uint rest = gid / half_d;                                                 \
    uint h    = rest % num_heads;                                             \
    uint row  = rest / num_heads;                                             \
    float c = cos_tbl[row * half_d + i];                                      \
    float s = sin_tbl[row * half_d + i];                                      \
    uint D = num_heads * head_dim;                                            \
    uint base_off = row * D + h * head_dim;                                   \
    float x0 = float(X[base_off + 2u * i]);                                   \
    float x1 = float(X[base_off + 2u * i + 1u]);                              \
    Y[base_off + 2u * i]      = T(x0 * c - x1 * s);                           \
    Y[base_off + 2u * i + 1u] = T(x0 * s + x1 * c);                           \
}

#define ROPE_APPLY_BW(NAME, T)                                                \
kernel void NAME(device const T*     dY      [[buffer(0)]],                   \
                 device const float* cos_tbl [[buffer(1)]],                   \
                 device const float* sin_tbl [[buffer(2)]],                   \
                 device T*           dX      [[buffer(3)]],                   \
                 constant uint& L         [[buffer(4)]],                      \
                 constant uint& num_heads [[buffer(5)]],                      \
                 constant uint& head_dim  [[buffer(6)]],                      \
                 uint gid [[thread_position_in_grid]]) {                      \
    uint half_d = head_dim / 2u;                                              \
    uint total  = L * num_heads * half_d;                                     \
    if (gid >= total) return;                                                 \
    uint i    = gid % half_d;                                                 \
    uint rest = gid / half_d;                                                 \
    uint h    = rest % num_heads;                                             \
    uint row  = rest / num_heads;                                             \
    float c = cos_tbl[row * half_d + i];                                      \
    float s = sin_tbl[row * half_d + i];                                      \
    uint D = num_heads * head_dim;                                            \
    uint base_off = row * D + h * head_dim;                                   \
    float dy0 = float(dY[base_off + 2u * i]);                                 \
    float dy1 = float(dY[base_off + 2u * i + 1u]);                            \
    dX[base_off + 2u * i]      = T( dy0 * c + dy1 * s);                       \
    dX[base_off + 2u * i + 1u] = T(-dy0 * s + dy1 * c);                       \
}

ROPE_APPLY_FW(k_rope_apply_fw_fp32, float)
ROPE_APPLY_FW(k_rope_apply_fw_fp16, half)
ROPE_APPLY_FW(k_rope_apply_fw_bf16, bfloat)
ROPE_APPLY_BW(k_rope_apply_bw_fp32, float)
ROPE_APPLY_BW(k_rope_apply_bw_fp16, half)
ROPE_APPLY_BW(k_rope_apply_bw_bf16, bfloat)
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
DEF_PSO(pso_fw_bf16, @"k_rope_fw_bf16")
DEF_PSO(pso_bw_fp32, @"k_rope_bw_fp32")
DEF_PSO(pso_bw_fp16, @"k_rope_bw_fp16")
DEF_PSO(pso_bw_bf16, @"k_rope_bw_bf16")
DEF_PSO(pso_apply_fw_fp32, @"k_rope_apply_fw_fp32")
DEF_PSO(pso_apply_fw_fp16, @"k_rope_apply_fw_fp16")
DEF_PSO(pso_apply_fw_bf16, @"k_rope_apply_fw_bf16")
DEF_PSO(pso_apply_bw_fp32, @"k_rope_apply_bw_fp32")
DEF_PSO(pso_apply_bw_fp16, @"k_rope_apply_bw_fp16")
DEF_PSO(pso_apply_bw_bf16, @"k_rope_apply_bw_bf16")
#undef DEF_PSO

void check_rope_tables(const Tensor& cos_tbl, const Tensor& sin_tbl,
                       const char* op, int L, int half) {
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) +
                                 ": cos_tbl / sin_tbl must be FP32");
    }
    if (cos_tbl.size() != L * half || sin_tbl.size() != L * half) {
        throw std::runtime_error(std::string(op) +
                                 ": cos_tbl / sin_tbl must each be (L, head_dim/2)");
    }
}

// One thread per (row, head, pair); cos/sin tables fed as extra buffers.
void launch_apply(id<MTLComputePipelineState> pso, NSUInteger total,
                  id<MTLBuffer> bin, NSUInteger oin,
                  id<MTLBuffer> bcos, NSUInteger ocos,
                  id<MTLBuffer> bsin, NSUInteger osin,
                  id<MTLBuffer> bout, NSUInteger oout,
                  uint32_t L, uint32_t num_heads, uint32_t head_dim) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bin  offset:oin  atIndex:0];
        [enc setBuffer:bcos offset:ocos atIndex:1];
        [enc setBuffer:bsin offset:osin atIndex:2];
        [enc setBuffer:bout offset:oout atIndex:3];
        [enc setBytes:&L         length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&head_dim  length:sizeof(uint32_t) atIndex:6];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

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
        ::brotensor::metal_impl::submit(cmd);
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
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_fw_fp16()
      : (X.dtype == Dtype::BF16) ? pso_fw_bf16()
      : pso_fw_fp32();
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
    id<MTLComputePipelineState> pso =
        (dY.dtype == Dtype::FP16) ? pso_bw_fp16()
      : (dY.dtype == Dtype::BF16) ? pso_bw_bf16()
      : pso_bw_fp32();
    launch(pso, total, buffer_for(dY), buffer_offset_for(dY),
           buffer_for(dX), buffer_offset_for(dX),
           (uint32_t)L, (uint32_t)num_heads, (uint32_t)head_dim,
           (int32_t)seq_offset, theta_base);
}

void rope_apply(const Tensor& X, const Tensor& cos_tbl, const Tensor& sin_tbl,
                int head_dim, int num_heads, Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("rope_apply: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    const int half = head_dim / 2;
    check_rope_tables(cos_tbl, sin_tbl, "rope_apply", L, half);
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(L) * num_heads * half;
    if (total == 0) return;
    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_apply_fw_fp16()
      : (X.dtype == Dtype::BF16) ? pso_apply_fw_bf16()
      : pso_apply_fw_fp32();
    launch_apply(pso, total, buffer_for(X), buffer_offset_for(X),
                 buffer_for(cos_tbl), buffer_offset_for(cos_tbl),
                 buffer_for(sin_tbl), buffer_offset_for(sin_tbl),
                 buffer_for(Y), buffer_offset_for(Y),
                 (uint32_t)L, (uint32_t)num_heads, (uint32_t)head_dim);
}

void rope_apply_backward(const Tensor& dY, const Tensor& cos_tbl,
                         const Tensor& sin_tbl, int head_dim, int num_heads,
                         Tensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_backward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_backward: num_heads must be positive");
    }
    if (dY.dtype != Dtype::FP32 && dY.dtype != Dtype::FP16 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("rope_apply_backward: dY must be FP32, FP16, or BF16");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    const int half = head_dim / 2;
    check_rope_tables(cos_tbl, sin_tbl, "rope_apply_backward", L, half);
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const NSUInteger total = static_cast<NSUInteger>(L) * num_heads * half;
    if (total == 0) return;
    id<MTLComputePipelineState> pso =
        (dY.dtype == Dtype::FP16) ? pso_apply_bw_fp16()
      : (dY.dtype == Dtype::BF16) ? pso_apply_bw_bf16()
      : pso_apply_bw_fp32();
    launch_apply(pso, total, buffer_for(dY), buffer_offset_for(dY),
                 buffer_for(cos_tbl), buffer_offset_for(cos_tbl),
                 buffer_for(sin_tbl), buffer_offset_for(sin_tbl),
                 buffer_for(dX), buffer_offset_for(dX),
                 (uint32_t)L, (uint32_t)num_heads, (uint32_t)head_dim);
}

} // namespace brotensor::detail::metal
