// Metal Q8_0 (W8A16-style) dequant + GEMV + batched GEMV. Q8_0 block = 34
// bytes / 32 elements: fp16 d + int8 qs[32]. Decoded value: y = d * qs[i].
// Mirrors src/cuda/q8_0.cu in shape contracts and accumulation order (FP32
// partials, FP16 store at the end). Batched path is a (B, M) GEMV loop —
// there is no fused SIMD-group-matrix variant yet (the CUDA WMMA path has
// no Metal equivalent here), but the per-row kernel is parallelised over
// batch so it matches the CUDA GEMV-loop fallback for B in a wide range.

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr int Q8_BLOCK_ELEMS = 32;
constexpr int Q8_BLOCK_BYTES = 34;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint Q8_BLOCK_ELEMS = 32;
constant uint Q8_BLOCK_BYTES = 34;

// One threadgroup per (row, super_block). 32 threads, one per element.
kernel void k_dequant_q8_0_to_fp16(device const uchar* W                 [[buffer(0)]],
                                   device half*        Wfp16             [[buffer(1)]],
                                   constant uint&      blocks_per_row    [[buffer(2)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint2 li [[thread_position_in_threadgroup]]) {
    uint row = tg.y;
    uint sb  = tg.x;
    uint t   = li.x;
    device const uchar* blk = W + (uint(row) * blocks_per_row + sb) * Q8_BLOCK_BYTES;
    half d_h = *((device const half*)blk);
    float d_f = float(d_h);
    int q = int(((device const char*)blk)[2 + t]);
    Wfp16[uint(row) * (blocks_per_row * Q8_BLOCK_ELEMS) + sb * Q8_BLOCK_ELEMS + t] =
        half(d_f * float(q));
}

// One threadgroup per (b, row); 32 threads (one simdgroup) loop blocks
// along K. simd_sum collapses the 32 partials into y[b, row]. The single-
// vector GEMV path uses B = 1.
kernel void k_linear_q8_0_fp16_gemv(device const uchar* W                 [[buffer(0)]],
                                    device const half*  X                 [[buffer(1)]],
                                    device const half*  bias              [[buffer(2)]],
                                    device half*        Y                 [[buffer(3)]],
                                    constant uint&      K                 [[buffer(4)]],
                                    constant uint&      blocks_per_row    [[buffer(5)]],
                                    constant uint&      out_dim           [[buffer(6)]],
                                    constant uint&      has_bias          [[buffer(7)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint2 li [[thread_position_in_threadgroup]]) {
    uint row = tg.x;
    uint b   = tg.y;
    uint t   = li.x;

    float partial = 0.0f;
    device const uchar* row_base = W + uint(row) * blocks_per_row * Q8_BLOCK_BYTES;
    device const half*  x_base   = X + uint(b)   * K;

    for (uint sb = 0; sb < blocks_per_row; ++sb) {
        device const uchar* blk = row_base + sb * Q8_BLOCK_BYTES;
        half  d_h = *((device const half*)blk);
        float d_f = float(d_h);
        int   q   = int(((device const char*)blk)[2 + t]);
        float xv  = float(x_base[sb * Q8_BLOCK_ELEMS + t]);
        partial += d_f * float(q) * xv;
    }
    float s = simd_sum(partial);
    if (t == 0) {
        float out = s;
        if (has_bias != 0u) out += float(bias[row]);
        Y[uint(b) * out_dim + row] = half(out);
    }
}
)msl";

id<MTLComputePipelineState> pso_dequant() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_dequant_q8_0_to_fp16"); });
    return pso;
}

id<MTLComputePipelineState> pso_gemv() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_linear_q8_0_fp16_gemv"); });
    return pso;
}

void validate_w_q8_0(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q8_0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q8_0");
    }
    if (W.cols % Q8_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 32");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

void launch_gemv(const Tensor& W_q8, const Tensor* bias,
                 const Tensor& X, Tensor& Y,
                 int B, int out_dim, int K) {
    const uint32_t blocks_per_row = static_cast<uint32_t>(K / Q8_BLOCK_ELEMS);
    const uint32_t uK   = static_cast<uint32_t>(K);
    const uint32_t uOut = static_cast<uint32_t>(out_dim);
    const uint32_t uHas = (bias && bias->size() > 0) ? 1u : 0u;

    id<MTLComputePipelineState> pso = pso_gemv();
    id<MTLBuffer> bW = buffer_for(W_q8);
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bY = buffer_for(Y);
    id<MTLBuffer> bB = (uHas != 0u) ? buffer_for(*bias) : bX;
    const NSUInteger oW = buffer_offset_for(W_q8);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oY = buffer_offset_for(Y);
    const NSUInteger oB = (uHas != 0u) ? buffer_offset_for(*bias) : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:oW atIndex:0];
        [enc setBuffer:bX offset:oX atIndex:1];
        [enc setBuffer:bB offset:oB atIndex:2];
        [enc setBuffer:bY offset:oY atIndex:3];
        [enc setBytes:&uK             length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&blocks_per_row length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&uOut           length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&uHas           length:sizeof(uint32_t) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(out_dim),
                                              static_cast<NSUInteger>(B), 1)
            threadsPerThreadgroup:MTLSizeMake(Q8_BLOCK_ELEMS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    validate_w_q8_0(W_q8, "dequant_q8_0_to_fp16");
    const int rows = W_q8.rows;
    const int K    = W_q8.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const uint32_t blocks_per_row = static_cast<uint32_t>(K / Q8_BLOCK_ELEMS);
    if (rows == 0 || blocks_per_row == 0) return;

    id<MTLComputePipelineState> pso = pso_dequant();
    id<MTLBuffer> bW = buffer_for(W_q8);
    id<MTLBuffer> bY = buffer_for(W_fp16);
    const NSUInteger oW = buffer_offset_for(W_q8);
    const NSUInteger oY = buffer_offset_for(W_fp16);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:oW atIndex:0];
        [enc setBuffer:bY offset:oY atIndex:1];
        [enc setBytes:&blocks_per_row length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(blocks_per_row,
                                              static_cast<NSUInteger>(rows), 1)
            threadsPerThreadgroup:MTLSizeMake(Q8_BLOCK_ELEMS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y) {
    validate_w_q8_0(W_q8, "linear_forward_q8_0_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x must be FP16");
    }
    const int out = W_q8.rows;
    const int K   = W_q8.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    // x is (in, 1) but the kernel reads it as a flat K-element vector; b=1.
    launch_gemv(W_q8, bias, x, y, /*B*/1, out, K);
}

void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q8_0(W_q8, "linear_forward_batched_q8_0_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q8.rows;
    if (W_q8.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q8_0_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q8_0_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q8_0_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    launch_gemv(W_q8, bias, X_BD, Y_BD, B, out, K);
}

} // namespace brotensor::detail::metal
