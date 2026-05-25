// Metal Q6_K (W6A16) dequant + GEMV + batched GEMV. Q6_K block = 210 bytes
// / 256 elements: uint8 ql[128], uint8 qh[64], int8 sc[16], fp16 d (offset
// 208). Decoded value: y = d * sc[sb] * (val6 - 32), where val6 = raw4 |
// (high2 << 4), see decode_element below (mirrors q6k_internal.cuh).
// Mirrors src/cuda/q6k.cu in contract and accumulation (FP32 partials, FP16
// store). 256 threads per threadgroup; cross-simdgroup reduction via
// threadgroup scratch.

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

constexpr int Q6K_BLOCK_ELEMS = 256;
constexpr int Q6K_BLOCK_BYTES = 210;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BLOCK_ELEMS = 256;
constant uint Q6K_BLOCK_BYTES = 210;
constant uint Q6K_QL_OFFSET   = 0;
constant uint Q6K_QH_OFFSET   = 128;
constant uint Q6K_SC_OFFSET   = 192;
constant uint Q6K_D_OFFSET    = 208;

// Recover the signed 6-bit value for element e in [0, 256) and the
// sub-block index (0..15) that selects the scale.
inline void decode_element(uint e,
                           threadgroup const uchar* ql,
                           threadgroup const uchar* qh,
                           thread int& sb_out,
                           thread int& val6_out) {
    uint group = e >> 7;             // 0..1
    uint local = e - (group << 7);   // 0..127
    uint quad  = local >> 5;         // 0..3
    uint l     = local - (quad << 5); // 0..31

    int sb = int((group << 3) + (quad << 1) + (l >> 4));   // 0..15
    uchar ql_b = ql[group * 64u + (quad & 1u) * 32u + l];
    uchar qh_b = qh[group * 32u + l];
    int raw4  = (quad < 2u) ? int(ql_b & 0x0Fu) : int(ql_b >> 4);
    int high2 = int((qh_b >> (quad * 2u)) & 0x03u);
    int val6  = (raw4 | (high2 << 4)) - 32;

    sb_out   = sb;
    val6_out = val6;
}

inline float block_reduce_sum_256(float v,
                                  threadgroup float* scratch,
                                  uint lane,
                                  uint sg_idx) {
    float s = simd_sum(v);
    if (lane == 0u) scratch[sg_idx] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0u) {
        float w = (lane < 8u) ? scratch[lane] : 0.0f;
        w = simd_sum(w);
        if (lane == 0u) scratch[0] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

kernel void k_dequant_q6k_to_fp16(device const uchar* W                 [[buffer(0)]],
                                  device half*        Wfp16             [[buffer(1)]],
                                  constant uint&      blocks_per_row    [[buffer(2)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint2 li [[thread_position_in_threadgroup]]) {
    uint row = tg.y;
    uint sb_idx = tg.x;
    uint t      = li.x;
    threadgroup uchar W_smem[Q6K_BLOCK_BYTES];
    threadgroup float d_f;

    device const uchar* blk = W + (uint(row) * blocks_per_row + sb_idx) * Q6K_BLOCK_BYTES;
    if (t < Q6K_BLOCK_BYTES) W_smem[t] = blk[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0u) {
        d_f = float(*((threadgroup const half*)(W_smem + Q6K_D_OFFSET)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int sb, val6;
    decode_element(t, W_smem + Q6K_QL_OFFSET, W_smem + Q6K_QH_OFFSET, sb, val6);
    int scv = int(((threadgroup const char*)W_smem)[Q6K_SC_OFFSET + uint(sb)]);
    float y = d_f * float(scv) * float(val6);

    Wfp16[uint(row) * (blocks_per_row * Q6K_BLOCK_ELEMS) + sb_idx * Q6K_BLOCK_ELEMS + t]
        = half(y);
}

kernel void k_linear_q6k_fp16_gemv(device const uchar* W                 [[buffer(0)]],
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
    // Apple-silicon simdgroup width is 32; t is laid out (256, 1, 1) so the
    // simdgroup index and lane can be recovered directly without needing the
    // dedicated attributes (which would conflict with the uint2 li above).
    uint lane   = t & 31u;
    uint sg_idx = t >> 5;

    threadgroup uchar W_smem[Q6K_BLOCK_BYTES];
    threadgroup half  X_smem[Q6K_BLOCK_ELEMS];
    threadgroup float d_f;
    threadgroup float red_scratch[8];

    float partial = 0.0f;
    device const uchar* row_base = W + uint(row) * blocks_per_row * Q6K_BLOCK_BYTES;
    device const half*  x_base   = X + uint(b)   * K;

    for (uint sb_idx = 0; sb_idx < blocks_per_row; ++sb_idx) {
        device const uchar* blk = row_base + sb_idx * Q6K_BLOCK_BYTES;
        if (t < Q6K_BLOCK_BYTES) W_smem[t] = blk[t];
        X_smem[t] = x_base[sb_idx * Q6K_BLOCK_ELEMS + t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t == 0u) {
            d_f = float(*((threadgroup const half*)(W_smem + Q6K_D_OFFSET)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int sb, val6;
        decode_element(t, W_smem + Q6K_QL_OFFSET, W_smem + Q6K_QH_OFFSET, sb, val6);
        int scv = int(((threadgroup const char*)W_smem)[Q6K_SC_OFFSET + uint(sb)]);
        float w = d_f * float(scv) * float(val6);
        partial += w * float(X_smem[t]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = block_reduce_sum_256(partial, red_scratch, lane, sg_idx);
    if (t == 0u) {
        float out = sum;
        if (has_bias != 0u) out += float(bias[row]);
        Y[uint(b) * out_dim + row] = half(out);
    }
}
)msl";

id<MTLComputePipelineState> pso_dequant() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_dequant_q6k_to_fp16"); });
    return pso;
}

id<MTLComputePipelineState> pso_gemv() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_linear_q6k_fp16_gemv"); });
    return pso;
}

void validate_w_q6k(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q6_K) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q6_K");
    }
    if (W.cols % Q6K_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 256");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

void launch_gemv(const Tensor& W_q6k, const Tensor* bias,
                 const Tensor& X, Tensor& Y,
                 int B, int out_dim, int K) {
    const uint32_t blocks_per_row = static_cast<uint32_t>(K / Q6K_BLOCK_ELEMS);
    const uint32_t uK   = static_cast<uint32_t>(K);
    const uint32_t uOut = static_cast<uint32_t>(out_dim);
    const uint32_t uHas = (bias && bias->size() > 0) ? 1u : 0u;

    id<MTLComputePipelineState> pso = pso_gemv();
    id<MTLBuffer> bW = buffer_for(W_q6k);
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bY = buffer_for(Y);
    id<MTLBuffer> bB = (uHas != 0u) ? buffer_for(*bias) : bX;
    const NSUInteger oW = buffer_offset_for(W_q6k);
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
            threadsPerThreadgroup:MTLSizeMake(Q6K_BLOCK_ELEMS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16) {
    validate_w_q6k(W_q6k, "dequant_q6k_to_fp16");
    const int rows = W_q6k.rows;
    const int K    = W_q6k.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const uint32_t blocks_per_row = static_cast<uint32_t>(K / Q6K_BLOCK_ELEMS);
    if (rows == 0 || blocks_per_row == 0) return;

    id<MTLComputePipelineState> pso = pso_dequant();
    id<MTLBuffer> bW = buffer_for(W_q6k);
    id<MTLBuffer> bY = buffer_for(W_fp16);
    const NSUInteger oW = buffer_offset_for(W_q6k);
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
            threadsPerThreadgroup:MTLSizeMake(Q6K_BLOCK_ELEMS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    validate_w_q6k(W_q6k, "linear_forward_q6k_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: x must be FP16");
    }
    const int out = W_q6k.rows;
    const int K   = W_q6k.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q6k_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q6k_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    launch_gemv(W_q6k, bias, x, y, /*B*/1, out, K);
}

void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q6k(W_q6k, "linear_forward_batched_q6k_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q6k.rows;
    if (W_q6k.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q6k_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q6k_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q6k_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    launch_gemv(W_q6k, bias, X_BD, Y_BD, B, out, K);
}

} // namespace brotensor::detail::metal
