// Metal Q4_K (W4A16) dequant + GEMV + batched GEMV. Q4_K block = 144 bytes
// / 256 elements: fp16 d, fp16 dmin, uint8 scales[12] (eight packed 6-bit
// (sc,m) pairs), uint8 qs[128] (256 nibbles, 8 sub-blocks of 32). Decoded
// value: y = d * sc[is] * nibble - dmin * m[is]. Mirrors src/cuda/q4k.cu in
// contract and accumulation (FP32 partials, FP16 store). 256 threads per
// threadgroup; cross-simdgroup reduction via threadgroup scratch.

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

constexpr int Q4K_BLOCK_ELEMS = 256;
constexpr int Q4K_BLOCK_BYTES = 144;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant uint Q4K_BLOCK_ELEMS = 256;
constant uint Q4K_BLOCK_BYTES = 144;

// Recovers the j-th 6-bit sub-scale `sc` and sub-min `m` from the 12-byte
// packed scales array. j in [0, 8). Matches q4k::unpack_sc_m in the CUDA
// header (q4k_internal.cuh).
inline void unpack_sc_m(uint j, threadgroup const uchar* s,
                        thread uchar& sc, thread uchar& m) {
    if (j < 4u) {
        sc = s[j]     & 0x3Fu;
        m  = s[j + 4u] & 0x3Fu;
    } else {
        sc = (s[j + 4u] & 0x0Fu) | ((s[j - 4u] >> 6) << 4);
        m  = (s[j + 4u] >> 4)    | ((s[j]      >> 6) << 4);
    }
}

// 256-thread block reduction. Uses simd_sum within each 32-wide simdgroup
// (8 per threadgroup), then a final simd_sum over the 8 partials in lane 0
// of simdgroup 0. scratch must hold at least 8 floats.
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

// One threadgroup per (row, super_block). 256 threads, one per element.
kernel void k_dequant_q4k_to_fp16(device const uchar* W                 [[buffer(0)]],
                                  device half*        Wfp16             [[buffer(1)]],
                                  constant uint&      blocks_per_row    [[buffer(2)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint2 li [[thread_position_in_threadgroup]]) {
    uint row = tg.y;
    uint sb  = tg.x;
    uint t   = li.x;
    threadgroup uchar W_smem[Q4K_BLOCK_BYTES];
    threadgroup float sc_f[8];
    threadgroup float m_f [8];
    threadgroup float d_f;
    threadgroup float dmin_f;

    device const uchar* blk = W + (uint(row) * blocks_per_row + sb) * Q4K_BLOCK_BYTES;
    if (t < Q4K_BLOCK_BYTES) W_smem[t] = blk[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0u) {
        d_f    = float(*((threadgroup const half*)W_smem));
        dmin_f = float(*((threadgroup const half*)(W_smem + 2)));
    }
    if (t < 8u) {
        uchar sc, m;
        unpack_sc_m(t, W_smem + 4, sc, m);
        sc_f[t] = float(sc);
        m_f [t] = float(m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint is   = t >> 5;
    uint l    = t & 31u;
    uint pair = is >> 1;
    uchar qb  = W_smem[16u + pair * 32u + l];
    int nib   = ((is & 1u) != 0u) ? int(qb >> 4) : int(qb & 0x0Fu);
    float y   = d_f * sc_f[is] * float(nib) - dmin_f * m_f[is];

    Wfp16[uint(row) * (blocks_per_row * Q4K_BLOCK_ELEMS) + sb * Q4K_BLOCK_ELEMS + t]
        = half(y);
}

// One threadgroup per (b, row); 256 threads loop super-blocks along K.
kernel void k_linear_q4k_fp16_gemv(device const uchar* W                 [[buffer(0)]],
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

    threadgroup uchar W_smem[Q4K_BLOCK_BYTES];
    threadgroup half  X_smem[Q4K_BLOCK_ELEMS];
    threadgroup float sc_f[8];
    threadgroup float m_f [8];
    threadgroup float d_f;
    threadgroup float dmin_f;
    threadgroup float red_scratch[8];

    uint is   = t >> 5;
    uint l    = t & 31u;
    uint pair = is >> 1;

    float partial = 0.0f;

    device const uchar* row_base = W + uint(row) * blocks_per_row * Q4K_BLOCK_BYTES;
    device const half*  x_base   = X + uint(b)   * K;

    for (uint sb = 0; sb < blocks_per_row; ++sb) {
        device const uchar* blk = row_base + sb * Q4K_BLOCK_BYTES;
        if (t < Q4K_BLOCK_BYTES) W_smem[t] = blk[t];
        X_smem[t] = x_base[sb * Q4K_BLOCK_ELEMS + t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t == 0u) {
            d_f    = float(*((threadgroup const half*)W_smem));
            dmin_f = float(*((threadgroup const half*)(W_smem + 2)));
        }
        if (t < 8u) {
            uchar sc, m;
            unpack_sc_m(t, W_smem + 4, sc, m);
            sc_f[t] = float(sc);
            m_f [t] = float(m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uchar qb = W_smem[16u + pair * 32u + l];
        int nib  = ((is & 1u) != 0u) ? int(qb >> 4) : int(qb & 0x0Fu);
        float w  = d_f * sc_f[is] * float(nib) - dmin_f * m_f[is];
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
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_dequant_q4k_to_fp16"); });
    return pso;
}

id<MTLComputePipelineState> pso_gemv() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_linear_q4k_fp16_gemv"); });
    return pso;
}

void validate_w_q4k(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q4_K) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q4_K");
    }
    if (W.cols % Q4K_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 256");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

void launch_gemv(const Tensor& W_q4k, const Tensor* bias,
                 const Tensor& X, Tensor& Y,
                 int B, int out_dim, int K) {
    const uint32_t blocks_per_row = static_cast<uint32_t>(K / Q4K_BLOCK_ELEMS);
    const uint32_t uK   = static_cast<uint32_t>(K);
    const uint32_t uOut = static_cast<uint32_t>(out_dim);
    const uint32_t uHas = (bias && bias->size() > 0) ? 1u : 0u;

    id<MTLComputePipelineState> pso = pso_gemv();
    id<MTLBuffer> bW = buffer_for(W_q4k);
    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bY = buffer_for(Y);
    id<MTLBuffer> bB = (uHas != 0u) ? buffer_for(*bias) : bX;
    const NSUInteger oW = buffer_offset_for(W_q4k);
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
            threadsPerThreadgroup:MTLSizeMake(Q4K_BLOCK_ELEMS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16) {
    validate_w_q4k(W_q4k, "dequant_q4k_to_fp16");
    const int rows = W_q4k.rows;
    const int K    = W_q4k.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const uint32_t blocks_per_row = static_cast<uint32_t>(K / Q4K_BLOCK_ELEMS);
    if (rows == 0 || blocks_per_row == 0) return;

    id<MTLComputePipelineState> pso = pso_dequant();
    id<MTLBuffer> bW = buffer_for(W_q4k);
    id<MTLBuffer> bY = buffer_for(W_fp16);
    const NSUInteger oW = buffer_offset_for(W_q4k);
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
            threadsPerThreadgroup:MTLSizeMake(Q4K_BLOCK_ELEMS, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    validate_w_q4k(W_q4k, "linear_forward_q4k_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: x must be FP16");
    }
    const int out = W_q4k.rows;
    const int K   = W_q4k.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q4k_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q4k_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    launch_gemv(W_q4k, bias, x, y, /*B*/1, out, K);
}

void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q4k(W_q4k, "linear_forward_batched_q4k_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q4k.rows;
    if (W_q4k.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q4k_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q4k_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q4k_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    launch_gemv(W_q4k, bias, X_BD, Y_BD, B, out, K);
}

} // namespace brotensor::detail::metal
