// Metal M-RoPE (Qwen2.5-VL / Qwen3-VL).
//
// Three per-axis (t, h, w) cos/sin tables + INT32 position-ID streams.
// head_dim is split into three sub-ranges of widths 2*d_t, 2*d_h, 2*d_w.
// Inline axis dispatch in the kernel: each thread (row, head, pair) figures
// out which axis its pair belongs to and looks up cos/sin at pos_a[row].
//
// CPU exposes pos_t/h/w as host pointers; Metal looks them up in the device
// buffer pool via pool_lookup() — same convention as
// flash_attention_varlen_forward's cu_seqlens.

#include <brotensor/runtime.h>

#include <cstdint>
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

#define MROPE_KERNEL(NAME, T)                                                 \
kernel void NAME(device const T*     X       [[buffer(0)]],                   \
                 device T*           Y       [[buffer(1)]],                   \
                 device const float* cos_t   [[buffer(2)]],                   \
                 device const float* sin_t   [[buffer(3)]],                   \
                 device const float* cos_h   [[buffer(4)]],                   \
                 device const float* sin_h   [[buffer(5)]],                   \
                 device const float* cos_w   [[buffer(6)]],                   \
                 device const float* sin_w   [[buffer(7)]],                   \
                 device const int*   pos_t   [[buffer(8)]],                   \
                 device const int*   pos_h   [[buffer(9)]],                   \
                 device const int*   pos_w   [[buffer(10)]],                  \
                 constant uint& L         [[buffer(11)]],                     \
                 constant uint& num_heads [[buffer(12)]],                     \
                 constant uint& head_dim  [[buffer(13)]],                     \
                 constant uint& d_t       [[buffer(14)]],                     \
                 constant uint& d_h       [[buffer(15)]],                     \
                 constant uint& d_w       [[buffer(16)]],                     \
                 uint gid [[thread_position_in_grid]]) {                      \
    uint half_d = head_dim / 2u;                                              \
    uint total  = L * num_heads * half_d;                                     \
    if (gid >= total) return;                                                 \
    uint i    = gid % half_d;                                                 \
    uint rest = gid / half_d;                                                 \
    uint h    = rest % num_heads;                                             \
    uint row  = rest / num_heads;                                             \
    uint D    = num_heads * head_dim;                                         \
    uint base_off = row * D + h * head_dim;                                   \
    float x0 = float(X[base_off + 2u * i]);                                   \
    float x1 = float(X[base_off + 2u * i + 1u]);                              \
    float c, s;                                                               \
    if (i < d_t) {                                                            \
        uint pos = uint(pos_t[row]);                                          \
        c = cos_t[pos * d_t + i];                                             \
        s = sin_t[pos * d_t + i];                                             \
    } else if (i < d_t + d_h) {                                               \
        uint local = i - d_t;                                                 \
        uint pos = uint(pos_h[row]);                                          \
        c = cos_h[pos * d_h + local];                                         \
        s = sin_h[pos * d_h + local];                                         \
    } else {                                                                  \
        uint local = i - d_t - d_h;                                           \
        uint pos = uint(pos_w[row]);                                          \
        c = cos_w[pos * d_w + local];                                         \
        s = sin_w[pos * d_w + local];                                         \
    }                                                                         \
    Y[base_off + 2u * i]      = T(x0 * c - x1 * s);                           \
    Y[base_off + 2u * i + 1u] = T(x0 * s + x1 * c);                           \
}

MROPE_KERNEL(k_rope_mrope_fp32, float)
MROPE_KERNEL(k_rope_mrope_fp16, half)
MROPE_KERNEL(k_rope_mrope_bf16, bfloat)
)msl";

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_mrope_fp32, @"k_rope_mrope_fp32")
DEF_PSO(pso_mrope_fp16, @"k_rope_mrope_fp16")
DEF_PSO(pso_mrope_bf16, @"k_rope_mrope_bf16")
#undef DEF_PSO

void check_axis_tbl(const Tensor& cos_a, const Tensor& sin_a,
                    const char* axis, int d_a) {
    if (d_a < 0) {
        throw std::runtime_error(std::string("rope_apply_mrope: d_") + axis +
                                 " must be non-negative");
    }
    if (d_a == 0) return;
    if (cos_a.dtype != Dtype::FP32 || sin_a.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis + " must be FP32");
    }
    if (cos_a.cols != d_a || sin_a.cols != d_a) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis +
                                 " must each have cols == d_" + axis);
    }
    if (cos_a.rows != sin_a.rows || cos_a.rows < 1) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis + " row count mismatch");
    }
}

} // namespace

void rope_apply_mrope(const Tensor& X,
                      const Tensor& cos_t, const Tensor& sin_t,
                      const Tensor& cos_h, const Tensor& sin_h,
                      const Tensor& cos_w, const Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h,
                      const int32_t* pos_w,
                      int head_dim, int num_heads,
                      int d_t, int d_h, int d_w,
                      Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_mrope: head_dim must be a "
                                 "positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_mrope: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("rope_apply_mrope: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_mrope: X.cols != "
                                 "num_heads * head_dim");
    }
    if (2 * (d_t + d_h + d_w) != head_dim) {
        throw std::runtime_error("rope_apply_mrope: 2*(d_t + d_h + d_w) "
                                 "must equal head_dim");
    }
    check_axis_tbl(cos_t, sin_t, "t", d_t);
    check_axis_tbl(cos_h, sin_h, "h", d_h);
    check_axis_tbl(cos_w, sin_w, "w", d_w);
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    if (L == 0 || num_heads == 0 || head_dim == 0) return;
    if (d_t > 0 && !pos_t) throw std::runtime_error("rope_apply_mrope: pos_t null");
    if (d_h > 0 && !pos_h) throw std::runtime_error("rope_apply_mrope: pos_h null");
    if (d_w > 0 && !pos_w) throw std::runtime_error("rope_apply_mrope: pos_w null");

    const NSUInteger total = (NSUInteger)L * num_heads * (head_dim / 2);
    if (total == 0) return;

    id<MTLComputePipelineState> pso =
        (X.dtype == Dtype::FP16) ? pso_mrope_fp16()
      : (X.dtype == Dtype::BF16) ? pso_mrope_bf16()
      : pso_mrope_fp32();

    id<MTLBuffer> bX = buffer_for(X); NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bY = buffer_for(Y); NSUInteger oY = buffer_offset_for(Y);
    id<MTLBuffer> bCT = d_t > 0 ? buffer_for(cos_t) : buffer_for(X);
    NSUInteger oCT    = d_t > 0 ? buffer_offset_for(cos_t) : 0;
    id<MTLBuffer> bST = d_t > 0 ? buffer_for(sin_t) : buffer_for(X);
    NSUInteger oST    = d_t > 0 ? buffer_offset_for(sin_t) : 0;
    id<MTLBuffer> bCH = d_h > 0 ? buffer_for(cos_h) : buffer_for(X);
    NSUInteger oCH    = d_h > 0 ? buffer_offset_for(cos_h) : 0;
    id<MTLBuffer> bSH = d_h > 0 ? buffer_for(sin_h) : buffer_for(X);
    NSUInteger oSH    = d_h > 0 ? buffer_offset_for(sin_h) : 0;
    id<MTLBuffer> bCW = d_w > 0 ? buffer_for(cos_w) : buffer_for(X);
    NSUInteger oCW    = d_w > 0 ? buffer_offset_for(cos_w) : 0;
    id<MTLBuffer> bSW = d_w > 0 ? buffer_for(sin_w) : buffer_for(X);
    NSUInteger oSW    = d_w > 0 ? buffer_offset_for(sin_w) : 0;
    id<MTLBuffer> bPT = d_t > 0 ? pool_lookup(pos_t) : buffer_for(X);
    NSUInteger oPT    = d_t > 0 ? pool_lookup_offset(pos_t) : 0;
    id<MTLBuffer> bPH = d_h > 0 ? pool_lookup(pos_h) : buffer_for(X);
    NSUInteger oPH    = d_h > 0 ? pool_lookup_offset(pos_h) : 0;
    id<MTLBuffer> bPW = d_w > 0 ? pool_lookup(pos_w) : buffer_for(X);
    NSUInteger oPW    = d_w > 0 ? pool_lookup_offset(pos_w) : 0;

    const uint32_t Lu = (uint32_t)L;
    const uint32_t Hu = (uint32_t)num_heads;
    const uint32_t Du = (uint32_t)head_dim;
    const uint32_t dtU = (uint32_t)d_t;
    const uint32_t dhU = (uint32_t)d_h;
    const uint32_t dwU = (uint32_t)d_w;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX  offset:oX  atIndex:0];
        [enc setBuffer:bY  offset:oY  atIndex:1];
        [enc setBuffer:bCT offset:oCT atIndex:2];
        [enc setBuffer:bST offset:oST atIndex:3];
        [enc setBuffer:bCH offset:oCH atIndex:4];
        [enc setBuffer:bSH offset:oSH atIndex:5];
        [enc setBuffer:bCW offset:oCW atIndex:6];
        [enc setBuffer:bSW offset:oSW atIndex:7];
        [enc setBuffer:bPT offset:oPT atIndex:8];
        [enc setBuffer:bPH offset:oPH atIndex:9];
        [enc setBuffer:bPW offset:oPW atIndex:10];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:11];
        [enc setBytes:&Hu  length:sizeof(uint32_t) atIndex:12];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:13];
        [enc setBytes:&dtU length:sizeof(uint32_t) atIndex:14];
        [enc setBytes:&dhU length:sizeof(uint32_t) atIndex:15];
        [enc setBytes:&dwU length:sizeof(uint32_t) atIndex:16];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace brotensor::detail::metal
