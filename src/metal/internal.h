#pragma once

// Internal helpers shared across the Metal backend translation units.
// NOT installed; not part of the public API.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <brotensor/tensor.h>

// The custom-kernel surface — device(), queue(), new_command_buffer(),
// pool_lookup[_offset](), buffer_for(), buffer_offset_for(), set_tensor(),
// compile_pipeline() — is the public, installed half of this header. The
// declarations below are the internal-only remainder (pool mutation,
// compile-pipeline archive flushing, the cuda_* lifecycle hooks).
#include <brotensor/metal_interop.h>

#include <cstddef>

namespace brotensor {

// ─── Metal backend runtime (internal) ──────────────────────────────────────
//
// The public runtime surface (init / sync / DeviceScope) lives in
// <brotensor/runtime.h> and src/init.cpp. These are the Metal backend's own
// private lifecycle hooks, kept under the historical `cuda_*` names for
// source continuity with the op .mm files. Declared here (not in any public
// header) so the op TUs can call cuda_init() lazily.
void  cuda_init();
void  cuda_sync();
void  cuda_set_stream(void* stream);
void* cuda_current_stream();
void  cuda_stream_sync(void* stream);
void  cuda_check_throw(int err, const char* expr_text,
                       const char* file, int line);

} // namespace brotensor

namespace brotensor::metal_impl {

// Pool mutation. The pool maps (data_ptr → MTLBuffer); device(), queue(),
// new_command_buffer(), pool_lookup[_offset]() and the buffer_for /
// set_tensor helpers are declared in <brotensor/metal_interop.h>. These two
// mutators stay internal — only tensor.mm's AllocVTable touches the pool.
void  pool_register(void* data_ptr, id<MTLBuffer> buf);
void  pool_release(void* data_ptr);

// The pooled buffer a raw device-pointer operand (a mask, indices, offsets)
// lives in, with its byte offset in `ofs`. A null `p` is "no operand": nil,
// offset 0. A non-null `p` no pooled buffer contains — a host pointer, or one
// from another backend or already freed — throws "brotensor: <op>: <name> is
// not a Metal device pointer" rather than leaving the op to bind some other
// buffer in its place.
id<MTLBuffer> pool_require(const void* p, NSUInteger& ofs,
                           const char* op, const char* name);

// Flush any newly compiled compute pipelines to the on-disk MTLBinaryArchive.
void flush_pipeline_archive();

} // namespace brotensor::metal_impl

namespace brotensor::detail::metal {

// Shared prefill fast path for the quantized batched-linear ops (Q4_K / Q6_K /
// Q8_0). Given an already-dequantized FP16 weight W_fp16 (out, K), computes
// Y(B, out) = X_BD(B, K) @ W_fp16^T (+ optional bias) via the simdgroup ABT
// GEMM in fp16_matmul.mm. Lets each quant op dequant its weight ONCE and reuse
// the tensor-core kernel, instead of re-decoding the weight per output token.
void quant_prefill_gemm_fp16(const ::brotensor::Tensor& W_fp16,
                             const ::brotensor::Tensor* bias,
                             const ::brotensor::Tensor& X_BD,
                             ::brotensor::Tensor& Y_BD);

// Batch size at/above which the quantized batched-linear ops switch from the
// per-token dequant+GEMV path to the dequant-once + tensor-core GEMM prefill
// path above. Below this, re-reading the (compact) quantized weight per token
// is cheaper than materializing a full FP16 weight; above it, amortizing the
// dequant and using simdgroup matmul wins.
inline constexpr int kQuantPrefillMinB = 8;

// Fused quantized-weight MirKind::GemmCm prefill fast path (in gemm_fp32.mm)
// for QF_INT8 / QF_FP8 / QF_Q8_0 / QF_Q4K / QF_Q6K with FP16 activations.
bool try_mir_quant_gemm_cm(std::uint32_t qb,
                             const void* a, const void* b, const void* c,
                             const void* bias, const void* scale,
                             ::brotensor::Dtype dt_out,
                             int M, int N, int K,
                             int lda, int ldb_bytes, int ldc,
                             int act, int epi, const char* op);

// Fused INT8-weight MirKind::ConvCm fast path (in conv2d.mm) for FP16 X.
bool try_conv2d_int8w_cm(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& W_int8,
                         const ::brotensor::Tensor& scales,
                         const ::brotensor::Tensor* bias,
                         int N, int C_in, int H, int W,
                         int C_out, int H_out, int W_out,
                         int kH, int kW,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dil_h, int dil_w,
                         int groups,
                         ::brotensor::Tensor& Y);

} // namespace brotensor::detail::metal
