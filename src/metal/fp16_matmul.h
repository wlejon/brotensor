#pragma once

// Tiled FP16 matmul for C(M,N) = A(M,K) @ B(N,K)^T, FP16 storage / FP32 accum.
// Defined in src/metal/fp16_matmul.mm. Used by gemm.mm (Linear forward) and
// flash_attention.mm for QK^T / PV inner products.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace brotensor::metal_impl {

// Batched A @ B^T with optional per-N bias (shared across the batch) and a
// fused epilogue activation (act code matches src/cuda/fp16_matmul.cu:
// 0 none, 1 relu, 2 gelu-tanh, 3 gelu-erf, 4 silu, 5 quick-gelu). Strides are
// element counts per batch matrix. FP16 storage / FP32 accumulate; a
// 64x64x32 simdgroup-matrix kernel handles M*N >= 1024 (masked on M/N/K, so
// any shape is valid), with a naive one-thread-per-output fallback below that
// and for K == 0. This backs the public matmul_abt op.
void launch_matmul_abt_fp16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act);

// Single (non-batched) contiguous A @ B^T, no bias, no activation — the shape
// the flash-attention QK^T/PV and Linear-forward inner products use. Thin
// wrapper over launch_matmul_abt_fp16_ex, so those callers ride the same
// simdgroup fast path.
void launch_matmul_abt_fp16(id<MTLBuffer> A, NSUInteger ofs_A,
                            id<MTLBuffer> B, NSUInteger ofs_B,
                            id<MTLBuffer> C, NSUInteger ofs_C,
                            int M, int N, int K);

// BF16 twins. Naive per-thread GEMM — simdgroup_matrix has no bfloat form, so
// there is no tiled fast path; the _ex form still carries batch/bias/act.
void launch_matmul_abt_bf16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act);
void launch_matmul_abt_bf16(id<MTLBuffer> A, NSUInteger ofs_A,
                            id<MTLBuffer> B, NSUInteger ofs_B,
                            id<MTLBuffer> C, NSUInteger ofs_C,
                            int M, int N, int K);

} // namespace brotensor::metal_impl
