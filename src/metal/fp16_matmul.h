#pragma once

// Tiled matmuls for C(M,N) = A(M,K) @ B(N,K)^T with FP32 accumulation.
// Defined in src/metal/fp16_matmul.mm. The FP16 / BF16 _ex launchers back the
// public matmul_abt op (matmul.mm); the mixed-precision launcher backs the
// batched linears (gemm.mm, linear_ex.mm) and the attention GEMMs
// (flash_attention*.mm).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::metal_impl {

// Batched A @ B^T with optional per-N bias (shared across the batch) and a
// fused epilogue activation (act code matches src/cuda/fp16_matmul.cu:
// 0 none, 1 relu, 2 gelu-tanh, 3 gelu-erf, 4 silu, 5 quick-gelu). Strides are
// element counts per batch matrix. FP16 storage / FP32 accumulate, on
// launch_matmul_abt_mixed (so any shape is valid). This backs the public
// matmul_abt op.
void launch_matmul_abt_fp16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act);

// BF16 twin (BF16 widens to FP32 in the GEMM).
void launch_matmul_abt_bf16_ex(id<MTLBuffer> A, NSUInteger ofs_A,
                               id<MTLBuffer> B, NSUInteger ofs_B,
                               id<MTLBuffer> C, NSUInteger ofs_C,
                               int batch, int M, int N, int K,
                               uint64_t strideA, uint64_t strideB, uint64_t strideC,
                               id<MTLBuffer> bias, NSUInteger ofs_bias, bool has_bias,
                               int act);

// ── Mixed-precision A @ B^T ─────────────────────────────────────────────────
//
// C[b](M, N) (op)= epilogue(alpha * A[b](M,K) @ B[b](N,K)^T + bias) with the
// input and output element types chosen independently (FP16 / BF16 / FP32 in,
// FP16 / BF16 / FP32 out) and explicit row strides, so a caller can read a
// head's column slice of an (L, D) tensor in place (lda = D) and write the
// result straight into another tensor's column slot (ldc = D). Accumulation is
// FP32 throughout; the output is rounded exactly once. FP16 x FP16 operands
// ride simdgroup_matrix<half>; any BF16 or FP32 operand widens both to FP32
// and rides simdgroup_matrix<float>. B may carry its own type (in_B), e.g. an
// FP16 weight under FP32 activations. bias (N entries, A's type) and act (same
// codes as above) apply before the epilogue:
//   kAbtStore       C = r
//   kAbtAccumulate  C = C + r          (C read in the output type)
//   kAbtGeglu       C[:, j] = r[:, 2j] * gelu_erf(r[:, 2j+1]); N even, C has
//                   N/2 columns (ldc counts output columns).
enum AbtType : int { kAbtF16 = 0, kAbtBF16 = 1, kAbtF32 = 2 };
enum AbtEpilogue : int { kAbtStore = 0, kAbtAccumulate = 1, kAbtGeglu = 2 };

struct AbtMixed {
    // A is (M, K) with row stride lda, or with transA (K, M) with row stride
    // lda; likewise B is (N, K) or, with transB, (K, N).
    id<MTLBuffer> A = nil;    NSUInteger ofs_A = 0;    uint64_t lda = 0;    bool transA = false;
    id<MTLBuffer> B = nil;    NSUInteger ofs_B = 0;    uint64_t ldb = 0;    bool transB = false;
    id<MTLBuffer> C = nil;    NSUInteger ofs_C = 0;    uint64_t ldc = 0;
    id<MTLBuffer> bias = nil; NSUInteger ofs_bias = 0;  // nil = no bias
    int M = 0, N = 0, K = 0;
    AbtType in = kAbtF16, out = kAbtF16;
    int in_B = -1;            // B's AbtType when it differs from `in`; -1 = `in`
    int act = 0;
    int epilogue = kAbtStore;
    float alpha = 1.0f;
    int batch = 1;
    uint64_t strideA = 0, strideB = 0, strideC = 0;  // elements, per batch
};

void launch_matmul_abt_mixed(const AbtMixed& g);

// The FP32-accumulate simdgroup GEMM (src/metal/gemm_fp32.mm) that
// launch_matmul_abt_mixed hands every K > 0 product of a useful size to: a
// 64x64 tiled kernel (half tiles, BK 32, for FP16 x FP16; float tiles, BK 16,
// otherwise) and a register-blocked kernel for M <= 8. Same AbtMixed
// contract; K must be positive.
void launch_gemm_fp32(const AbtMixed& g);

// The AbtType for a 16/32-bit float Dtype; throws on anything else.
AbtType abt_type(::brotensor::Dtype dt);

} // namespace brotensor::metal_impl
