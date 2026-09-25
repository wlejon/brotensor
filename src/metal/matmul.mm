// Public matmul_gpu: row-major C(M,N) = A(M,K) @ B(K,N), no bias.
// FP32 accumulation throughout, on the simdgroup GEMM in gemm_fp32.mm.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"
#include "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("matmul_gpu: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_gpu: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) {
        C.resize(M, N, A.dtype);
    }
    if (M == 0 || N == 0) return;
    if (K == 0) {
        C.zero();
        return;
    }
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_gpu: only FP32/FP16/BF16 supported");
    }

    // The simdgroup GEMM (gemm_fp32.mm), B read as (K, N).
    metal_impl::AbtMixed g;
    g.A = buffer_for(A); g.ofs_A = buffer_offset_for(A); g.lda = static_cast<uint64_t>(K);
    g.B = buffer_for(B); g.ofs_B = buffer_offset_for(B); g.ldb = static_cast<uint64_t>(N); g.transB = true;
    g.C = buffer_for(C); g.ofs_C = buffer_offset_for(C); g.ldc = static_cast<uint64_t>(N);
    g.M = M; g.N = N; g.K = K;
    g.in = g.out = metal_impl::abt_type(A.dtype);
    metal_impl::launch_matmul_abt_mixed(g);
}

void matmul_abt(const Tensor& A, const Tensor& B, Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const Tensor* bias, int act) {
    if (A.dtype != B.dtype || A.dtype != C.dtype) {
        throw std::runtime_error("matmul_abt: A, B, C must share dtype");
    }
    if (A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_abt: dtype must be FP16 or BF16");
    }
    if (bias && bias->dtype != A.dtype) {
        throw std::runtime_error("matmul_abt: bias dtype must match operands");
    }
    if (batch <= 0 || M == 0 || N == 0) return;

    id<MTLBuffer> bA = buffer_for(A);
    id<MTLBuffer> bB = buffer_for(B);
    id<MTLBuffer> bC = buffer_for(C);
    const NSUInteger oA = buffer_offset_for(A);
    const NSUInteger oB = buffer_offset_for(B);
    const NSUInteger oC = buffer_offset_for(C);
    id<MTLBuffer> bBias = bias ? buffer_for(*bias) : nil;
    const NSUInteger oBias = bias ? buffer_offset_for(*bias) : 0;

    const uint64_t sA = static_cast<uint64_t>(strideA);
    const uint64_t sB = static_cast<uint64_t>(strideB);
    const uint64_t sC = static_cast<uint64_t>(strideC);

    // The batched ABT kernels (FP16 simdgroup fast path + naive fallbacks, plus
    // the BF16 naive path) live in fp16_matmul.mm and are shared with the
    // attention / Linear inner-product callers. K == 0, small shapes, bias and
    // activation are all handled there.
    if (A.dtype == Dtype::FP16) {
        metal_impl::launch_matmul_abt_fp16_ex(bA, oA, bB, oB, bC, oC,
                                              batch, M, N, K, sA, sB, sC,
                                              bBias, oBias, bias != nullptr, act);
    } else {
        metal_impl::launch_matmul_abt_bf16_ex(bA, oA, bB, oB, bC, oC,
                                              batch, M, N, K, sA, sB, sC,
                                              bBias, oBias, bias != nullptr, act);
    }
}

void quant_prefill_gemm_fp16(const Tensor& W_fp16, const Tensor* bias,
                             const Tensor& X_BD, Tensor& Y_BD) {
    // Y(B, out) = X_BD(B, K) @ W_fp16(out, K)^T (+bias). This is exactly the
    // batch=1 ABT GEMM shape M=B, N=out, K=K, so it rides the simdgroup kernel.
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_fp16.rows;
    if (B == 0 || out == 0) return;

    id<MTLBuffer> bX = buffer_for(X_BD);
    id<MTLBuffer> bW = buffer_for(W_fp16);
    id<MTLBuffer> bY = buffer_for(Y_BD);
    const NSUInteger oX = buffer_offset_for(X_BD);
    const NSUInteger oW = buffer_offset_for(W_fp16);
    const NSUInteger oY = buffer_offset_for(Y_BD);

    const bool has_bias = bias && bias->size() > 0;
    id<MTLBuffer> bBias = has_bias ? buffer_for(*bias) : nil;
    const NSUInteger oBias = has_bias ? buffer_offset_for(*bias) : 0;

    metal_impl::launch_matmul_abt_fp16_ex(
        bX, oX, bW, oW, bY, oY,
        /*batch=*/1, /*M=*/B, /*N=*/out, /*K=*/K,
        static_cast<uint64_t>(B) * static_cast<uint64_t>(K),
        static_cast<uint64_t>(out) * static_cast<uint64_t>(K),
        static_cast<uint64_t>(B) * static_cast<uint64_t>(out),
        bBias, oBias, has_bias, /*act=*/0);
}

} // namespace brotensor::detail::metal
