// Backward of matmul_gpu (Metal): both products run on the simdgroup GEMM
// (gemm_fp32.mm) with an accumulating epilogue, for FP32, FP16 and BF16 alike
// (FP32 accumulate, one rounding into the caller-owned dA / dB).

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"
#import "fp16_matmul.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;

void matmul_backward(const Tensor& A,
                     const Tensor& B,
                     const Tensor& dC,
                     Tensor& dA,
                     Tensor& dB) {
    if (A.dtype != B.dtype || A.dtype != dC.dtype ||
        A.dtype != dA.dtype || A.dtype != dB.dtype) {
        throw std::runtime_error("matmul_backward_gpu: dtype mismatch");
    }
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16 && A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_backward_gpu: only FP32/FP16/BF16 supported");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_backward_gpu: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (dC.rows != M || dC.cols != N) {
        throw std::runtime_error("matmul_backward_gpu: dC shape mismatch");
    }
    if (dA.rows != M || dA.cols != K) {
        throw std::runtime_error("matmul_backward_gpu: dA must be pre-sized to (M, K)");
    }
    if (dB.rows != K || dB.cols != N) {
        throw std::runtime_error("matmul_backward_gpu: dB must be pre-sized to (K, N)");
    }
    if (M == 0 || N == 0 || K == 0) return;

    // Both products on the simdgroup GEMM (gemm_fp32.mm), FP32 accumulate,
    // added into dA / dB and rounded once into their dtype:
    //   dA(M, K) += dC(M, N) @ B(K, N)^T
    //   dB(K, N) += A(M, K)^T @ dC(M, N)
    id<MTLBuffer> bA  = buffer_for(A);
    id<MTLBuffer> bB  = buffer_for(B);
    id<MTLBuffer> bdC = buffer_for(dC);
    const NSUInteger oA  = buffer_offset_for(A);
    const NSUInteger oB  = buffer_offset_for(B);
    const NSUInteger odC = buffer_offset_for(dC);
    metal_impl::AbtMixed g;
    g.in = g.out = metal_impl::abt_type(A.dtype);
    g.epilogue = metal_impl::kAbtAccumulate;
    g.A = bdC; g.ofs_A = odC; g.lda = static_cast<uint64_t>(N);
    g.B = bB;  g.ofs_B = oB;  g.ldb = static_cast<uint64_t>(N);
    g.C = buffer_for(dA); g.ofs_C = buffer_offset_for(dA); g.ldc = static_cast<uint64_t>(K);
    g.M = M; g.N = K; g.K = N;
    metal_impl::launch_matmul_abt_mixed(g);
    g.A = bA;  g.ofs_A = oA;  g.lda = static_cast<uint64_t>(K); g.transA = true;
    g.B = bdC; g.ofs_B = odC; g.ldb = static_cast<uint64_t>(N); g.transB = true;
    g.C = buffer_for(dB); g.ofs_C = buffer_offset_for(dB); g.ldc = static_cast<uint64_t>(N);
    g.M = K; g.N = N; g.K = M;
    metal_impl::launch_matmul_abt_mixed(g);
}

} // namespace brotensor::detail::metal
