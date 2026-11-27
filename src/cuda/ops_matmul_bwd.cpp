// Backward of matmul backed by Brass MIR tiled GEMM kernels (EPI_ACCUM).
//   forward: C(M, N) = A(M, K) @ B(K, N)
//   dA(M, K) += dC(M, N) @ B^T(N, K)
//   dB(K, N) += A^T(K, M) @ dC(M, N)

#include "ops_gemm_mir.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/tensor.h>

#include <stdexcept>

namespace brotensor {
namespace detail::cuda {

void matmul_backward(const ::brotensor::Tensor& A,
                     const ::brotensor::Tensor& B,
                     const ::brotensor::Tensor& dC,
                     ::brotensor::Tensor& dA,
                     ::brotensor::Tensor& dB) {
    if (A.dtype != B.dtype || A.dtype != dC.dtype ||
        A.dtype != dA.dtype || A.dtype != dB.dtype) {
        throw std::runtime_error("matmul_backward: dtype mismatch");
    }
    if (A.dtype != Dtype::FP32 && A.dtype != Dtype::FP16 &&
        A.dtype != Dtype::BF16) {
        throw std::runtime_error("matmul_backward: only FP32/FP16/BF16 supported");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul_backward: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (dC.rows != M || dC.cols != N) {
        throw std::runtime_error("matmul_backward: dC shape mismatch");
    }
    if (dA.rows != M || dA.cols != K) {
        throw std::runtime_error("matmul_backward: dA must be pre-sized to (M, K)");
    }
    if (dB.rows != K || dB.cols != N) {
        throw std::runtime_error("matmul_backward: dB must be pre-sized to (K, N)");
    }
    if (M == 0 || N == 0 || K == 0) return;

    const Dtype dt = A.dtype;
    dispatch_mir_gemm_simt(dC.data, dt, B.data, dt, dA.data, dt,
                           nullptr, dt, 1, M, K, N,
                           N, N, K, 0, 0, 0,
                           /*ta=*/false, /*nb=*/false, EPI_ACCUM, 0, A.device);
    dispatch_mir_gemm_simt(A.data, dt, dC.data, dt, dB.data, dt,
                           nullptr, dt, 1, K, N, M,
                           K, N, N, 0, 0, 0,
                           /*ta=*/true, /*nb=*/true, EPI_ACCUM, 0, A.device);
}

}  // namespace detail::cuda
}  // namespace brotensor
