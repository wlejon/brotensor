// ─── CPU matmul ops (CHUNK 2) ──────────────────────────────────────────────
//
// FP32 scalar host implementations. Ports src/cuda/matmul.cu and
// src/cuda/matmul_backward.cu — FP32 path only, row-major throughout.
//
//   forward:  C(M, N) = A(M, K) @ B(K, N)
//   backward: dA(M, K) += dC(M, N) @ B^T(N, K)
//             dB(K, N) += A^T(K, M) @ dC(M, N)
//
// ACCUMULATION: the GPU backward kernels atomicAdd partial products into the
// caller's dA / dB buffers, so they ACCUMULATE (+=). The caller is responsible
// for zeroing dA / dB before the call if a fresh gradient is wanted; the GPU
// kernel also requires dA / dB to be pre-sized to (M, K) / (K, N).

#include <brotensor/tensor.h>

#include <stdexcept>

namespace brotensor::detail::cpu {

void matmul(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
            ::brotensor::Tensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("matmul: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("matmul: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N) C.resize(M, N);
    if (M == 0 || N == 0) return;
    if (K == 0) {
        C.zero();
        return;
    }
    const float* Ap = A.host_f32();
    const float* Bp = B.host_f32();
    float* Cp = C.host_f32_mut();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += Ap[m * K + k] * Bp[k * N + n];
            }
            Cp[m * N + n] = acc;
        }
    }
}

void matmul_backward(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                     const ::brotensor::Tensor& dC,
                     ::brotensor::Tensor& dA, ::brotensor::Tensor& dB) {
    if (A.dtype != B.dtype || A.dtype != dC.dtype ||
        A.dtype != dA.dtype || A.dtype != dB.dtype) {
        throw std::runtime_error("matmul_backward: dtype mismatch");
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

    const float* Ap = A.host_f32();
    const float* Bp = B.host_f32();
    const float* dCp = dC.host_f32();
    float* dAp = dA.host_f32_mut();
    float* dBp = dB.host_f32_mut();

    // dA[m, k] += sum_n dC[m, n] * B[k, n]  (accumulate — matches GPU atomicAdd)
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                acc += dCp[m * N + n] * Bp[k * N + n];
            }
            dAp[m * K + k] += acc;
        }
    }
    // dB[k, n] += sum_m A[m, k] * dC[m, n]  (accumulate — matches GPU atomicAdd)
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int m = 0; m < M; ++m) {
                acc += Ap[m * K + k] * dCp[m * N + n];
            }
            dBp[k * N + n] += acc;
        }
    }
}

} // namespace brotensor::detail::cpu
