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
#include <brotensor/detail/cpu/thread_pool.h>

#include <cmath>
#include <cstdint>
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
    // m-k-n order: broadcast A[m,k], walk B's row k and C's row m contiguously
    // in the innermost loop (both stride-1) instead of striding through B by N
    // floats per k step. Each m exclusively owns C's row m (B is read-only),
    // so the outer loop parallelizes across m with no cross-thread writes.
    parallel_for(static_cast<std::size_t>(M), [&](std::size_t mi) {
        const int m = static_cast<int>(mi);
        float* Crow = Cp + static_cast<size_t>(m) * N;
        for (int n = 0; n < N; ++n) Crow[n] = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float a_mk = Ap[m * K + k];
            const float* Brow = Bp + static_cast<size_t>(k) * N;
            for (int n = 0; n < N; ++n) {
                Crow[n] += a_mk * Brow[n];
            }
        }
    });
}

// Batched A @ B^T, 16-bit (FP16/BF16), FP32 accumulation. Reference triple
// loop mirroring fp16_internal::launch_matmul_ABT_batched_impl semantics:
//   C[b][m,n] = sum_k A[b][m,k] * B[b][n,k]  (+ bias[n], then activation).
// bias is per-N (broadcast over rows), length N. C is caller-sized.
void matmul_abt(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                ::brotensor::Tensor& C,
                int batch, int M, int N, int K,
                long long strideA, long long strideB, long long strideC,
                const ::brotensor::Tensor* bias, int act) {
    if (A.dtype != B.dtype || A.dtype != C.dtype) {
        throw std::runtime_error("matmul_abt: A, B, C must share dtype");
    }
    const bool is_fp16 = (A.dtype == ::brotensor::Dtype::FP16);
    const bool is_bf16 = (A.dtype == ::brotensor::Dtype::BF16);
    if (!is_fp16 && !is_bf16) {
        throw std::runtime_error("matmul_abt: dtype must be FP16 or BF16");
    }
    if (bias && bias->dtype != A.dtype) {
        throw std::runtime_error("matmul_abt: bias dtype must match operands");
    }
    if (batch <= 0 || M == 0 || N == 0) return;

    const uint16_t* Ap = is_fp16 ? A.host_fp16() : A.host_bf16();
    const uint16_t* Bp = is_fp16 ? B.host_fp16() : B.host_bf16();
    uint16_t*       Cp = is_fp16 ? C.host_fp16_mut() : C.host_bf16_mut();
    const uint16_t* bp =
        bias ? (is_fp16 ? bias->host_fp16() : bias->host_bf16()) : nullptr;

    auto to_f32 = [&](uint16_t b) {
        return is_fp16 ? ::brotensor::fp16_bits_to_fp32(b)
                       : ::brotensor::bf16_bits_to_fp32(b);
    };
    auto from_f32 = [&](float v) {
        return is_fp16 ? ::brotensor::fp32_to_fp16_bits(v)
                       : ::brotensor::fp32_to_bf16_bits(v);
    };
    auto apply_act = [&](float v) -> float {
        switch (act) {
            case 0: return v;                                  // none
            case 1: return v > 0.0f ? v : 0.0f;                // relu
            case 2: {                                          // gelu (tanh)
                const float c = 0.7978845608028654f;           // sqrt(2/pi)
                const float t = c * (v + 0.044715f * v * v * v);
                return 0.5f * v * (1.0f + std::tanh(t));
            }
            case 3:                                            // gelu (exact)
                return 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f));
            case 4:                                            // silu
                return v / (1.0f + std::exp(-v));
            case 5:                                            // quick gelu
                return v / (1.0f + std::exp(-1.702f * v));
            default: return v;
        }
    };

    for (int b = 0; b < batch; ++b) {
        const uint16_t* Ab = Ap + static_cast<long long>(b) * strideA;
        const uint16_t* Bb = Bp + static_cast<long long>(b) * strideB;
        uint16_t*       Cb = Cp + static_cast<long long>(b) * strideC;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += to_f32(Ab[m * K + k]) * to_f32(Bb[n * K + k]);
                }
                if (bp) acc += to_f32(bp[n]);
                Cb[m * N + n] = from_f32(apply_act(acc));
            }
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
    // Each m exclusively owns dA's row m (dC/B are read-only), so this
    // parallelizes across m with no cross-thread writes.
    parallel_for(static_cast<std::size_t>(M), [&](std::size_t mi) {
        const int m = static_cast<int>(mi);
        for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                acc += dCp[m * N + n] * Bp[k * N + n];
            }
            dAp[m * K + k] += acc;
        }
    });
    // dB[k, n] += sum_m A[m, k] * dC[m, n]  (accumulate — matches GPU atomicAdd)
    // k-m-n order: each k exclusively owns dB's row k (accumulated over all
    // m), so this axis parallelizes across k with no cross-thread writes —
    // unlike an m-outer order, which would need every thread to accumulate
    // into the same shared dB rows across m. dC's row m and dB's row k are
    // still walked contiguously (both stride-1) in the innermost loop; only
    // the A[m,k] broadcast load is strided, which doesn't block
    // vectorization of the n loop.
    parallel_for(static_cast<std::size_t>(K), [&](std::size_t ki) {
        const int k = static_cast<int>(ki);
        float* dBrow = dBp + static_cast<size_t>(k) * N;
        for (int m = 0; m < M; ++m) {
            const float a_mk = Ap[m * K + k];
            const float* dCrow = dCp + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) {
                dBrow[n] += a_mk * dCrow[n];
            }
        }
    });
}

} // namespace brotensor::detail::cpu
