// High-performance GEMM operations for AMD GPU (ROCm/HIP) backend via hipBLAS.
//
// Computes row-major matrix multiplications using hipBLAS:
// - matmul (NN):         C(M,N) = A(M,K) @ B(K,N)
// - matmul_abt (NT):     C(M,N) = A(M,K) @ B(N,K)^T (+ bias, + act)
// - batched linear:      Y(B,N) = X(B,K) @ W(N,K)^T (+ bias, + act)

#include "detail/hip_check.h"
#include "detail/hipblas_handle.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/ops/linear.h>
#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <stdexcept>
#include <string>

#define BROTENSOR_HIPBLAS_CHECK(expr)                                                 \
    do {                                                                               \
        hipblasStatus_t _bga_err = (expr);                                             \
        if (_bga_err != HIPBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("hipBLAS error: ") + #expr);          \
        }                                                                              \
    } while (0)

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

__device__ inline float apply_linear_act(int act, float v) {
    switch (act) {
        case 1:  return v > 0.0f ? v : 0.0f;                                  // relu
        case 2: {                                                              // gelu_tanh
            constexpr float kSqrt2OverPi = 0.7978845608f;
            const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + tanhf(u));
        }
        case 3: {                                                              // gelu_exact
            constexpr float kInvSqrt2 = 0.70710678118654752440f;
            return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
        }
        case 4:  return v / (1.0f + __expf(-v));                               // silu
        case 5:  return v / (1.0f + __expf(-1.702f * v));                      // quick_gelu
        default: return v;
    }
}

template <typename T>
__global__ void hip_linear_epilogue_kernel(
    T* __restrict__ Y,
    const T* __restrict__ bias,
    int batch, int M, int N,
    long long strideY,
    int act) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total_per_batch = static_cast<long long>(M) * N;
    const long long total = total_per_batch * batch;
    if (tid >= total) return;

    const int b = static_cast<int>(tid / total_per_batch);
    const int rem = static_cast<int>(tid % total_per_batch);
    const int col = rem % N;

    T* y_row = Y + b * strideY;
    float v = 0.0f;
    if constexpr (std::is_same_v<T, __half>) {
        v = __half2float(y_row[rem]);
        if (bias) v += __half2float(bias[col]);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        v = float(y_row[rem]);
        if (bias) v += float(bias[col]);
    } else {
        v = y_row[rem];
        if (bias) v += bias[col];
    }
    v = apply_linear_act(act, v);
    if constexpr (std::is_same_v<T, __half>) {
        y_row[rem] = __float2half(v);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
        y_row[rem] = hip_bfloat16(v);
    } else {
        y_row[rem] = v;
    }
}

template <typename T>
void launch_epilogue(T* Y, const T* bias, int batch, int M, int N,
                     long long strideY, int act, hipStream_t stream) {
    const long long total = static_cast<long long>(M) * N * batch;
    if (total == 0) return;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
    hip_linear_epilogue_kernel<T><<<blocks, threads, 0, stream>>>(
        Y, bias, batch, M, N, strideY, act);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace

void hip_matmul(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                ::brotensor::Tensor& C) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("hip_matmul: A and B must share dtype");
    }
    const int M = A.rows;
    const int K = A.cols;
    if (B.rows != K) {
        throw std::runtime_error("hip_matmul: shape mismatch (A.cols != B.rows)");
    }
    const int N = B.cols;
    if (C.rows != M || C.cols != N || C.dtype != A.dtype) {
        C.resize(M, N, A.dtype);
    }
    if (M == 0 || N == 0) return;

    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_current_stream(A.device.index));
    if (K == 0) {
        BROTENSOR_HIP_CHECK(hipMemsetAsync(C.data, 0, C.bytes(), stream));
        return;
    }

    hipblasHandle_t handle = hipblas_handle(A.device.index);

    if (A.dtype == Dtype::FP16) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,
            &alpha,
            B.data, HIP_R_16F, N,
            A.data, HIP_R_16F, K,
            &beta,
            C.data, HIP_R_16F, N,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    } else if (A.dtype == Dtype::BF16) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,
            &alpha,
            B.data, HIP_R_16BF, N,
            A.data, HIP_R_16BF, K,
            &beta,
            C.data, HIP_R_16BF, N,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    } else if (A.dtype == Dtype::FP32) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,
            &alpha,
            reinterpret_cast<const float*>(B.data), N,
            reinterpret_cast<const float*>(A.data), K,
            &beta,
            reinterpret_cast<float*>(C.data), N));
    } else {
        throw std::runtime_error("hip_matmul: unsupported dtype on HIP");
    }
}

void hip_matmul_abt(const ::brotensor::Tensor& A, const ::brotensor::Tensor& B,
                    ::brotensor::Tensor& C,
                    int batch, int M, int N, int K,
                    long long strideA, long long strideB, long long strideC,
                    const ::brotensor::Tensor* bias, int act) {
    if (A.dtype != B.dtype) {
        throw std::runtime_error("hip_matmul_abt: A and B must share dtype");
    }
    if (bias && bias->dtype != A.dtype) {
        throw std::runtime_error("hip_matmul_abt: bias must share operand dtype");
    }
    if (batch <= 0 || M == 0 || N == 0) return;

    if (strideA == 0) strideA = static_cast<long long>(M) * K;
    if (strideB == 0) strideB = static_cast<long long>(N) * K;
    if (strideC == 0) strideC = static_cast<long long>(M) * N;

    if (C.dtype != A.dtype || C.rows * C.cols < static_cast<long long>(batch) * strideC) {
        if (C.rows != (batch > 1 ? batch * M : M) || C.cols != N || C.dtype != A.dtype) {
            C.resize(batch > 1 ? batch * M : M, N, A.dtype);
        }
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_current_stream(A.device.index));
    if (K == 0) {
        BROTENSOR_HIP_CHECK(hipMemsetAsync(C.data, 0, C.bytes(), stream));
        return;
    }

    hipblasHandle_t handle = hipblas_handle(A.device.index);

    if (A.dtype == Dtype::FP16) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        if (batch == 1) {
            BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data, HIP_R_16F, K,
                A.data, HIP_R_16F, K,
                &beta,
                C.data, HIP_R_16F, N,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        } else {
            BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
                handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data, HIP_R_16F, K, strideB,
                A.data, HIP_R_16F, K, strideA,
                &beta,
                C.data, HIP_R_16F, N, strideC,
                batch,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        }
        if (bias || act != 0) {
            const __half* bp = bias ? reinterpret_cast<const __half*>(bias->data) : nullptr;
            launch_epilogue<__half>(
                reinterpret_cast<__half*>(C.data), bp, batch, M, N, strideC, act, stream);
        }
    } else if (A.dtype == Dtype::BF16) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        if (batch == 1) {
            BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data, HIP_R_16BF, K,
                A.data, HIP_R_16BF, K,
                &beta,
                C.data, HIP_R_16BF, N,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        } else {
            BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
                handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data, HIP_R_16BF, K, strideB,
                A.data, HIP_R_16BF, K, strideA,
                &beta,
                C.data, HIP_R_16BF, N, strideC,
                batch,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        }
        if (bias || act != 0) {
            const hip_bfloat16* bp = bias ? reinterpret_cast<const hip_bfloat16*>(bias->data) : nullptr;
            launch_epilogue<hip_bfloat16>(
                reinterpret_cast<hip_bfloat16*>(C.data), bp, batch, M, N, strideC, act, stream);
        }
    } else if (A.dtype == Dtype::FP32) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        if (batch == 1) {
            BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
                handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                reinterpret_cast<const float*>(B.data), K,
                reinterpret_cast<const float*>(A.data), K,
                &beta,
                reinterpret_cast<float*>(C.data), N));
        } else {
            BROTENSOR_HIPBLAS_CHECK(hipblasSgemmStridedBatched(
                handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                N, M, K,
                &alpha,
                reinterpret_cast<const float*>(B.data), K, strideB,
                reinterpret_cast<const float*>(A.data), K, strideA,
                &beta,
                reinterpret_cast<float*>(C.data), N, strideC,
                batch));
        }
        if (bias || act != 0) {
            const float* bp = bias ? reinterpret_cast<const float*>(bias->data) : nullptr;
            launch_epilogue<float>(
                reinterpret_cast<float*>(C.data), bp, batch, M, N, strideC, act, stream);
        }
    } else {
        throw std::runtime_error("hip_matmul_abt: unsupported dtype on HIP");
    }
}

void hip_linear_forward_batched_fp16_act(const ::brotensor::Tensor& W,
                                         const ::brotensor::Tensor* bias,
                                         const ::brotensor::Tensor& X_BD,
                                         int act,
                                         ::brotensor::Tensor& Y_BD) {
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("hip_linear_forward_batched_fp16_act: shape mismatch");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != X_BD.dtype) {
        Y_BD.resize(B, out_dim, X_BD.dtype);
    }
    hip_matmul_abt(X_BD, W, Y_BD, 1, B, out_dim, in_dim, 0, 0, 0, bias, act);
}

void hip_linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                     const ::brotensor::Tensor* bias,
                                     const ::brotensor::Tensor& X_BD,
                                     ::brotensor::Tensor& Y_BD) {
    hip_linear_forward_batched_fp16_act(W, bias, X_BD, 0, Y_BD);
}

__global__ void fp32_to_fp16_kernel(const float* __restrict__ src, __half* __restrict__ dst, long long n) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

template <typename T>
__global__ void hip_linear_ex_epilogue_kernel(
    const T* __restrict__ r,
    T* __restrict__ y,
    long long count, int N, int epi) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (epi == 1) { // accumulate
        float val = float(y[i]) + float(r[i]);
        y[i] = T(val);
    } else { // geglu: Y[:, j] = r[:, 2j] * gelu_exact(r[:, 2j+1])
        const int half = N / 2;
        const long long m = i / half;
        const int j = static_cast<int>(i - m * half);
        const float a = float(r[m * N + 2 * j]);
        const float g = float(r[m * N + 2 * j + 1]);
        constexpr float kInvSqrt2 = 0.70710678118654752440f;
        const float gelu_g = 0.5f * g * (1.0f + erff(g * kInvSqrt2));
        y[i] = T(a * gelu_g);
    }
}

void hip_linear_forward_batched(const ::brotensor::Tensor& W,
                                const ::brotensor::Tensor& bias,
                                const ::brotensor::Tensor& X_BD,
                                ::brotensor::Tensor& Y_BD) {
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("hip_linear_forward_batched: shape mismatch");
    }
    const ::brotensor::Tensor* bp = bias.size() > 0 ? &bias : nullptr;
    if (X_BD.dtype == W.dtype) {
        hip_matmul_abt(X_BD, W, Y_BD, 1, B, out_dim, in_dim, 0, 0, 0, bp, 0);
        return;
    }
    // Mixed precision: FP32 X, FP16 W -> FP32 Y
    if (X_BD.dtype == Dtype::FP32 && W.dtype == Dtype::FP16) {
        if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::FP32) {
            Y_BD.resize(B, out_dim, Dtype::FP32);
        }
        hipStream_t stream = reinterpret_cast<hipStream_t>(hip_current_stream(W.device.index));
        hipblasHandle_t handle = hipblas_handle(W.device.index);

        ::brotensor::Tensor X_16 = ::brotensor::Tensor::empty_on(X_BD.device, B, in_dim, Dtype::FP16);
        const long long total_x = static_cast<long long>(B) * in_dim;
        const int threads = 256;
        const unsigned blocks = static_cast<unsigned>((total_x + threads - 1) / threads);
        fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(X_BD.data),
            reinterpret_cast<__half*>(X_16.data),
            total_x);

        const float alpha = 1.0f, beta = 0.0f;
        BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
            out_dim, B, in_dim,
            &alpha,
            W.data, HIP_R_16F, in_dim,
            X_16.data, HIP_R_16F, in_dim,
            &beta,
            Y_BD.data, HIP_R_32F, out_dim,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        if (bp) {
            launch_epilogue<float>(reinterpret_cast<float*>(Y_BD.data),
                                   reinterpret_cast<const float*>(bp->data),
                                   1, B, out_dim, static_cast<long long>(B) * out_dim, 0, stream);
        }
        return;
    }
    throw std::runtime_error("hip_linear_forward_batched: unsupported dtype combination");
}

void hip_linear_forward_batched_ex(const ::brotensor::Tensor& W,
                                   const ::brotensor::Tensor* bias,
                                   const ::brotensor::Tensor& X_BD,
                                   int act, int epilogue,
                                   ::brotensor::Tensor* workspace,
                                   ::brotensor::Tensor& Y_BD) {
    (void)workspace;
    constexpr int kLinearEpiFastAccumFlag = 16;
    constexpr int kLinearEpiStoreCode     = 0;
    constexpr int kLinearEpiAccumulateCode = 1;
    constexpr int kLinearEpiGegluCode     = 2;
    const int epilogue_mode = epilogue & ~kLinearEpiFastAccumFlag;

    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("hip_linear_forward_batched_ex: shape mismatch");
    }
    const int actual_out = (epilogue_mode == kLinearEpiGegluCode) ? out_dim / 2 : out_dim;
    if (epilogue_mode == kLinearEpiAccumulateCode) {
        if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != X_BD.dtype) {
            throw std::runtime_error("hip_linear_forward_batched_ex: accumulate requires matching Y");
        }
    } else {
        if (Y_BD.rows != B || Y_BD.cols != actual_out || Y_BD.dtype != X_BD.dtype) {
            Y_BD.resize(B, actual_out, X_BD.dtype);
        }
    }
    if (B == 0 || out_dim == 0) return;

    if (epilogue_mode == kLinearEpiStoreCode) {
        hip_linear_forward_batched_fp16_act(W, bias, X_BD, act, Y_BD);
        return;
    }

    ::brotensor::Tensor r = ::brotensor::Tensor::empty_on(W.device, B, out_dim, W.dtype);
    hip_linear_forward_batched_fp16_act(W, bias, X_BD, act, r);

    const long long count = static_cast<long long>(B) * actual_out;
    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_current_stream(W.device.index));
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);

    if (W.dtype == Dtype::FP16) {
        hip_linear_ex_epilogue_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(r.data),
            reinterpret_cast<__half*>(Y_BD.data),
            count, out_dim, epilogue_mode);
    } else if (W.dtype == Dtype::BF16) {
        hip_linear_ex_epilogue_kernel<hip_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const hip_bfloat16*>(r.data),
            reinterpret_cast<hip_bfloat16*>(Y_BD.data),
            count, out_dim, epilogue_mode);
    } else {
        hip_linear_ex_epilogue_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(r.data),
            reinterpret_cast<float*>(Y_BD.data),
            count, out_dim, epilogue_mode);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                        const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (x.rows != in_dim || x.cols != 1) {
        throw std::runtime_error("hip_linear_forward: x must be (in_dim, 1)");
    }
    if (y.rows != out_dim || y.cols != 1 || y.dtype != x.dtype) {
        y.resize(out_dim, 1, x.dtype);
    }
    const ::brotensor::Tensor* bp = b.size() > 0 ? &b : nullptr;
    hip_matmul_abt(x, W, y, 1, 1, out_dim, in_dim, 0, 0, 0, bp, 0);
}

void fill_hip_vtable_gemm(::brotensor::detail::OpsVTable& v) {
    v.matmul                          = &hip_matmul;
    v.matmul_abt                      = &hip_matmul_abt;
    v.linear_forward                  = &hip_linear_forward;
    v.linear_forward_batched          = &hip_linear_forward_batched;
    v.linear_forward_batched_ex       = &hip_linear_forward_batched_ex;
    v.linear_forward_batched_fp16     = &hip_linear_forward_batched_fp16;
    v.linear_forward_batched_fp16_act = &hip_linear_forward_batched_fp16_act;
}

} // namespace brotensor::detail::hip
