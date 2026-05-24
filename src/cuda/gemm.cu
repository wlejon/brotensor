#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include "fp16_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor {
namespace detail::cuda {

namespace {

template <typename T> __device__ inline float g_load(const T* p);
template <> __device__ inline float g_load<float>(const float* p) { return *p; }
template <> __device__ inline float g_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float g_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void g_store(T* p, float v);
template <> __device__ inline void g_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void g_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void g_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// y[i] = b[i] + sum_j W[i, j] * x[j]
// One thread per output row. Uses shared memory to cache tiles of x (in FP32).
constexpr int LF_BLOCK = 128;
constexpr int LF_TILE  = 128;

template <typename T>
__global__ void linear_forward_kernel(const T* __restrict__ W,
                                      const T* __restrict__ b,
                                      const T* __restrict__ x,
                                      T* __restrict__ y,
                                      int out_dim, int in_dim) {
    __shared__ float xtile[LF_TILE];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < in_dim; t0 += LF_TILE) {
        const int t_len = (in_dim - t0) < LF_TILE ? (in_dim - t0) : LF_TILE;

        // Cooperatively load tile of x into shared memory.
        for (int k = tid; k < t_len; k += blockDim.x) {
            xtile[k] = g_load<T>(&x[t0 + k]);
        }
        __syncthreads();

        if (row < out_dim) {
            const T* wrow = W + static_cast<size_t>(row) * in_dim + t0;
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += g_load<T>(&wrow[k]) * xtile[k];
            }
        }
        __syncthreads();
    }

    if (row < out_dim) {
        g_store<T>(&y[row], g_load<T>(&b[row]) + acc);
    }
}

// dX[j] = sum_i W[i, j] * dY[i]
// One thread per input column. Uses shared memory to cache tiles of dY (FP32).
constexpr int LB_DX_BLOCK = 128;
constexpr int LB_DX_TILE  = 128;

template <typename T>
__global__ void linear_backward_dx_kernel(const T* __restrict__ W,
                                          const T* __restrict__ dY,
                                          T* __restrict__ dX,
                                          int out_dim, int in_dim) {
    __shared__ float dytile[LB_DX_TILE];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < out_dim; t0 += LB_DX_TILE) {
        const int t_len = (out_dim - t0) < LB_DX_TILE ? (out_dim - t0) : LB_DX_TILE;

        for (int k = tid; k < t_len; k += blockDim.x) {
            dytile[k] = g_load<T>(&dY[t0 + k]);
        }
        __syncthreads();

        if (col < in_dim) {
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += g_load<T>(&W[static_cast<size_t>(t0 + k) * in_dim + col]) * dytile[k];
            }
        }
        __syncthreads();
    }

    if (col < in_dim) {
        g_store<T>(&dX[col], acc);
    }
}

// dW[i, j] += dY[i] * x[j]. 2D grid: each thread one (i, j).
template <typename T>
__global__ void linear_backward_dw_kernel(const T* __restrict__ dY,
                                          const T* __restrict__ x,
                                          T* __restrict__ dW,
                                          int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    const size_t off = static_cast<size_t>(i) * in_dim + j;
    const float prev = g_load<T>(&dW[off]);
    const float upd  = g_load<T>(&dY[i]) * g_load<T>(&x[j]);
    g_store<T>(&dW[off], prev + upd);
}

// dB[i] += dY[i].
template <typename T>
__global__ void linear_backward_db_kernel(const T* __restrict__ dY,
                                          T* __restrict__ dB,
                                          int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    g_store<T>(&dB[i], g_load<T>(&dB[i]) + g_load<T>(&dY[i]));
}

inline void require_fp(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

inline void require_same_dtype(const ::brotensor::Tensor& a, const ::brotensor::Tensor& b,
                               const char* op) {
    if (a.dtype != b.dtype) {
        throw std::runtime_error(std::string("brotensor: ") + op +
                                 ": all FP operand dtypes must match");
    }
}

} // anonymous namespace

void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp(W, "linear_forward", "W");
    require_same_dtype(W, b, "linear_forward");
    require_same_dtype(W, x, "linear_forward");
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (y.rows != out_dim || y.cols != 1 || y.dtype != W.dtype) {
        y.resize(out_dim, 1, W.dtype);
    }
    if (out_dim == 0) return;

    const int blocks = (out_dim + LF_BLOCK - 1) / LF_BLOCK;
    if (W.dtype == Dtype::FP16) {
        linear_forward_kernel<__half><<<blocks, LF_BLOCK>>>(
            static_cast<const __half*>(W.data),
            static_cast<const __half*>(b.data),
            static_cast<const __half*>(x.data),
            static_cast<__half*>(y.data),
            out_dim, in_dim);
    } else if (W.dtype == Dtype::BF16) {
        linear_forward_kernel<__nv_bfloat16><<<blocks, LF_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(W.data),
            static_cast<const __nv_bfloat16*>(b.data),
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<__nv_bfloat16*>(y.data),
            out_dim, in_dim);
    } else {
        linear_forward_kernel<float><<<blocks, LF_BLOCK>>>(
            static_cast<const float*>(W.data),
            static_cast<const float*>(b.data),
            static_cast<const float*>(x.data),
            static_cast<float*>(y.data),
            out_dim, in_dim);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// FP16 batched linear forward: Y(B, out_dim) = X(B, in_dim) @ W(out_dim, in_dim)^T
// + optional broadcast bias. Same matmul kernel as cross-attention's matmul_ABT
// — X is the (M=B, K=in_dim) side, W is the (N=out_dim, K=in_dim) side. Bias
// is added in a tiny epilogue kernel.
namespace {
__global__ void fp16_bias_add_kernel(__half* __restrict__ Y,
                                     const __half* __restrict__ bias,
                                     int B, int out_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * out_dim;
    if (idx >= total) return;
    const int j = idx % out_dim;
    const float yv = __half2float(Y[idx]);
    const float bv = __half2float(bias[j]);
    Y[idx] = __float2half(yv + bv);
}
} // namespace

void linear_forward_batched_fp16(const ::brotensor::Tensor& W,
                                 const ::brotensor::Tensor* bias,
                                 const ::brotensor::Tensor& X_BD,
                                 ::brotensor::Tensor& Y_BD) {
    if (W.dtype != Dtype::FP16 || X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("linear_forward_batched_fp16: W and X must be FP16");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("linear_forward_batched_fp16: bias must be FP16");
    }
    const int B       = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W.rows;
    if (W.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out_dim, Dtype::FP16);
    }
    if (B == 0 || out_dim == 0) return;

    fp16_internal::launch_matmul_ABT(
        static_cast<const __half*>(X_BD.data),
        static_cast<const __half*>(W.data),
        static_cast<__half*>(Y_BD.data),
        B, out_dim, in_dim);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    if (bias && bias->size() > 0) {
        const int total = B * out_dim;
        const int blocks = (total + 255) / 256;
        fp16_bias_add_kernel<<<blocks, 256>>>(
            static_cast<__half*>(Y_BD.data),
            static_cast<const __half*>(bias->data),
            B, out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB) {
    require_fp(W, "linear_backward", "W");
    require_same_dtype(W, x,  "linear_backward");
    require_same_dtype(W, dY, "linear_backward");
    require_same_dtype(W, dW, "linear_backward");
    require_same_dtype(W, dB, "linear_backward");
    const int out_dim = W.rows;
    const int in_dim  = W.cols;

    if (dX.rows != in_dim || dX.cols != 1 || dX.dtype != W.dtype) {
        dX.resize(in_dim, 1, W.dtype);
    }

    // dX = W^T * dY (overwrite)
    if (in_dim > 0) {
        const int blocks = (in_dim + LB_DX_BLOCK - 1) / LB_DX_BLOCK;
        if (W.dtype == Dtype::FP16) {
            linear_backward_dx_kernel<__half><<<blocks, LB_DX_BLOCK>>>(
                static_cast<const __half*>(W.data),
                static_cast<const __half*>(dY.data),
                static_cast<__half*>(dX.data),
                out_dim, in_dim);
        } else if (W.dtype == Dtype::BF16) {
            linear_backward_dx_kernel<__nv_bfloat16><<<blocks, LB_DX_BLOCK>>>(
                static_cast<const __nv_bfloat16*>(W.data),
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<__nv_bfloat16*>(dX.data),
                out_dim, in_dim);
        } else {
            linear_backward_dx_kernel<float><<<blocks, LB_DX_BLOCK>>>(
                static_cast<const float*>(W.data),
                static_cast<const float*>(dY.data),
                static_cast<float*>(dX.data),
                out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    // dW += dY * x^T
    if (out_dim > 0 && in_dim > 0) {
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        if (W.dtype == Dtype::FP16) {
            linear_backward_dw_kernel<__half><<<grid, block>>>(
                static_cast<const __half*>(dY.data),
                static_cast<const __half*>(x.data),
                static_cast<__half*>(dW.data),
                out_dim, in_dim);
        } else if (W.dtype == Dtype::BF16) {
            linear_backward_dw_kernel<__nv_bfloat16><<<grid, block>>>(
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<__nv_bfloat16*>(dW.data),
                out_dim, in_dim);
        } else {
            linear_backward_dw_kernel<float><<<grid, block>>>(
                static_cast<const float*>(dY.data),
                static_cast<const float*>(x.data),
                static_cast<float*>(dW.data),
                out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    // dB += dY
    if (out_dim > 0) {
        const int blocks = (out_dim + 255) / 256;
        if (W.dtype == Dtype::FP16) {
            linear_backward_db_kernel<__half><<<blocks, 256>>>(
                static_cast<const __half*>(dY.data),
                static_cast<__half*>(dB.data), out_dim);
        } else if (W.dtype == Dtype::BF16) {
            linear_backward_db_kernel<__nv_bfloat16><<<blocks, 256>>>(
                static_cast<const __nv_bfloat16*>(dY.data),
                static_cast<__nv_bfloat16*>(dB.data), out_dim);
        } else {
            linear_backward_db_kernel<float><<<blocks, 256>>>(
                static_cast<const float*>(dY.data),
                static_cast<float*>(dB.data), out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace detail::cuda
} // namespace brotensor
