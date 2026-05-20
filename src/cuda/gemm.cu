#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include "fp16_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {
namespace detail::cuda {

namespace {

// y[i] = b[i] + sum_j W[i, j] * x[j]
// One thread per output row. Uses shared memory to cache tiles of x.
constexpr int LF_BLOCK = 128;
constexpr int LF_TILE  = 128;

__global__ void linear_forward_kernel(const float* __restrict__ W,
                                      const float* __restrict__ b,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int out_dim, int in_dim) {
    __shared__ float xtile[LF_TILE];
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < in_dim; t0 += LF_TILE) {
        const int t_len = (in_dim - t0) < LF_TILE ? (in_dim - t0) : LF_TILE;

        // Cooperatively load tile of x into shared memory.
        for (int k = tid; k < t_len; k += blockDim.x) {
            xtile[k] = x[t0 + k];
        }
        __syncthreads();

        if (row < out_dim) {
            const float* wrow = W + static_cast<size_t>(row) * in_dim + t0;
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += wrow[k] * xtile[k];
            }
        }
        __syncthreads();
    }

    if (row < out_dim) {
        y[row] = b[row] + acc;
    }
}

// dX[j] = sum_i W[i, j] * dY[i]
// One thread per input column. Uses shared memory to cache tiles of dY.
constexpr int LB_DX_BLOCK = 128;
constexpr int LB_DX_TILE  = 128;

__global__ void linear_backward_dx_kernel(const float* __restrict__ W,
                                          const float* __restrict__ dY,
                                          float* __restrict__ dX,
                                          int out_dim, int in_dim) {
    __shared__ float dytile[LB_DX_TILE];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int t0 = 0; t0 < out_dim; t0 += LB_DX_TILE) {
        const int t_len = (out_dim - t0) < LB_DX_TILE ? (out_dim - t0) : LB_DX_TILE;

        for (int k = tid; k < t_len; k += blockDim.x) {
            dytile[k] = dY[t0 + k];
        }
        __syncthreads();

        if (col < in_dim) {
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += W[static_cast<size_t>(t0 + k) * in_dim + col] * dytile[k];
            }
        }
        __syncthreads();
    }

    if (col < in_dim) {
        dX[col] = acc;
    }
}

// dW[i, j] += dY[i] * x[j]. 2D grid: each thread one (i, j).
__global__ void linear_backward_dw_kernel(const float* __restrict__ dY,
                                          const float* __restrict__ x,
                                          float* __restrict__ dW,
                                          int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    dW[static_cast<size_t>(i) * in_dim + j] += dY[i] * x[j];
}

// dB[i] += dY[i].
__global__ void linear_backward_db_kernel(const float* __restrict__ dY,
                                          float* __restrict__ dB,
                                          int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    dB[i] += dY[i];
}

} // anonymous namespace

void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    if (y.rows != out_dim || y.cols != 1) y.resize(out_dim, 1);
    if (out_dim == 0) return;

    const int blocks = (out_dim + LF_BLOCK - 1) / LF_BLOCK;
    linear_forward_kernel<<<blocks, LF_BLOCK>>>(
        static_cast<const float*>(W.data),
        static_cast<const float*>(b.data),
        static_cast<const float*>(x.data),
        static_cast<float*>(y.data),
        out_dim, in_dim);
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
    const int out_dim = W.rows;
    const int in_dim  = W.cols;

    if (dX.rows != in_dim || dX.cols != 1) dX.resize(in_dim, 1);

    // dX = W^T * dY (overwrite)
    if (in_dim > 0) {
        const int blocks = (in_dim + LB_DX_BLOCK - 1) / LB_DX_BLOCK;
        linear_backward_dx_kernel<<<blocks, LB_DX_BLOCK>>>(
            static_cast<const float*>(W.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            out_dim, in_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dW += dY * x^T
    if (out_dim > 0 && in_dim > 0) {
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        linear_backward_dw_kernel<<<grid, block>>>(
            static_cast<const float*>(dY.data),
            static_cast<const float*>(x.data),
            static_cast<float*>(dW.data),
            out_dim, in_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dB += dY
    if (out_dim > 0) {
        const int blocks = (out_dim + 255) / 256;
        linear_backward_db_kernel<<<blocks, 256>>>(
            static_cast<const float*>(dY.data),
            static_cast<float*>(dB.data),
            out_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace detail::cuda
} // namespace brotensor
