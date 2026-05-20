// SwiGLU activation (Llama FFN gate):
//   X is (B, 2D); split along last dim into A=(B,D) and B_half=(B,D).
//   Y(B, D) = silu(A) * B_half = (A * sigmoid(A)) * B_half.
// Mirrors elementwise.cu's geglu pattern exactly.

#include <brotensor/tensor.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor::detail::cuda {

namespace {

constexpr int SG_BLOCK = 256;

__device__ inline float silu_scalar(float v) {
    return v / (1.0f + __expf(-v));
}
__device__ inline float silu_grad_scalar(float v) {
    const float s = 1.0f / (1.0f + __expf(-v));
    return s * (1.0f + v * (1.0f - s));
}

__global__ void swiglu_forward_fp32_kernel(const float* __restrict__ X,
                                           float* __restrict__ Y,
                                           int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a  = X[b * two_d + d];
        const float bh = X[b * two_d + D + d];
        Y[idx] = silu_scalar(a) * bh;
    }
}

__global__ void swiglu_forward_fp16_kernel(const __half* __restrict__ X,
                                           __half* __restrict__ Y,
                                           int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a  = __half2float(X[b * two_d + d]);
        const float bh = __half2float(X[b * two_d + D + d]);
        Y[idx] = __float2half(silu_scalar(a) * bh);
    }
}

__global__ void swiglu_backward_fp32_kernel(const float* __restrict__ X,
                                            const float* __restrict__ dY,
                                            float* __restrict__ dX,
                                            int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a   = X[b * two_d + d];
        const float bh  = X[b * two_d + D + d];
        const float dy  = dY[idx];
        const float s   = silu_scalar(a);
        const float sp  = silu_grad_scalar(a);
        dX[b * two_d + d]     = dy * bh * sp;
        dX[b * two_d + D + d] = dy * s;
    }
}

__global__ void swiglu_backward_fp16_kernel(const __half* __restrict__ X,
                                            const __half* __restrict__ dY,
                                            __half* __restrict__ dX,
                                            int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a   = __half2float(X[b * two_d + d]);
        const float bh  = __half2float(X[b * two_d + D + d]);
        const float dy  = __half2float(dY[idx]);
        const float s   = silu_scalar(a);
        const float sp  = silu_grad_scalar(a);
        dX[b * two_d + d]     = __float2half(dy * bh * sp);
        dX[b * two_d + D + d] = __float2half(dy * s);
    }
}

__global__ void swiglu_forward_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                           __nv_bfloat16* __restrict__ Y,
                                           int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a  = __bfloat162float(X[b * two_d + d]);
        const float bh = __bfloat162float(X[b * two_d + D + d]);
        Y[idx] = __float2bfloat16(silu_scalar(a) * bh);
    }
}

__global__ void swiglu_backward_bf16_kernel(const __nv_bfloat16* __restrict__ X,
                                            const __nv_bfloat16* __restrict__ dY,
                                            __nv_bfloat16* __restrict__ dX,
                                            int B, int D) {
    const int total = B * D;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int b = idx / D;
        const int d = idx % D;
        const int two_d = 2 * D;
        const float a   = __bfloat162float(X[b * two_d + d]);
        const float bh  = __bfloat162float(X[b * two_d + D + d]);
        const float dy  = __bfloat162float(dY[idx]);
        const float s   = silu_scalar(a);
        const float sp  = silu_grad_scalar(a);
        dX[b * two_d + d]     = __float2bfloat16(dy * bh * sp);
        dX[b * two_d + D + d] = __float2bfloat16(dy * s);
    }
}

inline int grid_for(int n) { return (n + SG_BLOCK - 1) / SG_BLOCK; }

} // namespace

void swiglu_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_forward: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        swiglu_forward_fp16_kernel<<<grid_for(total), SG_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            B, D);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        swiglu_forward_bf16_kernel<<<grid_for(total), SG_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            B, D);
    } else {
        swiglu_forward_fp32_kernel<<<grid_for(total), SG_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void swiglu_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_backward: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        swiglu_backward_fp16_kernel<<<grid_for(total), SG_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            B, D);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        swiglu_backward_bf16_kernel<<<grid_for(total), SG_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data),
            B, D);
    } else {
        swiglu_backward_fp32_kernel<<<grid_for(total), SG_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
