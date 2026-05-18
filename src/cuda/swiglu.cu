// SwiGLU activation (Llama FFN gate):
//   X is (B, 2D); split along last dim into A=(B,D) and B_half=(B,D).
//   Y(B, D) = silu(A) * B_half = (A * sigmoid(A)) * B_half.
// Mirrors elementwise.cu's geglu pattern exactly.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

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

inline int grid_for(int n) { return (n + SG_BLOCK - 1) / SG_BLOCK; }

} // namespace

void swiglu_forward_gpu(const GpuTensor& X, GpuTensor& Y) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        swiglu_forward_fp16_kernel<<<grid_for(total), SG_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<__half*>(Y.data_fp16()),
            B, D);
    } else {
        swiglu_forward_fp32_kernel<<<grid_for(total), SG_BLOCK>>>(
            X.data, Y.data, B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void swiglu_backward_gpu(const GpuTensor& X, const GpuTensor& dY,
                        GpuTensor& dX) {
    if (X.cols % 2 != 0) {
        throw std::runtime_error("swiglu_backward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (dX.rows != B || dX.cols != 2 * D || dX.dtype != X.dtype) {
        dX.resize(B, 2 * D, X.dtype);
    }
    const int total = B * D;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        swiglu_backward_fp16_kernel<<<grid_for(total), SG_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<const __half*>(dY.data_fp16()),
            reinterpret_cast<__half*>(dX.data_fp16()),
            B, D);
    } else {
        swiglu_backward_fp32_kernel<<<grid_for(total), SG_BLOCK>>>(
            X.data, dY.data, dX.data, B, D);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
