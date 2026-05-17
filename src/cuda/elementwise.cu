#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int EW_BLOCK = 256;

__device__ inline float silu_scalar(float v) {
    return v / (1.0f + __expf(-v));
}

__device__ inline float gelu_tanh_scalar(float v) {
    // GELU with tanh approximation (matches PyTorch's approximate="tanh").
    constexpr float kSqrt2OverPi = 0.7978845608f;
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(u));
}

__global__ void relu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
    }
}

__global__ void tanh_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = tanhf(x[i]);
    }
}

__global__ void tanh_backward_kernel(const float* __restrict__ y,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * (1.0f - yv * yv);
    }
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ y,
                                        const float* __restrict__ dY,
                                        float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * yv * (1.0f - yv);
    }
}

__global__ void add_inplace_kernel(float* __restrict__ y,
                                   const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += x[i];
    }
}

__global__ void add_scalar_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += s;
    }
}

__global__ void scale_inplace_kernel(float* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] *= s;
    }
}

__global__ void build_slot_mask_kernel(const float* __restrict__ x,
                                       float* __restrict__ mask,
                                       int offset, int K, int stride) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const float v = x[offset + k * stride];
    mask[k] = v > 0.5f ? 1.0f : 0.0f;
}

__global__ void silu_forward_fp32_kernel(const float* __restrict__ x,
                                         float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = silu_scalar(x[i]);
    }
}

__global__ void silu_forward_fp16_kernel(const __half* __restrict__ x,
                                         __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(silu_scalar(__half2float(x[i])));
    }
}

__global__ void gelu_forward_fp32_kernel(const float* __restrict__ x,
                                         float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = gelu_tanh_scalar(x[i]);
    }
}

__global__ void gelu_forward_fp16_kernel(const __half* __restrict__ x,
                                         __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(gelu_tanh_scalar(__half2float(x[i])));
    }
}

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

} // anonymous namespace

void relu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    relu_forward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_backward_gpu(const GpuTensor& x, const GpuTensor& dY, GpuTensor& dX) {
    if (dX.rows != x.rows || dX.cols != x.cols) dX.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    relu_backward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, dY.data, dX.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    tanh_forward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols) dX.resize(y.rows, y.cols);
    const int n = y.size();
    if (n == 0) return;
    tanh_backward_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, dY.data, dX.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    const int n = x.size();
    if (n == 0) return;
    sigmoid_forward_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward_gpu(const GpuTensor& y, const GpuTensor& dY, GpuTensor& dX) {
    if (dX.rows != y.rows || dX.cols != y.cols) dX.resize(y.rows, y.cols);
    const int n = y.size();
    if (n == 0) return;
    sigmoid_backward_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, dY.data, dX.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    const int n = y.size();
    if (n == 0) return;
    add_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, x.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_scalar_inplace_gpu(GpuTensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    add_scalar_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, s, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void scale_inplace_gpu(GpuTensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    scale_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, s, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void silu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        silu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<const __half*>(x.data_fp16()),
            reinterpret_cast<__half*>(y.data_fp16()), n);
    } else {
        silu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void gelu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        gelu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<const __half*>(x.data_fp16()),
            reinterpret_cast<__half*>(y.data_fp16()), n);
    } else {
        gelu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_slot_mask_gpu(const GpuTensor& x, int offset, int K, int stride,
                         GpuTensor& mask) {
    if (mask.rows != K || mask.cols != 1) mask.resize(K, 1);
    if (K <= 0) return;
    const int blocks = (K + EW_BLOCK - 1) / EW_BLOCK;
    build_slot_mask_kernel<<<blocks, EW_BLOCK>>>(x.data, mask.data,
                                                 offset, K, stride);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
