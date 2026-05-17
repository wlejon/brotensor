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

__global__ void add_inplace_fp16_kernel(__half* __restrict__ y,
                                        const __half* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float a = __half2float(y[i]);
        const float b = __half2float(x[i]);
        y[i] = __float2half(a + b);
    }
}

__global__ void scale_inplace_fp16_kernel(__half* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(__half2float(y[i]) * s);
    }
}

__global__ void add_scalar_inplace_fp16_kernel(__half* __restrict__ y, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(__half2float(y[i]) + s);
    }
}

__global__ void clamp_fp32_kernel(float* __restrict__ y, float lo, float hi, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = y[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        y[i] = v;
    }
}

__global__ void clamp_fp16_kernel(__half* __restrict__ y, float lo, float hi, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = __half2float(y[i]);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        y[i] = __float2half(v);
    }
}

__device__ inline float quick_gelu_scalar(float v) {
    // OpenAI CLIP's QuickGELU: x * sigmoid(1.702 * x).
    return v / (1.0f + __expf(-1.702f * v));
}

__global__ void quick_gelu_forward_fp32_kernel(const float* __restrict__ x,
                                               float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = quick_gelu_scalar(x[i]);
    }
}

__global__ void quick_gelu_forward_fp16_kernel(const __half* __restrict__ x,
                                               __half* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = __float2half(quick_gelu_scalar(__half2float(x[i])));
    }
}

__global__ void mul_inplace_fp32_kernel(float* __restrict__ y,
                                        const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] *= x[i];
    }
}

__global__ void mul_inplace_fp16_kernel(__half* __restrict__ y,
                                        const __half* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float a = __half2float(y[i]);
        const float b = __half2float(x[i]);
        y[i] = __float2half(a * b);
    }
}

// Y(B, D) = X_a(B, D) * gelu(X_b(B, D)) where X is (B, 2D) — A is the first
// half along the last dim, B is the second half.
__global__ void geglu_forward_fp16_kernel(const __half* __restrict__ X,
                                          __half* __restrict__ Y,
                                          int B, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * D;
    if (idx >= total) return;
    const int b = idx / D;
    const int d = idx % D;
    const int two_d = 2 * D;
    const float a = __half2float(X[b * two_d + d]);
    const float gv_raw = __half2float(X[b * two_d + D + d]);
    Y[idx] = __float2half(a * gelu_tanh_scalar(gv_raw));
}

__global__ void causal_mask_row_kernel(float* __restrict__ mask, int L, int q) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= L) return;
    mask[k] = (k <= q) ? 1.0f : 0.0f;
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
    if (y.dtype == Dtype::FP16) {
        if (x.dtype != Dtype::FP16) {
            throw std::runtime_error("add_inplace_gpu: dtype mismatch");
        }
        add_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<__half*>(y.data_fp16()),
            reinterpret_cast<const __half*>(x.data_fp16()), n);
    } else {
        add_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, x.data, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_scalar_inplace_gpu(GpuTensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        add_scalar_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<__half*>(y.data_fp16()), s, n);
    } else {
        add_scalar_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, s, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void clamp_gpu(GpuTensor& y, float lo, float hi) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        clamp_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<__half*>(y.data_fp16()), lo, hi, n);
    } else {
        clamp_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, lo, hi, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void scale_inplace_gpu(GpuTensor& y, float s) {
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        scale_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<__half*>(y.data_fp16()), s, n);
    } else {
        scale_inplace_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, s, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void mul_inplace_gpu(GpuTensor& y, const GpuTensor& x) {
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("mul_inplace_gpu: shape/dtype mismatch");
    }
    const int n = y.size();
    if (n == 0) return;
    if (y.dtype == Dtype::FP16) {
        mul_inplace_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<__half*>(y.data_fp16()),
            reinterpret_cast<const __half*>(x.data_fp16()), n);
    } else {
        mul_inplace_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(y.data, x.data, n);
    }
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

void quick_gelu_forward_gpu(const GpuTensor& x, GpuTensor& y) {
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    const int n = x.size();
    if (n == 0) return;
    if (x.dtype == Dtype::FP16) {
        quick_gelu_forward_fp16_kernel<<<grid_for(n), EW_BLOCK>>>(
            reinterpret_cast<const __half*>(x.data_fp16()),
            reinterpret_cast<__half*>(y.data_fp16()), n);
    } else {
        quick_gelu_forward_fp32_kernel<<<grid_for(n), EW_BLOCK>>>(x.data, y.data, n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void geglu_forward_gpu(const GpuTensor& X, GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("geglu_forward_gpu: X must be FP16");
    }
    if (X.cols % 2 != 0) {
        throw std::runtime_error("geglu_forward_gpu: X.cols must be even (2*D)");
    }
    const int B = X.rows;
    const int D = X.cols / 2;
    if (Y.rows != B || Y.cols != D || Y.dtype != Dtype::FP16) {
        Y.resize(B, D, Dtype::FP16);
    }
    const int total = B * D;
    if (total == 0) return;
    geglu_forward_fp16_kernel<<<grid_for(total), EW_BLOCK>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        B, D);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void build_causal_mask_row_gpu(int L, int q, GpuTensor& mask) {
    if (mask.rows != L || mask.cols != 1 || mask.dtype != Dtype::FP32) {
        mask.resize(L, 1, Dtype::FP32);
    }
    if (L <= 0) return;
    const int blocks = (L + EW_BLOCK - 1) / EW_BLOCK;
    causal_mask_row_kernel<<<blocks, EW_BLOCK>>>(mask.data, L, q);
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
