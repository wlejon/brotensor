#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

#include "detail/cuda_check.h"

namespace brotensor {

namespace {

constexpr int RS_BLOCK = 256;

// ─── Forward kernels (FP16 + FP32) ─────────────────────────────────────────

template <typename T>
__device__ inline float to_f32(T v);
template <> __device__ inline float to_f32<float>(float v)   { return v; }
template <> __device__ inline float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ inline T from_f32(float v);
template <> __device__ inline float  from_f32<float>(float v)  { return v; }
template <> __device__ inline __half from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ void upsample_nearest_2x_kernel(const T* __restrict__ X,
                                           T* __restrict__ Y,
                                           int N, int C, int H, int W,
                                           int H_out, int W_out,
                                           int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;
    const int ih = oh / 2;
    const int iw = ow / 2;
    const int in_idx = ((n * C + c) * H + ih) * W + iw;
    Y[idx] = X[in_idx];
}

template <typename T>
__global__ void upsample_bilinear_2x_kernel(const T* __restrict__ X,
                                            T* __restrict__ Y,
                                            int N, int C, int H, int W,
                                            int H_out, int W_out,
                                            int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;

    const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
    const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
    const int y0 = static_cast<int>(floorf(src_y));
    const int x0 = static_cast<int>(floorf(src_x));
    const float fy = src_y - y0;
    const float fx = src_x - x0;
    const int y0c = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);
    const int x0c = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);
    const int y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= H ? H - 1 : (y0 + 1));
    const int x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= W ? W - 1 : (x0 + 1));

    const int base = (n * C + c) * H;
    const float v00 = to_f32<T>(X[(base + y0c) * W + x0c]);
    const float v01 = to_f32<T>(X[(base + y0c) * W + x1c]);
    const float v10 = to_f32<T>(X[(base + y1c) * W + x0c]);
    const float v11 = to_f32<T>(X[(base + y1c) * W + x1c]);
    const float top = v00 + (v01 - v00) * fx;
    const float bot = v10 + (v11 - v10) * fx;
    const float v   = top + (bot - top) * fy;
    Y[idx] = from_f32<T>(v);
}

template <typename T>
__global__ void downsample_avg_2x_kernel(const T* __restrict__ X,
                                         T* __restrict__ Y,
                                         int N, int C, int H, int W,
                                         int H_out, int W_out,
                                         int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;
    const int ih = oh * 2;
    const int iw = ow * 2;
    const int base = ((n * C + c) * H + ih) * W + iw;
    const float v00 = to_f32<T>(X[base]);
    const float v01 = to_f32<T>(X[base + 1]);
    const float v10 = to_f32<T>(X[base + W]);
    const float v11 = to_f32<T>(X[base + W + 1]);
    Y[idx] = from_f32<T>(0.25f * (v00 + v01 + v10 + v11));
}

// ─── Backward kernels ──────────────────────────────────────────────────────

// One thread per INPUT pixel; gather sum of 4 dY values, no atomics.
template <typename T>
__global__ void upsample_nearest_2x_backward_kernel(const T* __restrict__ dY,
                                                    T* __restrict__ dX,
                                                    int N, int C, int H, int W,
                                                    int H_out, int W_out,
                                                    int total_in) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in) return;
    const int iw = idx % W;
    int t = idx / W;
    const int ih = t % H;
    t /= H;
    const int c  = t % C;
    const int n  = t / C;
    const int base = (n * C + c) * H_out;
    const int oh0 = 2 * ih;
    const int ow0 = 2 * iw;
    const float v00 = to_f32<T>(dY[(base + oh0    ) * W_out + ow0    ]);
    const float v01 = to_f32<T>(dY[(base + oh0    ) * W_out + ow0 + 1]);
    const float v10 = to_f32<T>(dY[(base + oh0 + 1) * W_out + ow0    ]);
    const float v11 = to_f32<T>(dY[(base + oh0 + 1) * W_out + ow0 + 1]);
    dX[idx] = from_f32<T>(v00 + v01 + v10 + v11);
}

// One thread per OUTPUT pixel; atomicAdd into dX (FP32 path). For the FP16
// path, we run this kernel against an FP32 scratch buffer and then fold back.
__global__ void upsample_bilinear_2x_backward_scatter_fp32(
        const float* __restrict__ dY,
        float* __restrict__ dX,
        int N, int C, int H, int W,
        int H_out, int W_out,
        int total_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;

    const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
    const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
    const int y0 = static_cast<int>(floorf(src_y));
    const int x0 = static_cast<int>(floorf(src_x));
    const float fy = src_y - y0;
    const float fx = src_x - x0;
    const int y0c = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);
    const int x0c = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);
    const int y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= H ? H - 1 : (y0 + 1));
    const int x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= W ? W - 1 : (x0 + 1));

    const float w00 = (1.0f - fy) * (1.0f - fx);
    const float w01 = (1.0f - fy) * fx;
    const float w10 = fy * (1.0f - fx);
    const float w11 = fy * fx;
    const float g = dY[idx];

    const int base = (n * C + c) * H;
    atomicAdd(&dX[(base + y0c) * W + x0c], w00 * g);
    atomicAdd(&dX[(base + y0c) * W + x1c], w01 * g);
    atomicAdd(&dX[(base + y1c) * W + x0c], w10 * g);
    atomicAdd(&dX[(base + y1c) * W + x1c], w11 * g);
}

// FP16-source variant: reads FP16 dY but scatters into an FP32 scratch via
// atomicAdd, since FP16 atomicAdd isn't portable.
__global__ void upsample_bilinear_2x_backward_scatter_fp16(
        const __half* __restrict__ dY,
        float* __restrict__ dX_f32,
        int N, int C, int H, int W,
        int H_out, int W_out,
        int total_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;

    const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
    const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
    const int y0 = static_cast<int>(floorf(src_y));
    const int x0 = static_cast<int>(floorf(src_x));
    const float fy = src_y - y0;
    const float fx = src_x - x0;
    const int y0c = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);
    const int x0c = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);
    const int y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= H ? H - 1 : (y0 + 1));
    const int x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= W ? W - 1 : (x0 + 1));

    const float w00 = (1.0f - fy) * (1.0f - fx);
    const float w01 = (1.0f - fy) * fx;
    const float w10 = fy * (1.0f - fx);
    const float w11 = fy * fx;
    const float g = __half2float(dY[idx]);

    const int base = (n * C + c) * H;
    atomicAdd(&dX_f32[(base + y0c) * W + x0c], w00 * g);
    atomicAdd(&dX_f32[(base + y0c) * W + x1c], w01 * g);
    atomicAdd(&dX_f32[(base + y1c) * W + x0c], w10 * g);
    atomicAdd(&dX_f32[(base + y1c) * W + x1c], w11 * g);
}

// BF16-source variant: verbatim copy of fp16 scatter with __nv_bfloat16.
__global__ void upsample_bilinear_2x_backward_scatter_bf16(
        const __nv_bfloat16* __restrict__ dY,
        float* __restrict__ dX_f32,
        int N, int C, int H, int W,
        int H_out, int W_out,
        int total_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;

    const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
    const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
    const int y0 = static_cast<int>(floorf(src_y));
    const int x0 = static_cast<int>(floorf(src_x));
    const float fy = src_y - y0;
    const float fx = src_x - x0;
    const int y0c = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);
    const int x0c = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);
    const int y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= H ? H - 1 : (y0 + 1));
    const int x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= W ? W - 1 : (x0 + 1));

    const float w00 = (1.0f - fy) * (1.0f - fx);
    const float w01 = (1.0f - fy) * fx;
    const float w10 = fy * (1.0f - fx);
    const float w11 = fy * fx;
    const float g = __bfloat162float(dY[idx]);

    const int base = (n * C + c) * H;
    atomicAdd(&dX_f32[(base + y0c) * W + x0c], w00 * g);
    atomicAdd(&dX_f32[(base + y0c) * W + x1c], w01 * g);
    atomicAdd(&dX_f32[(base + y1c) * W + x0c], w10 * g);
    atomicAdd(&dX_f32[(base + y1c) * W + x1c], w11 * g);
}

// Fold an FP32 buffer into FP16 storage (overwrite).
__global__ void copy_fp32_to_fp16(const float* __restrict__ src,
                                  __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

// Fold an FP32 buffer into BF16 storage (overwrite).
__global__ void copy_fp32_to_bf16(const float* __restrict__ src,
                                  __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

// One thread per INPUT pixel; reads its single output-pixel grad, scales 1/4.
template <typename T>
__global__ void downsample_avg_2x_backward_kernel(const T* __restrict__ dY,
                                                  T* __restrict__ dX,
                                                  int N, int C, int H, int W,
                                                  int H_out, int W_out,
                                                  int total_in) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in) return;
    const int iw = idx % W;
    int t = idx / W;
    const int ih = t % H;
    t /= H;
    const int c  = t % C;
    const int n  = t / C;
    const int oh = ih / 2;
    const int ow = iw / 2;
    const int out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
    dX[idx] = from_f32<T>(0.25f * to_f32<T>(dY[out_idx]));
}

inline int grid_for(int n) {
    int b = (n + RS_BLOCK - 1) / RS_BLOCK;
    if (b < 1) b = 1;
    return b;
}

inline void check_dtype_fp(const ::brotensor::Tensor& t,
                           const char* op, const char* name) {
    if (t.dtype != Dtype::FP16 && t.dtype != Dtype::FP32 && t.dtype != Dtype::BF16) {
        throw std::runtime_error(std::string(op) + ": " + name + " must be FP16, BF16, or FP32");
    }
}

} // namespace

namespace detail::cuda {

// ─── Forward ───────────────────────────────────────────────────────────────

void upsample_nearest_2x(const ::brotensor::Tensor& X,
                         int N, int C, int H, int W,
                         ::brotensor::Tensor& Y) {
    check_dtype_fp(X, "upsample_nearest_2x", "X");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        upsample_nearest_2x_kernel<__half><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    } else if (X.dtype == Dtype::BF16) {
        upsample_nearest_2x_kernel<__nv_bfloat16><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    } else {
        upsample_nearest_2x_kernel<float><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void upsample_bilinear_2x(const ::brotensor::Tensor& X,
                          int N, int C, int H, int W,
                          ::brotensor::Tensor& Y) {
    check_dtype_fp(X, "upsample_bilinear_2x", "X");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        upsample_bilinear_2x_kernel<__half><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    } else if (X.dtype == Dtype::BF16) {
        upsample_bilinear_2x_kernel<__nv_bfloat16><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    } else {
        upsample_bilinear_2x_kernel<float><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void downsample_avg_2x(const ::brotensor::Tensor& X,
                       int N, int C, int H, int W,
                       ::brotensor::Tensor& Y) {
    check_dtype_fp(X, "downsample_avg_2x", "X");
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x: H and W must be even");
    }
    const int H_out = H / 2, W_out = W / 2;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        downsample_avg_2x_kernel<__half><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    } else if (X.dtype == Dtype::BF16) {
        downsample_avg_2x_kernel<__nv_bfloat16><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    } else {
        downsample_avg_2x_kernel<float><<<grid_for(total), RS_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            N, C, H, W, H_out, W_out, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── Backward ──────────────────────────────────────────────────────────────

void upsample_nearest_2x_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  ::brotensor::Tensor& dX) {
    check_dtype_fp(dY, "upsample_nearest_2x_backward", "dY");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    const int total_in = N * cols_in;
    if (total_in == 0) return;
    if (dY.dtype == Dtype::FP16) {
        upsample_nearest_2x_backward_kernel<__half><<<grid_for(total_in), RS_BLOCK>>>(
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            N, C, H, W, H_out, W_out, total_in);
    } else if (dY.dtype == Dtype::BF16) {
        upsample_nearest_2x_backward_kernel<__nv_bfloat16><<<grid_for(total_in), RS_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data),
            N, C, H, W, H_out, W_out, total_in);
    } else {
        upsample_nearest_2x_backward_kernel<float><<<grid_for(total_in), RS_BLOCK>>>(
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            N, C, H, W, H_out, W_out, total_in);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void upsample_bilinear_2x_backward(const ::brotensor::Tensor& dY,
                                   int N, int C, int H, int W,
                                   ::brotensor::Tensor& dX) {
    check_dtype_fp(dY, "upsample_bilinear_2x_backward", "dY");
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols_in = C * H * W;
    const int cols_out = C * H_out * W_out;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    const int total_in = N * cols_in;
    const int total_out = N * cols_out;
    if (total_out == 0) return;

    if (dY.dtype == Dtype::FP32) {
        // Zero dX, then atomic-scatter directly.
        BROTENSOR_CUDA_CHECK(cudaMemset(dX.data, 0, total_in * sizeof(float)));
        upsample_bilinear_2x_backward_scatter_fp32<<<grid_for(total_out), RS_BLOCK>>>(
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            N, C, H, W, H_out, W_out, total_out);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else if (dY.dtype == Dtype::FP16) {
        // FP16 storage: allocate FP32 scratch, scatter into it, fold back.
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        total_in * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemset(d_scratch, 0, total_in * sizeof(float)));
        upsample_bilinear_2x_backward_scatter_fp16<<<grid_for(total_out), RS_BLOCK>>>(
            static_cast<const __half*>(dY.data),
            d_scratch, N, C, H, W, H_out, W_out, total_out);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        copy_fp32_to_fp16<<<grid_for(total_in), RS_BLOCK>>>(
            d_scratch, static_cast<__half*>(dX.data), total_in);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    } else {
        // BF16 storage: same FP32-scratch pattern as FP16.
        float* d_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                        total_in * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemset(d_scratch, 0, total_in * sizeof(float)));
        upsample_bilinear_2x_backward_scatter_bf16<<<grid_for(total_out), RS_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(dY.data),
            d_scratch, N, C, H, W, H_out, W_out, total_out);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        copy_fp32_to_bf16<<<grid_for(total_in), RS_BLOCK>>>(
            d_scratch, static_cast<__nv_bfloat16*>(dX.data), total_in);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_scratch);
    }
}

void downsample_avg_2x_backward(const ::brotensor::Tensor& dY,
                                int N, int C, int H, int W,
                                ::brotensor::Tensor& dX) {
    check_dtype_fp(dY, "downsample_avg_2x_backward", "dY");
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x_backward: H and W must be even");
    }
    const int H_out = H / 2, W_out = W / 2;
    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    const int total_in = N * cols_in;
    if (total_in == 0) return;
    if (dY.dtype == Dtype::FP16) {
        downsample_avg_2x_backward_kernel<__half><<<grid_for(total_in), RS_BLOCK>>>(
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            N, C, H, W, H_out, W_out, total_in);
    } else if (dY.dtype == Dtype::BF16) {
        downsample_avg_2x_backward_kernel<__nv_bfloat16><<<grid_for(total_in), RS_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(dY.data),
            static_cast<__nv_bfloat16*>(dX.data),
            N, C, H, W, H_out, W_out, total_in);
    } else {
        downsample_avg_2x_backward_kernel<float><<<grid_for(total_in), RS_BLOCK>>>(
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            N, C, H, W, H_out, W_out, total_in);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda

} // namespace brotensor
