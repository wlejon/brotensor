#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor::detail::cuda {

namespace {

constexpr int GN_BLOCK = 256;

// ─── Forward kernels ───────────────────────────────────────────────────────
//
// One block per (sample, group) tile. Threads cooperatively reduce sum and
// sum-of-squares over the tile (channels_per_group * spatial elements), then
// normalize using per-channel gamma/beta. FP32 accumulation; storage dtype
// templated.

__global__ void group_norm_forward_kernel_fp16(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        const __half* __restrict__ beta,
        __half* __restrict__ Y,
        int C, int spatial,
        int channels_per_group,
        float eps) {
    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    const int tile_channels = channels_per_group;
    const int tile_size = tile_channels * spatial;
    const int chan_base = g * channels_per_group;
    const int sample_stride = C * spatial;
    const __half* x_tile = X + n * sample_stride + chan_base * spatial;
    __half*       y_tile = Y + n * sample_stride + chan_base * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = __half2float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }

    __shared__ float s_sum[GN_BLOCK];
    __shared__ float s_sumsq[GN_BLOCK];
    s_sum[tid]   = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean  = s_sum[0] * inv_n;
        const float var   = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = __half2float(gamma[channel]);
        const float bv = __half2float(beta[channel]);
        const float v  = __half2float(x_tile[i]);
        const float yn = (v - mean) * rstd;
        y_tile[i] = __float2half(yn * gv + bv);
    }
}

__global__ void group_norm_forward_kernel_fp32(
        const float* __restrict__ X,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ Y,
        int C, int spatial,
        int channels_per_group,
        float eps) {
    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    const int tile_size = channels_per_group * spatial;
    const int chan_base = g * channels_per_group;
    const int sample_stride = C * spatial;
    const float* x_tile = X + n * sample_stride + chan_base * spatial;
    float*       y_tile = Y + n * sample_stride + chan_base * spatial;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = x_tile[i];
        sum   += v;
        sumsq += v * v;
    }

    __shared__ float s_sum[GN_BLOCK];
    __shared__ float s_sumsq[GN_BLOCK];
    s_sum[tid]   = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean  = s_sum[0] * inv_n;
        const float var   = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = gamma[channel];
        const float bv = beta[channel];
        const float v  = x_tile[i];
        const float yn = (v - mean) * rstd;
        y_tile[i] = yn * gv + bv;
    }
}

// ─── Backward kernels ──────────────────────────────────────────────────────
//
// One block per (sample, group) tile. Three passes:
//   pass 1: reduce sum, sumsq over X tile → recompute mean, rstd.
//   pass 2: reduce sum1 = Σ dx̂  and sum2 = Σ dx̂ · x̂  over the tile,
//           where dx̂ = dY * γ_c and x̂ = (x - mean) * rstd.
//   pass 3: write dX = rstd * (dx̂ - (sum1 + x̂ * sum2) / M).
//
// Per-channel dGamma_c, dBeta_c are atomicAdded in FP32 scratch buffers
// (one per launch) and the host converts back to the storage dtype.

__global__ void group_norm_backward_kernel_fp16(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        const __half* __restrict__ dY,
        __half* __restrict__ dX,
        float* __restrict__ dGamma_acc,  // FP32 scratch, length C
        float* __restrict__ dBeta_acc,   // FP32 scratch, length C
        int C, int spatial,
        int channels_per_group,
        float eps) {
    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    const int tile_size = channels_per_group * spatial;
    const int chan_base = g * channels_per_group;
    const int sample_stride = C * spatial;
    const __half* x_tile  = X  + n * sample_stride + chan_base * spatial;
    const __half* dy_tile = dY + n * sample_stride + chan_base * spatial;
    __half*       dx_tile = dX + n * sample_stride + chan_base * spatial;

    // Pass 1: mean, var.
    float sum = 0.0f, sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = __half2float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }
    __shared__ float s_a[GN_BLOCK];
    __shared__ float s_b[GN_BLOCK];
    s_a[tid] = sum; s_b[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
            s_b[tid] += s_b[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean = s_a[0] * inv_n;
        const float var  = s_b[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    // Pass 2: sum1 = Σ dx̂, sum2 = Σ dx̂ * x̂.
    // Also accumulate dGamma/dBeta per channel (atomic into FP32 scratch).
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv  = __half2float(gamma[channel]);
        const float dyv = __half2float(dy_tile[i]);
        const float xv  = __half2float(x_tile[i]);
        const float xh  = (xv - mean) * rstd;
        const float dxh = dyv * gv;
        sum1 += dxh;
        sum2 += dxh * xh;
        // Per-channel accumulators (scattered across spatial within tile).
        atomicAdd(&dGamma_acc[channel], dyv * xh);
        atomicAdd(&dBeta_acc[channel],  dyv);
    }
    s_a[tid] = sum1; s_b[tid] = sum2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
            s_b[tid] += s_b[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_sum1, s_sum2;
    if (tid == 0) { s_sum1 = s_a[0]; s_sum2 = s_b[0]; }
    __syncthreads();
    const float sum1_t = s_sum1;
    const float sum2_t = s_sum2;
    const float inv_M = 1.0f / static_cast<float>(tile_size);

    // Pass 3: dX = rstd * (dx̂ - (sum1 + x̂ * sum2) / M).
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv  = __half2float(gamma[channel]);
        const float dyv = __half2float(dy_tile[i]);
        const float xv  = __half2float(x_tile[i]);
        const float xh  = (xv - mean) * rstd;
        const float dxh = dyv * gv;
        const float dx  = rstd * (dxh - (sum1_t + xh * sum2_t) * inv_M);
        dx_tile[i] = __float2half(dx);
    }
}

__global__ void group_norm_backward_kernel_fp32(
        const float* __restrict__ X,
        const float* __restrict__ gamma,
        const float* __restrict__ dY,
        float* __restrict__ dX,
        float* __restrict__ dGamma_acc,
        float* __restrict__ dBeta_acc,
        int C, int spatial,
        int channels_per_group,
        float eps) {
    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    const int tile_size = channels_per_group * spatial;
    const int chan_base = g * channels_per_group;
    const int sample_stride = C * spatial;
    const float* x_tile  = X  + n * sample_stride + chan_base * spatial;
    const float* dy_tile = dY + n * sample_stride + chan_base * spatial;
    float*       dx_tile = dX + n * sample_stride + chan_base * spatial;

    float sum = 0.0f, sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = x_tile[i];
        sum   += v;
        sumsq += v * v;
    }
    __shared__ float s_a[GN_BLOCK];
    __shared__ float s_b[GN_BLOCK];
    s_a[tid] = sum; s_b[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
            s_b[tid] += s_b[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean = s_a[0] * inv_n;
        const float var  = s_b[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv  = gamma[channel];
        const float dyv = dy_tile[i];
        const float xv  = x_tile[i];
        const float xh  = (xv - mean) * rstd;
        const float dxh = dyv * gv;
        sum1 += dxh;
        sum2 += dxh * xh;
        atomicAdd(&dGamma_acc[channel], dyv * xh);
        atomicAdd(&dBeta_acc[channel],  dyv);
    }
    s_a[tid] = sum1; s_b[tid] = sum2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
            s_b[tid] += s_b[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_sum1, s_sum2;
    if (tid == 0) { s_sum1 = s_a[0]; s_sum2 = s_b[0]; }
    __syncthreads();
    const float sum1_t = s_sum1;
    const float sum2_t = s_sum2;
    const float inv_M = 1.0f / static_cast<float>(tile_size);

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv  = gamma[channel];
        const float dyv = dy_tile[i];
        const float xv  = x_tile[i];
        const float xh  = (xv - mean) * rstd;
        const float dxh = dyv * gv;
        dx_tile[i] = rstd * (dxh - (sum1_t + xh * sum2_t) * inv_M);
    }
}

// Add FP32 scratch into FP16/FP32 dGamma/dBeta accumulators (caller-owned,
// previously zeroed). Storage dtype dispatched.
__global__ void add_fp32_into_fp16(const float* __restrict__ src,
                                   __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float prev = __half2float(dst[i]);
    dst[i] = __float2half(prev + src[i]);
}

__global__ void add_fp32_into_fp32(const float* __restrict__ src,
                                   float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

} // namespace

void group_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        int N, int C, int H, int W,
                        int num_groups,
                        float eps,
                        ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (gamma.dtype != X.dtype || beta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_forward: gamma/beta dtype must match X");
    }
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("group_norm_forward: X must be FP16 or FP32");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward: num_groups must divide C");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const int channels_per_group = C / num_groups;
    dim3 grid(num_groups, N, 1);
    if (X.dtype == Dtype::FP16) {
        group_norm_forward_kernel_fp16<<<grid, GN_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data),
            reinterpret_cast<const __half*>(gamma.data),
            reinterpret_cast<const __half*>(beta.data),
            reinterpret_cast<__half*>(Y.data),
            C, spatial, channels_per_group, eps);
    } else {
        group_norm_forward_kernel_fp32<<<grid, GN_BLOCK>>>(
            reinterpret_cast<const float*>(X.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(beta.data),
            reinterpret_cast<float*>(Y.data),
            C, spatial, channels_per_group, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void group_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W,
                         int num_groups,
                         float eps,
                         ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: gamma/dY dtype must match X");
    }
    if (X.dtype != Dtype::FP16 && X.dtype != Dtype::FP32) {
        throw std::runtime_error("group_norm_backward: X must be FP16 or FP32");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_backward: num_groups must divide C");
    }
    if (dGamma.dtype != X.dtype || dBeta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: dGamma/dBeta dtype must match X");
    }
    if (dGamma.rows != C || dGamma.cols != 1 ||
        dBeta.rows  != C || dBeta.cols  != 1) {
        throw std::runtime_error("group_norm_backward: dGamma/dBeta must be (C,1)");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (dX.rows != N || dX.cols != cols || dX.dtype != X.dtype) {
        dX.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    // FP32 scratch for per-channel grads — accumulate atomically in the
    // backward kernel, then add into the storage-dtype caller buffers.
    float* d_dG = nullptr;
    float* d_dB = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dG), C * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dB), C * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMemset(d_dG, 0, C * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMemset(d_dB, 0, C * sizeof(float)));

    const int channels_per_group = C / num_groups;
    dim3 grid(num_groups, N, 1);
    if (X.dtype == Dtype::FP16) {
        group_norm_backward_kernel_fp16<<<grid, GN_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data),
            reinterpret_cast<const __half*>(gamma.data),
            reinterpret_cast<const __half*>(dY.data),
            reinterpret_cast<__half*>(dX.data),
            d_dG, d_dB,
            C, spatial, channels_per_group, eps);
    } else {
        group_norm_backward_kernel_fp32<<<grid, GN_BLOCK>>>(
            reinterpret_cast<const float*>(X.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(dY.data),
            reinterpret_cast<float*>(dX.data),
            d_dG, d_dB,
            C, spatial, channels_per_group, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    const int block = 128;
    const int gridc = (C + block - 1) / block;
    if (X.dtype == Dtype::FP16) {
        add_fp32_into_fp16<<<gridc, block>>>(
            d_dG, reinterpret_cast<__half*>(dGamma.data), C);
        add_fp32_into_fp16<<<gridc, block>>>(
            d_dB, reinterpret_cast<__half*>(dBeta.data),  C);
    } else {
        add_fp32_into_fp32<<<gridc, block>>>(d_dG, reinterpret_cast<float*>(dGamma.data), C);
        add_fp32_into_fp32<<<gridc, block>>>(d_dB, reinterpret_cast<float*>(dBeta.data),  C);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    cudaFree(d_dG);
    cudaFree(d_dB);
}

} // namespace brotensor::detail::cuda
