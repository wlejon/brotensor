// Native AMD HIP BatchNorm suite for RDNA 3.5 (gfx1151).
//
// Supports batch_norm_forward, batch_norm_inference, and batch_norm_backward.

#include "detail/hip_check.h"
#include "detail/norm_internal.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int BN_BLOCK = 256;
constexpr int kWave32 = 32;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

__device__ __forceinline__ float to_f32(float v)           { return v; }
__device__ __forceinline__ float to_f32(__half v)          { return __half2float(v); }
__device__ __forceinline__ float to_f32(hip_bfloat16 v)    { return float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ hip_bfloat16 from_f32<hip_bfloat16>(float v) { return hip_bfloat16(v); }

__device__ __forceinline__ float wave32_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, kWave32);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* sdata) {
    val = wave32_reduce_sum(val);
    const int tid = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        sdata[wave_id] = val;
    }
    __syncthreads();
    float sum = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
    sum = wave32_reduce_sum(sum);
    return sum;
}

// ─── Forward Training ───────────────────────────────────────────────────────

__global__ void bn_forward_kernel(const float* __restrict__ X,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ running_mean,
                                  float* __restrict__ running_var,
                                  float* __restrict__ Y,
                                  float* __restrict__ saved_mean,
                                  float* __restrict__ saved_rstd,
                                  int N, int C, int spatial,
                                  float eps, float momentum) {
    const int c = blockIdx.x;
    if (c >= C) return;
    const int tid = threadIdx.x;
    const int M = N * spatial;
    const float M_f = static_cast<float>(M);
    const float inv_M = 1.0f / M_f;

    __shared__ float sdata[BN_BLOCK >> 5];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    // Pass 1: Mean
    float local_sum = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        const int n = i / spatial;
        const int s = i % spatial;
        const size_t idx = (static_cast<size_t>(n) * C + c) * spatial + s;
        local_sum += X[idx];
    }
    const float tot_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = tot_sum * inv_M;
    __syncthreads();
    const float mean = s_mean;

    // Pass 2: Var
    float local_sq = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        const int n = i / spatial;
        const int s = i % spatial;
        const size_t idx = (static_cast<size_t>(n) * C + c) * spatial + s;
        const float d = X[idx] - mean;
        local_sq += d * d;
    }
    const float tot_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) {
        const float var = tot_sq * inv_M;
        s_rstd = rsqrtf(var + eps);
        saved_mean[c] = mean;
        saved_rstd[c] = s_rstd;
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        const float unbiased_var = (M > 1) ? (tot_sq / static_cast<float>(M - 1)) : var;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    }
    __syncthreads();
    const float rstd = s_rstd;
    const float g = gamma[c];
    const float b = beta[c];

    // Pass 3: Write Y
    for (int i = tid; i < M; i += blockDim.x) {
        const int n = i / spatial;
        const int s = i % spatial;
        const size_t idx = (static_cast<size_t>(n) * C + c) * spatial + s;
        Y[idx] = g * (X[idx] - mean) * rstd + b;
    }
}

// ─── Inference ─────────────────────────────────────────────────────────────

template <typename T>
__global__ void bn_inference_kernel(const T* __restrict__ X,
                                    const T* __restrict__ gamma,
                                    const T* __restrict__ beta,
                                    const T* __restrict__ running_mean,
                                    const T* __restrict__ running_var,
                                    T* __restrict__ Y,
                                    long long total, int C, int spatial,
                                    float eps) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>((i / spatial) % C);
        const float mean = to_f32(running_mean[c]);
        const float var  = to_f32(running_var[c]);
        const float g    = to_f32(gamma[c]);
        const float b    = to_f32(beta[c]);
        const float rstd = rsqrtf(var + eps);
        const float v    = to_f32(X[i]);
        Y[i] = from_f32<T>(g * (v - mean) * rstd + b);
    }
}

// ─── Backward ──────────────────────────────────────────────────────────────

__global__ void bn_backward_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ saved_mean,
                                   const float* __restrict__ saved_rstd,
                                   const float* __restrict__ dY,
                                   int N, int C, int spatial,
                                   float* __restrict__ dX,
                                   float* __restrict__ dGamma,
                                   float* __restrict__ dBeta) {
    const int c = blockIdx.x;
    if (c >= C) return;
    const int tid = threadIdx.x;
    const int M = N * spatial;
    const float M_f = static_cast<float>(M);
    const float inv_M = 1.0f / M_f;

    const float mean = saved_mean[c];
    const float rstd = saved_rstd[c];
    const float g    = gamma[c];

    __shared__ float sdata[BN_BLOCK >> 5];
    __shared__ float s_sum_dy;
    __shared__ float s_sum_dy_xh;

    float local_dy = 0.0f;
    float local_dy_xh = 0.0f;

    for (int i = tid; i < M; i += blockDim.x) {
        const int n = i / spatial;
        const int s = i % spatial;
        const size_t idx = (static_cast<size_t>(n) * C + c) * spatial + s;
        const float dy = dY[idx];
        const float xh = (X[idx] - mean) * rstd;
        local_dy    += dy;
        local_dy_xh += dy * xh;
    }

    const float tot_dy    = block_reduce_sum(local_dy, sdata);
    if (tid == 0) s_sum_dy = tot_dy;
    __syncthreads();

    const float tot_dy_xh = block_reduce_sum(local_dy_xh, sdata);
    if (tid == 0) {
        s_sum_dy_xh = tot_dy_xh;
        dGamma[c]  += tot_dy_xh;
        dBeta[c]   += s_sum_dy;
    }
    __syncthreads();

    const float sum_dy    = s_sum_dy;
    const float sum_dy_xh = s_sum_dy_xh;

    for (int i = tid; i < M; i += blockDim.x) {
        const int n = i / spatial;
        const int s = i % spatial;
        const size_t idx = (static_cast<size_t>(n) * C + c) * spatial + s;
        const float dy = dY[idx];
        const float xh = (X[idx] - mean) * rstd;
        dX[idx] = g * rstd * inv_M * (M_f * dy - sum_dy - xh * sum_dy_xh);
    }
}

} // namespace

void batch_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        Tensor& running_mean, Tensor& running_var,
                        int N, int C, int H, int W, float eps, float momentum,
                        Tensor& Y, Tensor& saved_mean, Tensor& saved_rstd) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32) throw std::runtime_error("batch_norm_forward: X must be FP32");
    if (gamma.dtype != Dtype::FP32 || beta.dtype != Dtype::FP32)
        throw std::runtime_error("batch_norm_forward: gamma/beta must be FP32");
    if (running_mean.dtype != Dtype::FP32 || running_var.dtype != Dtype::FP32)
        throw std::runtime_error("batch_norm_forward: running_mean/var must be FP32");

    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != Dtype::FP32)
        Y.resize(X.rows, X.cols, Dtype::FP32);
    if (saved_mean.size() != C || saved_mean.dtype != Dtype::FP32)
        saved_mean.resize(C, 1, Dtype::FP32);
    if (saved_rstd.size() != C || saved_rstd.dtype != Dtype::FP32)
        saved_rstd.resize(C, 1, Dtype::FP32);

    const int spatial = H * W;
    if (N == 0 || C == 0 || spatial == 0) return;

    hipStream_t stream = cur_stream();
    bn_forward_kernel<<<C, BN_BLOCK, 0, stream>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(gamma.data),
        static_cast<const float*>(beta.data),
        static_cast<float*>(running_mean.data),
        static_cast<float*>(running_var.data),
        static_cast<float*>(Y.data),
        static_cast<float*>(saved_mean.data),
        static_cast<float*>(saved_rstd.data),
        N, C, spatial, eps, momentum);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void batch_norm_inference(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                          const Tensor& running_mean, const Tensor& running_var,
                          int N, int C, int H, int W, float eps, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error("batch_norm_inference: X must be FP32, FP16, or BF16");
    if (gamma.dtype != X.dtype || beta.dtype != X.dtype ||
        running_mean.dtype != X.dtype || running_var.dtype != X.dtype)
        throw std::runtime_error("batch_norm_inference: per-channel params must match X.dtype");

    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype)
        Y.resize(X.rows, X.cols, X.dtype);

    const long long total = static_cast<long long>(N) * C * H * W;
    if (total == 0) return;

    const int spatial = H * W;
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid > 65535) grid = 65535;

    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        bn_inference_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(gamma.data),
            static_cast<const float*>(beta.data), static_cast<const float*>(running_mean.data),
            static_cast<const float*>(running_var.data), static_cast<float*>(Y.data),
            total, C, spatial, eps);
    } else if (X.dtype == Dtype::FP16) {
        bn_inference_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(gamma.data),
            static_cast<const __half*>(beta.data), static_cast<const __half*>(running_mean.data),
            static_cast<const __half*>(running_var.data), static_cast<__half*>(Y.data),
            total, C, spatial, eps);
    } else {
        bn_inference_kernel<hip_bfloat16><<<grid, block, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(gamma.data),
            static_cast<const hip_bfloat16*>(beta.data), static_cast<const hip_bfloat16*>(running_mean.data),
            static_cast<const hip_bfloat16*>(running_var.data), static_cast<hip_bfloat16*>(Y.data),
            total, C, spatial, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void batch_norm_backward(const Tensor& X, const Tensor& gamma,
                         const Tensor& saved_mean, const Tensor& saved_rstd,
                         const Tensor& dY, int N, int C, int H, int W,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 || gamma.dtype != Dtype::FP32 || dY.dtype != Dtype::FP32)
        throw std::runtime_error("batch_norm_backward: tensors must be FP32");
    if (dX.rows != X.rows || dX.cols != X.cols || dX.dtype != Dtype::FP32)
        dX.resize(X.rows, X.cols, Dtype::FP32);
    if (dGamma.size() != C || dGamma.dtype != Dtype::FP32)
        dGamma.resize(C, 1, Dtype::FP32);
    if (dBeta.size() != C || dBeta.dtype != Dtype::FP32)
        dBeta.resize(C, 1, Dtype::FP32);

    const int spatial = H * W;
    if (N == 0 || C == 0 || spatial == 0) return;

    hipStream_t stream = cur_stream();
    bn_backward_kernel<<<C, BN_BLOCK, 0, stream>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(gamma.data),
        static_cast<const float*>(saved_mean.data),
        static_cast<const float*>(saved_rstd.data),
        static_cast<const float*>(dY.data),
        N, C, spatial,
        static_cast<float*>(dX.data),
        static_cast<float*>(dGamma.data),
        static_cast<float*>(dBeta.data));
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
