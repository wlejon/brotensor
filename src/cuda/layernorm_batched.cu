#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

namespace {

constexpr int LN_BLOCK = 256;

inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

template <typename T>
__device__ inline float to_f32(T x);
template <> __device__ inline float to_f32<float>(float x) { return x; }
template <> __device__ inline float to_f32<__half>(__half x) { return __half2float(x); }
template <> __device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_f32(float x);
template <> __device__ inline float from_f32<float>(float x) { return x; }
template <> __device__ inline __half from_f32<__half>(float x) { return __float2half(x); }
template <> __device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T>
__global__ void ln_fwd_batched_caches_kernel(const T* __restrict__ x,
                                             const T* __restrict__ gamma,
                                             const T* __restrict__ beta,
                                             T* __restrict__ y,
                                             T* __restrict__ xhat,
                                             float* __restrict__ mean_out,
                                             float* __restrict__ rstd_out,
                                             int R, int D, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;
    const T* xrow = x    + static_cast<size_t>(row) * D;
    T*       yrow = y    + static_cast<size_t>(row) * D;
    T*       hrow = xhat + static_cast<size_t>(row) * D;

    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local += to_f32<T>(xrow[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(D);
    __syncthreads();   // barrier before reusing sdata for variance — see FP32 variant

    float local_v = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = to_f32<T>(xrow[i]) - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var  = sdata[0] / static_cast<float>(D);
    const float rstd = rsqrtf(var + eps);

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (to_f32<T>(xrow[i]) - mean) * rstd;
        hrow[i] = from_f32<T>(xh);
        const float g = to_f32<T>(gamma[i]);
        const float b = to_f32<T>(beta[i]);
        yrow[i] = from_f32<T>(g * xh + b);
    }
}

// Backward kernel template. dGamma/dBeta accumulators are FP32 (either the
// caller's dGamma/dBeta for the FP32 path, or per-call FP32 scratch for the
// FP16/BF16 paths).
template <typename T>
__global__ void ln_bwd_batched_caches_kernel(const T* __restrict__ dY,
                                             const T* __restrict__ xhat,
                                             const T* __restrict__ gamma,
                                             const float* __restrict__ rstd_R,
                                             T* __restrict__ dX,
                                             float* __restrict__ dGamma_f32,
                                             float* __restrict__ dBeta_f32,
                                             int R, int D) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;
    const T* dyr = dY   + static_cast<size_t>(row) * D;
    const T* hr  = xhat + static_cast<size_t>(row) * D;
    T*       dxr = dX   + static_cast<size_t>(row) * D;
    const float rstd = rstd_R[row];

    // sum_dxh
    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local += to_f32<T>(dyr[i]) * to_f32<T>(gamma[i]);
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh = sdata[0];

    // sum_dxh_xhat
    float local2 = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local2 += to_f32<T>(dyr[i]) * to_f32<T>(gamma[i]) * to_f32<T>(hr[i]);
    }
    sdata[tid] = local2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh_xhat = sdata[0];

    const float nf = static_cast<float>(D);
    const float scale = rstd / nf;
    for (int i = tid; i < D; i += blockDim.x) {
        const float g  = to_f32<T>(dyr[i]);
        const float xh = to_f32<T>(hr[i]);
        const float dxh = g * to_f32<T>(gamma[i]);
        dxr[i] = from_f32<T>(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
        atomicAdd(&dGamma_f32[i], g * xh);
        atomicAdd(&dBeta_f32[i],  g);
    }
}

__global__ void ln_fold_fp32_into_fp16(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}

__global__ void ln_fold_fp32_into_bf16(const float* __restrict__ src,
                                       __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

} // namespace

void layernorm_forward_batched_with_caches(const ::brotensor::Tensor& X_RD,
                                           const ::brotensor::Tensor& gamma,
                                           const ::brotensor::Tensor& beta,
                                           ::brotensor::Tensor& Y_RD,
                                           ::brotensor::Tensor& Xhat_RD,
                                           ::brotensor::Tensor& Mean_R,
                                           ::brotensor::Tensor& Rstd_R,
                                           float eps) {
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 && X_RD.dtype != Dtype::BF16 &&
        X_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward_batched_with_caches: X must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != X_RD.dtype || beta.dtype != X_RD.dtype) {
        throw std::runtime_error("layernorm_forward_batched_with_caches: gamma/beta dtype must match X.dtype");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != X_RD.dtype) {
        Y_RD.resize(R, D, X_RD.dtype);
    }
    if (Xhat_RD.rows != R || Xhat_RD.cols != D || Xhat_RD.dtype != X_RD.dtype) {
        Xhat_RD.resize(R, D, X_RD.dtype);
    }
    if (Mean_R.rows != R || Mean_R.cols != 1 || Mean_R.dtype != Dtype::FP32) {
        Mean_R.resize(R, 1, Dtype::FP32);
    }
    if (Rstd_R.rows != R || Rstd_R.cols != 1 || Rstd_R.dtype != Dtype::FP32) {
        Rstd_R.resize(R, 1, Dtype::FP32);
    }
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    if (X_RD.dtype == Dtype::FP16) {
        ln_fwd_batched_caches_kernel<__half><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const __half*>(X_RD.data),
            reinterpret_cast<const __half*>(gamma.data),
            reinterpret_cast<const __half*>(beta.data),
            reinterpret_cast<__half*>(Y_RD.data),
            reinterpret_cast<__half*>(Xhat_RD.data),
            reinterpret_cast<float*>(Mean_R.data),
            reinterpret_cast<float*>(Rstd_R.data),
            R, D, eps);
    } else if (X_RD.dtype == Dtype::BF16) {
        ln_fwd_batched_caches_kernel<__nv_bfloat16><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(X_RD.data),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data),
            reinterpret_cast<const __nv_bfloat16*>(beta.data),
            reinterpret_cast<__nv_bfloat16*>(Y_RD.data),
            reinterpret_cast<__nv_bfloat16*>(Xhat_RD.data),
            reinterpret_cast<float*>(Mean_R.data),
            reinterpret_cast<float*>(Rstd_R.data),
            R, D, eps);
    } else {
        ln_fwd_batched_caches_kernel<float><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const float*>(X_RD.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(beta.data),
            reinterpret_cast<float*>(Y_RD.data),
            reinterpret_cast<float*>(Xhat_RD.data),
            reinterpret_cast<float*>(Mean_R.data),
            reinterpret_cast<float*>(Rstd_R.data),
            R, D, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward_batched_with_caches(const ::brotensor::Tensor& dY_RD,
                                            const ::brotensor::Tensor& Xhat_RD,
                                            const ::brotensor::Tensor& gamma,
                                            const ::brotensor::Tensor& Rstd_R,
                                            ::brotensor::Tensor& dX_RD,
                                            ::brotensor::Tensor& dGamma,
                                            ::brotensor::Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (dY_RD.dtype != Dtype::FP16 && dY_RD.dtype != Dtype::BF16 &&
        dY_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY must be FP16, BF16, or FP32");
    }
    if (Xhat_RD.dtype != dY_RD.dtype || gamma.dtype != dY_RD.dtype ||
        dGamma.dtype != dY_RD.dtype || dBeta.dtype != dY_RD.dtype) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY/Xhat/gamma/dGamma/dBeta must share dtype");
    }
    if (Rstd_R.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: Rstd_R must be FP32");
    }
    const int R = dY_RD.rows;
    const int D = dY_RD.cols;
    if (dX_RD.rows != R || dX_RD.cols != D || dX_RD.dtype != dY_RD.dtype) {
        dX_RD.resize(R, D, dY_RD.dtype);
    }
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);

    if (dY_RD.dtype == Dtype::FP32) {
        ln_bwd_batched_caches_kernel<float><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const float*>(dY_RD.data),
            reinterpret_cast<const float*>(Xhat_RD.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(Rstd_R.data),
            reinterpret_cast<float*>(dX_RD.data),
            reinterpret_cast<float*>(dGamma.data),
            reinterpret_cast<float*>(dBeta.data),
            R, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else {
        float* d_dg = static_cast<float*>(cuda_alloc(static_cast<size_t>(D) * sizeof(float)));
        float* d_db = static_cast<float*>(cuda_alloc(static_cast<size_t>(D) * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_dg, 0, D * sizeof(float), cur_stream()));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_db, 0, D * sizeof(float), cur_stream()));
        if (dY_RD.dtype == Dtype::FP16) {
            ln_bwd_batched_caches_kernel<__half><<<R, block, shmem, cur_stream()>>>(
                reinterpret_cast<const __half*>(dY_RD.data),
                reinterpret_cast<const __half*>(Xhat_RD.data),
                reinterpret_cast<const __half*>(gamma.data),
                reinterpret_cast<const float*>(Rstd_R.data),
                reinterpret_cast<__half*>(dX_RD.data),
                d_dg, d_db, R, D);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            const int blocks = (D + 255) / 256;
            ln_fold_fp32_into_fp16<<<blocks, 256, 0, cur_stream()>>>(
                d_dg, reinterpret_cast<__half*>(dGamma.data), D);
            ln_fold_fp32_into_fp16<<<blocks, 256, 0, cur_stream()>>>(
                d_db, reinterpret_cast<__half*>(dBeta.data), D);
        } else {
            ln_bwd_batched_caches_kernel<__nv_bfloat16><<<R, block, shmem, cur_stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(dY_RD.data),
                reinterpret_cast<const __nv_bfloat16*>(Xhat_RD.data),
                reinterpret_cast<const __nv_bfloat16*>(gamma.data),
                reinterpret_cast<const float*>(Rstd_R.data),
                reinterpret_cast<__nv_bfloat16*>(dX_RD.data),
                d_dg, d_db, R, D);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            const int blocks = (D + 255) / 256;
            ln_fold_fp32_into_bf16<<<blocks, 256, 0, cur_stream()>>>(
                d_dg, reinterpret_cast<__nv_bfloat16*>(dGamma.data), D);
            ln_fold_fp32_into_bf16<<<blocks, 256, 0, cur_stream()>>>(
                d_db, reinterpret_cast<__nv_bfloat16*>(dBeta.data), D);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(d_dg);
        cuda_free(d_db);
    }
}

} // namespace brotensor::detail::cuda
