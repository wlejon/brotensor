// Native AMD HIP LayerNorm and Norms Registration for RDNA 3.5 (gfx1151, Wave32).
//
// LayerNorm forward, backward, inference batched, and batched with caches
// supporting FP32, FP16 (__half), and BF16 (hip_bfloat16).

#include "detail/hip_check.h"
#include "detail/norm_internal.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int LN_BLOCK = 256;
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

// ─── Single-sample LayerNorm Forward ────────────────────────────────────────

template <typename T>
__global__ void layernorm_forward_kernel(const T* __restrict__ x,
                                         const T* __restrict__ gamma,
                                         const T* __restrict__ beta,
                                         T* __restrict__ y,
                                         T* __restrict__ xhat,
                                         float* __restrict__ scratch,
                                         int n, float eps) {
    __shared__ float sdata[LN_BLOCK >> 5];
    __shared__ float s_mean;
    __shared__ float s_rstd;
    const int tid = threadIdx.x;

    // Mean
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local_sum += to_f32(x[i]);
    const float tot_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = tot_sum / static_cast<float>(n);
    __syncthreads();
    const float mean = s_mean;

    // Variance
    float local_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = to_f32(x[i]) - mean;
        local_sq += d * d;
    }
    const float tot_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) {
        const float var = tot_sq / static_cast<float>(n);
        const float rstd = rsqrtf(var + eps);
        s_rstd = rstd;
        scratch[0] = mean;
        scratch[1] = rstd;
    }
    __syncthreads();
    const float rstd = s_rstd;

    for (int i = tid; i < n; i += blockDim.x) {
        const float xh = (to_f32(x[i]) - mean) * rstd;
        xhat[i] = from_f32<T>(xh);
        y[i]    = from_f32<T>(to_f32(gamma[i]) * xh + to_f32(beta[i]));
    }
}

// ─── Single-sample LayerNorm Backward ───────────────────────────────────────

template <typename T>
__global__ void layernorm_backward_kernel(const T* __restrict__ dY,
                                          const T* __restrict__ xhat,
                                          const T* __restrict__ gamma,
                                          float rstd,
                                          T* __restrict__ dX,
                                          float* __restrict__ dGamma_scratch,
                                          float* __restrict__ dBeta_scratch,
                                          int n) {
    __shared__ float sdata[LN_BLOCK >> 5];
    __shared__ float s_sum_dxh;
    __shared__ float s_sum_dxh_xh;
    const int tid = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) {
        const float dy = to_f32(dY[i]);
        const float xh = to_f32(xhat[i]);
        dGamma_scratch[i] = dy * xh;
        dBeta_scratch[i]  = dy;
    }

    float local_dxh = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_dxh += to_f32(dY[i]) * to_f32(gamma[i]);
    }
    const float tot_dxh = block_reduce_sum(local_dxh, sdata);
    if (tid == 0) s_sum_dxh = tot_dxh;
    __syncthreads();

    float local_dxh_xh = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_dxh_xh += to_f32(dY[i]) * to_f32(gamma[i]) * to_f32(xhat[i]);
    }
    const float tot_dxh_xh = block_reduce_sum(local_dxh_xh, sdata);
    if (tid == 0) s_sum_dxh_xh = tot_dxh_xh;
    __syncthreads();

    const float sum_dxh    = s_sum_dxh;
    const float sum_dxh_xh = s_sum_dxh_xh;
    const float nf         = static_cast<float>(n);
    const float scale      = rstd / nf;

    for (int i = tid; i < n; i += blockDim.x) {
        const float dxh = to_f32(dY[i]) * to_f32(gamma[i]);
        const float xh  = to_f32(xhat[i]);
        dX[i] = from_f32<T>(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xh));
    }
}

template <typename T>
__global__ void fold_fp32_into_T(const float* __restrict__ src, T* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_f32<T>(to_f32(dst[i]) + src[i]);
}

// ─── Batched Inference LayerNorm ────────────────────────────────────────────

template <typename T>
__global__ void ln_fwd_inference_batched_kernel(const T* __restrict__ x,
                                                const T* __restrict__ gamma,
                                                const T* __restrict__ beta,
                                                T* __restrict__ y,
                                                int R, int D, float eps) {
    __shared__ float sdata[LN_BLOCK >> 5];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const size_t off = static_cast<size_t>(row) * D;
    const T* xrow = x + off;
    T*       yrow = y + off;

    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local_sum += to_f32(xrow[i]);
    const float tot_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = tot_sum / static_cast<float>(D);
    __syncthreads();
    const float mean = s_mean;

    float local_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = to_f32(xrow[i]) - mean;
        local_sq += d * d;
    }
    const float tot_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) s_rstd = rsqrtf(tot_sq / static_cast<float>(D) + eps);
    __syncthreads();
    const float rstd = s_rstd;

    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (to_f32(xrow[i]) - mean) * rstd;
        const float g  = to_f32(gamma[i]);
        const float b  = beta ? to_f32(beta[i]) : 0.0f;
        yrow[i] = from_f32<T>(xh * g + b);
    }
}

// ─── Batched With Caches LayerNorm ──────────────────────────────────────────

template <typename T>
__global__ void ln_fwd_batched_caches_kernel(const T* __restrict__ X,
                                             const T* __restrict__ gamma,
                                             const T* __restrict__ beta,
                                             T* __restrict__ Y,
                                             T* __restrict__ Xhat,
                                             float* __restrict__ Mean_R,
                                             float* __restrict__ Rstd_R,
                                             int R, int D, float eps) {
    __shared__ float sdata[LN_BLOCK >> 5];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const size_t off = static_cast<size_t>(row) * D;
    const T* xr  = X    + off;
    T*       yr  = Y    + off;
    T*       xhr = Xhat + off;

    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local_sum += to_f32(xr[i]);
    const float tot_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = tot_sum / static_cast<float>(D);
    __syncthreads();
    const float mean = s_mean;

    float local_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = to_f32(xr[i]) - mean;
        local_sq += d * d;
    }
    const float tot_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) {
        const float rstd = rsqrtf(tot_sq / static_cast<float>(D) + eps);
        s_rstd = rstd;
        Mean_R[row] = mean;
        Rstd_R[row] = rstd;
    }
    __syncthreads();
    const float rstd = s_rstd;

    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (to_f32(xr[i]) - mean) * rstd;
        xhr[i] = from_f32<T>(xh);
        yr[i]  = from_f32<T>(to_f32(gamma[i]) * xh + to_f32(beta[i]));
    }
}

template <typename T>
__global__ void ln_bwd_batched_caches_kernel(const T* __restrict__ dY,
                                             const T* __restrict__ Xhat,
                                             const T* __restrict__ gamma,
                                             const float* __restrict__ rstd_R,
                                             T* __restrict__ dX,
                                             float* __restrict__ dGamma_f32,
                                             float* __restrict__ dBeta_f32,
                                             int R, int D) {
    __shared__ float sdata[LN_BLOCK >> 5];
    __shared__ float s_sum_dxh;
    __shared__ float s_sum_dxh_xh;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const size_t off = static_cast<size_t>(row) * D;
    const T* dyr = dY   + off;
    const T* hr  = Xhat + off;
    T*       dxr = dX   + off;
    const float rstd = rstd_R[row];

    float local_dxh = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_dxh += to_f32(dyr[i]) * to_f32(gamma[i]);
    }
    const float tot_dxh = block_reduce_sum(local_dxh, sdata);
    if (tid == 0) s_sum_dxh = tot_dxh;
    __syncthreads();

    float local_dxh_xh = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_dxh_xh += to_f32(dyr[i]) * to_f32(gamma[i]) * to_f32(hr[i]);
    }
    const float tot_dxh_xh = block_reduce_sum(local_dxh_xh, sdata);
    if (tid == 0) s_sum_dxh_xh = tot_dxh_xh;
    __syncthreads();

    const float sum_dxh    = s_sum_dxh;
    const float sum_dxh_xh = s_sum_dxh_xh;
    const float nf         = static_cast<float>(D);
    const float scale      = rstd / nf;

    for (int i = tid; i < D; i += blockDim.x) {
        const float g   = to_f32(dyr[i]);
        const float xh  = to_f32(hr[i]);
        const float dxh = g * to_f32(gamma[i]);
        dxr[i] = from_f32<T>(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xh));
        atomicAdd(&dGamma_f32[i], g * xh);
        atomicAdd(&dBeta_f32[i],  g);
    }
}

} // namespace

void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       Tensor& y, Tensor& xhat, float& mean_out, float& rstd_out, float eps) {
    using ::brotensor::Dtype;
    if (x.dtype != Dtype::FP16 && x.dtype != Dtype::BF16 && x.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward: x must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != x.dtype || beta.dtype != x.dtype) {
        throw std::runtime_error("layernorm_forward: gamma/beta dtype must match x.dtype");
    }
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    if (xhat.rows != x.rows || xhat.cols != x.cols || xhat.dtype != x.dtype) {
        xhat.resize(x.rows, x.cols, x.dtype);
    }
    if (n == 0) {
        mean_out = 0.0f;
        rstd_out = 0.0f;
        return;
    }

    hipStream_t stream = cur_stream();
    float* d_scratch = static_cast<float*>(hip_alloc(2 * sizeof(float)));

    if (x.dtype == Dtype::FP16) {
        layernorm_forward_kernel<__half><<<1, LN_BLOCK, 0, stream>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(gamma.data),
            static_cast<const __half*>(beta.data), static_cast<__half*>(y.data),
            static_cast<__half*>(xhat.data), d_scratch, n, eps);
    } else if (x.dtype == Dtype::BF16) {
        layernorm_forward_kernel<hip_bfloat16><<<1, LN_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(x.data), static_cast<const hip_bfloat16*>(gamma.data),
            static_cast<const hip_bfloat16*>(beta.data), static_cast<hip_bfloat16*>(y.data),
            static_cast<hip_bfloat16*>(xhat.data), d_scratch, n, eps);
    } else {
        layernorm_forward_kernel<float><<<1, LN_BLOCK, 0, stream>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(gamma.data),
            static_cast<const float*>(beta.data), static_cast<float*>(y.data),
            static_cast<float*>(xhat.data), d_scratch, n, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    float h[2] = {0.0f, 0.0f};
    BROTENSOR_HIP_CHECK(hipMemcpyAsync(h, d_scratch, 2 * sizeof(float), hipMemcpyDeviceToHost, stream));
    BROTENSOR_HIP_CHECK(hipStreamSynchronize(stream));
    hip_free(d_scratch);
    mean_out = h[0];
    rstd_out = h[1];
}

void layernorm_backward(const Tensor& dY, const Tensor& xhat, const Tensor& gamma,
                        float rstd, Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::BF16 && dY.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward: dY must be FP16, BF16, or FP32");
    }
    if (xhat.dtype != dY.dtype || gamma.dtype != dY.dtype) {
        throw std::runtime_error("layernorm_backward: xhat/gamma dtype must match dY.dtype");
    }
    const int n = dY.size();
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    if (dGamma.size() != n || dGamma.dtype != dY.dtype) {
        dGamma.resize(gamma.rows, gamma.cols, dY.dtype);
    }
    if (dBeta.size() != n || dBeta.dtype != dY.dtype) {
        dBeta.resize(gamma.rows, gamma.cols, dY.dtype);
    }
    if (n == 0) return;

    hipStream_t stream = cur_stream();
    float* dG_scratch = static_cast<float*>(hip_alloc(n * sizeof(float)));
    float* dB_scratch = static_cast<float*>(hip_alloc(n * sizeof(float)));

    if (dY.dtype == Dtype::FP16) {
        layernorm_backward_kernel<__half><<<1, LN_BLOCK, 0, stream>>>(
            static_cast<const __half*>(dY.data), static_cast<const __half*>(xhat.data),
            static_cast<const __half*>(gamma.data), rstd, static_cast<__half*>(dX.data),
            dG_scratch, dB_scratch, n);
    } else if (dY.dtype == Dtype::BF16) {
        layernorm_backward_kernel<hip_bfloat16><<<1, LN_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY.data), static_cast<const hip_bfloat16*>(xhat.data),
            static_cast<const hip_bfloat16*>(gamma.data), rstd, static_cast<hip_bfloat16*>(dX.data),
            dG_scratch, dB_scratch, n);
    } else {
        layernorm_backward_kernel<float><<<1, LN_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), static_cast<const float*>(xhat.data),
            static_cast<const float*>(gamma.data), rstd, static_cast<float*>(dX.data),
            dG_scratch, dB_scratch, n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    const int fold_blocks = (n + LN_BLOCK - 1) / LN_BLOCK;
    if (dY.dtype == Dtype::FP32) {
        fold_fp32_into_T<float><<<fold_blocks, LN_BLOCK, 0, stream>>>(dG_scratch, static_cast<float*>(dGamma.data), n);
        fold_fp32_into_T<float><<<fold_blocks, LN_BLOCK, 0, stream>>>(dB_scratch, static_cast<float*>(dBeta.data),  n);
    } else if (dY.dtype == Dtype::FP16) {
        fold_fp32_into_T<__half><<<fold_blocks, LN_BLOCK, 0, stream>>>(dG_scratch, static_cast<__half*>(dGamma.data), n);
        fold_fp32_into_T<__half><<<fold_blocks, LN_BLOCK, 0, stream>>>(dB_scratch, static_cast<__half*>(dBeta.data),  n);
    } else {
        fold_fp32_into_T<hip_bfloat16><<<fold_blocks, LN_BLOCK, 0, stream>>>(dG_scratch, static_cast<hip_bfloat16*>(dGamma.data), n);
        fold_fp32_into_T<hip_bfloat16><<<fold_blocks, LN_BLOCK, 0, stream>>>(dB_scratch, static_cast<hip_bfloat16*>(dBeta.data),  n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(dG_scratch);
    hip_free(dB_scratch);
}

void layernorm_forward_inference_batched_fp16(const Tensor& X_RD, const Tensor& gamma,
                                              const Tensor* beta, Tensor& Y_RD, float eps) {
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 || gamma.dtype != Dtype::FP16 || (beta && beta->dtype != Dtype::FP16)) {
        throw std::runtime_error("layernorm_forward_inference_batched_fp16: all tensors must be FP16");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != Dtype::FP16) {
        Y_RD.resize(R, D, Dtype::FP16);
    }
    if (R == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    ln_fwd_inference_batched_kernel<__half><<<R, LN_BLOCK, 0, stream>>>(
        static_cast<const __half*>(X_RD.data),
        static_cast<const __half*>(gamma.data),
        beta ? static_cast<const __half*>(beta->data) : nullptr,
        static_cast<__half*>(Y_RD.data),
        R, D, eps);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void layernorm_forward_inference_batched(const Tensor& X_RD, const Tensor& gamma,
                                         const Tensor* beta, Tensor& Y_RD, float eps) {
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 && X_RD.dtype != Dtype::BF16 && X_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward_inference_batched: X must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != X_RD.dtype || (beta && beta->dtype != X_RD.dtype)) {
        throw std::runtime_error("layernorm_forward_inference_batched: gamma/beta dtype must match X.dtype");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != X_RD.dtype) {
        Y_RD.resize(R, D, X_RD.dtype);
    }
    if (R == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    if (X_RD.dtype == Dtype::FP16) {
        ln_fwd_inference_batched_kernel<__half><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X_RD.data),
            static_cast<const __half*>(gamma.data),
            beta ? static_cast<const __half*>(beta->data) : nullptr,
            static_cast<__half*>(Y_RD.data),
            R, D, eps);
    } else if (X_RD.dtype == Dtype::BF16) {
        ln_fwd_inference_batched_kernel<hip_bfloat16><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X_RD.data),
            static_cast<const hip_bfloat16*>(gamma.data),
            beta ? static_cast<const hip_bfloat16*>(beta->data) : nullptr,
            static_cast<hip_bfloat16*>(Y_RD.data),
            R, D, eps);
    } else {
        ln_fwd_inference_batched_kernel<float><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const float*>(X_RD.data),
            static_cast<const float*>(gamma.data),
            beta ? static_cast<const float*>(beta->data) : nullptr,
            static_cast<float*>(Y_RD.data),
            R, D, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void layernorm_forward_batched_with_caches(const Tensor& X_RD, const Tensor& gamma, const Tensor& beta,
                                           Tensor& Y_RD, Tensor& Xhat_RD,
                                           Tensor& Mean_R, Tensor& Rstd_R, float eps) {
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 && X_RD.dtype != Dtype::BF16 && X_RD.dtype != Dtype::FP32) {
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

    hipStream_t stream = cur_stream();
    if (X_RD.dtype == Dtype::FP16) {
        ln_fwd_batched_caches_kernel<__half><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X_RD.data), static_cast<const __half*>(gamma.data),
            static_cast<const __half*>(beta.data), static_cast<__half*>(Y_RD.data),
            static_cast<__half*>(Xhat_RD.data), static_cast<float*>(Mean_R.data),
            static_cast<float*>(Rstd_R.data), R, D, eps);
    } else if (X_RD.dtype == Dtype::BF16) {
        ln_fwd_batched_caches_kernel<hip_bfloat16><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X_RD.data), static_cast<const hip_bfloat16*>(gamma.data),
            static_cast<const hip_bfloat16*>(beta.data), static_cast<hip_bfloat16*>(Y_RD.data),
            static_cast<hip_bfloat16*>(Xhat_RD.data), static_cast<float*>(Mean_R.data),
            static_cast<float*>(Rstd_R.data), R, D, eps);
    } else {
        ln_fwd_batched_caches_kernel<float><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const float*>(X_RD.data), static_cast<const float*>(gamma.data),
            static_cast<const float*>(beta.data), static_cast<float*>(Y_RD.data),
            static_cast<float*>(Xhat_RD.data), static_cast<float*>(Mean_R.data),
            static_cast<float*>(Rstd_R.data), R, D, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void layernorm_backward_batched_with_caches(const Tensor& dY_RD, const Tensor& Xhat_RD,
                                            const Tensor& gamma, const Tensor& Rstd_R,
                                            Tensor& dX_RD, Tensor& dGamma, Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (dY_RD.dtype != Dtype::FP16 && dY_RD.dtype != Dtype::BF16 && dY_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY must be FP16, BF16, or FP32");
    }
    if (Xhat_RD.dtype != dY_RD.dtype || gamma.dtype != dY_RD.dtype) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: Xhat/gamma dtype must match dY.dtype");
    }
    if (Rstd_R.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: Rstd_R must be FP32");
    }
    const int R = dY_RD.rows;
    const int D = dY_RD.cols;
    if (dX_RD.rows != R || dX_RD.cols != D || dX_RD.dtype != dY_RD.dtype) {
        dX_RD.resize(R, D, dY_RD.dtype);
    }
    if (dGamma.size() != D || dGamma.dtype != dY_RD.dtype) {
        dGamma.resize(gamma.rows, gamma.cols, dY_RD.dtype);
    }
    if (dBeta.size() != D || dBeta.dtype != dY_RD.dtype) {
        dBeta.resize(gamma.rows, gamma.cols, dY_RD.dtype);
    }
    if (R == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    float* dG_f32 = static_cast<float*>(hip_alloc(D * sizeof(float)));
    float* dB_f32 = static_cast<float*>(hip_alloc(D * sizeof(float)));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dG_f32, 0, D * sizeof(float), stream));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dB_f32, 0, D * sizeof(float), stream));

    if (dY_RD.dtype == Dtype::FP16) {
        ln_bwd_batched_caches_kernel<__half><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const __half*>(dY_RD.data), static_cast<const __half*>(Xhat_RD.data),
            static_cast<const __half*>(gamma.data), static_cast<const float*>(Rstd_R.data),
            static_cast<__half*>(dX_RD.data), dG_f32, dB_f32, R, D);
    } else if (dY_RD.dtype == Dtype::BF16) {
        ln_bwd_batched_caches_kernel<hip_bfloat16><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY_RD.data), static_cast<const hip_bfloat16*>(Xhat_RD.data),
            static_cast<const hip_bfloat16*>(gamma.data), static_cast<const float*>(Rstd_R.data),
            static_cast<hip_bfloat16*>(dX_RD.data), dG_f32, dB_f32, R, D);
    } else {
        ln_bwd_batched_caches_kernel<float><<<R, LN_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY_RD.data), static_cast<const float*>(Xhat_RD.data),
            static_cast<const float*>(gamma.data), static_cast<const float*>(Rstd_R.data),
            static_cast<float*>(dX_RD.data), dG_f32, dB_f32, R, D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    const int fold_blocks = (D + LN_BLOCK - 1) / LN_BLOCK;
    if (dY_RD.dtype == Dtype::FP32) {
        fold_fp32_into_T<float><<<fold_blocks, LN_BLOCK, 0, stream>>>(dG_f32, static_cast<float*>(dGamma.data), D);
        fold_fp32_into_T<float><<<fold_blocks, LN_BLOCK, 0, stream>>>(dB_f32, static_cast<float*>(dBeta.data),  D);
    } else if (dY_RD.dtype == Dtype::FP16) {
        fold_fp32_into_T<__half><<<fold_blocks, LN_BLOCK, 0, stream>>>(dG_f32, static_cast<__half*>(dGamma.data), D);
        fold_fp32_into_T<__half><<<fold_blocks, LN_BLOCK, 0, stream>>>(dB_f32, static_cast<__half*>(dBeta.data),  D);
    } else {
        fold_fp32_into_T<hip_bfloat16><<<fold_blocks, LN_BLOCK, 0, stream>>>(dG_f32, static_cast<hip_bfloat16*>(dGamma.data), D);
        fold_fp32_into_T<hip_bfloat16><<<fold_blocks, LN_BLOCK, 0, stream>>>(dB_f32, static_cast<hip_bfloat16*>(dBeta.data),  D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(dG_f32);
    hip_free(dB_f32);
}

void fill_hip_vtable_norms(::brotensor::detail::OpsVTable& v) {
    v.rms_norm_forward                        = &rms_norm_forward;
    v.rms_norm_backward                       = &rms_norm_backward;
    v.layernorm_forward                       = &layernorm_forward;
    v.layernorm_backward                      = &layernorm_backward;
    v.layernorm_forward_inference_batched     = &layernorm_forward_inference_batched;
    v.layernorm_forward_inference_batched_fp16 = &layernorm_forward_inference_batched_fp16;
    v.layernorm_forward_batched_with_caches   = &layernorm_forward_batched_with_caches;
    v.layernorm_backward_batched_with_caches  = &layernorm_backward_batched_with_caches;
    v.group_norm_forward                      = &group_norm_forward;
    v.group_norm_backward                     = &group_norm_backward;
    v.l2_norm_forward                         = &l2_norm_forward;
    v.l2_norm_backward                        = &l2_norm_backward;
    v.l2_normalize_nchw_forward               = &l2_normalize_nchw_forward;
    v.batch_norm_forward                      = &batch_norm_forward;
    v.batch_norm_inference                    = &batch_norm_inference;
    v.batch_norm_backward                     = &batch_norm_backward;
}

} // namespace brotensor::detail::hip
