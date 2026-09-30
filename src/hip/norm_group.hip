// Native AMD HIP GroupNorm forward & backward for Diffusion UNet / VAE.
//
// Supports FP32, FP16 (__half), and BF16 (hip_bfloat16).
// One block per (sample, group) tile.

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

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int GN_BLOCK = 256;
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

template <typename T>
__global__ void group_norm_forward_kernel(const T* __restrict__ X,
                                          const T* __restrict__ gamma,
                                          const T* __restrict__ beta,
                                          T* __restrict__ Y,
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
    const T* x_tile = X + static_cast<size_t>(n) * sample_stride + static_cast<size_t>(chan_base) * spatial;
    T*       y_tile = Y + static_cast<size_t>(n) * sample_stride + static_cast<size_t>(chan_base) * spatial;

    __shared__ float s_red[GN_BLOCK >> 5];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    float sum = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        sum += to_f32(x_tile[i]);
    }
    const float tot_sum = block_reduce_sum(sum, s_red);
    if (tid == 0) {
        s_mean = tot_sum / static_cast<float>(tile_size);
    }
    __syncthreads();
    const float mean = s_mean;

    float sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float d = to_f32(x_tile[i]) - mean;
        sumsq += d * d;
    }
    const float tot_sumsq = block_reduce_sum(sumsq, s_red);
    if (tid == 0) {
        s_rstd = rsqrtf(tot_sumsq / static_cast<float>(tile_size) + eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = to_f32(gamma[channel]);
        const float bv = to_f32(beta[channel]);
        const float v  = to_f32(x_tile[i]);
        const float yn = (v - mean) * rstd;
        y_tile[i] = from_f32<T>(yn * gv + bv);
    }
}

template <typename T>
__global__ void group_norm_backward_kernel(const T* __restrict__ X,
                                           const T* __restrict__ gamma,
                                           const T* __restrict__ dY,
                                           T* __restrict__ dX,
                                           float* __restrict__ dGamma_accum,
                                           float* __restrict__ dBeta_accum,
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

    const T* x_tile  = X  + static_cast<size_t>(n) * sample_stride + static_cast<size_t>(chan_base) * spatial;
    const T* dy_tile = dY + static_cast<size_t>(n) * sample_stride + static_cast<size_t>(chan_base) * spatial;
    T*       dx_tile = dX + static_cast<size_t>(n) * sample_stride + static_cast<size_t>(chan_base) * spatial;

    __shared__ float s_red[GN_BLOCK >> 5];
    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ float s_sum_dxh;
    __shared__ float s_sum_dxh_xh;

    // Pass 1: Mean
    float sum_x = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        sum_x += to_f32(x_tile[i]);
    }
    const float tot_x = block_reduce_sum(sum_x, s_red);
    if (tid == 0) {
        s_mean = tot_x / static_cast<float>(tile_size);
    }
    __syncthreads();
    const float mean = s_mean;

    // Pass 2: Rstd
    float sum_sq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float d = to_f32(x_tile[i]) - mean;
        sum_sq += d * d;
    }
    const float tot_sq = block_reduce_sum(sum_sq, s_red);
    if (tid == 0) {
        s_rstd = rsqrtf(tot_sq / static_cast<float>(tile_size) + eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    // Dynamic shared memory for per-block staging of dGamma / dBeta partials
    extern __shared__ float smem_staging[];
    float* s_dG = smem_staging;
    float* s_dB = smem_staging + channels_per_group;

    for (int c = tid; c < channels_per_group; c += blockDim.x) {
        s_dG[c] = 0.0f;
        s_dB[c] = 0.0f;
    }
    __syncthreads();

    // Pass 3: sum_dxh, sum_dxh_xh, and thread-local channel accumulation
    float local_sum_dxh = 0.0f;
    float local_sum_dxh_xh = 0.0f;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = to_f32(gamma[channel]);
        const float dy = to_f32(dy_tile[i]);
        const float xh = (to_f32(x_tile[i]) - mean) * rstd;
        const float dxh = dy * gv;

        local_sum_dxh += dxh;
        local_sum_dxh_xh += dxh * xh;

        atomicAdd(&s_dG[local_c], dy * xh);
        atomicAdd(&s_dB[local_c], dy);
    }

    const float tot_dxh = block_reduce_sum(local_sum_dxh, s_red);
    if (tid == 0) s_sum_dxh = tot_dxh;
    __syncthreads();

    const float tot_dxh_xh = block_reduce_sum(local_sum_dxh_xh, s_red);
    if (tid == 0) s_sum_dxh_xh = tot_dxh_xh;
    __syncthreads();

    const float sum_dxh    = s_sum_dxh;
    const float sum_dxh_xh = s_sum_dxh_xh;
    const float inv_M      = 1.0f / static_cast<float>(tile_size);
    const float M_f        = static_cast<float>(tile_size);

    // Pass 4: write dX
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = to_f32(gamma[channel]);
        const float dy = to_f32(dy_tile[i]);
        const float xh = (to_f32(x_tile[i]) - mean) * rstd;
        const float dxh = dy * gv;

        const float val = rstd * inv_M * (M_f * dxh - sum_dxh - xh * sum_dxh_xh);
        dx_tile[i] = from_f32<T>(val);
    }

    // Flush staging to global accumulators
    for (int c = tid; c < channels_per_group; c += blockDim.x) {
        atomicAdd(&dGamma_accum[chan_base + c], s_dG[c]);
        atomicAdd(&dBeta_accum[chan_base + c],  s_dB[c]);
    }
}

template <typename T>
__global__ void fold_fp32_into_T(const float* __restrict__ src, T* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_f32<T>(to_f32(dst[i]) + src[i]);
}

} // namespace

void group_norm_forward(const Tensor& X, const Tensor& gamma, const Tensor& beta,
                        int N, int C, int H, int W, int num_groups, float eps, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("group_norm_forward: X must be FP32, FP16, or BF16");
    }
    if (gamma.dtype != X.dtype || beta.dtype != X.dtype) {
        throw std::runtime_error("group_norm_forward: gamma/beta dtype must match X.dtype");
    }
    if (C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward: C must be divisible by num_groups");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(X.rows, X.cols, X.dtype);
    }
    const int spatial = H * W;
    if (N == 0 || C == 0 || spatial == 0) return;

    const int channels_per_group = C / num_groups;
    dim3 grid(num_groups, N, 1);
    hipStream_t stream = cur_stream();

    if (X.dtype == Dtype::FP32) {
        group_norm_forward_kernel<float><<<grid, GN_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(gamma.data),
            static_cast<const float*>(beta.data),
            static_cast<float*>(Y.data),
            C, spatial, channels_per_group, eps);
    } else if (X.dtype == Dtype::FP16) {
        group_norm_forward_kernel<__half><<<grid, GN_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(gamma.data),
            static_cast<const __half*>(beta.data),
            static_cast<__half*>(Y.data),
            C, spatial, channels_per_group, eps);
    } else {
        group_norm_forward_kernel<hip_bfloat16><<<grid, GN_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data),
            static_cast<const hip_bfloat16*>(gamma.data),
            static_cast<const hip_bfloat16*>(beta.data),
            static_cast<hip_bfloat16*>(Y.data),
            C, spatial, channels_per_group, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void group_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                         int N, int C, int H, int W, int num_groups, float eps,
                         Tensor& dX, Tensor& dGamma, Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("group_norm_backward: X must be FP32, FP16, or BF16");
    }
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype) {
        throw std::runtime_error("group_norm_backward: gamma and dY must match X.dtype");
    }
    if (C % num_groups != 0) {
        throw std::runtime_error("group_norm_backward: C must be divisible by num_groups");
    }
    if (dX.rows != X.rows || dX.cols != X.cols || dX.dtype != X.dtype) {
        dX.resize(X.rows, X.cols, X.dtype);
    }
    if (dGamma.size() != C || dGamma.dtype != X.dtype) {
        dGamma.resize(C, 1, X.dtype);
    }
    if (dBeta.size() != C || dBeta.dtype != X.dtype) {
        dBeta.resize(C, 1, X.dtype);
    }
    const int spatial = H * W;
    if (N == 0 || C == 0 || spatial == 0) return;

    hipStream_t stream = cur_stream();
    float* d_dG = static_cast<float*>(hip_alloc(static_cast<size_t>(C) * sizeof(float)));
    float* d_dB = static_cast<float*>(hip_alloc(static_cast<size_t>(C) * sizeof(float)));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(d_dG, 0, C * sizeof(float), stream));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(d_dB, 0, C * sizeof(float), stream));

    const int channels_per_group = C / num_groups;
    dim3 grid(num_groups, N, 1);
    const size_t chan_smem = static_cast<size_t>(channels_per_group) * 2 * sizeof(float);

    if (X.dtype == Dtype::FP32) {
        group_norm_backward_kernel<float><<<grid, GN_BLOCK, chan_smem, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(gamma.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            d_dG, d_dB,
            C, spatial, channels_per_group, eps);
    } else if (X.dtype == Dtype::FP16) {
        group_norm_backward_kernel<__half><<<grid, GN_BLOCK, chan_smem, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<const __half*>(gamma.data),
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            d_dG, d_dB,
            C, spatial, channels_per_group, eps);
    } else {
        group_norm_backward_kernel<hip_bfloat16><<<grid, GN_BLOCK, chan_smem, stream>>>(
            static_cast<const hip_bfloat16*>(X.data),
            static_cast<const hip_bfloat16*>(gamma.data),
            static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data),
            d_dG, d_dB,
            C, spatial, channels_per_group, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    const int block = 128;
    const int gridc = (C + block - 1) / block;
    if (X.dtype == Dtype::FP32) {
        fold_fp32_into_T<float><<<gridc, block, 0, stream>>>(d_dG, static_cast<float*>(dGamma.data), C);
        fold_fp32_into_T<float><<<gridc, block, 0, stream>>>(d_dB, static_cast<float*>(dBeta.data), C);
    } else if (X.dtype == Dtype::FP16) {
        fold_fp32_into_T<__half><<<gridc, block, 0, stream>>>(d_dG, static_cast<__half*>(dGamma.data), C);
        fold_fp32_into_T<__half><<<gridc, block, 0, stream>>>(d_dB, static_cast<__half*>(dBeta.data), C);
    } else {
        fold_fp32_into_T<hip_bfloat16><<<gridc, block, 0, stream>>>(d_dG, static_cast<hip_bfloat16*>(dGamma.data), C);
        fold_fp32_into_T<hip_bfloat16><<<gridc, block, 0, stream>>>(d_dB, static_cast<hip_bfloat16*>(dBeta.data), C);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(d_dG);
    hip_free(d_dB);
}

} // namespace brotensor::detail::hip
