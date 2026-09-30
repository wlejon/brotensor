// Native AMD HIP RMSNorm forward & backward for RDNA 3.5 (gfx1151, Wave32).
//
// Features:
// - Wave32 wavefront execution (kWave32 = 32).
// - 128-bit vectorized loads (float4 for FP32 = 4 elements; 16-byte for FP16/BF16 = 8 elements).
// - Fast cross-lane register reductions via __shfl_xor and __shfl_down.
// - Supports FP32, FP16 (__half), and BF16 (hip_bfloat16).
// - Gamma can match X.dtype or be FP32.

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

constexpr int kWave32 = 32;
constexpr int kBlockThreads = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

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

// ─── Forward Kernels ─────────────────────────────────────────────────────────

// Warp-per-row forward kernel (D <= 1024 or small rows): 1 Wave32 per row
template <typename T, typename GT>
__global__ void rms_forward_warp_kernel(const T* __restrict__ X,
                                        const GT* __restrict__ gamma,
                                        T* __restrict__ Y,
                                        int B, int D, float eps) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (b >= B) return;

    const size_t row_off = static_cast<size_t>(b) * D;
    const T* __restrict__ xrow = X + row_off;
    T* __restrict__ yrow = Y + row_off;

    float local_sum = 0.0f;

    // Vectorized 128-bit loads if aligned
    if constexpr (sizeof(T) == 4) {
        // FP32: 4 floats = 16 bytes
        const int vec_end = (D / 4) * 4;
        const float4* x4 = reinterpret_cast<const float4*>(xrow);
        for (int j = lane * 4; j < vec_end; j += kWave32 * 4) {
            float4 v4 = x4[j / 4];
            local_sum += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
        }
        for (int j = vec_end + lane; j < D; j += kWave32) {
            float v = to_f32(xrow[j]);
            local_sum += v * v;
        }
    } else {
        // FP16/BF16: 8 elements = 16 bytes
        const int vec_end = (D / 8) * 8;
        const int4* x4 = reinterpret_cast<const int4*>(xrow);
        for (int j = lane * 8; j < vec_end; j += kWave32 * 8) {
            int4 raw = x4[j / 8];
            const T* sub = reinterpret_cast<const T*>(&raw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = to_f32(sub[k]);
                local_sum += v * v;
            }
        }
        for (int j = vec_end + lane; j < D; j += kWave32) {
            float v = to_f32(xrow[j]);
            local_sum += v * v;
        }
    }

    const float sum = wave32_reduce_sum(local_sum);
    const float rrms = rsqrtf(sum / static_cast<float>(D) + eps);

    for (int j = lane; j < D; j += kWave32) {
        const float xv = to_f32(xrow[j]);
        const float gv = to_f32(gamma[j]);
        yrow[j] = from_f32<T>(xv * gv * rrms);
    }
}

// Block-per-row forward kernel (for large rows D > 1024): 256 threads per row
template <typename T, typename GT>
__global__ void rms_forward_block_kernel(const T* __restrict__ X,
                                         const GT* __restrict__ gamma,
                                         T* __restrict__ Y,
                                         int B, int D, float eps) {
    __shared__ float sdata[kBlockThreads >> 5];
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;

    const size_t row_off = static_cast<size_t>(b) * D;
    const T* __restrict__ xrow = X + row_off;
    T* __restrict__ yrow = Y + row_off;

    float local_sum = 0.0f;

    if constexpr (sizeof(T) == 4) {
        const int vec_end = (D / 4) * 4;
        const float4* x4 = reinterpret_cast<const float4*>(xrow);
        for (int j = tid * 4; j < vec_end; j += blockDim.x * 4) {
            float4 v4 = x4[j / 4];
            local_sum += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
        }
        for (int j = vec_end + tid; j < D; j += blockDim.x) {
            float v = to_f32(xrow[j]);
            local_sum += v * v;
        }
    } else {
        const int vec_end = (D / 8) * 8;
        const int4* x4 = reinterpret_cast<const int4*>(xrow);
        for (int j = tid * 8; j < vec_end; j += blockDim.x * 8) {
            int4 raw = x4[j / 8];
            const T* sub = reinterpret_cast<const T*>(&raw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = to_f32(sub[k]);
                local_sum += v * v;
            }
        }
        for (int j = vec_end + tid; j < D; j += blockDim.x) {
            float v = to_f32(xrow[j]);
            local_sum += v * v;
        }
    }

    const float sum = block_reduce_sum(local_sum, sdata);
    __shared__ float s_rrms;
    if (tid == 0) {
        s_rrms = rsqrtf(sum / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float rrms = s_rrms;

    for (int j = tid; j < D; j += blockDim.x) {
        const float xv = to_f32(xrow[j]);
        const float gv = to_f32(gamma[j]);
        yrow[j] = from_f32<T>(xv * gv * rrms);
    }
}

// ─── Backward Kernels ────────────────────────────────────────────────────────

template <typename T, typename GT>
__global__ void rms_backward_kernel(const T* __restrict__ X,
                                    const GT* __restrict__ gamma,
                                    const T* __restrict__ dY,
                                    T* __restrict__ dX,
                                    float* __restrict__ dGamma_scratch,
                                    int B, int D, float eps) {
    __shared__ float sdata[kBlockThreads >> 5];
    __shared__ float s_rrms;
    __shared__ float s_coeff;

    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;

    const size_t row_off = static_cast<size_t>(b) * D;
    const T* __restrict__ xrow  = X  + row_off;
    const T* __restrict__ dyrow = dY + row_off;
    T* __restrict__       dxrow = dX + row_off;

    // Pass 1: sum of squares
    float local_xx = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        const float v = to_f32(xrow[j]);
        local_xx += v * v;
    }
    const float sum_xx = block_reduce_sum(local_xx, sdata);
    if (tid == 0) {
        s_rrms = rsqrtf(sum_xx / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float rrms = s_rrms;

    // Pass 2: sum of x * dy * gamma
    float local_xdy = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        local_xdy += to_f32(xrow[j]) * to_f32(dyrow[j]) * to_f32(gamma[j]);
    }
    const float sum_xdy = block_reduce_sum(local_xdy, sdata);
    if (tid == 0) {
        const float inv_D = 1.0f / static_cast<float>(D);
        s_coeff = inv_D * rrms * rrms * sum_xdy;
    }
    __syncthreads();
    const float coeff = s_coeff;

    // Pass 3: write dX and accumulate into dGamma scratch
    for (int j = tid; j < D; j += blockDim.x) {
        const float g  = to_f32(gamma[j]);
        const float dy = to_f32(dyrow[j]);
        const float x  = to_f32(xrow[j]);
        dxrow[j] = from_f32<T>(rrms * (g * dy - x * coeff));
        atomicAdd(&dGamma_scratch[j], dy * x * rrms);
    }
}

template <typename T>
__global__ void fold_fp32_into_T(const float* __restrict__ src, T* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = from_f32<T>(to_f32(dst[i]) + src[i]);
}

} // namespace

void rms_norm_forward(const Tensor& X, const Tensor& gamma, float eps, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("rms_norm_forward: X must be FP32, FP16, or BF16");
    }
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) {
        throw std::runtime_error("rms_norm_forward: gamma.dtype must match X.dtype or be FP32");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (gamma.size() != D) {
        throw std::runtime_error("rms_norm_forward: gamma must have D elements");
    }
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    if (B == 0 || D == 0) return;

    hipStream_t stream = cur_stream();

    // Use warp-per-row for D <= 1024, block-per-row for D > 1024
    if (D <= 1024) {
        constexpr int warps_per_block = kBlockThreads / kWave32;
        const int blocks = (B + warps_per_block - 1) / warps_per_block;
        if (X.dtype == Dtype::FP32) {
            rms_forward_warp_kernel<float, float><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const float*>(X.data),
                static_cast<const float*>(gamma.data),
                static_cast<float*>(Y.data),
                B, D, eps);
        } else if (X.dtype == Dtype::FP16) {
            if (gamma.dtype == Dtype::FP32) {
                rms_forward_warp_kernel<__half, float><<<blocks, kBlockThreads, 0, stream>>>(
                    static_cast<const __half*>(X.data),
                    static_cast<const float*>(gamma.data),
                    static_cast<__half*>(Y.data),
                    B, D, eps);
            } else {
                rms_forward_warp_kernel<__half, __half><<<blocks, kBlockThreads, 0, stream>>>(
                    static_cast<const __half*>(X.data),
                    static_cast<const __half*>(gamma.data),
                    static_cast<__half*>(Y.data),
                    B, D, eps);
            }
        } else { // BF16
            if (gamma.dtype == Dtype::FP32) {
                rms_forward_warp_kernel<hip_bfloat16, float><<<blocks, kBlockThreads, 0, stream>>>(
                    static_cast<const hip_bfloat16*>(X.data),
                    static_cast<const float*>(gamma.data),
                    static_cast<hip_bfloat16*>(Y.data),
                    B, D, eps);
            } else {
                rms_forward_warp_kernel<hip_bfloat16, hip_bfloat16><<<blocks, kBlockThreads, 0, stream>>>(
                    static_cast<const hip_bfloat16*>(X.data),
                    static_cast<const hip_bfloat16*>(gamma.data),
                    static_cast<hip_bfloat16*>(Y.data),
                    B, D, eps);
            }
        }
    } else {
        if (X.dtype == Dtype::FP32) {
            rms_forward_block_kernel<float, float><<<B, kBlockThreads, 0, stream>>>(
                static_cast<const float*>(X.data),
                static_cast<const float*>(gamma.data),
                static_cast<float*>(Y.data),
                B, D, eps);
        } else if (X.dtype == Dtype::FP16) {
            if (gamma.dtype == Dtype::FP32) {
                rms_forward_block_kernel<__half, float><<<B, kBlockThreads, 0, stream>>>(
                    static_cast<const __half*>(X.data),
                    static_cast<const float*>(gamma.data),
                    static_cast<__half*>(Y.data),
                    B, D, eps);
            } else {
                rms_forward_block_kernel<__half, __half><<<B, kBlockThreads, 0, stream>>>(
                    static_cast<const __half*>(X.data),
                    static_cast<const __half*>(gamma.data),
                    static_cast<__half*>(Y.data),
                    B, D, eps);
            }
        } else { // BF16
            if (gamma.dtype == Dtype::FP32) {
                rms_forward_block_kernel<hip_bfloat16, float><<<B, kBlockThreads, 0, stream>>>(
                    static_cast<const hip_bfloat16*>(X.data),
                    static_cast<const float*>(gamma.data),
                    static_cast<hip_bfloat16*>(Y.data),
                    B, D, eps);
            } else {
                rms_forward_block_kernel<hip_bfloat16, hip_bfloat16><<<B, kBlockThreads, 0, stream>>>(
                    static_cast<const hip_bfloat16*>(X.data),
                    static_cast<const hip_bfloat16*>(gamma.data),
                    static_cast<hip_bfloat16*>(Y.data),
                    B, D, eps);
            }
        }
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void rms_norm_backward(const Tensor& X, const Tensor& gamma, const Tensor& dY,
                       float eps, Tensor& dX, Tensor& dGamma) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("rms_norm_backward: X must be FP32, FP16, or BF16");
    }
    if (dY.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_backward: dY.dtype must match X.dtype");
    }
    if (gamma.dtype != X.dtype && gamma.dtype != Dtype::FP32) {
        throw std::runtime_error("rms_norm_backward: gamma.dtype must match X.dtype or be FP32");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (dX.rows != B || dX.cols != D || dX.dtype != X.dtype) {
        dX.resize(B, D, X.dtype);
    }
    if (dGamma.size() != D || dGamma.dtype != gamma.dtype) {
        dGamma.resize(gamma.rows, gamma.cols, gamma.dtype);
    }
    if (B == 0 || D == 0) return;

    hipStream_t stream = cur_stream();
    float* dGamma_scratch = static_cast<float*>(hip_alloc(D * sizeof(float)));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dGamma_scratch, 0, D * sizeof(float), stream));

    if (X.dtype == Dtype::FP32) {
        rms_backward_kernel<float, float><<<B, kBlockThreads, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<const float*>(gamma.data),
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            dGamma_scratch, B, D, eps);
    } else if (X.dtype == Dtype::FP16) {
        if (gamma.dtype == Dtype::FP32) {
            rms_backward_kernel<__half, float><<<B, kBlockThreads, 0, stream>>>(
                static_cast<const __half*>(X.data),
                static_cast<const float*>(gamma.data),
                static_cast<const __half*>(dY.data),
                static_cast<__half*>(dX.data),
                dGamma_scratch, B, D, eps);
        } else {
            rms_backward_kernel<__half, __half><<<B, kBlockThreads, 0, stream>>>(
                static_cast<const __half*>(X.data),
                static_cast<const __half*>(gamma.data),
                static_cast<const __half*>(dY.data),
                static_cast<__half*>(dX.data),
                dGamma_scratch, B, D, eps);
        }
    } else { // BF16
        if (gamma.dtype == Dtype::FP32) {
            rms_backward_kernel<hip_bfloat16, float><<<B, kBlockThreads, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data),
                static_cast<const float*>(gamma.data),
                static_cast<const hip_bfloat16*>(dY.data),
                static_cast<hip_bfloat16*>(dX.data),
                dGamma_scratch, B, D, eps);
        } else {
            rms_backward_kernel<hip_bfloat16, hip_bfloat16><<<B, kBlockThreads, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data),
                static_cast<const hip_bfloat16*>(gamma.data),
                static_cast<const hip_bfloat16*>(dY.data),
                static_cast<hip_bfloat16*>(dX.data),
                dGamma_scratch, B, D, eps);
        }
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    // Fold dGamma_scratch into caller's dGamma
    const int fold_blocks = (D + kBlockThreads - 1) / kBlockThreads;
    if (dGamma.dtype == Dtype::FP32) {
        fold_fp32_into_T<float><<<fold_blocks, kBlockThreads, 0, stream>>>(
            dGamma_scratch, static_cast<float*>(dGamma.data), D);
    } else if (dGamma.dtype == Dtype::FP16) {
        fold_fp32_into_T<__half><<<fold_blocks, kBlockThreads, 0, stream>>>(
            dGamma_scratch, static_cast<__half*>(dGamma.data), D);
    } else {
        fold_fp32_into_T<hip_bfloat16><<<fold_blocks, kBlockThreads, 0, stream>>>(
            dGamma_scratch, static_cast<hip_bfloat16*>(dGamma.data), D);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(dGamma_scratch);
}

} // namespace brotensor::detail::hip
