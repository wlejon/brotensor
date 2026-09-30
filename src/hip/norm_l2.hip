// Native AMD HIP L2 Norm & L2 Normalize (nchw per-pixel) for RDNA 3.5 (gfx1151).
//
// Supports FP32, FP16 (__half), and BF16 (hip_bfloat16).

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

constexpr int L2_BLOCK = 128;
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

// ─── L2 Norm Forward & Backward (per-head, last-dim) ────────────────────────

template <typename T>
__global__ void l2_norm_forward_kernel(const T* __restrict__ X,
                                       T* __restrict__ Y,
                                       int L, int head_dim, int total_dim,
                                       float eps) {
    __shared__ float sdata[L2_BLOCK >> 5];
    const int row = blockIdx.y;
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= L) return;

    const size_t off = static_cast<size_t>(row) * total_dim + static_cast<size_t>(h) * head_dim;
    const T* x = X + off;
    T*       y = Y + off;

    float acc = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = to_f32(x[i]);
        acc += v * v;
    }
    const float sum = block_reduce_sum(acc, sdata);
    __shared__ float s_inv_norm;
    if (tid == 0) {
        s_inv_norm = rsqrtf(sum + eps);
    }
    __syncthreads();
    const float inv_norm = s_inv_norm;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        y[i] = from_f32<T>(to_f32(x[i]) * inv_norm);
    }
}

template <typename T>
__global__ void l2_norm_backward_kernel(const T* __restrict__ X,
                                        const T* __restrict__ dY,
                                        T* __restrict__ dX,
                                        int L, int head_dim, int total_dim,
                                        float eps) {
    __shared__ float sdata[L2_BLOCK >> 5];
    const int row = blockIdx.y;
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= L) return;

    const size_t off = static_cast<size_t>(row) * total_dim + static_cast<size_t>(h) * head_dim;
    const T* x  = X  + off;
    const T* dy = dY + off;
    T*       dx = dX + off;

    float acc_sq = 0.0f;
    float acc_dot = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float xv = to_f32(x[i]);
        const float yv = to_f32(dy[i]);
        acc_sq  += xv * xv;
        acc_dot += xv * yv;
    }
    const float sum_sq  = block_reduce_sum(acc_sq, sdata);
    __syncthreads();
    const float sum_dot = block_reduce_sum(acc_dot, sdata);

    __shared__ float s_inv_n;
    __shared__ float s_inv_n3;
    if (tid == 0) {
        const float n2 = sum_sq + eps;
        const float inv_n = rsqrtf(n2);
        s_inv_n  = inv_n;
        s_inv_n3 = inv_n / n2;
    }
    __syncthreads();
    const float inv_n  = s_inv_n;
    const float inv_n3 = s_inv_n3;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float xv = to_f32(x[i]);
        const float yv = to_f32(dy[i]);
        dx[i] = from_f32<T>(inv_n * yv - xv * sum_dot * inv_n3);
    }
}

// ─── L2 Normalize NCHW (per-pixel over channel axis) ────────────────────────

template <typename T>
__global__ void l2_normalize_nchw_kernel(const T* __restrict__ X,
                                         T* __restrict__ Y,
                                         int N, int C, int HW,
                                         float eps) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total_pixels = static_cast<long long>(N) * HW;
    if (idx >= total_pixels) return;

    const int n = static_cast<int>(idx / HW);
    const int hw = static_cast<int>(idx % HW);
    const size_t base_offset = static_cast<size_t>(n) * C * HW + hw;

    double sumsq = 0.0;
    for (int c = 0; c < C; ++c) {
        const double v = static_cast<double>(to_f32(X[base_offset + static_cast<size_t>(c) * HW]));
        sumsq += v * v;
    }
    const double norm = std::sqrt(sumsq);
    const double denom = (norm > static_cast<double>(eps)) ? norm : static_cast<double>(eps);
    const float scale = static_cast<float>(1.0 / denom);

    for (int c = 0; c < C; ++c) {
        const size_t off = base_offset + static_cast<size_t>(c) * HW;
        Y[off] = from_f32<T>(to_f32(X[off]) * scale);
    }
}

} // namespace

void l2_norm_forward(const Tensor& X, int head_dim, int num_heads, float eps, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("l2_norm_forward: X must be FP32, FP16, or BF16");
    }
    const int total_dim = head_dim * num_heads;
    if (X.cols != total_dim) {
        throw std::runtime_error("l2_norm_forward: X.cols must equal head_dim * num_heads");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != total_dim || Y.dtype != X.dtype) {
        Y.resize(L, total_dim, X.dtype);
    }
    if (L == 0 || total_dim == 0) return;

    dim3 grid(num_heads, L);
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        l2_norm_forward_kernel<float><<<grid, L2_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), L, head_dim, total_dim, eps);
    } else if (X.dtype == Dtype::FP16) {
        l2_norm_forward_kernel<__half><<<grid, L2_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), L, head_dim, total_dim, eps);
    } else {
        l2_norm_forward_kernel<hip_bfloat16><<<grid, L2_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), L, head_dim, total_dim, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void l2_norm_backward(const Tensor& X, int head_dim, int num_heads, float eps,
                      const Tensor& dY, Tensor& dX) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("l2_norm_backward: X must be FP32, FP16, or BF16");
    }
    if (dY.dtype != X.dtype) {
        throw std::runtime_error("l2_norm_backward: dY must match X.dtype");
    }
    const int total_dim = head_dim * num_heads;
    const int L = X.rows;
    if (dX.rows != L || dX.cols != total_dim || dX.dtype != X.dtype) {
        dX.resize(L, total_dim, X.dtype);
    }
    if (L == 0 || total_dim == 0) return;

    dim3 grid(num_heads, L);
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP32) {
        l2_norm_backward_kernel<float><<<grid, L2_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data), L, head_dim, total_dim, eps);
    } else if (X.dtype == Dtype::FP16) {
        l2_norm_backward_kernel<__half><<<grid, L2_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data), L, head_dim, total_dim, eps);
    } else {
        l2_norm_backward_kernel<hip_bfloat16><<<grid, L2_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(dY.data),
            static_cast<hip_bfloat16*>(dX.data), L, head_dim, total_dim, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void l2_normalize_nchw_forward(const Tensor& X, int N, int C, int H, int W, float eps, Tensor& Y) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("l2_normalize_nchw_forward: X must be FP32, FP16, or BF16");
    }
    const long long total = static_cast<long long>(N) * C * H * W;
    if (Y.rows != N || Y.cols != C * H * W || Y.dtype != X.dtype) {
        Y.resize(N, C * H * W, X.dtype);
    }
    if (total == 0) return;

    const int HW = H * W;
    const long long pixels = static_cast<long long>(N) * HW;
    const int block = 256;
    const int grid = static_cast<int>((pixels + block - 1) / block);
    hipStream_t stream = cur_stream();

    if (X.dtype == Dtype::FP32) {
        l2_normalize_nchw_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data), N, C, HW, eps);
    } else if (X.dtype == Dtype::FP16) {
        l2_normalize_nchw_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), N, C, HW, eps);
    } else {
        l2_normalize_nchw_kernel<hip_bfloat16><<<grid, block, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), N, C, HW, eps);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
