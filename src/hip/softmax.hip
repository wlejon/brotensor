// Native AMD HIP row-wise and vector softmax for RDNA 3.5 (gfx1151, Wave32).
//
// Optimized for Wave32 wavefronts:
// - Warp-per-row: One Wave32 (32 threads) computes row max, exp-sum, and
//   normalization using 0xffffffff cross-lane shuffles (__shfl_xor / __shfl_down).
//   Zero shared memory, zero barriers, all in registers.
// - Block-per-row: For large rows (> 1024), 256 threads (8 Wave32s) perform
//   two-level tree reductions.
// - Supports FP32, FP16 (__half), and BF16 (hip_bfloat16).

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/hip_check.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int kWave32 = 32;
constexpr int kBlockThreads = 256;
constexpr unsigned kWaveMask = 0xffffffffu;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

// ─── Wave32 Shuffles ──────────────────────────────────────────────────────────

__device__ __forceinline__ float wave32_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, kWave32));
    }
    return val;
}

__device__ __forceinline__ float wave32_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, kWave32);
    }
    return val;
}

// ─── Type Conversion Traits ──────────────────────────────────────────────────

template <typename T>
struct SoftmaxTraits;

template <>
struct SoftmaxTraits<float> {
    __device__ static __forceinline__ float to_f32(float v) { return v; }
    __device__ static __forceinline__ float from_f32(float v) { return v; }
};

template <>
struct SoftmaxTraits<__half> {
    __device__ static __forceinline__ float to_f32(__half v) { return __half2float(v); }
    __device__ static __forceinline__ __half from_f32(float v) { return __float2half(v); }
};

template <>
struct SoftmaxTraits<hip_bfloat16> {
    __device__ static __forceinline__ float to_f32(hip_bfloat16 v) { return float(v); }
    __device__ static __forceinline__ hip_bfloat16 from_f32(float v) { return hip_bfloat16(v); }
};

// ─── Row Softmax: Warp-per-Row (cols <= 1024) ────────────────────────────────

template <typename T>
__global__ void softmax_rows_warp_kernel(const T* __restrict__ X,
                                         T* __restrict__ Y,
                                         int rows, int cols) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (row >= rows) return;

    const size_t row_off = static_cast<size_t>(row) * cols;
    const T* __restrict__ in_row = X + row_off;
    T* __restrict__ out_row = Y + row_off;

    // Phase 1: Thread-local max & Wave32 all-reduce
    float local_max = -1e30f;
    for (int c = lane; c < cols; c += kWave32) {
        const float v = SoftmaxTraits<T>::to_f32(in_row[c]);
        local_max = fmaxf(local_max, v);
    }
    const float row_max = wave32_reduce_max(local_max);

    // Phase 2: Exponentiate, sum & Wave32 all-reduce
    float local_sum = 0.0f;
    for (int c = lane; c < cols; c += kWave32) {
        const float v = SoftmaxTraits<T>::to_f32(in_row[c]);
        local_sum += __expf(v - row_max);
    }
    const float row_sum = wave32_reduce_sum(local_sum);
    const float inv_sum = row_sum > 0.0f ? (1.0f / row_sum) : 0.0f;

    // Phase 3: Normalize and store
    for (int c = lane; c < cols; c += kWave32) {
        const float v = SoftmaxTraits<T>::to_f32(in_row[c]);
        const float p = __expf(v - row_max) * inv_sum;
        out_row[c] = SoftmaxTraits<T>::from_f32(p);
    }
}

// ─── Row Softmax: Block-per-Row (cols > 1024) ────────────────────────────────

template <typename T>
__global__ void softmax_rows_block_kernel(const T* __restrict__ X,
                                          T* __restrict__ Y,
                                          int rows, int cols) {
    __shared__ float s_wave[32]; // up to 32 waves (1024 threads)
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const size_t row_off = static_cast<size_t>(row) * cols;
    const T* __restrict__ in_row = X + row_off;
    T* __restrict__ out_row = Y + row_off;

    // Phase 1: Local max -> Wave32 reduce -> Block reduce
    float local_max = -1e30f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float v = SoftmaxTraits<T>::to_f32(in_row[c]);
        local_max = fmaxf(local_max, v);
    }
    local_max = wave32_reduce_max(local_max);
    if (lane == 0) s_wave[warp_id] = local_max;
    __syncthreads();

    float block_max = -1e30f;
    if (tid < num_warps) block_max = s_wave[tid];
    block_max = wave32_reduce_max(block_max);
    if (warp_id == 0) s_wave[lane] = block_max;
    __syncthreads();
    const float row_max = s_wave[0];
    __syncthreads();

    // Phase 2: Exponentiate, local sum -> Wave32 reduce -> Block reduce
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float v = SoftmaxTraits<T>::to_f32(in_row[c]);
        local_sum += __expf(v - row_max);
    }
    local_sum = wave32_reduce_sum(local_sum);
    if (lane == 0) s_wave[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < num_warps) block_sum = s_wave[tid];
    block_sum = wave32_reduce_sum(block_sum);
    if (warp_id == 0) s_wave[lane] = block_sum;
    __syncthreads();
    const float row_sum = s_wave[0];
    const float inv_sum = row_sum > 0.0f ? (1.0f / row_sum) : 0.0f;

    // Phase 3: Normalize and store
    for (int c = tid; c < cols; c += blockDim.x) {
        const float v = SoftmaxTraits<T>::to_f32(in_row[c]);
        const float p = __expf(v - row_max) * inv_sum;
        out_row[c] = SoftmaxTraits<T>::from_f32(p);
    }
}

// ─── Single-Vector Softmax (softmax_forward with optional mask) ───────────────

template <typename T>
__global__ void softmax_vector_forward_kernel(const T* __restrict__ logits,
                                              T* __restrict__ probs,
                                              const float* __restrict__ mask,
                                              int n) {
    __shared__ float s_wave[32];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // Phase 1: Max over unmasked entries
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) continue;
        const float v = SoftmaxTraits<T>::to_f32(logits[i]);
        local_max = fmaxf(local_max, v);
    }
    local_max = wave32_reduce_max(local_max);
    if (lane == 0) s_wave[warp_id] = local_max;
    __syncthreads();

    float block_max = -1e30f;
    if (tid < num_warps) block_max = s_wave[tid];
    block_max = wave32_reduce_max(block_max);
    if (warp_id == 0) s_wave[lane] = block_max;
    __syncthreads();
    const float m = s_wave[0];
    __syncthreads();

    // Phase 2: Exponentiate and accumulate sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) {
            probs[i] = SoftmaxTraits<T>::from_f32(0.0f);
            continue;
        }
        const float e = __expf(SoftmaxTraits<T>::to_f32(logits[i]) - m);
        local_sum += e;
    }
    local_sum = wave32_reduce_sum(local_sum);
    if (lane == 0) s_wave[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < num_warps) block_sum = s_wave[tid];
    block_sum = wave32_reduce_sum(block_sum);
    if (warp_id == 0) s_wave[lane] = block_sum;
    __syncthreads();
    const float sum = s_wave[0];
    const float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;

    // Phase 3: Normalize
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) continue;
        const float e = __expf(SoftmaxTraits<T>::to_f32(logits[i]) - m);
        probs[i] = SoftmaxTraits<T>::from_f32(e * inv);
    }
}

// ─── Single-Vector Softmax Backward ──────────────────────────────────────────

template <typename T>
__global__ void softmax_vector_backward_kernel(const T* __restrict__ probs,
                                               const T* __restrict__ dProbs,
                                               T* __restrict__ dLogits,
                                               int n) {
    __shared__ float s_wave[32];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    // dot = sum_j (dProbs[j] * probs[j])
    float local_dot = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float p = SoftmaxTraits<T>::to_f32(probs[i]);
        const float dp = SoftmaxTraits<T>::to_f32(dProbs[i]);
        local_dot += p * dp;
    }
    local_dot = wave32_reduce_sum(local_dot);
    if (lane == 0) s_wave[warp_id] = local_dot;
    __syncthreads();

    float block_dot = 0.0f;
    if (tid < num_warps) block_dot = s_wave[tid];
    block_dot = wave32_reduce_sum(block_dot);
    if (warp_id == 0) s_wave[lane] = block_dot;
    __syncthreads();
    const float dot = s_wave[0];

    // dLogits[i] = probs[i] * (dProbs[i] - dot)
    for (int i = tid; i < n; i += blockDim.x) {
        const float p = SoftmaxTraits<T>::to_f32(probs[i]);
        const float dp = SoftmaxTraits<T>::to_f32(dProbs[i]);
        dLogits[i] = SoftmaxTraits<T>::from_f32(p * (dp - dot));
    }
}

} // namespace

// ─── Public API Implementations ──────────────────────────────────────────────

void hip_softmax_forward(const ::brotensor::Tensor& logits,
                         ::brotensor::Tensor& probs,
                         const float* d_mask) {
    using ::brotensor::Dtype;
    if (logits.dtype != Dtype::FP16 && logits.dtype != Dtype::BF16 &&
        logits.dtype != Dtype::FP32) {
        throw std::runtime_error("hip_softmax_forward: logits must be FP16, BF16, or FP32");
    }
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols ||
        probs.dtype != logits.dtype) {
        probs.resize(logits.rows, logits.cols, logits.dtype);
    }
    if (n == 0) return;

    hipStream_t stream = cur_stream();
    constexpr int threads = kBlockThreads;
    if (logits.dtype == Dtype::FP16) {
        softmax_vector_forward_kernel<__half><<<1, threads, 0, stream>>>(
            static_cast<const __half*>(logits.data),
            static_cast<__half*>(probs.data),
            d_mask, n);
    } else if (logits.dtype == Dtype::BF16) {
        softmax_vector_forward_kernel<hip_bfloat16><<<1, threads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(logits.data),
            static_cast<hip_bfloat16*>(probs.data),
            d_mask, n);
    } else {
        softmax_vector_forward_kernel<float><<<1, threads, 0, stream>>>(
            static_cast<const float*>(logits.data),
            static_cast<float*>(probs.data),
            d_mask, n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_softmax_rows_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                              int rows, int cols) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_softmax_rows_forward: X must be FP32, FP16, or BF16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(X.rows, X.cols, X.dtype);
    }
    if (rows <= 0 || cols <= 0) return;

    hipStream_t stream = cur_stream();
    if (cols <= 1024) {
        // Warp-per-row path: 8 rows per block of 256 threads (Wave32)
        constexpr int warps_per_block = kBlockThreads / kWave32;
        const int blocks = (rows + warps_per_block - 1) / warps_per_block;
        if (X.dtype == Dtype::FP16) {
            softmax_rows_warp_kernel<__half><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), rows, cols);
        } else if (X.dtype == Dtype::BF16) {
            softmax_rows_warp_kernel<hip_bfloat16><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), rows, cols);
        } else {
            softmax_rows_warp_kernel<float><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const float*>(X.data), static_cast<float*>(Y.data), rows, cols);
        }
    } else {
        // Block-per-row path for wide rows (vocab logits, large contexts)
        const int blocks = rows;
        if (X.dtype == Dtype::FP16) {
            softmax_rows_block_kernel<__half><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const __half*>(X.data), static_cast<__half*>(Y.data), rows, cols);
        } else if (X.dtype == Dtype::BF16) {
            softmax_rows_block_kernel<hip_bfloat16><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data), rows, cols);
        } else {
            softmax_rows_block_kernel<float><<<blocks, kBlockThreads, 0, stream>>>(
                static_cast<const float*>(X.data), static_cast<float*>(Y.data), rows, cols);
        }
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_softmax_backward(const ::brotensor::Tensor& probs,
                          const ::brotensor::Tensor& dProbs,
                          ::brotensor::Tensor& dLogits) {
    using ::brotensor::Dtype;
    if (probs.dtype != Dtype::FP16 && probs.dtype != Dtype::BF16 &&
        probs.dtype != Dtype::FP32) {
        throw std::runtime_error("hip_softmax_backward: probs must be FP16, BF16, or FP32");
    }
    if (dProbs.dtype != probs.dtype) {
        throw std::runtime_error("hip_softmax_backward: dProbs.dtype must match probs.dtype");
    }
    const int n = probs.size();
    if (dLogits.rows != probs.rows || dLogits.cols != probs.cols ||
        dLogits.dtype != probs.dtype) {
        dLogits.resize(probs.rows, probs.cols, probs.dtype);
    }
    if (n == 0) return;

    hipStream_t stream = cur_stream();
    constexpr int threads = kBlockThreads;
    if (probs.dtype == Dtype::FP16) {
        softmax_vector_backward_kernel<__half><<<1, threads, 0, stream>>>(
            static_cast<const __half*>(probs.data),
            static_cast<const __half*>(dProbs.data),
            static_cast<__half*>(dLogits.data), n);
    } else if (probs.dtype == Dtype::BF16) {
        softmax_vector_backward_kernel<hip_bfloat16><<<1, threads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(probs.data),
            static_cast<const hip_bfloat16*>(dProbs.data),
            static_cast<hip_bfloat16*>(dLogits.data), n);
    } else {
        softmax_vector_backward_kernel<float><<<1, threads, 0, stream>>>(
            static_cast<const float*>(probs.data),
            static_cast<const float*>(dProbs.data),
            static_cast<float*>(dLogits.data), n);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_softmax(::brotensor::detail::OpsVTable& v) {
    v.softmax_forward      = &hip_softmax_forward;
    v.softmax_backward     = &hip_softmax_backward;
    v.softmax_rows_forward = &hip_softmax_rows_forward;
}

} // namespace brotensor::detail::hip
