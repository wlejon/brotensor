// ─── CUDA L2-norm (per-head, last-dim) ─────────────────────────────────────
//
// Mirrors src/cpu/l2_norm.cpp. Layout: (L, num_heads * head_dim), row-major;
// head h occupies columns [h*head_dim, (h+1)*head_dim). Used by the
// Qwen3-Next text path to L2-normalise q and k per head before the gated
// delta-rule recurrence.
//
// FP32-only. brolm's Qwen3-Next text path runs FP32 here per the public
// contract; if we ever need FP16/BF16 paths we'd extend along the same lines
// as rms_norm.cu.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int L2_BLOCK = 128;

// In-block reduction over `blockDim.x` threads using shared memory. Returns
// the sum from thread 0; other threads see the final value in sdata[0].
__device__ inline float block_sum(float v, float* sdata) {
    const int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

// One block per (row, head). blockDim.x = L2_BLOCK threads cooperate over
// head_dim. Shared memory: L2_BLOCK floats (single reduction scratch).
__global__ void l2_norm_forward_kernel(const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       int L, int num_heads, int head_dim,
                                       float eps) {
    extern __shared__ float sdata[];
    const int idx = blockIdx.x;
    const int r   = idx / num_heads;
    const int h   = idx - r * num_heads;
    if (r >= L) return;
    const int tid = threadIdx.x;
    const int D   = num_heads * head_dim;
    const int off = r * D + h * head_dim;
    const float* xrow = X + off;
    float*       yrow = Y + off;

    float local = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = xrow[d];
        local += v * v;
    }
    const float sumsq = block_sum(local, sdata);
    const float inv   = rsqrtf(sumsq + eps);

    for (int d = tid; d < head_dim; d += blockDim.x) {
        yrow[d] = xrow[d] * inv;
    }
}

// One block per (row, head). Shared memory: 2 * L2_BLOCK floats — one stripe
// for sumsq, one for dot(x, dY).
//
// dX_d = n * (dY_d - x_d * n^2 * dot(x, dY)), with n = 1 / sqrt(sumsq + eps).
__global__ void l2_norm_backward_kernel(const float* __restrict__ X,
                                        const float* __restrict__ dY,
                                        float* __restrict__ dX,
                                        int L, int num_heads, int head_dim,
                                        float eps) {
    extern __shared__ float sdata[];
    float* s_sumsq = sdata;
    float* s_dot   = sdata + blockDim.x;

    const int idx = blockIdx.x;
    const int r   = idx / num_heads;
    const int h   = idx - r * num_heads;
    if (r >= L) return;
    const int tid = threadIdx.x;
    const int D   = num_heads * head_dim;
    const int off = r * D + h * head_dim;
    const float* xrow = X  + off;
    const float* grow = dY + off;
    float*       drow = dX + off;

    float l_sumsq = 0.0f;
    float l_dot   = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v  = xrow[d];
        const float gv = grow[d];
        l_sumsq += v * v;
        l_dot   += v * gv;
    }
    s_sumsq[tid] = l_sumsq;
    s_dot[tid]   = l_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sumsq[tid] += s_sumsq[tid + s];
            s_dot[tid]   += s_dot[tid + s];
        }
        __syncthreads();
    }
    const float sumsq = s_sumsq[0];
    const float dot   = s_dot[0];
    const float n2    = 1.0f / (sumsq + eps);
    const float n     = sqrtf(n2);
    const float c     = dot * n2;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        drow[d] = n * (grow[d] - xrow[d] * c);
    }
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CUDA l2_norm is FP32-only)");
    }
}

inline void check_shape(const ::brotensor::Tensor& t,
                        int head_dim, int num_heads,
                        const char* op, const char* name) {
    if (head_dim <= 0) {
        throw std::runtime_error(std::string(op) + ": head_dim must be positive");
    }
    if (num_heads <= 0) {
        throw std::runtime_error(std::string(op) + ": num_heads must be positive");
    }
    if (t.cols != num_heads * head_dim) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 ".cols != num_heads * head_dim");
    }
}

} // namespace

void l2_norm_forward(const ::brotensor::Tensor& X,
                     int head_dim, int num_heads, float eps,
                     ::brotensor::Tensor& Y) {
    check_fp32(X, "l2_norm_forward", "X");
    check_shape(X, head_dim, num_heads, "l2_norm_forward", "X");
    const int L = X.rows;
    const int D = X.cols;
    if (Y.rows != L || Y.cols != D || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(L, D, ::brotensor::Dtype::FP32);
    }
    if (L == 0 || D == 0) return;

    const int blocks = L * num_heads;
    const int block  = L2_BLOCK;
    const size_t shmem = block * sizeof(float);
    l2_norm_forward_kernel<<<blocks, block, shmem>>>(
        static_cast<const float*>(X.data),
        static_cast<float*>(Y.data),
        L, num_heads, head_dim, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void l2_norm_backward(const ::brotensor::Tensor& X,
                      int head_dim, int num_heads, float eps,
                      const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX) {
    check_fp32(X,  "l2_norm_backward", "X");
    check_fp32(dY, "l2_norm_backward", "dY");
    check_shape(X,  head_dim, num_heads, "l2_norm_backward", "X");
    check_shape(dY, head_dim, num_heads, "l2_norm_backward", "dY");
    if (dY.rows != X.rows) {
        throw std::runtime_error("l2_norm_backward: dY.rows != X.rows");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (dX.rows != L || dX.cols != D || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(L, D, ::brotensor::Dtype::FP32);
    }
    if (L == 0 || D == 0) return;

    const int blocks = L * num_heads;
    const int block  = L2_BLOCK;
    const size_t shmem = 2 * block * sizeof(float);
    l2_norm_backward_kernel<<<blocks, block, shmem>>>(
        static_cast<const float*>(X.data),
        static_cast<const float*>(dY.data),
        static_cast<float*>(dX.data),
        L, num_heads, head_dim, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_l2_norm(::brotensor::detail::OpsVTable& v) {
    v.l2_norm_forward  = &l2_norm_forward;
    v.l2_norm_backward = &l2_norm_backward;
}

} // namespace brotensor::detail::cuda
