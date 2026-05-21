// Rotary position embedding (RoPE) forward + backward.
// Per-head: rotate pairs (x_{2i}, x_{2i+1}) by angle theta = pos * base^{-2i/hd}.

#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <math.h>
#include <stdexcept>
#include <string>

namespace brotensor {
namespace detail::cuda {

namespace {

constexpr int RP_BLOCK = 256;

__device__ inline float rope_theta(int pair_i, int head_dim, float base) {
    // theta_i = base^{-2i/head_dim} = exp(-2i/hd * log(base))
    return __expf(-static_cast<float>(2 * pair_i) /
                  static_cast<float>(head_dim) * __logf(base));
}

// One thread per pair (row, head, i). Forward.
__global__ void rope_forward_fp32_kernel(const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         int L, int num_heads, int head_dim,
                                         int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i      = idx % half;
        const int rest   = idx / half;
        const int h      = rest % num_heads;
        const int row    = rest / num_heads;
        const int pos    = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c    = __cosf(theta);
        const float s    = __sinf(theta);
        const int D      = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float x0 = X[base_off + 2 * i];
        const float x1 = X[base_off + 2 * i + 1];
        Y[base_off + 2 * i]     = x0 * c - x1 * s;
        Y[base_off + 2 * i + 1] = x0 * s + x1 * c;
    }
}

__global__ void rope_forward_fp16_kernel(const __half* __restrict__ X,
                                         __half* __restrict__ Y,
                                         int L, int num_heads, int head_dim,
                                         int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int pos  = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c  = __cosf(theta);
        const float s  = __sinf(theta);
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float x0 = __half2float(X[base_off + 2 * i]);
        const float x1 = __half2float(X[base_off + 2 * i + 1]);
        Y[base_off + 2 * i]     = __float2half(x0 * c - x1 * s);
        Y[base_off + 2 * i + 1] = __float2half(x0 * s + x1 * c);
    }
}

__global__ void rope_backward_fp32_kernel(const float* __restrict__ dY,
                                          float* __restrict__ dX,
                                          int L, int num_heads, int head_dim,
                                          int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int pos  = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c  = __cosf(theta);
        const float s  = __sinf(theta);
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float dy0 = dY[base_off + 2 * i];
        const float dy1 = dY[base_off + 2 * i + 1];
        // Inverse rotation (transpose of R(θ)).
        dX[base_off + 2 * i]     = dy0 * c + dy1 * s;
        dX[base_off + 2 * i + 1] = -dy0 * s + dy1 * c;
    }
}

__global__ void rope_backward_fp16_kernel(const __half* __restrict__ dY,
                                          __half* __restrict__ dX,
                                          int L, int num_heads, int head_dim,
                                          int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int pos  = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c  = __cosf(theta);
        const float s  = __sinf(theta);
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float dy0 = __half2float(dY[base_off + 2 * i]);
        const float dy1 = __half2float(dY[base_off + 2 * i + 1]);
        dX[base_off + 2 * i]     = __float2half(dy0 * c + dy1 * s);
        dX[base_off + 2 * i + 1] = __float2half(-dy0 * s + dy1 * c);
    }
}

inline int grid_for(int n) { return (n + RP_BLOCK - 1) / RP_BLOCK; }

// ─── RoPE with explicit cos/sin tables (rope_apply) ───────────────────────
//
// cos_tbl / sin_tbl are (L, head_dim/2) FP32: one angle per (row, pair),
// shared across heads. X / Y are typed (FP32 / FP16 / BF16); math in FP32.

__device__ inline float rp_ld(const float& x)         { return x; }
__device__ inline float rp_ld(const __half& x)        { return __half2float(x); }
__device__ inline float rp_ld(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ inline void  rp_st(float& d, float v)         { d = v; }
__device__ inline void  rp_st(__half& d, float v)        { d = __float2half(v); }
__device__ inline void  rp_st(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

template <typename T>
__global__ void rope_apply_fwd_kernel(const T* __restrict__ X,
                                      const float* __restrict__ cos_tbl,
                                      const float* __restrict__ sin_tbl,
                                      T* __restrict__ Y,
                                      int L, int num_heads, int head_dim) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const float c  = cos_tbl[row * half + i];
        const float s  = sin_tbl[row * half + i];
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float x0 = rp_ld(X[base_off + 2 * i]);
        const float x1 = rp_ld(X[base_off + 2 * i + 1]);
        rp_st(Y[base_off + 2 * i],     x0 * c - x1 * s);
        rp_st(Y[base_off + 2 * i + 1], x0 * s + x1 * c);
    }
}

template <typename T>
__global__ void rope_apply_bwd_kernel(const T* __restrict__ dY,
                                      const float* __restrict__ cos_tbl,
                                      const float* __restrict__ sin_tbl,
                                      T* __restrict__ dX,
                                      int L, int num_heads, int head_dim) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const float c  = cos_tbl[row * half + i];
        const float s  = sin_tbl[row * half + i];
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float dy0 = rp_ld(dY[base_off + 2 * i]);
        const float dy1 = rp_ld(dY[base_off + 2 * i + 1]);
        rp_st(dX[base_off + 2 * i],      dy0 * c + dy1 * s);
        rp_st(dX[base_off + 2 * i + 1], -dy0 * s + dy1 * c);
    }
}

} // namespace

void rope_forward(const ::brotensor::Tensor& X, int head_dim, int num_heads,
                  int seq_offset, float theta_base, ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_forward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_forward: num_heads must be positive");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_forward: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;
    const int blocks = grid_for(total);
    if (X.dtype == Dtype::FP16) {
        rope_forward_fp16_kernel<<<blocks, RP_BLOCK>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else {
        rope_forward_fp32_kernel<<<blocks, RP_BLOCK>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rope_backward(const ::brotensor::Tensor& dY, int head_dim, int num_heads,
                   int seq_offset, float theta_base, ::brotensor::Tensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_backward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_backward: num_heads must be positive");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;
    const int blocks = grid_for(total);
    if (dY.dtype == Dtype::FP16) {
        rope_backward_fp16_kernel<<<blocks, RP_BLOCK>>>(
            static_cast<const __half*>(dY.data),
            static_cast<__half*>(dX.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else {
        rope_backward_fp32_kernel<<<blocks, RP_BLOCK>>>(
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── RoPE-with-tables public ops ──────────────────────────────────────────

namespace {

void check_rope_tables(const ::brotensor::Tensor& cos_tbl,
                       const ::brotensor::Tensor& sin_tbl,
                       const char* op, int L, int half) {
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": cos_tbl / sin_tbl must be FP32");
    }
    if (cos_tbl.size() != L * half || sin_tbl.size() != L * half) {
        throw std::runtime_error(std::string(op) +
                                 ": cos_tbl / sin_tbl must each be (L, head_dim/2)");
    }
}

} // namespace

void rope_apply(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("rope_apply: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    const int half = head_dim / 2;
    check_rope_tables(cos_tbl, sin_tbl, "rope_apply", L, half);
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * half;
    if (total == 0) return;
    const int blocks = grid_for(total);
    const float* cos_p = static_cast<const float*>(cos_tbl.data);
    const float* sin_p = static_cast<const float*>(sin_tbl.data);
    switch (X.dtype) {
    case Dtype::FP32:
        rope_apply_fwd_kernel<float><<<blocks, RP_BLOCK>>>(
            static_cast<const float*>(X.data), cos_p, sin_p,
            static_cast<float*>(Y.data), L, num_heads, head_dim);
        break;
    case Dtype::FP16:
        rope_apply_fwd_kernel<__half><<<blocks, RP_BLOCK>>>(
            static_cast<const __half*>(X.data), cos_p, sin_p,
            static_cast<__half*>(Y.data), L, num_heads, head_dim);
        break;
    default:  // BF16
        rope_apply_fwd_kernel<__nv_bfloat16><<<blocks, RP_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data), cos_p, sin_p,
            static_cast<__nv_bfloat16*>(Y.data), L, num_heads, head_dim);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rope_apply_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& cos_tbl,
                         const ::brotensor::Tensor& sin_tbl,
                         int head_dim, int num_heads, ::brotensor::Tensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_backward: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_backward: num_heads must be positive");
    }
    if (dY.dtype != Dtype::FP32 && dY.dtype != Dtype::FP16 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("rope_apply_backward: dY must be FP32, FP16, or BF16");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    const int half = head_dim / 2;
    check_rope_tables(cos_tbl, sin_tbl, "rope_apply_backward", L, half);
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const int total = L * num_heads * half;
    if (total == 0) return;
    const int blocks = grid_for(total);
    const float* cos_p = static_cast<const float*>(cos_tbl.data);
    const float* sin_p = static_cast<const float*>(sin_tbl.data);
    switch (dY.dtype) {
    case Dtype::FP32:
        rope_apply_bwd_kernel<float><<<blocks, RP_BLOCK>>>(
            static_cast<const float*>(dY.data), cos_p, sin_p,
            static_cast<float*>(dX.data), L, num_heads, head_dim);
        break;
    case Dtype::FP16:
        rope_apply_bwd_kernel<__half><<<blocks, RP_BLOCK>>>(
            static_cast<const __half*>(dY.data), cos_p, sin_p,
            static_cast<__half*>(dX.data), L, num_heads, head_dim);
        break;
    default:  // BF16
        rope_apply_bwd_kernel<__nv_bfloat16><<<blocks, RP_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(dY.data), cos_p, sin_p,
            static_cast<__nv_bfloat16*>(dX.data), L, num_heads, head_dim);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
