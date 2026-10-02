// Native AMD HIP Rotary Position Embeddings (RoPE) for RDNA 3.5 (Wave32).
//
// Implements:
// - rope_forward / rope_backward (on-the-fly theta calculation)
// - rope_apply / rope_apply_backward (precomputed 2D cos/sin tables)
// - rope_apply_perhead (per-head cos/sin tables)
// - rope_qkv_packed_inplace (fused in-place Q/K rotation for packed encoder)
// - rope_apply_mrope (multimodal 3D RoPE for Qwen2.5-VL / Qwen3-VL)
//
// Optimized for Wave32 execution with vectorized pair processing.

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/hip_check.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int kRopeBlock = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int grid_for(int total) {
    return (total + kRopeBlock - 1) / kRopeBlock;
}

__device__ __forceinline__ float rope_theta(int pair_i, int head_dim, float base) {
    return __expf(-static_cast<float>(2 * pair_i) / static_cast<float>(head_dim) * __logf(base));
}

template <typename T> struct RopeScalar;
template <> struct RopeScalar<float> {
    __device__ static __forceinline__ float ld(float v) { return v; }
    __device__ static __forceinline__ float st(float v) { return v; }
};
template <> struct RopeScalar<__half> {
    __device__ static __forceinline__ float ld(__half v) { return __half2float(v); }
    __device__ static __forceinline__ __half st(float v) { return __float2half(v); }
};
template <> struct RopeScalar<hip_bfloat16> {
    __device__ static __forceinline__ float ld(hip_bfloat16 v) { return float(v); }
    __device__ static __forceinline__ hip_bfloat16 st(float v) { return hip_bfloat16(v); }
};

// ─── Forward / Backward Kernels (On-the-fly theta) ───────────────────────────

template <typename T>
__global__ void rope_fwd_kernel(const T* __restrict__ X,
                                T* __restrict__ Y,
                                int L, int num_heads, int head_dim,
                                int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i    = idx % half;
    const int rest = idx / half;
    const int h    = rest % num_heads;
    const int row  = rest / num_heads;
    const int pos  = row + seq_offset;

    const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
    float s, c;
    __sincosf(theta, &s, &c);

    const int D = num_heads * head_dim;
    const int base_off = row * D + h * head_dim;
    const float x0 = RopeScalar<T>::ld(X[base_off + 2 * i]);
    const float x1 = RopeScalar<T>::ld(X[base_off + 2 * i + 1]);

    Y[base_off + 2 * i]     = RopeScalar<T>::st(x0 * c - x1 * s);
    Y[base_off + 2 * i + 1] = RopeScalar<T>::st(x0 * s + x1 * c);
}

template <typename T>
__global__ void rope_bwd_kernel(const T* __restrict__ dY,
                                T* __restrict__ dX,
                                int L, int num_heads, int head_dim,
                                int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i    = idx % half;
    const int rest = idx / half;
    const int h    = rest % num_heads;
    const int row  = rest / num_heads;
    const int pos  = row + seq_offset;

    const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
    float s, c;
    __sincosf(theta, &s, &c);

    const int D = num_heads * head_dim;
    const int base_off = row * D + h * head_dim;
    const float dy0 = RopeScalar<T>::ld(dY[base_off + 2 * i]);
    const float dy1 = RopeScalar<T>::ld(dY[base_off + 2 * i + 1]);

    dX[base_off + 2 * i]     = RopeScalar<T>::st(dy0 * c + dy1 * s);
    dX[base_off + 2 * i + 1] = RopeScalar<T>::st(-dy0 * s + dy1 * c);
}

// ─── Table-Based RoPE Kernels ────────────────────────────────────────────────

template <typename T>
__global__ void rope_apply_fwd_kernel(const T* __restrict__ X,
                                      const float* __restrict__ cos_p,
                                      const float* __restrict__ sin_p,
                                      T* __restrict__ Y,
                                      int L, int num_heads, int head_dim) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i    = idx % half;
    const int rest = idx / half;
    const int h    = rest % num_heads;
    const int row  = rest / num_heads;

    const float c = cos_p[row * half + i];
    const float s = sin_p[row * half + i];

    const int D = num_heads * head_dim;
    const int base_off = row * D + h * head_dim;
    const float x0 = RopeScalar<T>::ld(X[base_off + 2 * i]);
    const float x1 = RopeScalar<T>::ld(X[base_off + 2 * i + 1]);

    Y[base_off + 2 * i]     = RopeScalar<T>::st(x0 * c - x1 * s);
    Y[base_off + 2 * i + 1] = RopeScalar<T>::st(x0 * s + x1 * c);
}

template <typename T>
__global__ void rope_apply_perhead_fwd_kernel(const T* __restrict__ X,
                                              const float* __restrict__ cos_p,
                                              const float* __restrict__ sin_p,
                                              T* __restrict__ Y,
                                              int L, int num_heads, int head_dim) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i    = idx % half;
    const int rest = idx / half;
    const int h    = rest % num_heads;
    const int row  = rest / num_heads;

    const int tbl_row = row * num_heads + h;
    const float c = cos_p[tbl_row * half + i];
    const float s = sin_p[tbl_row * half + i];

    const int D = num_heads * head_dim;
    const int base_off = row * D + h * head_dim;
    const float x0 = RopeScalar<T>::ld(X[base_off + 2 * i]);
    const float x1 = RopeScalar<T>::ld(X[base_off + 2 * i + 1]);

    Y[base_off + 2 * i]     = RopeScalar<T>::st(x0 * c - x1 * s);
    Y[base_off + 2 * i + 1] = RopeScalar<T>::st(x0 * s + x1 * c);
}

template <typename T>
__global__ void rope_apply_bwd_kernel(const T* __restrict__ dY,
                                      const float* __restrict__ cos_p,
                                      const float* __restrict__ sin_p,
                                      T* __restrict__ dX,
                                      int L, int num_heads, int head_dim) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i    = idx % half;
    const int rest = idx / half;
    const int h    = rest % num_heads;
    const int row  = rest / num_heads;

    const float c = cos_p[row * half + i];
    const float s = sin_p[row * half + i];

    const int D = num_heads * head_dim;
    const int base_off = row * D + h * head_dim;
    const float dy0 = RopeScalar<T>::ld(dY[base_off + 2 * i]);
    const float dy1 = RopeScalar<T>::ld(dY[base_off + 2 * i + 1]);

    dX[base_off + 2 * i]     = RopeScalar<T>::st(dy0 * c + dy1 * s);
    dX[base_off + 2 * i + 1] = RopeScalar<T>::st(-dy0 * s + dy1 * c);
}

// ─── Packed QKV In-place RoPE ────────────────────────────────────────────────

template <typename T>
__global__ void rope_qkv_packed_kernel(T* __restrict__ QKV,
                                       const float* __restrict__ cos_p,
                                       const float* __restrict__ sin_p,
                                       const int* __restrict__ pos_p,
                                       int D, int half) {
    const int row = blockIdx.y;
    const int p = pos_p[row];
    // One thread per (head, pair): D/2 pairs per section, each rotated in
    // both the Q and the K section of the row.
    const int pj = blockIdx.x * blockDim.x + threadIdx.x;
    if (pj >= D / 2) return;

    const int pair_idx = pj % half;
    const int off = (pj / half) * 2 * half + 2 * pair_idx;
    const float c = cos_p[static_cast<size_t>(p) * half + pair_idx];
    const float s = sin_p[static_cast<size_t>(p) * half + pair_idx];

    T* r = QKV + static_cast<size_t>(row) * (3 * D);

    // Rotate Q (first D)
    const float q0 = RopeScalar<T>::ld(r[off]);
    const float q1 = RopeScalar<T>::ld(r[off + 1]);
    r[off]     = RopeScalar<T>::st(q0 * c - q1 * s);
    r[off + 1] = RopeScalar<T>::st(q0 * s + q1 * c);

    // Rotate K (second D)
    const float k0 = RopeScalar<T>::ld(r[D + off]);
    const float k1 = RopeScalar<T>::ld(r[D + off + 1]);
    r[D + off]     = RopeScalar<T>::st(k0 * c - k1 * s);
    r[D + off + 1] = RopeScalar<T>::st(k0 * s + k1 * c);
}

// ─── M-RoPE Kernel (Qwen2.5-VL / Qwen3-VL 3D Positional Embedding) ───────────

template <typename T>
__global__ void rope_mrope_kernel(const T* __restrict__ X,
                                  T* __restrict__ Y,
                                  const float* __restrict__ cos_t,
                                  const float* __restrict__ sin_t,
                                  const float* __restrict__ cos_h,
                                  const float* __restrict__ sin_h,
                                  const float* __restrict__ cos_w,
                                  const float* __restrict__ sin_w,
                                  const int* __restrict__ pos_t,
                                  const int* __restrict__ pos_h,
                                  const int* __restrict__ pos_w,
                                  int L, int num_heads, int head_dim,
                                  int d_t, int d_h, int d_w) {
    const int half_d = head_dim >> 1;
    const int total  = L * num_heads * half_d;
    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int i    = idx % half_d;
    const int rest = idx / half_d;
    const int h    = rest % num_heads;
    const int row  = rest / num_heads;
    const int D    = num_heads * head_dim;
    const int base_off = row * D + h * head_dim;

    const float x0 = RopeScalar<T>::ld(X[base_off + 2 * i]);
    const float x1 = RopeScalar<T>::ld(X[base_off + 2 * i + 1]);

    float c, s;
    if (i < d_t) {
        const int p = pos_t[row];
        c = cos_t[p * d_t + i];
        s = sin_t[p * d_t + i];
    } else if (i < d_t + d_h) {
        const int local = i - d_t;
        const int p = pos_h[row];
        c = cos_h[p * d_h + local];
        s = sin_h[p * d_h + local];
    } else {
        const int local = i - d_t - d_h;
        const int p = pos_w[row];
        c = cos_w[p * d_w + local];
        s = sin_w[p * d_w + local];
    }

    Y[base_off + 2 * i]     = RopeScalar<T>::st(x0 * c - x1 * s);
    Y[base_off + 2 * i + 1] = RopeScalar<T>::st(x0 * s + x1 * c);
}

} // namespace

// ─── Public Function Implementations ─────────────────────────────────────────

void hip_rope_forward(const ::brotensor::Tensor& X, int head_dim, int num_heads,
                      int seq_offset, float theta_base, ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_forward: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_forward: num_heads must be positive");
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_forward: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("hip_rope_forward: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;

    const int blocks = grid_for(total);
    hipStream_t stream = cur_stream();
    if (X.dtype == Dtype::FP16) {
        rope_fwd_kernel<__half><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else if (X.dtype == Dtype::BF16) {
        rope_fwd_kernel<hip_bfloat16><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else {
        rope_fwd_kernel<float><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_rope_backward(const ::brotensor::Tensor& dY, int head_dim, int num_heads,
                       int seq_offset, float theta_base, ::brotensor::Tensor& dX) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_backward: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_backward: num_heads must be positive");
    if (dY.dtype != Dtype::FP32 && dY.dtype != Dtype::FP16 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_backward: dY must be FP32, FP16, or BF16");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("hip_rope_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;

    const int blocks = grid_for(total);
    hipStream_t stream = cur_stream();
    if (dY.dtype == Dtype::FP16) {
        rope_bwd_kernel<__half><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const __half*>(dY.data), static_cast<__half*>(dX.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else if (dY.dtype == Dtype::BF16) {
        rope_bwd_kernel<hip_bfloat16><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY.data), static_cast<hip_bfloat16*>(dX.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else {
        rope_bwd_kernel<float><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
            L, num_heads, head_dim, seq_offset, theta_base);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_rope_apply(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                    const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                    ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_apply: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_apply: num_heads must be positive");
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_apply: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("hip_rope_apply: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    const int half = head_dim / 2;
    if (cos_tbl.size() != L * half || sin_tbl.size() != L * half) {
        throw std::runtime_error("hip_rope_apply: cos_tbl / sin_tbl must be (L, head_dim/2)");
    }
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * half;
    if (total == 0) return;

    const int blocks = grid_for(total);
    const float* cos_p = static_cast<const float*>(cos_tbl.data);
    const float* sin_p = static_cast<const float*>(sin_tbl.data);
    hipStream_t stream = cur_stream();

    if (X.dtype == Dtype::FP16) {
        rope_apply_fwd_kernel<__half><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const __half*>(X.data), cos_p, sin_p, static_cast<__half*>(Y.data),
            L, num_heads, head_dim);
    } else if (X.dtype == Dtype::BF16) {
        rope_apply_fwd_kernel<hip_bfloat16><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), cos_p, sin_p, static_cast<hip_bfloat16*>(Y.data),
            L, num_heads, head_dim);
    } else {
        rope_apply_fwd_kernel<float><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const float*>(X.data), cos_p, sin_p, static_cast<float*>(Y.data),
            L, num_heads, head_dim);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_rope_apply_perhead(const ::brotensor::Tensor& X, const ::brotensor::Tensor& cos_tbl,
                            const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                            ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_apply_perhead: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_apply_perhead: num_heads must be positive");
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_apply_perhead: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("hip_rope_apply_perhead: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    const int half = head_dim / 2;
    if (cos_tbl.size() != L * num_heads * half || sin_tbl.size() != L * num_heads * half) {
        throw std::runtime_error("hip_rope_apply_perhead: cos_tbl/sin_tbl must be (L*num_heads, head_dim/2)");
    }
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * half;
    if (total == 0) return;

    const int blocks = grid_for(total);
    const float* cos_p = static_cast<const float*>(cos_tbl.data);
    const float* sin_p = static_cast<const float*>(sin_tbl.data);
    hipStream_t stream = cur_stream();

    if (X.dtype == Dtype::FP16) {
        rope_apply_perhead_fwd_kernel<__half><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const __half*>(X.data), cos_p, sin_p, static_cast<__half*>(Y.data),
            L, num_heads, head_dim);
    } else if (X.dtype == Dtype::BF16) {
        rope_apply_perhead_fwd_kernel<hip_bfloat16><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), cos_p, sin_p, static_cast<hip_bfloat16*>(Y.data),
            L, num_heads, head_dim);
    } else {
        rope_apply_perhead_fwd_kernel<float><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const float*>(X.data), cos_p, sin_p, static_cast<float*>(Y.data),
            L, num_heads, head_dim);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_rope_apply_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& cos_tbl,
                             const ::brotensor::Tensor& sin_tbl, int head_dim, int num_heads,
                             ::brotensor::Tensor& dX) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_apply_backward: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_apply_backward: num_heads must be positive");
    if (dY.dtype != Dtype::FP32 && dY.dtype != Dtype::FP16 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_apply_backward: dY must be FP32, FP16, or BF16");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("hip_rope_apply_backward: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    const int half = head_dim / 2;
    if (cos_tbl.size() != L * half || sin_tbl.size() != L * half) {
        throw std::runtime_error("hip_rope_apply_backward: cos_tbl/sin_tbl must be (L, head_dim/2)");
    }
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const int total = L * num_heads * half;
    if (total == 0) return;

    const int blocks = grid_for(total);
    const float* cos_p = static_cast<const float*>(cos_tbl.data);
    const float* sin_p = static_cast<const float*>(sin_tbl.data);
    hipStream_t stream = cur_stream();

    if (dY.dtype == Dtype::FP16) {
        rope_apply_bwd_kernel<__half><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const __half*>(dY.data), cos_p, sin_p, static_cast<__half*>(dX.data),
            L, num_heads, head_dim);
    } else if (dY.dtype == Dtype::BF16) {
        rope_apply_bwd_kernel<hip_bfloat16><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY.data), cos_p, sin_p, static_cast<hip_bfloat16*>(dX.data),
            L, num_heads, head_dim);
    } else {
        rope_apply_bwd_kernel<float><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const float*>(dY.data), cos_p, sin_p, static_cast<float*>(dX.data),
            L, num_heads, head_dim);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_rope_qkv_packed_inplace(::brotensor::Tensor& QKV, const ::brotensor::Tensor& cos_tbl,
                                 const ::brotensor::Tensor& sin_tbl, const ::brotensor::Tensor& pos,
                                 int num_heads, int head_dim) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_qkv_packed_inplace: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_qkv_packed_inplace: num_heads must be positive");
    if (QKV.cols != 3 * num_heads * head_dim) {
        throw std::runtime_error("hip_rope_qkv_packed_inplace: QKV.cols != 3 * num_heads * head_dim");
    }
    const int L = QKV.rows;
    const int D = num_heads * head_dim;
    const int half = head_dim / 2;
    if (pos.dtype != Dtype::INT32 || pos.rows != L) {
        throw std::runtime_error("hip_rope_qkv_packed_inplace: pos must be (L, 1) INT32");
    }
    if (cos_tbl.cols != half || sin_tbl.cols != half) {
        throw std::runtime_error("hip_rope_qkv_packed_inplace: cos_tbl/sin_tbl must have cols == head_dim/2");
    }
    if (cos_tbl.dtype != Dtype::FP32 || sin_tbl.dtype != Dtype::FP32) {
        throw std::runtime_error("hip_rope_qkv_packed_inplace: cos_tbl/sin_tbl must be FP32");
    }
    if (QKV.dtype != Dtype::FP32 && QKV.dtype != Dtype::FP16 && QKV.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_qkv_packed_inplace: QKV must be FP32, FP16, or BF16");
    }
    if (L == 0) return;

    const dim3 grid((D / 2 + 255) / 256, L);
    const float* c = static_cast<const float*>(cos_tbl.data);
    const float* s = static_cast<const float*>(sin_tbl.data);
    const int* p = static_cast<const int*>(pos.data);
    hipStream_t stream = cur_stream();

    if (QKV.dtype == Dtype::FP16) {
        rope_qkv_packed_kernel<__half><<<grid, 256, 0, stream>>>(
            static_cast<__half*>(QKV.data), c, s, p, D, half);
    } else if (QKV.dtype == Dtype::BF16) {
        rope_qkv_packed_kernel<hip_bfloat16><<<grid, 256, 0, stream>>>(
            static_cast<hip_bfloat16*>(QKV.data), c, s, p, D, half);
    } else {
        rope_qkv_packed_kernel<float><<<grid, 256, 0, stream>>>(
            static_cast<float*>(QKV.data), c, s, p, D, half);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void hip_rope_apply_mrope(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& cos_t, const ::brotensor::Tensor& sin_t,
                          const ::brotensor::Tensor& cos_h, const ::brotensor::Tensor& sin_h,
                          const ::brotensor::Tensor& cos_w, const ::brotensor::Tensor& sin_w,
                          const int32_t* pos_t, const int32_t* pos_h, const int32_t* pos_w,
                          int head_dim, int num_heads,
                          int d_t, int d_h, int d_w,
                          ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("hip_rope_apply_mrope: head_dim must be positive and even");
    }
    if (num_heads <= 0) throw std::runtime_error("hip_rope_apply_mrope: num_heads must be positive");
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("hip_rope_apply_mrope: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("hip_rope_apply_mrope: X.cols != num_heads * head_dim");
    }
    if (2 * (d_t + d_h + d_w) != head_dim) {
        throw std::runtime_error("hip_rope_apply_mrope: 2*(d_t + d_h + d_w) != head_dim");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;

    const float* ct = d_t > 0 ? static_cast<const float*>(cos_t.data) : nullptr;
    const float* st = d_t > 0 ? static_cast<const float*>(sin_t.data) : nullptr;
    const float* ch = d_h > 0 ? static_cast<const float*>(cos_h.data) : nullptr;
    const float* sh = d_h > 0 ? static_cast<const float*>(sin_h.data) : nullptr;
    const float* cw = d_w > 0 ? static_cast<const float*>(cos_w.data) : nullptr;
    const float* sw = d_w > 0 ? static_cast<const float*>(sin_w.data) : nullptr;
    const int*   pt = d_t > 0 ? pos_t : nullptr;
    const int*   ph = d_h > 0 ? pos_h : nullptr;
    const int*   pw = d_w > 0 ? pos_w : nullptr;

    const int blocks = grid_for(total);
    hipStream_t stream = cur_stream();

    if (X.dtype == Dtype::FP16) {
        rope_mrope_kernel<__half><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            ct, st, ch, sh, cw, sw, pt, ph, pw,
            L, num_heads, head_dim, d_t, d_h, d_w);
    } else if (X.dtype == Dtype::BF16) {
        rope_mrope_kernel<hip_bfloat16><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            ct, st, ch, sh, cw, sw, pt, ph, pw,
            L, num_heads, head_dim, d_t, d_h, d_w);
    } else {
        rope_mrope_kernel<float><<<blocks, kRopeBlock, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            ct, st, ch, sh, cw, sw, pt, ph, pw,
            L, num_heads, head_dim, d_t, d_h, d_w);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_rope(::brotensor::detail::OpsVTable& v) {
    v.rope_forward             = &hip_rope_forward;
    v.rope_backward            = &hip_rope_backward;
    v.rope_apply               = &hip_rope_apply;
    v.rope_apply_perhead       = &hip_rope_apply_perhead;
    v.rope_apply_backward      = &hip_rope_apply_backward;
    v.rope_qkv_packed_inplace  = &hip_rope_qkv_packed_inplace;
    v.rope_apply_mrope         = &hip_rope_apply_mrope;
}

} // namespace brotensor::detail::hip
