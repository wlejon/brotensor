// CUDA M-RoPE (Qwen2.5-VL / Qwen3-VL).
//
// head_dim is split into three contiguous sub-ranges of widths 2*d_t, 2*d_h,
// 2*d_w (in order). Each sub-range rotates pair-wise using its own per-axis
// (cos, sin) table indexed by its own per-axis position-ID stream.
//
// One thread per (row, head, pair). The thread figures out which axis its
// pair belongs to and reads the matching axis table at row -> pos_a[row].
// Avoids stitching a (seq_len, head_dim) cos/sin table host-side.
//
// Also hosts the fill_cuda_vtable_qwen3_vl_polish entry point that registers
// both ops (spatial_merge_2x2_forward + rope_apply_mrope) into the CUDA vtable.

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

// Defined in spatial_merge.cu.
void spatial_merge_2x2_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W,
                               bool channel_major,
                               ::brotensor::Tensor& Y);
void pixel_shuffle_upsample_2x_forward(const ::brotensor::Tensor& X,
                                       int N, int C_in, int H, int W,
                                       int C_out, ::brotensor::Tensor& Y);
void patch_unpack_forward(const ::brotensor::Tensor& tokens,
                          int hp, int wp, int P, int C_total, int C_keep,
                          bool channel_major, ::brotensor::Tensor& Y);

namespace {

constexpr int MR_BLOCK = 256;

__device__ inline float mr_ld(const float& x)         { return x; }
__device__ inline float mr_ld(const __half& x)        { return __half2float(x); }
__device__ inline float mr_ld(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ inline void  mr_st(float& d, float v)         { d = v; }
__device__ inline void  mr_st(__half& d, float v)        { d = __float2half(v); }
__device__ inline void  mr_st(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

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
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half_d;
        const int rest = idx / half_d;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float x0 = mr_ld(X[base_off + 2 * i]);
        const float x1 = mr_ld(X[base_off + 2 * i + 1]);

        // Axis dispatch on the pair index i.
        float c, s;
        if (i < d_t) {
            const int pos = pos_t[row];
            c = cos_t[pos * d_t + i];
            s = sin_t[pos * d_t + i];
        } else if (i < d_t + d_h) {
            const int local = i - d_t;
            const int pos = pos_h[row];
            c = cos_h[pos * d_h + local];
            s = sin_h[pos * d_h + local];
        } else {
            // i in [d_t + d_h, d_t + d_h + d_w == half_d).
            const int local = i - d_t - d_h;
            const int pos = pos_w[row];
            c = cos_w[pos * d_w + local];
            s = sin_w[pos * d_w + local];
        }
        mr_st(Y[base_off + 2 * i],     x0 * c - x1 * s);
        mr_st(Y[base_off + 2 * i + 1], x0 * s + x1 * c);
    }
}

inline int grid_for(int n) { return (n + MR_BLOCK - 1) / MR_BLOCK; }

void check_axis_tbl(const ::brotensor::Tensor& cos_a,
                    const ::brotensor::Tensor& sin_a,
                    const char* axis, int d_a) {
    if (d_a < 0) {
        throw std::runtime_error(std::string("rope_apply_mrope: d_") + axis +
                                 " must be non-negative");
    }
    if (d_a == 0) return;
    if (cos_a.dtype != Dtype::FP32 || sin_a.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis + " must be FP32");
    }
    if (cos_a.cols != d_a || sin_a.cols != d_a) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis +
                                 " must each have cols == d_" + axis);
    }
    if (cos_a.rows != sin_a.rows || cos_a.rows < 1) {
        throw std::runtime_error(std::string("rope_apply_mrope: cos_") + axis +
                                 " / sin_" + axis + " row count mismatch");
    }
}

} // namespace

void rope_apply_mrope(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& cos_t,
                      const ::brotensor::Tensor& sin_t,
                      const ::brotensor::Tensor& cos_h,
                      const ::brotensor::Tensor& sin_h,
                      const ::brotensor::Tensor& cos_w,
                      const ::brotensor::Tensor& sin_w,
                      const int32_t* pos_t, const int32_t* pos_h,
                      const int32_t* pos_w,
                      int head_dim, int num_heads,
                      int d_t, int d_h, int d_w,
                      ::brotensor::Tensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_apply_mrope: head_dim must be a "
                                 "positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_apply_mrope: num_heads must be positive");
    }
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("rope_apply_mrope: X must be FP32, FP16, or BF16");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_apply_mrope: X.cols != "
                                 "num_heads * head_dim");
    }
    if (2 * (d_t + d_h + d_w) != head_dim) {
        throw std::runtime_error("rope_apply_mrope: 2*(d_t + d_h + d_w) "
                                 "must equal head_dim");
    }
    check_axis_tbl(cos_t, sin_t, "t", d_t);
    check_axis_tbl(cos_h, sin_h, "h", d_h);
    check_axis_tbl(cos_w, sin_w, "w", d_w);
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    if (L == 0 || num_heads == 0 || head_dim == 0) return;
    if (d_t > 0 && !pos_t) throw std::runtime_error("rope_apply_mrope: pos_t null");
    if (d_h > 0 && !pos_h) throw std::runtime_error("rope_apply_mrope: pos_h null");
    if (d_w > 0 && !pos_w) throw std::runtime_error("rope_apply_mrope: pos_w null");

    const int half_d = head_dim / 2;
    const int total  = L * num_heads * half_d;
    if (total == 0) return;
    const int blocks = grid_for(total);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    const float* cos_tp = d_t > 0 ? static_cast<const float*>(cos_t.data) : nullptr;
    const float* sin_tp = d_t > 0 ? static_cast<const float*>(sin_t.data) : nullptr;
    const float* cos_hp = d_h > 0 ? static_cast<const float*>(cos_h.data) : nullptr;
    const float* sin_hp = d_h > 0 ? static_cast<const float*>(sin_h.data) : nullptr;
    const float* cos_wp = d_w > 0 ? static_cast<const float*>(cos_w.data) : nullptr;
    const float* sin_wp = d_w > 0 ? static_cast<const float*>(sin_w.data) : nullptr;
    const int* d_pt = reinterpret_cast<const int*>(pos_t);
    const int* d_ph = reinterpret_cast<const int*>(pos_h);
    const int* d_pw = reinterpret_cast<const int*>(pos_w);

    switch (X.dtype) {
    case Dtype::FP16:
        rope_mrope_kernel<__half><<<blocks, MR_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            cos_tp, sin_tp, cos_hp, sin_hp, cos_wp, sin_wp,
            d_pt, d_ph, d_pw, L, num_heads, head_dim, d_t, d_h, d_w);
        break;
    case Dtype::BF16:
        rope_mrope_kernel<__nv_bfloat16><<<blocks, MR_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            cos_tp, sin_tp, cos_hp, sin_hp, cos_wp, sin_wp,
            d_pt, d_ph, d_pw, L, num_heads, head_dim, d_t, d_h, d_w);
        break;
    default:  // FP32
        rope_mrope_kernel<float><<<blocks, MR_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            cos_tp, sin_tp, cos_hp, sin_hp, cos_wp, sin_wp,
            d_pt, d_ph, d_pw, L, num_heads, head_dim, d_t, d_h, d_w);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// Per-cluster vtable contribution for the Qwen3-VL polish ops.
void fill_cuda_vtable_qwen3_vl_polish(::brotensor::detail::OpsVTable& v) {
    v.spatial_merge_2x2_forward = &spatial_merge_2x2_forward;
    v.pixel_shuffle_upsample_2x_forward = &pixel_shuffle_upsample_2x_forward;
    v.patch_unpack_forward      = &patch_unpack_forward;
    v.rope_apply_mrope          = &rope_apply_mrope;
}

} // namespace detail::cuda
} // namespace brotensor
