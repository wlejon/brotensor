// ─── HIP INT8 Quantization (W8A16) & Quant VTable Registration ────────────
//
// AMD ROCm / HIP port of src/cuda/int8_quant.cu.
//
// Supports:
// - matmul_int8w_fp16
// - linear_forward_batched_int8w_fp16 (FP16 & BF16)
// - conv2d_int8w_fp16_forward
// - fill_hip_vtable_quant

#include "detail/hip_check.h"
#include "detail/quant_internal.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void* hip_current_stream(int dev = -1);
static inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

namespace {

constexpr int MM_TILE = 16;

template <typename T> __device__ inline float a16_load(const T* p);
template <> __device__ inline float a16_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float a16_load<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T> __device__ inline void a16_store(T* p, float v);
template <> __device__ inline void a16_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void a16_store<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

template <typename T>
__global__ void linear_batched_int8w_a16_kernel(const T*      __restrict__ X,
                                                const int8_t* __restrict__ W,
                                                const float*  __restrict__ scales,
                                                const T*      __restrict__ bias,
                                                T* __restrict__ Y,
                                                int B, int M, int K) {
    __shared__ float Xs[MM_TILE][MM_TILE];
    __shared__ float Ws[MM_TILE][MM_TILE];

    const int b = blockIdx.y * MM_TILE + threadIdx.y;
    const int m = blockIdx.x * MM_TILE + threadIdx.x;
    const float m_scale = (m < M) ? scales[m] : 0.0f;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int x_k = t * MM_TILE + threadIdx.x;
        const int w_k = t * MM_TILE + threadIdx.y;

        Xs[threadIdx.y][threadIdx.x] =
            (b < B && x_k < K) ? a16_load<T>(&X[b * K + x_k]) : 0.0f;
        Ws[threadIdx.y][threadIdx.x] =
            (m < M && w_k < K) ? (static_cast<float>(W[m * K + w_k]) * m_scale)
                               : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += Xs[threadIdx.y][k] * Ws[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (b < B && m < M) {
        if (bias) acc += a16_load<T>(&bias[m]);
        a16_store<T>(&Y[b * M + m], acc);
    }
}

__global__ void matmul_int8w_fp16_kernel(const int8_t* __restrict__ W,
                                         const float*  __restrict__ scales,
                                         const __half* __restrict__ X,
                                         __half* __restrict__ Y,
                                         int M, int N, int K) {
    __shared__ float Ws[MM_TILE][MM_TILE];
    __shared__ float Xs[MM_TILE][MM_TILE];

    const int row = blockIdx.y * MM_TILE + threadIdx.y;
    const int col = blockIdx.x * MM_TILE + threadIdx.x;

    const float row_scale = (row < M) ? scales[row] : 0.0f;

    float acc = 0.0f;
    const int n_tiles = (K + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int w_col = t * MM_TILE + threadIdx.x;
        const int x_row = t * MM_TILE + threadIdx.y;

        if (row < M && w_col < K) {
            Ws[threadIdx.y][threadIdx.x] =
                static_cast<float>(W[row * K + w_col]) * row_scale;
        } else {
            Ws[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (x_row < K && col < N) {
            Xs[threadIdx.y][threadIdx.x] = __half2float(X[x_row * N + col]);
        } else {
            Xs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += Ws[threadIdx.y][k] * Xs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc);
    }
}

__global__ void conv2d_int8w_fp16_forward_kernel(
        const __half* __restrict__ X,
        const int8_t* __restrict__ W,
        const float*  __restrict__ scales,
        const __half* __restrict__ bias,
        __half* __restrict__ Y,
        int N, int C_in, int H, int W_in_,
        int C_out, int kH, int kW,
        int H_out, int W_out,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dil_h, int dil_w,
        int /*groups*/, int Cg_in, int Cg_out,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int oc = t % C_out;
    const int n  = t / C_out;

    const int in_h_origin = oh * stride_h - pad_h;
    const int in_w_origin = ow * stride_w - pad_w;

    const float scale = scales[oc];

    const int g_out = oc / Cg_out;
    const int ic_abs_base = g_out * Cg_in;
    const int w_oc_base = oc * Cg_in * kH * kW;
    const int x_n_base = n * C_in * H * W_in_;

    float acc = 0.0f;
    for (int ic_local = 0; ic_local < Cg_in; ++ic_local) {
        const int ic = ic_abs_base + ic_local;
        const int w_ic_base = w_oc_base + ic_local * kH * kW;
        const int x_ic_base = x_n_base + ic * H * W_in_;
        for (int kh = 0; kh < kH; ++kh) {
            const int in_h = in_h_origin + kh * dil_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int in_w = in_w_origin + kw * dil_w;
                if (in_w < 0 || in_w >= W_in_) continue;
                const float xv = __half2float(X[x_ic_base + in_h * W_in_ + in_w]);
                const float wv = static_cast<float>(W[w_ic_base + kh * kW + kw]) * scale;
                acc += xv * wv;
            }
        }
    }
    if (bias) acc += __half2float(bias[oc]);
    Y[idx] = __float2half(acc);
}

} // namespace

void matmul_int8w_fp16(const Tensor& W_int8,
                       const Tensor& scales,
                       const Tensor& X,
                       Tensor& Y) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("matmul_int8w_fp16: W_int8 must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("matmul_int8w_fp16: scales must be FP32");
    }
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("matmul_int8w_fp16: X must be FP16");
    }
    const int M = W_int8.rows;
    const int K = W_int8.cols;
    if (X.rows != K) {
        throw std::runtime_error("matmul_int8w_fp16: K mismatch (W.cols != X.rows)");
    }
    if (scales.rows != M || scales.cols != 1) {
        throw std::runtime_error("matmul_int8w_fp16: scales shape must be (out, 1)");
    }
    const int Nb = X.cols;
    if (Y.rows != M || Y.cols != Nb || Y.dtype != Dtype::FP16) {
        Y.resize(M, Nb, Dtype::FP16);
    }
    if (M == 0 || Nb == 0) return;
    hipStream_t stream = cur_stream();
    if (K == 0) {
        BROTENSOR_HIP_CHECK(hipMemsetAsync(Y.data, 0, Y.bytes(), stream));
        return;
    }

    dim3 block(MM_TILE, MM_TILE);
    dim3 grid((Nb + MM_TILE - 1) / MM_TILE, (M + MM_TILE - 1) / MM_TILE);
    matmul_int8w_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const int8_t*>(W_int8.data),
        static_cast<const float*>(scales.data),
        static_cast<const __half*>(X.data),
        static_cast<__half*>(Y.data),
        M, Nb, K);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void linear_forward_batched_int8w_fp16(const Tensor& W_int8,
                                       const Tensor& scales,
                                       const Tensor* bias,
                                       const Tensor& X_BD,
                                       Tensor& Y_BD) {
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: scales must be FP32");
    }
    const Dtype dt = X_BD.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: X must be FP16 or BF16");
    }
    if (bias && bias->dtype != dt) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: bias dtype must match X");
    }
    const int B   = X_BD.rows;
    const int in_dim  = X_BD.cols;
    const int out_dim = W_int8.rows;
    if (W_int8.cols != in_dim) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (scales.rows != out_dim || scales.cols != 1) {
        throw std::runtime_error("linear_forward_batched_int8w_fp16: scales shape must be (out, 1)");
    }
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != dt) {
        Y_BD.resize(B, out_dim, dt);
    }
    if (B == 0 || out_dim == 0) return;
    hipStream_t stream = cur_stream();
    if (in_dim == 0) {
        BROTENSOR_HIP_CHECK(hipMemsetAsync(Y_BD.data, 0, Y_BD.bytes(), stream));
        return;
    }
    const void* b_p = (bias && bias->size() > 0) ? bias->data : nullptr;
    const int8_t* w_p = static_cast<const int8_t*>(W_int8.data);
    const float*  s_p = static_cast<const float*>(scales.data);

    dim3 block(MM_TILE, MM_TILE);
    dim3 grid((out_dim + MM_TILE - 1) / MM_TILE, (B + MM_TILE - 1) / MM_TILE);
    if (dt == Dtype::FP16) {
        linear_batched_int8w_a16_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(X_BD.data), w_p, s_p,
            static_cast<const __half*>(b_p),
            static_cast<__half*>(Y_BD.data),
            B, out_dim, in_dim);
    } else {  // BF16
        linear_batched_int8w_a16_kernel<hip_bfloat16><<<grid, block, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X_BD.data), w_p, s_p,
            static_cast<const hip_bfloat16*>(b_p),
            static_cast<hip_bfloat16*>(Y_BD.data),
            B, out_dim, in_dim);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void conv2d_int8w_fp16_forward(const Tensor& X,
                               const Tensor& W_int8,
                               const Tensor& scales,
                               const Tensor* bias,
                               int N, int C_in, int H, int W,
                               int C_out, int kH, int kW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dil_h, int dil_w, int groups,
                               Tensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: X must be FP16");
    }
    if (W_int8.dtype != Dtype::INT8) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: W must be INT8");
    }
    if (scales.dtype != Dtype::FP32) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: scales must be FP32");
    }
    if (bias && bias->dtype != Dtype::FP16) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: bias must be FP16");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error(
            "conv2d_int8w_fp16_forward: groups must be >=1 and divide both C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    if (W_int8.rows != C_out || W_int8.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: W shape mismatch");
    }
    if (scales.rows != C_out || scales.cols != 1) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: scales shape mismatch");
    }
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("conv2d_int8w_fp16_forward: non-positive output shape");
    }
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, out_cols, Dtype::FP16);
    }
    const int total = N * out_cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    const __half* b_p = bias ? static_cast<const __half*>(bias->data) : nullptr;

    constexpr int CONV_BLOCK = 256;
    const int blocks = (total + CONV_BLOCK - 1) / CONV_BLOCK;
    conv2d_int8w_fp16_forward_kernel<<<blocks, CONV_BLOCK, 0, stream>>>(
        static_cast<const __half*>(X.data),
        static_cast<const int8_t*>(W_int8.data),
        static_cast<const float*>(scales.data),
        b_p,
        static_cast<__half*>(Y.data),
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        groups, Cg_in, Cg_out, total);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_quant(OpsVTable& v) {
    v.matmul_int8w_fp16                 = &matmul_int8w_fp16;
    v.linear_forward_batched_int8w_fp16 = &linear_forward_batched_int8w_fp16;
    v.conv2d_int8w_fp16_forward         = &conv2d_int8w_fp16_forward;

    v.dequant_q4k_to_fp16               = &dequant_q4k_to_fp16;
    v.linear_forward_q4k_fp16           = &linear_forward_q4k_fp16;
    v.linear_forward_batched_q4k_fp16   = &linear_forward_batched_q4k_fp16;

    v.dequant_q8_0_to_fp16              = &dequant_q8_0_to_fp16;
    v.linear_forward_q8_0_fp16          = &linear_forward_q8_0_fp16;
    v.linear_forward_batched_q8_0_fp16  = &linear_forward_batched_q8_0_fp16;

    v.dequant_q6k_to_fp16               = &dequant_q6k_to_fp16;
    v.linear_forward_q6k_fp16           = &linear_forward_q6k_fp16;
    v.linear_forward_batched_q6k_fp16   = &linear_forward_batched_q6k_fp16;
}

} // namespace brotensor::detail::hip
