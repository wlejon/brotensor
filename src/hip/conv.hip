// Native AMD HIP 2D convolution suite for RDNA 3.5 (gfx1151, Wave32).
//
// Key performance features:
// - Fast 1x1 convs (forward & backward): mapped directly to hipBLAS GEMM
//   via hipblasGemmEx / hipblasGemmStridedBatchedEx / hipblasSgemm.
// - Fast im2col + hipBLAS GEMM for general 2D convs (3x3, 5x5, 7x7).
// - Dedicated direct Wave32 depthwise conv2d kernels (fwd, bwd input, bwd weight).
// - Direct fallback kernels for arbitrary grouped convolutions.
// - Backward bias with Wave32 tree reduction.

#include "detail/conv_internal.h"
#include "detail/hip_check.h"
#include "detail/hipblas_handle.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#define BROTENSOR_HIPBLAS_CHECK(expr)                                                 \
    do {                                                                               \
        hipblasStatus_t _bga_err = (expr);                                             \
        if (_bga_err != HIPBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("hipBLAS error: ") + #expr);          \
        }                                                                              \
    } while (0)

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int CONV_BLOCK = 256;
constexpr int kWave32 = 32;

inline hipStream_t cur_stream(int dev = -1) {
    return reinterpret_cast<hipStream_t>(hip_current_stream(dev));
}

template <typename T>
__device__ __forceinline__ float load_f32(const T* p);
template <> __device__ __forceinline__ float load_f32<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ __forceinline__ float load_f32<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T>
__device__ __forceinline__ void store_f32(T* p, float v);
template <> __device__ __forceinline__ void store_f32<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ __forceinline__ void store_f32<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

template <typename T>
__global__ void conv2d_add_bias_kernel(T* __restrict__ Y, const T* __restrict__ bias,
                                       int C_out, int hw_out, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int oc = (idx / hw_out) % C_out;
    float y_val = load_f32<T>(&Y[idx]);
    float b_val = load_f32<T>(&bias[oc]);
    store_f32<T>(&Y[idx], y_val + b_val);
}

template <typename T>
__global__ void conv2d_add_fp32_into_dst_kernel(const float* __restrict__ src, T* __restrict__ dst, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float cur = load_f32<T>(&dst[idx]);
    store_f32<T>(&dst[idx], cur + src[idx]);
}

// ─── Im2Col and Col2Im Gather ───────────────────────────────────────────────
template <typename T>
__global__ void im2col_kernel(const T* __restrict__ X, T* __restrict__ col,
                              int C_in, int H, int W,
                              int kH, int kW,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dil_h, int dil_w,
                              int H_out, int W_out,
                              long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int M_col = H_out * W_out;
    int sp = static_cast<int>(idx % M_col);
    int k  = static_cast<int>(idx / M_col);

    int ow = sp % W_out;
    int oh = sp / W_out;

    int kw = k % kW;
    int t = k / kW;
    int kh = t % kH;
    int ic = t / kH;

    int ih = oh * stride_h - pad_h + kh * dil_h;
    int iw = ow * stride_w - pad_w + kw * dil_w;

    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = load_f32<T>(&X[(ic * H + ih) * W + iw]);
    }
    store_f32<T>(&col[idx], val);
}

template <typename T>
__global__ void col2im_gather_kernel(const T* __restrict__ col, T* __restrict__ dX,
                                     int C_in, int H, int W,
                                     int kH, int kW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int dil_h, int dil_w,
                                     int H_out, int W_out,
                                     long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int iw = static_cast<int>(idx % W);
    long long t = idx / W;
    int ih = static_cast<int>(t % H);
    int ic = static_cast<int>(t / H);

    int M_col = H_out * W_out;
    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        int num_h = ih + pad_h - kh * dil_h;
        if (num_h < 0 || num_h % stride_h != 0) continue;
        int oh = num_h / stride_h;
        if (oh >= H_out) continue;

        for (int kw = 0; kw < kW; ++kw) {
            int num_w = iw + pad_w - kw * dil_w;
            if (num_w < 0 || num_w % stride_w != 0) continue;
            int ow = num_w / stride_w;
            if (ow >= W_out) continue;

            int k = (ic * kH + kh) * kW + kw;
            int sp = oh * W_out + ow;
            acc += load_f32<T>(&col[(long long)k * M_col + sp]);
        }
    }
    store_f32<T>(&dX[idx], acc);
}

// ─── Backward Bias Kernel ───────────────────────────────────────────────────
template <typename T>
__global__ void conv2d_backward_bias_kernel(const T* __restrict__ dY, float* __restrict__ dB_scratch,
                                            int N, int C_out, int H_out, int W_out) {
    int oc = blockIdx.x;
    if (oc >= C_out) return;
    int tid = threadIdx.x;
    int spatial = H_out * W_out;
    long long M = (long long)N * spatial;

    float acc = 0.0f;
    for (long long m = tid; m < M; m += blockDim.x) {
        int n  = static_cast<int>(m / spatial);
        int sp = static_cast<int>(m % spatial);
        long long idx = (((long long)n * C_out + oc) * H_out) * W_out + sp;
        acc += load_f32<T>(&dY[idx]);
    }

    __shared__ float s_acc[CONV_BLOCK];
    s_acc[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 16; stride >>= 1) {
        if (tid < stride) s_acc[tid] += s_acc[tid + stride];
        __syncthreads();
    }
    if (tid < 16) {
        float v = s_acc[tid] + s_acc[tid + 16];
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset, kWave32);
        }
        if (tid == 0) dB_scratch[oc] = v;
    }
}

} // namespace

void conv2d_forward(const ::brotensor::Tensor& X,
                    const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W,
                    int C_out, int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    int groups,
                    ::brotensor::Tensor& Y) {
    if (X.dtype != ::brotensor::Dtype::FP16 && X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error("conv2d_forward: X must be FP16, BF16 or FP32");
    }
    if (Wt.dtype != X.dtype) throw std::runtime_error("conv2d_forward: Wt dtype must match X");
    if (bias && bias->dtype != X.dtype) throw std::runtime_error("conv2d_forward: bias dtype must match X");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv2d_forward: groups must divide C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) throw std::runtime_error("conv2d_forward: non-positive output shape");
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    const long long total = (long long)N * out_cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream(X.device.index);
    hipblasHandle_t handle = hipblas_handle(X.device.index);

    // Fast Path 1: 1x1 convs
    if (groups == 1 && kH == 1 && kW == 1 &&
        pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1 &&
        dil_h == 1 && dil_w == 1) {
        const int M = C_out;
        const int K = C_in;
        const int N_sp = H * W;
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        if (X.dtype == ::brotensor::Dtype::FP16) {
            if (N == 1) {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    N_sp, M, K,
                    &alpha,
                    X.data, HIP_R_16F, N_sp,
                    Wt.data, HIP_R_16F, K,
                    &beta,
                    Y.data, HIP_R_16F, N_sp,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    N_sp, M, K,
                    &alpha,
                    X.data, HIP_R_16F, N_sp, (long long)K * N_sp,
                    Wt.data, HIP_R_16F, K, 0,
                    &beta,
                    Y.data, HIP_R_16F, N_sp, (long long)M * N_sp,
                    N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            }
        } else if (X.dtype == ::brotensor::Dtype::BF16) {
            if (N == 1) {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    N_sp, M, K,
                    &alpha,
                    X.data, HIP_R_16BF, N_sp,
                    Wt.data, HIP_R_16BF, K,
                    &beta,
                    Y.data, HIP_R_16BF, N_sp,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    N_sp, M, K,
                    &alpha,
                    X.data, HIP_R_16BF, N_sp, (long long)K * N_sp,
                    Wt.data, HIP_R_16BF, K, 0,
                    &beta,
                    Y.data, HIP_R_16BF, N_sp, (long long)M * N_sp,
                    N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            }
        } else {
            if (N == 1) {
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    N_sp, M, K,
                    &alpha,
                    static_cast<const float*>(X.data), N_sp,
                    static_cast<const float*>(Wt.data), K,
                    &beta,
                    static_cast<float*>(Y.data), N_sp));
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemmStridedBatched(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    N_sp, M, K,
                    &alpha,
                    static_cast<const float*>(X.data), N_sp, (long long)K * N_sp,
                    static_cast<const float*>(Wt.data), K, 0,
                    &beta,
                    static_cast<float*>(Y.data), N_sp, (long long)M * N_sp,
                    N));
            }
        }
        if (bias) {
            const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
            if (X.dtype == ::brotensor::Dtype::FP16) {
                conv2d_add_bias_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
                    static_cast<__half*>(Y.data), static_cast<const __half*>(bias->data),
                    C_out, N_sp, total);
            } else if (X.dtype == ::brotensor::Dtype::BF16) {
                conv2d_add_bias_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
                    static_cast<hip_bfloat16*>(Y.data), static_cast<const hip_bfloat16*>(bias->data),
                    C_out, N_sp, total);
            } else {
                conv2d_add_bias_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
                    static_cast<float*>(Y.data), static_cast<const float*>(bias->data),
                    C_out, N_sp, total);
            }
        }
        BROTENSOR_HIP_CHECK(hipGetLastError());
        return;
    }

    // Fast Path 2: Dedicated Wave32 Depthwise Conv2D
    if (groups == C_in && C_in == C_out) {
        depthwise_conv2d_forward(X, Wt, bias, N, C_in, H, W, kH, kW, H_out, W_out,
                                 stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total, Y);
        return;
    }

    // Fast Path 3: Fast im2col + hipBLAS GEMM for general single-group convs
    if (groups == 1) {
        const int K_col = C_in * kH * kW;
        const int M_col = H_out * W_out;
        const size_t elem_size = X.dtype == ::brotensor::Dtype::FP32 ? sizeof(float) : 2;
        const size_t needed = (size_t)K_col * M_col * elem_size;
        static thread_local void* s_col_buf = nullptr;
        static thread_local size_t s_col_cap = 0;
        if (needed > s_col_cap) {
            if (s_col_buf) hip_free(s_col_buf, X.device.index);
            s_col_cap = needed * 2;
            s_col_buf = hip_alloc(s_col_cap, X.device.index);
        }
        void* d_col = s_col_buf;

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        const int im2col_blocks = static_cast<int>(((long long)K_col * M_col + CONV_BLOCK - 1) / CONV_BLOCK);

        for (int n = 0; n < N; ++n) {
            const char* x_ptr = static_cast<const char*>(X.data) + (long long)n * C_in * H * W * elem_size;
            char* y_ptr       = static_cast<char*>(Y.data)       + (long long)n * C_out * M_col * elem_size;

            if (X.dtype == ::brotensor::Dtype::FP16) {
                im2col_kernel<__half><<<im2col_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<__half*>(d_col),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)K_col * M_col);
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    M_col, C_out, K_col,
                    &alpha,
                    d_col, HIP_R_16F, M_col,
                    Wt.data, HIP_R_16F, K_col,
                    &beta,
                    y_ptr, HIP_R_16F, M_col,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else if (X.dtype == ::brotensor::Dtype::BF16) {
                im2col_kernel<hip_bfloat16><<<im2col_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const hip_bfloat16*>(x_ptr), reinterpret_cast<hip_bfloat16*>(d_col),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)K_col * M_col);
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    M_col, C_out, K_col,
                    &alpha,
                    d_col, HIP_R_16BF, M_col,
                    Wt.data, HIP_R_16BF, K_col,
                    &beta,
                    y_ptr, HIP_R_16BF, M_col,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else {
                im2col_kernel<float><<<im2col_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(x_ptr), reinterpret_cast<float*>(d_col),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)K_col * M_col);
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                    M_col, C_out, K_col,
                    &alpha,
                    static_cast<const float*>(d_col), M_col,
                    static_cast<const float*>(Wt.data), K_col,
                    &beta,
                    reinterpret_cast<float*>(y_ptr), M_col));
            }
        }

        if (bias) {
            const int blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
            if (X.dtype == ::brotensor::Dtype::FP16) {
                conv2d_add_bias_kernel<__half><<<blocks, CONV_BLOCK, 0, stream>>>(
                    static_cast<__half*>(Y.data), static_cast<const __half*>(bias->data),
                    C_out, M_col, total);
            } else if (X.dtype == ::brotensor::Dtype::BF16) {
                conv2d_add_bias_kernel<hip_bfloat16><<<blocks, CONV_BLOCK, 0, stream>>>(
                    static_cast<hip_bfloat16*>(Y.data), static_cast<const hip_bfloat16*>(bias->data),
                    C_out, M_col, total);
            } else {
                conv2d_add_bias_kernel<float><<<blocks, CONV_BLOCK, 0, stream>>>(
                    static_cast<float*>(Y.data), static_cast<const float*>(bias->data),
                    C_out, M_col, total);
            }
        }
        BROTENSOR_HIP_CHECK(hipGetLastError());
        return;
    }

    // Fallback: Direct grouped convolution
    conv2d_direct_forward(X, Wt, bias, N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total, Y);
}

void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dil_h, int dil_w,
                           int groups,
                           ::brotensor::Tensor& dX) {
    if (Wt.dtype != ::brotensor::Dtype::FP16 && Wt.dtype != ::brotensor::Dtype::FP32 &&
        Wt.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_input: Wt must be FP16, BF16 or FP32");
    }
    if (dY.dtype != Wt.dtype) throw std::runtime_error("conv2d_backward_input: dY dtype must match Wt");
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv2d_backward_input: groups must divide C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) throw std::runtime_error("conv2d_backward_input: non-positive output shape");
    const int in_cols = C_in * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != Wt.dtype) {
        dX.resize(N, in_cols, Wt.dtype);
    }
    const long long total = (long long)N * in_cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream(Wt.device.index);
    hipblasHandle_t handle = hipblas_handle(Wt.device.index);

    // Fast Path 1: 1x1 convs backward input (resolves the 122s bottleneck!)
    if (groups == 1 && kH == 1 && kW == 1 &&
        pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1 &&
        dil_h == 1 && dil_w == 1) {
        const int M = C_in;
        const int K = C_out;
        const int N_sp = H * W;
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        if (Wt.dtype == ::brotensor::Dtype::FP16) {
            if (N == 1) {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    N_sp, M, K,
                    &alpha,
                    dY.data, HIP_R_16F, N_sp,
                    Wt.data, HIP_R_16F, M,
                    &beta,
                    dX.data, HIP_R_16F, N_sp,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    N_sp, M, K,
                    &alpha,
                    dY.data, HIP_R_16F, N_sp, (long long)K * N_sp,
                    Wt.data, HIP_R_16F, M, 0,
                    &beta,
                    dX.data, HIP_R_16F, N_sp, (long long)M * N_sp,
                    N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            }
        } else if (Wt.dtype == ::brotensor::Dtype::BF16) {
            if (N == 1) {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    N_sp, M, K,
                    &alpha,
                    dY.data, HIP_R_16BF, N_sp,
                    Wt.data, HIP_R_16BF, M,
                    &beta,
                    dX.data, HIP_R_16BF, N_sp,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    N_sp, M, K,
                    &alpha,
                    dY.data, HIP_R_16BF, N_sp, (long long)K * N_sp,
                    Wt.data, HIP_R_16BF, M, 0,
                    &beta,
                    dX.data, HIP_R_16BF, N_sp, (long long)M * N_sp,
                    N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            }
        } else {
            if (N == 1) {
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    N_sp, M, K,
                    &alpha,
                    static_cast<const float*>(dY.data), N_sp,
                    static_cast<const float*>(Wt.data), M,
                    &beta,
                    static_cast<float*>(dX.data), N_sp));
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemmStridedBatched(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    N_sp, M, K,
                    &alpha,
                    static_cast<const float*>(dY.data), N_sp, (long long)K * N_sp,
                    static_cast<const float*>(Wt.data), M, 0,
                    &beta,
                    static_cast<float*>(dX.data), N_sp, (long long)M * N_sp,
                    N));
            }
        }
        BROTENSOR_HIP_CHECK(hipGetLastError());
        return;
    }

    // Fast Path 2: Dedicated Wave32 Depthwise Conv2D Backward Input
    if (groups == C_in && C_in == C_out) {
        depthwise_conv2d_backward_input(Wt, dY, N, C_in, H, W, kH, kW, H_out, W_out,
                                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total, dX);
        return;
    }

    // Fast Path 3: Fast hipBLAS GEMM + col2im gather for general single-group convs
    if (groups == 1) {
        const int K_col = C_in * kH * kW;
        const int M_col = H_out * W_out;
        const size_t elem_size = Wt.dtype == ::brotensor::Dtype::FP32 ? sizeof(float) : 2;
        void* d_col = hip_alloc((size_t)K_col * M_col * elem_size, Wt.device.index);

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        const int gather_blocks = static_cast<int>(((long long)C_in * H * W + CONV_BLOCK - 1) / CONV_BLOCK);

        for (int n = 0; n < N; ++n) {
            const char* dy_ptr = static_cast<const char*>(dY.data) + (long long)n * C_out * M_col * elem_size;
            char* dx_ptr       = static_cast<char*>(dX.data)       + (long long)n * C_in * H * W * elem_size;

            if (Wt.dtype == ::brotensor::Dtype::FP16) {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    M_col, K_col, C_out,
                    &alpha,
                    dy_ptr, HIP_R_16F, M_col,
                    Wt.data, HIP_R_16F, K_col,
                    &beta,
                    d_col, HIP_R_16F, M_col,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
                col2im_gather_kernel<__half><<<gather_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __half*>(d_col), reinterpret_cast<__half*>(dx_ptr),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)C_in * H * W);
            } else if (Wt.dtype == ::brotensor::Dtype::BF16) {
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    M_col, K_col, C_out,
                    &alpha,
                    dy_ptr, HIP_R_16BF, M_col,
                    Wt.data, HIP_R_16BF, K_col,
                    &beta,
                    d_col, HIP_R_16BF, M_col,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
                col2im_gather_kernel<hip_bfloat16><<<gather_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const hip_bfloat16*>(d_col), reinterpret_cast<hip_bfloat16*>(dx_ptr),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)C_in * H * W);
            } else {
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
                    handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                    M_col, K_col, C_out,
                    &alpha,
                    reinterpret_cast<const float*>(dy_ptr), M_col,
                    static_cast<const float*>(Wt.data), K_col,
                    &beta,
                    static_cast<float*>(d_col), M_col));
                col2im_gather_kernel<float><<<gather_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(d_col), reinterpret_cast<float*>(dx_ptr),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)C_in * H * W);
            }
        }
        hip_free(d_col, Wt.device.index);
        BROTENSOR_HIP_CHECK(hipGetLastError());
        return;
    }

    // Fallback: Direct grouped convolution backward input
    conv2d_direct_backward_input(Wt, dY, N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                                 stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total, dX);
}

void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            int dil_h, int dil_w,
                            int groups,
                            ::brotensor::Tensor& dWt) {
    if (X.dtype != ::brotensor::Dtype::FP16 && X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_weight: X must be FP16, BF16 or FP32");
    }
    if (dY.dtype != X.dtype || dWt.dtype != X.dtype) {
        throw std::runtime_error("conv2d_backward_weight: dtypes must match");
    }
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::runtime_error("conv2d_backward_weight: groups must divide C_in and C_out");
    }
    const int Cg_in  = C_in  / groups;
    const int Cg_out = C_out / groups;
    const int H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) throw std::runtime_error("conv2d_backward_weight: non-positive output shape");
    const int total = C_out * Cg_in * kH * kW;
    if (dWt.rows != C_out || dWt.cols != Cg_in * kH * kW) {
        throw std::runtime_error("conv2d_backward_weight: dWt shape mismatch");
    }
    if (total == 0) return;

    hipStream_t stream = cur_stream(X.device.index);
    float* dWt_scratch = static_cast<float*>(hip_alloc(total * sizeof(float), X.device.index));
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dWt_scratch, 0, total * sizeof(float), stream));

    // Fast Path 1: Dedicated Wave32 Depthwise Conv2D Backward Weight
    if (groups == C_in && C_in == C_out) {
        depthwise_conv2d_backward_weight(X, dY, dWt_scratch, N, C_in, H, W, kH, kW, H_out, W_out,
                                         stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, total);
    } else if (groups == 1) {
        // Fast Path 2: im2col + hipBLAS GEMM for general conv2d
        hipblasHandle_t handle = hipblas_handle(X.device.index);
        const int K_col = C_in * kH * kW;
        const int M_col = H_out * W_out;
        const size_t elem_size = X.dtype == ::brotensor::Dtype::FP32 ? sizeof(float) : 2;
        void* d_col = hip_alloc((size_t)K_col * M_col * elem_size, X.device.index);

        const float alpha = 1.0f;
        const float beta  = 1.0f; // Accumulates across batch samples into dWt_scratch
        const int im2col_blocks = static_cast<int>(((long long)K_col * M_col + CONV_BLOCK - 1) / CONV_BLOCK);

        for (int n = 0; n < N; ++n) {
            const char* x_ptr  = static_cast<const char*>(X.data)  + (long long)n * C_in * H * W * elem_size;
            const char* dy_ptr = static_cast<const char*>(dY.data) + (long long)n * C_out * M_col * elem_size;

            if (X.dtype == ::brotensor::Dtype::FP16) {
                im2col_kernel<__half><<<im2col_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const __half*>(x_ptr), reinterpret_cast<__half*>(d_col),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)K_col * M_col);
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                    K_col, C_out, M_col,
                    &alpha,
                    d_col, HIP_R_16F, M_col,
                    dy_ptr, HIP_R_16F, M_col,
                    &beta,
                    dWt_scratch, HIP_R_32F, K_col,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else if (X.dtype == ::brotensor::Dtype::BF16) {
                im2col_kernel<hip_bfloat16><<<im2col_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const hip_bfloat16*>(x_ptr), reinterpret_cast<hip_bfloat16*>(d_col),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)K_col * M_col);
                BROTENSOR_HIPBLAS_CHECK(hipblasGemmEx(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                    K_col, C_out, M_col,
                    &alpha,
                    d_col, HIP_R_16BF, M_col,
                    dy_ptr, HIP_R_16BF, M_col,
                    &beta,
                    dWt_scratch, HIP_R_32F, K_col,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            } else {
                im2col_kernel<float><<<im2col_blocks, CONV_BLOCK, 0, stream>>>(
                    reinterpret_cast<const float*>(x_ptr), reinterpret_cast<float*>(d_col),
                    C_in, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                    H_out, W_out, (long long)K_col * M_col);
                BROTENSOR_HIPBLAS_CHECK(hipblasSgemm(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                    K_col, C_out, M_col,
                    &alpha,
                    static_cast<const float*>(d_col), M_col,
                    reinterpret_cast<const float*>(dy_ptr), M_col,
                    &beta,
                    dWt_scratch, K_col));
            }
        }
        hip_free(d_col, X.device.index);
    } else {
        // Fallback: Direct grouped conv2d backward weight
        conv2d_direct_backward_weight(X, dY, dWt_scratch, N, C_in, H, W, C_out, kH, kW, H_out, W_out,
                                      stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, Cg_in, Cg_out, total);
    }

    // Fold dWt_scratch (FP32) into destination dWt (adds to existing baseline)
    const int fold_blocks = static_cast<int>((total + CONV_BLOCK - 1) / CONV_BLOCK);
    if (dWt.dtype == ::brotensor::Dtype::FP16) {
        conv2d_add_fp32_into_dst_kernel<__half><<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            dWt_scratch, static_cast<__half*>(dWt.data), total);
    } else if (dWt.dtype == ::brotensor::Dtype::BF16) {
        conv2d_add_fp32_into_dst_kernel<hip_bfloat16><<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            dWt_scratch, static_cast<hip_bfloat16*>(dWt.data), total);
    } else {
        conv2d_add_fp32_into_dst_kernel<float><<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            dWt_scratch, static_cast<float*>(dWt.data), total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(dWt_scratch, X.device.index);
}

void conv2d_backward_bias(const ::brotensor::Tensor& dY,
                          int N, int C_out, int H_out, int W_out,
                          ::brotensor::Tensor& dB) {
    if (dY.dtype != ::brotensor::Dtype::FP16 && dY.dtype != ::brotensor::Dtype::FP32 &&
        dY.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error("conv2d_backward_bias: dY must be FP16, BF16 or FP32");
    }
    if (dB.dtype != dY.dtype) throw std::runtime_error("conv2d_backward_bias: dB dtype must match dY");
    if (dB.rows != C_out || dB.cols != 1) throw std::runtime_error("conv2d_backward_bias: dB shape mismatch");
    if (C_out == 0 || N == 0 || H_out == 0 || W_out == 0) return;

    hipStream_t stream = cur_stream(dY.device.index);
    float* dB_scratch = static_cast<float*>(hip_alloc(C_out * sizeof(float), dY.device.index));

    if (dY.dtype == ::brotensor::Dtype::FP16) {
        conv2d_backward_bias_kernel<__half><<<C_out, CONV_BLOCK, 0, stream>>>(
            static_cast<const __half*>(dY.data), dB_scratch, N, C_out, H_out, W_out);
    } else if (dY.dtype == ::brotensor::Dtype::BF16) {
        conv2d_backward_bias_kernel<hip_bfloat16><<<C_out, CONV_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(dY.data), dB_scratch, N, C_out, H_out, W_out);
    } else {
        conv2d_backward_bias_kernel<float><<<C_out, CONV_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), dB_scratch, N, C_out, H_out, W_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());

    const int fold_blocks = (C_out + CONV_BLOCK - 1) / CONV_BLOCK;
    if (dB.dtype == ::brotensor::Dtype::FP16) {
        conv2d_add_fp32_into_dst_kernel<__half><<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            dB_scratch, static_cast<__half*>(dB.data), C_out);
    } else if (dB.dtype == ::brotensor::Dtype::BF16) {
        conv2d_add_fp32_into_dst_kernel<hip_bfloat16><<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            dB_scratch, static_cast<hip_bfloat16*>(dB.data), C_out);
    } else {
        conv2d_add_fp32_into_dst_kernel<float><<<fold_blocks, CONV_BLOCK, 0, stream>>>(
            dB_scratch, static_cast<float*>(dB.data), C_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(dB_scratch, dY.device.index);
}

void fill_hip_vtable_conv(::brotensor::detail::OpsVTable& v) {
    // 2D conv
    v.conv2d_forward                   = &conv2d_forward;
    v.conv2d_backward_input            = &conv2d_backward_input;
    v.conv2d_backward_weight           = &conv2d_backward_weight;
    v.conv2d_backward_bias             = &conv2d_backward_bias;
    // 1D conv
    v.conv_transpose1d_forward         = &conv_transpose1d_forward;
    v.conv_transpose1d_backward_input  = &conv_transpose1d_backward_input;
    v.conv_transpose1d_backward_weight = &conv_transpose1d_backward_weight;
    v.conv_transpose1d_backward_bias   = &conv_transpose1d_backward_bias;
    v.causal_conv1d_update             = &causal_conv1d_update;
    v.pad1d_forward                    = &pad1d_forward;
    v.pad1d_backward                   = &pad1d_backward;
    // 2D transposed conv
    v.conv_transpose2d_forward         = &conv_transpose2d_forward;
    v.conv_transpose2d_backward_input  = &conv_transpose2d_backward_input;
    v.conv_transpose2d_backward_weight = &conv_transpose2d_backward_weight;
    v.conv_transpose2d_backward_bias   = &conv_transpose2d_backward_bias;
    // 3D conv
    v.conv3d_forward                   = &conv3d_forward;
    v.conv3d_int8w_fp16_forward        = &conv3d_int8w_fp16_forward;
    // Deformable conv
    v.deform_conv2d_forward            = &deform_conv2d_forward;
}

} // namespace brotensor::detail::hip
