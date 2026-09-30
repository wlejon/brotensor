// Native AMD HIP 2D pooling (adaptive_avg_pool2d + max_pool2d).
//
// Implements:
//   - adaptive_avg_pool2d_forward (FP32 / FP16 / BF16)
//   - adaptive_avg_pool2d_backward (FP32)
//   - max_pool2d_forward (FP32 / FP16 / BF16)
//   - max_pool2d_backward (FP32)
//   - fill_hip_vtable_pool
//
// Optimized for Wave32 wavefronts on RDNA 3.5.

#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>
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

constexpr int PL_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int pl_grid(long long n) {
    long long blocks = (n + PL_BLOCK - 1) / PL_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32");
    }
}

template <typename T>
__device__ inline float pl_load_f32(const T* p);
template <> __device__ inline float pl_load_f32<float>(const float* p)   { return *p; }
template <> __device__ inline float pl_load_f32<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float pl_load_f32<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T>
__device__ inline void pl_store_f32(T* p, float v);
template <> __device__ inline void pl_store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void pl_store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void pl_store_f32<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

template <typename T> struct pl_acc            { using type = float; };
template <>           struct pl_acc<float>     { using type = double; };

__device__ inline void adaptive_window(int o, int L, int L_out,
                                       int& start, int& end) {
    start = (o * L) / L_out;
    end   = ((o + 1) * L + L_out - 1) / L_out;
    if (end > L) end = L;
    if (start < 0) start = 0;
}

// ── adaptive_avg_pool2d forward kernel ──────────────────────────────────────
template <typename T>
__global__ void adaptive_avg_pool2d_forward_kernel(const T* __restrict__ X,
                                                   T* __restrict__ Y,
                                                   int N, int C,
                                                   int H, int W,
                                                   int H_out, int W_out) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        int h0, h1, w0, w1;
        adaptive_window(oh, H, H_out, h0, h1);
        adaptive_window(ow, W, W_out, w0, w1);
        const int area = (h1 - h0) * (w1 - w0);
        const long long xbase = ((long long)n * C + c) * H * W;
        typename pl_acc<T>::type acc = 0;
        for (int h = h0; h < h1; ++h) {
            const T* row = X + xbase + (long long)h * W;
            for (int w = w0; w < w1; ++w) acc += pl_load_f32<T>(&row[w]);
        }
        pl_store_f32<T>(&Y[idx], static_cast<float>(acc / area));
    }
}

// ── adaptive_avg_pool2d backward kernel ─────────────────────────────────────
__global__ void adaptive_avg_pool2d_backward_kernel(const float* __restrict__ dY,
                                                    float* __restrict__ dX,
                                                    int N, int C,
                                                    int H, int W,
                                                    int H_out, int W_out) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        int h0, h1, w0, w1;
        adaptive_window(oh, H, H_out, h0, h1);
        adaptive_window(ow, W, W_out, w0, w1);
        const int area = (h1 - h0) * (w1 - w0);
        const float g = dY[idx] / static_cast<float>(area);
        const long long xbase = ((long long)n * C + c) * H * W;
        for (int h = h0; h < h1; ++h) {
            float* row = dX + xbase + (long long)h * W;
            for (int w = w0; w < w1; ++w) atomicAdd(&row[w], g);
        }
    }
}

// ── max_pool2d forward kernel ───────────────────────────────────────────────
template <typename T>
__global__ void max_pool2d_forward_kernel(const T* __restrict__ X,
                                          T* __restrict__ Y,
                                          int32_t* __restrict__ Idx,
                                          int N, int C, int H, int W,
                                          int kH, int kW,
                                          int stride_h, int stride_w,
                                          int pad_h, int pad_w,
                                          int H_out, int W_out) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * H * W;
        const int h_base = oh * stride_h - pad_h;
        const int w_base = ow * stride_w - pad_w;
        float best_v = -1e38f;
        int32_t best_i = -1;
        for (int kh = 0; kh < kH; ++kh) {
            const int ih = h_base + kh;
            if (ih < 0 || ih >= H) continue;
            const T* row = X + xbase + (long long)ih * W;
            for (int kw = 0; kw < kW; ++kw) {
                const int iw = w_base + kw;
                if (iw < 0 || iw >= W) continue;
                const float v = pl_load_f32<T>(&row[iw]);
                if (v > best_v) {
                    best_v = v;
                    best_i = ih * W + iw;
                }
            }
        }
        pl_store_f32<T>(&Y[idx], best_v);
        Idx[idx] = best_i;
    }
}

// ── max_pool2d backward kernel ──────────────────────────────────────────────
__global__ void max_pool2d_backward_kernel(const float* __restrict__ dY,
                                           const int32_t* __restrict__ Idx,
                                           float* __restrict__ dX,
                                           int N, int C, int H, int W,
                                           int H_out, int W_out) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        long long t = idx / W_out;
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const int32_t flat = Idx[idx];
        if (flat < 0) continue;
        const long long xbase = ((long long)n * C + c) * H * W;
        atomicAdd(&dX[xbase + flat], dY[idx]);
    }
}

} // namespace

void adaptive_avg_pool2d_forward(const ::brotensor::Tensor& X,
                                 int N, int C, int H, int W,
                                 int H_out, int W_out,
                                 ::brotensor::Tensor& Y) {
    const char* op = "adaptive_avg_pool2d_forward";
    if (X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::FP16 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        fail(op, "X must be FP32, FP16, or BF16");
    }
    if (N < 0 || C < 0 || H < 0 || W < 0 || H_out < 0 || W_out < 0) {
        fail(op, "dimensions must be non-negative");
    }
    if ((H_out > 0 && H == 0) || (W_out > 0 && W == 0)) {
        fail(op, "input spatial dims must be > 0 when output dims are > 0");
    }
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must match (N, C*H*W)");
    }
    const int out_cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || C == 0 || H_out == 0 || W_out == 0) return;

    const long long total = (long long)N * out_cols;
    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        adaptive_avg_pool2d_forward_kernel<__half><<<pl_grid(total), PL_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        adaptive_avg_pool2d_forward_kernel<hip_bfloat16><<<pl_grid(total), PL_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            N, C, H, W, H_out, W_out);
    } else {
        adaptive_avg_pool2d_forward_kernel<float><<<pl_grid(total), PL_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, H_out, W_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void adaptive_avg_pool2d_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out,
                                  ::brotensor::Tensor& dX) {
    const char* op = "adaptive_avg_pool2d_backward";
    check_fp32(dY, op, "dY");
    if (N < 0 || C < 0 || H < 0 || W < 0 || H_out < 0 || W_out < 0) {
        fail(op, "dimensions must be non-negative");
    }
    if ((H_out > 0 && H == 0) || (W_out > 0 && W == 0)) {
        fail(op, "input spatial dims must be > 0 when output dims are > 0");
    }
    if (dY.rows != N || dY.cols != C * H_out * W_out) {
        fail(op, "dY shape must match (N, C*H_out*W_out)");
    }
    const int in_cols = C * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0 || H == 0 || W == 0) return;

    hipStream_t stream = cur_stream();
    const long long total_in = (long long)N * in_cols;
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dX.data, 0, total_in * sizeof(float), stream));
    const long long total_out = (long long)N * C * H_out * W_out;
    if (total_out > 0) {
        adaptive_avg_pool2d_backward_kernel<<<pl_grid(total_out), PL_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
            N, C, H, W, H_out, W_out);
        BROTENSOR_HIP_CHECK(hipGetLastError());
    }
}

void max_pool2d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int H, int W,
                        int kH, int kW,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        ::brotensor::Tensor& Y,
                        ::brotensor::Tensor& Idx) {
    const char* op = "max_pool2d_forward";
    if (X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::FP16 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        fail(op, "X must be FP32, FP16, or BF16");
    }
    if (kH <= 0 || kW <= 0 || stride_h <= 0 || stride_w <= 0 ||
        pad_h < 0 || pad_w < 0) {
        fail(op, "kernel/stride must be >0 and padding >=0");
    }
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must match (N, C*H*W)");
    }
    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) fail(op, "non-positive output shape");

    const int out_cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (Idx.rows != N || Idx.cols != out_cols || Idx.dtype != ::brotensor::Dtype::INT32) {
        Idx.resize(N, out_cols, ::brotensor::Dtype::INT32);
    }
    if (N == 0 || C == 0 || H_out == 0 || W_out == 0) return;

    const long long total = (long long)N * out_cols;
    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        max_pool2d_forward_kernel<__half><<<pl_grid(total), PL_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            static_cast<int32_t*>(Idx.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        max_pool2d_forward_kernel<hip_bfloat16><<<pl_grid(total), PL_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            static_cast<int32_t*>(Idx.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
    } else {
        max_pool2d_forward_kernel<float><<<pl_grid(total), PL_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            static_cast<int32_t*>(Idx.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void max_pool2d_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& Idx,
                         int N, int C, int H, int W,
                         int H_out, int W_out,
                         ::brotensor::Tensor& dX) {
    const char* op = "max_pool2d_backward";
    check_fp32(dY, op, "dY");
    if (Idx.dtype != ::brotensor::Dtype::INT32) {
        fail(op, "Idx must be INT32");
    }
    if (dY.rows != N || dY.cols != C * H_out * W_out) {
        fail(op, "dY shape must match (N, C*H_out*W_out)");
    }
    if (Idx.rows != N || Idx.cols != C * H_out * W_out) {
        fail(op, "Idx shape must match (N, C*H_out*W_out)");
    }
    const int in_cols = C * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, in_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0 || H == 0 || W == 0) return;

    hipStream_t stream = cur_stream();
    const long long total_in = (long long)N * in_cols;
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dX.data, 0, total_in * sizeof(float), stream));
    const long long total_out = (long long)N * C * H_out * W_out;
    if (total_out > 0) {
        max_pool2d_backward_kernel<<<pl_grid(total_out), PL_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), static_cast<const int32_t*>(Idx.data),
            static_cast<float*>(dX.data), N, C, H, W, H_out, W_out);
        BROTENSOR_HIP_CHECK(hipGetLastError());
    }
}

void fill_hip_vtable_pool(::brotensor::detail::OpsVTable& v) {
    v.max_pool2d_forward          = &max_pool2d_forward;
    v.max_pool2d_backward         = &max_pool2d_backward;
    v.adaptive_avg_pool2d_forward = &adaptive_avg_pool2d_forward;
    v.adaptive_avg_pool2d_backward= &adaptive_avg_pool2d_backward;
}

} // namespace brotensor::detail::hip
