// ─── CUDA 2D pooling: adaptive_avg_pool2d + max_pool2d ─────────────────────
//
// Port of src/cpu/pool2d.cpp. Contracts mirror CPU exactly:
//   * NCHW flat layout (rows = N, cols = C*H*W).
//   * max_pool2d Idx is INT32 per-channel flat-spatial (ih*W + iw); -1 if
//     the kernel window saw no valid (non-padding) pixel.
//   * Forward overwrites Y / Idx; backward overwrites dX (zero-then-scatter)
//     with atomicAdd because adaptive windows overlap (avg) and max-pool
//     stride < kernel can route multiple outputs to the same input pixel.
//
// The two forward ops are dtype-dispatched on X (FP32/FP16/BF16 — Y matches;
// the avg pool accumulates in FP32 for the 16-bit paths and keeps the
// original double accumulator for FP32, and the max-pool Idx tensor stays
// INT32 regardless). The backward ops remain FP32-only.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int PL_BLOCK = 256;

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
template <> __device__ inline float pl_load_f32<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__device__ inline void pl_store_f32(T* p, float v);
template <> __device__ inline void pl_store_f32<float>(float* p, float v)   { *p = v; }
template <> __device__ inline void pl_store_f32<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void pl_store_f32<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// Accumulator type for the avg pool: keep the original double accumulator on
// the FP32 path (bit-stable vs the existing CPU/CUDA behavior); FP32 is ample
// for 16-bit storage.
template <typename T> struct pl_acc            { using type = float; };
template <>           struct pl_acc<float>     { using type = double; };

__device__ inline void adaptive_window(int o, int L, int L_out,
                                       int& start, int& end) {
    start = (o * L) / L_out;
    end   = ((o + 1) * L + L_out - 1) / L_out;
    if (end > L) end = L;
    if (start < 0) start = 0;
}

// ── adaptive_avg_pool2d forward kernel ────────────────────────────────────
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

// ── adaptive_avg_pool2d backward kernel — scatter via atomicAdd ───────────
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

// ── max_pool2d forward kernel ─────────────────────────────────────────────
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
        float best_v = -CUDART_INF_F;
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

// ── max_pool2d backward kernel — scatter via atomicAdd ────────────────────
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

// ════════════════════════════════════════════════════════════════════════════
//  adaptive_avg_pool2d
// ════════════════════════════════════════════════════════════════════════════

void adaptive_avg_pool2d_forward(const ::brotensor::Tensor& X,
                                 int N, int C, int H, int W,
                                 int H_out, int W_out,
                                 ::brotensor::Tensor& Y) {
    const char* op = "adaptive_avg_pool2d_forward";
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        fail(op, "X must be FP32, FP16 or BF16");
    }
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 1 || W_out < 1)
        fail(op, "H_out and W_out must be >= 1");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype) {
        Y.resize(N, cols_out, X.dtype);
    }
    if (N == 0) return;

    const long long total = (long long)N * cols_out;
    if (X.dtype == Dtype::FP16) {
        adaptive_avg_pool2d_forward_kernel<__half><<<pl_grid(total), PL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, H_out, W_out);
    } else if (X.dtype == Dtype::BF16) {
        adaptive_avg_pool2d_forward_kernel<__nv_bfloat16><<<pl_grid(total), PL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, H_out, W_out);
    } else {
        adaptive_avg_pool2d_forward_kernel<float><<<pl_grid(total), PL_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void adaptive_avg_pool2d_backward(const ::brotensor::Tensor& dY,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out,
                                  ::brotensor::Tensor& dX) {
    const char* op = "adaptive_avg_pool2d_backward";
    check_fp32(dY, op, "dY");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 1 || W_out < 1)
        fail(op, "H_out and W_out must be >= 1");
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, cols_in, ::brotensor::Dtype::FP32);
    }
    if (N == 0) return;

    const long long total_in = (long long)N * cols_in;
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
        dX.data, 0, static_cast<size_t>(total_in) * sizeof(float), cur_stream()));

    const long long total_out = (long long)N * C * H_out * W_out;
    adaptive_avg_pool2d_backward_kernel<<<pl_grid(total_out), PL_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
        N, C, H, W, H_out, W_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  max_pool2d
// ════════════════════════════════════════════════════════════════════════════

void max_pool2d_forward(const ::brotensor::Tensor& X,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        ::brotensor::Tensor& Y, ::brotensor::Tensor& Idx) {
    const char* op = "max_pool2d_forward";
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        fail(op, "X must be FP32, FP16 or BF16");
    }
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH and kW must be >= 1");
    if (stride_h < 1 || stride_w < 1) fail(op, "strides must be >= 1");
    if (pad_h < 0 || pad_w < 0) fail(op, "pads must be >= 0");
    if (kH > H + 2 * pad_h || kW > W + 2 * pad_w)
        fail(op, "kernel larger than padded input");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype) {
        Y.resize(N, cols_out, X.dtype);
    }
    if (Idx.rows != N || Idx.cols != cols_out ||
        Idx.dtype != ::brotensor::Dtype::INT32) {
        Idx.resize(N, cols_out, ::brotensor::Dtype::INT32);
    }
    if (N == 0) return;

    const long long total = (long long)N * cols_out;
    if (X.dtype == Dtype::FP16) {
        max_pool2d_forward_kernel<__half><<<pl_grid(total), PL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            static_cast<int32_t*>(Idx.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w,
            H_out, W_out);
    } else if (X.dtype == Dtype::BF16) {
        max_pool2d_forward_kernel<__nv_bfloat16><<<pl_grid(total), PL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            static_cast<int32_t*>(Idx.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w,
            H_out, W_out);
    } else {
        max_pool2d_forward_kernel<float><<<pl_grid(total), PL_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            static_cast<int32_t*>(Idx.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w,
            H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void max_pool2d_backward(const ::brotensor::Tensor& dY,
                         const ::brotensor::Tensor& Idx,
                         int N, int C, int H, int W,
                         int H_out, int W_out,
                         ::brotensor::Tensor& dX) {
    const char* op = "max_pool2d_backward";
    check_fp32(dY, op, "dY");
    if (Idx.dtype != ::brotensor::Dtype::INT32)
        fail(op, "Idx must be INT32 (as produced by max_pool2d_forward)");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (H_out < 0 || W_out < 0)
        fail(op, "H_out and W_out must be >= 0");
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");
    if (Idx.rows != N || Idx.cols != C * H_out * W_out)
        fail(op, "Idx shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(N, cols_in, ::brotensor::Dtype::FP32);
    }
    if (N == 0) return;

    const long long total_in = (long long)N * cols_in;
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
        dX.data, 0, static_cast<size_t>(total_in) * sizeof(float), cur_stream()));

    if (H_out == 0 || W_out == 0) return;

    const long long total_out = (long long)N * C * H_out * W_out;
    max_pool2d_backward_kernel<<<pl_grid(total_out), PL_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dY.data),
        static_cast<const int32_t*>(Idx.data),
        static_cast<float*>(dX.data),
        N, C, H, W, H_out, W_out);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_pool2d(::brotensor::detail::OpsVTable& v) {
    v.adaptive_avg_pool2d_forward  = &adaptive_avg_pool2d_forward;
    v.adaptive_avg_pool2d_backward = &adaptive_avg_pool2d_backward;
    v.max_pool2d_forward           = &max_pool2d_forward;
    v.max_pool2d_backward          = &max_pool2d_backward;
}

} // namespace brotensor::detail::cuda
