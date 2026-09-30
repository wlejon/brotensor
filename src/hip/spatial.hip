// Native AMD HIP 2D spatial operations.
//
// Implements:
//   - pad2d_forward / pad2d_backward
//   - slice2d_forward / slice2d_backward
//   - unfold2d_forward
//   - pixel_shuffle_upsample_2x_forward
//   - fill_hip_vtable_spatial
//
// Optimized for Wave32 wavefronts on RDNA 3.5.

#include "detail/spatial_internal.h"
#include "detail/hip_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int SP_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int sp_grid(long long n) {
    long long blocks = (n + SP_BLOCK - 1) / SP_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp(const ::brotensor::Tensor& t,
                     const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        fail(op, std::string(name) + " must be FP32/FP16/BF16");
    }
}

template <typename T> __device__ inline float sp_load(const T* p);
template <> __device__ inline float sp_load<float>(const float* p) { return *p; }
template <> __device__ inline float sp_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float sp_load<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T> __device__ inline void sp_store(T* p, float v);
template <> __device__ inline void sp_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void sp_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void sp_store<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

// ── pad2d ───────────────────────────────────────────────────────────────────
__device__ inline int pad_src_d(int p, int L, int pad_left, int mode) {
    const int rel = p - pad_left;
    if (rel >= 0 && rel < L) return rel;
    if (mode == 0) return -1;
    if (mode == 2) return rel < 0 ? 0 : L - 1;
    if (L == 1) return 0;
    int q = rel % (2 * (L - 1));
    if (q < 0) q += 2 * (L - 1);
    return q < L ? q : 2 * (L - 1) - q;
}

template <typename T>
__global__ void pad2d_forward_kernel(const T* __restrict__ X,
                                     T* __restrict__ Y,
                                     int N, int C, int H, int W,
                                     int pad_top, int pad_left, int mode,
                                     int H_pad, int W_pad) {
    const long long total = (long long)N * C * H_pad * W_pad;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_pad);
        long long t = idx / W_pad;
        const int oh = static_cast<int>(t % H_pad);
        t /= H_pad;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const int src_h = pad_src_d(oh, H, pad_top, mode);
        const int src_w = pad_src_d(ow, W, pad_left, mode);
        float v = 0.0f;
        if (src_h >= 0 && src_w >= 0) {
            const long long xbase = ((long long)n * C + c) * H * W;
            v = sp_load<T>(&X[xbase + (long long)src_h * W + src_w]);
        }
        sp_store<T>(&Y[idx], v);
    }
}

template <typename T>
__global__ void pad2d_backward_kernel(const T* __restrict__ dY,
                                      float* __restrict__ dX_fp32,
                                      int N, int C, int H, int W,
                                      int pad_top, int pad_left, int mode,
                                      int H_pad, int W_pad) {
    const long long total = (long long)N * C * H_pad * W_pad;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_pad);
        long long t = idx / W_pad;
        const int oh = static_cast<int>(t % H_pad);
        t /= H_pad;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const int src_h = pad_src_d(oh, H, pad_top, mode);
        const int src_w = pad_src_d(ow, W, pad_left, mode);
        if (src_h >= 0 && src_w >= 0) {
            const long long xbase = ((long long)n * C + c) * H * W;
            atomicAdd(&dX_fp32[xbase + (long long)src_h * W + src_w],
                      sp_load<T>(&dY[idx]));
        }
    }
}

template <typename T>
__global__ void copy_fp32_to_t(const float* __restrict__ src,
                               T* __restrict__ dst, long long total) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        sp_store<T>(&dst[idx], src[idx]);
    }
}

// ── slice2d ─────────────────────────────────────────────────────────────────
template <typename T>
__global__ void slice2d_forward_kernel(const T* __restrict__ X,
                                       T* __restrict__ Y,
                                       int N, int C, int H, int W,
                                       int h0, int w0,
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
        Y[idx] = X[xbase + (long long)(h0 + oh) * W + (w0 + ow)];
    }
}

template <typename T>
__global__ void slice2d_backward_kernel(const T* __restrict__ dY,
                                        T* __restrict__ dX,
                                        int N, int C, int H, int W,
                                        int h0, int w0,
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
        dX[xbase + (long long)(h0 + oh) * W + (w0 + ow)] = dY[idx];
    }
}

// ── unfold2d ────────────────────────────────────────────────────────────────
__device__ inline int unf_src_d(int coord, int L, int mode) {
    if (coord >= 0 && coord < L) return coord;
    if (mode == 0) return -1;
    if (mode == 2) return coord < 0 ? 0 : L - 1;
    if (L == 1) return 0;
    int q = coord % (2 * (L - 1));
    if (q < 0) q += 2 * (L - 1);
    return q < L ? q : 2 * (L - 1) - q;
}

template <typename T>
__global__ void unfold2d_forward_kernel(const T* __restrict__ X,
                                        T* __restrict__ Y,
                                        int N, int C, int H, int W,
                                        int kH, int kW,
                                        int stride_h, int stride_w,
                                        int pad_top, int pad_left, int mode,
                                        int H_out, int W_out) {
    const int kK = kH * kW;
    const long long total = (long long)N * C * kK * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ox = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oy = static_cast<int>(t % H_out);
        t /= H_out;
        const int k = static_cast<int>(t % kK);
        t /= kK;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);

        const int ky = k / kW;
        const int kx = k % kW;
        const int sy = unf_src_d(oy * stride_h - pad_top + ky, H, mode);
        const int sx = unf_src_d(ox * stride_w - pad_left + kx, W, mode);
        float v = 0.0f;
        if (sy >= 0 && sx >= 0) {
            const long long xbase = ((long long)n * C + c) * H * W;
            v = sp_load<T>(&X[xbase + (long long)sy * W + sx]);
        }
        sp_store<T>(&Y[idx], v);
    }
}

// ── pixel_shuffle_upsample_2x ───────────────────────────────────────────────
template <typename T>
__global__ void pixel_shuffle_upsample_2x_kernel(const T* __restrict__ X,
                                                 T* __restrict__ Y,
                                                 int N, int C_in, int H, int W,
                                                 int C_out, int repeats,
                                                 int H_out, int W_out,
                                                 long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int w_out = static_cast<int>(idx % W_out);
    long long t     = idx / W_out;
    const int h_out = static_cast<int>(t % H_out);
    t              /= H_out;
    const int c_out = static_cast<int>(t % C_out);
    const int n     = static_cast<int>(t / C_out);
    const int i = h_out & 1;
    const int j = w_out & 1;
    const int h = h_out >> 1;
    const int w = w_out >> 1;
    const int src_c = (4 * c_out + 2 * i + j) / repeats;
    const long long x_idx = ((long long)n * C_in + src_c) * (H * W) + h * W + w;
    Y[idx] = X[x_idx];
}

} // namespace

void pad2d_forward(const ::brotensor::Tensor& X, int N, int C, int H, int W,
                   int pad_top, int pad_bottom, int pad_left, int pad_right,
                   int mode, ::brotensor::Tensor& Y) {
    const char* op = "pad2d_forward";
    check_fp(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1) {
        if (pad_top >= H || pad_bottom >= H) fail(op, "reflect padding requires pad_top and pad_bottom < H");
        if (pad_left >= W || pad_right >= W) fail(op, "reflect padding requires pad_left and pad_right < W");
    }
    if (X.rows != N || X.cols != C * H * W) fail(op, "X shape must be (N, C*H*W)");

    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    const int cols_pad = C * H_pad * W_pad;
    if (Y.rows != N || Y.cols != cols_pad || Y.dtype != X.dtype) {
        Y.resize(N, cols_pad, X.dtype);
    }
    if (N == 0 || cols_pad == 0) return;

    const long long total = (long long)N * cols_pad;
    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        pad2d_forward_kernel<__half><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        pad2d_forward_kernel<hip_bfloat16><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    } else {
        pad2d_forward_kernel<float><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void pad2d_backward(const ::brotensor::Tensor& dY, int N, int C, int H, int W,
                    int pad_top, int pad_bottom, int pad_left, int pad_right,
                    int mode, ::brotensor::Tensor& dX) {
    const char* op = "pad2d_backward";
    check_fp(dY, op, "dY");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1) {
        if (pad_top >= H || pad_bottom >= H) fail(op, "reflect padding requires pad_top and pad_bottom < H");
        if (pad_left >= W || pad_right >= W) fail(op, "reflect padding requires pad_left and pad_right < W");
    }

    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    const int cols_pad = C * H_pad * W_pad;
    if (dY.rows != N || dY.cols != cols_pad) fail(op, "dY shape mismatch");

    const int in_cols = C * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != dY.dtype) {
        dX.resize(N, in_cols, dY.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    hipStream_t stream = cur_stream();
    const long long total_in = (long long)N * in_cols;
    const long long total_out = (long long)N * cols_pad;

    if (dY.dtype == ::brotensor::Dtype::FP32) {
        BROTENSOR_HIP_CHECK(hipMemsetAsync(dX.data, 0, total_in * sizeof(float), stream));
        pad2d_backward_kernel<float><<<sp_grid(total_out), SP_BLOCK, 0, stream>>>(
            static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    } else {
        float* d_fp32 = nullptr;
        BROTENSOR_HIP_CHECK(hipMallocAsync(&d_fp32, total_in * sizeof(float), stream));
        BROTENSOR_HIP_CHECK(hipMemsetAsync(d_fp32, 0, total_in * sizeof(float), stream));
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            pad2d_backward_kernel<__half><<<sp_grid(total_out), SP_BLOCK, 0, stream>>>(
                static_cast<const __half*>(dY.data), d_fp32,
                N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
            copy_fp32_to_t<__half><<<sp_grid(total_in), SP_BLOCK, 0, stream>>>(
                d_fp32, static_cast<__half*>(dX.data), total_in);
        } else {
            pad2d_backward_kernel<hip_bfloat16><<<sp_grid(total_out), SP_BLOCK, 0, stream>>>(
                static_cast<const hip_bfloat16*>(dY.data), d_fp32,
                N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
            copy_fp32_to_t<hip_bfloat16><<<sp_grid(total_in), SP_BLOCK, 0, stream>>>(
                d_fp32, static_cast<hip_bfloat16*>(dX.data), total_in);
        }
        BROTENSOR_HIP_CHECK(hipFreeAsync(d_fp32, stream));
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void slice2d_forward(const ::brotensor::Tensor& X,
                     int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out,
                     ::brotensor::Tensor& Y) {
    const char* op = "slice2d_forward";
    check_fp(X, op, "X");
    if (N < 0 || C < 1 || H < 0 || W < 0) fail(op, "C must be >=1; N, H, W must be >=0");
    if (H_out < 0 || W_out < 0) fail(op, "H_out and W_out must be >=0");
    if (h0 < 0 || w0 < 0) fail(op, "h0 and w0 must be >=0");
    if (h0 + H_out > H || w0 + W_out > W) fail(op, "slice bounds exceed input size");
    if (X.rows != N || X.cols != C * H * W) fail(op, "X shape must be (N, C*H*W)");

    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype) {
        Y.resize(N, cols_out, X.dtype);
    }
    if (N == 0 || cols_out == 0) return;

    const long long total = (long long)N * cols_out;
    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        slice2d_forward_kernel<__half><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, h0, w0, H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        slice2d_forward_kernel<hip_bfloat16><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            N, C, H, W, h0, w0, H_out, W_out);
    } else {
        slice2d_forward_kernel<float><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, h0, w0, H_out, W_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void slice2d_backward(const ::brotensor::Tensor& dY,
                      int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out,
                      ::brotensor::Tensor& dX) {
    const char* op = "slice2d_backward";
    check_fp(dY, op, "dY");
    if (N < 0 || C < 1 || H < 0 || W < 0) fail(op, "C must be >=1; N, H, W must be >=0");
    if (H_out < 0 || W_out < 0) fail(op, "H_out and W_out must be >=0");
    if (h0 < 0 || w0 < 0) fail(op, "h0 and w0 must be >=0");
    if (h0 + H_out > H || w0 + W_out > W) fail(op, "slice bounds exceed input size");

    const int cols_out = C * H_out * W_out;
    if (dY.rows != N || dY.cols != cols_out) fail(op, "dY shape mismatch");

    const int in_cols = C * H * W;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != dY.dtype) {
        dX.resize(N, in_cols, dY.dtype);
    }
    if (N == 0 || in_cols == 0) return;

    hipStream_t stream = cur_stream();
    const size_t elem_size = ::brotensor::dtype_size_bytes(dY.dtype);
    BROTENSOR_HIP_CHECK(hipMemsetAsync(dX.data, 0, (size_t)N * in_cols * elem_size, stream));

    if (cols_out > 0) {
        const long long total = (long long)N * cols_out;
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            slice2d_backward_kernel<__half><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
                static_cast<const __half*>(dY.data), static_cast<__half*>(dX.data),
                N, C, H, W, h0, w0, H_out, W_out);
        } else if (dY.dtype == ::brotensor::Dtype::BF16) {
            slice2d_backward_kernel<hip_bfloat16><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
                static_cast<const hip_bfloat16*>(dY.data), static_cast<hip_bfloat16*>(dX.data),
                N, C, H, W, h0, w0, H_out, W_out);
        } else {
            slice2d_backward_kernel<float><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
                static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
                N, C, H, W, h0, w0, H_out, W_out);
        }
        BROTENSOR_HIP_CHECK(hipGetLastError());
    }
}

void unfold2d_forward(const ::brotensor::Tensor& X,
                      int N, int C, int H, int W,
                      int kH, int kW,
                      int stride_h, int stride_w,
                      int pad_top, int pad_bottom,
                      int pad_left, int pad_right,
                      int mode,
                      ::brotensor::Tensor& Y) {
    const char* op = "unfold2d_forward";
    check_fp(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH/kW must be >=1");
    if (stride_h < 1 || stride_w < 1) fail(op, "stride must be >=1");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2) fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (X.rows != N || X.cols != C * H * W) fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    const int W_out = (W + pad_left + pad_right - kW) / stride_w + 1;
    if (H_out < 1 || W_out < 1) fail(op, "kernel/padding/stride yield empty output");
    const int kK = kH * kW;
    const int cols_out = C * kK * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype) {
        Y.resize(N, cols_out, X.dtype);
    }
    if (N == 0 || cols_out == 0) return;

    const long long total = (long long)N * cols_out;
    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        unfold2d_forward_kernel<__half><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_left, mode, H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        unfold2d_forward_kernel<hip_bfloat16><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_left, mode, H_out, W_out);
    } else {
        unfold2d_forward_kernel<float><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_left, mode, H_out, W_out);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void pixel_shuffle_upsample_2x_forward(const ::brotensor::Tensor& X,
                                       int N, int C_in, int H, int W,
                                       int C_out,
                                       ::brotensor::Tensor& Y) {
    if (X.dtype != ::brotensor::Dtype::FP32 &&
        X.dtype != ::brotensor::Dtype::FP16 &&
        X.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: X must be FP32, FP16, or BF16");
    }
    if (N < 0 || C_in <= 0 || H < 0 || W < 0 || C_out <= 0) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: bad dimension");
    }
    if ((4 * C_out) % C_in != 0) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: C_in must divide 4*C_out");
    }
    const int repeats = (4 * C_out) / C_in;
    const int H_out = 2 * H;
    const int W_out = 2 * W;
    const int cols  = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const long long total = (long long)N * cols;
    if (total == 0) return;

    hipStream_t stream = cur_stream();
    if (X.dtype == ::brotensor::Dtype::FP16) {
        pixel_shuffle_upsample_2x_kernel<__half><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, repeats, H_out, W_out, total);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        pixel_shuffle_upsample_2x_kernel<hip_bfloat16><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            N, C_in, H, W, C_out, repeats, H_out, W_out, total);
    } else {
        pixel_shuffle_upsample_2x_kernel<float><<<sp_grid(total), SP_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C_in, H, W, C_out, repeats, H_out, W_out, total);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_spatial(::brotensor::detail::OpsVTable& v) {
    v.pad2d_forward                     = &pad2d_forward;
    v.pad2d_backward                    = &pad2d_backward;
    v.slice2d_forward                   = &slice2d_forward;
    v.slice2d_backward                  = &slice2d_backward;
    v.unfold2d_forward                  = &unfold2d_forward;
    v.window_partition_forward          = &window_partition_forward;
    v.window_reverse_forward            = &window_reverse_forward;
    v.convex_upsample_forward           = &convex_upsample_forward;
    v.pixel_shuffle_upsample_2x_forward = &pixel_shuffle_upsample_2x_forward;
}

} // namespace brotensor::detail::hip
