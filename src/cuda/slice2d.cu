// ─── CUDA 2D spatial slice / crop: slice2d_forward + slice2d_backward ─────
//
// CUDA port of src/cpu/slice2d.cpp. Contracts:
//   * NCHW flat layout (rows = N, cols = C*H*W for X / dX; cols = C*H_out*W_out
//     for Y / dY).
//   * Extracts the (H_out, W_out) sub-region of X starting at (h0, w0).
//   * Forward overwrites Y. Backward overwrites dX (zero via cudaMemset, then
//     scatter dY into the slice region — no aliasing so direct stores).
//
// CPU is FP32-only; CUDA additionally supports FP16/BF16 — this is pure data
// movement, so we templatise on element type with raw assignment (no math).

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int SL_BLOCK = 256;

inline int sl_grid(long long n) {
    long long blocks = (n + SL_BLOCK - 1) / SL_BLOCK;
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

inline void check_args(const char* op,
                       int N, int C, int H, int W,
                       int h0, int w0, int H_out, int W_out) {
    if (N < 0 || C < 1 || H < 0 || W < 0)
        fail(op, "C must be >=1; N, H, W must be >=0");
    if (H_out < 0 || W_out < 0)
        fail(op, "H_out and W_out must be >=0");
    if (h0 < 0 || w0 < 0)
        fail(op, "h0 and w0 must be >=0");
    if (h0 + H_out > H)
        fail(op, "h0 + H_out must be <= H");
    if (w0 + W_out > W)
        fail(op, "w0 + W_out must be <= W");
}

// ── slice2d_forward kernel — one thread per output pixel ──────────────────
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

// ── slice2d_backward kernel — direct scatter (no overlap) ─────────────────
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

inline size_t bytes_of(::brotensor::Dtype d) {
    return ::brotensor::dtype_size_bytes(d);
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════

void slice2d_forward(const ::brotensor::Tensor& X,
                     int N, int C, int H, int W,
                     int h0, int w0, int H_out, int W_out,
                     ::brotensor::Tensor& Y) {
    const char* op = "slice2d_forward";
    check_fp(X, op, "X");
    check_args(op, N, C, H, W, h0, w0, H_out, W_out);
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int cols_out = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype) {
        Y.resize(N, cols_out, X.dtype);
    }
    if (N == 0 || cols_out == 0) return;

    const long long total = (long long)N * cols_out;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        slice2d_forward_kernel<__half><<<sl_grid(total), SL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, h0, w0, H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        slice2d_forward_kernel<__nv_bfloat16><<<sl_grid(total), SL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, h0, w0, H_out, W_out);
    } else {
        slice2d_forward_kernel<float><<<sl_grid(total), SL_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, h0, w0, H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void slice2d_backward(const ::brotensor::Tensor& dY,
                      int N, int C, int H, int W,
                      int h0, int w0, int H_out, int W_out,
                      ::brotensor::Tensor& dX) {
    const char* op = "slice2d_backward";
    check_fp(dY, op, "dY");
    check_args(op, N, C, H, W, h0, w0, H_out, W_out);
    if (dY.rows != N || dY.cols != C * H_out * W_out)
        fail(op, "dY shape must be (N, C*H_out*W_out)");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    if (N == 0 || cols_in == 0) return;

    // All-bits-zero is +0.0 in IEEE 754 for FP32/FP16/BF16, so cudaMemset is
    // safe across dtypes.
    const long long total_in = (long long)N * cols_in;
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
        dX.data, 0, static_cast<size_t>(total_in) * bytes_of(dY.dtype), cur_stream()));

    if (H_out == 0 || W_out == 0) return;

    const long long total_out = (long long)N * C * H_out * W_out;
    if (dY.dtype == ::brotensor::Dtype::FP16) {
        slice2d_backward_kernel<__half><<<sl_grid(total_out), SL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(dY.data), static_cast<__half*>(dX.data),
            N, C, H, W, h0, w0, H_out, W_out);
    } else if (dY.dtype == ::brotensor::Dtype::BF16) {
        slice2d_backward_kernel<__nv_bfloat16><<<sl_grid(total_out), SL_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(dY.data), static_cast<__nv_bfloat16*>(dX.data),
            N, C, H, W, h0, w0, H_out, W_out);
    } else {
        slice2d_backward_kernel<float><<<sl_grid(total_out), SL_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
            N, C, H, W, h0, w0, H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_slice2d(::brotensor::detail::OpsVTable& v) {
    v.slice2d_forward  = &slice2d_forward;
    v.slice2d_backward = &slice2d_backward;
}

} // namespace brotensor::detail::cuda
