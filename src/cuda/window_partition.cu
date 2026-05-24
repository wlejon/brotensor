// ─── CUDA window partition / reverse ────────────────────────────────────────
//
// FP32-only port of src/cpu/window_partition.cpp. Both ops are pure layout
// shuffles (no math) and exact inverses of each other.
//
// Layout (NCHW <-> windowed batch):
//   X NCHW   : (N, C*H*W) at ((n*C + c)*H + h)*W + w
//   Y windowed: (N*nw_h*nw_w, C*window*window)
//               row index = n*nw_h*nw_w + nh*nw_w + nw
//               within-row = (c*window + lh)*window + lw
//   With (h, w) = (nh*window + lh, nw*window + lw).
//
// Both ops OVERWRITE the output. No accumulation.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int WP_BLOCK = 256;

inline int wp_grid(long long n) {
    long long blocks = (n + WP_BLOCK - 1) / WP_BLOCK;
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

inline void check_args(const char* op, int N, int C, int H, int W,
                       int window) {
    if (N < 0 || C < 1 || H < 1 || W < 1) {
        fail(op, "C/H/W must be >=1 and N >=0");
    }
    if (window < 1) {
        fail(op, "window must be >=1");
    }
    if (H % window != 0 || W % window != 0) {
        fail(op, "H and W must be multiples of window (use pad2d first if "
                 "the input doesn't align)");
    }
}

// Kernel-per-output-element. We iterate by output index in the windowed
// layout, decompose into (n, nh, nw, c, lh, lw), and source from NCHW.
__global__ void window_partition_forward_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int N, int C, int H, int W,
                                                int window,
                                                int nw_h, int nw_w) {
    const long long total =
        (long long)N * nw_h * nw_w * C * window * window;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int lw = static_cast<int>(idx % window);
        long long t = idx / window;
        const int lh = static_cast<int>(t % window);
        t /= window;
        const int c = static_cast<int>(t % C);
        t /= C;
        const int nw = static_cast<int>(t % nw_w);
        t /= nw_w;
        const int nh = static_cast<int>(t % nw_h);
        const int n  = static_cast<int>(t / nw_h);

        const int h = nh * window + lh;
        const int w = nw * window + lw;
        const long long x_off = (((long long)n * C + c) * H + h) * W + w;
        Y[idx] = X[x_off];
    }
}

// Kernel-per-output-element (NCHW side). Output index decomposes into
// (n, c, h, w); we recover (nh, lh, nw, lw) from (h, w) and source from the
// windowed layout.
__global__ void window_reverse_forward_kernel(const float* __restrict__ X,
                                              float* __restrict__ Y,
                                              int N, int C, int H, int W,
                                              int window,
                                              int nw_h, int nw_w) {
    const long long total = (long long)N * C * H * W;
    const int cols_in = C * window * window;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int w = static_cast<int>(idx % W);
        long long t = idx / W;
        const int h = static_cast<int>(t % H);
        t /= H;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);

        const int nh = h / window;
        const int lh = h - nh * window;
        const int nw = w / window;
        const int lw = w - nw * window;
        const int b_in = (n * nw_h + nh) * nw_w + nw;
        const long long x_off =
            (long long)b_in * cols_in +
            ((long long)c * window + lh) * window + lw;
        Y[idx] = X[x_off];
    }
}

} // namespace

void window_partition_forward(const ::brotensor::Tensor& X,
                              int N, int C, int H, int W, int window,
                              ::brotensor::Tensor& Y) {
    const char* op = "window_partition_forward";
    check_fp32(X, op, "X");
    check_args(op, N, C, H, W, window);
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must be (N, C*H*W)");
    }
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int B_out = N * nw_h * nw_w;
    const int cols_out = C * window * window;
    if (Y.rows != B_out || Y.cols != cols_out ||
        Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(B_out, cols_out, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    const long long total = (long long)B_out * cols_out;
    window_partition_forward_kernel<<<wp_grid(total), WP_BLOCK>>>(
        static_cast<const float*>(X.data),
        static_cast<float*>(Y.data),
        N, C, H, W, window, nw_h, nw_w);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void window_reverse_forward(const ::brotensor::Tensor& X,
                            int N, int C, int H, int W, int window,
                            ::brotensor::Tensor& Y) {
    const char* op = "window_reverse_forward";
    check_fp32(X, op, "X");
    check_args(op, N, C, H, W, window);
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int B_in = N * nw_h * nw_w;
    const int cols_in = C * window * window;
    if (X.rows != B_in || X.cols != cols_in) {
        fail(op, "X shape must be (N*nw_h*nw_w, C*window*window)");
    }
    const int cols_out = C * H * W;
    if (Y.rows != N || Y.cols != cols_out ||
        Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, cols_out, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || cols_out == 0) return;

    const long long total = (long long)N * cols_out;
    window_reverse_forward_kernel<<<wp_grid(total), WP_BLOCK>>>(
        static_cast<const float*>(X.data),
        static_cast<float*>(Y.data),
        N, C, H, W, window, nw_h, nw_w);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_window_partition(::brotensor::detail::OpsVTable& v) {
    v.window_partition_forward = &window_partition_forward;
    v.window_reverse_forward   = &window_reverse_forward;
}

} // namespace brotensor::detail::cuda
