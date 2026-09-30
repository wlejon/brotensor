// Native AMD HIP spatial operations (window partition / reverse & convex upsample).
//
// Implements:
//   - window_partition_forward
//   - window_reverse_forward
//   - convex_upsample_forward
//
// Optimized for Wave32 wavefronts on RDNA 3.5.

#include "detail/spatial_internal.h"
#include "detail/hip_check.h"

#include <brotensor/tensor.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int WP_BLOCK = 256;
constexpr int CU_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

inline int wp_grid(long long n) {
    long long blocks = (n + WP_BLOCK - 1) / WP_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

inline int cu_grid(long long n) {
    long long blocks = (n + CU_BLOCK - 1) / CU_BLOCK;
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

inline void check_fp(const ::brotensor::Tensor& t,
                     const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        fail(op, std::string(name) + " must be FP32/FP16/BF16");
    }
}

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename T> __device__ inline float cu_load(const T* p);
template <> __device__ inline float cu_load<float>(const float* p) { return *p; }
template <> __device__ inline float cu_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float cu_load<hip_bfloat16>(const hip_bfloat16* p) { return float(*p); }

template <typename T> __device__ inline void cu_store(T* p, float v);
template <> __device__ inline void cu_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void cu_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void cu_store<hip_bfloat16>(hip_bfloat16* p, float v) { *p = hip_bfloat16(v); }

// ── window_partition kernels ────────────────────────────────────────────────
__global__ void window_partition_forward_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int N, int C, int H, int W,
                                                int window,
                                                int nw_h, int nw_w) {
    const long long total = (long long)N * nw_h * nw_w * C * window * window;
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

__global__ void window_reverse_forward_kernel(const float* __restrict__ X,
                                              float* __restrict__ Y,
                                              int N, int C, int H, int W,
                                              int window,
                                              int nw_w) {
    const long long total = (long long)N * C * H * W;
    const int win2 = window * window;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int w = static_cast<int>(idx % W);
        long long t = idx / W;
        const int h = static_cast<int>(t % H);
        t /= H;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);

        const int nh = h / window, lh = h % window;
        const int nw = w / window, lw = w % window;
        const long long win_row = ((long long)n * (H / window) + nh) * nw_w + nw;
        const long long in_idx =
            (win_row * C + c) * win2 + (long long)lh * window + lw;
        Y[idx] = X[in_idx];
    }
}

// ── convex_upsample kernel ──────────────────────────────────────────────────
template <typename T>
__global__ void convex_upsample_kernel(const T* __restrict__ X,
                                       const T* __restrict__ Mask,
                                       T* __restrict__ Y,
                                       int N, int C, int H, int W, int scale) {
    const int HW = H * W;
    const int kk = scale * scale;
    const int oH = scale * H, oW = scale * W;
    const long long oHW = (long long)oH * oW;
    const long long spatial_total = (long long)N * oHW;

    __shared__ double w[9];

    for (long long sidx = blockIdx.x; sidx < spatial_total; sidx += gridDim.x) {
        const int ox = static_cast<int>(sidx % oW);
        long long t = sidx / oW;
        const int oy = static_cast<int>(t % oH);
        const int n = static_cast<int>(t / oH);

        const int y = oy / scale, sy = oy % scale;
        const int x = ox / scale, sx = ox % scale;
        const int sub = sy * scale + sx;
        const int pix = y * W + x;
        const T* m_img = Mask + (long long)n * 9 * kk * HW;

        if (threadIdx.x == 0) {
            double mx = -1e300;
            for (int m = 0; m < 9; ++m) {
                const double v = cu_load<T>(&m_img[((long long)m * kk + sub) * HW + pix]);
                if (v > mx) mx = v;
            }
            double sum = 0.0;
            for (int m = 0; m < 9; ++m) {
                const double e = exp(cu_load<T>(&m_img[((long long)m * kk + sub) * HW + pix]) - mx);
                w[m] = e; sum += e;
            }
            const double invs = 1.0 / sum;
            for (int m = 0; m < 9; ++m) w[m] *= invs;
        }
        __syncthreads();

        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const T* xc = X + ((long long)n * C + c) * HW;
            double acc = 0.0;
            for (int m = 0; m < 9; ++m) {
                const int ny = clampi(y - 1 + m / 3, 0, H - 1);
                const int nx = clampi(x - 1 + m % 3, 0, W - 1);
                acc += w[m] * cu_load<T>(&xc[(long long)ny * W + nx]);
            }
            const long long oidx = ((long long)n * C + c) * oHW + (long long)oy * oW + ox;
            cu_store<T>(&Y[oidx], static_cast<float>(acc));
        }
        __syncthreads();
    }
}

} // namespace

void window_partition_forward(const ::brotensor::Tensor& X,
                              int N, int C, int H, int W, int window,
                              ::brotensor::Tensor& Y) {
    const char* op = "window_partition_forward";
    check_fp32(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (window < 1) fail(op, "window must be >=1");
    if (H % window != 0 || W % window != 0) {
        fail(op, "H and W must be multiples of window");
    }
    if (X.rows != N || X.cols != C * H * W) {
        fail(op, "X shape must match (N, C*H*W)");
    }
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int out_rows = N * nw_h * nw_w;
    const int out_cols = C * window * window;
    if (Y.rows != out_rows || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(out_rows, out_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0) return;
    const long long total = (long long)out_rows * out_cols;
    window_partition_forward_kernel<<<wp_grid(total), WP_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<float*>(Y.data),
        N, C, H, W, window, nw_h, nw_w);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void window_reverse_forward(const ::brotensor::Tensor& X,
                            int N, int C, int H, int W, int window,
                            ::brotensor::Tensor& Y) {
    const char* op = "window_reverse_forward";
    check_fp32(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (window < 1) fail(op, "window must be >=1");
    if (H % window != 0 || W % window != 0) {
        fail(op, "H and W must be multiples of window");
    }
    const int nw_h = H / window;
    const int nw_w = W / window;
    const int in_rows = N * nw_h * nw_w;
    const int in_cols = C * window * window;
    if (X.rows != in_rows || X.cols != in_cols) {
        fail(op, "X shape must match windowed batch");
    }
    const int out_cols = C * H * W;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(N, out_cols, ::brotensor::Dtype::FP32);
    }
    if (N == 0 || C == 0) return;
    const long long total = (long long)N * out_cols;
    window_reverse_forward_kernel<<<wp_grid(total), WP_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(X.data), static_cast<float*>(Y.data),
        N, C, H, W, window, nw_w);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void convex_upsample_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             ::brotensor::Tensor& Y) {
    const char* op = "convex_upsample_forward";
    check_fp(X, op, "X");
    check_fp(Mask, op, "Mask");
    if (Mask.dtype != X.dtype) fail(op, "Mask dtype must match X");
    if (scale < 1) fail(op, "scale must be >= 1");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "dimensions must be non-negative / non-zero");

    const int kk = scale * scale;
    const long long mask_cols = (long long)9 * kk * H * W;
    if (X.rows != N || X.cols != C * H * W) fail(op, "X shape mismatch");
    if (Mask.rows != N || Mask.cols != mask_cols) fail(op, "Mask shape mismatch");

    const int oH = scale * H, oW = scale * W;
    const int out_cols = C * oH * oW;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || C == 0) return;

    const long long total_spatial = (long long)N * oH * oW;
    const int grid = cu_grid(total_spatial);
    hipStream_t stream = cur_stream();

    if (X.dtype == ::brotensor::Dtype::FP32) {
        convex_upsample_kernel<float><<<grid, CU_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(Mask.data),
            static_cast<float*>(Y.data), N, C, H, W, scale);
    } else if (X.dtype == ::brotensor::Dtype::FP16) {
        convex_upsample_kernel<__half><<<grid, CU_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(Mask.data),
            static_cast<__half*>(Y.data), N, C, H, W, scale);
    } else {
        convex_upsample_kernel<hip_bfloat16><<<grid, CU_BLOCK, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<const hip_bfloat16*>(Mask.data),
            static_cast<hip_bfloat16*>(Y.data), N, C, H, W, scale);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
