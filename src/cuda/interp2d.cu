// ─── CUDA arbitrary-scale 2D resample ───────────────────────────────────────
//
// CUDA port of src/cpu/interp2d.cpp. Storage dtype dispatched on X.dtype:
//   FP32 / FP16 / BF16  for nearest + bilinear (forward + backward).
//   FP32 only           for bicubic forward (backward not implemented;
//                       throws — matches the CPU contract).
//
// Math is always FP32; FP16 / BF16 paths read/write half-storage but
// accumulate / weight in single precision (parity with src/cuda/resample.cu).
//
// Sampling convention — PyTorch align_corners=False / half-pixel; identical
// to interp2d.cpp. With (H_out, W_out) == (2H, 2W) the output matches the
// upsample_*_2x kernels exactly (relied on by test_interp2d_parity).
//
// Accumulation:
//   interp2d_forward  — Y  OVERWRITTEN.
//   interp2d_backward — dX OVERWRITTEN (cudaMemset zero, then atomicAdd
//                       scatter — atomicAdd is FP32-only, so FP16/BF16
//                       inputs scatter into an FP32 scratch and the result
//                       is cast back at the end).

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int IP_BLOCK = 256;

inline int ip_grid(long long n) {
    long long blocks = (n + IP_BLOCK - 1) / IP_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_args(const char* op,
                       int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode,
                       bool allow_bicubic) {
    if (N < 0 || C < 0 || H_in < 0 || W_in < 0 ||
        H_out < 0 || W_out < 0) {
        fail(op, "N, C, H_in, W_in, H_out, W_out must be non-negative");
    }
    const int max_mode = allow_bicubic ? 3 : 1;
    if (mode < 0 || mode > max_mode) {
        fail(op, allow_bicubic
            ? "mode must be 0 (nearest), 1 (bilinear), 2 (bicubic a=-0.5, PIL), "
              "or 3 (bicubic a=-0.75, torch)"
            : "mode must be 0 (nearest) or 1 (bilinear) — bicubic backward "
              "is not implemented");
    }
    if ((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)) {
        fail(op, "input spatial dims must be > 0 when output spatial "
                 "dims are > 0");
    }
}

inline void check_dtype_forward(const ::brotensor::Tensor& t,
                                const char* op, const char* name,
                                int mode) {
    if (mode >= 2) {
        if (t.dtype != ::brotensor::Dtype::FP32) {
            fail(op, std::string(name) +
                     " must be FP32 (bicubic only supports FP32 on CUDA)");
        }
        return;
    }
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        fail(op, std::string(name) + " must be FP32, FP16, or BF16");
    }
}

// Corner-aligned source coordinate (torch align_corners=True): o*(in-1)/(out-1),
// with the out==1 degenerate case pinned to 0. align==0 keeps the half-pixel map.
__device__ inline double src_coord(int o, int in_dim, int out_dim,
                                   double scale, int align) {
    if (align) {
        if (out_dim <= 1) return 0.0;
        return static_cast<double>(o) * static_cast<double>(in_dim - 1) /
               static_cast<double>(out_dim - 1);
    }
    return (o + 0.5) * scale - 0.5;
}

// ── per-dtype storage adapters ─────────────────────────────────────────────
template <typename T> __device__ inline float to_f32(T v);
template <> __device__ inline float to_f32<float>(float v) { return v; }
template <> __device__ inline float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ inline T from_f32(float v);
template <> __device__ inline float        from_f32<float>(float v) { return v; }
template <> __device__ inline __half       from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Keys cubic-convolution kernel with coefficient `a`. a = -0.5 is Catmull-Rom
// (PIL BICUBIC); a = -0.75 matches torch interpolate(bicubic) and OpenCV.
// Mirrors src/cpu/interp2d.cpp.
__device__ inline float cubic_keys(float t, float a) {
    const float at = t < 0.0f ? -t : t;
    if (at < 1.0f) {
        return ((a + 2.0f) * at - (a + 3.0f)) * at * at + 1.0f;
    }
    if (at < 2.0f) {
        // a*t^3 - 5a*t^2 + 8a*t - 4a, Horner in |t|.
        return ((a * at - 5.0f * a) * at + 8.0f * a) * at - 4.0f * a;
    }
    return 0.0f;
}

// ── Forward kernel (nearest / bilinear) — templated on storage dtype ──────
//
// One thread per (n, c, oh, ow). Grid iterates if total > thread count.
template <typename T>
__global__ void interp2d_forward_kernel(const T* __restrict__ X,
                                        T* __restrict__ Y,
                                        int N, int C, int H_in, int W_in,
                                        int H_out, int W_out, int mode,
                                        double sy, double sx, int align) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * H_in * W_in;
        const double src_y = src_coord(oh, H_in, H_out, sy, align);
        const double src_x = src_coord(ow, W_in, W_out, sx, align);
        float out;
        if (mode == 0) {
            const int iy = clampi(static_cast<int>(nearbyint(src_y)),
                                  0, H_in - 1);
            const int ix = clampi(static_cast<int>(nearbyint(src_x)),
                                  0, W_in - 1);
            out = to_f32<T>(X[xbase + iy * W_in + ix]);
        } else {
            const int y0 = static_cast<int>(floor(src_y));
            const int x0 = static_cast<int>(floor(src_x));
            const float fy = static_cast<float>(src_y - y0);
            const float fx = static_cast<float>(src_x - x0);
            const int y0c = clampi(y0,     0, H_in - 1);
            const int y1c = clampi(y0 + 1, 0, H_in - 1);
            const int x0c = clampi(x0,     0, W_in - 1);
            const int x1c = clampi(x0 + 1, 0, W_in - 1);
            const float v00 = to_f32<T>(X[xbase + y0c * W_in + x0c]);
            const float v01 = to_f32<T>(X[xbase + y0c * W_in + x1c]);
            const float v10 = to_f32<T>(X[xbase + y1c * W_in + x0c]);
            const float v11 = to_f32<T>(X[xbase + y1c * W_in + x1c]);
            const float top = v00 + (v01 - v00) * fx;
            const float bot = v10 + (v11 - v10) * fx;
            out = top + (bot - top) * fy;
        }
        Y[idx] = from_f32<T>(out);
    }
}

// Bicubic forward — FP32 only. Same one-thread-per-output pattern.
__global__ void interp2d_bicubic_forward_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int N, int C, int H_in, int W_in,
                                                int H_out, int W_out,
                                                double sy, double sx, int align,
                                                float a) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * H_in * W_in;
        const double src_y = src_coord(oh, H_in, H_out, sy, align);
        const double src_x = src_coord(ow, W_in, W_out, sx, align);
        const int y0 = static_cast<int>(floor(src_y));
        const int x0 = static_cast<int>(floor(src_x));
        const float fy = static_cast<float>(src_y - y0);
        const float fx = static_cast<float>(src_x - x0);
        float wy[4], wx[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            wy[k] = cubic_keys(fy - (k - 1), a);
            wx[k] = cubic_keys(fx - (k - 1), a);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int iy = clampi(y0 + j - 1, 0, H_in - 1);
            float row = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int ix = clampi(x0 + i - 1, 0, W_in - 1);
                row += wx[i] * X[xbase + iy * W_in + ix];
            }
            acc += wy[j] * row;
        }
        Y[idx] = acc;
    }
}

// ── Backward kernels — scatter via atomicAdd into FP32 storage ────────────
//
// For FP32 dX we atomic-add into Y directly. For FP16/BF16, we run the
// kernel against an FP32 scratch buffer, then cast back. One thread per
// output gradient (n, c, oh, ow).
template <typename T>
__global__ void interp2d_backward_kernel(const T* __restrict__ dY,
                                         float* __restrict__ dX,
                                         int N, int C, int H_in, int W_in,
                                         int H_out, int W_out, int mode,
                                         double sy, double sx) {
    const long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ow = static_cast<int>(idx % W_out);
        long long t = idx / W_out;
        const int oh = static_cast<int>(t % H_out);
        t /= H_out;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);
        const long long xbase = ((long long)n * C + c) * H_in * W_in;
        const double src_y = (oh + 0.5) * sy - 0.5;
        const double src_x = (ow + 0.5) * sx - 0.5;
        const float g = to_f32<T>(dY[idx]);
        if (mode == 0) {
            const int iy = clampi(static_cast<int>(nearbyint(src_y)),
                                  0, H_in - 1);
            const int ix = clampi(static_cast<int>(nearbyint(src_x)),
                                  0, W_in - 1);
            atomicAdd(&dX[xbase + iy * W_in + ix], g);
        } else {
            const int y0 = static_cast<int>(floor(src_y));
            const int x0 = static_cast<int>(floor(src_x));
            const float fy = static_cast<float>(src_y - y0);
            const float fx = static_cast<float>(src_x - x0);
            const int y0c = clampi(y0,     0, H_in - 1);
            const int y1c = clampi(y0 + 1, 0, H_in - 1);
            const int x0c = clampi(x0,     0, W_in - 1);
            const int x1c = clampi(x0 + 1, 0, W_in - 1);
            const float w00 = (1.0f - fy) * (1.0f - fx);
            const float w01 = (1.0f - fy) * fx;
            const float w10 = fy          * (1.0f - fx);
            const float w11 = fy          * fx;
            atomicAdd(&dX[xbase + y0c * W_in + x0c], w00 * g);
            atomicAdd(&dX[xbase + y0c * W_in + x1c], w01 * g);
            atomicAdd(&dX[xbase + y1c * W_in + x0c], w10 * g);
            atomicAdd(&dX[xbase + y1c * W_in + x1c], w11 * g);
        }
    }
}

// FP32->FP16 / FP32->BF16 store kernel (used to fold the FP32 scratch back).
template <typename T>
__global__ void cast_fp32_store(const float* __restrict__ src,
                                T* __restrict__ dst, long long n) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < n; idx += (long long)blockDim.x * gridDim.x) {
        dst[idx] = from_f32<T>(src[idx]);
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrappers
// ════════════════════════════════════════════════════════════════════════════

// Shared launcher for the half-pixel and corner-aligned forward resamples —
// they differ only by the `align` flag threaded into the kernels.
static void launch_forward(const ::brotensor::Tensor& X,
                           int N, int C, int H_in, int W_in,
                           int H_out, int W_out, int mode, int align,
                           ::brotensor::Tensor& Y, const char* op) {
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/true);
    check_dtype_forward(X, op, "X", mode);

    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    if (N == 0 || cols == 0) return;

    const double sy = static_cast<double>(H_in) / static_cast<double>(H_out);
    const double sx = static_cast<double>(W_in) / static_cast<double>(W_out);
    const long long total = (long long)N * C * H_out * W_out;

    if (mode >= 2) {
        // bicubic — FP32-only (check_dtype_forward enforced).
        // mode 2: a=-0.5 (PIL); mode 3: a=-0.75 (torch).
        const float a = (mode == 3) ? -0.75f : -0.5f;
        interp2d_bicubic_forward_kernel<<<ip_grid(total), IP_BLOCK>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, sy, sx, align, a);
    } else if (X.dtype == ::brotensor::Dtype::FP32) {
        interp2d_forward_kernel<float><<<ip_grid(total), IP_BLOCK>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx, align);
    } else if (X.dtype == ::brotensor::Dtype::FP16) {
        interp2d_forward_kernel<__half><<<ip_grid(total), IP_BLOCK>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx, align);
    } else {
        interp2d_forward_kernel<__nv_bfloat16><<<ip_grid(total), IP_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx, align);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void interp2d_forward(const ::brotensor::Tensor& X,
                      int N, int C, int H_in, int W_in,
                      int H_out, int W_out, int mode,
                      ::brotensor::Tensor& Y) {
    launch_forward(X, N, C, H_in, W_in, H_out, W_out, mode,
                   /*align=*/0, Y, "interp2d_forward");
}

void interp2d_align_corners_forward(const ::brotensor::Tensor& X,
                                    int N, int C, int H_in, int W_in,
                                    int H_out, int W_out, int mode,
                                    ::brotensor::Tensor& Y) {
    launch_forward(X, N, C, H_in, W_in, H_out, W_out, mode,
                   /*align=*/1, Y, "interp2d_align_corners_forward");
}

void interp2d_backward(const ::brotensor::Tensor& dY,
                       int N, int C, int H_in, int W_in,
                       int H_out, int W_out, int mode,
                       ::brotensor::Tensor& dX) {
    check_args("interp2d_backward", N, C, H_in, W_in, H_out, W_out, mode,
               /*allow_bicubic=*/false);
    if (dY.dtype != ::brotensor::Dtype::FP32 &&
        dY.dtype != ::brotensor::Dtype::FP16 &&
        dY.dtype != ::brotensor::Dtype::BF16) {
        fail("interp2d_backward", "dY must be FP32, FP16, or BF16");
    }

    const int cols_in = C * H_in * W_in;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    if (N == 0 || cols_in == 0) return;

    const long long total_in = (long long)N * cols_in;
    const long long total_out = (long long)N * C * H_out * W_out;
    const double sy = static_cast<double>(H_in) / static_cast<double>(H_out);
    const double sx = static_cast<double>(W_in) / static_cast<double>(W_out);

    if (dY.dtype == ::brotensor::Dtype::FP32) {
        BROTENSOR_CUDA_CHECK(cudaMemset(
            dX.data, 0, static_cast<size_t>(total_in) * sizeof(float)));
        if (H_out == 0 || W_out == 0) return;
        interp2d_backward_kernel<float><<<ip_grid(total_out), IP_BLOCK>>>(
            static_cast<const float*>(dY.data),
            static_cast<float*>(dX.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // FP16 / BF16: scatter into FP32 scratch, then cast back.
    float* scratch = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&scratch,
        static_cast<size_t>(total_in) * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMemset(scratch, 0,
        static_cast<size_t>(total_in) * sizeof(float)));
    if (H_out > 0 && W_out > 0) {
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            interp2d_backward_kernel<__half><<<ip_grid(total_out), IP_BLOCK>>>(
                static_cast<const __half*>(dY.data), scratch,
                N, C, H_in, W_in, H_out, W_out, mode, sy, sx);
        } else {
            interp2d_backward_kernel<__nv_bfloat16>
                <<<ip_grid(total_out), IP_BLOCK>>>(
                    static_cast<const __nv_bfloat16*>(dY.data), scratch,
                    N, C, H_in, W_in, H_out, W_out, mode, sy, sx);
        }
    }
    if (dY.dtype == ::brotensor::Dtype::FP16) {
        cast_fp32_store<__half><<<ip_grid(total_in), IP_BLOCK>>>(
            scratch, static_cast<__half*>(dX.data), total_in);
    } else {
        cast_fp32_store<__nv_bfloat16><<<ip_grid(total_in), IP_BLOCK>>>(
            scratch, static_cast<__nv_bfloat16*>(dX.data), total_in);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    BROTENSOR_CUDA_CHECK(cudaFree(scratch));
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_interp2d(::brotensor::detail::OpsVTable& v) {
    v.interp2d_forward                = &interp2d_forward;
    v.interp2d_backward               = &interp2d_backward;
    v.interp2d_align_corners_forward  = &interp2d_align_corners_forward;
}

} // namespace brotensor::detail::cuda
