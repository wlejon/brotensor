// Native AMD HIP arbitrary-scale 2D resample (nearest / bilinear / bicubic).
//
// Implements:
//   - interp2d_forward (FP32 / FP16 / BF16 for nearest/bilinear; FP32 for bicubic)
//   - interp2d_backward (FP32 / FP16 / BF16 for nearest/bilinear)
//   - interp2d_align_corners_forward (modes 0, 1, 2, 3)
//   - fill_hip_vtable_interp
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
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);
void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);

namespace {

constexpr int IP_BLOCK = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

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
        throw std::invalid_argument(std::string("brotensor: ") + op +
                                    ": N, C, H_in, W_in, H_out, W_out must be non-negative");
    }
    if (mode < 0 || mode > 3) {
        throw std::invalid_argument(std::string("brotensor: ") + op +
            ": mode must be 0 (nearest), 1 (bilinear), 2 (bicubic a=-0.5, PIL), or 3 (bicubic a=-0.75, torch)");
    }
    if (!allow_bicubic && (mode == 2 || mode == 3)) {
        const char* detail = (mode == 2)
            ? "mode 2 (bicubic a=-0.5, PIL)"
            : "mode 3 (bicubic a=-0.75, torch)";
        throw std::runtime_error(std::string("brotensor: ") + op +
            ": bicubic backward is not implemented for " + detail +
            " — mode must be 0 (nearest) or 1 (bilinear)");
    }
    if ((H_out > 0 && H_in == 0) || (W_out > 0 && W_in == 0)) {
        throw std::invalid_argument(std::string("brotensor: ") + op +
                                    ": input spatial dims must be > 0 when output spatial dims are > 0");
    }
}

inline void check_dtype_forward(const ::brotensor::Tensor& t,
                                const char* op, const char* name,
                                int mode) {
    if (mode >= 2) {
        if (t.dtype != ::brotensor::Dtype::FP32) {
            fail(op, std::string(name) + " must be FP32 (bicubic only supports FP32 on HIP)");
        }
        return;
    }
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        fail(op, std::string(name) + " must be FP32, FP16, or BF16");
    }
}

__device__ inline double src_coord(int o, int in_dim, int out_dim,
                                   double scale, int align) {
    if (align) {
        if (out_dim <= 1) return 0.0;
        return static_cast<double>(o) * static_cast<double>(in_dim - 1) /
               static_cast<double>(out_dim - 1);
    }
    return (o + 0.5) * scale - 0.5;
}

template <typename T> __device__ inline float to_f32(T v);
template <> __device__ inline float to_f32<float>(float v) { return v; }
template <> __device__ inline float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ inline float to_f32<hip_bfloat16>(hip_bfloat16 v) { return float(v); }

template <typename T> __device__ inline T from_f32(float v);
template <> __device__ inline float        from_f32<float>(float v) { return v; }
template <> __device__ inline __half       from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ inline hip_bfloat16 from_f32<hip_bfloat16>(float v) { return hip_bfloat16(v); }

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float cubic_keys(float t, float a) {
    const float at = t < 0.0f ? -t : t;
    if (at < 1.0f) {
        return ((a + 2.0f) * at - (a + 3.0f)) * at * at + 1.0f;
    }
    if (at < 2.0f) {
        return ((a * at - 5.0f * a) * at + 8.0f * a) * at - 4.0f * a;
    }
    return 0.0f;
}

template <typename T>
__global__ void interp2d_forward_kernel(const T* __restrict__ X,
                                        T* __restrict__ Y,
                                        int N, int C, int H_in, int W_in,
                                        int H_out, int W_out, int mode,
                                        double sy, double sx, int align) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (ow >= W_out) return;
    const double src_x = src_coord(ow, W_in, W_out, sx, align);
    const int x0 = (mode == 0) ? 0 : static_cast<int>(floor(src_x));
    const float fx = (mode == 0) ? 0.0f : static_cast<float>(src_x - x0);
    const int x0c = (mode == 0) ? clampi(static_cast<int>(nearbyint(src_x)), 0, W_in - 1) : clampi(x0, 0, W_in - 1);
    const int x1c = (mode == 0) ? 0 : clampi(x0 + 1, 0, W_in - 1);

    for (int nc = blockIdx.z; nc < N * C; nc += gridDim.z) {
        const long long xbase = (long long)nc * H_in * W_in;
        const long long ybase_nc = (long long)nc * H_out * W_out;
        for (int oh = blockIdx.y * blockDim.y + threadIdx.y; oh < H_out; oh += blockDim.y * gridDim.y) {
            const double src_y = src_coord(oh, H_in, H_out, sy, align);
            float out;
            if (mode == 0) {
                const int iy = clampi(static_cast<int>(nearbyint(src_y)), 0, H_in - 1);
                out = to_f32<T>(X[xbase + iy * W_in + x0c]);
            } else {
                const int y0 = static_cast<int>(floor(src_y));
                const float fy = static_cast<float>(src_y - y0);
                const int y0c = clampi(y0,     0, H_in - 1);
                const int y1c = clampi(y0 + 1, 0, H_in - 1);
                const float v00 = to_f32<T>(X[xbase + y0c * W_in + x0c]);
                const float v01 = to_f32<T>(X[xbase + y0c * W_in + x1c]);
                const float v10 = to_f32<T>(X[xbase + y1c * W_in + x0c]);
                const float v11 = to_f32<T>(X[xbase + y1c * W_in + x1c]);
                const float top = v00 + (v01 - v00) * fx;
                const float bot = v10 + (v11 - v10) * fx;
                out = top + (bot - top) * fy;
            }
            Y[ybase_nc + (long long)oh * W_out + ow] = from_f32<T>(out);
        }
    }
}

__global__ void interp2d_bicubic_forward_kernel(const float* __restrict__ X,
                                                float* __restrict__ Y,
                                                int N, int C, int H_in, int W_in,
                                                int H_out, int W_out,
                                                double sy, double sx, int align,
                                                float a) {
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (ow >= W_out) return;
    const double src_x = src_coord(ow, W_in, W_out, sx, align);
    const int x0 = static_cast<int>(floor(src_x));
    const float fx = static_cast<float>(src_x - x0);
    float wx[4];
    int ix[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        wx[i] = cubic_keys(fx - (i - 1), a);
        ix[i] = clampi(x0 + i - 1, 0, W_in - 1);
    }

    for (int nc = blockIdx.z; nc < N * C; nc += gridDim.z) {
        const long long xbase = (long long)nc * H_in * W_in;
        const long long ybase_nc = (long long)nc * H_out * W_out;
        for (int oh = blockIdx.y * blockDim.y + threadIdx.y; oh < H_out; oh += blockDim.y * gridDim.y) {
            const double src_y = src_coord(oh, H_in, H_out, sy, align);
            const int y0 = static_cast<int>(floor(src_y));
            const float fy = static_cast<float>(src_y - y0);
            float wy[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) wy[k] = cubic_keys(fy - (k - 1), a);

            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int iy = clampi(y0 + j - 1, 0, H_in - 1);
                const float* xrow = &X[xbase + iy * W_in];
                float row = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    row += wx[i] * xrow[ix[i]];
                }
                acc += wy[j] * row;
            }
            Y[ybase_nc + (long long)oh * W_out + ow] = acc;
        }
    }
}

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
            const int iy = clampi(static_cast<int>(nearbyint(src_y)), 0, H_in - 1);
            const int ix = clampi(static_cast<int>(nearbyint(src_x)), 0, W_in - 1);
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

template <typename T>
__global__ void cast_fp32_store(const float* __restrict__ src,
                                T* __restrict__ dst, long long n) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < n; idx += (long long)blockDim.x * gridDim.x) {
        dst[idx] = from_f32<T>(src[idx]);
    }
}

static void launch_forward(const ::brotensor::Tensor& X,
                           int N, int C, int H_in, int W_in,
                           int H_out, int W_out, int mode, int align,
                           ::brotensor::Tensor& Y, const char* op) {
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode, /*allow_bicubic=*/true);
    check_dtype_forward(X, op, "X", mode);

    const int out_cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype) {
        Y.resize(N, out_cols, X.dtype);
    }
    if (N == 0 || C == 0 || H_out == 0 || W_out == 0) return;

    const double sy = static_cast<double>(H_in) / static_cast<double>(H_out);
    const double sx = static_cast<double>(W_in) / static_cast<double>(W_out);
    hipStream_t stream = cur_stream();
    dim3 block(16, 16);
    const uint32_t gx = (static_cast<uint32_t>(W_out) + block.x - 1) / block.x;
    const uint32_t gy = (static_cast<uint32_t>(H_out) + block.y - 1) / block.y;
    const uint32_t gz = static_cast<uint32_t>(N * C);
    dim3 grid(gx, std::max(1u, std::min(gy, 65535u)), std::max(1u, std::min(gz, 65535u)));

    if (mode == 2 || mode == 3) {
        const float a = (mode == 2) ? -0.5f : -0.75f;
        interp2d_bicubic_forward_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, sy, sx, align, a);
    } else if (X.dtype == ::brotensor::Dtype::FP32) {
        interp2d_forward_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx, align);
    } else if (X.dtype == ::brotensor::Dtype::FP16) {
        interp2d_forward_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx, align);
    } else {
        interp2d_forward_kernel<hip_bfloat16><<<grid, block, 0, stream>>>(
            static_cast<const hip_bfloat16*>(X.data), static_cast<hip_bfloat16*>(Y.data),
            N, C, H_in, W_in, H_out, W_out, mode, sy, sx, align);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace

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
    const char* op = "interp2d_backward";
    check_args(op, N, C, H_in, W_in, H_out, W_out, mode, /*allow_bicubic=*/false);
    if (dY.dtype != ::brotensor::Dtype::FP32 &&
        dY.dtype != ::brotensor::Dtype::FP16 &&
        dY.dtype != ::brotensor::Dtype::BF16) {
        fail(op, "dY must be FP32, FP16, or BF16");
    }

    const int in_cols = C * H_in * W_in;
    if (dX.rows != N || dX.cols != in_cols || dX.dtype != dY.dtype) {
        dX.resize(N, in_cols, dY.dtype);
    }
    if (N == 0 || C == 0 || H_in == 0 || W_in == 0) return;

    const long long total_in = (long long)N * in_cols;
    const long long total_out = (long long)N * C * H_out * W_out;
    const double sy = static_cast<double>(H_in) / static_cast<double>(H_out);
    const double sx = static_cast<double>(W_in) / static_cast<double>(W_out);
    hipStream_t stream = cur_stream();

    if (dY.dtype == ::brotensor::Dtype::FP32) {
        BROTENSOR_HIP_CHECK(hipMemsetAsync(dX.data, 0, total_in * sizeof(float), stream));
        if (total_out > 0) {
            interp2d_backward_kernel<float><<<ip_grid(total_out), IP_BLOCK, 0, stream>>>(
                static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
                N, C, H_in, W_in, H_out, W_out, mode, sy, sx);
        }
    } else {
        float* dX_f32 = static_cast<float*>(hip_alloc(total_in * sizeof(float), dY.device.index));
        BROTENSOR_HIP_CHECK(hipMemsetAsync(dX_f32, 0, total_in * sizeof(float), stream));
        if (total_out > 0) {
            if (dY.dtype == ::brotensor::Dtype::FP16) {
                interp2d_backward_kernel<__half><<<ip_grid(total_out), IP_BLOCK, 0, stream>>>(
                    static_cast<const __half*>(dY.data), dX_f32,
                    N, C, H_in, W_in, H_out, W_out, mode, sy, sx);
            } else {
                interp2d_backward_kernel<hip_bfloat16><<<ip_grid(total_out), IP_BLOCK, 0, stream>>>(
                    static_cast<const hip_bfloat16*>(dY.data), dX_f32,
                    N, C, H_in, W_in, H_out, W_out, mode, sy, sx);
            }
        }
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            cast_fp32_store<__half><<<ip_grid(total_in), IP_BLOCK, 0, stream>>>(
                dX_f32, static_cast<__half*>(dX.data), total_in);
        } else {
            cast_fp32_store<hip_bfloat16><<<ip_grid(total_in), IP_BLOCK, 0, stream>>>(
                dX_f32, static_cast<hip_bfloat16*>(dX.data), total_in);
        }
        hip_free(dX_f32, dY.device.index);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_interp(::brotensor::detail::OpsVTable& v) {
    v.interp2d_forward               = &interp2d_forward;
    v.interp2d_backward              = &interp2d_backward;
    v.interp2d_align_corners_forward = &interp2d_align_corners_forward;
}

} // namespace brotensor::detail::hip
