// ─── CUDA 2D padding: pad2d_forward + pad2d_backward ──────────────────────
//
// CUDA port of src/cpu/pad2d.cpp. Contracts:
//   * NCHW flat layout (rows = N, cols = C*H*W for X / dX; cols = C*H_pad*W_pad
//     for Y / dY).
//   * Modes: 0 = zero, 1 = reflect (no edge repeat; requires pad < H/W on that
//     axis), 2 = replicate (clamp to edge sample).
//   * Forward overwrites Y. Backward overwrites dX (zero-then-scatter). For
//     reflect/replicate multiple output positions may collapse onto the same
//     input pixel — use atomicAdd. Zero mode has no overlap but we still use
//     atomicAdd for uniformity.
//
// CPU is FP32-only; CUDA additionally supports FP16/BF16 — forward casts at
// load/store, backward accumulates into FP32 scratch (no native fp16/bf16
// atomicAdd across all arches) then casts into dX.

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

constexpr int PD_BLOCK = 256;

inline int pd_grid(long long n) {
    long long blocks = (n + PD_BLOCK - 1) / PD_BLOCK;
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

template <typename T> __device__ inline float pd_load(const T* p);
template <> __device__ inline float pd_load<float>(const float* p) { return *p; }
template <> __device__ inline float pd_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float pd_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void pd_store(T* p, float v);
template <> __device__ inline void pd_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void pd_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void pd_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

inline void check_args(const char* op,
                       int N, int C, int H, int W,
                       int pad_top, int pad_bottom,
                       int pad_left, int pad_right, int mode) {
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0)
        fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2)
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (mode == 1) {
        if (pad_top >= H || pad_bottom >= H)
            fail(op, "reflect padding requires pad_top and pad_bottom < H");
        if (pad_left >= W || pad_right >= W)
            fail(op, "reflect padding requires pad_left and pad_right < W");
    }
}

// Device-side mirror of the CPU `pad_src` helper. Returns the input index
// (in [0, L)) for output position p, or -1 for a zero-padded slot.
__device__ inline int pad_src_d(int p, int L, int pad_left, int mode) {
    const int rel = p - pad_left;
    if (rel >= 0 && rel < L) return rel;
    if (mode == 0) return -1;
    if (mode == 2) return rel < 0 ? 0 : L - 1;
    // mode == 1: numpy 'reflect' (no edge repeat).
    if (L == 1) return 0;
    int q = rel;
    const int period = 2 * (L - 1);
    q %= period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
}

// ── pad2d_forward kernel — one thread per output pixel ────────────────────
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
            v = pd_load<T>(&X[xbase + (long long)src_h * W + src_w]);
        }
        pd_store<T>(&Y[idx], v);
    }
}

// ── pad2d_backward kernel — scatter via atomicAdd (FP32 dst) ──────────────
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
        if (src_h < 0) continue;
        const int src_w = pad_src_d(ow, W, pad_left, mode);
        if (src_w < 0) continue;
        const long long xbase = ((long long)n * C + c) * H * W;
        atomicAdd(&dX_fp32[xbase + (long long)src_h * W + src_w],
                  pd_load<T>(&dY[idx]));
    }
}

template <typename T>
__global__ void pad2d_cast_fp32_to_T(const float* __restrict__ src,
                                     T* __restrict__ dst, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        pd_store<T>(&dst[i], src[i]);
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════

void pad2d_forward(const ::brotensor::Tensor& X,
                   int N, int C, int H, int W,
                   int pad_top, int pad_bottom,
                   int pad_left, int pad_right, int mode,
                   ::brotensor::Tensor& Y) {
    const char* op = "pad2d_forward";
    check_fp(X, op, "X");
    check_args(op, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode);
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    const int cols_out = C * H_pad * W_pad;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype) {
        Y.resize(N, cols_out, X.dtype);
    }
    if (N == 0 || C == 0) return;

    const long long total = (long long)N * cols_out;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        pad2d_forward_kernel<__half><<<pd_grid(total), PD_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        pad2d_forward_kernel<__nv_bfloat16><<<pd_grid(total), PD_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    } else {
        pad2d_forward_kernel<float><<<pd_grid(total), PD_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void pad2d_backward(const ::brotensor::Tensor& dY,
                    int N, int C, int H, int W,
                    int pad_top, int pad_bottom,
                    int pad_left, int pad_right, int mode,
                    ::brotensor::Tensor& dX) {
    const char* op = "pad2d_backward";
    check_fp(dY, op, "dY");
    check_args(op, N, C, H, W, pad_top, pad_bottom, pad_left, pad_right, mode);
    const int H_pad = H + pad_top + pad_bottom;
    const int W_pad = W + pad_left + pad_right;
    if (dY.rows != N || dY.cols != C * H_pad * W_pad)
        fail(op, "dY shape must be (N, C*(H+pt+pb)*(W+pl+pr))");

    const int cols_in = C * H * W;
    if (dX.rows != N || dX.cols != cols_in || dX.dtype != dY.dtype) {
        dX.resize(N, cols_in, dY.dtype);
    }
    if (N == 0 || C == 0) return;

    const long long total_in  = (long long)N * cols_in;
    const long long total_out = (long long)N * C * H_pad * W_pad;

    if (dY.dtype == ::brotensor::Dtype::FP32) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            dX.data, 0, static_cast<size_t>(total_in) * sizeof(float), cur_stream()));
        if (total_out == 0) return;
        pad2d_backward_kernel<float><<<pd_grid(total_out), PD_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
            N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // FP16/BF16: scatter-accumulate into FP32 scratch, then cast into dX.
    float* d_scratch = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                                    static_cast<size_t>(total_in) * sizeof(float)));
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_scratch, 0,
                                    static_cast<size_t>(total_in) * sizeof(float), cur_stream()));
    if (total_out > 0) {
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            pad2d_backward_kernel<__half><<<pd_grid(total_out), PD_BLOCK, 0, cur_stream()>>>(
                static_cast<const __half*>(dY.data), d_scratch,
                N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
        } else {
            pad2d_backward_kernel<__nv_bfloat16><<<pd_grid(total_out), PD_BLOCK, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY.data), d_scratch,
                N, C, H, W, pad_top, pad_left, mode, H_pad, W_pad);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    if (total_in > 0) {
        if (dY.dtype == ::brotensor::Dtype::FP16) {
            pad2d_cast_fp32_to_T<__half><<<pd_grid(total_in), PD_BLOCK, 0, cur_stream()>>>(
                d_scratch, static_cast<__half*>(dX.data), total_in);
        } else {
            pad2d_cast_fp32_to_T<__nv_bfloat16><<<pd_grid(total_in), PD_BLOCK, 0, cur_stream()>>>(
                d_scratch, static_cast<__nv_bfloat16*>(dX.data), total_in);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    cudaFree(d_scratch);
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_pad2d(::brotensor::detail::OpsVTable& v) {
    v.pad2d_forward  = &pad2d_forward;
    v.pad2d_backward = &pad2d_backward;
}

} // namespace brotensor::detail::cuda
