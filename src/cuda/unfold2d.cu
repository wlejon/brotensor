// ─── CUDA 2D neighborhood unfold (im2col, spatial-preserving) ───────────────
//
// CUDA port of src/cpu/unfold2d.cpp. One thread per output element. Contracts,
// layout, and `mode` convention are identical to the CPU file:
//   X : (N, C*H*W)            NCHW flat
//   Y : (N, C*kK*H_out*W_out) with kK = kH*kW, k = ky*kW + kx
//   Y[n,c,k,oy,ox] = X[n,c, oy*sh - pad_top + ky, ox*sw - pad_left + kx]
//   out-of-range source: mode 0 zero / 1 reflect / 2 replicate.
//
// CPU is FP32-only; CUDA additionally supports FP16/BF16 (cast at load/store).
// Forward OVERWRITES Y. Inference-only: no backward.

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

constexpr int UF_BLOCK = 256;

inline int uf_grid(long long n) {
    long long blocks = (n + UF_BLOCK - 1) / UF_BLOCK;
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
        t.dtype != ::brotensor::Dtype::BF16)
        fail(op, std::string(name) + " must be FP32/FP16/BF16");
}

template <typename T> __device__ inline float uf_load(const T* p);
template <> __device__ inline float uf_load<float>(const float* p) { return *p; }
template <> __device__ inline float uf_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float uf_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void uf_store(T* p, float v);
template <> __device__ inline void uf_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void uf_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void uf_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// Device mirror of the CPU unf_src helper.
__device__ inline int unf_src_d(int coord, int L, int mode) {
    if (coord >= 0 && coord < L) return coord;
    if (mode == 0) return -1;
    if (mode == 2) return coord < 0 ? 0 : L - 1;
    if (L == 1) return 0;
    int q = coord;
    const int period = 2 * (L - 1);
    q %= period;
    if (q < 0) q += period;
    return q < L ? q : period - q;
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
            v = uf_load<T>(&X[xbase + (long long)sy * W + sx]);
        }
        uf_store<T>(&Y[idx], v);
    }
}

} // namespace

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
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (kH < 1 || kW < 1) fail(op, "kH/kW must be >=1");
    if (stride_h < 1 || stride_w < 1) fail(op, "stride must be >=1");
    if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0)
        fail(op, "pad counts must be >=0");
    if (mode < 0 || mode > 2)
        fail(op, "mode must be 0 (zero), 1 (reflect) or 2 (replicate)");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");

    const int H_out = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    const int W_out = (W + pad_left + pad_right - kW) / stride_w + 1;
    if (H_out < 1 || W_out < 1)
        fail(op, "kernel/padding/stride yield empty output");
    const int kK = kH * kW;
    const int cols_out = C * kK * H_out * W_out;
    if (Y.rows != N || Y.cols != cols_out || Y.dtype != X.dtype)
        Y.resize(N, cols_out, X.dtype);
    if (N == 0) return;

    const long long total = (long long)N * cols_out;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        unfold2d_forward_kernel<__half><<<uf_grid(total), UF_BLOCK>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_left, mode,
            H_out, W_out);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        unfold2d_forward_kernel<__nv_bfloat16><<<uf_grid(total), UF_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_left, mode,
            H_out, W_out);
    } else {
        unfold2d_forward_kernel<float><<<uf_grid(total), UF_BLOCK>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, H, W, kH, kW, stride_h, stride_w, pad_top, pad_left, mode,
            H_out, W_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_unfold2d(::brotensor::detail::OpsVTable& v) {
    v.unfold2d_forward = &unfold2d_forward;
}

} // namespace brotensor::detail::cuda
