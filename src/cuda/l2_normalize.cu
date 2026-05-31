// ─── CUDA L2 normalization over the channel axis (NCHW) ─────────────────────
//
// CUDA port of src/cpu/l2_normalize.cpp. One thread per spatial position
// (n, h, w); each thread reduces over the C channels and rescales them:
//   Y[n,c,h,w] = X[n,c,h,w] / max(sqrt(sum_c X^2), eps)
// Reduction in double (matches the CPU path). CPU is FP32-only; CUDA adds
// FP16/BF16 (cast at load/store, double reduction). Y OVERWRITTEN; X and Y
// may alias. Inference-only: no backward.

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

constexpr int LN_BLOCK = 256;

inline int ln_grid(long long n) {
    long long blocks = (n + LN_BLOCK - 1) / LN_BLOCK;
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

template <typename T> __device__ inline float ln_load(const T* p);
template <> __device__ inline float ln_load<float>(const float* p) { return *p; }
template <> __device__ inline float ln_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float ln_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void ln_store(T* p, float v);
template <> __device__ inline void ln_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void ln_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void ln_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// One thread per (n, p) where p in [0, H*W). Strided gather over channels.
template <typename T>
__global__ void l2_normalize_nchw_kernel(const T* __restrict__ X,
                                         T* __restrict__ Y,
                                         int N, int C, int HW, double eps) {
    const long long total = (long long)N * HW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int p = static_cast<int>(idx % HW);
        const int n = static_cast<int>(idx / HW);
        const long long base = (long long)n * C * HW + p;
        double ss = 0.0;
        for (int c = 0; c < C; ++c) {
            const double v = ln_load<T>(&X[base + (long long)c * HW]);
            ss += v * v;
        }
        const double inv = 1.0 / fmax(sqrt(ss), eps);
        for (int c = 0; c < C; ++c) {
            const long long off = base + (long long)c * HW;
            ln_store<T>(&Y[off], static_cast<float>(ln_load<T>(&X[off]) * inv));
        }
    }
}

} // namespace

void l2_normalize_nchw_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W,
                               float eps,
                               ::brotensor::Tensor& Y) {
    const char* op = "l2_normalize_nchw_forward";
    check_fp(X, op, "X");
    if (N < 0 || C < 1 || H < 1 || W < 1)
        fail(op, "C/H/W must be >=1 and N >=0");
    if (X.rows != N || X.cols != C * H * W)
        fail(op, "X shape must be (N, C*H*W)");
    if (Y.rows != N || Y.cols != C * H * W || Y.dtype != X.dtype)
        Y.resize(N, C * H * W, X.dtype);
    if (N == 0) return;

    const int HW = H * W;
    const long long total = (long long)N * HW;
    const double epsd = static_cast<double>(eps);
    if (X.dtype == ::brotensor::Dtype::FP16) {
        l2_normalize_nchw_kernel<__half><<<ln_grid(total), LN_BLOCK>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C, HW, epsd);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        l2_normalize_nchw_kernel<__nv_bfloat16><<<ln_grid(total), LN_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<__nv_bfloat16*>(Y.data),
            N, C, HW, epsd);
    } else {
        l2_normalize_nchw_kernel<float><<<ln_grid(total), LN_BLOCK>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C, HW, epsd);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_l2_normalize(::brotensor::detail::OpsVTable& v) {
    v.l2_normalize_nchw_forward = &l2_normalize_nchw_forward;
}

} // namespace brotensor::detail::cuda
