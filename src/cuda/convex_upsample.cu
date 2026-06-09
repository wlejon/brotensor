// ─── CUDA convex (mask-based) upsample, NCHW ────────────────────────────────
//
// CUDA port of src/cpu/convex_upsample.cpp. One thread per OUTPUT element
// (n, c, oy, ox); each thread derives its source low-res pixel (y, x) and
// sub-position (sy, sx), softmaxes the 9 mask logits for that (sy, sx, y, x),
// and blends the 3×3 low-res neighborhood of channel c:
//   Y[n,c,k*y+sy,k*x+sx] = sum_m softmax_m(Mask[n,m,sy,sx,y,x]) * X[n,c,ny,nx]
//   neighbor m: ny=clamp(y-1+m/3), nx=clamp(x-1+m%3)  (replicate pad)
// Mask flat channel = (m*k*k + sy*k + sx). Softmax in double (matches CPU).
//
// The softmax is recomputed per output channel (redundant across C) for kernel
// simplicity — fine at the C/scale this targets. CPU is FP32-only; CUDA adds
// FP16/BF16 (cast at load/store). Y OVERWRITTEN. Inference-only: no backward.

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

constexpr int CU_BLOCK = 256;

inline int cu_grid(long long n) {
    long long blocks = (n + CU_BLOCK - 1) / CU_BLOCK;
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

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename T> __device__ inline float cu_load(const T* p);
template <> __device__ inline float cu_load<float>(const float* p) { return *p; }
template <> __device__ inline float cu_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float cu_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void cu_store(T* p, float v);
template <> __device__ inline void cu_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void cu_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void cu_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__global__ void convex_upsample_kernel(const T* __restrict__ X,
                                       const T* __restrict__ Mask,
                                       T* __restrict__ Y,
                                       int N, int C, int H, int W, int scale) {
    const int HW = H * W;
    const int kk = scale * scale;
    const int oH = scale * H, oW = scale * W;
    const long long oHW = (long long)oH * oW;
    const long long total = (long long)N * C * oHW;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int ox = static_cast<int>(idx % oW);
        long long t = idx / oW;
        const int oy = static_cast<int>(t % oH);
        t /= oH;
        const int c = static_cast<int>(t % C);
        const int n = static_cast<int>(t / C);

        const int y = oy / scale, sy = oy % scale;
        const int x = ox / scale, sx = ox % scale;
        const int sub = sy * scale + sx;
        const int pix = y * W + x;
        const T* m_img = Mask + (long long)n * 9 * kk * HW;

        double mx = -1e300;
        for (int m = 0; m < 9; ++m) {
            const double v = cu_load<T>(&m_img[((long long)m * kk + sub) * HW + pix]);
            if (v > mx) mx = v;
        }
        double w[9]; double sum = 0.0;
        for (int m = 0; m < 9; ++m) {
            const double e = exp(cu_load<T>(&m_img[((long long)m * kk + sub) * HW + pix]) - mx);
            w[m] = e; sum += e;
        }
        const double invs = 1.0 / sum;

        const T* xc = X + ((long long)n * C + c) * HW;
        double acc = 0.0;
        for (int m = 0; m < 9; ++m) {
            const int ny = clampi(y - 1 + m / 3, 0, H - 1);
            const int nx = clampi(x - 1 + m % 3, 0, W - 1);
            acc += (w[m] * invs) * cu_load<T>(&xc[(long long)ny * W + nx]);
        }
        cu_store<T>(&Y[idx], static_cast<float>(acc));
    }
}

} // namespace

void convex_upsample_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Mask,
                             int N, int C, int H, int W, int scale,
                             ::brotensor::Tensor& Y) {
    const char* op = "convex_upsample_forward";
    check_fp(X, op, "X");
    check_fp(Mask, op, "Mask");
    if (X.dtype != Mask.dtype) fail(op, "X and Mask must share dtype");
    if (N < 0 || C < 1 || H < 1 || W < 1) fail(op, "C/H/W must be >=1 and N >=0");
    if (scale < 1) fail(op, "scale must be >=1");
    const int HW = H * W;
    const int kk = scale * scale;
    if (X.rows != N || X.cols != C * HW) fail(op, "X shape must be (N, C*H*W)");
    if (Mask.rows != N || Mask.cols != 9 * kk * HW)
        fail(op, "Mask shape must be (N, 9*scale*scale*H*W)");
    const int oH = scale * H, oW = scale * W;
    const long long oHW = (long long)oH * oW;
    if (Y.rows != N || Y.cols != C * oHW || Y.dtype != X.dtype)
        Y.resize(N, static_cast<int>(C * oHW), X.dtype);
    if (N == 0) return;

    const long long total = (long long)N * C * oHW;
    if (X.dtype == ::brotensor::Dtype::FP16) {
        convex_upsample_kernel<__half><<<cu_grid(total), CU_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(Mask.data),
            static_cast<__half*>(Y.data), N, C, H, W, scale);
    } else if (X.dtype == ::brotensor::Dtype::BF16) {
        convex_upsample_kernel<__nv_bfloat16><<<cu_grid(total), CU_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<const __nv_bfloat16*>(Mask.data),
            static_cast<__nv_bfloat16*>(Y.data), N, C, H, W, scale);
    } else {
        convex_upsample_kernel<float><<<cu_grid(total), CU_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(Mask.data),
            static_cast<float*>(Y.data), N, C, H, W, scale);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_convex_upsample(::brotensor::detail::OpsVTable& v) {
    v.convex_upsample_forward = &convex_upsample_forward;
}

} // namespace brotensor::detail::cuda
