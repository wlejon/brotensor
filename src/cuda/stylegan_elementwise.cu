// ─── CUDA StyleGAN3 synthesis-input primitives ─────────────────────────────
//
// CUDA port of src/cpu/stylegan_elementwise.cpp:
//   sin / cos     y = sin/cos(x);   dX = dY*cos(x) / -dY*sin(x)
//   rsqrt         y = 1/sqrt(x);    dX = -0.5*dY*y^3   (backward reads y)
//   pixel_norm    per-row RMS-over-channel normalise; backward vs the
//                 same closed form as the CPU reference.
//
// sin/cos/rsqrt are elementwise; their backward OVERWRITES dX (no learnable
// parameters, x/y and dX/dY may alias). pixel_norm operates per row over the
// trailing (cols) axis — one block per row, block-reduction over the columns.
//
// rsqrt: the caller owns the x > 0 precondition (no guard — matching the CPU
// backend and log/exp), so the IEEE result for x<=0 surfaces loudly.
//
// CPU is FP32-only (per CLAUDE.md); CUDA additionally supports FP16/BF16 with
// math performed in FP32, casting at the load/store boundary.

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

constexpr int SG_BLOCK = 256;

inline int sg_grid(long long n) {
    long long blocks = (n + SG_BLOCK - 1) / SG_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

inline void require_fp(const char* op, const ::brotensor::Tensor& t,
                       const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

// ── load/store casts (mirror log_exp_round.cu) ──
template <typename T> __device__ inline float sg_load(const T* p);
template <> __device__ inline float sg_load<float>(const float* p) { return *p; }
template <> __device__ inline float sg_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float sg_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void sg_store(T* p, float v);
template <> __device__ inline void sg_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void sg_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void sg_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// ─── elementwise kernels ────────────────────────────────────────────────────

template <typename T>
__global__ void sin_forward_kernel(const T* __restrict__ x, long long n,
                                   T* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x)
        sg_store<T>(&y[i], sinf(sg_load<T>(&x[i])));
}

template <typename T>
__global__ void sin_backward_kernel(const T* __restrict__ x,
                                    const T* __restrict__ dY, long long n,
                                    T* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float xv = sg_load<T>(&x[i]);
        const float gv = sg_load<T>(&dY[i]);   // read both before write (alias)
        sg_store<T>(&dX[i], gv * cosf(xv));
    }
}

template <typename T>
__global__ void cos_forward_kernel(const T* __restrict__ x, long long n,
                                   T* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x)
        sg_store<T>(&y[i], cosf(sg_load<T>(&x[i])));
}

template <typename T>
__global__ void cos_backward_kernel(const T* __restrict__ x,
                                    const T* __restrict__ dY, long long n,
                                    T* __restrict__ dX) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float xv = sg_load<T>(&x[i]);
        const float gv = sg_load<T>(&dY[i]);
        sg_store<T>(&dX[i], -gv * sinf(xv));
    }
}

template <typename T>
__global__ void rsqrt_forward_kernel(const T* __restrict__ x, long long n,
                                     T* __restrict__ y) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x)
        sg_store<T>(&y[i], rsqrtf(sg_load<T>(&x[i])));
}

template <typename T>
__global__ void rsqrt_backward_kernel(const T* __restrict__ y,
                                      const T* __restrict__ dY, long long n,
                                      T* __restrict__ dX) {
    // y = x^{-1/2} ⇒ dy/dx = -1/2 y^3.  backward reads the OUTPUT y.
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float yv = sg_load<T>(&y[i]);
        const float gv = sg_load<T>(&dY[i]);
        sg_store<T>(&dX[i], -0.5f * gv * yv * yv * yv);
    }
}

// ─── pixel_norm kernels (one block per row) ─────────────────────────────────

__device__ inline float pn_block_sum(float v, float* sdata) {
    const int tid = threadIdx.x;
    __syncthreads();             // safe to call before reuse of sdata
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

template <typename T>
__global__ void pixel_norm_forward_kernel(const T* __restrict__ X, int C,
                                          float eps, T* __restrict__ Y) {
    extern __shared__ float sdata[];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const T* xr = X + static_cast<size_t>(r) * C;
    T*       yr = Y + static_cast<size_t>(r) * C;

    float local = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        const float v = sg_load<T>(&xr[c]);
        local += v * v;
    }
    const float ss = pn_block_sum(local, sdata);
    const float rinv = rsqrtf(ss / static_cast<float>(C) + eps);
    for (int c = tid; c < C; c += blockDim.x)
        sg_store<T>(&yr[c], sg_load<T>(&xr[c]) * rinv);
}

template <typename T>
__global__ void pixel_norm_backward_kernel(const T* __restrict__ X,
                                           const T* __restrict__ dY, int C,
                                           float eps, T* __restrict__ dX) {
    extern __shared__ float sdata[];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const T* xr  = X  + static_cast<size_t>(r) * C;
    const T* dyr = dY + static_cast<size_t>(r) * C;
    T*       dxr = dX + static_cast<size_t>(r) * C;
    const float invC = 1.0f / static_cast<float>(C);

    float l_ss = 0.0f, l_s = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        const float xv = sg_load<T>(&xr[c]);
        const float dv = sg_load<T>(&dyr[c]);
        l_ss += xv * xv;
        l_s  += dv * xv;
    }
    const float ss = pn_block_sum(l_ss, sdata);
    const float s  = pn_block_sum(l_s,  sdata);
    const float rinv = rsqrtf(ss * invC + eps);
    const float r3s  = rinv * rinv * rinv * s * invC;
    for (int c = tid; c < C; c += blockDim.x) {
        const float xv = sg_load<T>(&xr[c]);
        const float dv = sg_load<T>(&dyr[c]);
        sg_store<T>(&dxr[c], rinv * dv - r3s * xv);
    }
}

// ── launch helpers ──

template <typename FwdLaunch>
inline void run_unary_fwd(const char* op, const ::brotensor::Tensor& x,
                          ::brotensor::Tensor& y, FwdLaunch launch) {
    require_fp(op, x, "x");
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype)
        y.resize(x.rows, x.cols, x.dtype);
    const long long n = x.size();
    if (n == 0) return;
    launch(n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ─── sin ─────────────────────────────────────────────────────────────────────

void sin_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    run_unary_fwd("sin_forward", x, y, [&](long long n) {
        if (x.dtype == ::brotensor::Dtype::FP16)
            sin_forward_kernel<__half><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const __half*>(x.data), n, static_cast<__half*>(y.data));
        else if (x.dtype == ::brotensor::Dtype::BF16)
            sin_forward_kernel<__nv_bfloat16><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const __nv_bfloat16*>(x.data), n, static_cast<__nv_bfloat16*>(y.data));
        else
            sin_forward_kernel<float><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    });
}

void sin_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    require_fp("sin_backward", x, "x");
    require_fp("sin_backward", dY, "dY");
    if (dY.dtype != x.dtype)
        throw std::runtime_error("brotensor: sin_backward: dY.dtype must match x.dtype");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype)
        dX.resize(x.rows, x.cols, x.dtype);
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16)
        sin_backward_kernel<__half><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data),
            n, static_cast<__half*>(dX.data));
    else if (x.dtype == ::brotensor::Dtype::BF16)
        sin_backward_kernel<__nv_bfloat16><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data),
            n, static_cast<__nv_bfloat16*>(dX.data));
    else
        sin_backward_kernel<float><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(dY.data),
            n, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── cos ─────────────────────────────────────────────────────────────────────

void cos_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    run_unary_fwd("cos_forward", x, y, [&](long long n) {
        if (x.dtype == ::brotensor::Dtype::FP16)
            cos_forward_kernel<__half><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const __half*>(x.data), n, static_cast<__half*>(y.data));
        else if (x.dtype == ::brotensor::Dtype::BF16)
            cos_forward_kernel<__nv_bfloat16><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const __nv_bfloat16*>(x.data), n, static_cast<__nv_bfloat16*>(y.data));
        else
            cos_forward_kernel<float><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    });
}

void cos_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                  ::brotensor::Tensor& dX) {
    require_fp("cos_backward", x, "x");
    require_fp("cos_backward", dY, "dY");
    if (dY.dtype != x.dtype)
        throw std::runtime_error("brotensor: cos_backward: dY.dtype must match x.dtype");
    if (dX.rows != x.rows || dX.cols != x.cols || dX.dtype != x.dtype)
        dX.resize(x.rows, x.cols, x.dtype);
    const long long n = x.size();
    if (n == 0) return;
    if (x.dtype == ::brotensor::Dtype::FP16)
        cos_backward_kernel<__half><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const __half*>(x.data), static_cast<const __half*>(dY.data),
            n, static_cast<__half*>(dX.data));
    else if (x.dtype == ::brotensor::Dtype::BF16)
        cos_backward_kernel<__nv_bfloat16><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(dY.data),
            n, static_cast<__nv_bfloat16*>(dX.data));
    else
        cos_backward_kernel<float><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const float*>(x.data), static_cast<const float*>(dY.data),
            n, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── rsqrt ───────────────────────────────────────────────────────────────────

void rsqrt_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    run_unary_fwd("rsqrt_forward", x, y, [&](long long n) {
        if (x.dtype == ::brotensor::Dtype::FP16)
            rsqrt_forward_kernel<__half><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const __half*>(x.data), n, static_cast<__half*>(y.data));
        else if (x.dtype == ::brotensor::Dtype::BF16)
            rsqrt_forward_kernel<__nv_bfloat16><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const __nv_bfloat16*>(x.data), n, static_cast<__nv_bfloat16*>(y.data));
        else
            rsqrt_forward_kernel<float><<<sg_grid(n), SG_BLOCK>>>(
                static_cast<const float*>(x.data), n, static_cast<float*>(y.data));
    });
}

void rsqrt_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                    ::brotensor::Tensor& dX) {
    require_fp("rsqrt_backward", y, "y");
    require_fp("rsqrt_backward", dY, "dY");
    if (dY.dtype != y.dtype)
        throw std::runtime_error("brotensor: rsqrt_backward: dY.dtype must match y.dtype");
    if (dX.rows != y.rows || dX.cols != y.cols || dX.dtype != y.dtype)
        dX.resize(y.rows, y.cols, y.dtype);
    const long long n = y.size();
    if (n == 0) return;
    if (y.dtype == ::brotensor::Dtype::FP16)
        rsqrt_backward_kernel<__half><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const __half*>(y.data), static_cast<const __half*>(dY.data),
            n, static_cast<__half*>(dX.data));
    else if (y.dtype == ::brotensor::Dtype::BF16)
        rsqrt_backward_kernel<__nv_bfloat16><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const __nv_bfloat16*>(y.data), static_cast<const __nv_bfloat16*>(dY.data),
            n, static_cast<__nv_bfloat16*>(dX.data));
    else
        rsqrt_backward_kernel<float><<<sg_grid(n), SG_BLOCK>>>(
            static_cast<const float*>(y.data), static_cast<const float*>(dY.data),
            n, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── pixel_norm ──────────────────────────────────────────────────────────────

void pixel_norm_forward(const ::brotensor::Tensor& X, float eps,
                        ::brotensor::Tensor& Y) {
    require_fp("pixel_norm_forward", X, "X");
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype)
        Y.resize(X.rows, X.cols, X.dtype);
    const int R = X.rows, C = X.cols;
    if (R == 0 || C == 0) return;
    const size_t shmem = SG_BLOCK * sizeof(float);
    if (X.dtype == ::brotensor::Dtype::FP16)
        pixel_norm_forward_kernel<__half><<<R, SG_BLOCK, shmem>>>(
            static_cast<const __half*>(X.data), C, eps, static_cast<__half*>(Y.data));
    else if (X.dtype == ::brotensor::Dtype::BF16)
        pixel_norm_forward_kernel<__nv_bfloat16><<<R, SG_BLOCK, shmem>>>(
            static_cast<const __nv_bfloat16*>(X.data), C, eps, static_cast<__nv_bfloat16*>(Y.data));
    else
        pixel_norm_forward_kernel<float><<<R, SG_BLOCK, shmem>>>(
            static_cast<const float*>(X.data), C, eps, static_cast<float*>(Y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void pixel_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& dY, float eps,
                         ::brotensor::Tensor& dX) {
    require_fp("pixel_norm_backward", X, "X");
    require_fp("pixel_norm_backward", dY, "dY");
    if (dY.dtype != X.dtype)
        throw std::runtime_error("brotensor: pixel_norm_backward: dY.dtype must match X.dtype");
    if (dX.rows != X.rows || dX.cols != X.cols || dX.dtype != X.dtype)
        dX.resize(X.rows, X.cols, X.dtype);
    const int R = X.rows, C = X.cols;
    if (R == 0 || C == 0) return;
    const size_t shmem = SG_BLOCK * sizeof(float);
    if (X.dtype == ::brotensor::Dtype::FP16)
        pixel_norm_backward_kernel<__half><<<R, SG_BLOCK, shmem>>>(
            static_cast<const __half*>(X.data), static_cast<const __half*>(dY.data),
            C, eps, static_cast<__half*>(dX.data));
    else if (X.dtype == ::brotensor::Dtype::BF16)
        pixel_norm_backward_kernel<__nv_bfloat16><<<R, SG_BLOCK, shmem>>>(
            static_cast<const __nv_bfloat16*>(X.data), static_cast<const __nv_bfloat16*>(dY.data),
            C, eps, static_cast<__nv_bfloat16*>(dX.data));
    else
        pixel_norm_backward_kernel<float><<<R, SG_BLOCK, shmem>>>(
            static_cast<const float*>(X.data), static_cast<const float*>(dY.data),
            C, eps, static_cast<float*>(dX.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_stylegan_elementwise(::brotensor::detail::OpsVTable& v) {
    v.sin_forward          = &sin_forward;
    v.sin_backward         = &sin_backward;
    v.cos_forward          = &cos_forward;
    v.cos_backward         = &cos_backward;
    v.rsqrt_forward        = &rsqrt_forward;
    v.rsqrt_backward       = &rsqrt_backward;
    v.pixel_norm_forward   = &pixel_norm_forward;
    v.pixel_norm_backward  = &pixel_norm_backward;
}

} // namespace brotensor::detail::cuda
