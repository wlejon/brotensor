// ─── CUDA modulated_conv2d (StyleGAN3) ──────────────────────────────────────
//
// CUDA port of src/cpu/modulated_conv2d.cpp. The StyleGAN synthesis-layer core:
// per-sample style modulation of the conv weights, optional demodulation, then
// a standard stride-1 conv per sample.
//
// Realized by looping the batch and reusing the validated CUDA conv2d kernels
// (groups=1) on a per-sample weight — only the weight construction + demod are
// new (small fused reduction kernels). Layouts: X (N,C_in*H*W) NCHW;
// W (C_out, C_in*kH*kW) OIHW; s (N,C_in).
//
//   w'[o,i,kh,kw] = W[o,i,kh,kw] * s[n,i]
//   dcoef[n,o]    = demodulate ? rsqrt(Σ_{i,kh,kw} w'^2 + eps) : 1
//   w''           = w' * dcoef[n,o]
//   Y[n]          = conv2d(X[n], w'', pad, stride=1)
//
// Backward (per n; dw'' = conv2d_backward_weight(X[n],dY[n])):
//   g[o]    = Σ dw''[o,..] * w'[o,..]
//   dw'[o]  = demodulate ? dw''[o]*dcoef - g[o]*dcoef^3*w'[o] : dw''[o]
//   dW[o]  += Σ_n dw'[n,o] * s[n,i]      (accumulate — caller zeros dW)
//   ds[n,i] = Σ_{o,kh,kw} dw'[o,i,kh,kw] * W[o,i,kh,kw]   (overwrite)
//   dX[n]   = conv2d_backward_input(w''[n], dY[n])         (overwrite)
//
// dtype: FP32/FP16/BF16. The modulated weights (w'/w'') and gradients carry the
// storage dtype (so the reused conv2d kernels dispatch correctly); all
// reductions and the dW accumulation run in FP32. The demod coefficient cache
// `dcoef` is always FP32 — it is a small per-(n,o) scalar buffer whose
// precision matters for the rsqrt; the backward consumes it as FP32.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

// runtime.cu / tensor.cu (same namespace). Helper kernels launch on the current
// stream and the FP32 dW accumulator is pooled (cudaMallocAsync/cudaFreeAsync),
// so the per-sample inversion loop stops synchronizing the device on each op.
void* cuda_current_stream();
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

// Reused CUDA conv2d kernels (defined in conv2d.cu, same namespace).
void conv2d_forward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& Wt,
                    const ::brotensor::Tensor* bias,
                    int N, int C_in, int H, int W, int C_out, int kH, int kW,
                    int stride_h, int stride_w, int pad_h, int pad_w,
                    int dil_h, int dil_w, int groups, ::brotensor::Tensor& Y);
void conv2d_backward_input(const ::brotensor::Tensor& Wt,
                           const ::brotensor::Tensor& dY,
                           int N, int C_in, int H, int W,
                           int C_out, int kH, int kW,
                           int stride_h, int stride_w, int pad_h, int pad_w,
                           int dil_h, int dil_w, int groups,
                           ::brotensor::Tensor& dX);
void conv2d_backward_weight(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& dY,
                            int N, int C_in, int H, int W,
                            int C_out, int kH, int kW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dil_h, int dil_w, int groups,
                            ::brotensor::Tensor& dWt);

namespace {

constexpr int MC_BLOCK = 256;

inline void require_fp(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32 &&
        t.dtype != ::brotensor::Dtype::FP16 &&
        t.dtype != ::brotensor::Dtype::BF16) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32/FP16/BF16");
    }
}

inline int grid_for(long long n) {
    long long b = (n + MC_BLOCK - 1) / MC_BLOCK;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return static_cast<int>(b);
}

// Byte-correct row view honouring the tensor's element size.
inline ::brotensor::Tensor row_view(const ::brotensor::Tensor& T, int n, int cols) {
    const size_t elt = static_cast<size_t>(::brotensor::dtype_size_bytes(T.dtype));
    char* base = static_cast<char*>(T.data) + static_cast<size_t>(n) * cols * elt;
    return ::brotensor::Tensor::view(::brotensor::Device::CUDA, base, 1, cols, T.dtype);
}

template <typename T> __device__ inline float mc_load(const T* p);
template <> __device__ inline float mc_load<float>(const float* p) { return *p; }
template <> __device__ inline float mc_load<__half>(const __half* p) { return __half2float(*p); }
template <> __device__ inline float mc_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ inline void mc_store(T* p, float v);
template <> __device__ inline void mc_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void mc_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void mc_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ inline float mc_block_sum(float v, float* sdata) {
    const int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

// Forward: per output channel o (one block), build w' into Wn, reduce its sum
// of squares (FP32), write dcoef[o], then scale by the demod coefficient → w''.
template <typename T>
__global__ void modulate_build_kernel(const T* __restrict__ W,
                                      const T* __restrict__ sn,
                                      int C_out, int khw, int wk,
                                      int demodulate, float eps,
                                      T* __restrict__ Wn,
                                      float* __restrict__ dcoef) {
    extern __shared__ float sdata[];
    const int o = blockIdx.x;
    if (o >= C_out) return;
    const int tid = threadIdx.x;
    const T* Wo  = W  + static_cast<size_t>(o) * wk;
    T*       Wno = Wn + static_cast<size_t>(o) * wk;

    float local = 0.0f;
    for (int col = tid; col < wk; col += blockDim.x) {
        const int i = col / khw;
        const float wp = mc_load<T>(&Wo[col]) * mc_load<T>(&sn[i]);
        mc_store<T>(&Wno[col], wp);
        local += wp * wp;
    }
    const float ss = mc_block_sum(local, sdata);
    const float d = demodulate ? rsqrtf(ss + eps) : 1.0f;
    if (tid == 0) dcoef[o] = d;
    if (demodulate)
        for (int col = tid; col < wk; col += blockDim.x)
            mc_store<T>(&Wno[col], mc_load<T>(&Wno[col]) * d);
}

// Backward: rebuild w' (Wpr) and w'' (Wpp) from W, s, and the cached dcoef.
template <typename T>
__global__ void build_wpr_wpp_kernel(const T* __restrict__ W,
                                     const T* __restrict__ sn,
                                     const float* __restrict__ dcn,
                                     int khw, int wk, long long total,
                                     T* __restrict__ Wpr, T* __restrict__ Wpp) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int o = static_cast<int>(idx / wk);
        const int col = static_cast<int>(idx % wk);
        const int i = col / khw;
        const float wp = mc_load<T>(&W[idx]) * mc_load<T>(&sn[i]);
        mc_store<T>(&Wpr[idx], wp);
        mc_store<T>(&Wpp[idx], wp * dcn[o]);
    }
}

// Backward: dw'' (=dWpp from conv2d_backward_weight) through demod → dw' (dWpr).
template <typename T>
__global__ void demod_dwpr_kernel(const T* __restrict__ dWpp,
                                  const T* __restrict__ Wpr,
                                  const float* __restrict__ dcn,
                                  int C_out, int wk, int demodulate,
                                  T* __restrict__ dWpr) {
    extern __shared__ float sdata[];
    const int o = blockIdx.x;
    if (o >= C_out) return;
    const int tid = threadIdx.x;
    const size_t ob = static_cast<size_t>(o) * wk;
    if (!demodulate) {
        for (int col = tid; col < wk; col += blockDim.x)
            mc_store<T>(&dWpr[ob + col], mc_load<T>(&dWpp[ob + col]));
        return;
    }
    float local = 0.0f;
    for (int col = tid; col < wk; col += blockDim.x)
        local += mc_load<T>(&dWpp[ob + col]) * mc_load<T>(&Wpr[ob + col]);
    const float g = mc_block_sum(local, sdata);
    const float d = dcn[o];
    const float gd3 = g * d * d * d;
    for (int col = tid; col < wk; col += blockDim.x)
        mc_store<T>(&dWpr[ob + col],
                    mc_load<T>(&dWpp[ob + col]) * d - gd3 * mc_load<T>(&Wpr[ob + col]));
}

// Backward: accumulate dW_f32[o,col] += dw'[o,col] * s[n,i] in FP32 (one thread
// per (o,col); sequential across the n-loop, so no atomics needed).
template <typename T>
__global__ void accum_dW_kernel(const T* __restrict__ dWpr,
                                const T* __restrict__ sn,
                                int khw, int wk, long long total,
                                float* __restrict__ dW_f32) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int col = static_cast<int>(idx % wk);
        const int i = col / khw;
        dW_f32[idx] += mc_load<T>(&dWpr[idx]) * mc_load<T>(&sn[i]);
    }
}

// Merge the FP32 dW accumulator into the caller's dW (accumulate — caller zeros).
template <typename T>
__global__ void merge_dW_kernel(const float* __restrict__ dW_f32,
                                T* __restrict__ dW, long long total) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x)
        mc_store<T>(&dW[idx], mc_load<T>(&dW[idx]) + dW_f32[idx]);
}

// Backward: ds[n,i] = Σ_{o,t} dw'[o,i*khw+t] * W[o,i*khw+t]  (overwrite).
// One block per input channel i; reduce over (o, t) in FP32.
template <typename T>
__global__ void ds_kernel(const T* __restrict__ dWpr, const T* __restrict__ W,
                          int C_out, int C_in, int khw, int wk,
                          T* __restrict__ dsn) {
    extern __shared__ float sdata[];
    const int i = blockIdx.x;
    if (i >= C_in) return;
    const int tid = threadIdx.x;
    const long long inner = static_cast<long long>(C_out) * khw;
    float local = 0.0f;
    for (long long k = tid; k < inner; k += blockDim.x) {
        const int o = static_cast<int>(k / khw);
        const int t = static_cast<int>(k % khw);
        const size_t col = static_cast<size_t>(o) * wk + static_cast<size_t>(i) * khw + t;
        local += mc_load<T>(&dWpr[col]) * mc_load<T>(&W[col]);
    }
    const float acc = mc_block_sum(local, sdata);
    if (tid == 0) mc_store<T>(&dsn[i], acc);
}

// ─── templated forward / backward cores ─────────────────────────────────────

template <typename T>
void forward_impl(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W,
                  const ::brotensor::Tensor& s, int N, int C_in, int H, int Wd,
                  int C_out, int kH, int kW, int pad_h, int pad_w,
                  bool demodulate, float eps, int wk, int khw, int out_cols,
                  ::brotensor::Tensor& dcoef, ::brotensor::Tensor& Y) {
    const T* sp = static_cast<const T*>(s.data);
    float* dcp = static_cast<float*>(dcoef.data);
    ::brotensor::Tensor Wn = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA,
                                                           C_out, wk, X.dtype);
    const size_t shmem = MC_BLOCK * sizeof(float);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    for (int n = 0; n < N; ++n) {
        modulate_build_kernel<T><<<C_out, MC_BLOCK, shmem, stream>>>(
            static_cast<const T*>(W.data), sp + static_cast<size_t>(n) * C_in,
            C_out, khw, wk, demodulate ? 1 : 0, eps,
            static_cast<T*>(Wn.data), dcp + static_cast<size_t>(n) * C_out);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        ::brotensor::Tensor Xn = row_view(X, n, C_in * H * Wd);
        ::brotensor::Tensor Yn = row_view(Y, n, out_cols);
        conv2d_forward(Xn, Wn, nullptr, 1, C_in, H, Wd, C_out, kH, kW,
                       1, 1, pad_h, pad_w, 1, 1, 1, Yn);
    }
}

template <typename T>
void backward_impl(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W,
                   const ::brotensor::Tensor& s, const ::brotensor::Tensor& dcoef,
                   const ::brotensor::Tensor& dY, int N, int C_in, int H, int Wd,
                   int C_out, int kH, int kW, int pad_h, int pad_w,
                   bool demodulate, bool want_dW, int wk, int khw, int out_cols,
                   ::brotensor::Tensor& dX, ::brotensor::Tensor& dW, ::brotensor::Tensor& ds) {
    const T* sp = static_cast<const T*>(s.data);
    const float* dcp = static_cast<const float*>(dcoef.data);
    T* dsp = static_cast<T*>(ds.data);

    ::brotensor::Tensor Wpr  = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk, X.dtype);
    ::brotensor::Tensor Wpp  = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk, X.dtype);
    ::brotensor::Tensor dWpp = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk, X.dtype);
    ::brotensor::Tensor dWpr = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk, X.dtype);
    const long long wtotal = static_cast<long long>(C_out) * wk;
    const size_t shmem = MC_BLOCK * sizeof(float);
    const size_t dWpp_bytes = static_cast<size_t>(wtotal) *
                              static_cast<size_t>(::brotensor::dtype_size_bytes(X.dtype));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());

    // FP32 dW accumulator across the whole batch — pooled, stream-ordered.
    // Skipped entirely when the caller doesn't want the weight gradient (an
    // uncommitted dW), which is the common case during latent inversion: it
    // freezes the weights and discards dW. That drops the dW GEMM (accum_dW),
    // the final merge, and the scratch alloc/free for the whole backward.
    float* dW_f32 = nullptr;
    if (want_dW) {
        dW_f32 = static_cast<float*>(cuda_alloc(static_cast<size_t>(wtotal) * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(dW_f32, 0,
                                             static_cast<size_t>(wtotal) * sizeof(float), stream));
    }

    for (int n = 0; n < N; ++n) {
        const T* sn = sp + static_cast<size_t>(n) * C_in;
        const float* dcn = dcp + static_cast<size_t>(n) * C_out;

        build_wpr_wpp_kernel<T><<<grid_for(wtotal), MC_BLOCK, 0, stream>>>(
            static_cast<const T*>(W.data), sn, dcn, khw, wk, wtotal,
            static_cast<T*>(Wpr.data), static_cast<T*>(Wpp.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());

        ::brotensor::Tensor Xn  = row_view(X, n, C_in * H * Wd);
        ::brotensor::Tensor dYn = row_view(dY, n, out_cols);

        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(dWpp.data, 0, dWpp_bytes, stream));
        conv2d_backward_weight(Xn, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                               1, 1, pad_h, pad_w, 1, 1, 1, dWpp);
        ::brotensor::Tensor dXn = row_view(dX, n, C_in * H * Wd);
        conv2d_backward_input(Wpp, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                              1, 1, pad_h, pad_w, 1, 1, 1, dXn);

        demod_dwpr_kernel<T><<<C_out, MC_BLOCK, shmem, stream>>>(
            static_cast<const T*>(dWpp.data), static_cast<const T*>(Wpr.data),
            dcn, C_out, wk, demodulate ? 1 : 0, static_cast<T*>(dWpr.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());

        if (want_dW) {
            accum_dW_kernel<T><<<grid_for(wtotal), MC_BLOCK, 0, stream>>>(
                static_cast<const T*>(dWpr.data), sn, khw, wk, wtotal, dW_f32);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
        }

        ds_kernel<T><<<C_in, MC_BLOCK, shmem, stream>>>(
            static_cast<const T*>(dWpr.data), static_cast<const T*>(W.data),
            C_out, C_in, khw, wk, dsp + static_cast<size_t>(n) * C_in);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    if (want_dW) {
        merge_dW_kernel<T><<<grid_for(wtotal), MC_BLOCK, 0, stream>>>(
            dW_f32, static_cast<T*>(dW.data), wtotal);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(dW_f32);
    }
}

} // namespace

void modulated_conv2d_forward(const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& W,
                              const ::brotensor::Tensor& s,
                              int N, int C_in, int H, int Wd,
                              int C_out, int kH, int kW,
                              int pad_h, int pad_w,
                              bool demodulate, float eps,
                              ::brotensor::Tensor& dcoef,
                              ::brotensor::Tensor& Y) {
    require_fp(X, "modulated_conv2d_forward", "X");
    require_fp(W, "modulated_conv2d_forward", "W");
    require_fp(s, "modulated_conv2d_forward", "s");
    if (W.dtype != X.dtype || s.dtype != X.dtype)
        throw std::runtime_error("modulated_conv2d_forward: W/s dtype must match X");
    const int wk = C_in * kH * kW;
    const int khw = kH * kW;
    if (X.rows != N || X.cols != C_in * H * Wd)
        throw std::runtime_error("modulated_conv2d_forward: X shape mismatch");
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_forward: W shape mismatch");
    if (s.rows != N || s.cols != C_in)
        throw std::runtime_error("modulated_conv2d_forward: s shape mismatch");
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (H_out <= 0 || W_out <= 0)
        throw std::runtime_error("modulated_conv2d_forward: non-positive output shape");
    if (dcoef.rows != N || dcoef.cols != C_out || dcoef.dtype != ::brotensor::Dtype::FP32)
        dcoef.resize(N, C_out, ::brotensor::Dtype::FP32);
    const int out_cols = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != X.dtype)
        Y.resize(N, out_cols, X.dtype);
    if (N == 0 || out_cols == 0) return;

    if (X.dtype == ::brotensor::Dtype::FP16)
        forward_impl<__half>(X, W, s, N, C_in, H, Wd, C_out, kH, kW, pad_h, pad_w,
                             demodulate, eps, wk, khw, out_cols, dcoef, Y);
    else if (X.dtype == ::brotensor::Dtype::BF16)
        forward_impl<__nv_bfloat16>(X, W, s, N, C_in, H, Wd, C_out, kH, kW, pad_h, pad_w,
                                    demodulate, eps, wk, khw, out_cols, dcoef, Y);
    else
        forward_impl<float>(X, W, s, N, C_in, H, Wd, C_out, kH, kW, pad_h, pad_w,
                            demodulate, eps, wk, khw, out_cols, dcoef, Y);
}

void modulated_conv2d_backward(const ::brotensor::Tensor& X,
                               const ::brotensor::Tensor& W,
                               const ::brotensor::Tensor& s,
                               const ::brotensor::Tensor& dcoef,
                               const ::brotensor::Tensor& dY,
                               int N, int C_in, int H, int Wd,
                               int C_out, int kH, int kW,
                               int pad_h, int pad_w, bool demodulate, float eps,
                               ::brotensor::Tensor& dX,
                               ::brotensor::Tensor& dW,
                               ::brotensor::Tensor& ds) {
    require_fp(X, "modulated_conv2d_backward", "X");
    require_fp(W, "modulated_conv2d_backward", "W");
    require_fp(s, "modulated_conv2d_backward", "s");
    require_fp(dY, "modulated_conv2d_backward", "dY");
    if (W.dtype != X.dtype || s.dtype != X.dtype || dY.dtype != X.dtype)
        throw std::runtime_error("modulated_conv2d_backward: W/s/dY dtype must match X");
    if (dcoef.dtype != ::brotensor::Dtype::FP32)
        throw std::runtime_error("modulated_conv2d_backward: dcoef must be FP32");
    (void)eps;  // demod coefficient is precomputed (passed in as dcoef)
    const int wk = C_in * kH * kW;
    const int khw = kH * kW;
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: W shape mismatch");
    // dW is an optional output: an uncommitted (data == nullptr) dW means
    // "skip the weight gradient" — a straight speedup for inversion, which
    // freezes the weights. When committed it must match X's dtype and shape.
    const bool want_dW = (dW.data != nullptr);
    if (want_dW) {
        if (dW.dtype != X.dtype)
            throw std::runtime_error("modulated_conv2d_backward: dW dtype must match X");
        if (dW.rows != C_out || dW.cols != wk)
            throw std::runtime_error("modulated_conv2d_backward: dW shape mismatch");
    }
    if (dX.rows != N || dX.cols != C_in * H * Wd || dX.dtype != X.dtype)
        dX.resize(N, C_in * H * Wd, X.dtype);
    if (ds.rows != N || ds.cols != C_in || ds.dtype != X.dtype)
        ds.resize(N, C_in, X.dtype);
    if (N == 0) return;
    const int out_cols = C_out * H_out * W_out;

    if (X.dtype == ::brotensor::Dtype::FP16)
        backward_impl<__half>(X, W, s, dcoef, dY, N, C_in, H, Wd, C_out, kH, kW,
                              pad_h, pad_w, demodulate, want_dW, wk, khw, out_cols, dX, dW, ds);
    else if (X.dtype == ::brotensor::Dtype::BF16)
        backward_impl<__nv_bfloat16>(X, W, s, dcoef, dY, N, C_in, H, Wd, C_out, kH, kW,
                                     pad_h, pad_w, demodulate, want_dW, wk, khw, out_cols, dX, dW, ds);
    else
        backward_impl<float>(X, W, s, dcoef, dY, N, C_in, H, Wd, C_out, kH, kW,
                             pad_h, pad_w, demodulate, want_dW, wk, khw, out_cols, dX, dW, ds);
}

void fill_cuda_vtable_modulated_conv2d(::brotensor::detail::OpsVTable& v) {
    v.modulated_conv2d_forward  = &modulated_conv2d_forward;
    v.modulated_conv2d_backward = &modulated_conv2d_backward;
}

} // namespace brotensor::detail::cuda
