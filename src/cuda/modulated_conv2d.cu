// ─── CUDA modulated_conv2d (StyleGAN3) ──────────────────────────────────────
//
// CUDA port of src/cpu/modulated_conv2d.cpp. The StyleGAN synthesis-layer core:
// per-sample style modulation of the conv weights, optional demodulation, then
// a standard stride-1 conv per sample. FP32 (matches the CPU reference; the
// reused conv2d kernels carry the heavy lifting).
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

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

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

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32");
    }
}

inline int grid_for(long long n) {
    long long b = (n + MC_BLOCK - 1) / MC_BLOCK;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return static_cast<int>(b);
}

inline ::brotensor::Tensor row_view(const ::brotensor::Tensor& T, int n, int cols) {
    float* base = static_cast<float*>(T.data) + static_cast<size_t>(n) * cols;
    return ::brotensor::Tensor::view(::brotensor::Device::CUDA, base, 1, cols,
                                     ::brotensor::Dtype::FP32);
}

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
// of squares, write dcoef[o], then scale by the demod coefficient → w''.
__global__ void modulate_build_kernel(const float* __restrict__ W,
                                      const float* __restrict__ sn,
                                      int C_out, int khw, int wk,
                                      int demodulate, float eps,
                                      float* __restrict__ Wn,
                                      float* __restrict__ dcoef) {
    extern __shared__ float sdata[];
    const int o = blockIdx.x;
    if (o >= C_out) return;
    const int tid = threadIdx.x;
    const float* Wo = W  + static_cast<size_t>(o) * wk;
    float*       Wno = Wn + static_cast<size_t>(o) * wk;

    float local = 0.0f;
    for (int col = tid; col < wk; col += blockDim.x) {
        const int i = col / khw;
        const float wp = Wo[col] * sn[i];
        Wno[col] = wp;
        local += wp * wp;
    }
    const float ss = mc_block_sum(local, sdata);
    const float d = demodulate ? rsqrtf(ss + eps) : 1.0f;
    if (tid == 0) dcoef[o] = d;
    if (demodulate)
        for (int col = tid; col < wk; col += blockDim.x) Wno[col] *= d;
}

// Backward: rebuild w' (Wpr) and w'' (Wpp) from W, s, and the cached dcoef.
__global__ void build_wpr_wpp_kernel(const float* __restrict__ W,
                                     const float* __restrict__ sn,
                                     const float* __restrict__ dcn,
                                     int khw, int wk, long long total,
                                     float* __restrict__ Wpr,
                                     float* __restrict__ Wpp) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int o = static_cast<int>(idx / wk);
        const int col = static_cast<int>(idx % wk);
        const int i = col / khw;
        const float wp = W[idx] * sn[i];
        Wpr[idx] = wp;
        Wpp[idx] = wp * dcn[o];
    }
}

// Backward: dw'' (=dWpp from conv2d_backward_weight) through demod → dw' (dWpr).
__global__ void demod_dwpr_kernel(const float* __restrict__ dWpp,
                                  const float* __restrict__ Wpr,
                                  const float* __restrict__ dcn,
                                  int C_out, int wk, int demodulate,
                                  float* __restrict__ dWpr) {
    extern __shared__ float sdata[];
    const int o = blockIdx.x;
    if (o >= C_out) return;
    const int tid = threadIdx.x;
    const size_t ob = static_cast<size_t>(o) * wk;
    if (!demodulate) {
        for (int col = tid; col < wk; col += blockDim.x) dWpr[ob + col] = dWpp[ob + col];
        return;
    }
    float local = 0.0f;
    for (int col = tid; col < wk; col += blockDim.x)
        local += dWpp[ob + col] * Wpr[ob + col];
    const float g = mc_block_sum(local, sdata);
    const float d = dcn[o];
    const float gd3 = g * d * d * d;
    for (int col = tid; col < wk; col += blockDim.x)
        dWpr[ob + col] = dWpp[ob + col] * d - gd3 * Wpr[ob + col];
}

// Backward: accumulate dW[o,col] += dw'[o,col] * s[n,i] (caller zeroed dW).
__global__ void accum_dW_kernel(const float* __restrict__ dWpr,
                                const float* __restrict__ sn,
                                int khw, int wk, long long total,
                                float* __restrict__ dW) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total; idx += (long long)blockDim.x * gridDim.x) {
        const int col = static_cast<int>(idx % wk);
        const int i = col / khw;
        dW[idx] += dWpr[idx] * sn[i];
    }
}

// Backward: ds[n,i] = Σ_{o,t} dw'[o,i*khw+t] * W[o,i*khw+t]  (overwrite).
// One block per input channel i; reduce over (o, t).
__global__ void ds_kernel(const float* __restrict__ dWpr,
                          const float* __restrict__ W,
                          int C_out, int C_in, int khw, int wk,
                          float* __restrict__ dsn) {
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
        local += dWpr[col] * W[col];
    }
    const float acc = mc_block_sum(local, sdata);
    if (tid == 0) dsn[i] = acc;
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
    check_fp32(X, "modulated_conv2d_forward", "X");
    check_fp32(W, "modulated_conv2d_forward", "W");
    check_fp32(s, "modulated_conv2d_forward", "s");
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
    if (Y.rows != N || Y.cols != out_cols || Y.dtype != ::brotensor::Dtype::FP32)
        Y.resize(N, out_cols, ::brotensor::Dtype::FP32);
    if (N == 0 || out_cols == 0) return;

    const float* sp = static_cast<const float*>(s.data);
    float* dcp = static_cast<float*>(dcoef.data);
    ::brotensor::Tensor Wn = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk);
    const size_t shmem = MC_BLOCK * sizeof(float);

    for (int n = 0; n < N; ++n) {
        modulate_build_kernel<<<C_out, MC_BLOCK, shmem>>>(
            static_cast<const float*>(W.data), sp + static_cast<size_t>(n) * C_in,
            C_out, khw, wk, demodulate ? 1 : 0, eps,
            static_cast<float*>(Wn.data), dcp + static_cast<size_t>(n) * C_out);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());

        ::brotensor::Tensor Xn = row_view(X, n, C_in * H * Wd);
        ::brotensor::Tensor Yn = row_view(Y, n, out_cols);
        conv2d_forward(Xn, Wn, nullptr, 1, C_in, H, Wd, C_out, kH, kW,
                       1, 1, pad_h, pad_w, 1, 1, 1, Yn);
    }
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
    check_fp32(X, "modulated_conv2d_backward", "X");
    check_fp32(W, "modulated_conv2d_backward", "W");
    check_fp32(s, "modulated_conv2d_backward", "s");
    check_fp32(dcoef, "modulated_conv2d_backward", "dcoef");
    check_fp32(dY, "modulated_conv2d_backward", "dY");
    (void)eps;  // demod coefficient is precomputed (passed in as dcoef)
    const int wk = C_in * kH * kW;
    const int khw = kH * kW;
    const int H_out = H + 2 * pad_h - (kH - 1);
    const int W_out = Wd + 2 * pad_w - (kW - 1);
    if (W.rows != C_out || W.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: W shape mismatch");
    if (dW.rows != C_out || dW.cols != wk)
        throw std::runtime_error("modulated_conv2d_backward: dW shape mismatch");
    if (dX.rows != N || dX.cols != C_in * H * Wd || dX.dtype != ::brotensor::Dtype::FP32)
        dX.resize(N, C_in * H * Wd, ::brotensor::Dtype::FP32);
    if (ds.rows != N || ds.cols != C_in || ds.dtype != ::brotensor::Dtype::FP32)
        ds.resize(N, C_in, ::brotensor::Dtype::FP32);
    if (N == 0) return;

    const int out_cols = C_out * H_out * W_out;
    (void)out_cols;
    const float* sp = static_cast<const float*>(s.data);
    const float* dcp = static_cast<const float*>(dcoef.data);
    float* dsp = static_cast<float*>(ds.data);

    ::brotensor::Tensor Wpr  = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk); // w'
    ::brotensor::Tensor Wpp  = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk); // w''
    ::brotensor::Tensor dWpp = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk);
    ::brotensor::Tensor dWpr = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, C_out, wk);
    const long long wtotal = static_cast<long long>(C_out) * wk;
    const size_t shmem = MC_BLOCK * sizeof(float);

    for (int n = 0; n < N; ++n) {
        const float* sn  = sp  + static_cast<size_t>(n) * C_in;
        const float* dcn = dcp + static_cast<size_t>(n) * C_out;

        build_wpr_wpp_kernel<<<grid_for(wtotal), MC_BLOCK>>>(
            static_cast<const float*>(W.data), sn, dcn, khw, wk, wtotal,
            static_cast<float*>(Wpr.data), static_cast<float*>(Wpp.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());

        ::brotensor::Tensor Xn  = row_view(X, n, C_in * H * Wd);
        ::brotensor::Tensor dYn = row_view(dY, n, out_cols);

        // dw'' (conv2d_backward_weight accumulates → zero first), and dX[n].
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(dWpp.data, 0,
                                             static_cast<size_t>(wtotal) * sizeof(float)));
        conv2d_backward_weight(Xn, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                               1, 1, pad_h, pad_w, 1, 1, 1, dWpp);
        ::brotensor::Tensor dXn = row_view(dX, n, C_in * H * Wd);
        conv2d_backward_input(Wpp, dYn, 1, C_in, H, Wd, C_out, kH, kW,
                              1, 1, pad_h, pad_w, 1, 1, 1, dXn);

        // Through demod → dw'.
        demod_dwpr_kernel<<<C_out, MC_BLOCK, shmem>>>(
            static_cast<const float*>(dWpp.data), static_cast<const float*>(Wpr.data),
            dcn, C_out, wk, demodulate ? 1 : 0, static_cast<float*>(dWpr.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());

        // Accumulate dW and write ds[n].
        accum_dW_kernel<<<grid_for(wtotal), MC_BLOCK>>>(
            static_cast<const float*>(dWpr.data), sn, khw, wk, wtotal,
            static_cast<float*>(dW.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());

        ds_kernel<<<C_in, MC_BLOCK, shmem>>>(
            static_cast<const float*>(dWpr.data), static_cast<const float*>(W.data),
            C_out, C_in, khw, wk, dsp + static_cast<size_t>(n) * C_in);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

void fill_cuda_vtable_modulated_conv2d(::brotensor::detail::OpsVTable& v) {
    v.modulated_conv2d_forward  = &modulated_conv2d_forward;
    v.modulated_conv2d_backward = &modulated_conv2d_backward;
}

} // namespace brotensor::detail::cuda
