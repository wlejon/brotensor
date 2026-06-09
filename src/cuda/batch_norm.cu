// CUDA BatchNorm (NCHW, FP32-only — matches CPU contract).
//
// Three slots, all reducing over (N, H, W) per channel:
//
//   batch_norm_forward    — training; computes per-channel batch mean/var,
//                           writes Y, updates running_mean / running_var
//                           in-place using PyTorch convention
//                             running = (1 - momentum) * running
//                                       + momentum     * batch_stat
//                           (running_var uses the unbiased estimator; forward
//                           Y uses the biased estimator). Saves batch mean and
//                           rstd for the backward pass.
//
//   batch_norm_inference  — applies (gamma * (x - running_mean) /
//                           sqrt(running_var + eps) + beta); no state mutation.
//
//   batch_norm_backward   — caller zeros dX/dGamma/dBeta; op overwrites dX and
//                           accumulates into dGamma / dBeta.
//
// Kernel layout: one block per channel; threads cooperate over the M = N*H*W
// elements of that channel scattered through the NCHW tensor. Two passes per
// block (sum + write Y for forward; reduce + write dX for backward) over an
// inherently strided iteration (n stride = C*spatial; spatial stride = 1).

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Current CUDA stream for hot-op launches — so kernels join a non-default
// capture/replay stream instead of silently landing on the default stream.
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace {

constexpr int BN_BLOCK = 256;

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32");
    }
}

inline void check_per_channel(const ::brotensor::Tensor& t,
                              int C, const char* op, const char* name) {
    if (t.size() != C) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must have C elements");
    }
}

// One block per channel `c`. Threads iterate over the M = N*spatial elements
// of channel `c`, addressed as X[(n*C + c)*spatial + s].
__global__ void bn_forward_kernel(const float* __restrict__ X,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ running_mean,
                                  float* __restrict__ running_var,
                                  float* __restrict__ Y,
                                  float* __restrict__ saved_mean,
                                  float* __restrict__ saved_rstd,
                                  int N, int C, int spatial,
                                  float eps, float momentum) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int M = N * spatial;

    // Pass 1: sum + sumsq over channel `c`.
    float lsum = 0.0f, lsumsq = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* x_chan = X + (n * C + c) * spatial;
        for (int s = tid; s < spatial; s += blockDim.x) {
            const float v = x_chan[s];
            lsum   += v;
            lsumsq += v * v;
        }
    }
    __shared__ float s_sum[BN_BLOCK];
    __shared__ float s_sumsq[BN_BLOCK];
    s_sum[tid]   = lsum;
    s_sumsq[tid] = lsumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_mean, s_rstd, s_gv, s_bv;
    if (tid == 0) {
        const float inv_M = 1.0f / static_cast<float>(M);
        const float mean  = s_sum[0]   * inv_M;
        const float var_b = s_sumsq[0] * inv_M - mean * mean;  // biased
        const float rstd  = rsqrtf(var_b + eps);
        const float bessel = (M > 1)
            ? static_cast<float>(M) / static_cast<float>(M - 1)
            : 1.0f;
        const float var_unb = var_b * bessel;
        s_mean = mean;
        s_rstd = rstd;
        s_gv   = gamma[c];
        s_bv   = beta[c];

        // Save + update running stats (one writer per channel — no race).
        saved_mean[c] = mean;
        saved_rstd[c] = rstd;
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var_unb;
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;
    const float gv   = s_gv;
    const float bv   = s_bv;

    // Pass 2: Y = (x - mean) * rstd * gamma + beta.
    for (int n = 0; n < N; ++n) {
        const float* x_chan = X + (n * C + c) * spatial;
        float*       y_chan = Y + (n * C + c) * spatial;
        for (int s = tid; s < spatial; s += blockDim.x) {
            y_chan[s] = (x_chan[s] - mean) * rstd * gv + bv;
        }
    }
}

__global__ void bn_inference_kernel(const float* __restrict__ X,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ running_mean,
                                    const float* __restrict__ running_var,
                                    float* __restrict__ Y,
                                    int N, int C, int spatial, float eps) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const float inv = rsqrtf(running_var[c] + eps);
    const float a = gamma[c] * inv;
    const float b = beta[c] - running_mean[c] * a;
    for (int n = 0; n < N; ++n) {
        const float* x_chan = X + (n * C + c) * spatial;
        float*       y_chan = Y + (n * C + c) * spatial;
        for (int s = tid; s < spatial; s += blockDim.x) {
            y_chan[s] = x_chan[s] * a + b;
        }
    }
}

// One block per channel `c`. Two reduction passes (sum_dY + sum_dY_xh), then
// the dX write pass. dGamma / dBeta are accumulated (caller-zeroed) — single
// writer per channel, so no atomics needed.
__global__ void bn_backward_kernel(const float* __restrict__ X,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ saved_mean,
                                   const float* __restrict__ saved_rstd,
                                   const float* __restrict__ dY,
                                   float* __restrict__ dX,
                                   float* __restrict__ dGamma,
                                   float* __restrict__ dBeta,
                                   int N, int C, int spatial) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int M = N * spatial;
    const float mean = saved_mean[c];
    const float rstd = saved_rstd[c];
    const float gv   = gamma[c];

    float lsum_dY = 0.0f, lsum_dY_xh = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* x_chan  = X  + (n * C + c) * spatial;
        const float* dy_chan = dY + (n * C + c) * spatial;
        for (int s = tid; s < spatial; s += blockDim.x) {
            const float xh = (x_chan[s] - mean) * rstd;
            lsum_dY    += dy_chan[s];
            lsum_dY_xh += dy_chan[s] * xh;
        }
    }
    __shared__ float s_a[BN_BLOCK];
    __shared__ float s_b[BN_BLOCK];
    s_a[tid] = lsum_dY;
    s_b[tid] = lsum_dY_xh;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_a[tid] += s_a[tid + stride];
            s_b[tid] += s_b[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_sum_dY, s_sum_dY_xh;
    if (tid == 0) {
        s_sum_dY    = s_a[0];
        s_sum_dY_xh = s_b[0];
        dGamma[c] += s_b[0];   // accumulate
        dBeta[c]  += s_a[0];   // accumulate
    }
    __syncthreads();
    const float sum_dY    = s_sum_dY;
    const float sum_dY_xh = s_sum_dY_xh;
    const float sum1 = gv * sum_dY;
    const float sum2 = gv * sum_dY_xh;
    const float inv_M = 1.0f / static_cast<float>(M);

    for (int n = 0; n < N; ++n) {
        const float* x_chan  = X  + (n * C + c) * spatial;
        const float* dy_chan = dY + (n * C + c) * spatial;
        float*       dx_chan = dX + (n * C + c) * spatial;
        for (int s = tid; s < spatial; s += blockDim.x) {
            const float xh  = (x_chan[s] - mean) * rstd;
            const float dxh = dy_chan[s] * gv;
            dx_chan[s] = rstd * (dxh - (sum1 + xh * sum2) * inv_M);
        }
    }
}

} // namespace

void batch_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        ::brotensor::Tensor& running_mean,
                        ::brotensor::Tensor& running_var,
                        int N, int C, int H, int W,
                        float eps, float momentum,
                        ::brotensor::Tensor& Y,
                        ::brotensor::Tensor& saved_mean,
                        ::brotensor::Tensor& saved_rstd) {
    using ::brotensor::Dtype;
    check_fp32(X,            "batch_norm_forward", "X");
    check_fp32(gamma,        "batch_norm_forward", "gamma");
    check_fp32(beta,         "batch_norm_forward", "beta");
    check_fp32(running_mean, "batch_norm_forward", "running_mean");
    check_fp32(running_var,  "batch_norm_forward", "running_var");
    check_per_channel(gamma,        C, "batch_norm_forward", "gamma");
    check_per_channel(beta,         C, "batch_norm_forward", "beta");
    check_per_channel(running_mean, C, "batch_norm_forward", "running_mean");
    check_per_channel(running_var,  C, "batch_norm_forward", "running_var");

    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (saved_mean.rows != C || saved_mean.cols != 1 ||
        saved_mean.dtype != Dtype::FP32) {
        saved_mean.resize(C, 1, Dtype::FP32);
    }
    if (saved_rstd.rows != C || saved_rstd.cols != 1 ||
        saved_rstd.dtype != Dtype::FP32) {
        saved_rstd.resize(C, 1, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    bn_forward_kernel<<<C, BN_BLOCK, 0, cur_stream()>>>(
        reinterpret_cast<const float*>(X.data),
        reinterpret_cast<const float*>(gamma.data),
        reinterpret_cast<const float*>(beta.data),
        reinterpret_cast<float*>(running_mean.data),
        reinterpret_cast<float*>(running_var.data),
        reinterpret_cast<float*>(Y.data),
        reinterpret_cast<float*>(saved_mean.data),
        reinterpret_cast<float*>(saved_rstd.data),
        N, C, spatial, eps, momentum);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void batch_norm_inference(const ::brotensor::Tensor& X,
                          const ::brotensor::Tensor& gamma,
                          const ::brotensor::Tensor& beta,
                          const ::brotensor::Tensor& running_mean,
                          const ::brotensor::Tensor& running_var,
                          int N, int C, int H, int W,
                          float eps,
                          ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    check_fp32(X,            "batch_norm_inference", "X");
    check_fp32(gamma,        "batch_norm_inference", "gamma");
    check_fp32(beta,         "batch_norm_inference", "beta");
    check_fp32(running_mean, "batch_norm_inference", "running_mean");
    check_fp32(running_var,  "batch_norm_inference", "running_var");
    check_per_channel(gamma,        C, "batch_norm_inference", "gamma");
    check_per_channel(beta,         C, "batch_norm_inference", "beta");
    check_per_channel(running_mean, C, "batch_norm_inference", "running_mean");
    check_per_channel(running_var,  C, "batch_norm_inference", "running_var");

    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    bn_inference_kernel<<<C, BN_BLOCK, 0, cur_stream()>>>(
        reinterpret_cast<const float*>(X.data),
        reinterpret_cast<const float*>(gamma.data),
        reinterpret_cast<const float*>(beta.data),
        reinterpret_cast<const float*>(running_mean.data),
        reinterpret_cast<const float*>(running_var.data),
        reinterpret_cast<float*>(Y.data),
        N, C, spatial, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void batch_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& saved_mean,
                         const ::brotensor::Tensor& saved_rstd,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W,
                         ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta) {
    using ::brotensor::Dtype;
    check_fp32(X,          "batch_norm_backward", "X");
    check_fp32(gamma,      "batch_norm_backward", "gamma");
    check_fp32(saved_mean, "batch_norm_backward", "saved_mean");
    check_fp32(saved_rstd, "batch_norm_backward", "saved_rstd");
    check_fp32(dY,         "batch_norm_backward", "dY");
    check_fp32(dGamma,     "batch_norm_backward", "dGamma");
    check_fp32(dBeta,      "batch_norm_backward", "dBeta");
    check_per_channel(gamma,      C, "batch_norm_backward", "gamma");
    check_per_channel(saved_mean, C, "batch_norm_backward", "saved_mean");
    check_per_channel(saved_rstd, C, "batch_norm_backward", "saved_rstd");
    if (dGamma.rows != C || dGamma.cols != 1 ||
        dBeta.rows  != C || dBeta.cols  != 1) {
        throw std::runtime_error(
            "brotensor: batch_norm_backward: dGamma/dBeta must be (C,1)");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (dY.rows != N || dY.cols != cols) {
        throw std::runtime_error("brotensor: batch_norm_backward: dY shape mismatch");
    }
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("brotensor: batch_norm_backward: X shape mismatch");
    }
    if (dX.rows != N || dX.cols != cols || dX.dtype != Dtype::FP32) {
        dX.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    bn_backward_kernel<<<C, BN_BLOCK, 0, cur_stream()>>>(
        reinterpret_cast<const float*>(X.data),
        reinterpret_cast<const float*>(gamma.data),
        reinterpret_cast<const float*>(saved_mean.data),
        reinterpret_cast<const float*>(saved_rstd.data),
        reinterpret_cast<const float*>(dY.data),
        reinterpret_cast<float*>(dX.data),
        reinterpret_cast<float*>(dGamma.data),
        reinterpret_cast<float*>(dBeta.data),
        N, C, spatial);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_batch_norm(::brotensor::detail::OpsVTable& v) {
    v.batch_norm_forward   = &batch_norm_forward;
    v.batch_norm_inference = &batch_norm_inference;
    v.batch_norm_backward  = &batch_norm_backward;
}

} // namespace brotensor::detail::cuda
