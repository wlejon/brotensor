#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor::detail::cuda {

// NOTE on signature: per Subagent 1's spec, mean_out/rstd_out are host-side
// floats. We honour that: the kernel writes the two scalars to a tiny device
// scratch buffer, and we cudaMemcpy them back synchronously at the end of
// layernorm_forward. Backward consumes rstd as a host float (same).
// gamma/beta/xhat are device tensors as declared.

namespace {

constexpr int LN_BLOCK = 256;

__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         float* __restrict__ xhat,
                                         float* __restrict__ scratch, // [mean, rstd]
                                         int n, float eps) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    // Mean.
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += x[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(n);

    // Variance.
    float local_v = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = x[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var = sdata[0] / static_cast<float>(n);
    const float rstd = rsqrtf(var + eps);

    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }

    for (int i = tid; i < n; i += blockDim.x) {
        const float xh = (x[i] - mean) * rstd;
        xhat[i] = xh;
        y[i] = gamma[i] * xh + beta[i];
    }
}

// Backward.
//   dGamma_i += dY_i * xhat_i
//   dBeta_i  += dY_i
//   dxh_i = dY_i * gamma_i
//   sum_dxh = sum dxh
//   sum_dxh_xhat = sum dxh * xhat
//   dX_i = (rstd / N) * (N * dxh_i - sum_dxh - xhat_i * sum_dxh_xhat)
__global__ void layernorm_backward_kernel(const float* __restrict__ dY,
                                          const float* __restrict__ xhat,
                                          const float* __restrict__ gamma,
                                          float rstd,
                                          float* __restrict__ dX,
                                          float* __restrict__ dGamma,
                                          float* __restrict__ dBeta,
                                          int n) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    // Accumulate dGamma, dBeta in place.
    for (int i = tid; i < n; i += blockDim.x) {
        const float g = dY[i];
        dGamma[i] += g * xhat[i];
        dBeta[i]  += g;
    }

    // sum_dxh
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += dY[i] * gamma[i];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh = sdata[0];

    // sum_dxh_xhat
    float local2 = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local2 += dY[i] * gamma[i] * xhat[i];
    }
    sdata[tid] = local2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh_xhat = sdata[0];

    const float nf = static_cast<float>(n);
    const float scale = rstd / nf;
    for (int i = tid; i < n; i += blockDim.x) {
        const float dxh = dY[i] * gamma[i];
        dX[i] = scale * (nf * dxh - sum_dxh - xhat[i] * sum_dxh_xhat);
    }
}

// FP16 backward. Inputs/outputs in FP16 storage; FP32 reductions in shared
// memory. dGamma/dBeta are written into FP32 scratch buffers (size n) which
// the host then folds into the caller-owned FP16 accumulators.
__global__ void layernorm_backward_kernel_fp16(const __half* __restrict__ dY,
                                               const __half* __restrict__ xhat,
                                               const __half* __restrict__ gamma,
                                               float rstd,
                                               __half* __restrict__ dX,
                                               float* __restrict__ dGamma_scratch,
                                               float* __restrict__ dBeta_scratch,
                                               int n) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    // Per-feature dGamma/dBeta into FP32 scratch (single thread per feature
    // — no concurrent writers to same i, so no atomics).
    for (int i = tid; i < n; i += blockDim.x) {
        const float g  = __half2float(dY[i]);
        const float xh = __half2float(xhat[i]);
        dGamma_scratch[i] = g * xh;
        dBeta_scratch[i]  = g;
    }

    // sum_dxh
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += __half2float(dY[i]) * __half2float(gamma[i]);
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh = sdata[0];

    // sum_dxh_xhat
    float local2 = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local2 += __half2float(dY[i]) * __half2float(gamma[i]) * __half2float(xhat[i]);
    }
    sdata[tid] = local2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh_xhat = sdata[0];

    const float nf = static_cast<float>(n);
    const float scale = rstd / nf;
    for (int i = tid; i < n; i += blockDim.x) {
        const float dxh = __half2float(dY[i]) * __half2float(gamma[i]);
        const float xh  = __half2float(xhat[i]);
        dX[i] = __float2half(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
    }
}

__global__ void ln_add_fp32_into_fp16(const float* __restrict__ src,
                                      __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}

} // namespace

void layernorm_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& beta,
                       ::brotensor::Tensor& y, ::brotensor::Tensor& xhat,
                       float& mean_out, float& rstd_out,
                       float eps) {
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (xhat.rows != x.rows || xhat.cols != x.cols) xhat.resize(x.rows, x.cols);
    if (n == 0) {
        mean_out = 0.0f;
        rstd_out = 0.0f;
        return;
    }

    // Scratch buffer for [mean, rstd] on device.
    float* d_scratch = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                              2 * sizeof(float)));

    layernorm_forward_kernel<<<1, LN_BLOCK>>>(
        reinterpret_cast<const float*>(x.data),
        reinterpret_cast<const float*>(gamma.data),
        reinterpret_cast<const float*>(beta.data),
        reinterpret_cast<float*>(y.data),
        reinterpret_cast<float*>(xhat.data),
        d_scratch,
        n, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    float h[2] = {0.0f, 0.0f};
    BROTENSOR_CUDA_CHECK(cudaMemcpy(h, d_scratch, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_scratch);
    mean_out = h[0];
    rstd_out = h[1];
}

// Inference-only batched forward. Processes R independent rows of length D
// (R = rows, D = cols). One block per row; threads cooperate on mean/var
// reductions in shared memory. Does NOT cache xhat or write mean/rstd —
// no host syncs. Use when caches/backward aren't needed (e.g., inference
// pipeline through a TransformerEncoder).
namespace {
__global__ void layernorm_forward_inference_batched_kernel(
        const float* __restrict__ x,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ y,
        int R, int D, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const float* xrow = x + static_cast<size_t>(row) * D;
    float*       yrow = y + static_cast<size_t>(row) * D;

    // Mean.
    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local += xrow[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(D);

    // Variance.
    float local_v = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = xrow[i] - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var  = sdata[0] / static_cast<float>(D);
    const float rstd = rsqrtf(var + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (xrow[i] - mean) * rstd;
        yrow[i] = xh * gamma[i] + beta[i];
    }
}
} // namespace

namespace {
__global__ void layernorm_forward_inference_batched_fp16_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ gamma,
        const __half* __restrict__ beta,
        __half* __restrict__ y,
        int R, int D, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const __half* xrow = x + static_cast<size_t>(row) * D;
    __half*       yrow = y + static_cast<size_t>(row) * D;

    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local += __half2float(xrow[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(D);

    float local_v = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = __half2float(xrow[i]) - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var  = sdata[0] / static_cast<float>(D);
    const float rstd = rsqrtf(var + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (__half2float(xrow[i]) - mean) * rstd;
        const float g  = __half2float(gamma[i]);
        const float b  = __half2float(beta[i]);
        yrow[i] = __float2half(xh * g + b);
    }
}
} // namespace

void layernorm_forward_inference_batched_fp16(const ::brotensor::Tensor& X_RD,
                                              const ::brotensor::Tensor& gamma,
                                              const ::brotensor::Tensor& beta,
                                              ::brotensor::Tensor& Y_RD,
                                              float eps) {
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 || gamma.dtype != Dtype::FP16 ||
        beta.dtype != Dtype::FP16) {
        throw std::runtime_error("layernorm_forward_inference_batched_fp16: all tensors must be FP16");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != Dtype::FP16) {
        Y_RD.resize(R, D, Dtype::FP16);
    }
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    layernorm_forward_inference_batched_fp16_kernel<<<R, block, shmem>>>(
        reinterpret_cast<const __half*>(X_RD.data),
        reinterpret_cast<const __half*>(gamma.data),
        reinterpret_cast<const __half*>(beta.data),
        reinterpret_cast<__half*>(Y_RD.data),
        R, D, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void layernorm_forward_inference_batched(const ::brotensor::Tensor& X_RD,
                                         const ::brotensor::Tensor& gamma,
                                         const ::brotensor::Tensor& beta,
                                         ::brotensor::Tensor& Y_RD,
                                         float eps) {
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D) Y_RD.resize(R, D);
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    layernorm_forward_inference_batched_kernel<<<R, block, shmem>>>(
        reinterpret_cast<const float*>(X_RD.data),
        reinterpret_cast<const float*>(gamma.data),
        reinterpret_cast<const float*>(beta.data),
        reinterpret_cast<float*>(Y_RD.data),
        R, D, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& xhat,
                        const ::brotensor::Tensor& gamma, float rstd,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dGamma, ::brotensor::Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward: dY must be FP16 or FP32");
    }
    if (xhat.dtype != dY.dtype || gamma.dtype != dY.dtype ||
        dGamma.dtype != dY.dtype || dBeta.dtype != dY.dtype) {
        throw std::runtime_error("layernorm_backward: all tensors must share dtype");
    }
    const int n = dY.size();
    if (dX.rows != dY.rows || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(dY.rows, dY.cols, dY.dtype);
    }
    if (n == 0) return;

    if (dY.dtype == Dtype::FP32) {
        layernorm_backward_kernel<<<1, LN_BLOCK>>>(
            reinterpret_cast<const float*>(dY.data),
            reinterpret_cast<const float*>(xhat.data),
            reinterpret_cast<const float*>(gamma.data),
            rstd,
            reinterpret_cast<float*>(dX.data),
            reinterpret_cast<float*>(dGamma.data),
            reinterpret_cast<float*>(dBeta.data),
            n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else {
        float* d_dg = nullptr;
        float* d_db = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dg), n * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db), n * sizeof(float)));
        layernorm_backward_kernel_fp16<<<1, LN_BLOCK>>>(
            reinterpret_cast<const __half*>(dY.data),
            reinterpret_cast<const __half*>(xhat.data),
            reinterpret_cast<const __half*>(gamma.data),
            rstd,
            reinterpret_cast<__half*>(dX.data),
            d_dg, d_db, n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks = (n + 255) / 256;
        ln_add_fp32_into_fp16<<<blocks, 256>>>(
            d_dg, reinterpret_cast<__half*>(dGamma.data), n);
        ln_add_fp32_into_fp16<<<blocks, 256>>>(
            d_db, reinterpret_cast<__half*>(dBeta.data), n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dg);
        cudaFree(d_db);
    }
}

// ─── Vtable contribution ──────────────────────────────────────────────────
//
// Fills the GroupNorm + ResBlock + LayerNorm slots. Called by the CUDA
// backend's per-cluster vtable assembler.

void resblock_forward(const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,
                      const ::brotensor::Tensor& W1, const ::brotensor::Tensor* b1,
                      const ::brotensor::Tensor* t_emb_shift,
                      const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,
                      const ::brotensor::Tensor& W2, const ::brotensor::Tensor* b2,
                      const ::brotensor::Tensor* Wskip, const ::brotensor::Tensor* bskip,
                      int N, int C_in, int C_out, int H, int Wd,
                      int num_groups, float eps,
                      ::brotensor::Tensor& Y);

void resblock_forward_int8w_fp16(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,
                                 const ::brotensor::Tensor& W1_int8, const ::brotensor::Tensor& s1,
                                 const ::brotensor::Tensor* b1,
                                 const ::brotensor::Tensor* t_emb_shift,
                                 const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,
                                 const ::brotensor::Tensor& W2_int8, const ::brotensor::Tensor& s2,
                                 const ::brotensor::Tensor* b2,
                                 const ::brotensor::Tensor* Wskip_int8, const ::brotensor::Tensor* sskip,
                                 const ::brotensor::Tensor* bskip,
                                 int N, int C_in, int C_out, int H, int Wd,
                                 int num_groups, float eps,
                                 ::brotensor::Tensor& Y);

void resblock_backward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& gamma1, const ::brotensor::Tensor& beta1,
                       const ::brotensor::Tensor& W1, const ::brotensor::Tensor* b1,
                       const ::brotensor::Tensor* t_emb_shift,
                       const ::brotensor::Tensor& gamma2, const ::brotensor::Tensor& beta2,
                       const ::brotensor::Tensor& W2, const ::brotensor::Tensor* b2,
                       const ::brotensor::Tensor* Wskip, const ::brotensor::Tensor* bskip,
                       int N, int C_in, int C_out, int H, int Wd,
                       int num_groups, float eps,
                       const ::brotensor::Tensor& dY,
                       ::brotensor::Tensor& dX,
                       ::brotensor::Tensor& dGamma1, ::brotensor::Tensor& dBeta1,
                       ::brotensor::Tensor& dW1, ::brotensor::Tensor* db1,
                       ::brotensor::Tensor* dt_emb_shift,
                       ::brotensor::Tensor& dGamma2, ::brotensor::Tensor& dBeta2,
                       ::brotensor::Tensor& dW2, ::brotensor::Tensor* db2,
                       ::brotensor::Tensor* dWskip, ::brotensor::Tensor* dbskip);

// Forward decls from sibling group_norm.cu (this cluster, Phase 2E).
void group_norm_forward(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& gamma,
                        const ::brotensor::Tensor& beta,
                        int N, int C, int H, int W,
                        int num_groups, float eps,
                        ::brotensor::Tensor& Y);
void group_norm_backward(const ::brotensor::Tensor& X,
                         const ::brotensor::Tensor& gamma,
                         const ::brotensor::Tensor& dY,
                         int N, int C, int H, int W,
                         int num_groups, float eps,
                         ::brotensor::Tensor& dX,
                         ::brotensor::Tensor& dGamma,
                         ::brotensor::Tensor& dBeta);

// Masked mean-pool lives in reduce.cu (same brotensor::detail::cuda
// namespace); registered here so the norms/reduce slots are filled.
void masked_mean_pool_forward(const ::brotensor::Tensor& X, const float* d_mask,
                              ::brotensor::Tensor& y);
void masked_mean_pool_backward(const ::brotensor::Tensor& dY, const float* d_mask,
                               int K, ::brotensor::Tensor& dX);

void fill_cuda_vtable_norms(::brotensor::detail::OpsVTable& v) {
    v.layernorm_forward                              = &layernorm_forward;
    v.layernorm_backward                             = &layernorm_backward;
    v.layernorm_forward_inference_batched            = &layernorm_forward_inference_batched;
    v.layernorm_forward_inference_batched_fp16       = &layernorm_forward_inference_batched_fp16;
    v.group_norm_forward                             = &group_norm_forward;
    v.group_norm_backward                            = &group_norm_backward;
    v.resblock_forward                               = &resblock_forward;
    v.resblock_backward                              = &resblock_backward;
    v.resblock_forward_int8w_fp16                    = &resblock_forward_int8w_fp16;
    v.masked_mean_pool_forward                       = &masked_mean_pool_forward;
    v.masked_mean_pool_backward                      = &masked_mean_pool_backward;
}

} // namespace brotensor::detail::cuda
