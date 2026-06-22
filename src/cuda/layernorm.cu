#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Current CUDA stream for hot-op launches — so kernels join a non-default
// capture/replay stream instead of silently landing on the default stream.
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

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
    __syncthreads();   // barrier before reusing sdata for variance — see the
                       // batched variant; without it sdata[0] is clobbered by a
                       // fast thread before a slow thread reads `mean`.

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

// FP16 forward. Loads/stores in FP16; mean/variance reductions in FP32.
// Writes xhat in FP16 to round-trip with layernorm_backward (which dispatches
// on dY.dtype and requires xhat.dtype == dY.dtype).
__global__ void layernorm_forward_kernel_fp16(const __half* __restrict__ x,
                                              const __half* __restrict__ gamma,
                                              const __half* __restrict__ beta,
                                              __half* __restrict__ y,
                                              __half* __restrict__ xhat,
                                              float* __restrict__ scratch, // [mean, rstd]
                                              int n, float eps) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    // Mean.
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += __half2float(x[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(n);
    __syncthreads();   // barrier before reusing sdata for variance — see FP32 variant

    // Variance.
    float local_v = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = __half2float(x[i]) - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var  = sdata[0] / static_cast<float>(n);
    const float rstd = rsqrtf(var + eps);

    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }

    for (int i = tid; i < n; i += blockDim.x) {
        const float xh = (__half2float(x[i]) - mean) * rstd;
        xhat[i] = __float2half(xh);
        const float g = __half2float(gamma[i]);
        const float b = __half2float(beta[i]);
        y[i] = __float2half(g * xh + b);
    }
}

__global__ void layernorm_forward_kernel_bf16(const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ gamma,
                                              const __nv_bfloat16* __restrict__ beta,
                                              __nv_bfloat16* __restrict__ y,
                                              __nv_bfloat16* __restrict__ xhat,
                                              float* __restrict__ scratch, // [mean, rstd]
                                              int n, float eps) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += __bfloat162float(x[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(n);
    __syncthreads();   // barrier before reusing sdata for variance — see FP32 variant

    float local_v = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = __bfloat162float(x[i]) - mean;
        local_v += d * d;
    }
    sdata[tid] = local_v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float var  = sdata[0] / static_cast<float>(n);
    const float rstd = rsqrtf(var + eps);

    if (tid == 0) {
        scratch[0] = mean;
        scratch[1] = rstd;
    }

    for (int i = tid; i < n; i += blockDim.x) {
        const float xh = (__bfloat162float(x[i]) - mean) * rstd;
        xhat[i] = __float2bfloat16(xh);
        const float g = __bfloat162float(gamma[i]);
        const float b = __bfloat162float(beta[i]);
        y[i] = __float2bfloat16(g * xh + b);
    }
}

// BF16 backward — verbatim copy of layernorm_backward_kernel_fp16 with
// __half → __nv_bfloat16, __half2float → __bfloat162float, __float2half →
// __float2bfloat16. FP32 scratch/fold pattern mirrored exactly.
__global__ void layernorm_backward_kernel_bf16(const __nv_bfloat16* __restrict__ dY,
                                               const __nv_bfloat16* __restrict__ xhat,
                                               const __nv_bfloat16* __restrict__ gamma,
                                               float rstd,
                                               __nv_bfloat16* __restrict__ dX,
                                               float* __restrict__ dGamma_scratch,
                                               float* __restrict__ dBeta_scratch,
                                               int n) {
    __shared__ float sdata[LN_BLOCK];
    const int tid = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) {
        const float g  = __bfloat162float(dY[i]);
        const float xh = __bfloat162float(xhat[i]);
        dGamma_scratch[i] = g * xh;
        dBeta_scratch[i]  = g;
    }

    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local += __bfloat162float(dY[i]) * __bfloat162float(gamma[i]);
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh = sdata[0];

    float local2 = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local2 += __bfloat162float(dY[i]) * __bfloat162float(gamma[i]) * __bfloat162float(xhat[i]);
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
        const float dxh = __bfloat162float(dY[i]) * __bfloat162float(gamma[i]);
        const float xh  = __bfloat162float(xhat[i]);
        dX[i] = __float2bfloat16(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
    }
}

__global__ void ln_add_fp32_into_bf16(const float* __restrict__ src,
                                      __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

} // namespace

void layernorm_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& gamma, const ::brotensor::Tensor& beta,
                       ::brotensor::Tensor& y, ::brotensor::Tensor& xhat,
                       float& mean_out, float& rstd_out,
                       float eps) {
    using ::brotensor::Dtype;
    if (x.dtype != Dtype::FP16 && x.dtype != Dtype::BF16 && x.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward: x must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != x.dtype || beta.dtype != x.dtype) {
        throw std::runtime_error("layernorm_forward: gamma/beta dtype must match x.dtype");
    }
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != x.dtype) {
        y.resize(x.rows, x.cols, x.dtype);
    }
    if (xhat.rows != x.rows || xhat.cols != x.cols || xhat.dtype != x.dtype) {
        xhat.resize(x.rows, x.cols, x.dtype);
    }
    if (n == 0) {
        mean_out = 0.0f;
        rstd_out = 0.0f;
        return;
    }

    // Scratch buffer for [mean, rstd] on device.
    float* d_scratch = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                              2 * sizeof(float)));

    if (x.dtype == Dtype::FP16) {
        layernorm_forward_kernel_fp16<<<1, LN_BLOCK, 0, cur_stream()>>>(
            reinterpret_cast<const __half*>(x.data),
            reinterpret_cast<const __half*>(gamma.data),
            reinterpret_cast<const __half*>(beta.data),
            reinterpret_cast<__half*>(y.data),
            reinterpret_cast<__half*>(xhat.data),
            d_scratch, n, eps);
    } else if (x.dtype == Dtype::BF16) {
        layernorm_forward_kernel_bf16<<<1, LN_BLOCK, 0, cur_stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data),
            reinterpret_cast<const __nv_bfloat16*>(beta.data),
            reinterpret_cast<__nv_bfloat16*>(y.data),
            reinterpret_cast<__nv_bfloat16*>(xhat.data),
            d_scratch, n, eps);
    } else {
        layernorm_forward_kernel<<<1, LN_BLOCK, 0, cur_stream()>>>(
            reinterpret_cast<const float*>(x.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(beta.data),
            reinterpret_cast<float*>(y.data),
            reinterpret_cast<float*>(xhat.data),
            d_scratch, n, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    float h[2] = {0.0f, 0.0f};
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(h, d_scratch, 2 * sizeof(float),
                              cudaMemcpyDeviceToHost, cur_stream()));
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(cur_stream()));
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
    // Barrier: all threads must read the mean from sdata[0] BEFORE any thread
    // overwrites the shared buffer with its variance partial below. Without it a
    // fast thread's `sdata[tid] = local_v` clobbers sdata[0] while a slow thread
    // still reads `mean` — an intermittent shared-memory race that makes
    // layernorm nondeterministic (compute-sanitizer racecheck: read/write hazard).
    __syncthreads();

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
    __syncthreads();   // barrier before reusing sdata for variance — see FP32 variant

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

namespace {
__global__ void layernorm_forward_inference_batched_bf16_kernel(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ gamma,
        const __nv_bfloat16* __restrict__ beta,
        __nv_bfloat16* __restrict__ y,
        int R, int D, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;

    const __nv_bfloat16* xrow = x + static_cast<size_t>(row) * D;
    __nv_bfloat16*       yrow = y + static_cast<size_t>(row) * D;

    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local += __bfloat162float(xrow[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(D);
    __syncthreads();   // barrier before reusing sdata for variance — see FP32 variant

    float local_v = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = __bfloat162float(xrow[i]) - mean;
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
        const float xh = (__bfloat162float(xrow[i]) - mean) * rstd;
        const float g  = __bfloat162float(gamma[i]);
        const float b  = __bfloat162float(beta[i]);
        yrow[i] = __float2bfloat16(xh * g + b);
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
    layernorm_forward_inference_batched_fp16_kernel<<<R, block, shmem, cur_stream()>>>(
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
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 && X_RD.dtype != Dtype::BF16 &&
        X_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward_inference_batched: X must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != X_RD.dtype || beta.dtype != X_RD.dtype) {
        throw std::runtime_error("layernorm_forward_inference_batched: gamma/beta dtype must match X.dtype");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != X_RD.dtype) {
        Y_RD.resize(R, D, X_RD.dtype);
    }
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    if (X_RD.dtype == Dtype::FP16) {
        layernorm_forward_inference_batched_fp16_kernel<<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const __half*>(X_RD.data),
            reinterpret_cast<const __half*>(gamma.data),
            reinterpret_cast<const __half*>(beta.data),
            reinterpret_cast<__half*>(Y_RD.data),
            R, D, eps);
    } else if (X_RD.dtype == Dtype::BF16) {
        layernorm_forward_inference_batched_bf16_kernel<<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(X_RD.data),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data),
            reinterpret_cast<const __nv_bfloat16*>(beta.data),
            reinterpret_cast<__nv_bfloat16*>(Y_RD.data),
            R, D, eps);
    } else {
        layernorm_forward_inference_batched_kernel<<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const float*>(X_RD.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(beta.data),
            reinterpret_cast<float*>(Y_RD.data),
            R, D, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward(const ::brotensor::Tensor& dY, const ::brotensor::Tensor& xhat,
                        const ::brotensor::Tensor& gamma, float rstd,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dGamma, ::brotensor::Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (dY.dtype != Dtype::FP16 && dY.dtype != Dtype::FP32 && dY.dtype != Dtype::BF16) {
        throw std::runtime_error("layernorm_backward: dY must be FP16, BF16, or FP32");
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
        layernorm_backward_kernel<<<1, LN_BLOCK, 0, cur_stream()>>>(
            reinterpret_cast<const float*>(dY.data),
            reinterpret_cast<const float*>(xhat.data),
            reinterpret_cast<const float*>(gamma.data),
            rstd,
            reinterpret_cast<float*>(dX.data),
            reinterpret_cast<float*>(dGamma.data),
            reinterpret_cast<float*>(dBeta.data),
            n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else if (dY.dtype == Dtype::BF16) {
        float* d_dg = nullptr;
        float* d_db = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dg), n * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db), n * sizeof(float)));
        layernorm_backward_kernel_bf16<<<1, LN_BLOCK, 0, cur_stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(dY.data),
            reinterpret_cast<const __nv_bfloat16*>(xhat.data),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data),
            rstd,
            reinterpret_cast<__nv_bfloat16*>(dX.data),
            d_dg, d_db, n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks = (n + 255) / 256;
        ln_add_fp32_into_bf16<<<blocks, 256, 0, cur_stream()>>>(
            d_dg, reinterpret_cast<__nv_bfloat16*>(dGamma.data), n);
        ln_add_fp32_into_bf16<<<blocks, 256, 0, cur_stream()>>>(
            d_db, reinterpret_cast<__nv_bfloat16*>(dBeta.data), n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dg);
        cudaFree(d_db);
    } else {
        float* d_dg = nullptr;
        float* d_db = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dg), n * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db), n * sizeof(float)));
        layernorm_backward_kernel_fp16<<<1, LN_BLOCK, 0, cur_stream()>>>(
            reinterpret_cast<const __half*>(dY.data),
            reinterpret_cast<const __half*>(xhat.data),
            reinterpret_cast<const __half*>(gamma.data),
            rstd,
            reinterpret_cast<__half*>(dX.data),
            d_dg, d_db, n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks = (n + 255) / 256;
        ln_add_fp32_into_fp16<<<blocks, 256, 0, cur_stream()>>>(
            d_dg, reinterpret_cast<__half*>(dGamma.data), n);
        ln_add_fp32_into_fp16<<<blocks, 256, 0, cur_stream()>>>(
            d_db, reinterpret_cast<__half*>(dBeta.data), n);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dg);
        cudaFree(d_db);
    }
}

// ─── Batched (training) LayerNorm with caches ─────────────────────────────
//
// One block per row, R blocks. Forward writes Mean_R[row]/Rstd_R[row] in FP32
// regardless of X dtype (so backward dispatches identically across precisions).
// Backward accumulates dGamma/dBeta across rows via atomics: FP32 uses
// atomicAdd(float*) directly; FP16/BF16 go through FP32 scratch[D] + a fold
// kernel that adds into the FP16/BF16 accumulator preserving "caller zeros,
// op accumulates" semantics.
namespace {

template <typename T>
__device__ inline float to_f32(T x);
template <> __device__ inline float to_f32<float>(float x) { return x; }
template <> __device__ inline float to_f32<__half>(__half x) { return __half2float(x); }
template <> __device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_f32(float x);
template <> __device__ inline float from_f32<float>(float x) { return x; }
template <> __device__ inline __half from_f32<__half>(float x) { return __float2half(x); }
template <> __device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T>
__global__ void ln_fwd_batched_caches_kernel(const T* __restrict__ x,
                                             const T* __restrict__ gamma,
                                             const T* __restrict__ beta,
                                             T* __restrict__ y,
                                             T* __restrict__ xhat,
                                             float* __restrict__ mean_out,
                                             float* __restrict__ rstd_out,
                                             int R, int D, float eps) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;
    const T* xrow = x    + static_cast<size_t>(row) * D;
    T*       yrow = y    + static_cast<size_t>(row) * D;
    T*       hrow = xhat + static_cast<size_t>(row) * D;

    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local += to_f32<T>(xrow[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float mean = sdata[0] / static_cast<float>(D);
    __syncthreads();   // barrier before reusing sdata for variance — see FP32 variant

    float local_v = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float d = to_f32<T>(xrow[i]) - mean;
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

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    for (int i = tid; i < D; i += blockDim.x) {
        const float xh = (to_f32<T>(xrow[i]) - mean) * rstd;
        hrow[i] = from_f32<T>(xh);
        const float g = to_f32<T>(gamma[i]);
        const float b = to_f32<T>(beta[i]);
        yrow[i] = from_f32<T>(g * xh + b);
    }
}

// Backward kernel template. dGamma/dBeta accumulators are FP32 (either the
// caller's dGamma/dBeta for the FP32 path, or per-call FP32 scratch for the
// FP16/BF16 paths).
template <typename T>
__global__ void ln_bwd_batched_caches_kernel(const T* __restrict__ dY,
                                             const T* __restrict__ xhat,
                                             const T* __restrict__ gamma,
                                             const float* __restrict__ rstd_R,
                                             T* __restrict__ dX,
                                             float* __restrict__ dGamma_f32,
                                             float* __restrict__ dBeta_f32,
                                             int R, int D) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= R) return;
    const T* dyr = dY   + static_cast<size_t>(row) * D;
    const T* hr  = xhat + static_cast<size_t>(row) * D;
    T*       dxr = dX   + static_cast<size_t>(row) * D;
    const float rstd = rstd_R[row];

    // sum_dxh
    float local = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local += to_f32<T>(dyr[i]) * to_f32<T>(gamma[i]);
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
    for (int i = tid; i < D; i += blockDim.x) {
        local2 += to_f32<T>(dyr[i]) * to_f32<T>(gamma[i]) * to_f32<T>(hr[i]);
    }
    sdata[tid] = local2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum_dxh_xhat = sdata[0];

    const float nf = static_cast<float>(D);
    const float scale = rstd / nf;
    for (int i = tid; i < D; i += blockDim.x) {
        const float g  = to_f32<T>(dyr[i]);
        const float xh = to_f32<T>(hr[i]);
        const float dxh = g * to_f32<T>(gamma[i]);
        dxr[i] = from_f32<T>(scale * (nf * dxh - sum_dxh - xh * sum_dxh_xhat));
        atomicAdd(&dGamma_f32[i], g * xh);
        atomicAdd(&dBeta_f32[i],  g);
    }
}

__global__ void ln_fold_fp32_into_fp16(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}

__global__ void ln_fold_fp32_into_bf16(const float* __restrict__ src,
                                       __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

} // namespace

void layernorm_forward_batched_with_caches(const ::brotensor::Tensor& X_RD,
                                           const ::brotensor::Tensor& gamma,
                                           const ::brotensor::Tensor& beta,
                                           ::brotensor::Tensor& Y_RD,
                                           ::brotensor::Tensor& Xhat_RD,
                                           ::brotensor::Tensor& Mean_R,
                                           ::brotensor::Tensor& Rstd_R,
                                           float eps) {
    using ::brotensor::Dtype;
    if (X_RD.dtype != Dtype::FP16 && X_RD.dtype != Dtype::BF16 &&
        X_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_forward_batched_with_caches: X must be FP16, BF16, or FP32");
    }
    if (gamma.dtype != X_RD.dtype || beta.dtype != X_RD.dtype) {
        throw std::runtime_error("layernorm_forward_batched_with_caches: gamma/beta dtype must match X.dtype");
    }
    const int R = X_RD.rows;
    const int D = X_RD.cols;
    if (Y_RD.rows != R || Y_RD.cols != D || Y_RD.dtype != X_RD.dtype) {
        Y_RD.resize(R, D, X_RD.dtype);
    }
    if (Xhat_RD.rows != R || Xhat_RD.cols != D || Xhat_RD.dtype != X_RD.dtype) {
        Xhat_RD.resize(R, D, X_RD.dtype);
    }
    if (Mean_R.rows != R || Mean_R.cols != 1 || Mean_R.dtype != Dtype::FP32) {
        Mean_R.resize(R, 1, Dtype::FP32);
    }
    if (Rstd_R.rows != R || Rstd_R.cols != 1 || Rstd_R.dtype != Dtype::FP32) {
        Rstd_R.resize(R, 1, Dtype::FP32);
    }
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);
    if (X_RD.dtype == Dtype::FP16) {
        ln_fwd_batched_caches_kernel<__half><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const __half*>(X_RD.data),
            reinterpret_cast<const __half*>(gamma.data),
            reinterpret_cast<const __half*>(beta.data),
            reinterpret_cast<__half*>(Y_RD.data),
            reinterpret_cast<__half*>(Xhat_RD.data),
            reinterpret_cast<float*>(Mean_R.data),
            reinterpret_cast<float*>(Rstd_R.data),
            R, D, eps);
    } else if (X_RD.dtype == Dtype::BF16) {
        ln_fwd_batched_caches_kernel<__nv_bfloat16><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(X_RD.data),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data),
            reinterpret_cast<const __nv_bfloat16*>(beta.data),
            reinterpret_cast<__nv_bfloat16*>(Y_RD.data),
            reinterpret_cast<__nv_bfloat16*>(Xhat_RD.data),
            reinterpret_cast<float*>(Mean_R.data),
            reinterpret_cast<float*>(Rstd_R.data),
            R, D, eps);
    } else {
        ln_fwd_batched_caches_kernel<float><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const float*>(X_RD.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(beta.data),
            reinterpret_cast<float*>(Y_RD.data),
            reinterpret_cast<float*>(Xhat_RD.data),
            reinterpret_cast<float*>(Mean_R.data),
            reinterpret_cast<float*>(Rstd_R.data),
            R, D, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward_batched_with_caches(const ::brotensor::Tensor& dY_RD,
                                            const ::brotensor::Tensor& Xhat_RD,
                                            const ::brotensor::Tensor& gamma,
                                            const ::brotensor::Tensor& Rstd_R,
                                            ::brotensor::Tensor& dX_RD,
                                            ::brotensor::Tensor& dGamma,
                                            ::brotensor::Tensor& dBeta) {
    using ::brotensor::Dtype;
    if (dY_RD.dtype != Dtype::FP16 && dY_RD.dtype != Dtype::BF16 &&
        dY_RD.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY must be FP16, BF16, or FP32");
    }
    if (Xhat_RD.dtype != dY_RD.dtype || gamma.dtype != dY_RD.dtype ||
        dGamma.dtype != dY_RD.dtype || dBeta.dtype != dY_RD.dtype) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: dY/Xhat/gamma/dGamma/dBeta must share dtype");
    }
    if (Rstd_R.dtype != Dtype::FP32) {
        throw std::runtime_error("layernorm_backward_batched_with_caches: Rstd_R must be FP32");
    }
    const int R = dY_RD.rows;
    const int D = dY_RD.cols;
    if (dX_RD.rows != R || dX_RD.cols != D || dX_RD.dtype != dY_RD.dtype) {
        dX_RD.resize(R, D, dY_RD.dtype);
    }
    if (R == 0 || D == 0) return;
    const int block = LN_BLOCK;
    const size_t shmem = static_cast<size_t>(block) * sizeof(float);

    if (dY_RD.dtype == Dtype::FP32) {
        // Atomic-add directly into the caller's FP32 dGamma/dBeta.
        ln_bwd_batched_caches_kernel<float><<<R, block, shmem, cur_stream()>>>(
            reinterpret_cast<const float*>(dY_RD.data),
            reinterpret_cast<const float*>(Xhat_RD.data),
            reinterpret_cast<const float*>(gamma.data),
            reinterpret_cast<const float*>(Rstd_R.data),
            reinterpret_cast<float*>(dX_RD.data),
            reinterpret_cast<float*>(dGamma.data),
            reinterpret_cast<float*>(dBeta.data),
            R, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else {
        // Allocate + zero FP32 scratch[D] for dGamma and dBeta, accumulate via
        // atomics, then fold into caller's FP16/BF16 accumulators.
        float* d_dg = nullptr;
        float* d_db = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dg), D * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db), D * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_dg, 0, D * sizeof(float), cur_stream()));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_db, 0, D * sizeof(float), cur_stream()));
        if (dY_RD.dtype == Dtype::FP16) {
            ln_bwd_batched_caches_kernel<__half><<<R, block, shmem, cur_stream()>>>(
                reinterpret_cast<const __half*>(dY_RD.data),
                reinterpret_cast<const __half*>(Xhat_RD.data),
                reinterpret_cast<const __half*>(gamma.data),
                reinterpret_cast<const float*>(Rstd_R.data),
                reinterpret_cast<__half*>(dX_RD.data),
                d_dg, d_db, R, D);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            const int blocks = (D + 255) / 256;
            ln_fold_fp32_into_fp16<<<blocks, 256, 0, cur_stream()>>>(
                d_dg, reinterpret_cast<__half*>(dGamma.data), D);
            ln_fold_fp32_into_fp16<<<blocks, 256, 0, cur_stream()>>>(
                d_db, reinterpret_cast<__half*>(dBeta.data), D);
        } else {
            ln_bwd_batched_caches_kernel<__nv_bfloat16><<<R, block, shmem, cur_stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(dY_RD.data),
                reinterpret_cast<const __nv_bfloat16*>(Xhat_RD.data),
                reinterpret_cast<const __nv_bfloat16*>(gamma.data),
                reinterpret_cast<const float*>(Rstd_R.data),
                reinterpret_cast<__nv_bfloat16*>(dX_RD.data),
                d_dg, d_db, R, D);
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            const int blocks = (D + 255) / 256;
            ln_fold_fp32_into_bf16<<<blocks, 256, 0, cur_stream()>>>(
                d_dg, reinterpret_cast<__nv_bfloat16*>(dGamma.data), D);
            ln_fold_fp32_into_bf16<<<blocks, 256, 0, cur_stream()>>>(
                d_db, reinterpret_cast<__nv_bfloat16*>(dBeta.data), D);
        }
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

// Forward decls from sibling group_norm.cu (this cluster).
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
    v.layernorm_forward_batched_with_caches          = &layernorm_forward_batched_with_caches;
    v.layernorm_backward_batched_with_caches         = &layernorm_backward_batched_with_caches;
    v.group_norm_forward                             = &group_norm_forward;
    v.group_norm_backward                            = &group_norm_backward;
    v.resblock_forward                               = &resblock_forward;
    v.resblock_backward                              = &resblock_backward;
    v.resblock_forward_int8w_fp16                    = &resblock_forward_int8w_fp16;
    v.masked_mean_pool_forward                       = &masked_mean_pool_forward;
    v.masked_mean_pool_backward                      = &masked_mean_pool_backward;
}

} // namespace brotensor::detail::cuda
