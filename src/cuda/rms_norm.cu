// RMSNorm forward + backward.
//   rms[b] = sqrt(mean_j x[b, j]^2 + eps)
//   y[b, j] = x[b, j] * gamma[j] / rms[b]
// Per-row reductions inside a block; one block per row.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int RMS_BLOCK = 256;

__device__ inline float block_sum(float v, float* sdata) {
    const int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    return sdata[0];
}

__global__ void rms_forward_fp32_kernel(const float* __restrict__ X,
                                        const float* __restrict__ gamma,
                                        float* __restrict__ Y,
                                        int B, int D, float eps) {
    extern __shared__ float sdata[];
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const float* xrow = X + static_cast<size_t>(b) * D;
    float*       yrow = Y + static_cast<size_t>(b) * D;

    float local = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        const float v = xrow[j];
        local += v * v;
    }
    const float sum = block_sum(local, sdata);
    const float rrms = rsqrtf(sum / static_cast<float>(D) + eps);

    for (int j = tid; j < D; j += blockDim.x) {
        yrow[j] = xrow[j] * gamma[j] * rrms;
    }
}

__global__ void rms_forward_fp16_kernel(const __half* __restrict__ X,
                                        const __half* __restrict__ gamma,
                                        __half* __restrict__ Y,
                                        int B, int D, float eps) {
    extern __shared__ float sdata[];
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const __half* xrow = X + static_cast<size_t>(b) * D;
    __half*       yrow = Y + static_cast<size_t>(b) * D;

    float local = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        const float v = __half2float(xrow[j]);
        local += v * v;
    }
    const float sum = block_sum(local, sdata);
    const float rrms = rsqrtf(sum / static_cast<float>(D) + eps);

    for (int j = tid; j < D; j += blockDim.x) {
        const float xv = __half2float(xrow[j]);
        const float gv = __half2float(gamma[j]);
        yrow[j] = __float2half(xv * gv * rrms);
    }
}

// Backward math. With rms = sqrt(mean_j x_j^2 + eps), r = 1/rms,
//   y_j = x_j * g_j * r
// Let dx_j be dY_j * g_j * r = ŷ_j (without the x_j factor). Then
//   dRms_term: drms/dx_j = x_j / (D * rms)
//   dY_j contributes to rms through x_j only.
// Combining:
//   sum_xdy = sum_j x_j * dY_j * g_j           (== sum_j x_j * dx_j)
//   dX_j = r * (g_j * dY_j) - (x_j / D) * r^3 * sum_xdy
//        = r * (g_j * dY_j  - (x_j / D) * r^2 * sum_xdy)
//
// dGamma_j accumulated across batch:
//   dGamma_j += sum_b dY[b,j] * x[b,j] / rms[b]
__global__ void rms_backward_fp32_kernel(const float* __restrict__ X,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ dY,
                                         float* __restrict__ dX,
                                         float* __restrict__ dGamma,
                                         int B, int D, float eps) {
    extern __shared__ float sdata[];
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const float* xrow  = X  + static_cast<size_t>(b) * D;
    const float* dyrow = dY + static_cast<size_t>(b) * D;
    float*       dxrow = dX + static_cast<size_t>(b) * D;

    // Recompute rrms.
    float local = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        const float v = xrow[j];
        local += v * v;
    }
    const float sum_xx = block_sum(local, sdata);
    const float rrms = rsqrtf(sum_xx / static_cast<float>(D) + eps);

    // Need sum_xdy = sum_j x_j * dY_j * gamma_j.
    float local2 = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        local2 += xrow[j] * dyrow[j] * gamma[j];
    }
    const float sum_xdy = block_sum(local2, sdata);

    const float inv_D = 1.0f / static_cast<float>(D);
    const float coeff = inv_D * rrms * rrms * sum_xdy;  // multiplied by rrms below

    for (int j = tid; j < D; j += blockDim.x) {
        const float g = gamma[j];
        const float dy = dyrow[j];
        const float x  = xrow[j];
        dxrow[j] = rrms * (g * dy - x * coeff);
        // dGamma_j += dY * x * rrms.
        atomicAdd(&dGamma[j], dy * x * rrms);
    }
}

__global__ void rms_backward_fp16_kernel(const __half* __restrict__ X,
                                         const __half* __restrict__ gamma,
                                         const __half* __restrict__ dY,
                                         __half* __restrict__ dX,
                                         float* __restrict__ dGamma_scratch,
                                         int B, int D, float eps) {
    extern __shared__ float sdata[];
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const __half* xrow  = X  + static_cast<size_t>(b) * D;
    const __half* dyrow = dY + static_cast<size_t>(b) * D;
    __half*       dxrow = dX + static_cast<size_t>(b) * D;

    float local = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        const float v = __half2float(xrow[j]);
        local += v * v;
    }
    const float sum_xx = block_sum(local, sdata);
    const float rrms = rsqrtf(sum_xx / static_cast<float>(D) + eps);

    float local2 = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        local2 += __half2float(xrow[j]) * __half2float(dyrow[j]) *
                  __half2float(gamma[j]);
    }
    const float sum_xdy = block_sum(local2, sdata);

    const float inv_D = 1.0f / static_cast<float>(D);
    const float coeff = inv_D * rrms * rrms * sum_xdy;

    for (int j = tid; j < D; j += blockDim.x) {
        const float g  = __half2float(gamma[j]);
        const float dy = __half2float(dyrow[j]);
        const float x  = __half2float(xrow[j]);
        dxrow[j] = __float2half(rrms * (g * dy - x * coeff));
        atomicAdd(&dGamma_scratch[j], dy * x * rrms);
    }
}

__global__ void rms_fp32_into_fp16_kernel(const float* __restrict__ src,
                                          __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}

} // namespace

void rms_norm_forward_gpu(const GpuTensor& X, const GpuTensor& gamma,
                         float eps, GpuTensor& Y) {
    if (gamma.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_forward_gpu: gamma.dtype must match X.dtype");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (gamma.size() != D) {
        throw std::runtime_error("rms_norm_forward_gpu: gamma must have D elements");
    }
    if (Y.rows != B || Y.cols != D || Y.dtype != X.dtype) {
        Y.resize(B, D, X.dtype);
    }
    if (B == 0 || D == 0) return;
    const int block = RMS_BLOCK;
    const size_t shmem = block * sizeof(float);
    if (X.dtype == Dtype::FP16) {
        rms_forward_fp16_kernel<<<B, block, shmem>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<const __half*>(gamma.data_fp16()),
            reinterpret_cast<__half*>(Y.data_fp16()),
            B, D, eps);
    } else {
        rms_forward_fp32_kernel<<<B, block, shmem>>>(
            X.data, gamma.data, Y.data, B, D, eps);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rms_norm_backward_gpu(const GpuTensor& X, const GpuTensor& gamma,
                          const GpuTensor& dY, float eps,
                          GpuTensor& dX, GpuTensor& dGamma) {
    if (gamma.dtype != X.dtype || dY.dtype != X.dtype ||
        dGamma.dtype != X.dtype) {
        throw std::runtime_error("rms_norm_backward_gpu: dtypes must match");
    }
    const int B = X.rows;
    const int D = X.cols;
    if (dY.rows != B || dY.cols != D) {
        throw std::runtime_error("rms_norm_backward_gpu: dY shape mismatch");
    }
    if (gamma.size() != D || dGamma.size() != D) {
        throw std::runtime_error("rms_norm_backward_gpu: gamma/dGamma size mismatch");
    }
    if (dX.rows != B || dX.cols != D || dX.dtype != X.dtype) {
        dX.resize(B, D, X.dtype);
    }
    if (B == 0 || D == 0) return;
    const int block = RMS_BLOCK;
    const size_t shmem = block * sizeof(float);
    if (X.dtype == Dtype::FP32) {
        rms_backward_fp32_kernel<<<B, block, shmem>>>(
            X.data, gamma.data, dY.data, dX.data, dGamma.data,
            B, D, eps);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    } else {
        float* d_dg = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dg),
                                        D * sizeof(float)));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_dg, 0, D * sizeof(float)));
        rms_backward_fp16_kernel<<<B, block, shmem>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<const __half*>(gamma.data_fp16()),
            reinterpret_cast<const __half*>(dY.data_fp16()),
            reinterpret_cast<__half*>(dX.data_fp16()),
            d_dg, B, D, eps);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks = (D + 255) / 256;
        rms_fp32_into_fp16_kernel<<<blocks, 256>>>(
            d_dg, reinterpret_cast<__half*>(dGamma.data_fp16()), D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dg);
    }
}

} // namespace brotensor
