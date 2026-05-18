// Fused DDIM update (FP16). One elementwise kernel; FP32 internal math.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

namespace {

constexpr int DDIM_BLOCK = 256;

__global__ void ddim_step_kernel(const __half* __restrict__ x_t,
                                 const __half* __restrict__ eps_pred,
                                 __half* __restrict__ x_prev,
                                 float inv_sqrt_alpha_t,
                                 float sqrt_one_minus_alpha_t,
                                 float sqrt_alpha_prev,
                                 float dir_coef,
                                 int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float xt = __half2float(x_t[i]);
    const float eps = __half2float(eps_pred[i]);
    const float x0_pred = (xt - sqrt_one_minus_alpha_t * eps) * inv_sqrt_alpha_t;
    const float dir = dir_coef * eps;
    const float xp = sqrt_alpha_prev * x0_pred + dir;
    x_prev[i] = __float2half(xp);
}

} // namespace

void ddim_step_gpu(const GpuTensor& x_t, const GpuTensor& eps_pred,
                   float alpha_t, float alpha_prev, float sigma_t,
                   GpuTensor& x_prev) {
    if (x_t.dtype != Dtype::FP16 || eps_pred.dtype != Dtype::FP16) {
        throw std::runtime_error("ddim_step_gpu: x_t and eps_pred must be FP16");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("ddim_step_gpu: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != Dtype::FP16) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    const int total = x_t.size();
    if (total == 0) return;

    // Precompute scalar coefficients on host in FP32.
    const float sqrt_alpha_t      = std::sqrt(alpha_t);
    const float inv_sqrt_alpha_t  = sqrt_alpha_t > 0.0f ? 1.0f / sqrt_alpha_t : 0.0f;
    const float sqrt_1m_alpha_t   = std::sqrt(std::max(0.0f, 1.0f - alpha_t));
    const float sqrt_alpha_prev   = std::sqrt(std::max(0.0f, alpha_prev));
    const float dir_inner         = 1.0f - alpha_prev - sigma_t * sigma_t;
    const float dir_coef          = std::sqrt(std::max(0.0f, dir_inner));

    const int blocks = (total + DDIM_BLOCK - 1) / DDIM_BLOCK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    ddim_step_kernel<<<blocks, DDIM_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half*>(x_t.data_fp16()),
        reinterpret_cast<const __half*>(eps_pred.data_fp16()),
        reinterpret_cast<__half*>(x_prev.data_fp16()),
        inv_sqrt_alpha_t, sqrt_1m_alpha_t, sqrt_alpha_prev, dir_coef,
        total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
