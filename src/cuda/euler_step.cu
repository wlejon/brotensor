// Fused Euler-discrete sampler step (FP16). ε-prediction; σ convention
// matching diffusers' EulerDiscreteScheduler.
//
//   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred
//
// One elementwise kernel; FP32 internal math.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int EULER_BLOCK = 256;

__global__ void euler_step_kernel(const __half* __restrict__ x_t,
                                  const __half* __restrict__ eps_pred,
                                  __half* __restrict__ x_prev,
                                  float dsigma,
                                  int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float xt  = __half2float(x_t[i]);
    const float eps = __half2float(eps_pred[i]);
    x_prev[i] = __float2half(xt + dsigma * eps);
}

} // namespace

void euler_step_gpu(const GpuTensor& x_t, const GpuTensor& eps_pred,
                    float sigma_t, float sigma_prev,
                    GpuTensor& x_prev) {
    if (x_t.dtype != Dtype::FP16 || eps_pred.dtype != Dtype::FP16) {
        throw std::runtime_error("euler_step_gpu: x_t and eps_pred must be FP16");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("euler_step_gpu: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != Dtype::FP16) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    const int total = x_t.size();
    if (total == 0) return;

    const float dsigma = sigma_prev - sigma_t;

    const int blocks = (total + EULER_BLOCK - 1) / EULER_BLOCK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    euler_step_kernel<<<blocks, EULER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half*>(x_t.data_fp16()),
        reinterpret_cast<const __half*>(eps_pred.data_fp16()),
        reinterpret_cast<__half*>(x_prev.data_fp16()),
        dsigma, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
