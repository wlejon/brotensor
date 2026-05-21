// Fused first-order Euler sampler step (FP16 / BF16).
//
//   x_prev = x_t + (sigma_prev - sigma_t) * eps_pred
//
// The kernel never interprets `eps_pred` — it covers both ε / k-diffusion
// EulerDiscreteScheduler (pass the derivative) and flow-matching / rectified-
// flow velocity prediction (pass the velocity v; the update x += dσ·v is the
// same formula). See the euler_step doc in <brotensor/ops.h>.
// One elementwise kernel; FP32 internal math.

#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

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

__global__ void euler_step_bf16_kernel(const __nv_bfloat16* __restrict__ x_t,
                                       const __nv_bfloat16* __restrict__ eps_pred,
                                       __nv_bfloat16* __restrict__ x_prev,
                                       float dsigma,
                                       int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float xt  = __bfloat162float(x_t[i]);
    const float eps = __bfloat162float(eps_pred[i]);
    x_prev[i] = __float2bfloat16(xt + dsigma * eps);
}

} // namespace

void euler_step(const ::brotensor::Tensor& x_t,
                const ::brotensor::Tensor& eps_pred,
                float sigma_t, float sigma_prev,
                ::brotensor::Tensor& x_prev) {
    const Dtype dtype = x_t.dtype;
    if (dtype != Dtype::FP16 && dtype != Dtype::BF16) {
        throw std::runtime_error("euler_step: x_t and eps_pred must be FP16 or BF16");
    }
    if (eps_pred.dtype != dtype) {
        throw std::runtime_error("euler_step: x_t and eps_pred must have the same dtype");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols) {
        throw std::runtime_error("euler_step: shape mismatch between x_t and eps_pred");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != dtype) {
        x_prev.resize(x_t.rows, x_t.cols, dtype);
    }
    const int total = x_t.size();
    if (total == 0) return;

    const float dsigma = sigma_prev - sigma_t;

    const int blocks = (total + EULER_BLOCK - 1) / EULER_BLOCK;
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    if (dtype == Dtype::FP16) {
        euler_step_kernel<<<blocks, EULER_BLOCK, 0, stream>>>(
            static_cast<const __half*>(x_t.data),
            static_cast<const __half*>(eps_pred.data),
            static_cast<__half*>(x_prev.data),
            dsigma, total);
    } else {
        euler_step_bf16_kernel<<<blocks, EULER_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_t.data),
            static_cast<const __nv_bfloat16*>(eps_pred.data),
            static_cast<__nv_bfloat16*>(x_prev.data),
            dsigma, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
