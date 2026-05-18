// Fused DPM-Solver++ 2M sampler step (FP16). Multistep, ε-prediction.
//
// The caller maintains a running x0 cache. On step t the caller passes:
//   x_t       : current latent
//   eps_pred  : model output at step t (ε-prediction)
//   x0_prev   : cached denoised prediction from the previous step
// and three scalar linear-combination coefficients computed host-side from the
// scheduler's log-SNR / σ schedule. The kernel reconstructs x0_t = x_t -
// sigma_t * eps_pred, then evaluates:
//
//   x_prev    = c_xt * x_t + c_x0t * x0_t + c_x0prev * x0_prev
//   x0_out    = x0_t          (caller swaps it into x0_prev for the next step)
//
// Coefficient derivation (k-diffusion / DPM++ 2M, ε-prediction, α≡1):
//   h_last = lambda_t - lambda_last,  h = lambda_next - lambda_t,  r = h_last/h
//   D_t    = (1 + 1/(2r)) * x0_t - (1/(2r)) * x0_prev
//   x_prev = (sigma_next/sigma_t) * x_t - (exp(-h) - 1) * D_t
// →  c_xt     = sigma_next / sigma_t
//    c_x0t    = -(exp(-h) - 1) * (1 + 1/(2r))
//    c_x0prev = -(exp(-h) - 1) * (-1/(2r))
//
// First step (no x0_prev): use euler_step_gpu instead.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int DPMPP_BLOCK = 256;

__global__ void dpmpp_2m_step_kernel(const __half* __restrict__ x_t,
                                     const __half* __restrict__ eps_pred,
                                     const __half* __restrict__ x0_prev,
                                     __half* __restrict__ x_prev,
                                     __half* __restrict__ x0_out,
                                     float sigma_t,
                                     float c_xt, float c_x0t, float c_x0prev,
                                     int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float xt   = __half2float(x_t[i]);
    const float eps  = __half2float(eps_pred[i]);
    const float x0p  = __half2float(x0_prev[i]);
    const float x0t  = xt - sigma_t * eps;
    const float xp   = c_xt * xt + c_x0t * x0t + c_x0prev * x0p;
    x_prev[i] = __float2half(xp);
    x0_out[i] = __float2half(x0t);
}

} // namespace

void dpmpp_2m_step_gpu(const GpuTensor& x_t, const GpuTensor& eps_pred,
                       const GpuTensor& x0_prev,
                       float sigma_t,
                       float c_xt, float c_x0t, float c_x0prev,
                       GpuTensor& x_prev, GpuTensor& x0_out) {
    if (x_t.dtype != Dtype::FP16 || eps_pred.dtype != Dtype::FP16 ||
        x0_prev.dtype != Dtype::FP16) {
        throw std::runtime_error("dpmpp_2m_step_gpu: all inputs must be FP16");
    }
    if (x_t.rows != eps_pred.rows || x_t.cols != eps_pred.cols ||
        x_t.rows != x0_prev.rows  || x_t.cols != x0_prev.cols) {
        throw std::runtime_error("dpmpp_2m_step_gpu: shape mismatch");
    }
    if (x_prev.rows != x_t.rows || x_prev.cols != x_t.cols || x_prev.dtype != Dtype::FP16) {
        x_prev.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    if (x0_out.rows != x_t.rows || x0_out.cols != x_t.cols || x0_out.dtype != Dtype::FP16) {
        x0_out.resize(x_t.rows, x_t.cols, Dtype::FP16);
    }
    const int total = x_t.size();
    if (total == 0) return;

    const int blocks = (total + DPMPP_BLOCK - 1) / DPMPP_BLOCK;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    dpmpp_2m_step_kernel<<<blocks, DPMPP_BLOCK, 0, stream>>>(
        reinterpret_cast<const __half*>(x_t.data_fp16()),
        reinterpret_cast<const __half*>(eps_pred.data_fp16()),
        reinterpret_cast<const __half*>(x0_prev.data_fp16()),
        reinterpret_cast<__half*>(x_prev.data_fp16()),
        reinterpret_cast<__half*>(x0_out.data_fp16()),
        sigma_t, c_xt, c_x0t, c_x0prev, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
