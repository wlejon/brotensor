// Sinusoidal timestep embedding (FP32). Matches diffusers'
// get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0
// — the SD / SDXL default.
//
//   half      = dim / 2
//   freqs[j]  = exp(-log(max_period) * j / half)
//   args[i,j] = timesteps[i] * freqs[j]
//   Y[i, 0:half]      = cos(args[i, :])
//   Y[i, half:2*half] = sin(args[i, :])
//   if dim is odd: Y[i, dim-1] = 0
//
// Used for the diffusion timestep itself and for SDXL's added-cond
// micro-conditioning vector (original_size / crop / target_size).

#include <brotensor/runtime.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr int TE_BLOCK = 256;

__global__ void timestep_embedding_kernel(const float* __restrict__ ts,
                                          float* __restrict__ Y,
                                          int N, int dim, int half,
                                          float log_max_period) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * dim;
    if (idx >= total) return;
    const int i = idx / dim;
    const int j = idx - i * dim;
    if (j >= 2 * half) {
        // Odd-dim tail slot.
        Y[idx] = 0.0f;
        return;
    }
    const int k = j < half ? j : j - half;
    const float freq = expf(-log_max_period * (float)k / (float)half);
    const float arg  = ts[i] * freq;
    Y[idx] = j < half ? cosf(arg) : sinf(arg);
}

} // namespace

void timestep_embedding(const ::brotensor::Tensor& timesteps,
                        int dim, float max_period,
                        ::brotensor::Tensor& Y) {
    if (timesteps.dtype != Dtype::FP32) {
        throw std::runtime_error("timestep_embedding: timesteps must be FP32");
    }
    if (timesteps.cols != 1) {
        throw std::runtime_error("timestep_embedding: timesteps must be (N,1)");
    }
    if (dim <= 0) {
        throw std::runtime_error("timestep_embedding: dim must be positive");
    }
    const int N = timesteps.rows;
    if (Y.rows != N || Y.cols != dim || Y.dtype != Dtype::FP32) {
        Y.resize(N, dim, Dtype::FP32);
    }
    if (N == 0) return;

    const int half = dim / 2;
    const float log_max_period = std::log(max_period);
    const int total = N * dim;
    const int blocks = (total + TE_BLOCK - 1) / TE_BLOCK;
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    timestep_embedding_kernel<<<blocks, TE_BLOCK, 0, stream>>>(
        static_cast<const float*>(timesteps.data),
        static_cast<float*>(Y.data), N, dim, half, log_max_period);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
