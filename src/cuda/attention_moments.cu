// Per-text-token spatial moments of an FP16 cross-attention map. One block
// per text token k; threads stride over q = y * w_lat + x and reduce three
// running sums (mass, y-mass, x-mass) in shared memory; thread 0 divides and
// writes the (mass, centroid_y, centroid_x) outputs.

#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

// Forward declaration of the CUDA-internal stream getter. Defined in
// src/cuda/runtime.cu — kept out of the public header so non-GPU consumers
// don't pull cuda_runtime.h transitively.
void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr int MOM_BLOCK = 256;
constexpr float MOM_EPS = 1e-8f;

__global__ void attention_token_moments_kernel(const __half* __restrict__ Attn,
                                               int Lq, int Lk,
                                               int h_lat, int w_lat,
                                               float* __restrict__ mass,
                                               float* __restrict__ centroid) {
    __shared__ float sm_m[MOM_BLOCK];
    __shared__ float sm_y[MOM_BLOCK];
    __shared__ float sm_x[MOM_BLOCK];

    const int k = blockIdx.x;
    const int tid = threadIdx.x;
    if (k >= Lk) return;

    float am = 0.0f, ay = 0.0f, ax = 0.0f;
    for (int q = tid; q < Lq; q += blockDim.x) {
        const float a = __half2float(Attn[q * Lk + k]);
        const int y = q / w_lat;
        const int x = q - y * w_lat;
        am += a;
        ay += static_cast<float>(y) * a;
        ax += static_cast<float>(x) * a;
    }
    sm_m[tid] = am;
    sm_y[tid] = ay;
    sm_x[tid] = ax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm_m[tid] += sm_m[tid + s];
            sm_y[tid] += sm_y[tid + s];
            sm_x[tid] += sm_x[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float m = sm_m[0];
        mass[k] = m;
        if (m > MOM_EPS) {
            const float inv = 1.0f / m;
            centroid[k * 2 + 0] = sm_y[0] * inv;
            centroid[k * 2 + 1] = sm_x[0] * inv;
        } else {
            centroid[k * 2 + 0] = 0.0f;
            centroid[k * 2 + 1] = 0.0f;
        }
    }
}

} // namespace

void attention_token_moments(const ::brotensor::Tensor& Attn,
                             int h_lat, int w_lat,
                             ::brotensor::Tensor& mass,
                             ::brotensor::Tensor& centroid) {
    using ::brotensor::Dtype;
    if (Attn.dtype != Dtype::FP16) {
        throw std::runtime_error("attention_token_moments: Attn must be FP16");
    }
    if (h_lat <= 0 || w_lat <= 0) {
        throw std::runtime_error("attention_token_moments: h_lat and w_lat must be positive");
    }
    const int Lq = h_lat * w_lat;
    const int Lk = Attn.cols;
    if (Attn.rows != Lq) {
        throw std::runtime_error("attention_token_moments: Attn.rows must equal h_lat * w_lat");
    }
    if (mass.rows != Lk || mass.cols != 1 || mass.dtype != Dtype::FP32) {
        mass.resize(Lk, 1, Dtype::FP32);
    }
    if (centroid.rows != Lk || centroid.cols != 2 || centroid.dtype != Dtype::FP32) {
        centroid.resize(Lk, 2, Dtype::FP32);
    }
    if (Lk == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    attention_token_moments_kernel<<<Lk, MOM_BLOCK, 0, stream>>>(
        static_cast<const __half*>(Attn.data),
        Lq, Lk, h_lat, w_lat,
        static_cast<float*>(mass.data),
        static_cast<float*>(centroid.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
