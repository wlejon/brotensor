#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int GN_BLOCK = 256;

// One block per (sample, group) tile. Threads cooperatively reduce
// sum and sum-of-squares over the tile (C/num_groups * H * W elements),
// then normalize using per-channel gamma/beta. FP32 accumulation,
// FP16 storage.
__global__ void group_norm_forward_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ gamma,
        const __half* __restrict__ beta,
        __half* __restrict__ Y,
        int C, int spatial,
        int channels_per_group,
        int num_groups,
        float eps) {
    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    const int tile_channels = channels_per_group;
    const int tile_size = tile_channels * spatial;
    const int chan_base = g * channels_per_group;

    // Pointer to first element of this tile in X / Y.
    const int sample_stride = C * spatial;
    const __half* x_tile = X + n * sample_stride + chan_base * spatial;
    __half*       y_tile = Y + n * sample_stride + chan_base * spatial;

    // First pass: sum and sum-of-squares.
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = __half2float(x_tile[i]);
        sum   += v;
        sumsq += v * v;
    }

    __shared__ float s_sum[GN_BLOCK];
    __shared__ float s_sumsq[GN_BLOCK];
    s_sum[tid]   = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float s_mean;
    __shared__ float s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean  = s_sum[0] * inv_n;
        const float var   = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    // Second pass: normalize, scale, shift.
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;        // [0, channels_per_group)
        const int channel = chan_base + local_c;
        const float gv = __half2float(gamma[channel]);
        const float bv = __half2float(beta[channel]);
        const float v  = __half2float(x_tile[i]);
        const float yn = (v - mean) * rstd;
        y_tile[i] = __float2half(yn * gv + bv);
    }
    (void)num_groups;
}

} // namespace

void group_norm_forward_gpu(const GpuTensor& X,
                            const GpuTensor& gamma,
                            const GpuTensor& beta,
                            int N, int C, int H, int W,
                            int num_groups,
                            float eps,
                            GpuTensor& Y) {
    if (X.dtype != Dtype::FP16 || gamma.dtype != Dtype::FP16 ||
        beta.dtype != Dtype::FP16) {
        throw std::runtime_error("group_norm_forward_gpu: all tensors must be FP16");
    }
    if (num_groups <= 0 || C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward_gpu: num_groups must divide C");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, cols, Dtype::FP16);
    }
    if (N == 0 || cols == 0) return;

    const int channels_per_group = C / num_groups;
    dim3 grid(num_groups, N, 1);
    group_norm_forward_kernel<<<grid, GN_BLOCK>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<const __half*>(gamma.data_fp16()),
        reinterpret_cast<const __half*>(beta.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        C, spatial, channels_per_group, num_groups, eps);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
