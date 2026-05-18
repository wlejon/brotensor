// Rotary position embedding (RoPE) forward + backward.
// Per-head: rotate pairs (x_{2i}, x_{2i+1}) by angle theta = pos * base^{-2i/hd}.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>
#include <stdexcept>

namespace brotensor {

namespace {

constexpr int RP_BLOCK = 256;

__device__ inline float rope_theta(int pair_i, int head_dim, float base) {
    // theta_i = base^{-2i/head_dim} = exp(-2i/hd * log(base))
    return __expf(-static_cast<float>(2 * pair_i) /
                  static_cast<float>(head_dim) * __logf(base));
}

// One thread per pair (row, head, i). Forward.
__global__ void rope_forward_fp32_kernel(const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         int L, int num_heads, int head_dim,
                                         int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i      = idx % half;
        const int rest   = idx / half;
        const int h      = rest % num_heads;
        const int row    = rest / num_heads;
        const int pos    = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c    = __cosf(theta);
        const float s    = __sinf(theta);
        const int D      = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float x0 = X[base_off + 2 * i];
        const float x1 = X[base_off + 2 * i + 1];
        Y[base_off + 2 * i]     = x0 * c - x1 * s;
        Y[base_off + 2 * i + 1] = x0 * s + x1 * c;
    }
}

__global__ void rope_forward_fp16_kernel(const __half* __restrict__ X,
                                         __half* __restrict__ Y,
                                         int L, int num_heads, int head_dim,
                                         int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int pos  = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c  = __cosf(theta);
        const float s  = __sinf(theta);
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float x0 = __half2float(X[base_off + 2 * i]);
        const float x1 = __half2float(X[base_off + 2 * i + 1]);
        Y[base_off + 2 * i]     = __float2half(x0 * c - x1 * s);
        Y[base_off + 2 * i + 1] = __float2half(x0 * s + x1 * c);
    }
}

__global__ void rope_backward_fp32_kernel(const float* __restrict__ dY,
                                          float* __restrict__ dX,
                                          int L, int num_heads, int head_dim,
                                          int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int pos  = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c  = __cosf(theta);
        const float s  = __sinf(theta);
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float dy0 = dY[base_off + 2 * i];
        const float dy1 = dY[base_off + 2 * i + 1];
        // Inverse rotation (transpose of R(θ)).
        dX[base_off + 2 * i]     = dy0 * c + dy1 * s;
        dX[base_off + 2 * i + 1] = -dy0 * s + dy1 * c;
    }
}

__global__ void rope_backward_fp16_kernel(const __half* __restrict__ dY,
                                          __half* __restrict__ dX,
                                          int L, int num_heads, int head_dim,
                                          int seq_offset, float base) {
    const int half = head_dim / 2;
    const int total = L * num_heads * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const int i    = idx % half;
        const int rest = idx / half;
        const int h    = rest % num_heads;
        const int row  = rest / num_heads;
        const int pos  = row + seq_offset;
        const float theta = static_cast<float>(pos) * rope_theta(i, head_dim, base);
        const float c  = __cosf(theta);
        const float s  = __sinf(theta);
        const int D    = num_heads * head_dim;
        const int base_off = row * D + h * head_dim;
        const float dy0 = __half2float(dY[base_off + 2 * i]);
        const float dy1 = __half2float(dY[base_off + 2 * i + 1]);
        dX[base_off + 2 * i]     = __float2half(dy0 * c + dy1 * s);
        dX[base_off + 2 * i + 1] = __float2half(-dy0 * s + dy1 * c);
    }
}

inline int grid_for(int n) { return (n + RP_BLOCK - 1) / RP_BLOCK; }

} // namespace

void rope_forward_gpu(const GpuTensor& X, int head_dim, int num_heads,
                     int seq_offset, float theta_base, GpuTensor& Y) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_forward_gpu: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_forward_gpu: num_heads must be positive");
    }
    if (X.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_forward_gpu: X.cols != num_heads * head_dim");
    }
    const int L = X.rows;
    if (Y.rows != L || Y.cols != X.cols || Y.dtype != X.dtype) {
        Y.resize(L, X.cols, X.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;
    const int blocks = grid_for(total);
    if (X.dtype == Dtype::FP16) {
        rope_forward_fp16_kernel<<<blocks, RP_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<__half*>(Y.data_fp16()),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else {
        rope_forward_fp32_kernel<<<blocks, RP_BLOCK>>>(
            X.data, Y.data, L, num_heads, head_dim, seq_offset, theta_base);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rope_backward_gpu(const GpuTensor& dY, int head_dim, int num_heads,
                      int seq_offset, float theta_base, GpuTensor& dX) {
    if (head_dim <= 0 || (head_dim & 1) != 0) {
        throw std::runtime_error("rope_backward_gpu: head_dim must be a positive even integer");
    }
    if (num_heads <= 0) {
        throw std::runtime_error("rope_backward_gpu: num_heads must be positive");
    }
    if (dY.cols != num_heads * head_dim) {
        throw std::runtime_error("rope_backward_gpu: dY.cols != num_heads * head_dim");
    }
    const int L = dY.rows;
    if (dX.rows != L || dX.cols != dY.cols || dX.dtype != dY.dtype) {
        dX.resize(L, dY.cols, dY.dtype);
    }
    const int total = L * num_heads * (head_dim / 2);
    if (total == 0) return;
    const int blocks = grid_for(total);
    if (dY.dtype == Dtype::FP16) {
        rope_backward_fp16_kernel<<<blocks, RP_BLOCK>>>(
            reinterpret_cast<const __half*>(dY.data_fp16()),
            reinterpret_cast<__half*>(dX.data_fp16()),
            L, num_heads, head_dim, seq_offset, theta_base);
    } else {
        rope_backward_fp32_kernel<<<blocks, RP_BLOCK>>>(
            dY.data, dX.data, L, num_heads, head_dim, seq_offset, theta_base);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
