#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace brotensor {

namespace {

constexpr int TR_BLOCK = 256;

template <typename T>
__global__ void nchw_to_seq_kernel(const T* __restrict__ X,
                                   T* __restrict__ Y,
                                   int N, int C, int H, int W,
                                   int HW, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c     = idx % C;
    int t           = idx / C;
    const int p     = t % HW;
    const int n     = t / HW;
    const int x_idx = (n * C + c) * HW + p;
    Y[idx] = X[x_idx];
    (void)H; (void)W;
}

template <typename T>
__global__ void seq_to_nchw_kernel(const T* __restrict__ X,
                                   T* __restrict__ Y,
                                   int N, int C, int H, int W,
                                   int HW, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // idx walks output NCHW: idx = ((n*C + c)*H + h)*W + w
    const int p     = idx % HW;
    int t           = idx / HW;
    const int c     = t % C;
    const int n     = t / C;
    const int x_idx = (n * HW + p) * C + c;
    Y[idx] = X[x_idx];
    (void)H; (void)W;
}

inline int grid_for(int n) {
    int b = (n + TR_BLOCK - 1) / TR_BLOCK;
    if (b < 1) b = 1;
    return b;
}

void check_dims(const char* op, int N, int C, int H, int W) {
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error(std::string(op) + ": negative dimension");
    }
}

} // namespace

void nchw_to_sequence_gpu(const GpuTensor& X,
                          int N, int C, int H, int W,
                          GpuTensor& Y) {
    check_dims("nchw_to_sequence_gpu", N, C, H, W);
    const int HW = H * W;
    const int rows = N * HW;
    if (Y.rows != rows || Y.cols != C || Y.dtype != X.dtype) {
        Y.resize(rows, C, X.dtype);
    }
    const int total = rows * C;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        nchw_to_seq_kernel<__half><<<grid_for(total), TR_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<__half*>(Y.data_fp16()),
            N, C, H, W, HW, total);
    } else {
        nchw_to_seq_kernel<float><<<grid_for(total), TR_BLOCK>>>(
            X.data, Y.data, N, C, H, W, HW, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void sequence_to_nchw_gpu(const GpuTensor& X,
                          int N, int C, int H, int W,
                          GpuTensor& Y) {
    check_dims("sequence_to_nchw_gpu", N, C, H, W);
    const int HW = H * W;
    const int cols = C * HW;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    if (X.dtype == Dtype::FP16) {
        seq_to_nchw_kernel<__half><<<grid_for(total), TR_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data_fp16()),
            reinterpret_cast<__half*>(Y.data_fp16()),
            N, C, H, W, HW, total);
    } else {
        seq_to_nchw_kernel<float><<<grid_for(total), TR_BLOCK>>>(
            X.data, Y.data, N, C, H, W, HW, total);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
