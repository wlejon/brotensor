#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int RS_BLOCK = 256;

__global__ void upsample_nearest_2x_kernel(const __half* __restrict__ X,
                                           __half* __restrict__ Y,
                                           int N, int C, int H, int W,
                                           int H_out, int W_out,
                                           int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;
    const int ih = oh / 2;
    const int iw = ow / 2;
    const int in_idx  = ((n * C + c) * H + ih) * W + iw;
    Y[idx] = X[in_idx];
    (void)H_out; (void)W_out;
}

__global__ void upsample_bilinear_2x_kernel(const __half* __restrict__ X,
                                            __half* __restrict__ Y,
                                            int N, int C, int H, int W,
                                            int H_out, int W_out,
                                            int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;

    // align_corners=False mapping with scale=2:
    //   src = (dst + 0.5) / 2 - 0.5
    const float src_y = (oh + 0.5f) * 0.5f - 0.5f;
    const float src_x = (ow + 0.5f) * 0.5f - 0.5f;
    const int y0 = static_cast<int>(floorf(src_y));
    const int x0 = static_cast<int>(floorf(src_x));
    const float fy = src_y - y0;
    const float fx = src_x - x0;
    const int y0c = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);
    const int x0c = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);
    const int y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= H ? H - 1 : (y0 + 1));
    const int x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= W ? W - 1 : (x0 + 1));

    const int base = (n * C + c) * H;
    const float v00 = __half2float(X[(base + y0c) * W + x0c]);
    const float v01 = __half2float(X[(base + y0c) * W + x1c]);
    const float v10 = __half2float(X[(base + y1c) * W + x0c]);
    const float v11 = __half2float(X[(base + y1c) * W + x1c]);
    const float top = v00 + (v01 - v00) * fx;
    const float bot = v10 + (v11 - v10) * fx;
    const float v   = top + (bot - top) * fy;
    Y[idx] = __float2half(v);
    (void)H_out; (void)W_out;
}

__global__ void downsample_avg_2x_kernel(const __half* __restrict__ X,
                                         __half* __restrict__ Y,
                                         int N, int C, int H, int W,
                                         int H_out, int W_out,
                                         int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ow = idx % W_out;
    int t = idx / W_out;
    const int oh = t % H_out;
    t /= H_out;
    const int c  = t % C;
    const int n  = t / C;
    const int ih = oh * 2;
    const int iw = ow * 2;
    const int base = ((n * C + c) * H + ih) * W + iw;
    const float v00 = __half2float(X[base]);
    const float v01 = __half2float(X[base + 1]);
    const float v10 = __half2float(X[base + W]);
    const float v11 = __half2float(X[base + W + 1]);
    Y[idx] = __float2half(0.25f * (v00 + v01 + v10 + v11));
    (void)H_out; (void)W_out;
}

inline int grid_for(int n) {
    int b = (n + RS_BLOCK - 1) / RS_BLOCK;
    if (b < 1) b = 1;
    return b;
}

} // namespace

void upsample_nearest_2x_gpu(const GpuTensor& X,
                             int N, int C, int H, int W,
                             GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("upsample_nearest_2x_gpu: X must be FP16");
    }
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, cols, Dtype::FP16);
    }
    const int total = N * cols;
    if (total == 0) return;
    upsample_nearest_2x_kernel<<<grid_for(total), RS_BLOCK>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        N, C, H, W, H_out, W_out, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void upsample_bilinear_2x_gpu(const GpuTensor& X,
                              int N, int C, int H, int W,
                              GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("upsample_bilinear_2x_gpu: X must be FP16");
    }
    const int H_out = 2 * H, W_out = 2 * W;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, cols, Dtype::FP16);
    }
    const int total = N * cols;
    if (total == 0) return;
    upsample_bilinear_2x_kernel<<<grid_for(total), RS_BLOCK>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        N, C, H, W, H_out, W_out, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void downsample_avg_2x_gpu(const GpuTensor& X,
                           int N, int C, int H, int W,
                           GpuTensor& Y) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("downsample_avg_2x_gpu: X must be FP16");
    }
    if ((H & 1) || (W & 1)) {
        throw std::runtime_error("downsample_avg_2x_gpu: H and W must be even");
    }
    const int H_out = H / 2, W_out = W / 2;
    const int cols = C * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP16) {
        Y.resize(N, cols, Dtype::FP16);
    }
    const int total = N * cols;
    if (total == 0) return;
    downsample_avg_2x_kernel<<<grid_for(total), RS_BLOCK>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        N, C, H, W, H_out, W_out, total);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor
