// CUDA spatial 2x2 pixel-unshuffle (Qwen-VL merger / Flux.2 VAE).
//
// Pure gather: (N, C, H, W) -> (N, 4*C, H/2, W/2), stacking each 2x2 block
// into the channel axis. channel_major selects the output channel ordering
// (block-major Qwen-VL vs channel-major torch pixel_unshuffle). Supports
// FP32 / FP16 / BF16; one thread per output element. Registered through
// fill_cuda_vtable_qwen3_vl_polish (rope_mrope.cu).

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

constexpr int SM_BLOCK = 256;

template <typename T>
__global__ void spatial_merge_2x2_kernel(const T* __restrict__ X,
                                         T* __restrict__ Y,
                                         int N, int C, int H, int W,
                                         int H_out, int W_out,
                                         int total, bool channel_major) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // Output NCHW walk: idx = ((n*C_out + c_out)*H_out + h_out)*W_out + w_out.
    const int C_out = 4 * C;
    const int w_out = idx % W_out;
    int t           = idx / W_out;
    const int h_out = t % H_out;
    t              /= H_out;
    const int c_out = t % C_out;
    const int n     = t / C_out;
    // block = dh*2+dw. channel_major: c_out = c_in*4 + block; else block*C + c_in.
    const int block = channel_major ? (c_out & 3)  : (c_out / C);
    const int c_in  = channel_major ? (c_out >> 2) : (c_out - block * C);
    const int dh    = block >> 1;
    const int dw    = block & 1;
    const int h_in  = 2 * h_out + dh;
    const int w_in  = 2 * w_out + dw;
    const int HW    = H * W;
    const int x_idx = (n * C + c_in) * HW + h_in * W + w_in;
    Y[idx] = X[x_idx];
    (void)N;
}

inline int grid_for(int n) { return (n + SM_BLOCK - 1) / SM_BLOCK; }

} // namespace

void spatial_merge_2x2_forward(const ::brotensor::Tensor& X,
                               int N, int C, int H, int W,
                               bool channel_major,
                               ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("spatial_merge_2x2_forward: X must be FP32, "
                                 "FP16, or BF16");
    }
    if (N < 0 || C < 0 || H < 0 || W < 0) {
        throw std::runtime_error("spatial_merge_2x2_forward: negative dimension");
    }
    if ((H & 1) != 0 || (W & 1) != 0) {
        throw std::runtime_error("spatial_merge_2x2_forward: H and W must be even");
    }
    const int H_out = H / 2;
    const int W_out = W / 2;
    const int C_out = 4 * C;
    const int cols  = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    const int blocks = grid_for(total);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    switch (X.dtype) {
    case Dtype::FP16:
        spatial_merge_2x2_kernel<__half><<<blocks, SM_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data),
            static_cast<__half*>(Y.data),
            N, C, H, W, H_out, W_out, total, channel_major);
        break;
    case Dtype::BF16:
        spatial_merge_2x2_kernel<__nv_bfloat16><<<blocks, SM_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C, H, W, H_out, W_out, total, channel_major);
        break;
    default:  // FP32
        spatial_merge_2x2_kernel<float><<<blocks, SM_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data),
            static_cast<float*>(Y.data),
            N, C, H, W, H_out, W_out, total, channel_major);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── DC-AE up-shortcut: repeat_interleave + 2x pixel-shuffle ────────────────
//
// Y[n, c_out, 2h+i, 2w+j] = X[n, (4*c_out + 2i + j)/repeats, h, w],
// repeats = 4*C_out/C_in. Pure gather; one thread per output element.

namespace {

template <typename T>
__global__ void pixel_shuffle_upsample_2x_kernel(const T* __restrict__ X,
                                                 T* __restrict__ Y,
                                                 int N, int C_in, int H, int W,
                                                 int C_out, int repeats,
                                                 int H_out, int W_out,
                                                 int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int w_out = idx % W_out;
    int t           = idx / W_out;
    const int h_out = t % H_out;
    t              /= H_out;
    const int c_out = t % C_out;
    const int n     = t / C_out;
    const int i = h_out & 1;
    const int j = w_out & 1;
    const int h = h_out >> 1;
    const int w = w_out >> 1;
    const int src_c = (4 * c_out + 2 * i + j) / repeats;
    const int x_idx = (n * C_in + src_c) * (H * W) + h * W + w;
    Y[idx] = X[x_idx];
}

} // namespace

void pixel_shuffle_upsample_2x_forward(const ::brotensor::Tensor& X,
                                       int N, int C_in, int H, int W,
                                       int C_out,
                                       ::brotensor::Tensor& Y) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 &&
        X.dtype != Dtype::BF16) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: X must be "
                                 "FP32, FP16, or BF16");
    }
    if (N < 0 || C_in <= 0 || H < 0 || W < 0 || C_out <= 0) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: bad dimension");
    }
    if ((4 * C_out) % C_in != 0) {
        throw std::runtime_error("pixel_shuffle_upsample_2x_forward: C_in must "
                                 "divide 4*C_out");
    }
    const int repeats = (4 * C_out) / C_in;
    const int H_out = 2 * H;
    const int W_out = 2 * W;
    const int cols  = C_out * H_out * W_out;
    if (Y.rows != N || Y.cols != cols || Y.dtype != X.dtype) {
        Y.resize(N, cols, X.dtype);
    }
    const int total = N * cols;
    if (total == 0) return;
    const int blocks = grid_for(total);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    switch (X.dtype) {
    case Dtype::FP16:
        pixel_shuffle_upsample_2x_kernel<__half><<<blocks, SM_BLOCK, 0, stream>>>(
            static_cast<const __half*>(X.data), static_cast<__half*>(Y.data),
            N, C_in, H, W, C_out, repeats, H_out, W_out, total);
        break;
    case Dtype::BF16:
        pixel_shuffle_upsample_2x_kernel<__nv_bfloat16><<<blocks, SM_BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(X.data),
            static_cast<__nv_bfloat16*>(Y.data),
            N, C_in, H, W, C_out, repeats, H_out, W_out, total);
        break;
    default:  // FP32
        pixel_shuffle_upsample_2x_kernel<float><<<blocks, SM_BLOCK, 0, stream>>>(
            static_cast<const float*>(X.data), static_cast<float*>(Y.data),
            N, C_in, H, W, C_out, repeats, H_out, W_out, total);
        break;
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda
} // namespace brotensor
