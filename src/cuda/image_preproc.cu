// CUDA image preprocessing helpers (FP32).
//
//   image_normalize           — per-channel (X - mean[c]) / std[c] on NCHW.
//                               One element per thread; per-channel mean / std
//                               are looked up from the C-length tables.
//
//   image_u8_to_f32_nhwc_to_nchw — convert packed uint8 NHWC into FP32 NCHW
//                               with Y = src * scale + bias. On CUDA the
//                               `src` pointer is a *device* pointer (same
//                               convention as embedding_lookup_forward's
//                               `const int32_t* d_idx`).

#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int BLK = 256;

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + name +
                                 " must be FP32");
    }
}

// One thread per element of NCHW Y. We can decode (n, c, h, w) from the
// linear index, or even simpler: (linear / spatial) % C → channel index.
__global__ void image_normalize_kernel(const float* __restrict__ X,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ inv_std,
                                       float* __restrict__ Y,
                                       int total, int C, int spatial) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (i / spatial) % C;
    Y[i] = (X[i] - mean[c]) * inv_std[c];
}

__global__ void image_inv_std_kernel(const float* __restrict__ std_,
                                     float* __restrict__ inv,
                                     int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    // We'd like to throw on std==0 like the CPU does, but throwing from a
    // device kernel isn't a thing. Producing inf via division is the closest
    // diagnostic — and we don't expect a 0-std preprocess in practice.
    inv[c] = 1.0f / std_[c];
}

// One thread per output element. Maps Y[(n, c, h, w)] back to src[n*H*W*C +
// (h*W + w)*C + c].
__global__ void image_u8_to_f32_nhwc_to_nchw_kernel(
        const uint8_t* __restrict__ src,
        float* __restrict__ Y,
        int N, int H, int W, int C,
        float scale, float bias) {
    const int total = N * C * H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int spatial = H * W;
    // Decode (n, c, h, w) from NCHW linear index.
    const int n  =  idx / (C * spatial);
    const int rm =  idx % (C * spatial);
    const int c  =  rm / spatial;
    const int s  =  rm % spatial;
    const int h  =  s / W;
    const int w  =  s % W;
    const int src_idx = ((n * H + h) * W + w) * C + c;
    Y[idx] = static_cast<float>(src[src_idx]) * scale + bias;
}

} // namespace

void image_normalize(const ::brotensor::Tensor& X,
                     const ::brotensor::Tensor& mean,
                     const ::brotensor::Tensor& std_,
                     int N, int C, int H, int W,
                     ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    check_fp32(X,    "image_normalize", "X");
    check_fp32(mean, "image_normalize", "mean");
    check_fp32(std_, "image_normalize", "std");
    if (mean.size() != C || std_.size() != C) {
        throw std::runtime_error("brotensor: image_normalize: mean/std must have C elements");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (X.rows != N || X.cols != cols) {
        throw std::runtime_error("brotensor: image_normalize: X shape mismatch");
    }
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    // Precompute 1/std on device. (Cheap C-length scratch; avoids redundant
    // division in the per-element kernel.)
    float* d_inv = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inv),
                                    C * sizeof(float)));
    image_inv_std_kernel<<<(C + BLK - 1) / BLK, BLK, 0, cur_stream()>>>(
        reinterpret_cast<const float*>(std_.data), d_inv, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    const int total = N * cols;
    image_normalize_kernel<<<(total + BLK - 1) / BLK, BLK, 0, cur_stream()>>>(
        reinterpret_cast<const float*>(X.data),
        reinterpret_cast<const float*>(mean.data),
        d_inv,
        reinterpret_cast<float*>(Y.data),
        total, C, spatial);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    cudaFree(d_inv);
}

void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  ::brotensor::Tensor& Y) {
    using ::brotensor::Dtype;
    if (src == nullptr && N > 0 && H > 0 && W > 0 && C > 0) {
        throw std::runtime_error("brotensor: image_u8_to_f32_nhwc_to_nchw: src is null");
    }
    if (N < 0 || H < 0 || W < 0 || C < 0) {
        throw std::runtime_error("brotensor: image_u8_to_f32_nhwc_to_nchw: negative dim");
    }
    const int spatial = H * W;
    const int cols = C * spatial;
    if (Y.rows != N || Y.cols != cols || Y.dtype != Dtype::FP32) {
        Y.resize(N, cols, Dtype::FP32);
    }
    if (N == 0 || cols == 0) return;

    const int total = N * cols;
    image_u8_to_f32_nhwc_to_nchw_kernel<<<(total + BLK - 1) / BLK, BLK, 0, cur_stream()>>>(
        src,
        reinterpret_cast<float*>(Y.data),
        N, H, W, C, scale, bias);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fill_cuda_vtable_image_preproc(::brotensor::detail::OpsVTable& v) {
    v.image_normalize              = &image_normalize;
    v.image_u8_to_f32_nhwc_to_nchw = &image_u8_to_f32_nhwc_to_nchw;
}

} // namespace brotensor::detail::cuda
