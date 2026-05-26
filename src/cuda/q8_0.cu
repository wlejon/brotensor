// CUDA Q8_0 (W8A16-style) dequant + GEMV. Q8_0 block = 34 bytes / 32
// elements: fp16 d + int8 qs[32]. y = d * qs[i]. Dequant kernel uses one
// CTA per (row, block) with 32 threads (one per element). GEMV kernel uses
// one CTA per output row with 32 threads (one warp), looping super-blocks
// along K with warp-shuffle reduction at the end.

#include "detail/cuda_check.h"
#include "q8_0_internal.cuh"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void* cuda_current_stream();

namespace q8_0_wmma_internal {
bool launch_linear_q8_0_fp16_wmma(const __half* X, const uint8_t* W_q8,
                                  const __half* bias, __half* Y,
                                  int B, int M, int K, cudaStream_t stream);
}

namespace {

constexpr int Q8_BLOCK_ELEMS = q8_0::kBlockElems;
constexpr int Q8_BLOCK_BYTES = q8_0::kBlockBytes;

// One CTA per (row, block). 32 threads, one per element.
__global__ void dequant_q8_0_to_fp16_kernel(const uint8_t* __restrict__ W,
                                            __half* __restrict__ Wfp16,
                                            int rows, int blocks_per_row) {
    const int row = blockIdx.y;
    const int sb  = blockIdx.x;
    const int t   = threadIdx.x;

    const uint8_t* blk = W + (static_cast<size_t>(row) * blocks_per_row + sb) * Q8_BLOCK_BYTES;
    const __half d_h = *reinterpret_cast<const __half*>(blk + q8_0::kDOffset);
    const float  d_f = __half2float(d_h);
    const int8_t q   = static_cast<int8_t>(blk[q8_0::kQsOffset + t]);

    Wfp16[static_cast<size_t>(row) * (blocks_per_row * Q8_BLOCK_ELEMS)
          + sb * Q8_BLOCK_ELEMS + t] = __float2half_rn(d_f * static_cast<float>(q));
}

// One CTA per output row; one warp (32 threads) loops blocks along K.
__global__ void linear_q8_0_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                             const __half*  __restrict__ x,
                                             const __half*  __restrict__ bias,
                                             __half*        __restrict__ y,
                                             int K, int blocks_per_row) {
    const int row = blockIdx.x;
    const int t   = threadIdx.x;

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q8_BLOCK_BYTES;

    for (int sb = 0; sb < blocks_per_row; ++sb) {
        const uint8_t* blk = row_base + sb * Q8_BLOCK_BYTES;
        const __half d_h = *reinterpret_cast<const __half*>(blk + q8_0::kDOffset);
        const float  d_f = __half2float(d_h);
        const int8_t q   = static_cast<int8_t>(blk[q8_0::kQsOffset + t]);
        const float  xv  = __half2float(x[sb * Q8_BLOCK_ELEMS + t]);
        partial += d_f * static_cast<float>(q) * xv;
    }

    // Warp-shuffle reduction across the 32 threads (one warp).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, off);
    }
    if (t == 0) {
        float out = partial;
        if (bias) out += __half2float(bias[row]);
        y[row] = __float2half_rn(out);
    }
}

void validate_w_q8_0(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q8_0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q8_0");
    }
    if (W.cols % Q8_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 32");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

} // namespace

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    validate_w_q8_0(W_q8, "dequant_q8_0_to_fp16");
    const int rows = W_q8.rows;
    const int K    = W_q8.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const int blocks_per_row = K / Q8_BLOCK_ELEMS;
    if (rows == 0 || blocks_per_row == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    // CUDA caps gridDim.y at 65535; chunk rows to stay within the limit.
    constexpr int kMaxGridY = 65535;
    const uint8_t* W_p = static_cast<const uint8_t*>(W_q8.data);
    __half*        Y_p = static_cast<__half*>(W_fp16.data);
    const size_t row_bytes_w = static_cast<size_t>(blocks_per_row) * Q8_BLOCK_BYTES;
    const size_t row_elems_y = static_cast<size_t>(blocks_per_row) * Q8_BLOCK_ELEMS;
    for (int r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int r_chunk = (rows - r0) < kMaxGridY ? (rows - r0) : kMaxGridY;
        dim3 grid(blocks_per_row, r_chunk);
        dim3 block(Q8_BLOCK_ELEMS);
        dequant_q8_0_to_fp16_kernel<<<grid, block, 0, stream>>>(
            W_p + static_cast<size_t>(r0) * row_bytes_w,
            Y_p + static_cast<size_t>(r0) * row_elems_y,
            r_chunk, blocks_per_row);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y) {
    validate_w_q8_0(W_q8, "linear_forward_q8_0_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x must be FP16");
    }
    const int out = W_q8.rows;
    const int K   = W_q8.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    const int blocks_per_row = K / Q8_BLOCK_ELEMS;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = (bias && bias->size() > 0)
        ? static_cast<const __half*>(bias->data)
        : nullptr;

    linear_q8_0_fp16_gemv_kernel<<<out, Q8_BLOCK_ELEMS, 0, stream>>>(
        static_cast<const uint8_t*>(W_q8.data),
        static_cast<const __half*>(x.data),
        b_p,
        static_cast<__half*>(y.data),
        K, blocks_per_row);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q8_0(W_q8, "linear_forward_batched_q8_0_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q8.rows;
    if (W_q8.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q8_0_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q8_0_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q8_0_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    const int blocks_per_row = K / Q8_BLOCK_ELEMS;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = (bias && bias->size() > 0)
        ? static_cast<const __half*>(bias->data)
        : nullptr;

    if (q8_0_wmma_internal::launch_linear_q8_0_fp16_wmma(
            static_cast<const __half*>(X_BD.data),
            static_cast<const uint8_t*>(W_q8.data),
            b_p,
            static_cast<__half*>(Y_BD.data),
            B, out, K, stream)) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int b = 0; b < B; ++b) {
        const __half* x_p = static_cast<const __half*>(X_BD.data) + static_cast<size_t>(b) * K;
        __half*       y_p = static_cast<__half*>(Y_BD.data)       + static_cast<size_t>(b) * out;
        linear_q8_0_fp16_gemv_kernel<<<out, Q8_BLOCK_ELEMS, 0, stream>>>(
            static_cast<const uint8_t*>(W_q8.data),
            x_p, b_p, y_p,
            K, blocks_per_row);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
