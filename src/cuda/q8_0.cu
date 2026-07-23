// CUDA Q8_0 (W8A16-style) dequant + GEMV. Q8_0 block = 34 bytes / 32
// elements: fp16 d + int8 qs[32]. y = d * qs[i]. Dequant kernel uses one
// CTA per (row, block) with 32 threads (one per element). GEMV kernel uses
// one CTA per output row, four elements per thread, walking blocks along K
// with a shuffle reduction at the end; see the kernel for the load shape and
// for how the row widens when the grid is too small to fill the GPU.

#include "detail/cuda_check.h"
#include "detail/load_align.cuh"
#include "q8_0_internal.cuh"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <atomic>
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

// One CTA per output row. Each thread takes four consecutive elements of a
// 32-element block, so eight threads cover a block and a WARPS-warp CTA keeps
// 4*WARPS blocks in flight.
//
// Granularity is the point. The shape this replaces handed every thread a
// single int8 and a single half per block - one- and two-byte requests, the
// worst granularity on offer, and the same defect that capped the Q4_K and
// Q6_K GEMVs before they were rebuilt. Four elements per thread turns that
// into one 8-byte x load and one 4-byte weight read per block.
//
// The weight read cannot be a plain uint32 load: a block is 34 bytes, so qs
// starts on an arbitrary even boundary and half the blocks are not 4-byte
// aligned. It goes out as the usual 16-bit pair (detail/load_align.cuh). x
// needs no such care - element sb*32 + lane*4 sits at byte 64*sb + 8*lane,
// always 8-byte aligned.
//
// WARPS widens the row for the same reason as q4k/q6k: a single-warp row has
// almost no memory-level parallelism, so at low output widths the grid cannot
// keep enough requests outstanding and the kernel runs latency-bound.
template <int WARPS>
__global__ void linear_q8_0_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                             const __half*  __restrict__ x,
                                             const __half*  __restrict__ bias,
                                             __half*        __restrict__ y,
                                             int K, int blocks_per_row) {
    constexpr int NGRP = WARPS * 4;        // blocks walked concurrently

    const int row  = blockIdx.x;
    const int grp  = threadIdx.x >> 3;     // which of the NGRP blocks
    const int lane = threadIdx.x & 7;      // element quad within that block

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q8_BLOCK_BYTES;

    for (int sb = grp; sb < blocks_per_row; sb += NGRP) {
        const uint8_t* blk = row_base + sb * Q8_BLOCK_BYTES;

        const float d_f = __half2float(
            __ldg(reinterpret_cast<const __half*>(blk + q8_0::kDOffset)));
        const uint32_t q4 = load_u32_align2(blk + q8_0::kQsOffset + lane * 4);
        const uint2 xpack = __ldg(reinterpret_cast<const uint2*>(
            x + sb * Q8_BLOCK_ELEMS + lane * 4));
        const __half2* xh = reinterpret_cast<const __half2*>(&xpack);

        // Scale once per block rather than once per element: d is shared by
        // all 32, and the products are exact int8*fp16 either way.
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int qv = static_cast<int8_t>((q4 >> (c * 8)) & 0xFFu);
            const float xv = (c & 1) ? __high2float(xh[c >> 1])
                                     : __low2float (xh[c >> 1]);
            acc += static_cast<float>(qv) * xv;
        }
        partial += d_f * acc;
    }

    // Reduce within each warp, then across the CTA's warps.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    }
    if (WARPS == 1) {
        if (threadIdx.x == 0) {
            float out = partial;
            if (bias) out += __half2float(bias[row]);
            y[row] = __float2half_rn(out);
        }
        return;
    }
    __shared__ float warp_sum[WARPS];
    const int tid = threadIdx.x;
    if ((tid & 31) == 0) warp_sum[tid >> 5] = partial;
    __syncthreads();
    if (tid == 0) {
        float out = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) out += warp_sum[w];
        if (bias) out += __half2float(bias[row]);
        y[row] = __float2half_rn(out);
    }
}

// Group count and dispatch, same rule as q4k_gemv_groups / q6k_gemv_groups:
// widen the row only when the plain one-warp grid does not already span two
// block-waves, with the threshold read from the device so it tracks SM count.
inline int q8_0_gemv_warps(int out, int blocks_per_row) {
    static std::atomic<int> two_waves{0};   // benign double-compute, no lock
    int tw = two_waves.load(std::memory_order_relaxed);
    if (tw == 0) {
        int dev = 0, sms = 1, blocks = 1;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&blocks, cudaDevAttrMaxBlocksPerMultiprocessor, dev);
        tw = 2 * sms * blocks;
        two_waves.store(tw, std::memory_order_relaxed);
    }
    if (out >= tw) return 1;
    // Warps whose four blocks fall past the row's block count would sit idle.
    int w = 4;
    while (w > 1 && w * 4 > blocks_per_row) w >>= 1;
    return w;
}

inline void q8_0_gemv_launch(const uint8_t* W, const __half* x,
                             const __half* bias, __half* y,
                             int out, int K, int blocks_per_row,
                             cudaStream_t stream) {
    switch (q8_0_gemv_warps(out, blocks_per_row)) {
        case 4:
            linear_q8_0_fp16_gemv_kernel<4><<<out, 4 * 32, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
        case 2:
            linear_q8_0_fp16_gemv_kernel<2><<<out, 2 * 32, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
        default:
            linear_q8_0_fp16_gemv_kernel<1><<<out, 32, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
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

    q8_0_gemv_launch(static_cast<const uint8_t*>(W_q8.data),
                     static_cast<const __half*>(x.data),
                     b_p, static_cast<__half*>(y.data),
                     out, K, blocks_per_row, stream);
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
        q8_0_gemv_launch(static_cast<const uint8_t*>(W_q8.data),
                         x_p, b_p, y_p, out, K, blocks_per_row, stream);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
