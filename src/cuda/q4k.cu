// CUDA Q4_K (W4A16) dequant + GEMV. Q4_K block = 144 bytes / 256 elements:
//   fp16 d, fp16 dmin, uint8 scales[12] (eight 6-bit (sc,m) packed pairs),
//   uint8 qs[128] (256 nibbles). 8 sub-blocks of 32; element value
//   y = d * sc[is] * nibble - dmin * m[is]. The dequant kernel writes a
//   row-major FP16 weight; the GEMV kernel fuses dequant into a one-row dot
//   product with FP32 accumulation. Block size 256 (one thread per element of
//   a super-block); cross-warp reduction via shared memory.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

// Current stream helper (defined in runtime.cu).
void* cuda_current_stream();

namespace {

constexpr int Q4K_BLOCK_ELEMS = 256;
constexpr int Q4K_BLOCK_BYTES = 144;

// Recover the j-th (sc, m) pair (0 <= j < 8) from the 12-byte scales array.
// Mirrors llama.cpp's get_scale_min_k4 — re-expressed.
__device__ __forceinline__ void q4k_unpack_sc_m(int j, const uint8_t* s,
                                                uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = s[j]     & 0x3F;
        *m  = s[j + 4] & 0x3F;
    } else {
        *sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        *m  = (s[j + 4] >> 4)   | ((s[j - 0] >> 6) << 4);
    }
}

// One CTA per (out_row, super_block). 256 threads, one per element.
__global__ void dequant_q4k_to_fp16_kernel(const uint8_t* __restrict__ W,
                                           __half* __restrict__ Wfp16,
                                           int rows, int blocks_per_row) {
    const int row = blockIdx.y;
    const int sb  = blockIdx.x;
    const int t   = threadIdx.x;

    __shared__ uint8_t W_smem[Q4K_BLOCK_BYTES];
    __shared__ float   sc_f[8];
    __shared__ float   m_f [8];
    __shared__ float   d_f;
    __shared__ float   dmin_f;

    const uint8_t* blk = W + (static_cast<size_t>(row) * blocks_per_row + sb) * Q4K_BLOCK_BYTES;

    if (t < Q4K_BLOCK_BYTES) W_smem[t] = blk[t];
    __syncthreads();

    if (t == 0) {
        const __half d_h    = *reinterpret_cast<const __half*>(W_smem);
        const __half dmin_h = *reinterpret_cast<const __half*>(W_smem + 2);
        d_f    = __half2float(d_h);
        dmin_f = __half2float(dmin_h);
    }
    if (t < 8) {
        uint8_t sc, m;
        q4k_unpack_sc_m(t, W_smem + 4, &sc, &m);
        sc_f[t] = static_cast<float>(sc);
        m_f [t] = static_cast<float>(m);
    }
    __syncthreads();

    const int is   = t >> 5;            // sub-block index 0..7
    const int l    = t & 31;            // index within sub-block
    const int pair = is >> 1;           // pair of sub-blocks 0..3
    const uint8_t qb = W_smem[16 + pair * 32 + l];
    const int nib  = (is & 1) ? (qb >> 4) : (qb & 0x0F);

    const float y = d_f * sc_f[is] * static_cast<float>(nib)
                  - dmin_f * m_f[is];

    Wfp16[static_cast<size_t>(row) * (blocks_per_row * Q4K_BLOCK_ELEMS)
          + sb * Q4K_BLOCK_ELEMS + t] = __float2half_rn(y);
}

// In-block reduction across 256 threads (8 warps). Uses warp shuffles + a
// shared scratch buffer.
__device__ __forceinline__ float block_reduce_sum_256(float v, float* scratch) {
    // Warp reduce.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < 8) ? scratch[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, off);
        }
        if (lane == 0) scratch[0] = s;
    }
    __syncthreads();
    return scratch[0];
}

// One CTA per output row; loops over super-blocks along K. 256 threads.
__global__ void linear_q4k_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                            const __half*  __restrict__ x,
                                            const __half*  __restrict__ bias,
                                            __half*        __restrict__ y,
                                            int K, int blocks_per_row) {
    const int row = blockIdx.x;
    const int t   = threadIdx.x;

    __shared__ uint8_t W_smem[Q4K_BLOCK_BYTES];
    __shared__ __half  X_smem[Q4K_BLOCK_ELEMS];
    __shared__ float   sc_f[8];
    __shared__ float   m_f [8];
    __shared__ float   d_f;
    __shared__ float   dmin_f;
    __shared__ float   red_scratch[8];

    const int is   = t >> 5;
    const int l    = t & 31;
    const int pair = is >> 1;

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q4K_BLOCK_BYTES;

    for (int sb = 0; sb < blocks_per_row; ++sb) {
        const uint8_t* blk = row_base + sb * Q4K_BLOCK_BYTES;

        if (t < Q4K_BLOCK_BYTES) W_smem[t] = blk[t];
        X_smem[t] = x[sb * Q4K_BLOCK_ELEMS + t];
        __syncthreads();

        if (t == 0) {
            __half d_h, dmin_h;
            d_h    = *reinterpret_cast<const __half*>(W_smem);
            dmin_h = *reinterpret_cast<const __half*>(W_smem + 2);
            d_f    = __half2float(d_h);
            dmin_f = __half2float(dmin_h);
        }
        if (t < 8) {
            uint8_t sc, m;
            q4k_unpack_sc_m(t, W_smem + 4, &sc, &m);
            sc_f[t] = static_cast<float>(sc);
            m_f [t] = static_cast<float>(m);
        }
        __syncthreads();

        const uint8_t qb = W_smem[16 + pair * 32 + l];
        const int nib  = (is & 1) ? (qb >> 4) : (qb & 0x0F);
        const float w  = d_f * sc_f[is] * static_cast<float>(nib)
                       - dmin_f * m_f[is];
        partial += w * __half2float(X_smem[t]);
        __syncthreads();
    }

    const float sum = block_reduce_sum_256(partial, red_scratch);
    if (t == 0) {
        float out = sum;
        if (bias) out += __half2float(bias[row]);
        y[row] = __float2half_rn(out);
    }
}

void validate_w_q4k(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q4_K) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q4_K");
    }
    if (W.cols % Q4K_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 256");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

} // namespace

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16) {
    validate_w_q4k(W_q4k, "dequant_q4k_to_fp16");
    const int rows = W_q4k.rows;
    const int K    = W_q4k.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const int blocks_per_row = K / Q4K_BLOCK_ELEMS;
    if (rows == 0 || blocks_per_row == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    dim3 grid(blocks_per_row, rows);
    dim3 block(Q4K_BLOCK_ELEMS);
    dequant_q4k_to_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(W_q4k.data),
        static_cast<__half*>(W_fp16.data),
        rows, blocks_per_row);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    validate_w_q4k(W_q4k, "linear_forward_q4k_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: x must be FP16");
    }
    const int out = W_q4k.rows;
    const int K   = W_q4k.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q4k_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q4k_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    const int blocks_per_row = K / Q4K_BLOCK_ELEMS;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = (bias && bias->size() > 0)
        ? static_cast<const __half*>(bias->data)
        : nullptr;

    linear_q4k_fp16_gemv_kernel<<<out, Q4K_BLOCK_ELEMS, 0, stream>>>(
        static_cast<const uint8_t*>(W_q4k.data),
        static_cast<const __half*>(x.data),
        b_p,
        static_cast<__half*>(y.data),
        K, blocks_per_row);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q4k(W_q4k, "linear_forward_batched_q4k_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q4k.rows;
    if (W_q4k.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q4k_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q4k_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q4k_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    const int blocks_per_row = K / Q4K_BLOCK_ELEMS;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = (bias && bias->size() > 0)
        ? static_cast<const __half*>(bias->data)
        : nullptr;

    // Chunk 2: per-row GEMV loop. Chunk 3 will fuse into a real GEMM.
    for (int b = 0; b < B; ++b) {
        const __half* x_p = static_cast<const __half*>(X_BD.data) + static_cast<size_t>(b) * K;
        __half*       y_p = static_cast<__half*>(Y_BD.data)       + static_cast<size_t>(b) * out;
        linear_q4k_fp16_gemv_kernel<<<out, Q4K_BLOCK_ELEMS, 0, stream>>>(
            static_cast<const uint8_t*>(W_q4k.data),
            x_p, b_p, y_p,
            K, blocks_per_row);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
