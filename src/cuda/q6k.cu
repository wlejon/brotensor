// CUDA Q6_K (W6A16) dequant + GEMV. Q6_K block = 210 bytes / 256 elements:
//   uint8 ql[128]  -- low 4 bits, 256 nibbles
//   uint8 qh[64]   -- high 2 bits, packed 4-per-byte
//   int8  sc[16]   -- 16 sub-block signed scales
//   fp16  d        -- super-block scale
// Decoded value: y = d * sc[sb] * (val6 - 32), see q6k_internal.cuh
// for the index mapping. The dequant kernel writes a row-major FP16 weight;
// the GEMV kernel fuses dequant into a one-row dot product with FP32
// accumulation. The dequant kernel runs 256 threads per CTA (one per
// element); the GEMV kernel runs 64, four elements each.

#include "detail/cuda_check.h"
#include "q6k_internal.cuh"

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

namespace q6k_wmma_internal {
bool launch_linear_q6k_fp16_wmma(const __half* X, const uint8_t* W_q6k,
                                 const __half* bias, __half* Y,
                                 int B, int M, int K, cudaStream_t stream);
}

namespace {

constexpr int Q6K_BLOCK_ELEMS = q6k::kBlockElems;
constexpr int Q6K_BLOCK_BYTES = q6k::kBlockBytes;

// GEMV threads per CTA: four elements each covers one 256-element super-block.
constexpr int Q6K_GEMV_THREADS = Q6K_BLOCK_ELEMS / 4;

// One CTA per (out_row, super_block). 256 threads, one per element.
__global__ void dequant_q6k_to_fp16_kernel(const uint8_t* __restrict__ W,
                                           __half* __restrict__ Wfp16,
                                           int rows, int blocks_per_row) {
    const int row = blockIdx.y;
    const int sb_idx = blockIdx.x;
    const int t   = threadIdx.x;

    __shared__ uint8_t W_smem[Q6K_BLOCK_BYTES];
    __shared__ float   d_f;

    const uint8_t* blk = W + (static_cast<size_t>(row) * blocks_per_row + sb_idx) * Q6K_BLOCK_BYTES;

    // Cooperative load of 210 bytes by 256 threads (~1 byte each).
    if (t < Q6K_BLOCK_BYTES) W_smem[t] = blk[t];
    __syncthreads();

    if (t == 0) {
        const __half d_h = *reinterpret_cast<const __half*>(W_smem + q6k::kDOffset);
        d_f = __half2float(d_h);
    }
    __syncthreads();

    int sb, val6;
    q6k::decode_element(t, W_smem + q6k::kQlOffset, W_smem + q6k::kQhOffset,
                        sb, val6);
    const int8_t scv = static_cast<int8_t>(W_smem[q6k::kScalesOffset + sb]);
    const float y = d_f * static_cast<float>(scv) * static_cast<float>(val6);

    Wfp16[static_cast<size_t>(row) * (blocks_per_row * Q6K_BLOCK_ELEMS)
          + sb_idx * Q6K_BLOCK_ELEMS + t] = __float2half_rn(y);
}

__device__ __forceinline__ float block_reduce_sum_256(float v, float* scratch) {
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

// One CTA per output row; loops over super-blocks along K. 64 threads, four
// consecutive elements each.
//
// Nothing is staged in shared and there is no barrier in the loop. The
// previous version copied the whole 210-byte block and all 256 x-halves into
// shared behind three __syncthreads() per super-block, with thread 0 decoding
// `d` serially while the other 255 waited at the barrier, then had every
// thread re-derive the ql/qh layout for its single element.
//
// A run of four consecutive elements shares its group, quad and sub-block, so
// it needs four consecutive ql bytes, four consecutive qh bytes, one scale
// byte and four x-halves. The ql/qh reads go through load_u32_align2: a block
// is 210 bytes and 210 % 4 == 2, so every other block start is only 2-byte
// aligned and a single uint32 load would be misaligned. See q6k_internal.cuh.
// One CTA per output row, with the row's super-blocks split across SBG groups
// of 64 threads. See q4k.cu for why: a one-group row has only two warps of
// memory-level parallelism, so at low output widths the grid cannot keep enough
// requests outstanding and the kernel runs latency-bound.
template <int SBG>
__global__ void linear_q6k_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                            const __half*  __restrict__ x,
                                            const __half*  __restrict__ bias,
                                            __half*        __restrict__ y,
                                            int K, int blocks_per_row) {
    const int row = blockIdx.x;
    const int grp = threadIdx.x >> 6;      // which super-block group
    const int t   = threadIdx.x & 63;      // lane within the group

    // Four consecutive elements per thread; 64 threads cover the super-block.
    const q6k::QuadDesc qd = q6k::quad_desc(t * 4);
    const int xoff = t * 4;

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q6K_BLOCK_BYTES;

    for (int sb_idx = grp; sb_idx < blocks_per_row; sb_idx += SBG) {
        const uint8_t* blk = row_base + sb_idx * Q6K_BLOCK_BYTES;

        const uint32_t ql4 = load_u32_align2(blk + q6k::kQlOffset + qd.ql_off);
        const uint32_t qh4 = load_u32_align2(blk + q6k::kQhOffset + qd.qh_off);
        const float d_f = __half2float(
            __ldg(reinterpret_cast<const __half*>(blk + q6k::kDOffset)));
        const int8_t scv = static_cast<int8_t>(
            __ldg(blk + q6k::kScalesOffset + qd.sb));
        const float wscale = d_f * static_cast<float>(scv);

        const uint2 xpack = __ldg(reinterpret_cast<const uint2*>(
            x + sb_idx * Q6K_BLOCK_ELEMS + xoff));
        const __half2* xh = reinterpret_cast<const __half2*>(&xpack);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int val6 = q6k::decode_packed(qd, ql4, qh4, c);
            const float xv = (c & 1) ? __high2float(xh[c >> 1])
                                     : __low2float (xh[c >> 1]);
            partial += wscale * static_cast<float>(val6) * xv;
        }
    }

    // Reduce within each warp, then across the CTA's 2*SBG warps.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    }
    constexpr int NWARP = SBG * 2;
    __shared__ float warp_sum[NWARP];
    const int tid = threadIdx.x;
    if ((tid & 31) == 0) warp_sum[tid >> 5] = partial;
    __syncthreads();
    if (tid == 0) {
        float out = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARP; ++w) out += warp_sum[w];
        if (bias) out += __half2float(bias[row]);
        y[row] = __float2half_rn(out);
    }
}

// Group count and dispatch, same rule as q4k_gemv_groups: widen the row only
// when the plain one-group grid does not already span two block-waves, with the
// threshold read from the device so it tracks SM count.
inline int q6k_gemv_groups(int out, int blocks_per_row) {
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
    int g = 4;
    while (g > 1 && g > blocks_per_row) g >>= 1;
    return g;
}

inline void q6k_gemv_launch(const uint8_t* W, const __half* x,
                            const __half* bias, __half* y,
                            int out, int K, int blocks_per_row,
                            cudaStream_t stream) {
    switch (q6k_gemv_groups(out, blocks_per_row)) {
        case 4:
            linear_q6k_fp16_gemv_kernel<4><<<out, 4 * Q6K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
        case 2:
            linear_q6k_fp16_gemv_kernel<2><<<out, 2 * Q6K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
        default:
            linear_q6k_fp16_gemv_kernel<1><<<out, Q6K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
    }
}

void validate_w_q6k(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q6_K) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q6_K");
    }
    if (W.cols % Q6K_BLOCK_ELEMS != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W.cols must be a multiple of 256");
    }
    if (W.rows <= 0 || W.cols <= 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W has non-positive shape");
    }
}

} // namespace

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16) {
    validate_w_q6k(W_q6k, "dequant_q6k_to_fp16");
    const int rows = W_q6k.rows;
    const int K    = W_q6k.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const int blocks_per_row = K / Q6K_BLOCK_ELEMS;
    if (rows == 0 || blocks_per_row == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    // CUDA caps gridDim.y at 65535; chunk rows to stay within the limit.
    constexpr int kMaxGridY = 65535;
    const uint8_t* W_p = static_cast<const uint8_t*>(W_q6k.data);
    __half*        Y_p = static_cast<__half*>(W_fp16.data);
    const size_t row_bytes_w = static_cast<size_t>(blocks_per_row) * Q6K_BLOCK_BYTES;
    const size_t row_elems_y = static_cast<size_t>(blocks_per_row) * Q6K_BLOCK_ELEMS;
    for (int r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int r_chunk = (rows - r0) < kMaxGridY ? (rows - r0) : kMaxGridY;
        dim3 grid(blocks_per_row, r_chunk);
        dim3 block(Q6K_BLOCK_ELEMS);
        dequant_q6k_to_fp16_kernel<<<grid, block, 0, stream>>>(
            W_p + static_cast<size_t>(r0) * row_bytes_w,
            Y_p + static_cast<size_t>(r0) * row_elems_y,
            r_chunk, blocks_per_row);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_forward_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    validate_w_q6k(W_q6k, "linear_forward_q6k_fp16");
    if (x.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: x must be FP16");
    }
    const int out = W_q6k.rows;
    const int K   = W_q6k.cols;
    if (x.rows != K || x.cols != 1) {
        throw std::runtime_error("brotensor: linear_forward_q6k_fp16: x shape must be (in, 1)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_q6k_fp16: bias must be FP16");
        }
        if (bias->rows != out || bias->cols != 1) {
            throw std::runtime_error("brotensor: linear_forward_q6k_fp16: bias shape must be (out, 1)");
        }
    }
    if (y.rows != out || y.cols != 1 || y.dtype != Dtype::FP16) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;
    const int blocks_per_row = K / Q6K_BLOCK_ELEMS;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = (bias && bias->size() > 0)
        ? static_cast<const __half*>(bias->data)
        : nullptr;

    q6k_gemv_launch(static_cast<const uint8_t*>(W_q6k.data),
                    static_cast<const __half*>(x.data),
                    b_p,
                    static_cast<__half*>(y.data),
                    out, K, blocks_per_row, stream);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_forward_batched_q6k_fp16(const Tensor& W_q6k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q6k(W_q6k, "linear_forward_batched_q6k_fp16");
    if (X_BD.dtype != Dtype::FP16) {
        throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: X must be FP16");
    }
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q6k.rows;
    if (W_q6k.cols != K) {
        throw std::runtime_error(
            "brotensor: linear_forward_batched_q6k_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q6k_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error(
                "brotensor: linear_forward_batched_q6k_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    const int blocks_per_row = K / Q6K_BLOCK_ELEMS;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    const __half* b_p = (bias && bias->size() > 0)
        ? static_cast<const __half*>(bias->data)
        : nullptr;

    if (q6k_wmma_internal::launch_linear_q6k_fp16_wmma(
            static_cast<const __half*>(X_BD.data),
            static_cast<const uint8_t*>(W_q6k.data),
            b_p,
            static_cast<__half*>(Y_BD.data),
            B, out, K, stream)) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int b = 0; b < B; ++b) {
        const __half* x_p = static_cast<const __half*>(X_BD.data) + static_cast<size_t>(b) * K;
        __half*       y_p = static_cast<__half*>(Y_BD.data)       + static_cast<size_t>(b) * out;
        q6k_gemv_launch(static_cast<const uint8_t*>(W_q6k.data),
                        x_p, b_p, y_p, out, K, blocks_per_row, stream);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
