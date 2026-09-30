// ─── Native AMD Wave32 GGUF dequant & linear forward (Q4_K, Q8_0, Q6_K) ──────
//
// AMD ROCm / HIP implementation optimized for RDNA 3.5 (gfx1151, Wave32).
//
// Supports:
// - Q4_K: 256-element blocks (144 bytes).
// - Q8_0: 32-element blocks (34 bytes).
// - Q6_K: 256-element blocks (210 bytes).
//
// Uses Wave32 shuffles (__shfl_down with width 32), 64-thread super-block groups
// (4 elements per thread), and 2-level reductions for high-throughput GEMV.

#include "detail/hip_check.h"
#include "detail/quant_internal.h"

#include <brotensor/tensor.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

void* hip_current_stream(int dev = -1);
static inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

namespace {

constexpr int kWaveSize = 32;

inline int get_cu_count() {
    static std::atomic<int> s_cus{0};
    int cus = s_cus.load(std::memory_order_relaxed);
    if (cus == 0) {
        int dev = 0;
        hipGetDevice(&dev);
        hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev);
        if (cus <= 0) cus = 16;
        s_cus.store(cus, std::memory_order_relaxed);
    }
    return cus;
}

// ════════════════════════════════════════════════════════════════════════════
//  Q4_K (W4A16)
// ════════════════════════════════════════════════════════════════════════════

constexpr int Q4K_BLOCK_ELEMS  = q4k::kBlockElems;
constexpr int Q4K_BLOCK_BYTES  = q4k::kBlockBytes;
constexpr int Q4K_GEMV_THREADS = Q4K_BLOCK_ELEMS / 4; // 64 threads

__global__ void dequant_q4k_to_fp16_kernel(const uint8_t* __restrict__ W,
                                           __half* __restrict__ Wfp16,
                                           int rows, int blocks_per_row) {
    const int row = blockIdx.y;
    const int sb  = blockIdx.x;
    const int t   = threadIdx.x;

    __shared__ uint8_t W_smem[Q4K_BLOCK_BYTES];
    __shared__ float   sc_f[8];
    __shared__ float   m_f[8];
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
        q4k::unpack_sc_m(t, W_smem + 4, sc, m);
        sc_f[t] = static_cast<float>(sc);
        m_f[t]  = static_cast<float>(m);
    }
    __syncthreads();

    const int is   = t >> 5;            // sub-block index 0..7
    const int l    = t & 31;            // index within sub-block
    const int pair = is >> 1;           // pair of sub-blocks 0..3
    const uint8_t qb = W_smem[16 + pair * 32 + l];
    const int nib  = (is & 1) ? (qb >> 4) : (qb & 0x0F);

    const float y = d_f * sc_f[is] * static_cast<float>(nib) - dmin_f * m_f[is];

    Wfp16[static_cast<size_t>(row) * (blocks_per_row * Q4K_BLOCK_ELEMS)
          + sb * Q4K_BLOCK_ELEMS + t] = __float2half_rn(y);
}

template <int SBG>
__global__ void linear_q4k_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                            const __half*  __restrict__ x,
                                            const __half*  __restrict__ bias,
                                            __half*        __restrict__ y,
                                            int K, int blocks_per_row, int out) {
    const int row = blockIdx.x;
    const int b   = blockIdx.y;
    const int grp = threadIdx.x >> 6;      // which super-block group (0..SBG-1)
    const int t   = threadIdx.x & 63;      // lane within the group

    const int is   = t >> 3;         // sub-block 0..7
    const int lg   = t & 7;          // element quad within the sub-block
    const int qoff = q4k::kQsOffset + (is >> 1) * 32 + lg * 4;
    const int xoff = is * 32 + lg * 4;
    const int hi   = is & 1;         // high nibble flag

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q4K_BLOCK_BYTES;
    const __half*  xb       = x + static_cast<size_t>(b) * K;

    for (int sb = grp; sb < blocks_per_row; sb += SBG) {
        const uint8_t* blk = row_base + sb * Q4K_BLOCK_BYTES;

        const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
        const float d_f    = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xFFFFu)));
        const float dmin_f = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
        uint32_t sc, m;
        q4k::unpack_sc_m(is, hdr, sc, m);
        const float wscale = d_f * static_cast<float>(sc);
        const float wmin   = dmin_f * static_cast<float>(m);

        const uint32_t q4 = *reinterpret_cast<const uint32_t*>(blk + qoff);
        const uint2 xpack = *reinterpret_cast<const uint2*>(xb + sb * Q4K_BLOCK_ELEMS + xoff);
        const __half2* xh = reinterpret_cast<const __half2*>(&xpack);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t qb  = (q4 >> (c * 8)) & 0xFFu;
            const uint32_t nib = hi ? (qb >> 4) : (qb & 0x0Fu);
            const float xv = (c & 1) ? __high2float(xh[c >> 1])
                                     : __low2float(xh[c >> 1]);
            partial += (wscale * static_cast<float>(nib) - wmin) * xv;
        }
    }

    // Wave32 reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down(partial, off, 32);
    }

    constexpr int NWARP = SBG * 2;
    __shared__ float warp_sum[NWARP];
    const int tid = threadIdx.x;
    if ((tid & 31) == 0) warp_sum[tid >> 5] = partial;
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARP; ++w) sum += warp_sum[w];
        if (bias) sum += __half2float(bias[row]);
        y[static_cast<size_t>(b) * out + row] = __float2half_rn(sum);
    }
}

inline int q4k_gemv_groups(int out, int blocks_per_row) {
    const int tw = 2 * get_cu_count() * 16;
    if (out >= tw) return 1;
    int g = 4;
    while (g > 1 && g > blocks_per_row) g >>= 1;
    return g;
}

inline void q4k_gemv_launch(const uint8_t* W, const __half* x,
                            const __half* bias, __half* y,
                            int out, int K, int blocks_per_row, int B,
                            hipStream_t stream) {
    dim3 grid(out, B);
    switch (q4k_gemv_groups(out, blocks_per_row)) {
        case 4:
            linear_q4k_fp16_gemv_kernel<4><<<grid, 4 * Q4K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
        case 2:
            linear_q4k_fp16_gemv_kernel<2><<<grid, 2 * Q4K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
        default:
            linear_q4k_fp16_gemv_kernel<1><<<grid, Q4K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
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

// ════════════════════════════════════════════════════════════════════════════
//  Q8_0 (W8A16)
// ════════════════════════════════════════════════════════════════════════════

constexpr int Q8_BLOCK_ELEMS = q8_0::kBlockElems;
constexpr int Q8_BLOCK_BYTES = q8_0::kBlockBytes;

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

template <int WARPS>
__global__ void linear_q8_0_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                             const __half*  __restrict__ x,
                                             const __half*  __restrict__ bias,
                                             __half*        __restrict__ y,
                                             int K, int blocks_per_row, int out) {
    constexpr int NGRP = WARPS * 4;

    const int row  = blockIdx.x;
    const int b    = blockIdx.y;
    const int grp  = threadIdx.x >> 3;
    const int lane = threadIdx.x & 7;

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q8_BLOCK_BYTES;
    const __half*  xb       = x + static_cast<size_t>(b) * K;

    for (int sb = grp; sb < blocks_per_row; sb += NGRP) {
        const uint8_t* blk = row_base + sb * Q8_BLOCK_BYTES;

        const float d_f = __half2float(*reinterpret_cast<const __half*>(blk + q8_0::kDOffset));
        const uint32_t q4 = load_u32_align2(blk + q8_0::kQsOffset + lane * 4);
        const uint2 xpack = *reinterpret_cast<const uint2*>(xb + sb * Q8_BLOCK_ELEMS + lane * 4);
        const __half2* xh = reinterpret_cast<const __half2*>(&xpack);

        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int qv = static_cast<int8_t>((q4 >> (c * 8)) & 0xFFu);
            const float xv = (c & 1) ? __high2float(xh[c >> 1])
                                     : __low2float(xh[c >> 1]);
            acc += static_cast<float>(qv) * xv;
        }
        partial += d_f * acc;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down(partial, off, 32);
    }

    if (WARPS == 1) {
        if (threadIdx.x == 0) {
            float sum = partial;
            if (bias) sum += __half2float(bias[row]);
            y[static_cast<size_t>(b) * out + row] = __float2half_rn(sum);
        }
        return;
    }

    __shared__ float warp_sum[WARPS];
    const int tid = threadIdx.x;
    if ((tid & 31) == 0) warp_sum[tid >> 5] = partial;
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += warp_sum[w];
        if (bias) sum += __half2float(bias[row]);
        y[static_cast<size_t>(b) * out + row] = __float2half_rn(sum);
    }
}

inline int q8_0_gemv_warps(int out, int blocks_per_row) {
    const int tw = 2 * get_cu_count() * 16;
    if (out >= tw) return 1;
    int w = 4;
    while (w > 1 && w * 4 > blocks_per_row) w >>= 1;
    return w;
}

inline void q8_0_gemv_launch(const uint8_t* W, const __half* x,
                             const __half* bias, __half* y,
                             int out, int K, int blocks_per_row, int B,
                             hipStream_t stream) {
    dim3 grid(out, B);
    switch (q8_0_gemv_warps(out, blocks_per_row)) {
        case 4:
            linear_q8_0_fp16_gemv_kernel<4><<<grid, 4 * 32, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
        case 2:
            linear_q8_0_fp16_gemv_kernel<2><<<grid, 2 * 32, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
        default:
            linear_q8_0_fp16_gemv_kernel<1><<<grid, 32, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
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

// ════════════════════════════════════════════════════════════════════════════
//  Q6_K (W6A16)
// ════════════════════════════════════════════════════════════════════════════

constexpr int Q6K_BLOCK_ELEMS  = q6k::kBlockElems;
constexpr int Q6K_BLOCK_BYTES  = q6k::kBlockBytes;
constexpr int Q6K_GEMV_THREADS = Q6K_BLOCK_ELEMS / 4;

__global__ void dequant_q6k_to_fp16_kernel(const uint8_t* __restrict__ W,
                                           __half* __restrict__ Wfp16,
                                           int rows, int blocks_per_row) {
    const int row    = blockIdx.y;
    const int sb_idx = blockIdx.x;
    const int t      = threadIdx.x;

    __shared__ uint8_t W_smem[Q6K_BLOCK_BYTES];
    __shared__ float   d_f;

    const uint8_t* blk = W + (static_cast<size_t>(row) * blocks_per_row + sb_idx) * Q6K_BLOCK_BYTES;

    if (t < Q6K_BLOCK_BYTES) W_smem[t] = blk[t];
    __syncthreads();

    if (t == 0) {
        const __half d_h = *reinterpret_cast<const __half*>(W_smem + q6k::kDOffset);
        d_f = __half2float(d_h);
    }
    __syncthreads();

    int sb, val6;
    q6k::decode_element(t, W_smem + q6k::kQlOffset, W_smem + q6k::kQhOffset, sb, val6);
    const int8_t scv = static_cast<int8_t>(W_smem[q6k::kScalesOffset + sb]);
    const float y = d_f * static_cast<float>(scv) * static_cast<float>(val6);

    Wfp16[static_cast<size_t>(row) * (blocks_per_row * Q6K_BLOCK_ELEMS)
          + sb_idx * Q6K_BLOCK_ELEMS + t] = __float2half_rn(y);
}

template <int SBG>
__global__ void linear_q6k_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                            const __half*  __restrict__ x,
                                            const __half*  __restrict__ bias,
                                            __half*        __restrict__ y,
                                            int K, int blocks_per_row, int out) {
    const int row = blockIdx.x;
    const int b   = blockIdx.y;
    const int grp = threadIdx.x >> 6;
    const int t   = threadIdx.x & 63;

    const q6k::QuadDesc qd = q6k::quad_desc(t * 4);
    const int xoff = t * 4;

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q6K_BLOCK_BYTES;
    const __half*  xb       = x + static_cast<size_t>(b) * K;

    for (int sb_idx = grp; sb_idx < blocks_per_row; sb_idx += SBG) {
        const uint8_t* blk = row_base + sb_idx * Q6K_BLOCK_BYTES;

        const uint32_t ql4 = load_u32_align2(blk + q6k::kQlOffset + qd.ql_off);
        const uint32_t qh4 = load_u32_align2(blk + q6k::kQhOffset + qd.qh_off);
        const float d_f = __half2float(*reinterpret_cast<const __half*>(blk + q6k::kDOffset));
        const int8_t scv = static_cast<int8_t>(*(blk + q6k::kScalesOffset + qd.sb));
        const float wscale = d_f * static_cast<float>(scv);

        const uint2 xpack = *reinterpret_cast<const uint2*>(xb + sb_idx * Q6K_BLOCK_ELEMS + xoff);
        const __half2* xh = reinterpret_cast<const __half2*>(&xpack);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int val6 = q6k::decode_packed(qd, ql4, qh4, c);
            const float xv = (c & 1) ? __high2float(xh[c >> 1])
                                     : __low2float(xh[c >> 1]);
            partial += wscale * static_cast<float>(val6) * xv;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down(partial, off, 32);
    }

    constexpr int NWARP = SBG * 2;
    __shared__ float warp_sum[NWARP];
    const int tid = threadIdx.x;
    if ((tid & 31) == 0) warp_sum[tid >> 5] = partial;
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARP; ++w) sum += warp_sum[w];
        if (bias) sum += __half2float(bias[row]);
        y[static_cast<size_t>(b) * out + row] = __float2half_rn(sum);
    }
}

inline int q6k_gemv_groups(int out, int blocks_per_row) {
    const int tw = 2 * get_cu_count() * 16;
    if (out >= tw) return 1;
    int g = 4;
    while (g > 1 && g > blocks_per_row) g >>= 1;
    return g;
}

inline void q6k_gemv_launch(const uint8_t* W, const __half* x,
                            const __half* bias, __half* y,
                            int out, int K, int blocks_per_row, int B,
                            hipStream_t stream) {
    dim3 grid(out, B);
    switch (q6k_gemv_groups(out, blocks_per_row)) {
        case 4:
            linear_q6k_fp16_gemv_kernel<4><<<grid, 4 * Q6K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
        case 2:
            linear_q6k_fp16_gemv_kernel<2><<<grid, 2 * Q6K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
            break;
        default:
            linear_q6k_fp16_gemv_kernel<1><<<grid, Q6K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row, out);
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

// ════════════════════════════════════════════════════════════════════════════
//  Q4_K Dispatch
// ════════════════════════════════════════════════════════════════════════════

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16) {
    validate_w_q4k(W_q4k, "dequant_q4k_to_fp16");
    const int rows = W_q4k.rows;
    const int K    = W_q4k.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const int blocks_per_row = K / Q4K_BLOCK_ELEMS;
    if (rows == 0 || blocks_per_row == 0) return;

    hipStream_t stream = cur_stream();
    constexpr int kMaxGridY = 65535;
    const uint8_t* W_p = static_cast<const uint8_t*>(W_q4k.data);
    __half*        Y_p = static_cast<__half*>(W_fp16.data);
    const size_t row_bytes_w = static_cast<size_t>(blocks_per_row) * Q4K_BLOCK_BYTES;
    const size_t row_elems_y = static_cast<size_t>(blocks_per_row) * Q4K_BLOCK_ELEMS;
    for (int r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int r_chunk = (rows - r0) < kMaxGridY ? (rows - r0) : kMaxGridY;
        dim3 grid(blocks_per_row, r_chunk);
        dim3 block(Q4K_BLOCK_ELEMS);
        dequant_q4k_to_fp16_kernel<<<grid, block, 0, stream>>>(
            W_p + static_cast<size_t>(r0) * row_bytes_w,
            Y_p + static_cast<size_t>(r0) * row_elems_y,
            r_chunk, blocks_per_row);
    }
    BROTENSOR_HIP_CHECK(hipGetLastError());
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

    hipStream_t stream = cur_stream();
    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;

    q4k_gemv_launch(static_cast<const uint8_t*>(W_q4k.data),
                    static_cast<const __half*>(x.data),
                    b_p,
                    static_cast<__half*>(y.data),
                    out, K, blocks_per_row, /*B=*/1, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
        throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    const int blocks_per_row = K / Q4K_BLOCK_ELEMS;

    hipStream_t stream = cur_stream();
    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;

    q4k_gemv_launch(static_cast<const uint8_t*>(W_q4k.data),
                    static_cast<const __half*>(X_BD.data),
                    b_p,
                    static_cast<__half*>(Y_BD.data),
                    out, K, blocks_per_row, B, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  Q8_0 Dispatch
// ════════════════════════════════════════════════════════════════════════════

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    validate_w_q8_0(W_q8, "dequant_q8_0_to_fp16");
    const int rows = W_q8.rows;
    const int K    = W_q8.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const int blocks_per_row = K / Q8_BLOCK_ELEMS;
    if (rows == 0 || blocks_per_row == 0) return;

    hipStream_t stream = cur_stream();
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
    BROTENSOR_HIP_CHECK(hipGetLastError());
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

    hipStream_t stream = cur_stream();
    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;

    q8_0_gemv_launch(static_cast<const uint8_t*>(W_q8.data),
                     static_cast<const __half*>(x.data),
                     b_p,
                     static_cast<__half*>(y.data),
                     out, K, blocks_per_row, /*B=*/1, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
        throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    const int blocks_per_row = K / Q8_BLOCK_ELEMS;

    hipStream_t stream = cur_stream();
    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;

    q8_0_gemv_launch(static_cast<const uint8_t*>(W_q8.data),
                     static_cast<const __half*>(X_BD.data),
                     b_p,
                     static_cast<__half*>(Y_BD.data),
                     out, K, blocks_per_row, B, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  Q6_K Dispatch
// ════════════════════════════════════════════════════════════════════════════

void dequant_q6k_to_fp16(const Tensor& W_q6k, Tensor& W_fp16) {
    validate_w_q6k(W_q6k, "dequant_q6k_to_fp16");
    const int rows = W_q6k.rows;
    const int K    = W_q6k.cols;
    if (W_fp16.rows != rows || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(rows, K, Dtype::FP16);
    }
    const int blocks_per_row = K / Q6K_BLOCK_ELEMS;
    if (rows == 0 || blocks_per_row == 0) return;

    hipStream_t stream = cur_stream();
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
    BROTENSOR_HIP_CHECK(hipGetLastError());
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

    hipStream_t stream = cur_stream();
    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;

    q6k_gemv_launch(static_cast<const uint8_t*>(W_q6k.data),
                    static_cast<const __half*>(x.data),
                    b_p,
                    static_cast<__half*>(y.data),
                    out, K, blocks_per_row, /*B=*/1, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
        throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: shape mismatch (W.cols != X.cols)");
    }
    if (bias) {
        if (bias->dtype != Dtype::FP16) {
            throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: bias must be FP16");
        }
        const bool ok = (bias->rows == out && bias->cols == 1) ||
                        (bias->rows == 1 && bias->cols == out);
        if (!ok) {
            throw std::runtime_error("brotensor: linear_forward_batched_q6k_fp16: bias shape must be (out,1) or (1,out)");
        }
    }
    if (Y_BD.rows != B || Y_BD.cols != out || Y_BD.dtype != Dtype::FP16) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;
    const int blocks_per_row = K / Q6K_BLOCK_ELEMS;

    hipStream_t stream = cur_stream();
    const __half* b_p = (bias && bias->size() > 0) ? static_cast<const __half*>(bias->data) : nullptr;

    q6k_gemv_launch(static_cast<const uint8_t*>(W_q6k.data),
                    static_cast<const __half*>(X_BD.data),
                    b_p,
                    static_cast<__half*>(Y_BD.data),
                    out, K, blocks_per_row, B, stream);
    BROTENSOR_HIP_CHECK(hipGetLastError());
}

} // namespace brotensor::detail::hip
