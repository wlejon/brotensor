// CUDA Q4_K (W4A16) dequant + GEMV. Q4_K block = 144 bytes / 256 elements:
//   fp16 d, fp16 dmin, uint8 scales[12] (eight 6-bit (sc,m) packed pairs),
//   uint8 qs[128] (256 nibbles). 8 sub-blocks of 32; element value
//   y = d * sc[is] * nibble - dmin * m[is]. The dequant kernel writes a
//   row-major FP16 weight (256 threads, one per element of a super-block);
//   the GEMV kernel fuses dequant into a one-row dot product with FP32
//   accumulation (64 threads, four elements each — see the comment on it).

#include "detail/cuda_check.h"
#include "q4k_internal.cuh"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;
using ::brotensor::Dtype;

// Current stream helper (defined in runtime.cu).
void* cuda_current_stream();

// Forward decl of fused WMMA launcher implemented in q4k_wmma.cu.
namespace q4k_wmma_internal {
bool launch_linear_q4k_fp16_wmma(const __half* X, const uint8_t* W_q4k,
                                 const __half* bias, __half* Y,
                                 int B, int M, int K, cudaStream_t stream);
}  // namespace q4k_wmma_internal

namespace {

constexpr int Q4K_BLOCK_ELEMS = q4k::kBlockElems;
constexpr int Q4K_BLOCK_BYTES = q4k::kBlockBytes;

// GEMV threads per CTA: four elements each covers one 256-element super-block.
constexpr int Q4K_GEMV_THREADS = Q4K_BLOCK_ELEMS / 4;

__device__ __forceinline__ void q4k_unpack_sc_m(int j, const uint8_t* s,
                                                uint8_t* sc, uint8_t* m) {
    uint8_t sc_v, m_v;
    q4k::unpack_sc_m(j, s, sc_v, m_v);
    *sc = sc_v;
    *m  = m_v;
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

// One CTA per output row; loops over super-blocks along K. 64 threads, four
// elements each — one CTA still covers a whole 256-element super-block per
// iteration, but in a quarter of the threads with four times the load width.
//
// Thread t owns sub-block is = t>>3 and the four consecutive elements
// l = (t&7)*4 .. +3 within it, which is exactly:
//
//   one uint32 of qs   at kQsOffset + (is>>1)*32 + (t&7)*4   (4-byte aligned)
//   one uint2  of x    at is*32 + (t&7)*4                    (4 halves, 8B)
//   one uint4  header  at the block start                    (16B aligned:
//                                                             144 = 9 * 16)
//
// Three load instructions per thread per super-block, all naturally aligned,
// against the eleven byte- and half-wide ones the one-element-per-thread map
// needed. The sector count is unchanged — it is fixed by the data — but the
// instruction count and the per-instruction coalescing are not, and this
// kernel is request-throughput bound rather than DRAM bound: Q8_0's GEMV
// issues *more* total requests than Q4_K's did and still finished sooner.
//
// Nothing is staged in shared and there is no barrier in the loop. (An
// earlier version copied the 144-byte block and all 256 x-halves into shared
// behind three __syncthreads() per super-block, with thread 0 decoding the
// header and threads 0..7 the scales serially while the other 248 waited.)
//
// 64-thread CTAs still reach full occupancy on sm_89: the 24-block-per-SM cap
// times 64 threads is 1536, which is the SM's thread limit.
// One CTA per output row. The row's super-blocks are split across SBG groups
// of 64 threads — a group covers one super-block's 256 elements at 4 per
// thread — so SBG scales how many of a row's loads are in flight at once
// without changing the CTA count.
//
// That is the knob that matters here, because this kernel is latency-bound
// rather than bandwidth-bound once the output is narrow. At one 64-thread
// group per row, a 4096-row projection reached 442 GB/s where a 12288-row one
// reached 601 on identical per-row work: the only difference was how many CTAs
// were resident to keep requests outstanding. Widening the row itself supplies
// that memory-level parallelism regardless of how many rows there are.
template <int SBG>
__global__ void linear_q4k_fp16_gemv_kernel(const uint8_t* __restrict__ W,
                                            const __half*  __restrict__ x,
                                            const __half*  __restrict__ bias,
                                            __half*        __restrict__ y,
                                            int K, int blocks_per_row) {
    const int row = blockIdx.x;
    const int grp = threadIdx.x >> 6;      // which super-block group
    const int t   = threadIdx.x & 63;      // lane within the group

    const int is   = t >> 3;         // sub-block 0..7
    const int lg   = t & 7;          // element quad within the sub-block
    const int qoff = q4k::kQsOffset + (is >> 1) * 32 + lg * 4;
    const int xoff = is * 32 + lg * 4;
    const int hi   = is & 1;         // this sub-block takes the high nibbles

    float partial = 0.0f;

    const uint8_t* row_base = W + static_cast<size_t>(row) * blocks_per_row * Q4K_BLOCK_BYTES;

    for (int sb = grp; sb < blocks_per_row; sb += SBG) {
        const uint8_t* blk = row_base + sb * Q4K_BLOCK_BYTES;

        const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
        const float d_f    = __half2float(__ushort_as_half(
            static_cast<unsigned short>(hdr.x & 0xFFFFu)));
        const float dmin_f = __half2float(__ushort_as_half(
            static_cast<unsigned short>(hdr.x >> 16)));
        uint32_t sc, m;
        q4k::unpack_sc_m(is, hdr, sc, m);
        const float wscale = d_f * static_cast<float>(sc);
        const float wmin   = dmin_f * static_cast<float>(m);

        const uint32_t q4 = __ldg(reinterpret_cast<const uint32_t*>(blk + qoff));
        const uint2 xpack = __ldg(reinterpret_cast<const uint2*>(
            x + sb * Q4K_BLOCK_ELEMS + xoff));
        const __half2* xh = reinterpret_cast<const __half2*>(&xpack);

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t qb  = (q4 >> (c * 8)) & 0xFFu;
            const uint32_t nib = hi ? (qb >> 4) : (qb & 0x0Fu);
            const float xv = (c & 1) ? __high2float(xh[c >> 1])
                                     : __low2float (xh[c >> 1]);
            partial += (wscale * static_cast<float>(nib) - wmin) * xv;
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

// Pick the group count for a row of `blocks_per_row` super-blocks.
//
// Widening a row only pays when the plain one-group grid cannot already fill
// the machine. Measured on a 4090 (128 SMs, 24 resident blocks/SM, so 3072
// CTAs to a wave), three runs each:
//
//   shape                     out     1 group     4 groups
//   qkv_proj   6144 x  4096   6144    576 GB/s    512-532
//   gate/up   12288 x  4096  12288    628-643     564
//   o_proj     4096 x  4096   4096    446-461     463-485
//   down_proj  4096 x 12288   4096    453-469     553
//
// The crossover lands exactly at two block-waves. At or above it the grid
// already has the CTAs to keep requests outstanding, and the coarser 8-warp
// CTAs only cost scheduling freedom; below it the CTAs are not there, and the
// parallelism has to come from inside the row instead. So the threshold is
// read from the device rather than hard-coded — it moves with SM count.
inline int q4k_gemv_groups(int out, int blocks_per_row) {
    // Relaxed atomic, not a guarded static: every caller computes the same
    // value, so a racing double-compute is harmless and no lock is needed.
    static std::atomic<int> two_waves{0};
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
    // Groups past the row's super-block count would sit idle.
    int g = 4;
    while (g > 1 && g > blocks_per_row) g >>= 1;
    return g;
}

inline void q4k_gemv_launch(const uint8_t* W, const __half* x,
                            const __half* bias, __half* y,
                            int out, int K, int blocks_per_row,
                            cudaStream_t stream) {
    switch (q4k_gemv_groups(out, blocks_per_row)) {
        case 4:
            linear_q4k_fp16_gemv_kernel<4><<<out, 4 * Q4K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
        case 2:
            linear_q4k_fp16_gemv_kernel<2><<<out, 2 * Q4K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
            break;
        default:
            linear_q4k_fp16_gemv_kernel<1><<<out, Q4K_GEMV_THREADS, 0, stream>>>(
                W, x, bias, y, K, blocks_per_row);
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
    // CUDA caps gridDim.y at 65535; chunk rows to stay within the limit.
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

    q4k_gemv_launch(static_cast<const uint8_t*>(W_q4k.data),
                    static_cast<const __half*>(x.data),
                    b_p,
                    static_cast<__half*>(y.data),
                    out, K, blocks_per_row, stream);
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

    // Chunk 3: try the fused WMMA GEMM first. It returns false for shapes
    // outside its supported envelope (small B, K%256 != 0, etc), in which
    // case we fall back to the chunk-2 per-row GEMV loop.
    if (q4k_wmma_internal::launch_linear_q4k_fp16_wmma(
            static_cast<const __half*>(X_BD.data),
            static_cast<const uint8_t*>(W_q4k.data),
            b_p,
            static_cast<__half*>(Y_BD.data),
            B, out, K, stream)) {
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int b = 0; b < B; ++b) {
        const __half* x_p = static_cast<const __half*>(X_BD.data) + static_cast<size_t>(b) * K;
        __half*       y_p = static_cast<__half*>(Y_BD.data)       + static_cast<size_t>(b) * out;
        q4k_gemv_launch(static_cast<const uint8_t*>(W_q4k.data),
                        x_p, b_p, y_p, out, K, blocks_per_row, stream);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
