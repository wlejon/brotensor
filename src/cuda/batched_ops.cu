// Batched (inference-only) CUDA ops.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Defined in tensor.cu (same namespace). The pooled allocator draws from a
// stream-ordered memory pool (cudaMallocAsync/cudaFreeAsync) instead of the
// synchronizing cudaMalloc/cudaFree pair.
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

using ::brotensor::Tensor;
using ::brotensor::Dtype;

namespace {

constexpr int BL_ROWS_PER_BLOCK = 64;
constexpr int BL_TILE           = 64;

// Current CUDA stream for hot-op launches — so kernels join a non-default
// capture/replay stream instead of silently landing on the default stream.
inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

__global__ void linear_forward_batched_kernel(const float* __restrict__ W,
                                              const float* __restrict__ bias,
                                              const float* __restrict__ X,
                                              float* __restrict__ Y,
                                              int B, int out_dim, int in_dim) {
    __shared__ float xtile[BL_TILE];

    const int b   = blockIdx.y;
    const int row = blockIdx.x * BL_ROWS_PER_BLOCK + threadIdx.x;
    if (b >= B) return;

    const float* x_row = X + static_cast<size_t>(b) * in_dim;
    float acc = 0.0f;

    for (int t0 = 0; t0 < in_dim; t0 += BL_TILE) {
        const int t_len = (in_dim - t0) < BL_TILE ? (in_dim - t0) : BL_TILE;

        for (int k = threadIdx.x; k < t_len; k += blockDim.x) {
            xtile[k] = x_row[t0 + k];
        }
        __syncthreads();

        if (row < out_dim) {
            const float* wrow = W + static_cast<size_t>(row) * in_dim + t0;
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += wrow[k] * xtile[k];
            }
        }
        __syncthreads();
    }

    if (row < out_dim) {
        Y[static_cast<size_t>(b) * out_dim + row] = bias[row] + acc;
    }
}

// GEMV fast path for skinny batches (B small — the autoregressive decode
// regime, where every linear is (B<=2, in_dim) @ W^T). The tiled kernel above
// puts one THREAD on each output row walking W serially, so at any k the warp's
// 32 threads hit addresses in_dim*4 bytes apart — 4 useful bytes per 128-byte
// transaction (~1/10 of DRAM bandwidth measured) and only out_dim threads of
// parallelism. Here one WARP owns each output row: lanes stride the K dimension
// with float4 loads (consecutive lanes -> consecutive 16-byte words, fully
// coalesced), then a shuffle tree reduces the row. X rows are small and shared
// across all warps, so they ride L2. Requires in_dim % 4 == 0 (every row then
// stays 16-byte aligned); other shapes keep the tiled kernel.
constexpr int GV_BLOCK = 128;   // threads cooperating on one output row

__global__ void linear_gemv_kernel(const float* __restrict__ W,
                                   const float* __restrict__ bias,
                                   const float* __restrict__ X,
                                   float* __restrict__ Y,
                                   int B, int out_dim, int in_dim) {
    // One BLOCK per (row, batch): out_dim*B blocks keep every SM saturated
    // with outstanding loads even at decode widths (out_dim ~1-3k), where a
    // warp-per-row mapping tops out at a couple hundred threads per SM.
    const int row = blockIdx.x;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;

    const float4* w4 = reinterpret_cast<const float4*>(
        W + static_cast<size_t>(row) * in_dim);
    const float4* x4 = reinterpret_cast<const float4*>(
        X + static_cast<size_t>(b) * in_dim);
    const int in4 = in_dim >> 2;

    float acc = 0.0f;
    for (int k = tid; k < in4; k += GV_BLOCK) {
        const float4 w = w4[k];
        const float4 x = x4[k];
        acc += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float red[GV_BLOCK / 32];
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float s = red[0];
        #pragma unroll
        for (int w = 1; w < GV_BLOCK / 32; ++w) s += red[w];
        Y[static_cast<size_t>(b) * out_dim + row] = bias[row] + s;
    }
}

constexpr int EW_BLOCK = 256;

constexpr int LBB_DX_BLOCK = 64;

template <typename T> __device__ inline float lbb_load(const T* p);
template <> __device__ inline float lbb_load<float>(const float* p)  { return *p; }
template <> __device__ inline float lbb_load<__half>(const __half* p){ return __half2float(*p); }
template <> __device__ inline float lbb_load<__nv_bfloat16>(const __nv_bfloat16* p){ return __bfloat162float(*p); }
template <typename T> __device__ inline void lbb_store(T* p, float v);
template <> __device__ inline void lbb_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void lbb_store<__half>(__half* p, float v) { *p = __float2half(v); }
template <> __device__ inline void lbb_store<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// 16-bit-weight twins of the two linear kernels above: W stored FP16/BF16,
// bias/X/Y and accumulation FP32. Autoregressive decode is weight-bandwidth
// bound at B<=2, so halving the weight bytes halves the per-step floor; the
// activations stay FP32 so the surrounding graph (norms, attention, residual
// stream) is untouched. Lanes load adjacent T2 words (consecutive 4-byte
// words across the warp — fully coalesced); requires in_dim % 2 == 0, which
// also keeps every weight row 4-byte aligned.
template <typename T2> __device__ inline float lbb_lo2(T2 v);
template <typename T2> __device__ inline float lbb_hi2(T2 v);
template <> __device__ inline float lbb_lo2<__half2>(__half2 v) { return __low2float(v); }
template <> __device__ inline float lbb_hi2<__half2>(__half2 v) { return __high2float(v); }
template <> __device__ inline float lbb_lo2<__nv_bfloat162>(__nv_bfloat162 v) { return __low2float(v); }
template <> __device__ inline float lbb_hi2<__nv_bfloat162>(__nv_bfloat162 v) { return __high2float(v); }

template <typename WT, typename WT2>
__global__ void linear_gemv_w16_kernel(const WT* __restrict__ W,
                                       const float* __restrict__ bias,
                                       const float* __restrict__ X,
                                       float* __restrict__ Y,
                                       int B, int out_dim, int in_dim) {
    const int row = blockIdx.x;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;

    const WT2* w2 = reinterpret_cast<const WT2*>(
        W + static_cast<size_t>(row) * in_dim);
    const float2* x2 = reinterpret_cast<const float2*>(
        X + static_cast<size_t>(b) * in_dim);
    const int in2 = in_dim >> 1;

    float acc = 0.0f;
    for (int k = 2 * tid; k < in2; k += 2 * GV_BLOCK) {  // two adjacent WT2 words
        const WT2 w0   = w2[k];
        const float2 x0 = x2[k];
        acc += lbb_lo2(w0) * x0.x + lbb_hi2(w0) * x0.y;
        const int k1 = k + 1;
        if (k1 < in2) {
            const WT2 w1   = w2[k1];
            const float2 x1 = x2[k1];
            acc += lbb_lo2(w1) * x1.x + lbb_hi2(w1) * x1.y;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    __shared__ float red[GV_BLOCK / 32];
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float s = red[0];
        #pragma unroll
        for (int w = 1; w < GV_BLOCK / 32; ++w) s += red[w];
        Y[static_cast<size_t>(b) * out_dim + row] = bias[row] + s;
    }
}

template <typename WT>
__global__ void linear_forward_batched_w16_kernel(const WT* __restrict__ W,
                                                  const float* __restrict__ bias,
                                                  const float* __restrict__ X,
                                                  float* __restrict__ Y,
                                                  int B, int out_dim, int in_dim) {
    __shared__ float xtile[BL_TILE];

    const int b   = blockIdx.y;
    const int row = blockIdx.x * BL_ROWS_PER_BLOCK + threadIdx.x;
    if (b >= B) return;

    const float* x_row = X + static_cast<size_t>(b) * in_dim;
    float acc = 0.0f;

    for (int t0 = 0; t0 < in_dim; t0 += BL_TILE) {
        const int t_len = (in_dim - t0) < BL_TILE ? (in_dim - t0) : BL_TILE;

        for (int k = threadIdx.x; k < t_len; k += blockDim.x) {
            xtile[k] = x_row[t0 + k];
        }
        __syncthreads();

        if (row < out_dim) {
            const WT* wrow = W + static_cast<size_t>(row) * in_dim + t0;
            #pragma unroll 8
            for (int k = 0; k < t_len; ++k) {
                acc += lbb_load<WT>(&wrow[k]) * xtile[k];
            }
        }
        __syncthreads();
    }

    if (row < out_dim) {
        Y[static_cast<size_t>(b) * out_dim + row] = bias[row] + acc;
    }
}

template <typename T>
__global__ void relu_forward_batched_kernel(const T* __restrict__ x,
                                            T* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = lbb_load<T>(&x[i]);
        lbb_store<T>(&y[i], v > 0.0f ? v : 0.0f);
    }
}

template <typename T>
__global__ void tanh_forward_batched_kernel(const T* __restrict__ x,
                                            T* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        lbb_store<T>(&y[i], tanhf(lbb_load<T>(&x[i])));
    }
}

template <typename T>
__global__ void add_inplace_batched_kernel(T* __restrict__ y,
                                           const T* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        lbb_store<T>(&y[i], lbb_load<T>(&y[i]) + lbb_load<T>(&x[i]));
    }
}

template <typename T>
__global__ void relu_backward_batched_kernel(const T* __restrict__ x,
                                             const T* __restrict__ dY,
                                             T* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float xv  = lbb_load<T>(&x[i]);
        const float dyv = lbb_load<T>(&dY[i]);
        lbb_store<T>(&dX[i], xv > 0.0f ? dyv : 0.0f);
    }
}

template <typename T>
__global__ void tanh_backward_batched_kernel(const T* __restrict__ y,
                                             const T* __restrict__ dY,
                                             T* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv  = lbb_load<T>(&y[i]);
        const float dyv = lbb_load<T>(&dY[i]);
        lbb_store<T>(&dX[i], dyv * (1.0f - yv * yv));
    }
}

template <typename T>
__global__ void linear_backward_batched_dx_kernel(const T* __restrict__ W,
                                                  const T* __restrict__ dY,
                                                  T* __restrict__ dX,
                                                  int B, int out_dim, int in_dim) {
    const int b = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= in_dim) return;
    const T* dY_row = dY + static_cast<size_t>(b) * out_dim;
    float acc = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        acc += lbb_load<T>(&W[static_cast<size_t>(i) * in_dim + j]) *
               lbb_load<T>(&dY_row[i]);
    }
    lbb_store<T>(&dX[static_cast<size_t>(b) * in_dim + j], acc);
}

template <typename T>
__global__ void linear_backward_batched_dw_kernel(const T* __restrict__ dY,
                                                  const T* __restrict__ X,
                                                  float* __restrict__ dW_scratch,
                                                  int B, int out_dim, int in_dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_dim || j >= in_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += lbb_load<T>(&dY[static_cast<size_t>(b) * out_dim + i]) *
               lbb_load<T>(&X [static_cast<size_t>(b) * in_dim  + j]);
    }
    dW_scratch[static_cast<size_t>(i) * in_dim + j] = acc;
}

template <typename T>
__global__ void linear_backward_batched_db_kernel(const T* __restrict__ dY,
                                                  float* __restrict__ dB_scratch,
                                                  int B, int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_dim) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        acc += lbb_load<T>(&dY[static_cast<size_t>(b) * out_dim + i]);
    }
    dB_scratch[i] = acc;
}

__global__ void lbb_add_fp32_into_fp16(const float* __restrict__ src,
                                       __half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(dst[i]) + src[i]);
}
__global__ void lbb_add_fp32_into_fp32(const float* __restrict__ src,
                                       float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}
__global__ void lbb_add_fp32_into_bf16(const float* __restrict__ src,
                                       __nv_bfloat16* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + src[i]);
}

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

// CUDA per-launch limit for gridDim.y / gridDim.z on every supported arch.
constexpr int LBB_MAX_GRID_Y = 65535;

} // anonymous namespace

void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD) {
    const bool w16 = (W.dtype == Dtype::FP16 || W.dtype == Dtype::BF16);
    if ((W.dtype != Dtype::FP32 && !w16) || X_BD.dtype != Dtype::FP32 ||
        bias.dtype != Dtype::FP32) {
        throw std::runtime_error(
            "linear_forward_batched: X and bias must be FP32, W FP32/FP16/BF16 "
            "(use linear_forward_batched_fp16 for 16-bit activations)");
    }
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim || Y_BD.dtype != Dtype::FP32) {
        Y_BD.resize(B, out_dim, Dtype::FP32);
    }
    if (B == 0 || out_dim == 0) return;

    const float* bias_p = static_cast<const float*>(bias.data);
    const float* X_p    = static_cast<const float*>(X_BD.data);
    float*       Y_p    = static_cast<float*>(Y_BD.data);

    if (w16) {
        // 16-bit weights, FP32 activations/accumulation. Same skinny/wide
        // split as the FP32 path below; the GEMV needs in_dim % 2 == 0 (T2
        // loads), which every transformer width satisfies.
        if (B <= 32 && in_dim % 2 == 0 && in_dim > 0) {
            dim3 grid(out_dim, B);
            if (W.dtype == Dtype::FP16) {
                linear_gemv_w16_kernel<__half, __half2><<<grid, GV_BLOCK, 0, cur_stream()>>>(
                    static_cast<const __half*>(W.data), bias_p, X_p, Y_p,
                    B, out_dim, in_dim);
            } else {
                linear_gemv_w16_kernel<__nv_bfloat16, __nv_bfloat162><<<grid, GV_BLOCK, 0, cur_stream()>>>(
                    static_cast<const __nv_bfloat16*>(W.data), bias_p, X_p, Y_p,
                    B, out_dim, in_dim);
            }
            BROTENSOR_CUDA_CHECK(cudaGetLastError());
            return;
        }
        dim3 block(BL_ROWS_PER_BLOCK, 1);
        const int grid_x = (out_dim + BL_ROWS_PER_BLOCK - 1) / BL_ROWS_PER_BLOCK;
        for (int b0 = 0; b0 < B; b0 += LBB_MAX_GRID_Y) {
            const int b_chunk = (B - b0) < LBB_MAX_GRID_Y ? (B - b0) : LBB_MAX_GRID_Y;
            dim3 grid(grid_x, b_chunk);
            if (W.dtype == Dtype::FP16) {
                linear_forward_batched_w16_kernel<__half><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const __half*>(W.data), bias_p,
                    X_p + static_cast<size_t>(b0) * in_dim,
                    Y_p + static_cast<size_t>(b0) * out_dim,
                    b_chunk, out_dim, in_dim);
            } else {
                linear_forward_batched_w16_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const __nv_bfloat16*>(W.data), bias_p,
                    X_p + static_cast<size_t>(b0) * in_dim,
                    Y_p + static_cast<size_t>(b0) * out_dim,
                    b_chunk, out_dim, in_dim);
            }
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    const float* W_p = static_cast<const float*>(W.data);

    // Skinny batches take the warp-per-row GEMV (see linear_gemv_kernel); the
    // tiled thread-per-row kernel remains for wide batches and odd in_dim.
    if (B <= 32 && in_dim % 4 == 0 && in_dim > 0) {
        dim3 grid(out_dim, B);
        linear_gemv_kernel<<<grid, GV_BLOCK, 0, cur_stream()>>>(
            W_p, bias_p, X_p, Y_p, B, out_dim, in_dim);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

    dim3 block(BL_ROWS_PER_BLOCK, 1);
    const int grid_x = (out_dim + BL_ROWS_PER_BLOCK - 1) / BL_ROWS_PER_BLOCK;
    for (int b0 = 0; b0 < B; b0 += LBB_MAX_GRID_Y) {
        const int b_chunk = (B - b0) < LBB_MAX_GRID_Y ? (B - b0) : LBB_MAX_GRID_Y;
        dim3 grid(grid_x, b_chunk);
        linear_forward_batched_kernel<<<grid, block, 0, cur_stream()>>>(
            W_p, bias_p,
            X_p + static_cast<size_t>(b0) * in_dim,
            Y_p + static_cast<size_t>(b0) * out_dim,
            b_chunk, out_dim, in_dim);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols ||
        Y_BD.dtype != X_BD.dtype) {
        Y_BD.resize(X_BD.rows, X_BD.cols, X_BD.dtype);
    }
    const int n = X_BD.size();
    if (n == 0) return;
    if (X_BD.dtype == Dtype::FP16) {
        relu_forward_batched_kernel<__half><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X_BD.data),
            static_cast<__half*>(Y_BD.data), n);
    } else if (X_BD.dtype == Dtype::BF16) {
        relu_forward_batched_kernel<__nv_bfloat16><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X_BD.data),
            static_cast<__nv_bfloat16*>(Y_BD.data), n);
    } else {
        relu_forward_batched_kernel<float><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X_BD.data),
            static_cast<float*>(Y_BD.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward_batched(const Tensor& X_BD, Tensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols ||
        Y_BD.dtype != X_BD.dtype) {
        Y_BD.resize(X_BD.rows, X_BD.cols, X_BD.dtype);
    }
    const int n = X_BD.size();
    if (n == 0) return;
    if (X_BD.dtype == Dtype::FP16) {
        tanh_forward_batched_kernel<__half><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X_BD.data),
            static_cast<__half*>(Y_BD.data), n);
    } else if (X_BD.dtype == Dtype::BF16) {
        tanh_forward_batched_kernel<__nv_bfloat16><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X_BD.data),
            static_cast<__nv_bfloat16*>(Y_BD.data), n);
    } else {
        tanh_forward_batched_kernel<float><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X_BD.data),
            static_cast<float*>(Y_BD.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_inplace_batched(Tensor& Y_BD, const Tensor& X_BD) {
    if (Y_BD.dtype != X_BD.dtype) {
        throw std::runtime_error("add_inplace_batched: dtype mismatch");
    }
    const int n = Y_BD.size();
    if (n == 0) return;
    if (Y_BD.dtype == Dtype::FP16) {
        add_inplace_batched_kernel<__half><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__half*>(Y_BD.data),
            static_cast<const __half*>(X_BD.data), n);
    } else if (Y_BD.dtype == Dtype::BF16) {
        add_inplace_batched_kernel<__nv_bfloat16><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<__nv_bfloat16*>(Y_BD.data),
            static_cast<const __nv_bfloat16*>(X_BD.data), n);
    } else {
        add_inplace_batched_kernel<float><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<float*>(Y_BD.data),
            static_cast<const float*>(X_BD.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_backward_batched(const Tensor& X_BD, const Tensor& dY_BD,
                           Tensor& dX_BD) {
    if (dX_BD.rows != X_BD.rows || dX_BD.cols != X_BD.cols ||
        dX_BD.dtype != X_BD.dtype) {
        dX_BD.resize(X_BD.rows, X_BD.cols, X_BD.dtype);
    }
    const int n = X_BD.size();
    if (n == 0) return;
    if (X_BD.dtype == Dtype::FP16) {
        relu_backward_batched_kernel<__half><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(X_BD.data),
            static_cast<const __half*>(dY_BD.data),
            static_cast<__half*>(dX_BD.data), n);
    } else if (X_BD.dtype == Dtype::BF16) {
        relu_backward_batched_kernel<__nv_bfloat16><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(X_BD.data),
            static_cast<const __nv_bfloat16*>(dY_BD.data),
            static_cast<__nv_bfloat16*>(dX_BD.data), n);
    } else {
        relu_backward_batched_kernel<float><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(X_BD.data),
            static_cast<const float*>(dY_BD.data),
            static_cast<float*>(dX_BD.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_backward_batched(const Tensor& Y_BD, const Tensor& dY_BD,
                           Tensor& dX_BD) {
    if (dX_BD.rows != Y_BD.rows || dX_BD.cols != Y_BD.cols ||
        dX_BD.dtype != Y_BD.dtype) {
        dX_BD.resize(Y_BD.rows, Y_BD.cols, Y_BD.dtype);
    }
    const int n = Y_BD.size();
    if (n == 0) return;
    if (Y_BD.dtype == Dtype::FP16) {
        tanh_backward_batched_kernel<__half><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __half*>(Y_BD.data),
            static_cast<const __half*>(dY_BD.data),
            static_cast<__half*>(dX_BD.data), n);
    } else if (Y_BD.dtype == Dtype::BF16) {
        tanh_backward_batched_kernel<__nv_bfloat16><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const __nv_bfloat16*>(Y_BD.data),
            static_cast<const __nv_bfloat16*>(dY_BD.data),
            static_cast<__nv_bfloat16*>(dX_BD.data), n);
    } else {
        tanh_backward_batched_kernel<float><<<grid_for(n), EW_BLOCK, 0, cur_stream()>>>(
            static_cast<const float*>(Y_BD.data),
            static_cast<const float*>(dY_BD.data),
            static_cast<float*>(dX_BD.data), n);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_backward_batched(const Tensor& W, const Tensor& X_BD,
                             const Tensor& dY_BD,
                             Tensor& dX_BD,
                             Tensor& dW, Tensor& dB) {
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::FP32 && W.dtype != Dtype::BF16) {
        throw std::runtime_error("linear_backward_batched: W must be FP16, BF16, or FP32");
    }
    if (X_BD.dtype != W.dtype || dY_BD.dtype != W.dtype ||
        dW.dtype != W.dtype || dB.dtype != W.dtype) {
        throw std::runtime_error("linear_backward_batched: all tensors must share dtype");
    }
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;

    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != W.dtype) {
        dX_BD.resize(B, in_dim, W.dtype);
    }
    if (B == 0) return;

    const bool is_fp16 = (W.dtype == Dtype::FP16);
    const bool is_bf16 = (W.dtype == Dtype::BF16);

    if (in_dim > 0 && out_dim > 0) {
        dim3 block(LBB_DX_BLOCK, 1);
        const int grid_x = (in_dim + LBB_DX_BLOCK - 1) / LBB_DX_BLOCK;
        for (int b0 = 0; b0 < B; b0 += LBB_MAX_GRID_Y) {
            const int b_chunk = (B - b0) < LBB_MAX_GRID_Y ? (B - b0) : LBB_MAX_GRID_Y;
            dim3 grid(grid_x, b_chunk);
            if (is_fp16) {
                linear_backward_batched_dx_kernel<__half><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const __half*>(W.data),
                    static_cast<const __half*>(dY_BD.data)
                        + static_cast<size_t>(b0) * out_dim,
                    static_cast<__half*>(dX_BD.data)
                        + static_cast<size_t>(b0) * in_dim,
                    b_chunk, out_dim, in_dim);
            } else if (is_bf16) {
                linear_backward_batched_dx_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const __nv_bfloat16*>(W.data),
                    static_cast<const __nv_bfloat16*>(dY_BD.data)
                        + static_cast<size_t>(b0) * out_dim,
                    static_cast<__nv_bfloat16*>(dX_BD.data)
                        + static_cast<size_t>(b0) * in_dim,
                    b_chunk, out_dim, in_dim);
            } else {
                linear_backward_batched_dx_kernel<float><<<grid, block, 0, cur_stream()>>>(
                    static_cast<const float*>(W.data),
                    static_cast<const float*>(dY_BD.data)
                        + static_cast<size_t>(b0) * out_dim,
                    static_cast<float*>(dX_BD.data)
                        + static_cast<size_t>(b0) * in_dim,
                    b_chunk, out_dim, in_dim);
            }
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    if (out_dim > 0 && in_dim > 0) {
        const int dw_n = out_dim * in_dim;
        float* d_dw_scratch = static_cast<float*>(cuda_alloc(static_cast<size_t>(dw_n) * sizeof(float)));
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        if (is_fp16) {
            linear_backward_batched_dw_kernel<__half><<<grid, block, 0, cur_stream()>>>(
                static_cast<const __half*>(dY_BD.data),
                static_cast<const __half*>(X_BD.data),
                d_dw_scratch, B, out_dim, in_dim);
        } else if (is_bf16) {
            linear_backward_batched_dw_kernel<__nv_bfloat16><<<grid, block, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY_BD.data),
                static_cast<const __nv_bfloat16*>(X_BD.data),
                d_dw_scratch, B, out_dim, in_dim);
        } else {
            linear_backward_batched_dw_kernel<float><<<grid, block, 0, cur_stream()>>>(
                static_cast<const float*>(dY_BD.data),
                static_cast<const float*>(X_BD.data),
                d_dw_scratch, B, out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (dw_n + 255) / 256;
        if (is_fp16) {
            lbb_add_fp32_into_fp16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_dw_scratch, static_cast<__half*>(dW.data), dw_n);
        } else if (is_bf16) {
            lbb_add_fp32_into_bf16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_dw_scratch, static_cast<__nv_bfloat16*>(dW.data), dw_n);
        } else {
            lbb_add_fp32_into_fp32<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_dw_scratch, static_cast<float*>(dW.data), dw_n);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(d_dw_scratch);
    }

    if (out_dim > 0) {
        float* d_db_scratch = static_cast<float*>(cuda_alloc(static_cast<size_t>(out_dim) * sizeof(float)));
        const int blocks = (out_dim + 255) / 256;
        if (is_fp16) {
            linear_backward_batched_db_kernel<__half><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const __half*>(dY_BD.data),
                d_db_scratch, B, out_dim);
        } else if (is_bf16) {
            linear_backward_batched_db_kernel<__nv_bfloat16><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const __nv_bfloat16*>(dY_BD.data),
                d_db_scratch, B, out_dim);
        } else {
            linear_backward_batched_db_kernel<float><<<blocks, 256, 0, cur_stream()>>>(
                static_cast<const float*>(dY_BD.data),
                d_db_scratch, B, out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (out_dim + 255) / 256;
        if (is_fp16) {
            lbb_add_fp32_into_fp16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_db_scratch, static_cast<__half*>(dB.data), out_dim);
        } else if (is_bf16) {
            lbb_add_fp32_into_bf16<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_db_scratch, static_cast<__nv_bfloat16*>(dB.data), out_dim);
        } else {
            lbb_add_fp32_into_fp32<<<blocks_fold, 256, 0, cur_stream()>>>(
                d_db_scratch, static_cast<float*>(dB.data), out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cuda_free(d_db_scratch);
    }
}

} // namespace brotensor::detail::cuda
