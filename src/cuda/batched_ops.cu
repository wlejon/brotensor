// Batched (inference-only) GPU kernels for the BatchedInferenceServer.
//
// Each op runs B independent forward passes in a single kernel launch. We do
// not modify the existing single-sample kernels — these are additive.
//
// Tensor layout: a (B, D) row-major tensor, so sample b occupies the
// half-open row range [b*D, (b+1)*D). Linear shares one weight matrix W of
// shape (out_dim, in_dim) across all B rows.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

// ─── linear_forward_batched ────────────────────────────────────────────────
//
// One block per (sample, output-row tile). Within a block, threadIdx.x walks
// the output rows of the tile and threadIdx.y indexes the sample. Because B
// can be small (1) and out_dim moderate, we keep the parallelism in the
// out_dim axis and one block per sample.
//
// We tile the in_dim axis through shared memory, sized for the whole block.

constexpr int BL_ROWS_PER_BLOCK = 64;   // output rows per block
constexpr int BL_TILE           = 64;   // in_dim tile width

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

        // Cooperatively load a tile of x_row into shared memory.
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

// ─── elementwise batched ───────────────────────────────────────────────────
//
// (B, D) is just a flat buffer of size B*D as far as elementwise ops are
// concerned. We reuse the same grid-stride pattern as the single-sample
// elementwise kernels.

constexpr int EW_BLOCK = 256;

__global__ void relu_forward_batched_kernel(const float* __restrict__ x,
                                            float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void tanh_forward_batched_kernel(const float* __restrict__ x,
                                            float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = tanhf(x[i]);
    }
}

__global__ void add_inplace_batched_kernel(float* __restrict__ y,
                                           const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        y[i] += x[i];
    }
}

__global__ void relu_backward_batched_kernel(const float* __restrict__ x,
                                             const float* __restrict__ dY,
                                             float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
    }
}

__global__ void tanh_backward_batched_kernel(const float* __restrict__ y,
                                             const float* __restrict__ dY,
                                             float* __restrict__ dX, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float yv = y[i];
        dX[i] = dY[i] * (1.0f - yv * yv);
    }
}

// Linear backward over a B-row minibatch. Kept in the batched_ops TU so the
// matched forward and backward live together. Dtype-dispatched (FP32/FP16);
// internal FP32 accumulation. For FP16 storage of dW/dB, we use an FP32
// scratch + fold-back epilogue (Bundle 2 pattern).

constexpr int LBB_DX_BLOCK = 64;     // threads per block (in_dim axis)

template <typename T> __device__ inline float lbb_load(const T* p);
template <> __device__ inline float lbb_load<float>(const float* p)  { return *p; }
template <> __device__ inline float lbb_load<__half>(const __half* p){ return __half2float(*p); }
template <typename T> __device__ inline void lbb_store(T* p, float v);
template <> __device__ inline void lbb_store<float>(float* p, float v) { *p = v; }
template <> __device__ inline void lbb_store<__half>(__half* p, float v) { *p = __float2half(v); }

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

// dW_scratch[i, j] = sum_b dY[b, i] * X[b, j] (FP32 scratch, then folded).
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

// dB_scratch[i] = sum_b dY[b, i] (FP32 scratch).
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

inline int grid_for(int n) {
    int blocks = (n + EW_BLOCK - 1) / EW_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return blocks;
}

} // anonymous namespace

void linear_forward_batched_gpu(const GpuTensor& W, const GpuTensor& bias,
                                const GpuTensor& X_BD, GpuTensor& Y_BD) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim) Y_BD.resize(B, out_dim);
    if (B == 0 || out_dim == 0) return;

    dim3 block(BL_ROWS_PER_BLOCK, 1);
    dim3 grid((out_dim + BL_ROWS_PER_BLOCK - 1) / BL_ROWS_PER_BLOCK, B);
    linear_forward_batched_kernel<<<grid, block>>>(
        W.data, bias.data, X_BD.data, Y_BD.data, B, out_dim, in_dim);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_forward_batched_gpu(const GpuTensor& X_BD, GpuTensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols)
        Y_BD.resize(X_BD.rows, X_BD.cols);
    const int n = X_BD.size();
    if (n == 0) return;
    relu_forward_batched_kernel<<<grid_for(n), EW_BLOCK>>>(X_BD.data, Y_BD.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_forward_batched_gpu(const GpuTensor& X_BD, GpuTensor& Y_BD) {
    if (Y_BD.rows != X_BD.rows || Y_BD.cols != X_BD.cols)
        Y_BD.resize(X_BD.rows, X_BD.cols);
    const int n = X_BD.size();
    if (n == 0) return;
    tanh_forward_batched_kernel<<<grid_for(n), EW_BLOCK>>>(X_BD.data, Y_BD.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void add_inplace_batched_gpu(GpuTensor& Y_BD, const GpuTensor& X_BD) {
    const int n = Y_BD.size();
    if (n == 0) return;
    add_inplace_batched_kernel<<<grid_for(n), EW_BLOCK>>>(Y_BD.data, X_BD.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void relu_backward_batched_gpu(const GpuTensor& X_BD, const GpuTensor& dY_BD,
                               GpuTensor& dX_BD) {
    if (dX_BD.rows != X_BD.rows || dX_BD.cols != X_BD.cols)
        dX_BD.resize(X_BD.rows, X_BD.cols);
    const int n = X_BD.size();
    if (n == 0) return;
    relu_backward_batched_kernel<<<grid_for(n), EW_BLOCK>>>(
        X_BD.data, dY_BD.data, dX_BD.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void tanh_backward_batched_gpu(const GpuTensor& Y_BD, const GpuTensor& dY_BD,
                               GpuTensor& dX_BD) {
    if (dX_BD.rows != Y_BD.rows || dX_BD.cols != Y_BD.cols)
        dX_BD.resize(Y_BD.rows, Y_BD.cols);
    const int n = Y_BD.size();
    if (n == 0) return;
    tanh_backward_batched_kernel<<<grid_for(n), EW_BLOCK>>>(
        Y_BD.data, dY_BD.data, dX_BD.data, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void linear_backward_batched_gpu(const GpuTensor& W, const GpuTensor& X_BD,
                                 const GpuTensor& dY_BD,
                                 GpuTensor& dX_BD,
                                 GpuTensor& dW, GpuTensor& dB) {
    if (W.dtype != Dtype::FP16 && W.dtype != Dtype::FP32) {
        throw std::runtime_error("linear_backward_batched_gpu: W must be FP16 or FP32");
    }
    if (X_BD.dtype != W.dtype || dY_BD.dtype != W.dtype ||
        dW.dtype != W.dtype || dB.dtype != W.dtype) {
        throw std::runtime_error("linear_backward_batched_gpu: all tensors must share dtype");
    }
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;

    if (dX_BD.rows != B || dX_BD.cols != in_dim || dX_BD.dtype != W.dtype) {
        dX_BD.resize(B, in_dim, W.dtype);
    }
    if (B == 0) return;

    const bool is_fp16 = (W.dtype == Dtype::FP16);

    // dX = dY @ W (per row).
    if (in_dim > 0 && out_dim > 0) {
        dim3 block(LBB_DX_BLOCK, 1);
        dim3 grid((in_dim + LBB_DX_BLOCK - 1) / LBB_DX_BLOCK, B);
        if (is_fp16) {
            linear_backward_batched_dx_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(W.data_fp16()),
                reinterpret_cast<const __half*>(dY_BD.data_fp16()),
                reinterpret_cast<__half*>(dX_BD.data_fp16()),
                B, out_dim, in_dim);
        } else {
            linear_backward_batched_dx_kernel<float><<<grid, block>>>(
                W.data, dY_BD.data, dX_BD.data, B, out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dW += dY^T @ X (sum over B). FP32 scratch + fold.
    if (out_dim > 0 && in_dim > 0) {
        const int dw_n = out_dim * in_dim;
        float* d_dw_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dw_scratch),
                                        dw_n * sizeof(float)));
        dim3 block(16, 16);
        dim3 grid((in_dim + 15) / 16, (out_dim + 15) / 16);
        if (is_fp16) {
            linear_backward_batched_dw_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(dY_BD.data_fp16()),
                reinterpret_cast<const __half*>(X_BD.data_fp16()),
                d_dw_scratch, B, out_dim, in_dim);
        } else {
            linear_backward_batched_dw_kernel<float><<<grid, block>>>(
                dY_BD.data, X_BD.data, d_dw_scratch, B, out_dim, in_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (dw_n + 255) / 256;
        if (is_fp16) {
            lbb_add_fp32_into_fp16<<<blocks_fold, 256>>>(
                d_dw_scratch, reinterpret_cast<__half*>(dW.data_fp16()), dw_n);
        } else {
            lbb_add_fp32_into_fp32<<<blocks_fold, 256>>>(
                d_dw_scratch, dW.data, dw_n);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_dw_scratch);
    }

    // dB += sum_b dY[b]. FP32 scratch + fold.
    if (out_dim > 0) {
        float* d_db_scratch = nullptr;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_db_scratch),
                                        out_dim * sizeof(float)));
        const int blocks = (out_dim + 255) / 256;
        if (is_fp16) {
            linear_backward_batched_db_kernel<__half><<<blocks, 256>>>(
                reinterpret_cast<const __half*>(dY_BD.data_fp16()),
                d_db_scratch, B, out_dim);
        } else {
            linear_backward_batched_db_kernel<float><<<blocks, 256>>>(
                dY_BD.data, d_db_scratch, B, out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        const int blocks_fold = (out_dim + 255) / 256;
        if (is_fp16) {
            lbb_add_fp32_into_fp16<<<blocks_fold, 256>>>(
                d_db_scratch, reinterpret_cast<__half*>(dB.data_fp16()), out_dim);
        } else {
            lbb_add_fp32_into_fp32<<<blocks_fold, 256>>>(
                d_db_scratch, dB.data, out_dim);
        }
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        cudaFree(d_db_scratch);
    }
}

} // namespace brotensor
