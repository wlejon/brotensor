#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

// Current CUDA stream for hot-op launches — so kernels join a non-default
// capture/replay stream instead of silently landing on the default stream.
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

// Naive correctness-first kernels mirroring src/nn/attention.cpp.
//
// Shapes:
//   X       : (N, D)
//   Wq/Wk/Wv/Wo : (D, D)        rows = output dim, cols = input dim
//   Q/K/V   : (N, D)
//   Attn    : (N, N)
//   Y_pre_Wo: (N, D)            == Attn @ V
//   O       : (N, D)
//   d_mask  : optional length N (1 valid, 0 invalid). Invalid keys excluded
//             from softmax denom; invalid query rows produce zero output.

namespace {

// Y(i, j) = sum_k X(i, k) * W(j, k)   — i.e. Y = X @ W^T.
// One thread per (i, j).
__global__ void matmul_xwT_kernel(const float* __restrict__ X,
                                  const float* __restrict__ W,
                                  float* __restrict__ Y,
                                  int N, int D_in, int D_out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= D_out) return;
    const float* xr = X + static_cast<size_t>(i) * D_in;
    const float* wr = W + static_cast<size_t>(j) * D_in;
    float acc = 0.0f;
    for (int k = 0; k < D_in; ++k) acc += xr[k] * wr[k];
    Y[static_cast<size_t>(i) * D_out + j] = acc;
}

// scores(i, j) = (Q_i . K_j) / sqrt(D)
// One thread per (i, j) of the (N, N) score matrix.
__global__ void scores_kernel(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              float* __restrict__ S,
                              int N, int D, float inv_sqrtd) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    const float* qr = Q + static_cast<size_t>(i) * D;
    const float* kr = K + static_cast<size_t>(j) * D;
    float s = 0.0f;
    for (int k = 0; k < D; ++k) s += qr[k] * kr[k];
    S[static_cast<size_t>(i) * N + j] = s * inv_sqrtd;
}

// Per-row masked softmax of an (N, N) matrix in place into Attn.
// One block per row i. Invalid query rows (mask[i]==0) get all-zero rows.
// Within a valid row, masked columns (mask[j]==0) get probability 0.
constexpr int ROW_SM_BLOCK = 256;
__global__ void row_masked_softmax_kernel(const float* __restrict__ scores,
                                          float* __restrict__ Attn,
                                          const float* __restrict__ mask,
                                          int N) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(i) * N;
    float* arow = Attn + static_cast<size_t>(i) * N;

    // Invalid query → zero row.
    if (mask && mask[i] < 0.5f) {
        for (int j = tid; j < N; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    // Phase 1: max over valid keys.
    float local_max = -1e30f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        const float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    // Phase 2: exp(x - m), zero on masked, accumulate sum.
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            arow[j] = 0.0f;
            continue;
        }
        const float e = expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float sum = sdata[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        arow[j] = arow[j] * inv;
    }
}

// Y(i, k) = sum_j Attn(i, j) * V(j, k).  One thread per (i, k).
__global__ void attn_apply_v_kernel(const float* __restrict__ Attn,
                                    const float* __restrict__ V,
                                    float* __restrict__ Y,
                                    int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    const float* arow = Attn + static_cast<size_t>(i) * N;
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) acc += arow[j] * V[static_cast<size_t>(j) * D + k];
    Y[static_cast<size_t>(i) * D + k] = acc;
}

// O(i, c) = mask[i] ? sum_k Y(i, k) * Wo(c, k) : 0
__global__ void output_proj_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ Wo,
                                   const float* __restrict__ mask,
                                   float* __restrict__ O,
                                   int N, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || c >= D) return;
    if (mask && mask[i] < 0.5f) {
        O[static_cast<size_t>(i) * D + c] = 0.0f;
        return;
    }
    const float* yr = Y + static_cast<size_t>(i) * D;
    const float* wr = Wo + static_cast<size_t>(c) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
    O[static_cast<size_t>(i) * D + c] = acc;
}

// ─── Backward ─────────────────────────────────────────────────────────────

// dY(i, k) = mask[i] ? sum_c Wo(c, k) * dO(i, c) : 0
// dWo(c, k) += sum_i mask[i] * dO(i, c) * Y(i, k)   — but we accumulate per (c,k)
//
// Step A: compute dY  (one thread per (i, k))
__global__ void wo_back_dY_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ Wo,
                                  const float* __restrict__ mask,
                                  float* __restrict__ dY,
                                  int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    if (mask && mask[i] < 0.5f) {
        dY[static_cast<size_t>(i) * D + k] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (int c = 0; c < D; ++c) {
        acc += Wo[static_cast<size_t>(c) * D + k] * dO[static_cast<size_t>(i) * D + c];
    }
    dY[static_cast<size_t>(i) * D + k] = acc;
}

// Step B: dWo(c, k) += sum_i (mask[i] ? dO(i,c) * Y(i,k) : 0)
__global__ void wo_back_dW_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ Y,
                                  const float* __restrict__ mask,
                                  float* __restrict__ dWo,
                                  int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (mask && mask[i] < 0.5f) continue;
        acc += dO[static_cast<size_t>(i) * D + c] * Y[static_cast<size_t>(i) * D + k];
    }
    dWo[static_cast<size_t>(c) * D + k] += acc;
}

// dAttn(i, j) = sum_k dY(i, k) * V(j, k)    — one thread per (i, j)
__global__ void dAttn_kernel(const float* __restrict__ dY,
                             const float* __restrict__ V,
                             float* __restrict__ dAttn,
                             int N, int D) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) {
        acc += dY[static_cast<size_t>(i) * D + k] * V[static_cast<size_t>(j) * D + k];
    }
    dAttn[static_cast<size_t>(i) * N + j] = acc;
}

// dV(j, k) = sum_i Attn(i, j) * dY(i, k)    — one thread per (j, k)
__global__ void dV_kernel(const float* __restrict__ Attn,
                          const float* __restrict__ dY,
                          float* __restrict__ dV,
                          int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= N || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += Attn[static_cast<size_t>(i) * N + j] * dY[static_cast<size_t>(i) * D + k];
    }
    dV[static_cast<size_t>(j) * D + k] = acc;
}

// Per-row softmax backward into dScores.
// Row i with mask[i]==0 → all zero.
// Otherwise: row_dz[j] = p_j * (dp_j - sum_l dp_l * p_l), then zero where mask[j]==0,
// finally scale by inv_sqrtd.
__global__ void row_softmax_back_kernel(const float* __restrict__ Attn,
                                        const float* __restrict__ dAttn,
                                        const float* __restrict__ mask,
                                        float* __restrict__ dScores,
                                        int N, float inv_sqrtd) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const float* prow  = Attn  + static_cast<size_t>(i) * N;
    const float* dprow = dAttn + static_cast<size_t>(i) * N;
    float* drow = dScores + static_cast<size_t>(i) * N;

    if (mask && mask[i] < 0.5f) {
        for (int j = tid; j < N; j += blockDim.x) drow[j] = 0.0f;
        return;
    }

    float local = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        local += dprow[j] * prow[j];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int j = tid; j < N; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtd;
        }
    }
}

// dQ(i, k) = sum_j dScores(i, j) * K(j, k)
__global__ void dQ_kernel(const float* __restrict__ dScores,
                          const float* __restrict__ K,
                          float* __restrict__ dQ,
                          int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += dScores[static_cast<size_t>(i) * N + j] * K[static_cast<size_t>(j) * D + k];
    }
    dQ[static_cast<size_t>(i) * D + k] = acc;
}

// dK(j, k) = sum_i dScores(i, j) * Q(i, k)
__global__ void dK_kernel(const float* __restrict__ dScores,
                          const float* __restrict__ Q,
                          float* __restrict__ dK,
                          int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= N || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += dScores[static_cast<size_t>(i) * N + j] * Q[static_cast<size_t>(i) * D + k];
    }
    dK[static_cast<size_t>(j) * D + k] = acc;
}

// Final input-projection backward, mirroring CPU's fused loop:
//   dWq(j, k) += sum_i dQ(i, j) * X(i, k)        (and likewise dWk, dWv)
//   dX(i, k)   = sum_j dQ(i, j) * Wq(j, k)
//              + sum_j dK(i, j) * Wk(j, k)
//              + sum_j dV(i, j) * Wv(j, k)
//
// Two kernels: dW accumulation (one thread per (j, k)) and dX (one per (i, k)).
__global__ void dWqkv_kernel(const float* __restrict__ dQ,
                             const float* __restrict__ dK,
                             const float* __restrict__ dV,
                             const float* __restrict__ X,
                             float* __restrict__ dWq,
                             float* __restrict__ dWk,
                             float* __restrict__ dWv,
                             int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= D || k >= D) return;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float xv = X[static_cast<size_t>(i) * D + k];
        aq += dQ[static_cast<size_t>(i) * D + j] * xv;
        ak += dK[static_cast<size_t>(i) * D + j] * xv;
        av += dV[static_cast<size_t>(i) * D + j] * xv;
    }
    const size_t idx = static_cast<size_t>(j) * D + k;
    dWq[idx] += aq;
    dWk[idx] += ak;
    dWv[idx] += av;
}

__global__ void dX_proj_kernel(const float* __restrict__ dQ,
                               const float* __restrict__ dK,
                               const float* __restrict__ dV,
                               const float* __restrict__ Wq,
                               const float* __restrict__ Wk,
                               const float* __restrict__ Wv,
                               float* __restrict__ dX,
                               int N, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= D) return;
    float acc = 0.0f;
    for (int j = 0; j < D; ++j) {
        const size_t widx = static_cast<size_t>(j) * D + k;
        acc += dQ[static_cast<size_t>(i) * D + j] * Wq[widx]
             + dK[static_cast<size_t>(i) * D + j] * Wk[widx]
             + dV[static_cast<size_t>(i) * D + j] * Wv[widx];
    }
    dX[static_cast<size_t>(i) * D + k] = acc;
}

inline dim3 grid2d(int x, int y, dim3 block) {
    return dim3((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);
}

} // namespace

void attention_forward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                       const float* d_mask,
                       ::brotensor::Tensor& Q, ::brotensor::Tensor& K, ::brotensor::Tensor& V,
                       ::brotensor::Tensor& Attn, ::brotensor::Tensor& Y_pre_Wo,
                       ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    const int N = X.rows;
    const int D = X.cols;
    if (Q.rows != N || Q.cols != D) Q.resize(N, D, Dtype::FP32);
    if (K.rows != N || K.cols != D) K.resize(N, D, Dtype::FP32);
    if (V.rows != N || V.cols != D) V.resize(N, D, Dtype::FP32);
    if (Attn.rows != N || Attn.cols != N) Attn.resize(N, N, Dtype::FP32);
    if (Y_pre_Wo.rows != N || Y_pre_Wo.cols != D) Y_pre_Wo.resize(N, D, Dtype::FP32);
    if (O.rows != N || O.cols != D) O.resize(N, D, Dtype::FP32);
    if (N == 0 || D == 0) return;

    const dim3 block_nd(16, 16);
    const dim3 block_nn(16, 16);
    const dim3 block_dd(16, 16);

    const float* X_p   = static_cast<const float*>(X.data);
    const float* Wq_p  = static_cast<const float*>(Wq.data);
    const float* Wk_p  = static_cast<const float*>(Wk.data);
    const float* Wv_p  = static_cast<const float*>(Wv.data);
    const float* Wo_p  = static_cast<const float*>(Wo.data);
    float* Q_p         = static_cast<float*>(Q.data);
    float* K_p         = static_cast<float*>(K.data);
    float* V_p         = static_cast<float*>(V.data);
    float* Attn_p      = static_cast<float*>(Attn.data);
    float* Y_p         = static_cast<float*>(Y_pre_Wo.data);
    float* O_p         = static_cast<float*>(O.data);

    // Q, K, V projections.
    {
        dim3 grid = grid2d(D, N, block_nd);
        matmul_xwT_kernel<<<grid, block_nd, 0, cur_stream()>>>(X_p, Wq_p, Q_p, N, D, D);
        matmul_xwT_kernel<<<grid, block_nd, 0, cur_stream()>>>(X_p, Wk_p, K_p, N, D, D);
        matmul_xwT_kernel<<<grid, block_nd, 0, cur_stream()>>>(X_p, Wv_p, V_p, N, D, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Scores (N, N).
    Tensor scores = Tensor::empty_on(Device::CUDA, N, N, Dtype::FP32);
    float* scores_p = static_cast<float*>(scores.data);
    {
        const float inv_sqrtd = 1.0f / sqrtf(static_cast<float>(D));
        dim3 grid = grid2d(N, N, block_nn);
        scores_kernel<<<grid, block_nn, 0, cur_stream()>>>(Q_p, K_p, scores_p,
                                          N, D, inv_sqrtd);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Row-masked softmax → Attn.
    row_masked_softmax_kernel<<<N, ROW_SM_BLOCK, 0, cur_stream()>>>(scores_p, Attn_p,
                                                   d_mask, N);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // Y_pre_Wo = Attn @ V.
    {
        dim3 grid = grid2d(D, N, block_nd);
        attn_apply_v_kernel<<<grid, block_nd, 0, cur_stream()>>>(Attn_p, V_p, Y_p,
                                                N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // O = Y @ Wo^T (with row mask zeroing).
    {
        dim3 grid = grid2d(D, N, block_nd);
        output_proj_kernel<<<grid, block_nd, 0, cur_stream()>>>(Y_p, Wo_p, d_mask,
                                               O_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    (void)block_dd;
}

void attention_backward(const ::brotensor::Tensor& dO,
                        const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K,
                        const ::brotensor::Tensor& V, const ::brotensor::Tensor& Attn,
                        const ::brotensor::Tensor& Y_pre_Wo,
                        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                        const float* d_mask,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                        ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    const int N = X.rows;
    const int D = X.cols;
    if (dX.rows != N || dX.cols != D) dX.resize(N, D, Dtype::FP32);
    if (N == 0 || D == 0) return;

    const float inv_sqrtd = 1.0f / sqrtf(static_cast<float>(D));
    const dim3 block_nd(16, 16);
    const dim3 block_nn(16, 16);
    const dim3 block_dd(16, 16);

    const float* dO_p   = static_cast<const float*>(dO.data);
    const float* X_p    = static_cast<const float*>(X.data);
    const float* Q_p    = static_cast<const float*>(Q.data);
    const float* K_p    = static_cast<const float*>(K.data);
    const float* V_p    = static_cast<const float*>(V.data);
    const float* Attn_p = static_cast<const float*>(Attn.data);
    const float* Y_p    = static_cast<const float*>(Y_pre_Wo.data);
    const float* Wq_p   = static_cast<const float*>(Wq.data);
    const float* Wk_p   = static_cast<const float*>(Wk.data);
    const float* Wv_p   = static_cast<const float*>(Wv.data);
    const float* Wo_p   = static_cast<const float*>(Wo.data);
    float* dX_p         = static_cast<float*>(dX.data);
    float* dWq_p        = static_cast<float*>(dWq.data);
    float* dWk_p        = static_cast<float*>(dWk.data);
    float* dWv_p        = static_cast<float*>(dWv.data);
    float* dWo_p        = static_cast<float*>(dWo.data);

    // dY (N, D) and dWo accumulation.
    Tensor dY = Tensor::empty_on(Device::CUDA, N, D, Dtype::FP32);
    float* dY_p = static_cast<float*>(dY.data);
    {
        dim3 grid = grid2d(D, N, block_nd);
        wo_back_dY_kernel<<<grid, block_nd, 0, cur_stream()>>>(dO_p, Wo_p, d_mask,
                                              dY_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid = grid2d(D, D, block_dd);
        wo_back_dW_kernel<<<grid, block_dd, 0, cur_stream()>>>(dO_p, Y_p, d_mask,
                                              dWo_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dAttn (N, N), dV (N, D).
    Tensor dAttn = Tensor::empty_on(Device::CUDA, N, N, Dtype::FP32);
    Tensor dV    = Tensor::empty_on(Device::CUDA, N, D, Dtype::FP32);
    float* dAttn_p = static_cast<float*>(dAttn.data);
    float* dV_p    = static_cast<float*>(dV.data);
    {
        dim3 grid = grid2d(N, N, block_nn);
        dAttn_kernel<<<grid, block_nn, 0, cur_stream()>>>(dY_p, V_p, dAttn_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid = grid2d(D, N, block_nd);
        dV_kernel<<<grid, block_nd, 0, cur_stream()>>>(Attn_p, dY_p, dV_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dScores via per-row softmax backward, scaled by inv_sqrtd.
    Tensor dScores = Tensor::empty_on(Device::CUDA, N, N, Dtype::FP32);
    float* dScores_p = static_cast<float*>(dScores.data);
    row_softmax_back_kernel<<<N, ROW_SM_BLOCK, 0, cur_stream()>>>(Attn_p, dAttn_p, d_mask,
                                                 dScores_p, N, inv_sqrtd);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // dQ, dK.
    Tensor dQ = Tensor::empty_on(Device::CUDA, N, D, Dtype::FP32);
    Tensor dK = Tensor::empty_on(Device::CUDA, N, D, Dtype::FP32);
    float* dQ_p = static_cast<float*>(dQ.data);
    float* dK_p = static_cast<float*>(dK.data);
    {
        dim3 grid = grid2d(D, N, block_nd);
        dQ_kernel<<<grid, block_nd, 0, cur_stream()>>>(dScores_p, K_p, dQ_p, N, D);
        dK_kernel<<<grid, block_nd, 0, cur_stream()>>>(dScores_p, Q_p, dK_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dWq/dWk/dWv accumulation and dX.
    {
        dim3 grid = grid2d(D, D, block_dd);
        dWqkv_kernel<<<grid, block_dd, 0, cur_stream()>>>(dQ_p, dK_p, dV_p, X_p,
                                         dWq_p, dWk_p, dWv_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid = grid2d(D, N, block_nd);
        dX_proj_kernel<<<grid, block_nd, 0, cur_stream()>>>(dQ_p, dK_p, dV_p,
                                           Wq_p, Wk_p, Wv_p,
                                           dX_p, N, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace brotensor::detail::cuda
