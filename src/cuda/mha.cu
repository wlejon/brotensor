#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

namespace brotensor::detail::cuda {

// Naive correctness-first kernels mirroring src/nn/multi_head_attention.cpp.
// Heads are stored stacked along rows:
//   Qh / Kh / Vh   : (h * K, dh)   — head hh = rows [hh*K, (hh+1)*K)
//   Attnh          : (h * K, K)
//   Yconcat        : (K, D)        — head hh occupies cols [hh*dh, (hh+1)*dh)

namespace {

constexpr int ROW_SM_BLOCK = 256;

inline dim3 grid2d(int x, int y, dim3 block) {
    return dim3((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);
}

// Per-head Q/K/V projection: out(i, j) = sum_k X(i, k) * W(hh*dh + j, k)
//   X: (K, D)
//   W: (D, D)
//   Out: (h*K, dh) — writes head hh slice [hh*K, (hh+1)*K)
// One thread per (i, j) within head hh; grid.z = h.
__global__ void mha_proj_kernel(const float* __restrict__ X,
                                const float* __restrict__ W,
                                float* __restrict__ Out,
                                int K, int D, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= dh) return;
    const int row_off = hh * dh;
    const float* xr = X + static_cast<size_t>(i) * D;
    const float* wr = W + static_cast<size_t>(row_off + j) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
    const size_t out_row = static_cast<size_t>(hh) * K + i;
    Out[out_row * dh + j] = acc;
}

// Per-head scores: S(i, j) = (Q_h(i) . K_h(j)) / sqrt(dh)
//   Qh, Kh: (h*K, dh)  S: (h*K, K)
__global__ void mha_scores_kernel(const float* __restrict__ Qh,
                                  const float* __restrict__ Kh,
                                  float* __restrict__ S,
                                  int K, int dh, float inv_sqrtdh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= K) return;
    const size_t qrow = (static_cast<size_t>(hh) * K + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * K + j) * dh;
    float s = 0.0f;
    for (int k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    const size_t srow = (static_cast<size_t>(hh) * K + i) * K;
    S[srow + j] = s * inv_sqrtdh;
}

// Per-row masked softmax of (h*K, K). One block per (head, row).
// blockIdx.x = row index across heads (== hh*K + i).
__global__ void mha_row_softmax_kernel(const float* __restrict__ scores,
                                       float* __restrict__ Attn,
                                       const float* __restrict__ mask,
                                       int K) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int i_within = row % K;   // query index within head
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(row) * K;
    float* arow = Attn + static_cast<size_t>(row) * K;

    if (mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < K; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < K; j += blockDim.x) {
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

    float local_sum = 0.0f;
    for (int j = tid; j < K; j += blockDim.x) {
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

    for (int j = tid; j < K; j += blockDim.x) {
        arow[j] = arow[j] * inv;
    }
}

// Per-head Y_h(i, k) = sum_j Attnh(i, j) * Vh(j, k), then write into
// Yconcat(i, hh*dh + k).
__global__ void mha_attn_apply_v_kernel(const float* __restrict__ Attnh,
                                        const float* __restrict__ Vh,
                                        float* __restrict__ Yconcat,
                                        int K, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || k >= dh) return;
    const size_t arow = (static_cast<size_t>(hh) * K + i) * K;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const size_t vrow = (static_cast<size_t>(hh) * K + j) * dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
}

// O(i, c) = mask[i] ? sum_k Yconcat(i, k) * Wo(c, k) : 0.
__global__ void mha_output_proj_kernel(const float* __restrict__ Y,
                                       const float* __restrict__ Wo,
                                       const float* __restrict__ mask,
                                       float* __restrict__ O,
                                       int K, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || c >= D) return;
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

// ─── Backward kernels ─────────────────────────────────────────────────────

// dWo(c, k) += sum_i mask_i * dO(i, c) * Yconcat(i, k)   (one thread per (c, k))
__global__ void mha_wo_back_dW_kernel(const float* __restrict__ dO,
                                      const float* __restrict__ Y,
                                      const float* __restrict__ mask,
                                      float* __restrict__ dWo,
                                      int K, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        if (mask && mask[i] < 0.5f) continue;
        acc += dO[static_cast<size_t>(i) * D + c] * Y[static_cast<size_t>(i) * D + k];
    }
    dWo[static_cast<size_t>(c) * D + k] += acc;
}

// dYconcat(i, k) = mask_i ? sum_c Wo(c, k) * dO(i, c) : 0
__global__ void mha_wo_back_dY_kernel(const float* __restrict__ dO,
                                      const float* __restrict__ Wo,
                                      const float* __restrict__ mask,
                                      float* __restrict__ dY,
                                      int K, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || k >= D) return;
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

// Per-head dAttn(hh, i, j) = sum_k dYh(hh, i, k) * Vh(hh, j, k).
// dYh is the appropriate slice of dYconcat; we read it directly with col offset.
__global__ void mha_dAttn_kernel(const float* __restrict__ dYconcat,
                                 const float* __restrict__ Vh,
                                 float* __restrict__ dAttn,
                                 int K, int dh, int D) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= K) return;
    float acc = 0.0f;
    for (int k = 0; k < dh; ++k) {
        const float dy = dYconcat[static_cast<size_t>(i) * D + (hh * dh + k)];
        const float vv = Vh[(static_cast<size_t>(hh) * K + j) * dh + k];
        acc += dy * vv;
    }
    dAttn[(static_cast<size_t>(hh) * K + i) * K + j] = acc;
}

// dVh(hh, j, k) = sum_i Attnh(hh, i, j) * dYh(hh, i, k)
__global__ void mha_dV_kernel(const float* __restrict__ Attnh,
                              const float* __restrict__ dYconcat,
                              float* __restrict__ dVh,
                              int K, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (j >= K || k >= dh) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float a  = Attnh[(static_cast<size_t>(hh) * K + i) * K + j];
        const float dy = dYconcat[static_cast<size_t>(i) * D + (hh * dh + k)];
        acc += a * dy;
    }
    dVh[(static_cast<size_t>(hh) * K + j) * dh + k] = acc;
}

// Per-row softmax backward. blockIdx.x indexes rows (hh*K + i).
__global__ void mha_row_softmax_back_kernel(const float* __restrict__ Attn,
                                            const float* __restrict__ dAttn,
                                            const float* __restrict__ mask,
                                            float* __restrict__ dScores,
                                            int K, float inv_sqrtdh) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int i_within = row % K;
    const int tid = threadIdx.x;
    const float* prow  = Attn  + static_cast<size_t>(row) * K;
    const float* dprow = dAttn + static_cast<size_t>(row) * K;
    float* drow = dScores + static_cast<size_t>(row) * K;

    if (mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < K; j += blockDim.x) drow[j] = 0.0f;
        return;
    }

    float local = 0.0f;
    for (int j = tid; j < K; j += blockDim.x) {
        local += dprow[j] * prow[j];
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int j = tid; j < K; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
        }
    }
}

// dQh(hh, i, k) = sum_j dScores(hh, i, j) * Kh(hh, j, k)
__global__ void mha_dQ_kernel(const float* __restrict__ dScores,
                              const float* __restrict__ Kh,
                              float* __restrict__ dQh,
                              int K, int dh) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || k >= dh) return;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const float ds = dScores[(static_cast<size_t>(hh) * K + i) * K + j];
        const float kk = Kh[(static_cast<size_t>(hh) * K + j) * dh + k];
        acc += ds * kk;
    }
    dQh[(static_cast<size_t>(hh) * K + i) * dh + k] = acc;
}

// dKh(hh, j, k) = sum_i dScores(hh, i, j) * Qh(hh, i, k)
__global__ void mha_dK_kernel(const float* __restrict__ dScores,
                              const float* __restrict__ Qh,
                              float* __restrict__ dKh,
                              int K, int dh) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (j >= K || k >= dh) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float ds = dScores[(static_cast<size_t>(hh) * K + i) * K + j];
        const float qq = Qh[(static_cast<size_t>(hh) * K + i) * dh + k];
        acc += ds * qq;
    }
    dKh[(static_cast<size_t>(hh) * K + j) * dh + k] = acc;
}

// Final input-projection backward, mirroring CPU:
//   dWq(hh*dh + j, k) += sum_i dQh(hh, i, j) * X(i, k)        (and Wk, Wv)
// One thread per global (D, D) cell of the dWq/dWk/dWv accumulators.
__global__ void mha_dWqkv_kernel(const float* __restrict__ dQh,
                                 const float* __restrict__ dKh,
                                 const float* __restrict__ dVh,
                                 const float* __restrict__ X,
                                 float* __restrict__ dWq,
                                 float* __restrict__ dWk,
                                 float* __restrict__ dWv,
                                 int K, int D, int dh, int H) {
    const int k_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int wrow  = blockIdx.y * blockDim.y + threadIdx.y;
    if (wrow >= D || k_col >= D) return;
    const int hh = wrow / dh;
    const int j  = wrow % dh;
    (void)H;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float xv = X[static_cast<size_t>(i) * D + k_col];
        aq += dQh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
        ak += dKh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
        av += dVh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
    }
    const size_t idx = static_cast<size_t>(wrow) * D + k_col;
    dWq[idx] += aq;
    dWk[idx] += ak;
    dWv[idx] += av;
}

// dX(i, k) = sum over heads, j: dQh*Wq + dKh*Wk + dVh*Wv at (hh*dh+j, k).
__global__ void mha_dX_proj_kernel(const float* __restrict__ dQh,
                                   const float* __restrict__ dKh,
                                   const float* __restrict__ dVh,
                                   const float* __restrict__ Wq,
                                   const float* __restrict__ Wk,
                                   const float* __restrict__ Wv,
                                   float* __restrict__ dX,
                                   int K, int D, int dh, int H) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || k >= D) return;
    float acc = 0.0f;
    for (int hh = 0; hh < H; ++hh) {
        for (int j = 0; j < dh; ++j) {
            const int wrow = hh * dh + j;
            const size_t widx = static_cast<size_t>(wrow) * D + k;
            const float gq = dQh[(static_cast<size_t>(hh) * K + i) * dh + j];
            const float gk = dKh[(static_cast<size_t>(hh) * K + i) * dh + j];
            const float gv = dVh[(static_cast<size_t>(hh) * K + i) * dh + j];
            acc += gq * Wq[widx] + gk * Wk[widx] + gv * Wv[widx];
        }
    }
    dX[static_cast<size_t>(i) * D + k] = acc;
}

} // namespace

void mha_forward(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                 const float* d_mask,
                 int num_heads,
                 ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,
                 ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat,
                 ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = D / H;
    if (Qh.rows != H * K || Qh.cols != dh) Qh.resize(H * K, dh, Dtype::FP32);
    if (Kh.rows != H * K || Kh.cols != dh) Kh.resize(H * K, dh, Dtype::FP32);
    if (Vh.rows != H * K || Vh.cols != dh) Vh.resize(H * K, dh, Dtype::FP32);
    if (Attnh.rows != H * K || Attnh.cols != K) Attnh.resize(H * K, K, Dtype::FP32);
    if (Yconcat.rows != K || Yconcat.cols != D) Yconcat.resize(K, D, Dtype::FP32);
    if (O.rows != K || O.cols != D) O.resize(K, D, Dtype::FP32);
    if (K == 0 || D == 0 || H == 0) return;

    const dim3 block_kdh(16, 16);
    const dim3 block_kk(16, 16);
    const dim3 block_kd(16, 16);
    const dim3 block_dd(16, 16);

    const float* X_p   = static_cast<const float*>(X.data);
    const float* Wq_p  = static_cast<const float*>(Wq.data);
    const float* Wk_p  = static_cast<const float*>(Wk.data);
    const float* Wv_p  = static_cast<const float*>(Wv.data);
    const float* Wo_p  = static_cast<const float*>(Wo.data);
    float* Qh_p        = static_cast<float*>(Qh.data);
    float* Kh_p        = static_cast<float*>(Kh.data);
    float* Vh_p        = static_cast<float*>(Vh.data);
    float* Attnh_p     = static_cast<float*>(Attnh.data);
    float* Yconcat_p   = static_cast<float*>(Yconcat.data);
    float* O_p         = static_cast<float*>(O.data);

    // Q/K/V projections.
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_proj_kernel<<<grid, block_kdh>>>(X_p, Wq_p, Qh_p, K, D, dh);
        mha_proj_kernel<<<grid, block_kdh>>>(X_p, Wk_p, Kh_p, K, D, dh);
        mha_proj_kernel<<<grid, block_kdh>>>(X_p, Wv_p, Vh_p, K, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Scores → reuse Attnh storage temporarily for raw scores.
    Tensor scores = Tensor::empty_on(Device::CUDA, H * K, K, Dtype::FP32);
    float* scores_p = static_cast<float*>(scores.data);
    {
        const float inv_sqrtdh = 1.0f / sqrtf(static_cast<float>(dh));
        dim3 grid(((K) + block_kk.x - 1) / block_kk.x,
                  ((K) + block_kk.y - 1) / block_kk.y,
                  H);
        mha_scores_kernel<<<grid, block_kk>>>(Qh_p, Kh_p, scores_p,
                                              K, dh, inv_sqrtdh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Row-masked softmax across all H*K rows.
    mha_row_softmax_kernel<<<H * K, ROW_SM_BLOCK>>>(scores_p, Attnh_p,
                                                    d_mask, K);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // Attn @ V → Yconcat.
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_attn_apply_v_kernel<<<grid, block_kdh>>>(Attnh_p, Vh_p,
                                                     Yconcat_p, K, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // Output projection.
    {
        dim3 grid(((D) + block_kd.x - 1) / block_kd.x,
                  ((K) + block_kd.y - 1) / block_kd.y);
        mha_output_proj_kernel<<<grid, block_kd>>>(Yconcat_p, Wo_p, d_mask,
                                                   O_p, K, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    (void)block_dd;
    (void)grid2d;
}

void mha_backward(const ::brotensor::Tensor& dO,
                  const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                  const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                  const ::brotensor::Tensor& Yconcat,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                  const float* d_mask,
                  int num_heads,
                  ::brotensor::Tensor& dX,
                  ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                  ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = D / H;
    if (dX.rows != K || dX.cols != D) dX.resize(K, D, Dtype::FP32);
    if (K == 0 || D == 0 || H == 0) return;

    const float inv_sqrtdh = 1.0f / sqrtf(static_cast<float>(dh));
    const dim3 block_kd(16, 16);
    const dim3 block_kk(16, 16);
    const dim3 block_dd(16, 16);
    const dim3 block_kdh(16, 16);

    const float* dO_p     = static_cast<const float*>(dO.data);
    const float* X_p      = static_cast<const float*>(X.data);
    const float* Qh_p     = static_cast<const float*>(Qh.data);
    const float* Kh_p     = static_cast<const float*>(Kh.data);
    const float* Vh_p     = static_cast<const float*>(Vh.data);
    const float* Attnh_p  = static_cast<const float*>(Attnh.data);
    const float* Yconcat_p = static_cast<const float*>(Yconcat.data);
    const float* Wq_p     = static_cast<const float*>(Wq.data);
    const float* Wk_p     = static_cast<const float*>(Wk.data);
    const float* Wv_p     = static_cast<const float*>(Wv.data);
    const float* Wo_p     = static_cast<const float*>(Wo.data);
    float* dX_p           = static_cast<float*>(dX.data);
    float* dWq_p          = static_cast<float*>(dWq.data);
    float* dWk_p          = static_cast<float*>(dWk.data);
    float* dWv_p          = static_cast<float*>(dWv.data);
    float* dWo_p          = static_cast<float*>(dWo.data);

    // dYconcat (K, D) and dWo accumulation.
    Tensor dYconcat = Tensor::empty_on(Device::CUDA, K, D, Dtype::FP32);
    float* dYconcat_p = static_cast<float*>(dYconcat.data);
    {
        dim3 grid(((D) + block_kd.x - 1) / block_kd.x,
                  ((K) + block_kd.y - 1) / block_kd.y);
        mha_wo_back_dY_kernel<<<grid, block_kd>>>(dO_p, Wo_p, d_mask,
                                                  dYconcat_p, K, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(((D) + block_dd.x - 1) / block_dd.x,
                  ((D) + block_dd.y - 1) / block_dd.y);
        mha_wo_back_dW_kernel<<<grid, block_dd>>>(dO_p, Yconcat_p, d_mask,
                                                  dWo_p, K, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dAttn (H*K, K), dVh (H*K, dh).
    Tensor dAttn = Tensor::empty_on(Device::CUDA, H * K, K, Dtype::FP32);
    Tensor dVh   = Tensor::empty_on(Device::CUDA, H * K, dh, Dtype::FP32);
    float* dAttn_p = static_cast<float*>(dAttn.data);
    float* dVh_p   = static_cast<float*>(dVh.data);
    {
        dim3 grid(((K) + block_kk.x - 1) / block_kk.x,
                  ((K) + block_kk.y - 1) / block_kk.y,
                  H);
        mha_dAttn_kernel<<<grid, block_kk>>>(dYconcat_p, Vh_p, dAttn_p,
                                             K, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_dV_kernel<<<grid, block_kdh>>>(Attnh_p, dYconcat_p, dVh_p,
                                           K, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dScores (H*K, K) via per-row softmax backward.
    Tensor dScores = Tensor::empty_on(Device::CUDA, H * K, K, Dtype::FP32);
    float* dScores_p = static_cast<float*>(dScores.data);
    mha_row_softmax_back_kernel<<<H * K, ROW_SM_BLOCK>>>(Attnh_p, dAttn_p,
                                                         d_mask, dScores_p,
                                                         K, inv_sqrtdh);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // dQh, dKh.
    Tensor dQh = Tensor::empty_on(Device::CUDA, H * K, dh, Dtype::FP32);
    Tensor dKh = Tensor::empty_on(Device::CUDA, H * K, dh, Dtype::FP32);
    float* dQh_p = static_cast<float*>(dQh.data);
    float* dKh_p = static_cast<float*>(dKh.data);
    {
        dim3 grid(((dh) + block_kdh.x - 1) / block_kdh.x,
                  ((K)  + block_kdh.y - 1) / block_kdh.y,
                  H);
        mha_dQ_kernel<<<grid, block_kdh>>>(dScores_p, Kh_p, dQh_p,
                                           K, dh);
        mha_dK_kernel<<<grid, block_kdh>>>(dScores_p, Qh_p, dKh_p,
                                           K, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dWq/dWk/dWv accumulation and dX.
    {
        dim3 grid(((D) + block_dd.x - 1) / block_dd.x,
                  ((D) + block_dd.y - 1) / block_dd.y);
        mha_dWqkv_kernel<<<grid, block_dd>>>(dQh_p, dKh_p, dVh_p, X_p,
                                             dWq_p, dWk_p, dWv_p,
                                             K, D, dh, H);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(((D) + block_kd.x - 1) / block_kd.x,
                  ((K) + block_kd.y - 1) / block_kd.y);
        mha_dX_proj_kernel<<<grid, block_kd>>>(dQh_p, dKh_p, dVh_p,
                                               Wq_p, Wk_p, Wv_p,
                                               dX_p, K, D, dh, H);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ─── Forward declarations of sibling-file ops (for vtable fill) ────────────

void attention_forward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                       const float* d_mask,
                       ::brotensor::Tensor& Q, ::brotensor::Tensor& K, ::brotensor::Tensor& V,
                       ::brotensor::Tensor& Attn, ::brotensor::Tensor& Y_pre_Wo,
                       ::brotensor::Tensor& O);
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
                        ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo);
void cross_attention_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Ctx,
                             const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                             const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             ::brotensor::Tensor& O);
void cross_attention_forward_train(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& Ctx,
                                   const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                   const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                   const float* d_mask,
                                   int num_heads,
                                   ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,
                                   ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat,
                                   ::brotensor::Tensor& O);
void cross_attention_backward(const ::brotensor::Tensor& dO,
                              const ::brotensor::Tensor& X,
                              const ::brotensor::Tensor& Ctx,
                              const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                              const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                              const ::brotensor::Tensor& Yconcat,
                              const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                              const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                              const float* d_mask,
                              int num_heads,
                              ::brotensor::Tensor& dX,
                              ::brotensor::Tensor& dCtx,
                              ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                              ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo);
void cross_attention_forward_with_attn(const ::brotensor::Tensor& X,
                                       const ::brotensor::Tensor& Ctx,
                                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                       const float* d_mask,
                                       const ::brotensor::Tensor* attn_logit_bias,
                                       int num_heads,
                                       ::brotensor::Tensor& O,
                                       ::brotensor::Tensor& AttnAvg);
void self_attention_forward(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                            const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                            const float* d_mask, int num_heads,
                            ::brotensor::Tensor& O);
void self_attention_forward_train(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                  const float* d_mask, int num_heads,
                                  ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,
                                  ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat,
                                  ::brotensor::Tensor& O);
void self_attention_backward(const ::brotensor::Tensor& dO,
                             const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                             const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                             const ::brotensor::Tensor& Yconcat,
                             const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                             const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                             const float* d_mask, int num_heads,
                             ::brotensor::Tensor& dX,
                             ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                             ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo);
void attention_token_moments(const ::brotensor::Tensor& Attn,
                             int h_lat, int w_lat,
                             ::brotensor::Tensor& mass,
                             ::brotensor::Tensor& centroid);
// Defined in self_attention_bias.cu.
void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq,
                                 const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv,
                                 const ::brotensor::Tensor& Wo,
                                 const float* d_mask,
                                 const ::brotensor::Tensor* attn_bias,
                                 int num_heads, float scale,
                                 ::brotensor::Tensor& O);

void fill_cuda_vtable_attention(::brotensor::detail::OpsVTable& v) {
    v.mha_forward                            = &mha_forward;
    v.mha_backward                           = &mha_backward;
    v.attention_forward                      = &attention_forward;
    v.attention_backward                     = &attention_backward;
    v.cross_attention_forward                = &cross_attention_forward;
    v.cross_attention_forward_train          = &cross_attention_forward_train;
    v.cross_attention_backward               = &cross_attention_backward;
    v.cross_attention_forward_with_attn      = &cross_attention_forward_with_attn;
    v.self_attention_forward                 = &self_attention_forward;
    v.self_attention_bias_forward            = &self_attention_bias_forward;
    v.self_attention_forward_train           = &self_attention_forward_train;
    v.self_attention_backward                = &self_attention_backward;
    v.attention_token_moments                = &attention_token_moments;
}

} // namespace brotensor::detail::cuda
