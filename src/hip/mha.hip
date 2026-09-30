// Native AMD HIP Multi-Head Attention (MHA) forward and backward for RDNA 3.5 (Wave32).
//
// Computes multi-head self/cross attention:
//   Qh = X @ Wq^T (+ bq)
//   Kh = X @ Wk^T (+ bk)
//   Vh = X @ Wv^T (+ bv)
//   Scores = (Qh @ Kh^T) / sqrt(dh)
//   Attnh = row_softmax(Scores, mask)
//   Yconcat = concat(Attnh @ Vh)
//   Output = Yconcat @ Wo^T (+ bo)
//
// Includes full backward pass for training. Wave32 reductions in softmax.

#include <brotensor/detail/dispatch.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include "detail/hip_check.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brotensor::detail::hip {

void* hip_current_stream(int dev = -1);

namespace {

constexpr int kWave32 = 32;
constexpr int kRowSmBlock = 256;

inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

__device__ __forceinline__ float wave32_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, kWave32));
    }
    return val;
}

__device__ __forceinline__ float wave32_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, kWave32);
    }
    return val;
}

inline dim3 grid2d(int x, int y, dim3 block) {
    return dim3((x + block.x - 1) / block.x, (y + block.y - 1) / block.y);
}

// ─── Forward Kernels ─────────────────────────────────────────────────────────

// Per-head projection: Out(i, j) = sum_k X(i, k) * W(hh*dh + j, k)
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

__global__ void mha_proj_bias_add_kernel(float* __restrict__ Out,
                                         const float* __restrict__ bias,
                                         int K, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= K || j >= dh) return;
    const size_t row = static_cast<size_t>(hh) * K + i;
    Out[row * dh + j] += bias[hh * dh + j];
}

// Per-head scores: S(i, j) = (Q_h(i) . K_h(j)) / sqrt(dh)
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

// Per-row masked softmax with Wave32 reductions. One block per (head, row).
__global__ void mha_row_softmax_kernel(const float* __restrict__ scores,
                                       float* __restrict__ Attn,
                                       const float* __restrict__ mask,
                                       int K) {
    __shared__ float s_wave[8];
    const int row = blockIdx.x;
    const int i_within = row % K;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

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
        local_max = fmaxf(local_max, v);
    }
    local_max = wave32_reduce_max(local_max);
    if (lane == 0) s_wave[warp_id] = local_max;
    __syncthreads();

    float block_max = -1e30f;
    if (tid < num_warps) block_max = s_wave[tid];
    block_max = wave32_reduce_max(block_max);
    if (warp_id == 0) s_wave[lane] = block_max;
    __syncthreads();
    const float m = s_wave[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < K; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            arow[j] = 0.0f;
            continue;
        }
        const float e = __expf(srow[j] - m);
        arow[j] = e;
        local_sum += e;
    }
    local_sum = wave32_reduce_sum(local_sum);
    if (lane == 0) s_wave[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < num_warps) block_sum = s_wave[tid];
    block_sum = wave32_reduce_sum(block_sum);
    if (warp_id == 0) s_wave[lane] = block_sum;
    __syncthreads();
    const float sum = s_wave[0];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int j = tid; j < K; j += blockDim.x) {
        arow[j] = arow[j] * inv;
    }
}

// Per-head Y_h(i, k) = sum_j Attnh(i, j) * Vh(j, k) -> Yconcat(i, hh*dh + k)
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

// O(i, c) = mask[i] ? sum_k Yconcat(i, k) * Wo(c, k) + (bo ? bo[c] : 0) : 0
__global__ void mha_output_proj_kernel(const float* __restrict__ Y,
                                       const float* __restrict__ Wo,
                                       const float* __restrict__ bo,
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
    if (bo) acc += bo[c];
    O[static_cast<size_t>(i) * D + c] = acc;
}

// ─── Backward Kernels ────────────────────────────────────────────────────────

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

__global__ void mha_wo_back_dbo_kernel(const float* __restrict__ dO,
                                       const float* __restrict__ mask,
                                       float* __restrict__ dbo,
                                       int K, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        if (mask && mask[i] < 0.5f) continue;
        acc += dO[static_cast<size_t>(i) * D + c];
    }
    dbo[c] += acc;
}

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

__global__ void mha_row_softmax_back_kernel(const float* __restrict__ Attn,
                                            const float* __restrict__ dAttn,
                                            const float* __restrict__ mask,
                                            float* __restrict__ dScores,
                                            int K, float inv_sqrtdh) {
    __shared__ float s_wave[8];
    const int row = blockIdx.x;
    const int i_within = row % K;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

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
    local = wave32_reduce_sum(local);
    if (lane == 0) s_wave[warp_id] = local;
    __syncthreads();

    float block_dot = 0.0f;
    if (tid < num_warps) block_dot = s_wave[tid];
    block_dot = wave32_reduce_sum(block_dot);
    if (warp_id == 0) s_wave[lane] = block_dot;
    __syncthreads();
    const float dot = s_wave[0];

    for (int j = tid; j < K; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) {
            drow[j] = 0.0f;
        } else {
            drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
        }
    }
}

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

__global__ void mha_dWqkv_kernel(const float* __restrict__ dQh,
                                 const float* __restrict__ dKh,
                                 const float* __restrict__ dVh,
                                 const float* __restrict__ X,
                                 float* __restrict__ dWq,
                                 float* __restrict__ dWk,
                                 float* __restrict__ dWv,
                                 int K, int D, int dh) {
    const int k_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int wrow  = blockIdx.y * blockDim.y + threadIdx.y;
    if (wrow >= D || k_col >= D) return;
    const int hh = wrow / dh;
    const int j  = wrow % dh;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (int i = 0; i < K; ++i) {
        const float xv = X[static_cast<size_t>(i) * D + k_col];
        aq += dQh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
        ak += dKh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
        av += dVh[(static_cast<size_t>(hh) * K + i) * dh + j] * xv;
    }
    const size_t off = static_cast<size_t>(wrow) * D + k_col;
    dWq[off] += aq;
    dWk[off] += ak;
    dWv[off] += av;
}

__global__ void mha_dbqkv_kernel(const float* __restrict__ dGh,
                                 float* __restrict__ db,
                                 int K, int D, int dh) {
    const int wrow = blockIdx.x * blockDim.x + threadIdx.x;
    if (wrow >= D) return;
    const int hh = wrow / dh;
    const int j  = wrow % dh;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        acc += dGh[(static_cast<size_t>(hh) * K + i) * dh + j];
    }
    db[wrow] += acc;
}

__global__ void mha_dX_kernel(const float* __restrict__ dQh,
                              const float* __restrict__ dKh,
                              const float* __restrict__ dVh,
                              const float* __restrict__ Wq,
                              const float* __restrict__ Wk,
                              const float* __restrict__ Wv,
                              float* __restrict__ dX,
                              int K, int D, int dh, int H) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= K || c >= D) return;
    float acc = 0.0f;
    for (int hh = 0; hh < H; ++hh) {
        for (int j = 0; j < dh; ++j) {
            const int wrow = hh * dh + j;
            const float dq = dQh[(static_cast<size_t>(hh) * K + i) * dh + j];
            const float dk = dKh[(static_cast<size_t>(hh) * K + i) * dh + j];
            const float dv = dVh[(static_cast<size_t>(hh) * K + i) * dh + j];
            acc += dq * Wq[static_cast<size_t>(wrow) * D + c];
            acc += dk * Wk[static_cast<size_t>(wrow) * D + c];
            acc += dv * Wv[static_cast<size_t>(wrow) * D + c];
        }
    }
    dX[static_cast<size_t>(i) * D + c] = acc;
}

} // namespace

// ─── Public MHA Forward ──────────────────────────────────────────────────────

void hip_mha_forward(const ::brotensor::Tensor& X,
                     const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                     const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                     const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                     const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                     const float* d_mask,
                     int num_heads,
                     ::brotensor::Tensor& Qh,
                     ::brotensor::Tensor& Kh,
                     ::brotensor::Tensor& Vh,
                     ::brotensor::Tensor& Attnh,
                     ::brotensor::Tensor& Yconcat,
                     ::brotensor::Tensor& Output) {
    const int K = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("hip_mha_forward: num_heads must divide D");
    }
    const int dh = D / num_heads;
    const int H = num_heads;

    Qh.resize(H * K, dh);
    Kh.resize(H * K, dh);
    Vh.resize(H * K, dh);
    Attnh.resize(H * K, K);
    Yconcat.resize(K, D);
    Output.resize(K, D);

    hipStream_t stream = cur_stream();
    dim3 b16(16, 16);

    // 1. Projections
    dim3 g_proj = grid2d(dh, K, b16);
    g_proj.z = H;
    mha_proj_kernel<<<g_proj, b16, 0, stream>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(Wq.data),
        static_cast<float*>(Qh.data), K, D, dh);
    mha_proj_kernel<<<g_proj, b16, 0, stream>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(Wk.data),
        static_cast<float*>(Kh.data), K, D, dh);
    mha_proj_kernel<<<g_proj, b16, 0, stream>>>(
        static_cast<const float*>(X.data), static_cast<const float*>(Wv.data),
        static_cast<float*>(Vh.data), K, D, dh);

    if (bq && bq->size() > 0)
        mha_proj_bias_add_kernel<<<g_proj, b16, 0, stream>>>(
            static_cast<float*>(Qh.data), static_cast<const float*>(bq->data), K, dh);
    if (bk && bk->size() > 0)
        mha_proj_bias_add_kernel<<<g_proj, b16, 0, stream>>>(
            static_cast<float*>(Kh.data), static_cast<const float*>(bk->data), K, dh);
    if (bv && bv->size() > 0)
        mha_proj_bias_add_kernel<<<g_proj, b16, 0, stream>>>(
            static_cast<float*>(Vh.data), static_cast<const float*>(bv->data), K, dh);

    // 2. Scores
    ::brotensor::Tensor S = ::brotensor::Tensor::empty_on(X.device, H * K, K, X.dtype);
    dim3 g_scores = grid2d(K, K, b16);
    g_scores.z = H;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
    mha_scores_kernel<<<g_scores, b16, 0, stream>>>(
        static_cast<const float*>(Qh.data), static_cast<const float*>(Kh.data),
        static_cast<float*>(S.data), K, dh, inv_sqrtdh);

    // 3. Row softmax
    mha_row_softmax_kernel<<<H * K, kRowSmBlock, 0, stream>>>(
        static_cast<const float*>(S.data), static_cast<float*>(Attnh.data), d_mask, K);

    // 4. Attnh @ Vh -> Yconcat
    dim3 g_apply = grid2d(dh, K, b16);
    g_apply.z = H;
    mha_attn_apply_v_kernel<<<g_apply, b16, 0, stream>>>(
        static_cast<const float*>(Attnh.data), static_cast<const float*>(Vh.data),
        static_cast<float*>(Yconcat.data), K, dh, D);

    // 5. Output projection
    dim3 g_out = grid2d(D, K, b16);
    const float* bo_ptr = (bo && bo->size() > 0) ? static_cast<const float*>(bo->data) : nullptr;
    mha_output_proj_kernel<<<g_out, b16, 0, stream>>>(
        static_cast<const float*>(Yconcat.data), static_cast<const float*>(Wo.data),
        bo_ptr, d_mask, static_cast<float*>(Output.data), K, D);

    BROTENSOR_HIP_CHECK(hipGetLastError());
}

// ─── Public MHA Backward ─────────────────────────────────────────────────────

void hip_mha_backward(const ::brotensor::Tensor& dOutput,
                      const ::brotensor::Tensor& X,
                      const ::brotensor::Tensor& Qh,
                      const ::brotensor::Tensor& Kh,
                      const ::brotensor::Tensor& Vh,
                      const ::brotensor::Tensor& Attnh,
                      const ::brotensor::Tensor& Yconcat,
                      const ::brotensor::Tensor& Wq,
                      const ::brotensor::Tensor& Wk,
                      const ::brotensor::Tensor& Wv,
                      const ::brotensor::Tensor& Wo,
                      const float* d_mask,
                      int num_heads,
                      ::brotensor::Tensor& dX,
                      ::brotensor::Tensor& dWq,
                      ::brotensor::Tensor& dWk,
                      ::brotensor::Tensor& dWv,
                      ::brotensor::Tensor& dWo,
                      ::brotensor::Tensor* dbq,
                      ::brotensor::Tensor* dbk,
                      ::brotensor::Tensor* dbv,
                      ::brotensor::Tensor* dbo) {
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = D / H;

    hipStream_t stream = cur_stream();
    dim3 b16(16, 16);

    // 1. Output projection backward
    dim3 g_DD = grid2d(D, D, b16);
    mha_wo_back_dW_kernel<<<g_DD, b16, 0, stream>>>(
        static_cast<const float*>(dOutput.data), static_cast<const float*>(Yconcat.data),
        d_mask, static_cast<float*>(dWo.data), K, D);

    if (dbo && dbo->size() > 0) {
        mha_wo_back_dbo_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(dOutput.data), d_mask,
            static_cast<float*>(dbo->data), K, D);
    }

    ::brotensor::Tensor dYconcat = ::brotensor::Tensor::empty_on(X.device, K, D, X.dtype);
    dim3 g_KD = grid2d(D, K, b16);
    mha_wo_back_dY_kernel<<<g_KD, b16, 0, stream>>>(
        static_cast<const float*>(dOutput.data), static_cast<const float*>(Wo.data),
        d_mask, static_cast<float*>(dYconcat.data), K, D);

    // 2. Attention backward: dAttn and dV
    ::brotensor::Tensor dAttnh = ::brotensor::Tensor::empty_on(X.device, H * K, K, X.dtype);
    ::brotensor::Tensor dVh = ::brotensor::Tensor::empty_on(X.device, H * K, dh, X.dtype);

    dim3 g_KK = grid2d(K, K, b16); g_KK.z = H;
    mha_dAttn_kernel<<<g_KK, b16, 0, stream>>>(
        static_cast<const float*>(dYconcat.data), static_cast<const float*>(Vh.data),
        static_cast<float*>(dAttnh.data), K, dh, D);

    dim3 g_dhK = grid2d(dh, K, b16); g_dhK.z = H;
    mha_dV_kernel<<<g_dhK, b16, 0, stream>>>(
        static_cast<const float*>(Attnh.data), static_cast<const float*>(dYconcat.data),
        static_cast<float*>(dVh.data), K, dh, D);

    // 3. Softmax backward -> dScores
    ::brotensor::Tensor dScores = ::brotensor::Tensor::empty_on(X.device, H * K, K, X.dtype);
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
    mha_row_softmax_back_kernel<<<H * K, kRowSmBlock, 0, stream>>>(
        static_cast<const float*>(Attnh.data), static_cast<const float*>(dAttnh.data),
        d_mask, static_cast<float*>(dScores.data), K, inv_sqrtdh);

    // 4. dQ and dK
    ::brotensor::Tensor dQh = ::brotensor::Tensor::empty_on(X.device, H * K, dh, X.dtype);
    ::brotensor::Tensor dKh = ::brotensor::Tensor::empty_on(X.device, H * K, dh, X.dtype);

    dim3 g_Q = grid2d(dh, K, b16); g_Q.z = H;
    mha_dQ_kernel<<<g_Q, b16, 0, stream>>>(
        static_cast<const float*>(dScores.data), static_cast<const float*>(Kh.data),
        static_cast<float*>(dQh.data), K, dh);
    mha_dK_kernel<<<g_Q, b16, 0, stream>>>(
        static_cast<const float*>(dScores.data), static_cast<const float*>(Qh.data),
        static_cast<float*>(dKh.data), K, dh);

    // 5. Input projections backward: dWq, dWk, dWv, dbq, dbk, dbv, dX
    mha_dWqkv_kernel<<<g_DD, b16, 0, stream>>>(
        static_cast<const float*>(dQh.data), static_cast<const float*>(dKh.data),
        static_cast<const float*>(dVh.data), static_cast<const float*>(X.data),
        static_cast<float*>(dWq.data), static_cast<float*>(dWk.data), static_cast<float*>(dWv.data),
        K, D, dh);

    if (dbq && dbq->size() > 0)
        mha_dbqkv_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(dQh.data), static_cast<float*>(dbq->data), K, D, dh);
    if (dbk && dbk->size() > 0)
        mha_dbqkv_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(dKh.data), static_cast<float*>(dbk->data), K, D, dh);
    if (dbv && dbv->size() > 0)
        mha_dbqkv_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(dVh.data), static_cast<float*>(dbv->data), K, D, dh);

    dX.resize(K, D);
    mha_dX_kernel<<<g_KD, b16, 0, stream>>>(
        static_cast<const float*>(dQh.data), static_cast<const float*>(dKh.data),
        static_cast<const float*>(dVh.data), static_cast<const float*>(Wq.data),
        static_cast<const float*>(Wk.data), static_cast<const float*>(Wv.data),
        static_cast<float*>(dX.data), K, D, dh, H);

    BROTENSOR_HIP_CHECK(hipGetLastError());
}

void fill_hip_vtable_mha(::brotensor::detail::OpsVTable& v) {
    v.mha_forward  = &hip_mha_forward;
    v.mha_backward = &hip_mha_backward;
}

} // namespace brotensor::detail::hip
