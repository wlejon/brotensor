#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace brotensor::detail::cuda {

// FP16 inference path (unchanged from prior bundle): cross_attention and
// self_attention public forwards delegate to the flash-attention kernel. A
// previous hand-rolled core kernel produced incorrect outputs at large
// block counts on this architecture; the flash kernel's tiled
// online-softmax path is numerically robust at every shape exercised by
// the U-Net / cross-attn pipeline.
//
// FP32 training path (this bundle): the *_train functions below mirror
// the mha math but accept a separate Ctx tensor for K/V projection and
// rectangular Wk/Wv: (D, D_ctx). Self-attention training is a thin wrapper
// over mha_forward / mha_backward (Lq == Lk, D_ctx == D, X == Ctx).

// ─── Sibling-cluster forward decls (Phase 2A / same cluster) ──────────────
void mha_forward(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                 const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                 const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                 const float* d_mask,
                 int num_heads,
                 ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,
                 ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat,
                 ::brotensor::Tensor& O);
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
                  ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo,
                  ::brotensor::Tensor* dbq, ::brotensor::Tensor* dbk,
                  ::brotensor::Tensor* dbv, ::brotensor::Tensor* dbo);
void flash_attention_qkvo_forward(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor* Ctx,
                                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
                                  const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
                                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
                                  const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
                                  const float* d_mask, int num_heads, bool causal,
                                  ::brotensor::Tensor& O);

namespace {

constexpr int ROW_SM_BLOCK = 256;

// ─── Forward kernels ──────────────────────────────────────────────────────

// Per-head projection: out(i, j) = sum_k In(i, k) * W(hh*dh + j, k)
// In:  (L, Din), W: (D, Din), Out: (h*L, dh).
// One thread per (i, j) within head hh; grid.z = h.
__global__ void cx_proj_kernel(const float* __restrict__ In,
                               const float* __restrict__ W,
                               float* __restrict__ Out,
                               int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int row_off = hh * dh;
    const float* xr = In + static_cast<size_t>(i) * Din;
    const float* wr = W  + static_cast<size_t>(row_off + j) * Din;
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) acc += xr[k] * wr[k];
    const size_t out_row = static_cast<size_t>(hh) * L + i;
    Out[out_row * dh + j] = acc;
}

// Per-head scores S(i, j) = (Q_h(i) . K_h(j)) / sqrt(dh)
//   Qh: (h*Lq, dh), Kh: (h*Lk, dh), S: (h*Lq, Lk)
__global__ void cx_scores_kernel(const float* __restrict__ Qh,
                                 const float* __restrict__ Kh,
                                 float* __restrict__ S,
                                 int Lq, int Lk, int dh, float inv_sqrtdh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= Lq || j >= Lk) return;
    const size_t qrow = (static_cast<size_t>(hh) * Lq + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * Lk + j) * dh;
    float s = 0.0f;
    for (int k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    const size_t srow = (static_cast<size_t>(hh) * Lq + i) * Lk;
    S[srow + j] = s * inv_sqrtdh;
}

// Per-row masked softmax over (h*Lq, Lk). One block per (head, row).
// mask is length Lk (key validity); query-side validity for masking is N/A
// here — the mha kernel uses the SAME mask buffer for both, but in
// cross-attention there is no per-query mask. To preserve the mha convention
// (mask zeros out invalid query rows) we ALSO interpret mask[i_within_query]
// in the self-attention degenerate case; cross-attn callers always pass
// a length-Lk mask. To remain consistent with mha and the existing
// inference path, we apply the mask in two places: keys (excluded from
// softmax denom) AND queries (only when Lq == Lk and the caller intends
// self-attn). We disable the query-side gate when Lq != Lk.
__global__ void cx_row_softmax_kernel(const float* __restrict__ scores,
                                      float* __restrict__ Attn,
                                      const float* __restrict__ mask,
                                      int Lq, int Lk, int gate_query) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int i_within = row % Lq;
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(row) * Lk;
    float* arow = Attn + static_cast<size_t>(row) * Lk;

    if (gate_query && mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < Lk; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < Lk; j += blockDim.x) {
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
    for (int j = tid; j < Lk; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
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

    for (int j = tid; j < Lk; j += blockDim.x) arow[j] = arow[j] * inv;
}

// Per-head Y_h(i, k) = sum_j Attnh(i, j) * Vh(j, k); writes Yconcat(i, hh*dh+k).
__global__ void cx_attn_apply_v_kernel(const float* __restrict__ Attnh,
                                       const float* __restrict__ Vh,
                                       float* __restrict__ Yconcat,
                                       int Lq, int Lk, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= Lq || k >= dh) return;
    const size_t arow = (static_cast<size_t>(hh) * Lq + i) * Lk;
    float acc = 0.0f;
    for (int j = 0; j < Lk; ++j) {
        const size_t vrow = (static_cast<size_t>(hh) * Lk + j) * dh;
        acc += Attnh[arow + j] * Vh[vrow + k];
    }
    Yconcat[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
}

// O(i, c) = mask_q[i] ? sum_k Yconcat(i, k) * Wo(c, k) : 0.
// Query-side mask gating only when gate_query is true.
__global__ void cx_output_proj_kernel(const float* __restrict__ Y,
                                      const float* __restrict__ Wo,
                                      const float* __restrict__ mask,
                                      int gate_query,
                                      float* __restrict__ O,
                                      int Lq, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Lq || c >= D) return;
    if (gate_query && mask && mask[i] < 0.5f) {
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

// dWo(c, k) += sum_i mask_q[i] * dO(i, c) * Yconcat(i, k)
__global__ void cx_wo_back_dW_kernel(const float* __restrict__ dO,
                                     const float* __restrict__ Y,
                                     const float* __restrict__ mask,
                                     int gate_query,
                                     float* __restrict__ dWo,
                                     int Lq, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= D || k >= D) return;
    float acc = 0.0f;
    for (int i = 0; i < Lq; ++i) {
        if (gate_query && mask && mask[i] < 0.5f) continue;
        acc += dO[static_cast<size_t>(i) * D + c] * Y[static_cast<size_t>(i) * D + k];
    }
    dWo[static_cast<size_t>(c) * D + k] += acc;
}

// dYconcat(i, k) = mask_q[i] ? sum_c Wo(c, k) * dO(i, c) : 0
__global__ void cx_wo_back_dY_kernel(const float* __restrict__ dO,
                                     const float* __restrict__ Wo,
                                     const float* __restrict__ mask,
                                     int gate_query,
                                     float* __restrict__ dY,
                                     int Lq, int D) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Lq || k >= D) return;
    if (gate_query && mask && mask[i] < 0.5f) {
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
__global__ void cx_dAttn_kernel(const float* __restrict__ dYconcat,
                                const float* __restrict__ Vh,
                                float* __restrict__ dAttn,
                                int Lq, int Lk, int dh, int D) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= Lq || j >= Lk) return;
    float acc = 0.0f;
    for (int k = 0; k < dh; ++k) {
        const float dy = dYconcat[static_cast<size_t>(i) * D + (hh * dh + k)];
        const float vv = Vh[(static_cast<size_t>(hh) * Lk + j) * dh + k];
        acc += dy * vv;
    }
    dAttn[(static_cast<size_t>(hh) * Lq + i) * Lk + j] = acc;
}

// dVh(hh, j, k) = sum_i Attnh(hh, i, j) * dYh(hh, i, k)
__global__ void cx_dV_kernel(const float* __restrict__ Attnh,
                             const float* __restrict__ dYconcat,
                             float* __restrict__ dVh,
                             int Lq, int Lk, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (j >= Lk || k >= dh) return;
    float acc = 0.0f;
    for (int i = 0; i < Lq; ++i) {
        const float a  = Attnh[(static_cast<size_t>(hh) * Lq + i) * Lk + j];
        const float dy = dYconcat[static_cast<size_t>(i) * D + (hh * dh + k)];
        acc += a * dy;
    }
    dVh[(static_cast<size_t>(hh) * Lk + j) * dh + k] = acc;
}

// Per-row softmax backward over (h*Lq, Lk).
__global__ void cx_row_softmax_back_kernel(const float* __restrict__ Attn,
                                           const float* __restrict__ dAttn,
                                           const float* __restrict__ mask,
                                           int gate_query,
                                           float* __restrict__ dScores,
                                           int Lq, int Lk, float inv_sqrtdh) {
    __shared__ float sdata[ROW_SM_BLOCK];
    const int row = blockIdx.x;
    const int i_within = row % Lq;
    const int tid = threadIdx.x;
    const float* prow  = Attn  + static_cast<size_t>(row) * Lk;
    const float* dprow = dAttn + static_cast<size_t>(row) * Lk;
    float* drow = dScores + static_cast<size_t>(row) * Lk;

    if (gate_query && mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < Lk; j += blockDim.x) drow[j] = 0.0f;
        return;
    }

    float local = 0.0f;
    for (int j = tid; j < Lk; j += blockDim.x) local += dprow[j] * prow[j];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float dot = sdata[0];

    for (int j = tid; j < Lk; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) drow[j] = 0.0f;
        else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
    }
}

// dQh(hh, i, k) = sum_j dScores(hh, i, j) * Kh(hh, j, k)
__global__ void cx_dQ_kernel(const float* __restrict__ dScores,
                             const float* __restrict__ Kh,
                             float* __restrict__ dQh,
                             int Lq, int Lk, int dh) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= Lq || k >= dh) return;
    float acc = 0.0f;
    for (int j = 0; j < Lk; ++j) {
        const float ds = dScores[(static_cast<size_t>(hh) * Lq + i) * Lk + j];
        const float kk = Kh[(static_cast<size_t>(hh) * Lk + j) * dh + k];
        acc += ds * kk;
    }
    dQh[(static_cast<size_t>(hh) * Lq + i) * dh + k] = acc;
}

// dKh(hh, j, k) = sum_i dScores(hh, i, j) * Qh(hh, i, k)
__global__ void cx_dK_kernel(const float* __restrict__ dScores,
                             const float* __restrict__ Qh,
                             float* __restrict__ dKh,
                             int Lq, int Lk, int dh) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int j  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (j >= Lk || k >= dh) return;
    float acc = 0.0f;
    for (int i = 0; i < Lq; ++i) {
        const float ds = dScores[(static_cast<size_t>(hh) * Lq + i) * Lk + j];
        const float qq = Qh[(static_cast<size_t>(hh) * Lq + i) * dh + k];
        acc += ds * qq;
    }
    dKh[(static_cast<size_t>(hh) * Lk + j) * dh + k] = acc;
}

// Per-W accumulator. Generic: dW(hh*dh + j, k) += sum_i dHh(hh, i, j) * In(i, k).
// L is the row count of `In` and of dHh (per head); Din is cols of `In` and W.
// One thread per (wrow, k_col) covering (D, Din).
__global__ void cx_dW_proj_kernel(const float* __restrict__ dHh,
                                  const float* __restrict__ In,
                                  float* __restrict__ dW,
                                  int L, int D, int Din, int dh) {
    const int k_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int wrow  = blockIdx.y * blockDim.y + threadIdx.y;
    if (wrow >= D || k_col >= Din) return;
    const int hh = wrow / dh;
    const int j  = wrow % dh;
    float acc = 0.0f;
    for (int i = 0; i < L; ++i) {
        const float xv = In[static_cast<size_t>(i) * Din + k_col];
        acc += dHh[(static_cast<size_t>(hh) * L + i) * dh + j] * xv;
    }
    dW[static_cast<size_t>(wrow) * Din + k_col] += acc;
}

// dX(i, k) = sum over heads, j: dQh(hh, i, j) * Wq(hh*dh + j, k).
// X-side: only Q feeds back into X. D == Din here for Wq.
__global__ void cx_dX_kernel(const float* __restrict__ dQh,
                             const float* __restrict__ Wq,
                             float* __restrict__ dX,
                             int Lq, int D, int dh, int H) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Lq || k >= D) return;
    float acc = 0.0f;
    for (int hh = 0; hh < H; ++hh) {
        for (int j = 0; j < dh; ++j) {
            const int wrow = hh * dh + j;
            const float gq = dQh[(static_cast<size_t>(hh) * Lq + i) * dh + j];
            acc += gq * Wq[static_cast<size_t>(wrow) * D + k];
        }
    }
    dX[static_cast<size_t>(i) * D + k] = acc;
}

// dCtx(j, k) = sum over heads, m: dKh*Wk + dVh*Wv at (hh*dh + m, k).
// Ctx-side: K and V both feed back. Din == D_ctx for Wk/Wv.
__global__ void cx_dCtx_kernel(const float* __restrict__ dKh,
                               const float* __restrict__ dVh,
                               const float* __restrict__ Wk,
                               const float* __restrict__ Wv,
                               float* __restrict__ dCtx,
                               int Lk, int D, int Dctx, int dh, int H) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= Lk || k >= Dctx) return;
    float acc = 0.0f;
    for (int hh = 0; hh < H; ++hh) {
        for (int m = 0; m < dh; ++m) {
            const int wrow = hh * dh + m;
            const size_t widx = static_cast<size_t>(wrow) * Dctx + k;
            const float gk = dKh[(static_cast<size_t>(hh) * Lk + j) * dh + m];
            const float gv = dVh[(static_cast<size_t>(hh) * Lk + j) * dh + m];
            acc += gk * Wk[widx] + gv * Wv[widx];
        }
    }
    dCtx[static_cast<size_t>(j) * Dctx + k] = acc;
}

inline void check_fp32(const ::brotensor::Tensor& t, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("cross_attention training path requires FP32 ") + name);
    }
}

void cross_attention_forward_train_core(const ::brotensor::Tensor& X,
                                        const ::brotensor::Tensor& Ctx,
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
    check_fp32(X, "X");
    check_fp32(Ctx, "Ctx");
    check_fp32(Wq, "Wq"); check_fp32(Wk, "Wk");
    check_fp32(Wv, "Wv"); check_fp32(Wo, "Wo");

    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;
    if (Qh.rows != H * Lq || Qh.cols != dh) Qh.resize(H * Lq, dh, Dtype::FP32);
    if (Kh.rows != H * Lk || Kh.cols != dh) Kh.resize(H * Lk, dh, Dtype::FP32);
    if (Vh.rows != H * Lk || Vh.cols != dh) Vh.resize(H * Lk, dh, Dtype::FP32);
    if (Attnh.rows != H * Lq || Attnh.cols != Lk) Attnh.resize(H * Lq, Lk, Dtype::FP32);
    if (Yconcat.rows != Lq || Yconcat.cols != D) Yconcat.resize(Lq, D, Dtype::FP32);
    if (O.rows != Lq || O.cols != D) O.resize(Lq, D, Dtype::FP32);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const int gate_query = (Lq == Lk) ? 1 : 0;
    const dim3 block(16, 16);

    const float* X_p   = static_cast<const float*>(X.data);
    const float* Ctx_p = static_cast<const float*>(Ctx.data);
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

    // Q projection from X (D, D).
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cx_proj_kernel<<<grid, block>>>(X_p, Wq_p, Qh_p, Lq, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    // K and V projections from Ctx (D, D_ctx).
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lk + block.y - 1) / block.y, H);
        cx_proj_kernel<<<grid, block>>>(Ctx_p, Wk_p, Kh_p, Lk, Dctx, dh);
        cx_proj_kernel<<<grid, block>>>(Ctx_p, Wv_p, Vh_p, Lk, Dctx, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    Tensor scores = Tensor::empty_on(Device::CUDA, H * Lq, Lk, Dtype::FP32);
    float* scores_p = static_cast<float*>(scores.data);
    {
        const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
        dim3 grid((Lk + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cx_scores_kernel<<<grid, block>>>(Qh_p, Kh_p, scores_p,
                                          Lq, Lk, dh, inv_sqrtdh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    cx_row_softmax_kernel<<<H * Lq, ROW_SM_BLOCK>>>(scores_p, Attnh_p,
                                                    d_mask, Lq, Lk, gate_query);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cx_attn_apply_v_kernel<<<grid, block>>>(Attnh_p, Vh_p,
                                                Yconcat_p, Lq, Lk, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 grid((D  + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y);
        cx_output_proj_kernel<<<grid, block>>>(Yconcat_p, Wo_p, d_mask,
                                               gate_query, O_p, Lq, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

// ─── FP32 self-attention training wrappers ────────────────────────────────

void self_attention_forward_train(const ::brotensor::Tensor& X,
                                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                  const float* d_mask,
                                  int num_heads,
                                  ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,
                                  ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat,
                                  ::brotensor::Tensor& O) {
    mha_forward(X, Wq, Wk, Wv, Wo,
                nullptr, nullptr, nullptr, nullptr,
                d_mask, num_heads,
                Qh, Kh, Vh, Attnh, Yconcat, O);
}

void self_attention_backward(const ::brotensor::Tensor& dO,
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
    mha_backward(dO, X, Qh, Kh, Vh, Attnh, Yconcat,
                 Wq, Wk, Wv, Wo, d_mask, num_heads,
                 dX, dWq, dWk, dWv, dWo,
                 nullptr, nullptr, nullptr, nullptr);
}

// ─── FP32 cross-attention training ────────────────────────────────────────

void cross_attention_forward_train(const ::brotensor::Tensor& X,
                                   const ::brotensor::Tensor& Ctx,
                                   const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                                   const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                                   const float* d_mask,
                                   int num_heads,
                                   ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh, ::brotensor::Tensor& Vh,
                                   ::brotensor::Tensor& Attnh, ::brotensor::Tensor& Yconcat,
                                   ::brotensor::Tensor& O) {
    cross_attention_forward_train_core(X, Ctx, Wq, Wk, Wv, Wo, d_mask,
                                       num_heads, Qh, Kh, Vh, Attnh,
                                       Yconcat, O);
}

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
                              ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    check_fp32(dO, "dO"); check_fp32(X, "X"); check_fp32(Ctx, "Ctx");

    const int Lq   = X.rows;
    const int D    = X.cols;
    const int Lk   = Ctx.rows;
    const int Dctx = Ctx.cols;
    const int H    = num_heads;
    const int dh   = (H > 0) ? D / H : 0;
    if (dX.rows != Lq || dX.cols != D) dX.resize(Lq, D, Dtype::FP32);
    if (dCtx.rows != Lk || dCtx.cols != Dctx) dCtx.resize(Lk, Dctx, Dtype::FP32);
    if (Lq == 0 || Lk == 0 || D == 0 || H == 0) return;

    const int gate_query = (Lq == Lk) ? 1 : 0;
    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));
    const dim3 block(16, 16);

    const float* dO_p      = static_cast<const float*>(dO.data);
    const float* X_p       = static_cast<const float*>(X.data);
    const float* Ctx_p     = static_cast<const float*>(Ctx.data);
    const float* Qh_p      = static_cast<const float*>(Qh.data);
    const float* Kh_p      = static_cast<const float*>(Kh.data);
    const float* Vh_p      = static_cast<const float*>(Vh.data);
    const float* Attnh_p   = static_cast<const float*>(Attnh.data);
    const float* Yconcat_p = static_cast<const float*>(Yconcat.data);
    const float* Wq_p      = static_cast<const float*>(Wq.data);
    const float* Wk_p      = static_cast<const float*>(Wk.data);
    const float* Wv_p      = static_cast<const float*>(Wv.data);
    const float* Wo_p      = static_cast<const float*>(Wo.data);
    float* dX_p            = static_cast<float*>(dX.data);
    float* dCtx_p          = static_cast<float*>(dCtx.data);
    float* dWq_p           = static_cast<float*>(dWq.data);
    float* dWk_p           = static_cast<float*>(dWk.data);
    float* dWv_p           = static_cast<float*>(dWv.data);
    float* dWo_p           = static_cast<float*>(dWo.data);

    // dYconcat = dO @ Wo (gated by query mask). Accumulate dWo.
    Tensor dYconcat = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP32);
    float* dYconcat_p = static_cast<float*>(dYconcat.data);
    {
        dim3 grid((D  + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y);
        cx_wo_back_dY_kernel<<<grid, block>>>(dO_p, Wo_p, d_mask,
                                              gate_query, dYconcat_p, Lq, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((D + block.x - 1) / block.x,
                  (D + block.y - 1) / block.y);
        cx_wo_back_dW_kernel<<<grid, block>>>(dO_p, Yconcat_p, d_mask,
                                              gate_query, dWo_p, Lq, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dAttn (H*Lq, Lk), dVh (H*Lk, dh).
    Tensor dAttn = Tensor::empty_on(Device::CUDA, H * Lq, Lk, Dtype::FP32);
    Tensor dVh   = Tensor::empty_on(Device::CUDA, H * Lk, dh, Dtype::FP32);
    float* dAttn_p = static_cast<float*>(dAttn.data);
    float* dVh_p   = static_cast<float*>(dVh.data);
    {
        dim3 grid((Lk + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cx_dAttn_kernel<<<grid, block>>>(dYconcat_p, Vh_p, dAttn_p,
                                         Lq, Lk, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lk + block.y - 1) / block.y, H);
        cx_dV_kernel<<<grid, block>>>(Attnh_p, dYconcat_p, dVh_p,
                                      Lq, Lk, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dScores via row softmax backward.
    Tensor dScores = Tensor::empty_on(Device::CUDA, H * Lq, Lk, Dtype::FP32);
    float* dScores_p = static_cast<float*>(dScores.data);
    cx_row_softmax_back_kernel<<<H * Lq, ROW_SM_BLOCK>>>(
            Attnh_p, dAttn_p, d_mask, gate_query,
            dScores_p, Lq, Lk, inv_sqrtdh);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // dQh, dKh.
    Tensor dQh = Tensor::empty_on(Device::CUDA, H * Lq, dh, Dtype::FP32);
    Tensor dKh = Tensor::empty_on(Device::CUDA, H * Lk, dh, Dtype::FP32);
    float* dQh_p = static_cast<float*>(dQh.data);
    float* dKh_p = static_cast<float*>(dKh.data);
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y, H);
        cx_dQ_kernel<<<grid, block>>>(dScores_p, Kh_p, dQh_p,
                                      Lq, Lk, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (Lk + block.y - 1) / block.y, H);
        cx_dK_kernel<<<grid, block>>>(dScores_p, Qh_p, dKh_p,
                                      Lq, Lk, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dWq (D, D) accumulates against X.
    {
        dim3 grid((D + block.x - 1) / block.x,
                  (D + block.y - 1) / block.y);
        cx_dW_proj_kernel<<<grid, block>>>(dQh_p, X_p, dWq_p,
                                           Lq, D, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    // dWk, dWv (D, D_ctx) accumulate against Ctx.
    {
        dim3 grid((Dctx + block.x - 1) / block.x,
                  (D    + block.y - 1) / block.y);
        cx_dW_proj_kernel<<<grid, block>>>(dKh_p, Ctx_p, dWk_p,
                                           Lk, D, Dctx, dh);
        cx_dW_proj_kernel<<<grid, block>>>(dVh_p, Ctx_p, dWv_p,
                                           Lk, D, Dctx, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }

    // dX and dCtx.
    {
        dim3 grid((D  + block.x - 1) / block.x,
                  (Lq + block.y - 1) / block.y);
        cx_dX_kernel<<<grid, block>>>(dQh_p, Wq_p, dX_p,
                                      Lq, D, dh, H);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((Dctx + block.x - 1) / block.x,
                  (Lk   + block.y - 1) / block.y);
        cx_dCtx_kernel<<<grid, block>>>(dKh_p, dVh_p, Wk_p, Wv_p,
                                        dCtx_p, Lk, D, Dctx, dh, H);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ─── Public forward dispatch ──────────────────────────────────────────────

void cross_attention_forward(const ::brotensor::Tensor& X,
                             const ::brotensor::Tensor& Ctx,
                             const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                             const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                             const float* d_mask,
                             int num_heads,
                             ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Dtype;
    if (X.dtype == Dtype::FP16) {
        if (Ctx.dtype != Dtype::FP16) {
            throw std::runtime_error("cross_attention_forward: Ctx dtype must match X dtype");
        }
        flash_attention_qkvo_forward(X, &Ctx,
                                     Wq, nullptr, Wk, nullptr,
                                     Wv, nullptr, Wo, nullptr,
                                     d_mask, num_heads, /*causal=*/false, O);
        return;
    }
    if (X.dtype == Dtype::BF16) {
        // BF16 inference: delegate to the flash path exactly as FP16 does.
        if (Ctx.dtype != Dtype::BF16) {
            throw std::runtime_error("cross_attention_forward: Ctx dtype must match X dtype");
        }
        flash_attention_qkvo_forward(X, &Ctx,
                                     Wq, nullptr, Wk, nullptr,
                                     Wv, nullptr, Wo, nullptr,
                                     d_mask, num_heads, /*causal=*/false, O);
        return;
    }
    if (Ctx.dtype != Dtype::FP32) {
        throw std::runtime_error("cross_attention_forward: Ctx dtype must match X dtype");
    }
    Tensor Qh      = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Kh      = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Vh      = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Attnh   = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Yconcat = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    cross_attention_forward_train_core(X, Ctx, Wq, Wk, Wv, Wo, d_mask,
                                       num_heads, Qh, Kh, Vh, Attnh,
                                       Yconcat, O);
}

void self_attention_forward(const ::brotensor::Tensor& X,
                            const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                            const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                            const float* d_mask,
                            int num_heads,
                            ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Dtype;
    if (X.dtype == Dtype::FP16 || X.dtype == Dtype::BF16) {
        // FP16/BF16 inference: delegate to the flash path (self-attention =
        // Ctx == nullptr). BF16 routes through the same call as FP16.
        flash_attention_qkvo_forward(X, nullptr,
                                     Wq, nullptr, Wk, nullptr,
                                     Wv, nullptr, Wo, nullptr,
                                     d_mask, num_heads, /*causal=*/false, O);
        return;
    }
    Tensor Qh      = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Kh      = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Vh      = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Attnh   = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    Tensor Yconcat = Tensor::empty_on(::brotensor::Device::CUDA, 0, 0, Dtype::FP32);
    mha_forward(X, Wq, Wk, Wv, Wo,
                nullptr, nullptr, nullptr, nullptr,
                d_mask, num_heads,
                Qh, Kh, Vh, Attnh, Yconcat, O);
}

} // namespace brotensor::detail::cuda
