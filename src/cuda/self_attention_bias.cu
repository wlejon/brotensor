// ─── CUDA self-attention with additive pre-softmax bias ────────────────────
//
// Multi-head self-attention that adds an optional per-head (L, L) bias to the
// attention logits before softmax — the general primitive behind T5's
// relative-position bias and ALiBi-style biases.
//
//   S[h,q,k] = scale * (Q_h[q] . K_h[k]) + attn_bias[h*L+q, k]
//   O        = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// Scores are materialised (L, L) per head — intended for encoder-length
// sequences (T5 ≤ 512), not long-context decoding. Dispatched on X.dtype
// (FP32 / FP16 / BF16): the projection inputs/outputs are typed, every
// intermediate (Q/K/V/scores/softmax) is FP32 scratch, math is FP32.
// attn_bias is FP32 on every backend.

#include <brotensor/tensor.h>

#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor {
namespace detail::cuda {

namespace {

constexpr int SAB_SM_BLOCK = 256;

__device__ inline float sab_ld(const float& x)         { return x; }
__device__ inline float sab_ld(const __half& x)        { return __half2float(x); }
__device__ inline float sab_ld(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ inline void  sab_st(float& d, float v)         { d = v; }
__device__ inline void  sab_st(__half& d, float v)        { d = __float2half(v); }
__device__ inline void  sab_st(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

// Per-head projection: Out[(hh*L+i), j] = sum_k In[i,k] * W[hh*dh+j, k].
// In: (L, Din) typed, W: (D, Din) typed, Out: (H*L, dh) FP32. grid.z = H.
template <typename T>
__global__ void sab_proj_kernel(const T* __restrict__ In,
                                const T* __restrict__ W,
                                float* __restrict__ Out,
                                int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const T* xr = In + static_cast<size_t>(i) * Din;
    const T* wr = W  + static_cast<size_t>(hh * dh + j) * Din;
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) acc += sab_ld(xr[k]) * sab_ld(wr[k]);
    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc;
}

// S[(hh*L+i), j] = scale * (Q_h[i] . K_h[j]) + bias[(hh*L+i), j].
// Qh, Kh: (H*L, dh) FP32; bias: (H*L, L) FP32 or null; S: (H*L, L) FP32.
__global__ void sab_scores_kernel(const float* __restrict__ Qh,
                                  const float* __restrict__ Kh,
                                  const float* __restrict__ bias,
                                  float* __restrict__ S,
                                  int L, int dh, float scale) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= L) return;
    const size_t qrow = (static_cast<size_t>(hh) * L + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * L + j) * dh;
    float s = 0.0f;
    for (int k = 0; k < dh; ++k) s += Qh[qrow + k] * Kh[krow + k];
    const size_t srow = (static_cast<size_t>(hh) * L + i) * L;
    s *= scale;
    if (bias) s += bias[srow + j];
    S[srow + j] = s;
}

// Per-row masked softmax over (H*L, L). One block per (head, query row).
// mask is length L (key validity); a padded query row produces a zero row.
__global__ void sab_softmax_kernel(const float* __restrict__ scores,
                                   float* __restrict__ Attn,
                                   const float* __restrict__ mask,
                                   int L) {
    __shared__ float sdata[SAB_SM_BLOCK];
    const int row = blockIdx.x;            // hh*L + i
    const int i_within = row % L;
    const int tid = threadIdx.x;
    const float* srow = scores + static_cast<size_t>(row) * L;
    float* arow = Attn + static_cast<size_t>(row) * L;

    if (mask && mask[i_within] < 0.5f) {
        for (int j = tid; j < L; j += blockDim.x) arow[j] = 0.0f;
        return;
    }

    float local_max = -1e30f;
    for (int j = tid; j < L; j += blockDim.x) {
        if (mask && mask[j] < 0.5f) continue;
        const float v = srow[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < L; j += blockDim.x) {
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
    for (int j = tid; j < L; j += blockDim.x) arow[j] *= inv;
}

// Yconcat[i, hh*dh+k] = sum_j Attn[(hh*L+i), j] * Vh[(hh*L+j), k].
__global__ void sab_apply_v_kernel(const float* __restrict__ Attn,
                                   const float* __restrict__ Vh,
                                   float* __restrict__ Yconcat,
                                   int L, int dh, int D) {
    const int k  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || k >= dh) return;
    const size_t arow = (static_cast<size_t>(hh) * L + i) * L;
    float acc = 0.0f;
    for (int j = 0; j < L; ++j) {
        const size_t vrow = (static_cast<size_t>(hh) * L + j) * dh;
        acc += Attn[arow + j] * Vh[vrow + k];
    }
    Yconcat[static_cast<size_t>(i) * D + (hh * dh + k)] = acc;
}

// O[i, c] = mask[i] ? sum_k Yconcat[i,k] * Wo[c,k] : 0.
template <typename T>
__global__ void sab_output_kernel(const float* __restrict__ Y,
                                  const T* __restrict__ Wo,
                                  const float* __restrict__ mask,
                                  T* __restrict__ O, int L, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= L || c >= D) return;
    if (mask && mask[i] < 0.5f) { sab_st(O[static_cast<size_t>(i) * D + c], 0.0f); return; }
    const float* yr = Y + static_cast<size_t>(i) * D;
    const T* wr = Wo + static_cast<size_t>(c) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * sab_ld(wr[k]);
    sab_st(O[static_cast<size_t>(i) * D + c], acc);
}

// Run the full pipeline for a concrete storage type T.
template <typename T>
void run_sab(const ::brotensor::Tensor& X,
             const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
             const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
             const float* d_mask, const float* bias_p,
             int num_heads, float scale, ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    using ::brotensor::Dtype;
    const int L  = X.rows;
    const int D  = X.cols;
    const int H  = num_heads;
    const int dh = D / H;

    Tensor Qh = Tensor::empty_on(Device::CUDA, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::CUDA, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::CUDA, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::CUDA, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::CUDA, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::CUDA, L, D, Dtype::FP32);

    const T* X_p  = static_cast<const T*>(X.data);
    const T* Wq_p = static_cast<const T*>(Wq.data);
    const T* Wk_p = static_cast<const T*>(Wk.data);
    const T* Wv_p = static_cast<const T*>(Wv.data);
    const T* Wo_p = static_cast<const T*>(Wo.data);
    float* Qh_p = static_cast<float*>(Qh.data);
    float* Kh_p = static_cast<float*>(Kh.data);
    float* Vh_p = static_cast<float*>(Vh.data);
    float* S_p  = static_cast<float*>(S.data);
    float* A_p  = static_cast<float*>(A.data);
    float* Yc_p = static_cast<float*>(Yc.data);

    const dim3 block(16, 16);
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (L  + block.y - 1) / block.y, H);
        sab_proj_kernel<T><<<grid, block>>>(X_p, Wq_p, Qh_p, L, D, dh);
        sab_proj_kernel<T><<<grid, block>>>(X_p, Wk_p, Kh_p, L, D, dh);
        sab_proj_kernel<T><<<grid, block>>>(X_p, Wv_p, Vh_p, L, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((L + block.x - 1) / block.x,
                  (L + block.y - 1) / block.y, H);
        sab_scores_kernel<<<grid, block>>>(Qh_p, Kh_p, bias_p, S_p, L, dh, scale);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    sab_softmax_kernel<<<H * L, SAB_SM_BLOCK>>>(S_p, A_p, d_mask, L);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    {
        dim3 grid((dh + block.x - 1) / block.x,
                  (L  + block.y - 1) / block.y, H);
        sab_apply_v_kernel<<<grid, block>>>(A_p, Vh_p, Yc_p, L, dh, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((D + block.x - 1) / block.x,
                  (L + block.y - 1) / block.y);
        sab_output_kernel<T><<<grid, block>>>(Yc_p, Wo_p, d_mask,
                                              static_cast<T*>(O.data), L, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq,
                                 const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv,
                                 const ::brotensor::Tensor& Wo,
                                 const float* d_mask,
                                 const ::brotensor::Tensor* attn_bias,
                                 int num_heads, float scale,
                                 ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("self_attention_bias_forward: X must be FP32, FP16, or BF16");
    }
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo dtype must match X");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("self_attention_bias_forward: num_heads must divide D");
    }
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo must be (D, D)");
    }
    const float* bias_p = nullptr;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be FP32");
        }
        if (attn_bias->size() != num_heads * L * L) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be (num_heads*L, L)");
        }
        bias_p = static_cast<const float*>(attn_bias->data);
    }
    if (O.rows != L || O.cols != D || O.dtype != X.dtype) {
        O.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    switch (X.dtype) {
    case Dtype::FP32:
        run_sab<float>(X, Wq, Wk, Wv, Wo, d_mask, bias_p, num_heads, scale, O);
        break;
    case Dtype::FP16:
        run_sab<__half>(X, Wq, Wk, Wv, Wo, d_mask, bias_p, num_heads, scale, O);
        break;
    default:  // BF16
        run_sab<__nv_bfloat16>(X, Wq, Wk, Wv, Wo, d_mask, bias_p, num_heads, scale, O);
        break;
    }
}

} // namespace detail::cuda
} // namespace brotensor
