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
#include <cstdint>
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

// Per-head projection: Out[(hh*L+i), j] = (bias?bias[hh*dh+j]:0) +
//                                          sum_k In[i,k] * W[hh*dh+j, k].
// In: (L, Din) typed, W: (D, Din) typed, bias: (D,1) typed or null,
// Out: (H*L, dh) FP32. grid.z = H.
template <typename T>
__global__ void sab_proj_kernel(const T* __restrict__ In,
                                const T* __restrict__ W,
                                const T* __restrict__ bias,  // (D,1) or null
                                float* __restrict__ Out,
                                int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int o = hh * dh + j;
    const T* xr = In + static_cast<size_t>(i) * Din;
    const T* wr = W  + static_cast<size_t>(o) * Din;
    float acc = bias ? sab_ld(bias[o]) : 0.0f;
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

// O[i, c] = mask[i] ? (bias?bias[c]:0) + sum_k Yconcat[i,k] * Wo[c,k] : 0.
template <typename T>
__global__ void sab_output_kernel(const float* __restrict__ Y,
                                  const T* __restrict__ Wo,
                                  const T* __restrict__ bias,  // (D,1) or null
                                  const float* __restrict__ mask,
                                  T* __restrict__ O, int L, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= L || c >= D) return;
    if (mask && mask[i] < 0.5f) { sab_st(O[static_cast<size_t>(i) * D + c], 0.0f); return; }
    const float* yr = Y + static_cast<size_t>(i) * D;
    const T* wr = Wo + static_cast<size_t>(c) * D;
    float acc = bias ? sab_ld(bias[c]) : 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * sab_ld(wr[k]);
    sab_st(O[static_cast<size_t>(i) * D + c], acc);
}

// Run the full pipeline for a concrete storage type T.
template <typename T>
void run_sab(const ::brotensor::Tensor& X,
             const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
             const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
             const T* bq, const T* bk, const T* bv, const T* bo,
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
        sab_proj_kernel<T><<<grid, block>>>(X_p, Wq_p, bq, Qh_p, L, D, dh);
        sab_proj_kernel<T><<<grid, block>>>(X_p, Wk_p, bk, Kh_p, L, D, dh);
        sab_proj_kernel<T><<<grid, block>>>(X_p, Wv_p, bv, Vh_p, L, D, dh);
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
        sab_output_kernel<T><<<grid, block>>>(Yc_p, Wo_p, bo, d_mask,
                                              static_cast<T*>(O.data), L, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ── INT8 weight-only (W8A16) projection / output kernels ───────────────────
//
// Same dot products as sab_proj_kernel / sab_output_kernel, but the weight is
// an INT8 (D, Din) matrix paired with an FP32 per-output-row dequant scale.
// The row `wrow` accumulates against int8 weights, then the whole sum is
// multiplied by scales[wrow] — equivalent to dequantising the row first,
// since one scale covers the entire row.
template <typename T>
__global__ void sab_proj_kernel_int8(const T* __restrict__ In,
                                     const int8_t* __restrict__ W,
                                     const float* __restrict__ scales,
                                     float* __restrict__ Out,
                                     int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int wrow = hh * dh + j;
    const T* xr = In + static_cast<size_t>(i) * Din;
    const int8_t* wr = W + static_cast<size_t>(wrow) * Din;
    float acc = 0.0f;
    for (int k = 0; k < Din; ++k) acc += sab_ld(xr[k]) * static_cast<float>(wr[k]);
    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc * scales[wrow];
}

template <typename T>
__global__ void sab_output_kernel_int8(const float* __restrict__ Y,
                                       const int8_t* __restrict__ Wo,
                                       const float* __restrict__ scales,
                                       const float* __restrict__ mask,
                                       T* __restrict__ O, int L, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= L || c >= D) return;
    if (mask && mask[i] < 0.5f) { sab_st(O[static_cast<size_t>(i) * D + c], 0.0f); return; }
    const float* yr = Y + static_cast<size_t>(i) * D;
    const int8_t* wr = Wo + static_cast<size_t>(c) * D;
    float acc = 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * static_cast<float>(wr[k]);
    sab_st(O[static_cast<size_t>(i) * D + c], acc * scales[c]);
}

// INT8-weight variant of run_sab: the four projection matmuls consume INT8
// weights + FP32 per-row scales; the attention core (scores / softmax / PV)
// is byte-identical to the FP16 path.
template <typename T>
void run_sab_int8(const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& sq,
                  const ::brotensor::Tensor& Wk, const ::brotensor::Tensor& sk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& sv,
                  const ::brotensor::Tensor& Wo, const ::brotensor::Tensor& so,
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
    const int8_t* Wq_p = static_cast<const int8_t*>(Wq.data);
    const int8_t* Wk_p = static_cast<const int8_t*>(Wk.data);
    const int8_t* Wv_p = static_cast<const int8_t*>(Wv.data);
    const int8_t* Wo_p = static_cast<const int8_t*>(Wo.data);
    const float* sq_p = static_cast<const float*>(sq.data);
    const float* sk_p = static_cast<const float*>(sk.data);
    const float* sv_p = static_cast<const float*>(sv.data);
    const float* so_p = static_cast<const float*>(so.data);
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
        sab_proj_kernel_int8<T><<<grid, block>>>(X_p, Wq_p, sq_p, Qh_p, L, D, dh);
        sab_proj_kernel_int8<T><<<grid, block>>>(X_p, Wk_p, sk_p, Kh_p, L, D, dh);
        sab_proj_kernel_int8<T><<<grid, block>>>(X_p, Wv_p, sv_p, Vh_p, L, D, dh);
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
        sab_output_kernel_int8<T><<<grid, block>>>(Yc_p, Wo_p, so_p, d_mask,
                                                   static_cast<T*>(O.data), L, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ── Decomposed 2D relative-position attention (SAM / ViTDet) ───────────────
//
// Same pipeline as run_sab, but (a) the qkv/output projections carry optional
// biases and (b) the pre-softmax bias is the decomposed rel-pos term computed
// from Q inline in the scores kernel — never materialised as a separate buffer.
// A token i maps to grid coords (i/grid_w, i%grid_w); the bias for key j is
//   q . rel_pos_h[(qh-kh)+grid_h-1] + q . rel_pos_w[(qw-kw)+grid_w-1].
// Like run_sab the (H*L, L) scores/softmax scratch is materialised, so the
// global 64x64 SAM blocks (L=4096) are memory-heavy — inherent to stock SAM,
// and identical to the sibling self_attention_bias path. A tiled/flash variant
// that avoids the (H*L, L) scratch is a later optimisation.

// Out[(hh*L+i), j] = (bias ? bias[hh*dh+j] : 0) + sum_k In[i,k]*W[(hh*dh+j), k].
template <typename T>
__global__ void sardp_proj_kernel(const T* __restrict__ In,
                                  const T* __restrict__ W,
                                  const T* __restrict__ bias,  // (D,1) or null
                                  float* __restrict__ Out,
                                  int L, int Din, int dh) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= dh) return;
    const int o = hh * dh + j;
    const T* xr = In + static_cast<size_t>(i) * Din;
    const T* wr = W  + static_cast<size_t>(o) * Din;
    float acc = bias ? sab_ld(bias[o]) : 0.0f;
    for (int k = 0; k < Din; ++k) acc += sab_ld(xr[k]) * sab_ld(wr[k]);
    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc;
}

// S[(hh*L+i), j] = scale*(Q_h[i].K_h[j]) + q.rel_pos_h[..] + q.rel_pos_w[..].
// rel_h: (2*gh-1, dh) typed; rel_w: (2*gw-1, dh) typed (model dtype T).
template <typename T>
__global__ void sardp_scores_kernel(const float* __restrict__ Qh,
                                    const float* __restrict__ Kh,
                                    const T* __restrict__ rel_h,
                                    const T* __restrict__ rel_w,
                                    float* __restrict__ S,
                                    int L, int dh, float scale,
                                    int gh, int gw) {
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int hh = blockIdx.z;
    if (i >= L || j >= L) return;
    const size_t qrow = (static_cast<size_t>(hh) * L + i) * dh;
    const size_t krow = (static_cast<size_t>(hh) * L + j) * dh;
    const int qh = i / gw, qw = i % gw;
    const int kh = j / gw, kw = j % gw;
    const T* rhr = rel_h + static_cast<size_t>(qh - kh + gh - 1) * dh;
    const T* rwr = rel_w + static_cast<size_t>(qw - kw + gw - 1) * dh;
    float s = 0.0f, bh = 0.0f, bw = 0.0f;
    for (int k = 0; k < dh; ++k) {
        const float q = Qh[qrow + k];
        s  += q * Kh[krow + k];
        bh += q * sab_ld(rhr[k]);
        bw += q * sab_ld(rwr[k]);
    }
    S[(static_cast<size_t>(hh) * L + i) * L + j] = s * scale + bh + bw;
}

// O[i, c] = (bias ? bias[c] : 0) + sum_k Yconcat[i,k] * Wo[c,k].
template <typename T>
__global__ void sardp_output_kernel(const float* __restrict__ Y,
                                    const T* __restrict__ Wo,
                                    const T* __restrict__ bias,  // (D,1) or null
                                    T* __restrict__ O, int L, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= L || c >= D) return;
    const float* yr = Y + static_cast<size_t>(i) * D;
    const T* wr = Wo + static_cast<size_t>(c) * D;
    float acc = bias ? sab_ld(bias[c]) : 0.0f;
    for (int k = 0; k < D; ++k) acc += yr[k] * sab_ld(wr[k]);
    sab_st(O[static_cast<size_t>(i) * D + c], acc);
}

template <typename T>
void run_sardp(const ::brotensor::Tensor& X,
               const ::brotensor::Tensor& Wq, const T* bq,
               const ::brotensor::Tensor& Wk, const T* bk,
               const ::brotensor::Tensor& Wv, const T* bv,
               const ::brotensor::Tensor& Wo, const T* bo,
               const ::brotensor::Tensor& rel_h, const ::brotensor::Tensor& rel_w,
               int num_heads, int gh, int gw, float scale,
               ::brotensor::Tensor& O) {
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
    const T* rh_p = static_cast<const T*>(rel_h.data);
    const T* rw_p = static_cast<const T*>(rel_w.data);
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
        sardp_proj_kernel<T><<<grid, block>>>(X_p, Wq_p, bq, Qh_p, L, D, dh);
        sardp_proj_kernel<T><<<grid, block>>>(X_p, Wk_p, bk, Kh_p, L, D, dh);
        sardp_proj_kernel<T><<<grid, block>>>(X_p, Wv_p, bv, Vh_p, L, D, dh);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid((L + block.x - 1) / block.x,
                  (L + block.y - 1) / block.y, H);
        sardp_scores_kernel<T><<<grid, block>>>(Qh_p, Kh_p, rh_p, rw_p, S_p,
                                                L, dh, scale, gh, gw);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
    sab_softmax_kernel<<<H * L, SAB_SM_BLOCK>>>(S_p, A_p, /*mask=*/nullptr, L);
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
        sardp_output_kernel<T><<<grid, block>>>(Yc_p, Wo_p, bo,
                                                static_cast<T*>(O.data), L, D);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    }
}

// ─── Windowed decomposed-rel-pos attention (SAM windowed encoder block) ──────
//
// Gathers the (grid_h, grid_w) token grid into a contiguous per-window batch
// (zero-padding the bottom/right up to a multiple of `window`), runs run_sardp
// once per window over a view of that window's rows, then scatters the result
// back — dropping the padded tokens. Pure layout work around the existing
// single-grid kernel.
//
// Partition row r = w_idx*window*window + lh*window + lw maps to grid token
// (h, w) = (nh*window+lh, nw*window+lw) with (nh, nw) = (w_idx/nw_w, w_idx%nw_w).

template <typename T>
__global__ void win_gather_kernel(const T* __restrict__ X, T* __restrict__ P,
                                  int grid_h, int grid_w, int window,
                                  int nw_w, int D, int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= D || row >= nrows) return;
    const int ww  = window * window;
    const int loc = row % ww;
    const int wi  = row / ww;
    const int h   = (wi / nw_w) * window + loc / window;
    const int w   = (wi % nw_w) * window + loc % window;
    T* dst = P + static_cast<size_t>(row) * D + col;
    if (h < grid_h && w < grid_w)
        *dst = X[static_cast<size_t>(h * grid_w + w) * D + col];
    else
        sab_st(*dst, 0.0f);
}

template <typename T>
__global__ void win_scatter_kernel(const T* __restrict__ P, T* __restrict__ O,
                                   int grid_h, int grid_w, int window,
                                   int nw_w, int D, int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= D || row >= nrows) return;
    const int ww  = window * window;
    const int loc = row % ww;
    const int wi  = row / ww;
    const int h   = (wi / nw_w) * window + loc / window;
    const int w   = (wi % nw_w) * window + loc % window;
    if (h < grid_h && w < grid_w)
        O[static_cast<size_t>(h * grid_w + w) * D + col] =
            P[static_cast<size_t>(row) * D + col];
}

template <typename T>
void run_windowed_sardp(const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& Wq, const T* bq,
                        const ::brotensor::Tensor& Wk, const T* bk,
                        const ::brotensor::Tensor& Wv, const T* bv,
                        const ::brotensor::Tensor& Wo, const T* bo,
                        const ::brotensor::Tensor& rel_h,
                        const ::brotensor::Tensor& rel_w,
                        int num_heads, int grid_h, int grid_w, int window,
                        float scale, ::brotensor::Tensor& O) {
    using ::brotensor::Tensor;
    using ::brotensor::Device;
    const int D     = X.cols;
    const auto dt   = X.dtype;
    const int pad_h = (window - grid_h % window) % window;
    const int pad_w = (window - grid_w % window) % window;
    const int nw_h  = (grid_h + pad_h) / window;
    const int nw_w  = (grid_w + pad_w) / window;
    const int nW    = nw_h * nw_w;
    const int ww    = window * window;
    const int nrows = nW * ww;

    Tensor Pin  = Tensor::empty_on(Device::CUDA, nrows, D, dt);
    Tensor Pout = Tensor::empty_on(Device::CUDA, nrows, D, dt);

    const dim3 block(16, 16);
    const dim3 grid((D + block.x - 1) / block.x,
                    (nrows + block.y - 1) / block.y);
    win_gather_kernel<T><<<grid, block>>>(static_cast<const T*>(X.data),
                                          static_cast<T*>(Pin.data),
                                          grid_h, grid_w, window, nw_w, D, nrows);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // Per-window attention over contiguous row-slices of the partition buffers.
    // run_sardp writes O.data without resizing, so non-owning views are safe.
    const size_t row_stride = static_cast<size_t>(ww) * D * sizeof(T);
    for (int w = 0; w < nW; ++w) {
        char* in_p  = static_cast<char*>(Pin.data)  + static_cast<size_t>(w) * row_stride;
        char* out_p = static_cast<char*>(Pout.data) + static_cast<size_t>(w) * row_stride;
        Tensor Xv = Tensor::view(Device::CUDA, in_p,  ww, D, dt);
        Tensor Ov = Tensor::view(Device::CUDA, out_p, ww, D, dt);
        run_sardp<T>(Xv, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                     rel_h, rel_w, num_heads, window, window, scale, Ov);
    }

    win_scatter_kernel<T><<<grid, block>>>(static_cast<const T*>(Pout.data),
                                           static_cast<T*>(O.data),
                                           grid_h, grid_w, window, nw_w, D, nrows);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void self_attention_bias_forward(const ::brotensor::Tensor& X,
                                 const ::brotensor::Tensor& Wq,
                                 const ::brotensor::Tensor& Wk,
                                 const ::brotensor::Tensor& Wv,
                                 const ::brotensor::Tensor& Wo,
                                 const ::brotensor::Tensor* bq,
                                 const ::brotensor::Tensor* bk,
                                 const ::brotensor::Tensor* bv,
                                 const ::brotensor::Tensor* bo,
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
    auto check_proj_bias = [&](const ::brotensor::Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string("self_attention_bias_forward: ") +
                                         name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string("self_attention_bias_forward: ") +
                                         name + " must have D entries");
        }
    };
    check_proj_bias(bq, "bq"); check_proj_bias(bk, "bk");
    check_proj_bias(bv, "bv"); check_proj_bias(bo, "bo");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype) {
        O.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    auto bp = [](const ::brotensor::Tensor* b) {
        return (b && b->data) ? b->data : nullptr;
    };
    switch (X.dtype) {
    case Dtype::FP32:
        run_sab<float>(X, Wq, Wk, Wv, Wo,
                       static_cast<const float*>(bp(bq)), static_cast<const float*>(bp(bk)),
                       static_cast<const float*>(bp(bv)), static_cast<const float*>(bp(bo)),
                       d_mask, bias_p, num_heads, scale, O);
        break;
    case Dtype::FP16:
        run_sab<__half>(X, Wq, Wk, Wv, Wo,
                        static_cast<const __half*>(bp(bq)), static_cast<const __half*>(bp(bk)),
                        static_cast<const __half*>(bp(bv)), static_cast<const __half*>(bp(bo)),
                        d_mask, bias_p, num_heads, scale, O);
        break;
    default:  // BF16
        run_sab<__nv_bfloat16>(X, Wq, Wk, Wv, Wo,
                               static_cast<const __nv_bfloat16*>(bp(bq)), static_cast<const __nv_bfloat16*>(bp(bk)),
                               static_cast<const __nv_bfloat16*>(bp(bv)), static_cast<const __nv_bfloat16*>(bp(bo)),
                               d_mask, bias_p, num_heads, scale, O);
        break;
    }
}

void self_attention_bias_int8w_fp16(const ::brotensor::Tensor& X,
                                    const ::brotensor::Tensor& Wq_int8,
                                    const ::brotensor::Tensor& sq,
                                    const ::brotensor::Tensor& Wk_int8,
                                    const ::brotensor::Tensor& sk,
                                    const ::brotensor::Tensor& Wv_int8,
                                    const ::brotensor::Tensor& sv,
                                    const ::brotensor::Tensor& Wo_int8,
                                    const ::brotensor::Tensor& so,
                                    const float* d_mask,
                                    const ::brotensor::Tensor* attn_bias,
                                    int num_heads, float scale,
                                    ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("self_attention_bias_int8w_fp16: X must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be INT8");
    }
    if (sq.dtype != Dtype::FP32 || sk.dtype != Dtype::FP32 ||
        sv.dtype != Dtype::FP32 || so.dtype != Dtype::FP32) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: scales must be FP32");
    }
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: num_heads must divide D");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D ||
        Wv_int8.rows != D || Wv_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be (D, D)");
    }
    if (sq.size() != D || sk.size() != D || sv.size() != D || so.size() != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: each scale tensor must have D entries");
    }
    const float* bias_p = nullptr;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be FP32");
        }
        if (attn_bias->size() != num_heads * L * L) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be (num_heads*L, L)");
        }
        bias_p = static_cast<const float*>(attn_bias->data);
    }
    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(L, D, Dtype::FP16);
    }
    if (L == 0 || D == 0) return;

    run_sab_int8<__half>(X, Wq_int8, sq, Wk_int8, sk, Wv_int8, sv, Wo_int8, so,
                         d_mask, bias_p, num_heads, scale, O);
}

void self_attention_decomposed_rel_pos_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h,
        const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale,
        ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const char* fn = "self_attention_decomposed_rel_pos_forward";
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error(std::string(fn) + ": X must be FP32, FP16, or BF16");
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype ||
        rel_pos_h.dtype != X.dtype || rel_pos_w.dtype != X.dtype)
        throw std::runtime_error(std::string(fn) +
            ": Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(std::string(fn) + ": num_heads must divide D");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(std::string(fn) + ": Wq/Wk/Wv/Wo must be (D, D)");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * grid_h - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*grid_h-1, head_dim)");
    if (rel_pos_w.rows != 2 * grid_w - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*grid_w-1, head_dim)");
    auto check_bias = [&](const ::brotensor::Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string(fn) + ": " + name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string(fn) + ": " + name + " must have D entries");
        }
    };
    check_bias(bq, "bq"); check_bias(bk, "bk");
    check_bias(bv, "bv"); check_bias(bo, "bo");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    auto bp = [](const ::brotensor::Tensor* b) {
        return (b && b->data) ? b->data : nullptr;
    };
    switch (X.dtype) {
    case Dtype::FP32:
        run_sardp<float>(X, Wq, static_cast<const float*>(bp(bq)),
                         Wk, static_cast<const float*>(bp(bk)),
                         Wv, static_cast<const float*>(bp(bv)),
                         Wo, static_cast<const float*>(bp(bo)),
                         rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, scale, O);
        break;
    case Dtype::FP16:
        run_sardp<__half>(X, Wq, static_cast<const __half*>(bp(bq)),
                          Wk, static_cast<const __half*>(bp(bk)),
                          Wv, static_cast<const __half*>(bp(bv)),
                          Wo, static_cast<const __half*>(bp(bo)),
                          rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, scale, O);
        break;
    default:  // BF16
        run_sardp<__nv_bfloat16>(X, Wq, static_cast<const __nv_bfloat16*>(bp(bq)),
                                 Wk, static_cast<const __nv_bfloat16*>(bp(bk)),
                                 Wv, static_cast<const __nv_bfloat16*>(bp(bv)),
                                 Wo, static_cast<const __nv_bfloat16*>(bp(bo)),
                                 rel_pos_h, rel_pos_w, num_heads, grid_h, grid_w, scale, O);
        break;
    }
}

void self_attention_decomposed_rel_pos_windowed_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h,
        const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const char* fn = "self_attention_decomposed_rel_pos_windowed_forward";
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error(std::string(fn) + ": X must be FP32, FP16, or BF16");
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype ||
        rel_pos_h.dtype != X.dtype || rel_pos_w.dtype != X.dtype)
        throw std::runtime_error(std::string(fn) +
            ": Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    const int L = X.rows;
    const int D = X.cols;
    if (window <= 0)
        throw std::runtime_error(std::string(fn) + ": window must be >= 1");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(std::string(fn) + ": num_heads must divide D");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(std::string(fn) + ": Wq/Wk/Wv/Wo must be (D, D)");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * window - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*window-1, head_dim)");
    if (rel_pos_w.rows != 2 * window - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*window-1, head_dim)");
    auto check_bias = [&](const ::brotensor::Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string(fn) + ": " + name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string(fn) + ": " + name + " must have D entries");
        }
    };
    check_bias(bq, "bq"); check_bias(bk, "bk");
    check_bias(bv, "bv"); check_bias(bo, "bo");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    auto bp = [](const ::brotensor::Tensor* b) {
        return (b && b->data) ? b->data : nullptr;
    };
    switch (X.dtype) {
    case Dtype::FP32:
        run_windowed_sardp<float>(X, Wq, static_cast<const float*>(bp(bq)),
                                  Wk, static_cast<const float*>(bp(bk)),
                                  Wv, static_cast<const float*>(bp(bv)),
                                  Wo, static_cast<const float*>(bp(bo)),
                                  rel_pos_h, rel_pos_w, num_heads,
                                  grid_h, grid_w, window, scale, O);
        break;
    case Dtype::FP16:
        run_windowed_sardp<__half>(X, Wq, static_cast<const __half*>(bp(bq)),
                                   Wk, static_cast<const __half*>(bp(bk)),
                                   Wv, static_cast<const __half*>(bp(bv)),
                                   Wo, static_cast<const __half*>(bp(bo)),
                                   rel_pos_h, rel_pos_w, num_heads,
                                   grid_h, grid_w, window, scale, O);
        break;
    default:  // BF16
        run_windowed_sardp<__nv_bfloat16>(X, Wq, static_cast<const __nv_bfloat16*>(bp(bq)),
                                          Wk, static_cast<const __nv_bfloat16*>(bp(bk)),
                                          Wv, static_cast<const __nv_bfloat16*>(bp(bv)),
                                          Wo, static_cast<const __nv_bfloat16*>(bp(bo)),
                                          rel_pos_h, rel_pos_w, num_heads,
                                          grid_h, grid_w, window, scale, O);
        break;
    }
}

} // namespace detail::cuda
} // namespace brotensor
