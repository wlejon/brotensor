#include <brotensor/runtime.h>
#include <brotensor/detail/dispatch.h>

#include "fp16_internal.cuh"
#include "flash_fused_internal.cuh"
#include "gemm_mma.cuh"
#include "ops_fa_mir.h"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace {

// ─── Per-head fallback kernels (flash_attention_forward, uncovered head_dim) ─

template <typename T> struct fa_cvt;
template <> struct fa_cvt<__half> {
    __device__ static float to_f32(__half v) { return __half2float(v); }
    __device__ static __half from_f32(float v) { return __float2half(v); }
};
template <> struct fa_cvt<__nv_bfloat16> {
    __device__ static float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ static __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
};

template <typename T>
__global__ void fa_gather_head_kernel(const T* __restrict__ X, T* __restrict__ Y,
                                      int L, int D, int head_off, int hd, int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * hd) return;
    const int l = idx / hd, d = idx % hd;
    Y[static_cast<size_t>(l) * ldY + d] = X[static_cast<size_t>(l) * D + head_off + d];
}

template <typename T>
__global__ void fa_gather_head_T_kernel(const T* __restrict__ X, T* __restrict__ Y,
                                        int L, int D, int head_off, int hd, int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * hd) return;
    const int l = idx / hd, d = idx % hd;
    Y[static_cast<size_t>(d) * ldY + l] = X[static_cast<size_t>(l) * D + head_off + d];
}

template <typename T>
__global__ void fa_scatter_head_kernel(const T* __restrict__ Y, T* __restrict__ Out,
                                       int L, int D, int head_off, int hd, int ldY) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * hd) return;
    const int l = idx / hd, d = idx % hd;
    Out[static_cast<size_t>(l) * D + head_off + d] = Y[static_cast<size_t>(l) * ldY + d];
}

__device__ __forceinline__ float fa_block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : -1e30f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    return v;
}
__device__ __forceinline__ float fa_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = lane < nw ? red[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    return v;
}

template <typename T>
__global__ void fa_softmax_rows_f32_kernel(const float* __restrict__ S, T* __restrict__ P,
                                           int Lk, int ld, float scale,
                                           const float* __restrict__ mask) {
    __shared__ float red[32];
    const float* srow = S + static_cast<size_t>(blockIdx.x) * ld;
    T* prow = P + static_cast<size_t>(blockIdx.x) * ld;

    float mx = -1e30f;
    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        if (!mask || mask[k] > 0.5f) mx = fmaxf(mx, srow[k] * scale);
    }
    mx = fa_block_max(mx, red);
    const bool empty = mx <= -1e29f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < Lk; k += blockDim.x) {
        if (!mask || mask[k] > 0.5f) sum += __expf(srow[k] * scale - mx);
    }
    sum = fa_block_sum(sum, red);
    const float inv = (!empty && sum > 0.0f) ? 1.0f / sum : 0.0f;

    for (int k = threadIdx.x; k < ld; k += blockDim.x) {
        const bool valid = k < Lk && (!mask || mask[k] > 0.5f);
        prow[k] = fa_cvt<T>::from_f32(valid ? __expf(srow[k] * scale - mx) * inv : 0.0f);
    }
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

} // namespace

namespace detail::cuda {

namespace {

constexpr size_t kFaScoreBudget = size_t(32) << 20;

template <typename T>
bool fa_per_head_forward(const T* Q, const T* K, const T* V, const float* mask, T* O,
                         Dtype dt, int Lq, int Lk, int D, int num_heads, int hd,
                         cudaStream_t stream) {
    const int hdp = (hd + 7) & ~7;
    const int Lk_pad = (Lk + 7) & ~7;
    int rows = Lq;
    if (static_cast<size_t>(Lq) * Lk_pad > kFaScoreBudget) {
        rows = static_cast<int>(kFaScoreBudget / Lk_pad) / 128 * 128;
        if (rows < 128) rows = 128;
        if (rows > Lq) rows = Lq;
    }

    Tensor Qh  = Tensor::empty_on(Device::CUDA, Lq, hdp, dt);
    Tensor Kh  = Tensor::empty_on(Device::CUDA, Lk_pad, hdp, dt);
    Tensor Vth = Tensor::empty_on(Device::CUDA, hdp, Lk_pad, dt);
    Tensor Oh  = Tensor::empty_on(Device::CUDA, Lq, hdp, dt);
    Tensor S   = Tensor::empty_on(Device::CUDA, rows, Lk_pad, Dtype::FP32);
    Tensor P   = Tensor::empty_on(Device::CUDA, rows, Lk_pad, dt);
    T* qh = static_cast<T*>(Qh.data);
    T* kh = static_cast<T*>(Kh.data);
    T* vth = static_cast<T*>(Vth.data);
    T* oh = static_cast<T*>(Oh.data);
    float* s = static_cast<float*>(S.data);
    T* p = static_cast<T*>(P.data);

    if (hdp != hd) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(qh, 0, sizeof(T) * size_t(Lq) * hdp, stream));
    }
    if (hdp != hd || Lk_pad != Lk) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(kh, 0, sizeof(T) * size_t(Lk_pad) * hdp, stream));
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(vth, 0, sizeof(T) * size_t(hdp) * Lk_pad, stream));
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(hd));
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    constexpr int CP = 256;

    for (int h = 0; h < num_heads; ++h) {
        const int off = h * hd;
        fa_gather_head_kernel<T><<<grid_for(Lq * hd, CP), CP, 0, stream>>>(Q, qh, Lq, D, off, hd, hdp);
        fa_gather_head_kernel<T><<<grid_for(Lk * hd, CP), CP, 0, stream>>>(K, kh, Lk, D, off, hd, hdp);
        fa_gather_head_T_kernel<T><<<grid_for(Lk * hd, CP), CP, 0, stream>>>(V, vth, Lk, D, off, hd, Lk_pad);
        for (int q0 = 0; q0 < Lq; q0 += rows) {
            const int m = Lq - q0 < rows ? Lq - q0 : rows;
            if (!detail::cuda::mma_gemm::launch_f32out(qh + size_t(q0) * hdp, kh, s, m, Lk_pad, hdp,
                                                       stream)) {
                return false;
            }
            fa_softmax_rows_f32_kernel<T><<<m, sm_block, 0, stream>>>(s, p, Lk, Lk_pad, scale, mask);
            fp16_internal::launch_matmul_ABT(p, vth, oh + size_t(q0) * hdp, m, hdp, Lk_pad);
        }
        fa_scatter_head_kernel<T><<<grid_for(Lq * hd, CP), CP, 0, stream>>>(oh, O, Lq, D, off, hd, hdp);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

void flash_attention_forward(const Tensor& Q,
                             const Tensor& K,
                             const Tensor& V,
                             const float* d_mask,
                             int num_heads,
                             bool causal,
                             Tensor& O) {
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_forward: Q, K, V must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt) {
        throw std::runtime_error("flash_attention_forward: Q, K, V dtype must match");
    }
    const bool bf16 = (dt == Dtype::BF16);
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_forward: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_forward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_forward: causal requires Lq == Lk");
    }
    const int head_dim = D / num_heads;
    if (O.rows != Lq || O.cols != D || O.dtype != dt) {
        O.resize(Lq, D, dt);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    if (flash_fused::supported(head_dim)) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        if (bf16) {
            flash_fused::launch(
                reinterpret_cast<const __nv_bfloat16*>(Q.data),
                reinterpret_cast<const __nv_bfloat16*>(K.data),
                reinterpret_cast<const __nv_bfloat16*>(V.data),
                d_mask,
                reinterpret_cast<__nv_bfloat16*>(O.data),
                Lq, Lk, D, num_heads, head_dim, causal, stream);
        } else {
            flash_fused::launch(
                reinterpret_cast<const __half*>(Q.data),
                reinterpret_cast<const __half*>(K.data),
                reinterpret_cast<const __half*>(V.data),
                d_mask,
                reinterpret_cast<__half*>(O.data),
                Lq, Lk, D, num_heads, head_dim, causal, stream);
        }
        return;
    }

    if (!causal) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
        const bool done = bf16
            ? fa_per_head_forward(reinterpret_cast<const __nv_bfloat16*>(Q.data),
                                  reinterpret_cast<const __nv_bfloat16*>(K.data),
                                  reinterpret_cast<const __nv_bfloat16*>(V.data), d_mask,
                                  reinterpret_cast<__nv_bfloat16*>(O.data), dt,
                                  Lq, Lk, D, num_heads, head_dim, stream)
            : fa_per_head_forward(reinterpret_cast<const __half*>(Q.data),
                                  reinterpret_cast<const __half*>(K.data),
                                  reinterpret_cast<const __half*>(V.data), d_mask,
                                  reinterpret_cast<__half*>(O.data), dt,
                                  Lq, Lk, D, num_heads, head_dim, stream);
        if (done) return;
    }

    detail::cuda::dispatch_fa_rows(
        detail::cuda::kFaModeRows, dt, Q.device,
        Q.data, K.data, V.data, O.data,
        d_mask, nullptr, nullptr,
        Lq, Lk, D, D, D,
        num_heads, num_heads, head_dim,
        0, causal, false, 0, 0, 0.0f);
}

// Forward declarations of flash attention ops in sibling files.
void flash_attention_gqa_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 const float* d_mask, int num_q_heads, int num_kv_heads,
                                 bool causal, Tensor& O);
void flash_attention_windowed_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                      const float* d_mask, int num_heads, int window,
                                      Tensor& O, bool causal);
void flash_attention_varlen_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
                                    int batch_size, int max_seqlen_q, int max_seqlen_k,
                                    int num_heads, int head_dim, bool causal, Tensor& O);
void flash_attention_qkvo_forward(const Tensor& X, const Tensor* Ctx,
                                  const Tensor& Wq, const Tensor* bq,
                                  const Tensor& Wk, const Tensor* bk,
                                  const Tensor& Wv, const Tensor* bv,
                                  const Tensor& Wo, const Tensor* bo,
                                  const float* d_mask, int num_heads, bool causal, Tensor& O);
void flash_attention_qkvo_backward(
    const Tensor& X, const Tensor* Ctx,
    const Tensor& Wq, const Tensor* bq,
    const Tensor& Wk, const Tensor* bk,
    const Tensor& Wv, const Tensor* bv,
    const Tensor& Wo, const Tensor* bo,
    const float* d_mask, int num_heads, bool causal, const Tensor& dO,
    Tensor& dX, Tensor* dCtx,
    Tensor& dWq, Tensor* dbq,
    Tensor& dWk, Tensor* dbk,
    Tensor& dWv, Tensor* dbv,
    Tensor& dWo, Tensor* dbo);
void flash_attention_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& O, const Tensor& dO,
                              const float* d_mask, int num_heads, bool causal,
                              Tensor& dQ, Tensor& dK, Tensor& dV);
void flash_attention_varlen_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                     const Tensor& O, const Tensor& dO,
                                     const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k,
                                     int num_heads, int head_dim, bool causal,
                                     Tensor& dQ, Tensor& dK, Tensor& dV);
void flash_attention_project_kv(const Tensor& ctx,
                                const Tensor& Wk, const Tensor* bk,
                                const Tensor& Wv, const Tensor* bv,
                                Tensor& K_out, Tensor& V_out);
void flash_attention_q_with_kv_cached_forward(const Tensor& X, const Tensor& K, const Tensor& V,
                                              const Tensor& Wq, const Tensor* bq,
                                              const Tensor& Wo, const Tensor* bo,
                                              const float* d_mask, int num_heads,
                                              bool causal, Tensor& O);
void flash_attention_qkvo_int8w_fp16(const Tensor& X, const Tensor* Ctx,
                                     const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                     const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                     const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                     const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                     const float* d_mask, int num_heads, bool causal, Tensor& O);
void flash_attention_q_with_kv_cached_int8w_fp16(const Tensor& X, const Tensor& K, const Tensor& V,
                                                 const Tensor& Wq_int8, const Tensor& sq, const Tensor* bq,
                                                 const Tensor& Wo_int8, const Tensor& so, const Tensor* bo,
                                                 const float* d_mask, int num_heads,
                                                 bool causal, Tensor& O);
void flash_attention_project_kv_int8w_fp16(const Tensor& ctx,
                                           const Tensor& Wk_int8, const Tensor& sk, const Tensor* bk,
                                           const Tensor& Wv_int8, const Tensor& sv, const Tensor* bv,
                                           Tensor& K_out, Tensor& V_out);

void fill_cuda_vtable_flash_attention(::brotensor::detail::OpsVTable& v) {
    v.flash_attention_forward                     = &flash_attention_forward;
    v.flash_attention_gqa_forward                 = &flash_attention_gqa_forward;
    v.flash_attention_windowed_forward            = &flash_attention_windowed_forward;
    v.flash_attention_varlen_forward              = &flash_attention_varlen_forward;
    v.flash_attention_qkvo_forward                = &flash_attention_qkvo_forward;
    v.flash_attention_qkvo_backward               = &flash_attention_qkvo_backward;
    v.flash_attention_backward                    = &flash_attention_backward;
    v.flash_attention_varlen_backward             = &flash_attention_varlen_backward;
    v.flash_attention_project_kv                  = &flash_attention_project_kv;
    v.flash_attention_q_with_kv_cached_forward    = &flash_attention_q_with_kv_cached_forward;
    v.flash_attention_qkvo_int8w_fp16             = &flash_attention_qkvo_int8w_fp16;
    v.flash_attention_q_with_kv_cached_int8w_fp16 = &flash_attention_q_with_kv_cached_int8w_fp16;
    v.flash_attention_project_kv_int8w_fp16       = &flash_attention_project_kv_int8w_fp16;
}

} // namespace detail::cuda

} // namespace brotensor
