// Backward of flash_attention_forward_gpu — bare attention core, no
// projection weights. Recompute-based: reproduces the per-head softmax to
// obtain P, then runs the standard FlashAttention-2 backward (dV = P^T·dO,
// dP = dO·V^T, dS = P*(dP - D_q)*inv_sqrt, dQ = dS·K, dK = dS^T·Q) per head
// and packs results back into (Lq, D) / (Lk, D).
//
// Numerically this is the same per-head sweep used inside
// flash_attention_qkvo_backward_gpu (src/cuda/flash_attention.cu) — the
// helper kernels (extract_head_LD, pack_head_LD, extract_head_DL,
// fa_scale_mask_causal_softmax_rows_kernel, fa_dP_kernel, fa_dS_from_P_dP_kernel,
// fa_dVh_kernel, fa_dQh_kernel, fa_dKh_kernel) live in flash_attention.cu.
// We forward-declare them here at namespace scope so we can call them
// without duplicating code. They have internal linkage in flash_attention.cu;
// to keep this file self-contained, we re-define equivalent helpers in our
// own anonymous namespace.

#include <brotensor/runtime.h>

#include "fp16_internal.cuh"
#include "detail/cuda_check.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

namespace brotensor {

void* cuda_current_stream();

namespace {

constexpr int FAB_BLOCK = 128;

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return b;
}

// ───── Per-head extract / pack kernels (mirror flash_attention.cu) ─────────

__global__ void fab_extract_head_LD_kernel(const __half* __restrict__ X,
                                           __half* __restrict__ Y,
                                           int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[l * head_dim + d] = X[l * D + head_off + d];
}

__global__ void fab_extract_head_DL_kernel(const __half* __restrict__ X,
                                           __half* __restrict__ Y,
                                           int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Y[d * L + l] = X[l * D + head_off + d];
}

__global__ void fab_pack_head_LD_kernel(const __half* __restrict__ Y,
                                        __half* __restrict__ Out,
                                        int L, int D, int head_off, int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = L * head_dim;
    if (idx >= total) return;
    const int l = idx / head_dim;
    const int d = idx % head_dim;
    Out[l * D + head_off + d] = Y[l * head_dim + d];
}

// Row-wise scaled, optionally masked, optionally causal softmax. One block
// per query row. Equivalent to fa_scale_mask_causal_softmax_rows_kernel.
__global__ void fab_softmax_rows_kernel(__half* __restrict__ S,
                                        int Lq, int Lk,
                                        float scale,
                                        const float* __restrict__ mask,
                                        int causal) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __half* row = S + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __half2float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        if (v > local_max) local_max = v;
    }
    ssm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float o = ssm[tid + s];
            if (o > ssm[tid]) ssm[tid] = o;
        }
        __syncthreads();
    }
    const float rmax = ssm[0];
    const bool empty = (rmax <= -1e29f);

    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        float v = __half2float(row[k]) * scale;
        if (mask && mask[k] <= 0.5f) v = -1e30f;
        if (causal && k > q) v = -1e30f;
        const float e = empty ? 0.0f : __expf(v - rmax);
        row[k] = __float2half(e);
        local_sum += e;
    }
    ssm[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float rsum = ssm[0];
    const float inv = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        const float e = __half2float(row[k]);
        row[k] = __float2half(e * inv);
    }
}

// dP[q, k] = sum_d dO_h[q, d] * V_h[k, d]   (Lq, Lk), FP32 accumulation
__global__ void fab_dP_kernel(const __half* __restrict__ dOh,
                              const __half* __restrict__ Vh,
                              __half* __restrict__ dP,
                              int Lq, int Lk, int hd) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || k >= Lk) return;
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) {
        acc += __half2float(dOh[q * hd + d]) * __half2float(Vh[k * hd + d]);
    }
    dP[q * Lk + k] = __float2half(acc);
}

// In-place dS = P * (dP - D_q) * scale  where D_q = sum_k P[q,k]*dP[q,k].
__global__ void fab_dS_from_P_dP_kernel(__half* __restrict__ P_dS,
                                        const __half* __restrict__ dP,
                                        int Lq, int Lk,
                                        float scale) {
    extern __shared__ float ssm[];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    __half* prow = P_dS + static_cast<size_t>(q) * static_cast<size_t>(Lk);
    const __half* dprow = dP + static_cast<size_t>(q) * static_cast<size_t>(Lk);

    float local = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        local += __half2float(prow[k]) * __half2float(dprow[k]);
    }
    ssm[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssm[tid] += ssm[tid + s];
        __syncthreads();
    }
    const float Dq = ssm[0];

    for (int k = tid; k < Lk; k += blockDim.x) {
        const float p  = __half2float(prow[k]);
        const float dp = __half2float(dprow[k]);
        prow[k] = __float2half(p * (dp - Dq) * scale);
    }
}

// dV_h[k, d] = sum_q P[q, k] * dO_h[q, d]
__global__ void fab_dVh_kernel(const __half* __restrict__ P,
                               const __half* __restrict__ dOh,
                               __half* __restrict__ dVh,
                               int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += __half2float(P[q * Lk + k]) * __half2float(dOh[q * hd + d]);
    }
    dVh[k * hd + d] = __float2half(acc);
}

// dQ_h[q, d] = sum_k dS[q, k] * K_h[k, d]
__global__ void fab_dQh_kernel(const __half* __restrict__ dS,
                               const __half* __restrict__ Kh,
                               __half* __restrict__ dQh,
                               int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= Lq || d >= hd) return;
    float acc = 0.0f;
    for (int k = 0; k < Lk; ++k) {
        acc += __half2float(dS[q * Lk + k]) * __half2float(Kh[k * hd + d]);
    }
    dQh[q * hd + d] = __float2half(acc);
}

// dK_h[k, d] = sum_q dS[q, k] * Q_h[q, d]
__global__ void fab_dKh_kernel(const __half* __restrict__ dS,
                               const __half* __restrict__ Qh,
                               __half* __restrict__ dKh,
                               int Lq, int Lk, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= Lk || d >= hd) return;
    float acc = 0.0f;
    for (int q = 0; q < Lq; ++q) {
        acc += __half2float(dS[q * Lk + k]) * __half2float(Qh[q * hd + d]);
    }
    dKh[k * hd + d] = __float2half(acc);
}

} // namespace

namespace detail::cuda {

// O is consumed by the qkvo backward only via Wo backward; here we don't have
// Wo. O is not actually needed for the recompute path — we re-derive P from
// (Q, K) + softmax. We accept O in the signature to mirror standard
// flash-attn backward APIs and to allow a future caller-supplied-cache path,
// but the parameter is currently unused.
void flash_attention_backward(const Tensor& Q,
                              const Tensor& K,
                              const Tensor& V,
                              const Tensor& O,
                              const Tensor& dO,
                              const float* d_mask,
                              int num_heads,
                              bool causal,
                              Tensor& dQ,
                              Tensor& dK,
                              Tensor& dV) {
    (void)O;  // recompute-based; O retained in API for symmetry.

    if (Q.dtype != Dtype::FP16 || K.dtype != Dtype::FP16 ||
        V.dtype != Dtype::FP16 || dO.dtype != Dtype::FP16) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must be FP16");
    }
    const int Lq = Q.rows;
    const int Lk = K.rows;
    const int D  = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dQ.rows != Lq || dQ.cols != D || dQ.dtype != Dtype::FP16) {
        dQ.resize(Lq, D, Dtype::FP16);
    }
    if (dK.rows != Lk || dK.cols != D || dK.dtype != Dtype::FP16) {
        dK.resize(Lk, D, Dtype::FP16);
    }
    if (dV.rows != Lk || dV.cols != D || dV.dtype != Dtype::FP16) {
        dV.resize(Lk, D, Dtype::FP16);
    }
    dQ.zero();
    dK.zero();
    dV.zero();

    if (Lq == 0 || Lk == 0 || D == 0) return;

    constexpr int CP_BLOCK = 256;
    int sm_block = 32;
    while (sm_block < Lk && sm_block < 1024) sm_block *= 2;
    if (sm_block > 1024) sm_block = 1024;
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));

    Tensor Qh = Tensor::empty_on(Device::CUDA, Lq, hd, Dtype::FP16);
    Tensor Kh = Tensor::empty_on(Device::CUDA, Lk, hd, Dtype::FP16);
    Tensor Vh = Tensor::empty_on(Device::CUDA, Lk, hd, Dtype::FP16);
    Tensor dOh = Tensor::empty_on(Device::CUDA, Lq, hd, Dtype::FP16);
    Tensor P = Tensor::empty_on(Device::CUDA, Lq, Lk, Dtype::FP16);
    Tensor dP = Tensor::empty_on(Device::CUDA, Lq, Lk, Dtype::FP16);
    Tensor dQh = Tensor::empty_on(Device::CUDA, Lq, hd, Dtype::FP16);
    Tensor dKh = Tensor::empty_on(Device::CUDA, Lk, hd, Dtype::FP16);
    Tensor dVh = Tensor::empty_on(Device::CUDA, Lk, hd, Dtype::FP16);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_current_stream());
    (void)FAB_BLOCK;

    for (int h = 0; h < num_heads; ++h) {
        const int head_off = h * hd;
        const int total_q = Lq * hd;
        const int total_k = Lk * hd;

        // Extract per-head buffers (Q_h, K_h, V_h, dO_h).
        fab_extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(Q.data),
            reinterpret_cast<__half*>(Qh.data),
            Lq, D, head_off, hd);
        fab_extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(K.data),
            reinterpret_cast<__half*>(Kh.data),
            Lk, D, head_off, hd);
        fab_extract_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(V.data),
            reinterpret_cast<__half*>(Vh.data),
            Lk, D, head_off, hd);
        fab_extract_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dO.data),
            reinterpret_cast<__half*>(dOh.data),
            Lq, D, head_off, hd);

        // Recompute P: S(Lq, Lk) = Qh · Kh^T then row-softmax(scale, mask, causal).
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(Qh.data),
            reinterpret_cast<const __half*>(Kh.data),
            reinterpret_cast<__half*>(P.data),
            Lq, Lk, hd);
        {
            const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);
            fab_softmax_rows_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P.data),
                Lq, Lk, inv_sqrt, d_mask, causal ? 1 : 0);
        }

        // dV_h = P^T · dO_h
        {
            dim3 block(16, 16);
            dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
            fab_dVh_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(P.data),
                reinterpret_cast<const __half*>(dOh.data),
                reinterpret_cast<__half*>(dVh.data),
                Lq, Lk, hd);
        }

        // dP = dO_h · V_h^T
        {
            dim3 block(16, 16);
            dim3 grid((Lk + 15) / 16, (Lq + 15) / 16);
            fab_dP_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(dOh.data),
                reinterpret_cast<const __half*>(Vh.data),
                reinterpret_cast<__half*>(dP.data),
                Lq, Lk, hd);
        }

        // dS = P * (dP - D_q) * inv_sqrt  (in-place over P)
        {
            const size_t shmem = static_cast<size_t>(sm_block) * sizeof(float);
            fab_dS_from_P_dP_kernel<<<Lq, sm_block, shmem, stream>>>(
                reinterpret_cast<__half*>(P.data),
                reinterpret_cast<const __half*>(dP.data),
                Lq, Lk, inv_sqrt);
        }

        // dQ_h = dS · K_h
        {
            dim3 block(16, 16);
            dim3 grid((hd + 15) / 16, (Lq + 15) / 16);
            fab_dQh_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(P.data),
                reinterpret_cast<const __half*>(Kh.data),
                reinterpret_cast<__half*>(dQh.data),
                Lq, Lk, hd);
        }

        // dK_h = dS^T · Q_h
        {
            dim3 block(16, 16);
            dim3 grid((hd + 15) / 16, (Lk + 15) / 16);
            fab_dKh_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(P.data),
                reinterpret_cast<const __half*>(Qh.data),
                reinterpret_cast<__half*>(dKh.data),
                Lq, Lk, hd);
        }

        // Pack per-head grads back into the (L, D) slot for this head.
        fab_pack_head_LD_kernel<<<grid_for(total_q, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dQh.data),
            reinterpret_cast<__half*>(dQ.data),
            Lq, D, head_off, hd);
        fab_pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dKh.data),
            reinterpret_cast<__half*>(dK.data),
            Lk, D, head_off, hd);
        fab_pack_head_LD_kernel<<<grid_for(total_k, CP_BLOCK), CP_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(dVh.data),
            reinterpret_cast<__half*>(dV.data),
            Lk, D, head_off, hd);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace detail::cuda

} // namespace brotensor
