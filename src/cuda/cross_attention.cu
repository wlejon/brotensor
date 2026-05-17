#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include "fp16_internal.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace brotensor {

namespace {

constexpr int CA_BLOCK = 128;

// One CUDA block per (query, head) tile. Computes scores against all Lk
// keys, applies numerically-stable softmax with optional mask, then writes
// the head's output slice as a weighted sum of V rows. Dynamic shared
// memory holds the Lk-long scores row in FP32.
__global__ void cross_attention_core_kernel(
        const __half* __restrict__ Q,    // (Lq, D)
        const __half* __restrict__ K,    // (Lk, D)
        const __half* __restrict__ V,    // (Lk, D)
        const float*  __restrict__ mask, // (Lk,) may be null
        __half* __restrict__ Out,        // (Lq, D)
        int Lq, int Lk, int D, int head_dim) {
    extern __shared__ float scratch[];
    float* scores = scratch;                       // size Lk
    float* s_red  = scratch + Lk;                  // size blockDim.x

    const int q  = blockIdx.x;
    const int h  = blockIdx.y;
    const int tid = threadIdx.x;
    const int head_off = h * head_dim;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    // 1. scores[k] = (Q[q, head_off:head_off+hd] · K[k, head_off:...]) * inv_sqrt
    //                + (-INF on mask=0). Single pass; each thread strides.
    for (int k = tid; k < Lk; k += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(Q[q * D + head_off + d]) *
                   __half2float(K[k * D + head_off + d]);
        }
        float s = dot * inv_sqrt;
        if (mask && mask[k] <= 0.5f) {
            s = -1e30f;
        }
        scores[k] = s;
    }
    __syncthreads();

    // 2. Numerically stable softmax: find max.
    float local_max = -1e30f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        if (scores[k] > local_max) local_max = scores[k];
    }
    s_red[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = s_red[tid + stride];
            if (other > s_red[tid]) s_red[tid] = other;
        }
        __syncthreads();
    }
    const float max_v = s_red[0];

    // 3. Exponentiate and sum.
    float local_sum = 0.0f;
    for (int k = tid; k < Lk; k += blockDim.x) {
        const float e = __expf(scores[k] - max_v);
        scores[k] = e;
        local_sum += e;
    }
    s_red[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        __syncthreads();
    }
    const float denom = s_red[0];
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

    // 4. Out[q, head_off + d] = sum_k (scores[k] * inv_denom) * V[k, head_off + d]
    //    Each thread handles a subset of d.
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < Lk; ++k) {
            acc += scores[k] * inv_denom *
                   __half2float(V[k * D + head_off + d]);
        }
        Out[q * D + head_off + d] = __float2half(acc);
    }
}

} // namespace

void cross_attention_forward_gpu(const GpuTensor& X,
                                 const GpuTensor& Ctx,
                                 const GpuTensor& Wq, const GpuTensor& Wk,
                                 const GpuTensor& Wv, const GpuTensor& Wo,
                                 const float* d_mask,
                                 int num_heads,
                                 GpuTensor& O) {
    if (X.dtype != Dtype::FP16 || Ctx.dtype != Dtype::FP16 ||
        Wq.dtype != Dtype::FP16 || Wk.dtype != Dtype::FP16 ||
        Wv.dtype != Dtype::FP16 || Wo.dtype != Dtype::FP16) {
        throw std::runtime_error("cross_attention_forward_gpu: all tensors must be FP16");
    }
    const int Lq = X.rows;
    const int Lk = Ctx.rows;
    const int D  = X.cols;
    if (Ctx.cols != D || Wq.rows != D || Wq.cols != D ||
        Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D ||
        Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("cross_attention_forward_gpu: shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("cross_attention_forward_gpu: num_heads must divide D");
    }
    const int head_dim = D / num_heads;

    if (O.rows != Lq || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(Lq, D, Dtype::FP16);
    }
    if (Lq == 0 || Lk == 0 || D == 0) return;

    // Scratch projections.
    GpuTensor Qp(Lq, D, Dtype::FP16);
    GpuTensor Kp(Lk, D, Dtype::FP16);
    GpuTensor Vp(Lk, D, Dtype::FP16);
    GpuTensor Op(Lq, D, Dtype::FP16);

    auto launch_matmul_ABT = [](const GpuTensor& A, const GpuTensor& B,
                                 GpuTensor& C, int M, int N, int K) {
        fp16_internal::launch_matmul_ABT(
            reinterpret_cast<const __half*>(A.data_fp16()),
            reinterpret_cast<const __half*>(B.data_fp16()),
            reinterpret_cast<__half*>(C.data_fp16()),
            M, N, K);
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
    };
    // Q = X @ Wq^T  →  shape (Lq, D), reading X(Lq, D) and Wq(D, D).
    launch_matmul_ABT(X,   Wq, Qp, Lq, D, D);
    launch_matmul_ABT(Ctx, Wk, Kp, Lk, D, D);
    launch_matmul_ABT(Ctx, Wv, Vp, Lk, D, D);

    // Attention core: dynamic shared mem = Lk floats (scores) + CA_BLOCK floats (reduction).
    const size_t shmem = (static_cast<size_t>(Lk) + CA_BLOCK) * sizeof(float);
    dim3 grid(Lq, num_heads, 1);
    cross_attention_core_kernel<<<grid, CA_BLOCK, shmem>>>(
        reinterpret_cast<const __half*>(Qp.data_fp16()),
        reinterpret_cast<const __half*>(Kp.data_fp16()),
        reinterpret_cast<const __half*>(Vp.data_fp16()),
        d_mask,
        reinterpret_cast<__half*>(Op.data_fp16()),
        Lq, Lk, D, head_dim);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // Output projection: O = Op @ Wo^T.
    launch_matmul_ABT(Op, Wo, O, Lq, D, D);
}

// Self-attention: thin wrapper that aliases X as the context. We can't pass
// the same GpuTensor twice as separate args without aliasing surprises, so
// build a non-owning FP16 view over X.data and pass that in as Ctx.
void self_attention_forward_gpu(const GpuTensor& X,
                                const GpuTensor& Wq, const GpuTensor& Wk,
                                const GpuTensor& Wv, const GpuTensor& Wo,
                                const float* d_mask,
                                int num_heads,
                                GpuTensor& O) {
    GpuTensor Ctx_view = GpuTensor::view_fp16(
        const_cast<uint16_t*>(X.data_fp16()), X.rows, X.cols);
    cross_attention_forward_gpu(X, Ctx_view, Wq, Wk, Wv, Wo,
                                d_mask, num_heads, O);
}

} // namespace brotensor
