// CUDA loss kernels. Phase 2G port — kernel bodies unchanged.
//
// Losses are FP32-by-design: even in mixed-precision pipelines, logits feeding
// xent and prediction/target feeding mse are conventionally upcast to FP32 to
// preserve numerical headroom. We enforce FP32 explicitly on every operand and
// pass FP32 explicitly to every resize() so a stale committed-output dtype
// can't silently flip our reinterpret-casts to garbage.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void require_fp32(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (loss ops are FP32-by-design)");
    }
}

inline void require_fp32_out(const Tensor& t, const char* op, const char* name) {
    if (t.data != nullptr && t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (loss ops are FP32-by-design)");
    }
}


constexpr int LOSS_BLOCK = 256;

__global__ void mse_forward_kernel(const float* __restrict__ pred,
                                   const float* __restrict__ target,
                                   float* __restrict__ out_sum, int n) {
    __shared__ float sdata[LOSS_BLOCK];
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = pred[i] - target[i];
        local += d * d;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out_sum = sdata[0];
}

__global__ void mse_backward_kernel(const float* __restrict__ pred,
                                    const float* __restrict__ target,
                                    float* __restrict__ dPred, int n,
                                    float scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        dPred[i] = scale * (pred[i] - target[i]);
    }
}

__global__ void softmax_xent_fused_kernel(const float* __restrict__ logits,
                                          const float* __restrict__ target,
                                          const float* __restrict__ mask,
                                          float* __restrict__ probs,
                                          float* __restrict__ dLogits,
                                          float* __restrict__ out_loss,
                                          int n) {
    __shared__ float sdata[LOSS_BLOCK];
    const int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) continue;
        const float v = logits[i];
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
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) {
            probs[i] = 0.0f;
            continue;
        }
        const float e = expf(logits[i] - m);
        probs[i] = e;
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

    float local_loss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) {
            dLogits[i] = 0.0f;
            continue;
        }
        const float p = probs[i] * inv;
        probs[i] = p;
        const float t = target[i];
        if (t > 0.0f) {
            const float pc = p > 1e-12f ? p : 1e-12f;
            local_loss -= t * logf(pc);
        }
        dLogits[i] = p - t;
    }
    sdata[tid] = local_loss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out_loss = sdata[0];
}

inline int grid_for(int n, int block) {
    int b = (n + block - 1) / block;
    if (b < 1) b = 1;
    if (b > 4096) b = 4096;
    return b;
}

__global__ void mse_per_sample_kernel(const float* __restrict__ pred,
                                      const float* __restrict__ target,
                                      float* __restrict__ dPred,
                                      float* __restrict__ loss, int B) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B;
         b += blockDim.x * gridDim.x) {
        const float d = pred[b] - target[b];
        dPred[b] = d;
        loss[b]  = 0.5f * d * d;
    }
}

__global__ void softmax_xent_fused_batched_kernel(
        const float* __restrict__ logits,
        const float* __restrict__ target,
        const float* __restrict__ mask,
        const int*   __restrict__ head_offsets,
        float* __restrict__ probs,
        float* __restrict__ dLogits,
        float* __restrict__ loss_per_sample,
        int B, int n_heads, int n_act) {
    __shared__ float sdata[LOSS_BLOCK];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (b >= B || h >= n_heads) return;

    const int off = head_offsets[h];
    const int end = head_offsets[h + 1];
    const int len = end - off;

    const int row_off = b * n_act + off;
    const float* logits_row  = logits  + row_off;
    const float* target_row  = target  + row_off;
    const float* mask_row    = mask ? (mask + row_off) : nullptr;
    float*       probs_row   = probs   + row_off;
    float*       dLogits_row = dLogits + row_off;

    const int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < len; i += blockDim.x) {
        if (mask_row && mask_row[i] == 0.0f) continue;
        const float v = logits_row[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = sdata[tid];
            const float c = sdata[tid + s];
            sdata[tid] = a > c ? a : c;
        }
        __syncthreads();
    }
    const float m = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
        if (mask_row && mask_row[i] == 0.0f) {
            probs_row[i] = 0.0f;
            continue;
        }
        const float e = expf(logits_row[i] - m);
        probs_row[i] = e;
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

    float local_loss = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
        if (mask_row && mask_row[i] == 0.0f) {
            dLogits_row[i] = 0.0f;
            continue;
        }
        const float p = probs_row[i] * inv;
        probs_row[i] = p;
        const float t = target_row[i];
        if (t > 0.0f) {
            const float pc = p > 1e-12f ? p : 1e-12f;
            local_loss -= t * logf(pc);
        }
        dLogits_row[i] = p - t;
    }
    sdata[tid] = local_loss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&loss_per_sample[b], sdata[0]);
}

} // namespace

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    require_fp32(pred,   "mse_vec_forward", "pred");
    require_fp32(target, "mse_vec_forward", "target");
    const int n = pred.size();
    if (n == 0) return 0.0f;
    float* d_sum = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    mse_forward_kernel<<<1, LOSS_BLOCK>>>(
        static_cast<const float*>(pred.data),
        static_cast<const float*>(target.data),
        d_sum, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    float h_sum = 0.0f;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_sum);
    return h_sum / static_cast<float>(n);
}

void mse_vec_backward(const Tensor& pred, const Tensor& target,
                      Tensor& dPred) {
    require_fp32(pred,   "mse_vec_backward", "pred");
    require_fp32(target, "mse_vec_backward", "target");
    require_fp32_out(dPred, "mse_vec_backward", "dPred");
    const int n = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols ||
        dPred.dtype != ::brotensor::Dtype::FP32) {
        dPred.resize(pred.rows, pred.cols, ::brotensor::Dtype::FP32);
    }
    if (n == 0) return;
    const float scale = 2.0f / static_cast<float>(n);
    mse_backward_kernel<<<grid_for(n, LOSS_BLOCK), LOSS_BLOCK>>>(
        static_cast<const float*>(pred.data),
        static_cast<const float*>(target.data),
        static_cast<float*>(dPred.data), n, scale);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample) {
    require_fp32(pred,   "mse_vec_per_sample", "pred");
    require_fp32(target, "mse_vec_per_sample", "target");
    require_fp32_out(dPred,           "mse_vec_per_sample", "dPred");
    require_fp32_out(loss_per_sample, "mse_vec_per_sample", "loss_per_sample");
    const int B = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols ||
        dPred.dtype != ::brotensor::Dtype::FP32)
        dPred.resize(pred.rows, pred.cols, ::brotensor::Dtype::FP32);
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32)
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    if (B == 0) return;
    mse_per_sample_kernel<<<grid_for(B, LOSS_BLOCK), LOSS_BLOCK>>>(
        static_cast<const float*>(pred.data),
        static_cast<const float*>(target.data),
        static_cast<float*>(dPred.data),
        static_cast<float*>(loss_per_sample.data), B);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void softmax_xent_fused_batched(const Tensor& logits_BL,
                                const Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets,
                                int n_heads,
                                Tensor& probs_BL,
                                Tensor& dLogits_BL,
                                Tensor& loss_per_sample) {
    require_fp32(logits_BL, "softmax_xent_fused_batched", "logits_BL");
    require_fp32(target_BL, "softmax_xent_fused_batched", "target_BL");
    require_fp32_out(probs_BL,        "softmax_xent_fused_batched", "probs_BL");
    require_fp32_out(dLogits_BL,      "softmax_xent_fused_batched", "dLogits_BL");
    require_fp32_out(loss_per_sample, "softmax_xent_fused_batched", "loss_per_sample");
    const int B     = logits_BL.rows;
    const int n_act = logits_BL.cols;
    if (probs_BL.rows != B || probs_BL.cols != n_act ||
        probs_BL.dtype != ::brotensor::Dtype::FP32)
        probs_BL.resize(B, n_act, ::brotensor::Dtype::FP32);
    if (dLogits_BL.rows != B || dLogits_BL.cols != n_act ||
        dLogits_BL.dtype != ::brotensor::Dtype::FP32)
        dLogits_BL.resize(B, n_act, ::brotensor::Dtype::FP32);
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32)
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    if (B == 0 || n_act == 0 || n_heads <= 0) return;

    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(loss_per_sample.data, 0,
                                   sizeof(float) * B));

    // CUDA caps gridDim.y at 65535; chunk B to stay within the limit.
    constexpr int kMaxGridY = 65535;
    const auto* logits_p  = static_cast<const float*>(logits_BL.data);
    const auto* target_p  = static_cast<const float*>(target_BL.data);
    auto*       probs_p   = static_cast<float*>(probs_BL.data);
    auto*       dLogits_p = static_cast<float*>(dLogits_BL.data);
    auto*       loss_p    = static_cast<float*>(loss_per_sample.data);
    for (int b0 = 0; b0 < B; b0 += kMaxGridY) {
        const int b_chunk = (B - b0) < kMaxGridY ? (B - b0) : kMaxGridY;
        dim3 grid(n_heads, b_chunk);
        softmax_xent_fused_batched_kernel<<<grid, LOSS_BLOCK>>>(
            logits_p  + static_cast<size_t>(b0) * n_act,
            target_p  + static_cast<size_t>(b0) * n_act,
            d_mask_BL ? (d_mask_BL + static_cast<size_t>(b0) * n_act) : nullptr,
            d_head_offsets,
            probs_p   + static_cast<size_t>(b0) * n_act,
            dLogits_p + static_cast<size_t>(b0) * n_act,
            loss_p    + b0,
            b_chunk, n_heads, n_act);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask,
                         Tensor& probs, Tensor& dLogits) {
    require_fp32(logits, "softmax_xent_fused", "logits");
    require_fp32(target, "softmax_xent_fused", "target");
    require_fp32_out(probs,   "softmax_xent_fused", "probs");
    require_fp32_out(dLogits, "softmax_xent_fused", "dLogits");
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols ||
        probs.dtype != ::brotensor::Dtype::FP32) {
        probs.resize(logits.rows, logits.cols, ::brotensor::Dtype::FP32);
    }
    if (dLogits.rows != logits.rows || dLogits.cols != logits.cols ||
        dLogits.dtype != ::brotensor::Dtype::FP32) {
        dLogits.resize(logits.rows, logits.cols, ::brotensor::Dtype::FP32);
    }
    if (n == 0) return 0.0f;

    float* d_loss = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    softmax_xent_fused_kernel<<<1, LOSS_BLOCK>>>(
        static_cast<const float*>(logits.data),
        static_cast<const float*>(target.data),
        d_mask,
        static_cast<float*>(probs.data),
        static_cast<float*>(dLogits.data),
        d_loss, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    float h_loss = 0.0f;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_loss);
    return h_loss;
}

} // namespace brotensor::detail::cuda
