// CUDA loss kernels.
//
// Losses are FP32-by-design: even in mixed-precision pipelines, logits feeding
// xent and prediction/target feeding mse are conventionally upcast to FP32 to
// preserve numerical headroom. We enforce FP32 explicitly on every operand and
// pass FP32 explicitly to every resize() so a stale committed-output dtype
// can't silently flip our reinterpret-casts to garbage.

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;

// Defined in tensor.cu (same namespace). The pooled allocator draws from a
// stream-ordered memory pool (cudaMallocAsync/cudaFreeAsync) instead of the
// synchronizing cudaMalloc/cudaFree pair — used below for the per-call
// scalar-loss scratch instead of raw cudaMalloc/cudaFree.
void* cuda_alloc(std::size_t bytes);
void  cuda_free(void* ptr);

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


// mse_forward_kernel / softmax_xent_fused_kernel (below) reduce over a
// SINGLE flat vector — one shared reduction domain — so they always launch
// exactly one block regardless of n; a vocab-sized n still only occupies one
// SM no matter the block size, but 1024 threads (vs. 256) still cuts the
// per-thread grid-stride factor 4x. The batched siblings below (mse_per_sample,
// softmax_xent_fused_batched, bce_with_logits_fused_batched) already scale
// their grid with B (see grid_for()/dim3(n_heads, b_chunk)), so bumping the
// shared block-size constant is neutral-to-positive there too.
constexpr int LOSS_BLOCK = 1024;

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

} // namespace

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    require_fp32(pred,   "mse_vec_forward", "pred");
    require_fp32(target, "mse_vec_forward", "target");
    const int n = pred.size();
    if (n == 0) return 0.0f;
    // Pooled/stream-ordered scratch alloc — this function returns a host
    // float per its signature, so the cudaStreamSynchronize below is
    // unavoidable (the caller needs the loss value on the host now), but the
    // allocation side no longer forces its own separate device-wide sync.
    float* d_sum = static_cast<float*>(cuda_alloc(sizeof(float)));
    mse_forward_kernel<<<1, LOSS_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(pred.data),
        static_cast<const float*>(target.data),
        d_sum, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    float h_sum = 0.0f;
    BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_sum, sizeof(float),
                              cudaMemcpyDeviceToHost, cur_stream()));
    BROTENSOR_CUDA_CHECK(cudaStreamSynchronize(cur_stream()));
    cuda_free(d_sum);
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
    mse_backward_kernel<<<grid_for(n, LOSS_BLOCK), LOSS_BLOCK, 0, cur_stream()>>>(
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
    mse_per_sample_kernel<<<grid_for(B, LOSS_BLOCK), LOSS_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(pred.data),
        static_cast<const float*>(target.data),
        static_cast<float*>(dPred.data),
        static_cast<float*>(loss_per_sample.data), B);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brotensor::detail::cuda
