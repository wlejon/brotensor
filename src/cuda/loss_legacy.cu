// CUDA port of the older non-batched scalar-loss surface from
// src/cpu/ops_impl.cpp: softmax_xent / softmax_xent_segment / mse_scalar.
//
// Contract notes:
//   * softmax_xent_segment takes raw float* — on CUDA these are *device*
//     pointers (the dispatcher only lands here when the operand device is
//     CUDA). The mask pointer is device-side as well.
//   * softmax_xent takes Tensor& args; mask is device-side, matching the
//     softmax_xent_fused convention.
//   * mse_scalar is pure host scalar math — no device work. It is registered
//     under the CUDA vtable so dispatch on Device::CUDA doesn't throw; the
//     same host function as CPU is reused.
//
// Numerics mirror src/cpu/ops_impl.cpp byte-for-byte (stable softmax,
// xent loss = -sum(t * log(max(p, 1e-12))), dLogits = p - t).

#include "detail/cuda_check.h"

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Tensor;

namespace {

constexpr int LOSS_LEGACY_BLOCK = 256;

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// Single-block stable softmax + cross-entropy over a contiguous segment of
// length n. Matches src/cpu/ops_impl.cpp::softmax_xent_segment exactly.
__global__ void softmax_xent_segment_kernel(const float* __restrict__ logits,
                                            const float* __restrict__ target,
                                            const float* __restrict__ mask,
                                            float* __restrict__ probs,
                                            float* __restrict__ dLogits,
                                            float* __restrict__ out_loss,
                                            int n) {
    __shared__ float sdata[LOSS_LEGACY_BLOCK];
    const int tid = threadIdx.x;

    // 1) max over unmasked elements.
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

    // 2) exp + sum.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) { probs[i] = 0.0f; continue; }
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

    // 3) normalise + xent loss + dLogits = p - t.
    float local_loss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (mask && mask[i] < 0.5f) { dLogits[i] = 0.0f; continue; }
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

float run_softmax_xent_segment(const float* d_logits, const float* d_target,
                               float* d_probs, float* d_dLogits, int n,
                               const float* d_mask) {
    if (n <= 0) return 0.0f;

    float* d_loss = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    softmax_xent_segment_kernel<<<1, LOSS_LEGACY_BLOCK>>>(
        d_logits, d_target, d_mask, d_probs, d_dLogits, d_loss, n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    float h_loss = 0.0f;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float),
                                    cudaMemcpyDeviceToHost));
    cudaFree(d_loss);
    return h_loss;
}

} // namespace

// ─── public ops ─────────────────────────────────────────────────────────────

float softmax_xent_segment(const float* logits, const float* target,
                           float* probs, float* dLogits,
                           int n, const float* mask) {
    // Raw pointers: device-side on CUDA (dispatcher invariant).
    return run_softmax_xent_segment(logits, target, probs, dLogits, n, mask);
}

float softmax_xent(const Tensor& logits, const Tensor& target,
                   Tensor& probs, Tensor& dLogits, const float* mask) {
    // FP32-by-design: logits/target/probs/dLogits must all be FP32 (mixed-
    // precision pipelines upcast logits before xent). Reject any non-FP32
    // operand so reinterpret-casts to float* are always sound.
    if (logits.dtype != ::brotensor::Dtype::FP32)
        fail("softmax_xent", "logits must be FP32 (loss ops are FP32-by-design)");
    if (target.dtype != ::brotensor::Dtype::FP32)
        fail("softmax_xent", "target must be FP32 (loss ops are FP32-by-design)");
    if (probs.data != nullptr && probs.dtype != ::brotensor::Dtype::FP32)
        fail("softmax_xent", "probs must be FP32 (loss ops are FP32-by-design)");
    if (dLogits.data != nullptr && dLogits.dtype != ::brotensor::Dtype::FP32)
        fail("softmax_xent", "dLogits must be FP32 (loss ops are FP32-by-design)");
    const int n = logits.size();
    if (probs.rows != logits.rows || probs.cols != logits.cols ||
        probs.dtype != ::brotensor::Dtype::FP32) {
        probs.resize(logits.rows, logits.cols, ::brotensor::Dtype::FP32);
    }
    if (dLogits.rows != logits.rows || dLogits.cols != logits.cols ||
        dLogits.dtype != ::brotensor::Dtype::FP32) {
        dLogits.resize(logits.rows, logits.cols, ::brotensor::Dtype::FP32);
    }
    if (target.size() != n) fail("softmax_xent", "target size mismatch");
    return run_softmax_xent_segment(
        static_cast<const float*>(logits.data),
        static_cast<const float*>(target.data),
        static_cast<float*>(probs.data),
        static_cast<float*>(dLogits.data),
        n, mask);
}

// Pure host scalar math — same impl as CPU. Registered under the CUDA vtable
// so dispatch on Device::CUDA doesn't throw.
float mse_scalar(float pred, float target, float& dPred) {
    const float d = pred - target;
    dPred = d;
    return 0.5f * d * d;
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_loss_legacy(::brotensor::detail::OpsVTable& v) {
    v.softmax_xent         = &softmax_xent;
    v.softmax_xent_segment = &softmax_xent_segment;
    v.mse_scalar           = &mse_scalar;
}

} // namespace brotensor::detail::cuda
