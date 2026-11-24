// CUDA softmax cross-entropy and BCE-with-logits fused loss/gradient kernels
// backed by Brass MIR -> PTX (MirKind::Xent).

#include "cu_mir_cache.h"
#include "detail/cuda_check.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

void need_f32(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + what + " must be FP32");
    }
}

void need_f32_out(const Tensor& t, const char* op, const char* what) {
    if (t.data != nullptr && t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": " + what + " must be FP32");
    }
}

void like(const Tensor& src, Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != Dtype::FP32) {
        dst.resize(src.rows, src.cols, Dtype::FP32);
    }
}

void run_xent(const void* logits, const void* target, const float* mask, const int* off,
              int rows, int n, int heads, void* probs, void* dlog, void* loss,
              int mode = 0, float pos_weight = 1.0f) {
    const std::uint8_t flags = static_cast<std::uint8_t>((mask ? 1u : 0u) | (off ? 2u : 0u));
    CUfunction fn = mir_function(
        {MirKind::Xent, static_cast<std::uint32_t>(mode), Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    mir_dispatch_xent(
        fn, static_cast<std::uint32_t>(std::min(rows, 65535)),
        logits, target, mask, off, probs, dlog, loss,
        static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(n),
        static_cast<std::uint32_t>(heads), pos_weight);
}

float run_single(const char* op, ::brotensor::Device dev, const void* logits, const void* target,
                 const float* mask, void* probs, void* dlog, long long n) {
    if (n <= 0) return 0.0f;
    if (n > 0x7fffffffLL) throw std::runtime_error(std::string("brotensor: ") + op + ": too large");
    Tensor loss = Tensor::empty_on(dev, 1, 1, Dtype::FP32);
    run_xent(logits, target, mask, nullptr, 1, static_cast<int>(n), 1, probs, dlog, loss.data);
    float v = 0.0f;
    loss.copy_to_host_raw(&v, sizeof v);
    return v;
}

float single_tensor(const char* op, const Tensor& logits, const Tensor& target, const float* mask,
                    Tensor& probs, Tensor& dlog) {
    need_f32(logits, op, "logits");
    need_f32(target, op, "target");
    need_f32_out(probs, op, "probs");
    need_f32_out(dlog, op, "dLogits");
    if (target.size() != logits.size()) throw std::runtime_error(std::string("brotensor: ") + op + ": target shape");
    like(logits, probs);
    like(logits, dlog);
    return run_single(op, logits.device, logits.data, target.data, mask, probs.data, dlog.data, logits.size());
}

}  // namespace

float softmax_xent_segment(const float* logits, const float* target, float* probs, float* dLogits,
                           int n, const float* mask) {
    int dev_idx = 0;
    BROTENSOR_CUDA_CHECK(cudaGetDevice(&dev_idx));
    return run_single("softmax_xent_segment", ::brotensor::Device::cuda(dev_idx),
                      logits, target, mask, probs, dLogits, n);
}

float softmax_xent(const Tensor& logits, const Tensor& target, Tensor& probs, Tensor& dLogits,
                   const float* mask) {
    return single_tensor("softmax_xent", logits, target, mask, probs, dLogits);
}

float softmax_xent_fused(const Tensor& logits, const Tensor& target, const float* d_mask,
                         Tensor& probs, Tensor& dLogits) {
    return single_tensor("softmax_xent_fused", logits, target, d_mask, probs, dLogits);
}

void softmax_xent_fused_batched(const Tensor& logits_BL, const Tensor& target_BL, const float* d_mask_BL,
                                const int* d_head_offsets, int n_heads, Tensor& probs_BL,
                                Tensor& dLogits_BL, Tensor& loss_per_sample) {
    constexpr const char* op = "softmax_xent_fused_batched";
    need_f32(logits_BL, op, "logits_BL");
    need_f32(target_BL, op, "target_BL");
    need_f32_out(probs_BL, op, "probs_BL");
    need_f32_out(dLogits_BL, op, "dLogits_BL");
    need_f32_out(loss_per_sample, op, "loss_per_sample");
    if (target_BL.size() != logits_BL.size()) throw std::runtime_error(std::string("brotensor: ") + op + ": target shape");
    like(logits_BL, probs_BL);
    like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 || loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0 || n_heads <= 0) return;
    if (!d_head_offsets) throw std::runtime_error("brotensor: softmax_xent_fused_batched: d_head_offsets is null");
    run_xent(logits_BL.data, target_BL.data, d_mask_BL, d_head_offsets, logits_BL.rows, logits_BL.cols, n_heads,
             probs_BL.data, dLogits_BL.data, loss_per_sample.data);
}

void bce_with_logits_fused_batched(const Tensor& logits_BL, const Tensor& target_BL, const float* d_mask_BL,
                                   float pos_weight, Tensor& probs_BL, Tensor& dLogits_BL,
                                   Tensor& loss_per_sample) {
    constexpr const char* op = "bce_with_logits_fused_batched";
    need_f32(logits_BL, op, "logits_BL");
    need_f32(target_BL, op, "target_BL");
    need_f32_out(probs_BL, op, "probs_BL");
    need_f32_out(dLogits_BL, op, "dLogits_BL");
    need_f32_out(loss_per_sample, op, "loss_per_sample");
    if (target_BL.size() != logits_BL.size()) throw std::runtime_error(std::string("brotensor: ") + op + ": target shape");
    like(logits_BL, probs_BL);
    like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 || loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0) return;
    run_xent(logits_BL.data, target_BL.data, d_mask_BL, nullptr, logits_BL.rows, logits_BL.cols, 1,
             probs_BL.data, dLogits_BL.data, loss_per_sample.data, 1, pos_weight);
}

float mse_scalar(float pred, float target, float& dPred) {
    const float d = pred - target;
    dPred = d;
    return 0.5f * d * d;
}

void fill_cuda_vtable_loss_legacy(::brotensor::detail::OpsVTable& v) {
    v.softmax_xent = &softmax_xent;
    v.softmax_xent_segment = &softmax_xent_segment;
    v.mse_scalar = &mse_scalar;
}

}  // namespace brotensor::detail::cuda
