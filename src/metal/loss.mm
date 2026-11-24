// Metal loss ops via single-source brass MIR kernels:
//   softmax_xent / softmax_xent_fused / softmax_xent_segment /
//   softmax_xent_fused_batched / bce_with_logits_fused_batched -> MirKind::Xent
//   mse_vec_forward / mse_vec_backward / mse_vec_per_sample    -> MirKind::Misc

#include "mtl_mir_cache.h"

#include "../vulkan/shaders/op_codes.h"

#include <brotensor/runtime.h>

#import "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

using metal_impl::pool_require;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void need_f32(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(what) + " must be FP32");
}

void ensure_f32_like(const Tensor& like, Tensor& t) {
    if (t.rows != like.rows || t.cols != like.cols || t.dtype != Dtype::FP32) {
        t.resize(like.rows, like.cols, Dtype::FP32);
    }
}

std::uint32_t count32(const Tensor& t, const char* op) {
    const long long n = t.size();
    if (n < 0 || n > 0x7fffffffLL) fail(op, "too large");
    return static_cast<std::uint32_t>(n);
}

void require_ptr(const void* p, const char* op, const char* name) {
    if (!p) return;
    NSUInteger ofs = 0;
    pool_require(p, ofs, op, name);
}

void run_xent_mir(const char* op, const void* logits, const void* target,
                  const float* mask, const int* off, int rows, int n, int heads,
                  void* probs, void* dlog, void* loss, int mode = 0,
                  float pos_weight = 1.0f) {
    const std::uint8_t flags =
        static_cast<std::uint8_t>((mask ? 1u : 0u) | (off ? 2u : 0u));
    const MirKernel mk = mir_kernel(
        {MirKind::Xent, static_cast<std::uint32_t>(mode), Dtype::FP32,
         Dtype::FP32, 1, Dtype::FP32, flags});
    mir_dispatch_xent(mk, static_cast<std::uint32_t>(std::min(rows, 65535)), op,
                      logits, target, mask, off, probs, dlog, loss,
                      static_cast<std::uint32_t>(rows),
                      static_cast<std::uint32_t>(n),
                      static_cast<std::uint32_t>(heads), pos_weight);
}

float run_xent_single(const char* op, const Tensor& logits, const Tensor& target,
                      const float* mask, Tensor& probs, Tensor& dlog) {
    need_f32(logits, op, "logits");
    need_f32(target, op, "target");
    if (target.size() != logits.size()) fail(op, "target shape mismatch");
    ensure_f32_like(logits, probs);
    ensure_f32_like(logits, dlog);
    const std::uint32_t n = count32(logits, op);
    if (n == 0) return 0.0f;
    require_ptr(mask, op, "mask");
    Tensor loss = Tensor::empty_on(logits.device, 1, 1, Dtype::FP32);
    run_xent_mir(op, logits.data, target.data, mask, nullptr, 1,
                 static_cast<int>(n), 1, probs.data, dlog.data, loss.data);
    float v = 0.0f;
    loss.copy_to_host_raw(&v, sizeof v);
    return v;
}

}  // namespace

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    constexpr const char* op = "mse_vec_forward";
    need_f32(pred, op, "pred");
    need_f32(target, op, "target");
    if (target.size() != pred.size()) fail(op, "size mismatch");
    const std::uint32_t n = count32(pred, op);
    if (n == 0) return 0.0f;
    Tensor out = Tensor::empty_on(pred.device, 1, 1, Dtype::FP32);
    const MirKernel mk = mir_kernel({MirKind::Misc, MI_MSE_SUM});
    const void* ptrs[] = {pred.data, target.data, out.data};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    mir_dispatch_args(mk, 1u, op, ptrs, ints, {});
    float s = 0.0f;
    out.copy_to_host_raw(&s, sizeof s);
    return s / static_cast<float>(n);
}

void mse_vec_backward(const Tensor& pred, const Tensor& target, Tensor& dPred) {
    constexpr const char* op = "mse_vec_backward";
    need_f32(pred, op, "pred");
    need_f32(target, op, "target");
    if (target.size() != pred.size()) fail(op, "size mismatch");
    ensure_f32_like(pred, dPred);
    const std::uint32_t n = count32(pred, op);
    if (n == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Misc, MI_MSE_BWD});
    const void* ptrs[] = {pred.data, target.data, dPred.data};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    const float flts[] = {2.0f / static_cast<float>(n)};
    mir_dispatch_args(mk, mir_groups_1d(n, 1), op, ptrs, ints, flts);
}

float mse_scalar(float pred, float target, float& dPred) {
    const float d = pred - target;
    dPred = d;
    return 0.5f * d * d;
}

void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample) {
    constexpr const char* op = "mse_vec_per_sample";
    need_f32(pred, op, "pred");
    need_f32(target, op, "target");
    if (target.size() != pred.size()) fail(op, "size mismatch");
    ensure_f32_like(pred, dPred);
    const std::uint32_t n = count32(pred, op);
    if (loss_per_sample.rows != static_cast<int>(n) || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(static_cast<int>(n), 1, Dtype::FP32);
    }
    if (n == 0) return;
    const MirKernel mk = mir_kernel({MirKind::Misc, MI_MSE_SAMPLE});
    const void* ptrs[] = {pred.data, target.data, dPred.data, loss_per_sample.data};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    mir_dispatch_args(mk, mir_groups_1d(n, 1), op, ptrs, ints, {});
}

float softmax_xent_fused(const Tensor& logits, const Tensor& target,
                         const float* d_mask, Tensor& probs, Tensor& dLogits) {
    return run_xent_single("softmax_xent_fused", logits, target, d_mask, probs, dLogits);
}

float softmax_xent(const Tensor& logits, const Tensor& target, Tensor& probs,
                   Tensor& dLogits, const float* mask) {
    return run_xent_single("softmax_xent", logits, target, mask, probs, dLogits);
}

float softmax_xent_segment(const float* logits, const float* target,
                           float* probs, float* dLogits, int n,
                           const float* mask) {
    constexpr const char* op = "softmax_xent_segment";
    if (n <= 0) return 0.0f;
    require_ptr(logits, op, "logits");
    require_ptr(target, op, "target");
    require_ptr(probs, op, "probs");
    require_ptr(dLogits, op, "dLogits");
    require_ptr(mask, op, "mask");
    Tensor loss = Tensor::empty_on(Device::Metal, 1, 1, Dtype::FP32);
    run_xent_mir(op, logits, target, mask, nullptr, 1, n, 1, probs, dLogits, loss.data);
    float v = 0.0f;
    loss.copy_to_host_raw(&v, sizeof v);
    return v;
}

void softmax_xent_fused_batched(const Tensor& logits_BL, const Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets, int n_heads,
                                Tensor& probs_BL, Tensor& dLogits_BL,
                                Tensor& loss_per_sample) {
    constexpr const char* op = "softmax_xent_fused_batched";
    need_f32(logits_BL, op, "logits");
    need_f32(target_BL, op, "target");
    if (target_BL.size() != logits_BL.size()) fail(op, "target shape mismatch");
    ensure_f32_like(logits_BL, probs_BL);
    ensure_f32_like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0 || n_heads <= 0) return;
    if (!d_head_offsets) fail(op, "d_head_offsets is null");
    require_ptr(d_mask_BL, op, "mask");
    require_ptr(d_head_offsets, op, "head_offsets");
    run_xent_mir(op, logits_BL.data, target_BL.data, d_mask_BL, d_head_offsets,
                 logits_BL.rows, logits_BL.cols, n_heads,
                 probs_BL.data, dLogits_BL.data, loss_per_sample.data);
}

void bce_with_logits_fused_batched(const Tensor& logits_BL,
                                   const Tensor& target_BL,
                                   const float* d_mask_BL, float pos_weight,
                                   Tensor& probs_BL, Tensor& dLogits_BL,
                                   Tensor& loss_per_sample) {
    constexpr const char* op = "bce_with_logits_fused_batched";
    need_f32(logits_BL, op, "logits");
    need_f32(target_BL, op, "target");
    if (target_BL.size() != logits_BL.size()) fail(op, "target shape mismatch");
    ensure_f32_like(logits_BL, probs_BL);
    ensure_f32_like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0) return;
    require_ptr(d_mask_BL, op, "mask");
    run_xent_mir(op, logits_BL.data, target_BL.data, d_mask_BL, nullptr,
                 logits_BL.rows, logits_BL.cols, 1,
                 probs_BL.data, dLogits_BL.data, loss_per_sample.data,
                 /*mode=*/1, pos_weight);
}

}  // namespace brotensor::detail::metal
