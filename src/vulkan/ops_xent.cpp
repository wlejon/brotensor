// Vulkan softmax cross-entropy: softmax_xent, softmax_xent_fused (one
// segment, the loss returned to the host, which syncs) and
// softmax_xent_fused_batched (per-row heads from a device offset table, a
// loss per row). FP32, as on the CPU and HIP backends; kernel
// shaders/xent.comp. softmax_xent_segment takes raw pointers only and stays
// on the CPU (src/ops.cpp routes it there).

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct XentPush {
    std::uint64_t logits, target, mask, off, probs, dlog, loss;
    std::uint32_t rows, n, heads;
};

void need_f32(const Tensor& t, const char* op, const char* what) {
    if (t.dtype != Dtype::FP32) throw std::runtime_error(std::string("brotensor: ") + op + ": " + what + " must be FP32");
}

void like(const Tensor& src, Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.dtype != Dtype::FP32) dst.resize(src.rows, src.cols, Dtype::FP32);
}

void run(const char* op, const Tensor& logits, const Tensor& target, const float* mask, const int* off, int rows, int n,
         int heads, Tensor& probs, Tensor& dlog, std::uint64_t loss) {
    need_f32(logits, op, "logits");
    need_f32(target, op, "target");
    if (target.size() != logits.size()) throw std::runtime_error(std::string("brotensor: ") + op + ": target shape");
    DeviceCtx& d = device_of(logits);
    const Kernel& k = d.pipelines().get(ShaderId::xent);
    const XentPush pc{addr(logits.data), addr(target.data), addr(mask), addr(off), addr(probs.data), addr(dlog.data),
                      loss, static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(n),
                      static_cast<std::uint32_t>(heads)};
    launch(d, k, pc, static_cast<std::uint32_t>(std::min(rows, 65535)));
}

float single(const char* op, const Tensor& logits, const Tensor& target, const float* mask, Tensor& probs,
             Tensor& dlog) {
    like(logits, probs);
    like(logits, dlog);
    const long long n = logits.size();
    if (n == 0) return 0.0f;
    if (n > 0x7fffffffLL) throw std::runtime_error(std::string("brotensor: ") + op + ": too large");
    Tensor loss = Tensor::empty_on(logits.device, 1, 1, Dtype::FP32);
    run(op, logits, target, mask, nullptr, 1, static_cast<int>(n), 1, probs, dlog, addr(loss.data));
    float v = 0.0f;
    loss.copy_to_host_raw(&v, sizeof v);
    return v;
}

}  // namespace

float softmax_xent(const Tensor& logits, const Tensor& target, Tensor& probs, Tensor& dLogits, const float* mask) {
    return single("softmax_xent", logits, target, mask, probs, dLogits);
}

float softmax_xent_fused(const Tensor& logits, const Tensor& target, const float* d_mask, Tensor& probs,
                         Tensor& dLogits) {
    return single("softmax_xent_fused", logits, target, d_mask, probs, dLogits);
}

void softmax_xent_fused_batched(const Tensor& logits_BL, const Tensor& target_BL, const float* d_mask_BL,
                                const int* d_head_offsets, int n_heads, Tensor& probs_BL, Tensor& dLogits_BL,
                                Tensor& loss_per_sample) {
    like(logits_BL, probs_BL);
    like(logits_BL, dLogits_BL);
    if (loss_per_sample.rows != logits_BL.rows || loss_per_sample.cols != 1 || loss_per_sample.dtype != Dtype::FP32) {
        loss_per_sample.resize(logits_BL.rows, 1, Dtype::FP32);
    }
    if (logits_BL.rows == 0 || logits_BL.cols == 0 || n_heads <= 0) return;
    if (!d_head_offsets) throw std::runtime_error("brotensor: softmax_xent_fused_batched: d_head_offsets is null");
    run("softmax_xent_fused_batched", logits_BL, target_BL, d_mask_BL, d_head_offsets, logits_BL.rows, logits_BL.cols,
        n_heads, probs_BL, dLogits_BL, addr(loss_per_sample.data));
}

void fill_vulkan_vtable_xent(::brotensor::detail::OpsVTable& v) {
    v.softmax_xent = &softmax_xent;
    v.softmax_xent_fused = &softmax_xent_fused;
    v.softmax_xent_fused_batched = &softmax_xent_fused_batched;
}

}  // namespace brotensor::detail::vulkan
