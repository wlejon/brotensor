// Vulkan per-row selection ops next to attention: top_k_rows (an exact rank
// count, any k up to the row length) and segment_softmax_stats (Laya's
// per-segment softmax features). Kernel: shaders/topk_seg.comp. Contracts
// follow the CPU reference (src/cpu/top_k.cpp, packed_encoder.cpp): Vals and
// the stats are FP32, Idx INT32; X may also be FP16 / BF16 here.

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct TopkPush {
    std::uint64_t x, vals, idx, off;
    std::uint32_t rows, cols, k, nseg;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

}  // namespace

void top_k_rows(const Tensor& X, int k, Tensor& Vals, Tensor& Idx) {
    constexpr const char* op = "top_k_rows";
    dt_code(X.dtype, op);
    const int R = X.rows, C = X.cols;
    if (k < 1) fail(op, "k must be >= 1");
    if (k > C) fail(op, "k must be <= C (per-row length)");
    if (Vals.rows != R || Vals.cols != k || Vals.dtype != Dtype::FP32) Vals.resize(R, k, Dtype::FP32);
    if (Idx.rows != R || Idx.cols != k || Idx.dtype != Dtype::INT32) Idx.resize(R, k, Dtype::INT32);
    if (R == 0) return;
    const std::uint64_t groups = (static_cast<std::uint64_t>(C) + 255) / 256;
    if (groups > 65535) fail(op, "rows longer than 16M elements");
    DeviceCtx& d = device_of(X);
    const Kernel& kern = d.pipelines().get(dt_variant(ShaderId::topk_seg_f32, X.dtype, op), {0u});
    const std::uint64_t es = X.dtype == Dtype::FP32 ? 4 : 2;
    for (int r0 = 0; r0 < R; r0 += 65535) {
        const int n = std::min(65535, R - r0);
        const TopkPush pc{addr(X.data) + static_cast<std::uint64_t>(r0) * C * es,
                          addr(Vals.data) + static_cast<std::uint64_t>(r0) * k * 4,
                          addr(Idx.data) + static_cast<std::uint64_t>(r0) * k * 4, 0,
                          static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(C), static_cast<std::uint32_t>(k), 0};
        launch(d, kern, pc, static_cast<std::uint32_t>(groups), static_cast<std::uint32_t>(n));
    }
}

void segment_softmax_stats(const Tensor& logits, const Tensor& seg_offsets, Tensor& out) {
    constexpr const char* op = "segment_softmax_stats";
    dt_code(logits.dtype, op);
    if (seg_offsets.dtype != Dtype::INT32 || seg_offsets.rows < 1) fail(op, "seg_offsets must be (S+1, 1) INT32");
    const int S = seg_offsets.rows - 1;
    // In the logits' dtype, as CUDA / HIP (the CPU reference is FP32-only).
    if (out.rows != S || out.cols != 4 || out.dtype != logits.dtype) out.resize(S, 4, logits.dtype);
    if (S == 0) return;
    // Offsets are device data: the kernel clamps them to the logits (the CPU
    // reference throws on a bad table instead).
    DeviceCtx& d = device_of(logits);
    const Kernel& kern = d.pipelines().get(dt_variant(ShaderId::topk_seg_f32, logits.dtype, op), {1u});
    const long long total = logits.size();
    if (total > 0x7fffffffLL) fail(op, "logits too large");
    const TopkPush pc{addr(logits.data), addr(out.data), 0, addr(seg_offsets.data), 0,
                      static_cast<std::uint32_t>(total), 0, static_cast<std::uint32_t>(S)};
    launch(d, kern, pc, static_cast<std::uint32_t>(std::min(S, 65535)));
}

void fill_vulkan_vtable_topk(::brotensor::detail::OpsVTable& v) {
    v.top_k_rows = &top_k_rows;
    v.segment_softmax_stats = &segment_softmax_stats;
}

}  // namespace brotensor::detail::vulkan
