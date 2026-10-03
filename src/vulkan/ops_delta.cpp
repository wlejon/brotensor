// Vulkan gated delta rule (Qwen3.5's linear-attention layers):
// gated_delta_rule_step (decode) and gated_delta_rule_chunked (prefill) share
// one kernel, shaders/delta_rule.comp, as on CUDA (src/cuda/gated_delta_rule.cu).
// The recurrence is serial in tokens but every row of a head's (d_v, d_k)
// state evolves on its own, so a lane group per (head, row) keeps that row
// in registers across the whole sequence: the state is read and written once
// per call whatever L is. Contract: FP32 throughout, state (num_heads,
// d_v * d_k) updated in place, O resized to (L, num_heads * d_v).

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct DeltaPush {
    std::uint64_t q, k, v, a, b, loga, s, o;
    std::uint32_t L, H, dk, dv, ldq, ldv;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string(op) + ": " + why);
}

void scan(const char* op, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& a_raw, const Tensor& beta,
          const Tensor& log_A, int H, int dk, int dv, Tensor& state, Tensor& O) {
    for (const Tensor* t : std::initializer_list<const Tensor*>{&Q, &K, &V, &a_raw, &beta, &log_A, &state}) {
        if (t->dtype != Dtype::FP32) fail(op, "Q, K, V, a_raw, beta, log_A and state must be FP32");
    }
    if (H <= 0 || dk <= 0 || dv <= 0) fail(op, "num_heads, d_k, d_v must be positive");
    if (Q.cols != H * dk || K.cols != H * dk) fail(op, "Q/K cols must equal num_heads * d_k");
    if (V.cols != H * dv) fail(op, "V.cols must equal num_heads * d_v");
    if (K.rows != Q.rows || V.rows != Q.rows) fail(op, "Q/K/V row count mismatch");
    if (a_raw.rows != Q.rows || a_raw.cols != H) fail(op, "a_raw must be (L, num_heads)");
    if (beta.rows != Q.rows || beta.cols != H) fail(op, "beta must be (L, num_heads)");
    if (log_A.rows != H || log_A.cols != 1) fail(op, "log_A must be (num_heads, 1)");
    if (state.rows != H || state.cols != dv * dk) fail(op, "state must be (num_heads, d_v*d_k)");
    const int L = Q.rows;
    if (O.rows != L || O.cols != V.cols || O.dtype != Dtype::FP32) O.resize(L, V.cols, Dtype::FP32);
    if (L == 0) return;

    DeviceCtx& d = device_of(Q);
    const PhysInfo& info = d.info();
    // 32 lanes per row, pinned when the subgroup size can be required, else
    // the device's own subgroup size if it is smaller.
    std::uint32_t lpr = 32, pin = 0;
    if (info.subgroup_size_control && info.min_subgroup <= 32 && info.max_subgroup >= 32) {
        pin = 32;
    } else {
        lpr = std::min<std::uint32_t>(32, std::max<std::uint32_t>(1, info.subgroup_size));
    }
    const std::uint32_t ne = (static_cast<std::uint32_t>(dk) + lpr - 1) / lpr;
    if (ne > 16) fail(op, "d_k is above 16 x the lanes per row (512 at subgroup 32) on Vulkan");
    const Kernel& k = d.pipelines().get(ShaderId::delta_rule, {lpr, ne}, pin);
    const DeltaPush pc{addr(Q.data), addr(K.data), addr(V.data), addr(a_raw.data), addr(beta.data),
                       addr(log_A.data), addr(state.data), addr(O.data),
                       static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(dk),
                       static_cast<std::uint32_t>(dv), static_cast<std::uint32_t>(Q.cols),
                       static_cast<std::uint32_t>(V.cols)};
    const std::uint64_t lanes = static_cast<std::uint64_t>(H) * dv * lpr;
    const std::uint64_t groups = (lanes + k.local[0] - 1) / k.local[0];
    if (groups > 65535) fail(op, "too many state rows for one dispatch");
    launch(d, k, pc, static_cast<std::uint32_t>(groups));
}

}  // namespace

void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& a_raw,
                              const Tensor& beta, const Tensor& log_A, int num_heads, int d_k, int d_v,
                              Tensor& state, Tensor& O) {
    scan("gated_delta_rule_chunked", Q, K, V, a_raw, beta, log_A, num_heads, d_k, d_v, state, O);
}

void gated_delta_rule_step(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& a_raw,
                           const Tensor& beta, const Tensor& log_A, int num_heads, int d_k, int d_v, Tensor& state,
                           Tensor& O) {
    scan("gated_delta_rule_step", Q, K, V, a_raw, beta, log_A, num_heads, d_k, d_v, state, O);
}

void fill_vulkan_vtable_delta(::brotensor::detail::OpsVTable& v) {
    v.gated_delta_rule_chunked = &gated_delta_rule_chunked;
    v.gated_delta_rule_step = &gated_delta_rule_step;
}

}  // namespace brotensor::detail::vulkan
