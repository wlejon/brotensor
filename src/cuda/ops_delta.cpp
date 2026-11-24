// CUDA gated delta rule scan backed by Brass MIR -> PTX (MirKind::DeltaRule,
// 32-lane warp register-resident state).

#include "cu_mir_cache.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string(op) + ": " + why);
}

void scan(const char* op, const Tensor& Q, const Tensor& K, const Tensor& V,
          const Tensor& a_raw, const Tensor& beta, const Tensor& log_A,
          int H, int dk, int dv, Tensor& state, Tensor& O) {
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

    constexpr std::uint32_t kWarp = 32;
    const std::uint32_t ne = (static_cast<std::uint32_t>(dk) + kWarp - 1) / kWarp;
    if (ne > 16) fail(op, "d_k is above 16 x 32 (512) on CUDA MIR");
    CUfunction fn = mir_function(
        {MirKind::DeltaRule, kWarp, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, static_cast<std::uint8_t>(ne)});
    const std::uint64_t lanes = static_cast<std::uint64_t>(H) * dv * kWarp;
    const std::uint64_t groups = (lanes + 255) / 256;
    if (groups > 65535) fail(op, "too many state rows for one dispatch");
    mir_dispatch_delta(
        fn, static_cast<std::uint32_t>(groups),
        Q.data, K.data, V.data, a_raw.data, beta.data, log_A.data, state.data, O.data,
        static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(H), static_cast<std::uint32_t>(dk),
        static_cast<std::uint32_t>(dv), static_cast<std::uint32_t>(Q.cols), static_cast<std::uint32_t>(V.cols));
}

}  // namespace

void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& a_raw, const Tensor& beta, const Tensor& log_A,
                              int num_heads, int d_k, int d_v, Tensor& state, Tensor& O) {
    scan("gated_delta_rule_chunked", Q, K, V, a_raw, beta, log_A, num_heads, d_k, d_v, state, O);
}

void gated_delta_rule_step(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& a_raw, const Tensor& beta, const Tensor& log_A,
                           int num_heads, int d_k, int d_v, Tensor& state, Tensor& O) {
    scan("gated_delta_rule_step", Q, K, V, a_raw, beta, log_A, num_heads, d_k, d_v, state, O);
}

void fill_cuda_vtable_gated_delta_rule(::brotensor::detail::OpsVTable& v) {
    v.gated_delta_rule_chunked = &gated_delta_rule_chunked;
    v.gated_delta_rule_step = &gated_delta_rule_step;
}

}  // namespace brotensor::detail::cuda
