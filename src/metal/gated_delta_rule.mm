// Gated Delta Rule (Metal): gated_delta_rule_chunked (prefill) and
// gated_delta_rule_step (decode) via single-source brass MIR kernel
// (src/kernels/mir_delta.cpp -> mtl_mir_cache.mm).
//
// Each (head, v-row) is owned by one 32-lane Apple Silicon simdgroup that
// keeps the d_k state row in registers across the entire sequence L.

#include "mtl_mir_cache.h"

#include <brotensor/runtime.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::metal {

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string(op) + ": " + why);
}

void run_scan(const Tensor& Q, const Tensor& K, const Tensor& V,
              const Tensor& a_raw, const Tensor& beta, const Tensor& log_A,
              int num_heads, int d_k, int d_v,
              Tensor& state, Tensor& O, const char* op) {
    for (const Tensor* t :
         std::initializer_list<const Tensor*>{&Q, &K, &V, &a_raw, &beta, &log_A, &state}) {
        if (t->dtype != Dtype::FP32) {
            fail(op, "Q, K, V, a_raw, beta, log_A and state must be FP32");
        }
    }
    if (num_heads <= 0 || d_k <= 0 || d_v <= 0) {
        fail(op, "num_heads, d_k, d_v must be positive");
    }
    if (Q.cols != num_heads * d_k || K.cols != num_heads * d_k) {
        fail(op, "Q/K cols must equal num_heads * d_k");
    }
    if (V.cols != num_heads * d_v) {
        fail(op, "V.cols must equal num_heads * d_v");
    }
    if (K.rows != Q.rows || V.rows != Q.rows) {
        fail(op, "Q/K/V row count mismatch");
    }
    if (a_raw.rows != Q.rows || a_raw.cols != num_heads) {
        fail(op, "a_raw must be (L, num_heads)");
    }
    if (beta.rows != Q.rows || beta.cols != num_heads) {
        fail(op, "beta must be (L, num_heads)");
    }
    if (log_A.rows != num_heads || log_A.cols != 1) {
        fail(op, "log_A must be (num_heads, 1)");
    }
    if (state.rows != num_heads || state.cols != d_v * d_k) {
        fail(op, "state must be (num_heads, d_v*d_k)");
    }

    const int L = Q.rows;
    if (O.rows != L || O.cols != V.cols || O.dtype != Dtype::FP32) {
        O.resize(L, V.cols, Dtype::FP32);
    }
    if (L == 0) return;

    constexpr std::uint32_t kLpr = 32;  // Apple Silicon simdgroup width
    const std::uint32_t ne = (static_cast<std::uint32_t>(d_k) + kLpr - 1u) / kLpr;
    if (ne > 16u) {
        fail(op, "d_k is above 16 x the lanes per row (512 at subgroup 32) on Metal");
    }
    const MirKernel mk = mir_kernel(
        {MirKind::DeltaRule, kLpr, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32,
         static_cast<std::uint8_t>(ne)});
    const std::uint64_t lanes = static_cast<std::uint64_t>(num_heads) * d_v * kLpr;
    const std::uint64_t groups = (lanes + 255u) / 256u;
    if (groups > 65535u) fail(op, "too many state rows for one dispatch");
    mir_dispatch_delta(mk, static_cast<std::uint32_t>(groups), op,
                       Q.data, K.data, V.data, a_raw.data, beta.data,
                       log_A.data, state.data, O.data,
                       static_cast<std::uint32_t>(L),
                       static_cast<std::uint32_t>(num_heads),
                       static_cast<std::uint32_t>(d_k),
                       static_cast<std::uint32_t>(d_v),
                       static_cast<std::uint32_t>(Q.cols),
                       static_cast<std::uint32_t>(V.cols));
}

}  // namespace

void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& a_raw, const Tensor& beta,
                              const Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              Tensor& state, Tensor& O) {
    run_scan(Q, K, V, a_raw, beta, log_A, num_heads, d_k, d_v,
             state, O, "gated_delta_rule_chunked");
}

void gated_delta_rule_step(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& a_raw, const Tensor& beta,
                           const Tensor& log_A,
                           int num_heads, int d_k, int d_v,
                           Tensor& state, Tensor& O) {
    run_scan(Q, K, V, a_raw, beta, log_A, num_heads, d_k, d_v,
             state, O, "gated_delta_rule_step");
}

}  // namespace brotensor::detail::metal
