// Gated Delta Rule (Metal). Mirrors src/cpu/gated_delta_rule.cpp.
//
// Qwen3-Next text-path matrix-valued recurrence (3-of-4 layers; the 1-of-4
// gated attention layer goes through flash_attention_decode). Per head h,
// per token t:
//   alpha_t = exp(-softplus(a_raw_t) * exp(log_A_h))      (decay gate, in (0,1])
//   beta_t  = sigmoid(beta_raw_t)
//   u_t     = S_{t-1} k_t
//   S_t     = alpha_t * S_{t-1} + beta_t * (v_t - u_t) k_t^T
//   o_t     = S_t q_t
// State S is (d_v, d_k) per head, laid out as state[h, v*d_k + k].
//
// Per-head parallel, per-token sequential (the inner recurrence is intrinsic
// to the algorithm). One threadgroup per head; threads inside the group
// partition the d_v dimension of S. Each thread owns a disjoint set of v rows
// for the lifetime of the head's scan — no cross-thread sync is needed within
// or across tokens because every state-row access is owned by exactly one
// thread.
//
// FP32-only — matches the CPU contract.

#include <brotensor/runtime.h>

#include <stdexcept>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

constexpr NSUInteger GDR_TG = 256;

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void k_gdr_fp32(device const float* Q     [[buffer(0)]],
                       device const float* K     [[buffer(1)]],
                       device const float* V     [[buffer(2)]],
                       device const float* a_raw [[buffer(3)]],
                       device const float* beta  [[buffer(4)]],
                       device const float* log_A [[buffer(5)]],
                       device float*       S     [[buffer(6)]],
                       device float*       O     [[buffer(7)]],
                       constant uint& L          [[buffer(8)]],
                       constant uint& num_heads  [[buffer(9)]],
                       constant uint& d_k        [[buffer(10)]],
                       constant uint& d_v        [[buffer(11)]],
                       uint h    [[threadgroup_position_in_grid]],
                       uint tid  [[thread_position_in_threadgroup]],
                       uint tgs  [[threads_per_threadgroup]]) {
    if (h >= num_heads) return;
    device float* Sh = S + h * (d_v * d_k);
    float exp_A = exp(log_A[h]);
    uint D_q = num_heads * d_k;
    uint D_v = num_heads * d_v;

    for (uint t = 0; t < L; ++t) {
        device const float* qt = Q + t * D_q + h * d_k;
        device const float* kt = K + t * D_q + h * d_k;
        device const float* vt = V + t * D_v + h * d_v;

        float a_raw_t = a_raw[t * num_heads + h];
        float beta_raw = beta[t * num_heads + h];
        // sigmoid(beta_raw), numerically stable
        float beta_t = (beta_raw >= 0.0f)
            ? (1.0f / (1.0f + exp(-beta_raw)))
            : (exp(beta_raw) / (1.0f + exp(beta_raw)));
        // softplus(a_raw_t), numerically stable
        float sp = max(a_raw_t, 0.0f) + log(1.0f + exp(-fabs(a_raw_t)));
        float alpha = exp(-sp * exp_A);

        device float* orow = O + t * D_v + h * d_v;

        // Each thread owns a disjoint set of v rows. For every owned v:
        //   u_v = sum_k S[v,k] * k[k]      (pass A — read S row v)
        //   S[v,k] = alpha*S[v,k] + beta*(v[v]-u_v)*k[k]   (update — write S row v)
        //   o_v = sum_k S[v,k] * q[k]      (pass B — read updated S row v)
        // No inter-thread dependency: row v is touched only by its owning thread.
        for (uint v = tid; v < d_v; v += tgs) {
            device float* Sv = Sh + v * d_k;
            float u = 0.0f;
            for (uint kk = 0; kk < d_k; ++kk) u += Sv[kk] * kt[kk];
            float scale = beta_t * (vt[v] - u);
            for (uint kk = 0; kk < d_k; ++kk) {
                Sv[kk] = alpha * Sv[kk] + scale * kt[kk];
            }
            float o = 0.0f;
            for (uint kk = 0; kk < d_k; ++kk) o += Sv[kk] * qt[kk];
            orow[v] = o;
        }
    }
}
)msl";

id<MTLComputePipelineState> pso_gdr() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, @"k_gdr_fp32"); });
    return pso;
}

void check_fp32(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU/Metal parity is FP32-only here)");
    }
}

void run_scan(const Tensor& Q, const Tensor& K, const Tensor& V,
              const Tensor& a_raw, const Tensor& beta, const Tensor& log_A,
              int num_heads, int d_k, int d_v,
              Tensor& state, Tensor& O, const char* op) {
    check_fp32(Q,     op, "Q");
    check_fp32(K,     op, "K");
    check_fp32(V,     op, "V");
    check_fp32(a_raw, op, "a_raw");
    check_fp32(beta,  op, "beta");
    check_fp32(log_A, op, "log_A");
    check_fp32(state, op, "state");

    if (num_heads <= 0 || d_k <= 0 || d_v <= 0) {
        throw std::runtime_error(std::string(op) +
                                 ": num_heads, d_k, d_v must be positive");
    }
    if (Q.cols != num_heads * d_k || K.cols != num_heads * d_k) {
        throw std::runtime_error(std::string(op) +
                                 ": Q/K cols must equal num_heads * d_k");
    }
    if (V.cols != num_heads * d_v) {
        throw std::runtime_error(std::string(op) +
                                 ": V.cols must equal num_heads * d_v");
    }
    if (K.rows != Q.rows || V.rows != Q.rows) {
        throw std::runtime_error(std::string(op) + ": Q/K/V row count mismatch");
    }
    if (a_raw.rows != Q.rows || a_raw.cols != num_heads) {
        throw std::runtime_error(std::string(op) + ": a_raw must be (L, num_heads)");
    }
    if (beta.rows != Q.rows || beta.cols != num_heads) {
        throw std::runtime_error(std::string(op) + ": beta must be (L, num_heads)");
    }
    if (log_A.rows != num_heads || log_A.cols != 1) {
        throw std::runtime_error(std::string(op) + ": log_A must be (num_heads, 1)");
    }
    if (state.rows != num_heads || state.cols != d_v * d_k) {
        throw std::runtime_error(std::string(op) + ": state must be (num_heads, d_v*d_k)");
    }

    const int L  = Q.rows;
    const int Dv = V.cols;
    if (O.rows != L || O.cols != Dv || O.dtype != Dtype::FP32) {
        O.resize(L, Dv, Dtype::FP32);
    }
    if (L == 0) return;

    id<MTLBuffer> bQ  = buffer_for(Q);
    id<MTLBuffer> bK  = buffer_for(K);
    id<MTLBuffer> bV  = buffer_for(V);
    id<MTLBuffer> bA  = buffer_for(a_raw);
    id<MTLBuffer> bB  = buffer_for(beta);
    id<MTLBuffer> bLA = buffer_for(log_A);
    id<MTLBuffer> bS  = buffer_for(state);
    id<MTLBuffer> bO  = buffer_for(O);
    const NSUInteger oQ  = buffer_offset_for(Q);
    const NSUInteger oK  = buffer_offset_for(K);
    const NSUInteger oV  = buffer_offset_for(V);
    const NSUInteger oA  = buffer_offset_for(a_raw);
    const NSUInteger oB  = buffer_offset_for(beta);
    const NSUInteger oLA = buffer_offset_for(log_A);
    const NSUInteger oS  = buffer_offset_for(state);
    const NSUInteger oO  = buffer_offset_for(O);

    const uint32_t Lu  = static_cast<uint32_t>(L);
    const uint32_t Hu  = static_cast<uint32_t>(num_heads);
    const uint32_t Dku = static_cast<uint32_t>(d_k);
    const uint32_t Dvu = static_cast<uint32_t>(d_v);

    // Threads per group: cap at GDR_TG, but no need to exceed d_v.
    NSUInteger tpg = (d_v < (int)GDR_TG) ? (NSUInteger)d_v : GDR_TG;
    if (tpg < 1) tpg = 1;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_gdr()];
        [enc setBuffer:bQ  offset:oQ  atIndex:0];
        [enc setBuffer:bK  offset:oK  atIndex:1];
        [enc setBuffer:bV  offset:oV  atIndex:2];
        [enc setBuffer:bA  offset:oA  atIndex:3];
        [enc setBuffer:bB  offset:oB  atIndex:4];
        [enc setBuffer:bLA offset:oLA atIndex:5];
        [enc setBuffer:bS  offset:oS  atIndex:6];
        [enc setBuffer:bO  offset:oO  atIndex:7];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&Hu  length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&Dku length:sizeof(uint32_t) atIndex:10];
        [enc setBytes:&Dvu length:sizeof(uint32_t) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& a_raw, const Tensor& beta,
                              const Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              Tensor& state, Tensor& O) {
    // Same scan as the streaming step on Metal too — the GPU "chunked" variant
    // would buy throughput at long prefill, but parity-first gets us a single
    // correct kernel that satisfies both contracts.
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

} // namespace brotensor::detail::metal
