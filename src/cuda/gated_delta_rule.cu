// ─── CUDA Gated Delta Rule ─────────────────────────────────────────────────
//
// Mirrors src/cpu/gated_delta_rule.cpp (FLA / HF Qwen3.5 ordering — decay
// BEFORE the delta read). Per token t, per head h:
//   alpha_t  = exp(-softplus(a_raw_t) * exp(log_A_h))    ∈ (0, 1]
//   beta_t   = sigmoid(beta_raw_t)
//   S_pre_t  = alpha_t * S_{t-1}
//   u_t      = S_pre_t k_t                               (predicted v)
//   S_t      = S_pre_t + beta_t * (v_t - u_t) k_t^T
//   o_t      = S_t q_t
// per-head state S has shape (d_v, d_k); o_t in R^{d_v}, q_t/k_t in R^{d_k},
// v_t in R^{d_v}.
//
// One CUDA block per head. The token loop stays sequential inside the block
// (the recurrence is fundamentally serial in t); threads parallelise the
// (d_v, d_k) state. Each token sees three passes:
//   A0) S[v,k] *= alpha                                 (decay, parallel over v*k)
//   A1) compute delta_v = v[v] - sum_k S_decayed[v,k] * k[k] (parallel over v)
//   A2) S[v,k] += beta * delta[v] * k[k]                (parallel over v*k)
//   B)  o_v   = sum_k S[v,k] * q[k]                     (parallel over v)
// delta lives in shared memory of size d_v. Both chunked and step share the
// same kernel — see CPU note: the chunked WY/UT transform is a GPU-throughput
// optimisation we can layer in later; the contract is identical either way.
//
// FP32-only. brolm's text path runs FP32 here per the public contract.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int GDR_BLOCK = 256;

__device__ inline float gdr_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = __expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = __expf(x);
        return e / (1.0f + e);
    }
}

__device__ inline float gdr_softplus(float x) {
    // max(x, 0) + log1p(exp(-|x|)) — numerically stable on both branches.
    return fmaxf(x, 0.0f) + log1pf(__expf(-fabsf(x)));
}

__global__ void gated_delta_rule_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        const float* __restrict__ a_raw,
                                        const float* __restrict__ beta,
                                        const float* __restrict__ log_A,
                                        int L, int num_heads, int d_k, int d_v,
                                        float* __restrict__ state,
                                        float* __restrict__ O) {
    extern __shared__ float sdelta[];      // size = d_v
    const int h = blockIdx.x;
    if (h >= num_heads) return;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    const int Dq = num_heads * d_k;
    const int Dv = num_heads * d_v;
    const int VK = d_v * d_k;

    float* S = state + h * VK;             // per-head state
    const float exp_A = __expf(log_A[h]);

    for (int t = 0; t < L; ++t) {
        const float* qt = Q + t * Dq + h * d_k;
        const float* kt = K + t * Dq + h * d_k;
        const float* vt = V + t * Dv + h * d_v;

        const float a_raw_t = a_raw[t * num_heads + h];
        const float b_raw_t = beta [t * num_heads + h];
        const float beta_t  = gdr_sigmoid(b_raw_t);
        const float alpha   = __expf(-gdr_softplus(a_raw_t) * exp_A);

        // FLA / HF ordering: decay S in place BEFORE computing the u read.
        // Pass A0: S[v,k] *= alpha
        for (int idx = tid; idx < VK; idx += bdim) {
            S[idx] *= alpha;
        }
        __syncthreads();

        // Pass A1: delta[v] = v[v] - sum_k S_decayed[v,k] * k[k]
        for (int v = tid; v < d_v; v += bdim) {
            const float* Sv = S + v * d_k;
            float u = 0.0f;
            for (int k = 0; k < d_k; ++k) u += Sv[k] * kt[k];
            sdelta[v] = vt[v] - u;
        }
        __syncthreads();

        // Pass A2: S[v,k] += beta * delta[v] * k[k]
        for (int idx = tid; idx < VK; idx += bdim) {
            const int v = idx / d_k;
            const int k = idx - v * d_k;
            S[idx] += beta_t * sdelta[v] * kt[k];
        }
        __syncthreads();

        // Pass B: o[v] = sum_k S[v,k] * q[k]
        float* orow = O + t * Dv + h * d_v;
        for (int v = tid; v < d_v; v += bdim) {
            const float* Sv = S + v * d_k;
            float o = 0.0f;
            for (int k = 0; k < d_k; ++k) o += Sv[k] * qt[k];
            orow[v] = o;
        }
        __syncthreads();  // ensure all S reads finished before next t's A2 writes
    }
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CUDA gated_delta_rule is FP32-only)");
    }
}

void run_scan(const ::brotensor::Tensor& Q,
              const ::brotensor::Tensor& K,
              const ::brotensor::Tensor& V,
              const ::brotensor::Tensor& a_raw,
              const ::brotensor::Tensor& beta,
              const ::brotensor::Tensor& log_A,
              int num_heads, int d_k, int d_v,
              ::brotensor::Tensor& state,
              ::brotensor::Tensor& O,
              const char* op) {
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
        throw std::runtime_error(std::string(op) +
                                 ": Q/K/V row count mismatch");
    }
    if (a_raw.rows != Q.rows || a_raw.cols != num_heads) {
        throw std::runtime_error(std::string(op) +
                                 ": a_raw must be (L, num_heads)");
    }
    if (beta.rows != Q.rows || beta.cols != num_heads) {
        throw std::runtime_error(std::string(op) +
                                 ": beta must be (L, num_heads)");
    }
    if (log_A.rows != num_heads || log_A.cols != 1) {
        throw std::runtime_error(std::string(op) +
                                 ": log_A must be (num_heads, 1)");
    }
    if (state.rows != num_heads || state.cols != d_v * d_k) {
        throw std::runtime_error(std::string(op) +
                                 ": state must be (num_heads, d_v*d_k)");
    }

    const int L  = Q.rows;
    const int Dv = V.cols;
    if (O.rows != L || O.cols != Dv || O.dtype != ::brotensor::Dtype::FP32) {
        O.resize(L, Dv, ::brotensor::Dtype::FP32);
    }
    if (L == 0) return;

    // Cap block at d_v * d_k (no point launching more threads than work units
    // in the largest pass) and at GDR_BLOCK to keep occupancy reasonable.
    int block = GDR_BLOCK;
    const int max_work = d_v * d_k;
    if (block > max_work) block = max_work;
    // Round to nearest power of two ≤ max so block_sum-style reductions (if
    // we add them later) stay clean; reductions aren't used here but the
    // shared-memory contract is simpler with a power-of-two block.
    int pow2 = 1;
    while (pow2 * 2 <= block) pow2 *= 2;
    block = pow2;
    if (block < 1) block = 1;

    const size_t shmem = static_cast<size_t>(d_v) * sizeof(float);
    gated_delta_rule_kernel<<<num_heads, block, shmem>>>(
        static_cast<const float*>(Q.data),
        static_cast<const float*>(K.data),
        static_cast<const float*>(V.data),
        static_cast<const float*>(a_raw.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(log_A.data),
        L, num_heads, d_k, d_v,
        static_cast<float*>(state.data),
        static_cast<float*>(O.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gated_delta_rule_chunked(const ::brotensor::Tensor& Q,
                              const ::brotensor::Tensor& K,
                              const ::brotensor::Tensor& V,
                              const ::brotensor::Tensor& a_raw,
                              const ::brotensor::Tensor& beta,
                              const ::brotensor::Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              ::brotensor::Tensor& state,
                              ::brotensor::Tensor& O) {
    run_scan(Q, K, V, a_raw, beta, log_A,
             num_heads, d_k, d_v, state, O, "gated_delta_rule_chunked");
}

void gated_delta_rule_step(const ::brotensor::Tensor& Q,
                           const ::brotensor::Tensor& K,
                           const ::brotensor::Tensor& V,
                           const ::brotensor::Tensor& a_raw,
                           const ::brotensor::Tensor& beta,
                           const ::brotensor::Tensor& log_A,
                           int num_heads, int d_k, int d_v,
                           ::brotensor::Tensor& state,
                           ::brotensor::Tensor& O) {
    run_scan(Q, K, V, a_raw, beta, log_A,
             num_heads, d_k, d_v, state, O, "gated_delta_rule_step");
}

void fill_cuda_vtable_gated_delta_rule(::brotensor::detail::OpsVTable& v) {
    v.gated_delta_rule_chunked = &gated_delta_rule_chunked;
    v.gated_delta_rule_step    = &gated_delta_rule_step;
}

} // namespace brotensor::detail::cuda
