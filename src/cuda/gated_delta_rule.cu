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
//
// OCCUPANCY: this launches exactly `num_heads` blocks (<<<num_heads, block>>>
// below), so a batch of B sequences occupies only num_heads SMs per serial
// per-sequence launch rather than num_heads*B — the natural fix is folding a
// batch axis into the grid, but the public contract (delta_rule.h) is
// single-sequence (Q/K/V have no batch dimension), so that would be an
// ABI-visible shape change, not a purely internal one — deferred pending an
// explicit decision to widen the op's contract. The WY/UT chunked-parallel
// transform noted above is the other lever and doesn't require an API
// change, but is a substantially larger algorithmic undertaking; also
// deferred. Both are documented here rather than attempted blind.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

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

__global__ void gated_delta_rule_step_fast_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ a_raw,
    const float* __restrict__ beta,
    const float* __restrict__ log_A,
    int num_heads, int d_k, int d_v,
    float* __restrict__ state,
    float* __restrict__ O) {

    extern __shared__ float shmem[];
    float* s_k  = shmem;
    float* s_q  = shmem + d_k;
    float* s_kq = shmem + 2 * d_k;

    const int h = blockIdx.x;
    if (h >= num_heads) return;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    const int VK = d_v * d_k;

    const float* q_h = Q + h * d_k;
    const float* k_h = K + h * d_k;
    const float* v_h = V + h * d_v;
    float* S_h       = state + h * VK;
    float* o_h       = O + h * d_v;

    // Load scalars for this head
    const float a_raw_val = a_raw[h];
    const float b_raw_val = beta[h];
    const float beta_val  = gdr_sigmoid(b_raw_val);
    const float alpha_val = __expf(-gdr_softplus(a_raw_val) * __expf(log_A[h]));

    // Stage k and q into shared memory
    for (int j = tid; j < d_k; j += bdim) {
        s_k[j] = k_h[j];
        s_q[j] = q_h[j];
    }
    __syncthreads();

    // Compute scalar dot product: kq = sum_j s_k[j] * s_q[j]
    float p_kq = 0.0f;
    for (int j = tid; j < d_k; j += bdim) {
        p_kq += s_k[j] * s_q[j];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        p_kq += __shfl_down_sync(0xffffffffu, p_kq, off);
    }
    __shared__ float red_kq[8];
    if ((tid & 31) == 0) red_kq[tid >> 5] = p_kq;
    __syncthreads();
    if (tid == 0) {
        float total_kq = red_kq[0];
        const int nwarps = bdim >> 5;
        for (int w = 1; w < nwarps; ++w) total_kq += red_kq[w];
        *s_kq = total_kq;
    }
    __syncthreads();
    const float kq_val = *s_kq;

    // Now, each warp handles rows of S
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int num_warps = bdim >> 5;

    const bool is_aligned16 = ((reinterpret_cast<uintptr_t>(S_h) & 15) == 0) &&
                              ((reinterpret_cast<uintptr_t>(s_k) & 15) == 0) &&
                              ((reinterpret_cast<uintptr_t>(s_q) & 15) == 0);
    const int d_k4 = d_k >> 2;

    if (((d_k & 3) == 0) && is_aligned16 && d_k4 <= 32) {
        // Ultra-fast path: d_k <= 128 and multiple of 4 (e.g. d_k=128)
        // 1 load of S, all math in registers, 1 store of S
        const float4* s_k4 = reinterpret_cast<const float4*>(s_k);
        const float4* s_q4 = reinterpret_cast<const float4*>(s_q);
        const float4 kj = (lane_id < d_k4) ? s_k4[lane_id] : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 qj = (lane_id < d_k4) ? s_q4[lane_id] : make_float4(0.f, 0.f, 0.f, 0.f);

        for (int r = warp_id; r < d_v; r += num_warps) {
            float4* S_row4 = reinterpret_cast<float4*>(S_h + r * d_k);
            float4 s = (lane_id < d_k4) ? S_row4[lane_id] : make_float4(0.f, 0.f, 0.f, 0.f);

            float dot_k = s.x * kj.x + s.y * kj.y + s.z * kj.z + s.w * kj.w;
            float dot_q = s.x * qj.x + s.y * qj.y + s.z * qj.z + s.w * qj.w;

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot_k += __shfl_down_sync(0xffffffffu, dot_k, off);
                dot_q += __shfl_down_sync(0xffffffffu, dot_q, off);
            }

            float beta_delta = 0.0f;
            if (lane_id == 0) {
                const float u = alpha_val * dot_k;
                const float delta = v_h[r] - u;
                o_h[r] = alpha_val * dot_q + beta_val * delta * kq_val;
                beta_delta = beta_val * delta;
            }
            beta_delta = __shfl_sync(0xffffffffu, beta_delta, 0);

            if (lane_id < d_k4) {
                s.x = alpha_val * s.x + beta_delta * kj.x;
                s.y = alpha_val * s.y + beta_delta * kj.y;
                s.z = alpha_val * s.z + beta_delta * kj.z;
                s.w = alpha_val * s.w + beta_delta * kj.w;
                S_row4[lane_id] = s;
            }
        }
    } else if (((d_k & 3) == 0) && is_aligned16) {
        // Fast path for d_k > 128 and multiple of 4
        const float4* s_k4 = reinterpret_cast<const float4*>(s_k);
        const float4* s_q4 = reinterpret_cast<const float4*>(s_q);

        for (int r = warp_id; r < d_v; r += num_warps) {
            float4* S_row4 = reinterpret_cast<float4*>(S_h + r * d_k);
            float dot_k = 0.0f;
            float dot_q = 0.0f;

            for (int j4 = lane_id; j4 < d_k4; j4 += 32) {
                const float4 s  = S_row4[j4];
                const float4 kj = s_k4[j4];
                const float4 qj = s_q4[j4];
                dot_k += s.x * kj.x + s.y * kj.y + s.z * kj.z + s.w * kj.w;
                dot_q += s.x * qj.x + s.y * qj.y + s.z * qj.z + s.w * qj.w;
            }

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot_k += __shfl_down_sync(0xffffffffu, dot_k, off);
                dot_q += __shfl_down_sync(0xffffffffu, dot_q, off);
            }

            float beta_delta = 0.0f;
            if (lane_id == 0) {
                const float u = alpha_val * dot_k;
                const float delta = v_h[r] - u;
                o_h[r] = alpha_val * dot_q + beta_val * delta * kq_val;
                beta_delta = beta_val * delta;
            }
            beta_delta = __shfl_sync(0xffffffffu, beta_delta, 0);

            for (int j4 = lane_id; j4 < d_k4; j4 += 32) {
                float4 s = S_row4[j4];
                const float4 kj = s_k4[j4];
                s.x = alpha_val * s.x + beta_delta * kj.x;
                s.y = alpha_val * s.y + beta_delta * kj.y;
                s.z = alpha_val * s.z + beta_delta * kj.z;
                s.w = alpha_val * s.w + beta_delta * kj.w;
                S_row4[j4] = s;
            }
        }
    } else {
        // General path for arbitrary d_k
        for (int r = warp_id; r < d_v; r += num_warps) {
            float* S_row = S_h + r * d_k;
            float dot_k = 0.0f;
            float dot_q = 0.0f;

            for (int j = lane_id; j < d_k; j += 32) {
                const float s = S_row[j];
                dot_k += s * s_k[j];
                dot_q += s * s_q[j];
            }

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot_k += __shfl_down_sync(0xffffffffu, dot_k, off);
                dot_q += __shfl_down_sync(0xffffffffu, dot_q, off);
            }

            float beta_delta = 0.0f;
            if (lane_id == 0) {
                const float u = alpha_val * dot_k;
                const float delta = v_h[r] - u;
                o_h[r] = alpha_val * dot_q + beta_val * delta * kq_val;
                beta_delta = beta_val * delta;
            }
            beta_delta = __shfl_sync(0xffffffffu, beta_delta, 0);

            for (int j = lane_id; j < d_k; j += 32) {
                S_row[j] = alpha_val * S_row[j] + beta_delta * s_k[j];
            }
        }
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

    if (L == 1) {
        int num_warps = (d_v < 8) ? d_v : 8;
        if (num_warps < 1) num_warps = 1;
        const int block = num_warps * 32;
        const size_t shmem = static_cast<size_t>(2 * d_k + 1) * sizeof(float);
        gated_delta_rule_step_fast_kernel<<<num_heads, block, shmem, cur_stream()>>>(
            static_cast<const float*>(Q.data),
            static_cast<const float*>(K.data),
            static_cast<const float*>(V.data),
            static_cast<const float*>(a_raw.data),
            static_cast<const float*>(beta.data),
            static_cast<const float*>(log_A.data),
            num_heads, d_k, d_v,
            static_cast<float*>(state.data),
            static_cast<float*>(O.data));
        BROTENSOR_CUDA_CHECK(cudaGetLastError());
        return;
    }

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
    gated_delta_rule_kernel<<<num_heads, block, shmem, cur_stream()>>>(
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
