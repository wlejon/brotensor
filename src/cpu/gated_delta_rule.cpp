// ─── CPU Gated Delta Rule ──────────────────────────────────────────────────
//
// FP32-only host implementation of the Gated DeltaNet matrix-valued recurrence
// used by hybrid linear-attention text decoders (the linear-attention layers
// alternate with standard gated attention, handled via flash_attention_decode).
//
// Per token t, per head h (FLA / HF Qwen3.5 ordering — decay applied BEFORE
// the delta read, so u_t is computed against the decayed state):
//   alpha_t  = exp(-softplus(a_raw_t) * exp(log_A_h))     (decay gate, in (0,1])
//   beta_t   = sigmoid(beta_raw_t)                         (write strength)
//   S_pre_t  = alpha_t * S_{t-1}                           (decayed state)
//   u_t      = S_pre_t k_t                                 (predicted v)
//   S_t      = S_pre_t + beta_t * (v_t - u_t) k_t^T
//   o_t      = S_t q_t
// per-head state S has shape (d_v, d_k); o_t in R^{d_v}, q_t/k_t in R^{d_k},
// v_t in R^{d_v}.
//
// The chunked WY/UT-transform exists for GPU throughput; on CPU a plain
// sequential scan is the same complexity and clearer. gated_delta_rule_chunked
// and gated_delta_rule_step therefore share the same inner loop; the only
// reason for two ops is the GPU split (where chunked prefill matters).

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

// log(1 + exp(x)) — numerically stable.
inline float softplus(float x) {
    return std::max(x, 0.0f) + std::log1p(std::exp(-std::abs(x)));
}

// 1 / (1 + exp(-x)).
inline float sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = std::exp(x);
        return e / (1.0f + e);
    }
}

// Shared per-call validation and inner scan. Both chunked and step share these
// rules; only the op name differs in error messages.
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
    const int Dq = Q.cols;
    const int Dv = V.cols;
    if (O.rows != L || O.cols != Dv || O.dtype != Dtype::FP32) {
        O.resize(L, Dv, Dtype::FP32);
    }
    if (L == 0) return;

    const float* Qp = Q.host_f32();
    const float* Kp = K.host_f32();
    const float* Vp = V.host_f32();
    const float* Ap = a_raw.host_f32();      // (L, num_heads)
    const float* Bp = beta.host_f32();       // (L, num_heads)
    const float* logA = log_A.host_f32();    // (num_heads, 1)
    float* Sp = state.host_f32_mut();        // (num_heads, d_v * d_k)
    float* Op = O.host_f32_mut();

    const int Sh_stride = d_v * d_k;         // per-head state stride

    // Per head independently — each head has its own S, and there is no
    // cross-head interaction. We loop heads-outer / tokens-inner so the
    // d_v * d_k state stays hot in cache for the whole sequence.
    for (int h = 0; h < num_heads; ++h) {
        float* S = Sp + h * Sh_stride;        // S[v, k] = S[v*d_k + k]
        const float exp_A = std::exp(logA[h]);

        for (int t = 0; t < L; ++t) {
            const float* qt = Qp + t * Dq + h * d_k;
            const float* kt = Kp + t * Dq + h * d_k;
            const float* vt = Vp + t * Dv + h * d_v;

            const float a_raw_t = Ap[t * num_heads + h];
            const float beta_t  = sigmoid(Bp[t * num_heads + h]);
            // alpha = exp(-softplus(a_raw) * exp(log_A))  ∈ (0, 1]
            const float alpha = std::exp(-softplus(a_raw_t) * exp_A);

            // u = S * k  (shape d_v).  S row v stride d_k.
            // We hold u on-stack via direct accumulation into the update;
            // for clarity and to allow the (v - u) compute, materialise it.
            // d_v is typically <= 256 in practice, so a stack-sized array
            // would suffice, but we use a small local heap buffer to avoid
            // VLA portability concerns.
            //
            // To keep the kernel allocation-light, fold the three steps:
            //   1) u_v = sum_k S[v,k] * k[k]
            //   2) delta_v = v[v] - u_v
            //   3) S[v,k] = alpha * S[v,k] + beta * delta_v * k[k]
            //   4) o_v    = sum_k S[v,k] * q[k]
            // into two passes per head per token:
            //   pass A: compute delta_v (needs u_v), then update S in place.
            //   pass B: compute o_v from the updated S.
            // pass A reads S once and writes S once; pass B reads S once.
            // Total: 3 * d_v * d_k flops + d_v * d_k memory traffic / token.
            //
            // We use a small temp for delta; size <= 4 KiB at d_v=1024.
            // For d_v that fits on stack we could VLA, but a local static
            // std::vector is allocated once per call.

            // FLA / HF Qwen3.5 ordering: decay S FIRST, then compute u against
            // the decayed S, then add the delta-write. Without this ordering,
            // brotensor's recurrence diverges from HF's `torch_recurrent_gated
            // _delta_rule` at every token after the first by a factor that
            // depends on alpha, producing percent-level logit drift in
            // Qwen3.5-VL.
            //
            // pass A1 — decay S in place
            for (int v = 0; v < d_v; ++v) {
                float* Sv = S + v * d_k;
                for (int k = 0; k < d_k; ++k) Sv[k] *= alpha;
            }
            // pass A2 — compute u against the decayed S; stash delta in orow.
            float* orow = Op + t * Dv + h * d_v;
            for (int v = 0; v < d_v; ++v) {
                const float* Sv = S + v * d_k;
                float u_v = 0.0f;
                for (int k = 0; k < d_k; ++k) u_v += Sv[k] * kt[k];
                orow[v] = vt[v] - u_v;       // stash delta in orow
            }
            // pass A3 — add the delta-write: S += beta * delta_v * k_t.
            for (int v = 0; v < d_v; ++v) {
                float* Sv = S + v * d_k;
                const float scale_v = beta_t * orow[v]; // beta * delta_v
                for (int k = 0; k < d_k; ++k) {
                    Sv[k] += scale_v * kt[k];
                }
            }
            // pass B — o = S @ q (overwrites the stashed delta)
            for (int v = 0; v < d_v; ++v) {
                const float* Sv = S + v * d_k;
                float o_v = 0.0f;
                for (int k = 0; k < d_k; ++k) o_v += Sv[k] * qt[k];
                orow[v] = o_v;
            }
        }
    }
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

} // namespace brotensor::detail::cpu
