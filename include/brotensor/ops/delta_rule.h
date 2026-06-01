#pragma once

// brotensor ops/delta_rule.h — Gated Delta Rule (linear attention): chunked prefill + streaming step.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// Gated Delta Rule — chunked prefill. Runs the matrix-valued recurrence
// (FLA / HF Qwen3.5 ordering: decay BEFORE the delta read):
//   alpha_t = exp(-softplus(a_raw_t) * exp(log_A))      (per token, per head)
//   beta_t  = sigmoid(beta_raw_t)
//   S_pre_t = alpha_t * S_{t-1}
//   S_t     = S_pre_t + beta_t * (v_t - S_pre_t k_t) k_t^T
//   o_t     = S_t q_t                                  (per head)
// over L tokens, sequentially within each head. The chunked WY/UT-transform
// is an internal optimisation — the contract is exactly the per-token rule.
//   Q, K: (L, num_heads*d_k).      V: (L, num_heads*d_v).
//         Heads contiguous within each row, exactly as rope_forward / rms_norm.
//   a_raw, beta: (L, num_heads) FP32 — per-token gate / write inputs (raw).
//                softplus / sigmoid are applied inside the op.
//   log_A: (num_heads, 1) FP32 — per-head learnable decay scale.
//   state: (num_heads, d_v*d_k) FP32 — initial S per head (caller zero-fills
//          for a fresh sequence; row h is S_h[v, k] = state[h, v*d_k + k]).
//          Read AND updated in place; on return holds S after token L-1.
//   O: (L, num_heads*d_v) — output, resized + dtype-set to match Q.
// Q/K/V/O are dispatched on Q.dtype (FP32 on CPU; FP16 or FP32 on GPU). state,
// a_raw, beta, log_A are FP32 on every backend (accumulator + gate precision).
// num_heads*d_k must equal Q.cols and K.cols; num_heads*d_v must equal V.cols
// and O.cols. d_k and d_v may differ. FP32 accumulation. Forward-only.
void gated_delta_rule_chunked(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& a_raw, const Tensor& beta,
                              const Tensor& log_A,
                              int num_heads, int d_k, int d_v,
                              Tensor& state, Tensor& O);


// Gated Delta Rule — streaming step. Same math as gated_delta_rule_chunked
// but for L_step new tokens against an existing state. With L_step == 1 this
// is the per-step recurrence; with L_step > 1 it's a plain non-chunked scan
// (correct, just without the WY/UT speedup).
//   Q, K: (L_step, num_heads*d_k).    V: (L_step, num_heads*d_v).
//   a_raw, beta: (L_step, num_heads) FP32.   log_A: (num_heads, 1) FP32.
//   state: (num_heads, d_v*d_k) FP32 — read AND overwritten with S after the
//          last new token, ready for the next call.
//   O: (L_step, num_heads*d_v), resized + dtype-set to match Q.
// Dtype rules identical to gated_delta_rule_chunked.
void gated_delta_rule_step(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& a_raw, const Tensor& beta,
                           const Tensor& log_A,
                           int num_heads, int d_k, int d_v,
                           Tensor& state, Tensor& O);

}  // namespace brotensor
