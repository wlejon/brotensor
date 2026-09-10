#pragma once

// brotensor ops/sampling.h — Logit sampling + Philox RNG (randn / uniform / bernoulli / truncated).

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// ─── Autoregressive logit sampling ─────────────────────────────────────────
//
// FP32-only, implemented on all three backends (CPU / CUDA / Metal). The
// next-token sampler for autoregressive generation loops — a general LLM /
// codec-LM sampler, not audio-specific.

// Draw one token id per row of an (N, V) logit matrix (N independent streams,
// V = vocabulary size), applying, in order:
//   1. temperature   logit /= temperature
//   2. softmax       p = softmax(logit)
//   3. top-k         (top_k > 0) keep the top_k highest-p tokens, rest -> 0
//   4. top-p         (top_p < 1) keep the smallest highest-p set with cumulative
//                    probability >= top_p, rest -> 0 (applied after top-k)
//   5. renormalize   kept probabilities rescaled to sum to 1
//   6. draw          inverse-CDF lookup of a uniform u in [0,1)
// Greedy: temperature == 0 is deterministic argmax — steps 2-6 skipped, no RNG
// consumed, ties keep the lowest index. top_k == 1 is likewise deterministic.
//
// RNG: counter-based Philox 4x32-10, seeded by two scalar args (so dispatch
// resolves on `logits`):
//   key     — 64-bit run seed.
//   counter — 64-bit base counter offset.
// Row n draws from the Philox counter block for the 64-bit value (counter + n),
// converting the first output word to a uniform via its top 24 bits / 2^24, so
// row n's draw depends only on (key, counter + n) — reproducible, independent
// of N and row order. To get fresh draws across decode steps, advance `counter`
// by the rows sampled so far and keep `key` fixed.
//
// Metal's FP32 reductions are not bit-identical to the CPU op's FP64
// accumulators, so a draw landing within a few ulp of a CDF-bucket boundary may
// pick a different token; with well-separated logits the backends agree.
//
//   logits:  (N,V) FP32 input.
//   indices: (N,1) INT32 output — resized + dtype-set to INT32.
// Throws ("brotensor: sample_logits: <reason>") for temperature < 0, top_k < 0,
// top_p < 0, or V == 0 while N > 0. No backward.
void sample_logits(const Tensor& logits, float temperature, int top_k,
                   float top_p, uint64_t key, uint64_t counter,
                   Tensor& indices);


// ─── Graph-capturable autoregressive sampler ───────────────────────────────
//
// Same draw as sample_logits (byte-identical for the same effective base
// counter), but written so a whole autoregressive decode step can be recorded
// into a CUDA graph and replayed with a single launch. Three changes make it
// capture-safe:
//
//   counter — a device tensor (>= 1 element, INT32) holding the base counter as
//             counter[0]. Row n draws from substream (counter[0] + n); the op
//             then advances counter[0] += N entirely on-device. So a captured
//             replay reads the value left by the previous step and advances it
//             again — fresh draws every launch with NO host involvement (the
//             host never reads or writes the counter). Mirrors the device-side
//             Philox offset PyTorch uses for graph-safe RNG.
//   scratch — caller-owned reusable workspace, FP32, with at least 3*N*V
//             elements (carved into prob / sort-work / order). No per-call
//             cudaMalloc/cudaFree — those are illegal during capture and force
//             a device sync otherwise. Sized once and reused across steps.
//   indices — (N,1) INT32, MUST be pre-sized by the caller; written in place,
//             never resized (a resize would allocate mid-capture).
//
// temperature == 0 is the deterministic argmax (no RNG, counter untouched), so
// the same buffer set works for a greedy step too. Validation mirrors
// sample_logits; additionally throws if counter is not INT32 with >= 1 element,
// if scratch has fewer than 3*N*V FP32 elements, or if indices is not a
// pre-sized (N,1) INT32 on the logits' device. No backward.
void sample_logits_into(const Tensor& logits, float temperature, int top_k,
                        float top_p, uint64_t key, Tensor& counter,
                        Tensor& scratch, Tensor& indices);


// ─── Counter-based noise generation (Philox 4x32-10) ───────────────────────
//
// PyTorch/JAX-compatible Philox 4x32-10 stream. (key, counter) seeds the
// stream; element i (row-major linear index) is drawn from substream
// (counter + i), so the result is reproducible and parallel-safe across
// backends. The CPU, CUDA and Metal implementations all use the same
// Philox construction byte-for-byte (see src/cpu/sample_logits.cpp).
//
// All four ops require Y FP32 and pre-sized to the desired (rows, cols);
// no resize is performed.

// Standard normal N(0, 1). One Philox call per element: ctr[0]/ctr[1] form
// (u1, u2) for one Box-Muller pair; ctr[2..3] are discarded so the per-
// element substream mapping stays trivial. Use `counter += rows*cols` to
// advance the stream past a generated tensor.
void randn(uint64_t key, uint64_t counter, Tensor& Y);


// Uniform U[0, 1). Top 24 bits of ctr[0] / 2^24 — identical to the uniform
// used inside randn / sample_logits.
void rand_uniform(uint64_t key, uint64_t counter, Tensor& Y);


// Bernoulli mask: Y[i] = (uniform < p) ? 1.0 : 0.0. p must be in [0, 1].
// Useful for dropout / stochastic-depth masks.
void rand_bernoulli(float p, uint64_t key, uint64_t counter, Tensor& Y);


// Truncated standard normal in [lo, hi] (lo < hi required). Rejection
// sampling on top of Box-Muller; per element, retries advance the substream
// by rows*cols (so each element's retry stream is independent of every
// other element's). Capped at 64 retries — for any interval covering at
// least a couple percent of mass that's failure prob ~10^-50; the final
// sample is clamped to [lo, hi] as a last-resort safety net.
void randn_truncated(float lo, float hi,
                     uint64_t key, uint64_t counter,
                     Tensor& Y);


// ─── Masked-diffusion token selection (OmniVoice-style codebook grids) ─────
//
// One step of a masked-diffusion language model over a (C codebooks, T frames)
// token grid with vocabulary V, where id `mask_id` marks a still-masked cell.
// Two ops split the step: `masked_diffusion_scores` fuses classifier-free
// guidance, log-softmax, per-cell prediction and the confidence score every
// cell competes with; the host then picks the k best cells (top_k_rows over
// the scores viewed as one (1, C*T) row) and `masked_diffusion_commit` writes
// the chosen predictions back. FP32-only, implemented on CPU, CUDA and Metal.
//
// Math per cell (c, t), with c_logits / u_logits its conditional and
// unconditional logit rows (upstream: OmniVoice._predict_tokens_with_scoring):
//   if guidance_scale != 0:
//       c = log_softmax(c_logits); u = log_softmax(u_logits)
//       log_probs = log_softmax(c + guidance_scale * (c - u))
//   else:
//       log_probs = log_softmax(c_logits)
//   log_probs[mask_id] = -inf
//   if class_temperature > 0:
//       k = ceil(class_top_frac * V), clamped to [1, V]
//       filtered = log_probs with all but its k largest entries set to -inf
//                  (ties at the k-th value keep the lower vocabulary index)
//       pred = argmax(filtered / class_temperature + gumbel(u_class[v]))
//   else:
//       pred = argmax(log_probs)                       (ties: lowest index)
//   confidence = max(log_probs)                        (UNfiltered, always)
//   score = confidence - c * layer_penalty
//   if position_temperature > 0:
//       score = score / position_temperature + gumbel(u_pos)
//   score = -inf where tokens[c, t] != mask_id         (already decided)
//   gumbel(u) = -log(-log(u + 1e-10) + 1e-10), u ~ U[0, 1)
//
// Noise: the uniforms are a counter-based hash (detail/hash_rng.h,
// splitmix64) of (seed, cell index c*T + t) for u_pos and of (seed, cell
// index * V + v) for u_class, under two distinct domain constants — so the
// CPU and CUDA backends draw bit-identical noise for the same `seed`, and the
// caller gets fresh noise by varying `seed` per step. Nothing is consumed or
// advanced; the same (inputs, seed) always yields the same result.
//
// Numerics: log-sum-exp accumulates in FP64 on CPU and CUDA (Metal, which has
// no FP64, accumulates in FP32); everything else is FP32 with the same
// operation order on every backend, so CPU and CUDA agree to the last ulp
// except where their libm exp/log differ, and `pred` can only diverge on an
// exact FP32 near-tie.
//
//   logits: (R, C*V) FP32; column c*V + v. R == 2*T when guidance_scale != 0
//           (rows [0, T) conditional, rows [T, 2T) unconditional), R == T
//           otherwise.
//   tokens: (C, T) INT32 current grid; cell (c, t) is at c*T + t and is masked
//           iff tokens == mask_id.
//   pred:   (C, T) INT32 output, resized + dtype-set. The predicted id for
//           every cell (never mask_id unless the whole row is -inf).
//   scores: (C, T) FP32 output, resized + dtype-set. -inf where tokens !=
//           mask_id.
// Throws ("brotensor: masked_diffusion_scores: <reason>") for a non-FP32
// logits / non-INT32 tokens, T/C/V < 1, mask_id outside [0, V), or a shape
// that does not match (T, C, V, guidance_scale).
void masked_diffusion_scores(const Tensor& logits, const Tensor& tokens,
                             int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty,
                             float position_temperature, float class_temperature,
                             float class_top_frac, std::uint64_t seed,
                             Tensor& pred, Tensor& scores);


// Commit the k selected cells: for i in [0, k): p = idx[i]; tokens[p] =
// pred[p]; unmask_step[p] = step. `idx` holds flat cell indices c*T + t into
// the (C, T) grids — the layout top_k_rows returns over `scores` viewed as
// (1, C*T). tokens / unmask_step / pred are (C, T) INT32 (all pre-sized,
// nothing is resized); idx is INT32 with at least k elements, of which only
// the first k are read. k == 0 is a no-op. An index outside [0, C*T) is
// ignored on every backend (a device kernel cannot throw), so the host should
// only ever pass what top_k_rows produced. Throws for a dtype / shape / k
// mismatch.
void masked_diffusion_commit(const Tensor& pred, const Tensor& idx, int k, int step,
                             Tensor& tokens, Tensor& unmask_step);

}  // namespace brotensor
