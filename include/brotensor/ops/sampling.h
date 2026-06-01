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

}  // namespace brotensor
