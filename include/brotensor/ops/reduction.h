#pragma once

// brotensor ops/reduction.h — Reductions: sum_rows, sum_cols, argmax_rows, top_k_rows.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// Per-row top-k. For each row of X: select the k largest values, returning
// them in descending order in `Vals` with their column indices in `Idx`.
// Ties broken by smaller column index. The companion to argmax_rows for
// classification heads (top-5), NMS pre-filter, and beam-search candidates.
//   X:    (R, C) FP32.
//   Vals: (R, k) FP32 (resized + dtype-set).
//   Idx:  (R, k) INT32 (resized + dtype-set).
// k > C or k < 1 throws. Not differentiable — no backward.
void top_k_rows(const Tensor& X, int k, Tensor& Vals, Tensor& Idx);


// ─── Public reductions ─────────────────────────────────────────────────────

// Row-wise sum: Y[m,0] = sum_n X[m,n]. X:(M,N), Y:(M,1) — same dtype
// (FP32/FP16), resized as needed.
void sum_rows(const Tensor& X, Tensor& Y);


// Column-wise sum: Y[0,n] = sum_m X[m,n]. X:(M,N), Y:(1,N) — same dtype
// (FP32/FP16), resized as needed.
void sum_cols(const Tensor& X, Tensor& Y);


// Row-wise argmax: Idx[m,0] = argmax_n X[m,n]. X:(M,N) FP32/FP16/BF16. Ties
// keep the lowest index. The output dtype is opt-in: pass an INT32-typed `Idx`
// to get the index as a device int32 (directly consumable as a gather index —
// no host round-trip, the AR-decode hot path); any other `Idx` (e.g. a fresh
// default tensor) yields the index as FP32. `Idx` is resized to (M,1) of the
// selected dtype.
void argmax_rows(const Tensor& X, Tensor& Idx);


// Per-row above-threshold counts at two thresholds in one pass:
//   counts[r][0] = #{ c : X[r][c] > t_lo }
//   counts[r][1] = #{ c : X[r][c] > t_hi }
// Strict >: elements exactly at a threshold are NOT counted. Backs SAM AMG's
// device-side stability score (intersection/union of the mask at logit
// thresholds +/- the stability offset) without downloading the logits.
//   X:      (R, C) FP32 or FP16.
//   counts: (R, 2) INT32, resized + dtype-set. Not differentiable.
void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts);


// Softmax summary statistics of variable-length segments of a logit column —
// the confidence features a decision head reads off each item's option scores
// without a host round-trip. Segment s covers logits rows
// [seg_offsets[s], seg_offsets[s+1]); with p = softmax(segment) (FP32) and
// k = max(2, segment length):
//   out[s] = [ top1, top1 - top2, -sum p*log(max(p,1e-9)) / log(k), k / 255 ]
// top2 is 0 for a one-element segment; an empty segment writes zeros.
//   logits: (K, 1) FP32/FP16/BF16.  seg_offsets: (S+1, 1) INT32, non-decreasing,
//   last entry <= K.  out: (S, 4), same dtype as logits, resized as needed.
void segment_softmax_stats(const Tensor& logits, const Tensor& seg_offsets,
                           Tensor& out);

}  // namespace brotensor
