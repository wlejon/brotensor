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

}  // namespace brotensor
