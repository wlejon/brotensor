#pragma once

// brotensor ops/embedding.h — Embedding / gather-scatter row lookups.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// Embedding lookup: out[b,:] = table[d_idx[b],:].
//   table: (V,D) FP32 or FP16.  d_idx: B int32 indices, each in [0,V).
//   out: (B,D), resized AND dtype-set to match table if mis-shaped/-typed.
void embedding_lookup_forward(const Tensor& table,
                              const int32_t* d_idx, int B,
                              Tensor& out);


// Scatter-accumulate backward of embedding_lookup. Dtype-dispatched (FP32/FP16);
// dOut and dTable share dtype.
//   dOut: (B,D) upstream.  d_idx: the B forward indices.
//   dTable: (V,D) accumulated — caller zeros; repeated indices sum.
void embedding_lookup_backward(const Tensor& dOut,
                               const int32_t* d_idx, int B,
                               Tensor& dTable);


// Gather rows of X by index (the general "look up these rows of this 2D
// tensor" op). Same compute as embedding_lookup_forward but takes Idx as
// a Tensor instead of a raw pointer + length, so it routes through the
// normal dispatcher and is usable on any backend. Used by SAM's prompt
// encoder (point indices → positional embeddings), DETR object queries,
// per-row attention reorderings, etc.
//   X:   (R, C) FP32.
//   Idx: (M, 1) INT32 with values in [0, R).
//   Y:   (M, C) FP32, resized + dtype-set.
// An OOB Idx value triggers undefined behavior — caller owns the bounds.
// (Validating each index on the CPU would dominate the run time and the
// GPU port can't afford it; aligns with embedding_lookup's contract.)
void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y);


// Adjoint of gather_rows: scatter-add each dY row into the dX row named by
// Idx. dX is OVERWRITTEN — it's zeroed first, then dY rows are summed onto
// their dX[Idx[m]] target. Duplicate indices in Idx accumulate (the same
// behavior as embedding_lookup_backward). R is the output's row count
// (i.e. the original X.rows from the forward call — not derivable from
// dY or Idx alone).
//   dY:  (M, C) FP32.
//   Idx: (M, 1) INT32.
//   dX:  (R, C) FP32, resized + dtype-set.
void scatter_rows_add(const Tensor& dY, const Tensor& Idx, int R, Tensor& dX);

}  // namespace brotensor
