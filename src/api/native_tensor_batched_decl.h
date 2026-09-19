#pragma once

// native_tensor_batched_decl.h — the C entry points of the restored "batched"
// group: the B-row batched dense / activation forwards and backwards, the row
// gather / scatter / top-k primitives, and batched LayerNorm with its training
// caches. Bodies in native_tensor_batched.cpp, JS wrappers in
// js/tensor_batched.js, registration table at the bottom of the .cpp.
//
// The JS signatures in the comments are the QuickJS binding's
// (bro-quickjs-oracle src/js/tensor_bindings.cpp) — same argument order, same
// optionality. A `|null` slot crosses as `dynamic` (the raw Value bits).

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Batched forwards ------------------------------------------------------

// linearForwardBatched(W, bias, X_BD, Y_BD)
void bro_tensor_linearForwardBatched(void* W, void* bias, void* X_BD, void* Y_BD);
// linearForwardBatchedFp16(W, bias|null, X_BD, Y_BD) — 16-bit storage
// throughout (FP16 or BF16); GPU-only, the CPU backend leaves the slot null.
void bro_tensor_linearForwardBatchedFp16(void* W, uint64_t bias_bits, void* X_BD, void* Y_BD);
// reluForwardBatched(X_BD, Y_BD)
void bro_tensor_reluForwardBatched(void* X_BD, void* Y_BD);
// tanhForwardBatched(X_BD, Y_BD)
void bro_tensor_tanhForwardBatched(void* X_BD, void* Y_BD);
// addInplaceBatched(Y_BD, X_BD)
void bro_tensor_addInplaceBatched(void* Y_BD, void* X_BD);

// --- Batched-training backwards -------------------------------------------

// linearBackwardBatched(W, X_BD, dY_BD, dX_BD, dW, dB) — dW/dB accumulate.
void bro_tensor_linearBackwardBatched(void* W, void* X_BD, void* dY_BD, void* dX_BD, void* dW, void* dB);
// reluBackwardBatched(X_BD, dY_BD, dX_BD) — reads the forward *input*.
void bro_tensor_reluBackwardBatched(void* X_BD, void* dY_BD, void* dX_BD);
// tanhBackwardBatched(Y_BD, dY_BD, dX_BD) — reads the forward *output*.
void bro_tensor_tanhBackwardBatched(void* Y_BD, void* dY_BD, void* dX_BD);

// --- row gather / scatter / top-k ------------------------------------------

// gatherRows(X, Idx, Y): Y[m,:] = X[Idx[m],:]. Idx is an (M,1) index tensor.
void bro_tensor_gatherRows(void* X, void* Idx, void* Y);
// scatterRowsAdd(dY, Idx, R, dX): dX is (R,C), zeroed then scatter-added.
void bro_tensor_scatterRowsAdd(void* dY, void* Idx, int32_t R, void* dX);
// topKRows(X, k, Vals, Idx): Vals (R,k) FP32 descending, Idx (R,k) INT32.
void bro_tensor_topKRows(void* X, int32_t k, void* Vals, void* Idx);

// --- batched LayerNorm with training caches --------------------------------

// layernormForwardBatchedWithCaches(X, gamma, beta, Y, Xhat, Mean, Rstd, eps)
void bro_tensor_layernormForwardBatchedWithCaches(void* X, void* gamma, void* beta, void* Y, void* Xhat, void* Mean, void* Rstd, double eps);
// layernormBackwardBatchedWithCaches(dY, Xhat, gamma, Rstd, dX, dGamma, dBeta)
// — dGamma/dBeta accumulate into (D,) tensors the caller sized and zeroed.
void bro_tensor_layernormBackwardBatchedWithCaches(void* dY, void* Xhat, void* gamma, void* Rstd, void* dX, void* dGamma, void* dBeta);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_batched(std::string* error); }
#endif
