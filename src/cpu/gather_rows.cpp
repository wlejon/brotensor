// ─── CPU row gather + scatter-add ───────────────────────────────────────────
//
// FP32 scalar host implementation. General "look up these rows" — the same
// compute as embedding_lookup_forward but taking Idx as a Tensor (so it
// routes through the dispatcher and works on any backend). Used by SAM's
// prompt encoder (point indices -> positional embeddings), DETR object
// queries, per-row attention reorderings, etc.
//
//   gather_rows       : Y[m, :] = X[Idx[m], :]
//   scatter_rows_add  : dX[Idx[m], :] += dY[m, :]  (with dX zeroed first;
//                       duplicate indices in Idx accumulate — adjoint of
//                       gather_rows).
//
// Idx layout: (M, 1) INT32, matching argmax_rows / top_k_rows' INT32 idx
// conventions. M is read from Idx.rows (cols must be 1).
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   gather_rows      — Y  OVERWRITTEN (pure copy from X).
//   scatter_rows_add — dX OVERWRITTEN (zeroed, then scatter-add).
//
// Bounds: an OOB Idx value triggers undefined behavior on all backends.
// The caller owns the precondition (same as embedding_lookup).

#include <brotensor/tensor.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) +
                 " must be FP32 (CPU backend is FP32-only)");
    }
}

inline void check_idx(const ::brotensor::Tensor& Idx, const char* op) {
    if (Idx.dtype != Dtype::INT32) {
        fail(op, "Idx must be INT32");
    }
    if (Idx.cols != 1) {
        fail(op, "Idx must be shaped (M, 1)");
    }
}

} // namespace

void gather_rows(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Idx,
                 ::brotensor::Tensor& Y) {
    const char* op = "gather_rows";
    check_fp32(X, op, "X");
    check_idx(Idx, op);
    const int M = Idx.rows;
    const int C = X.cols;
    if (Y.rows != M || Y.cols != C || Y.dtype != Dtype::FP32) {
        Y.resize(M, C, Dtype::FP32);
    }
    if (M == 0 || C == 0) return;

    const float* Xp = X.host_f32();
    const int32_t* Ip = static_cast<const int32_t*>(Idx.data);
    float* Yp = Y.host_f32_mut();

    for (int m = 0; m < M; ++m) {
        const int r = Ip[m];
        std::memcpy(Yp + static_cast<long>(m) * C,
                    Xp + static_cast<long>(r) * C,
                    static_cast<size_t>(C) * sizeof(float));
    }
}

void scatter_rows_add(const ::brotensor::Tensor& dY,
                      const ::brotensor::Tensor& Idx, int R,
                      ::brotensor::Tensor& dX) {
    const char* op = "scatter_rows_add";
    check_fp32(dY, op, "dY");
    check_idx(Idx, op);
    if (R < 0) fail(op, "R must be >= 0");
    const int M = Idx.rows;
    if (dY.rows != M) {
        fail(op, "dY.rows must equal Idx.rows");
    }
    const int C = dY.cols;
    if (dX.rows != R || dX.cols != C || dX.dtype != Dtype::FP32) {
        dX.resize(R, C, Dtype::FP32);
    }
    if (R == 0 || C == 0) return;

    const float* dYp = dY.host_f32();
    const int32_t* Ip = static_cast<const int32_t*>(Idx.data);
    float* dXp = dX.host_f32_mut();

    const long total = static_cast<long>(R) * C;
    for (long i = 0; i < total; ++i) dXp[i] = 0.0f;

    for (int m = 0; m < M; ++m) {
        const int r = Ip[m];
        const float* src = dYp + static_cast<long>(m) * C;
        float* dst = dXp + static_cast<long>(r) * C;
        for (int c = 0; c < C; ++c) dst[c] += src[c];
    }
}

} // namespace brotensor::detail::cpu
