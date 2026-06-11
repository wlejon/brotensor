// ─── CPU public reductions (CHUNK 1) ───────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU kernels in
// src/cuda/public_reductions.cu — FP32 path only.
//
//   sum_rows         — per-row sum; out is (M, 1).
//   sum_cols         — per-column sum; out is (1, N).
//   argmax_rows      — per-row argmax index, written as FP32; out is (M, 1).
//   rows_count_above — per-row strict-> counts at two thresholds; counts is
//                      (R, 2) INT32. Accepts FP32 or FP16 X (matches CUDA).
//
// GPU argmax: ties keep the lowest index (strict `v > best_v`); the scalar
// loop here matches that.

#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cpu {

void sum_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != M || Y.cols != 1 || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(M, 1, ::brotensor::Dtype::FP32);
    }
    if (M == 0) return;
    if (N == 0) { Y.zero(); return; }
    const float* xp = X.host_f32();
    float* yp = Y.host_f32_mut();
    for (int m = 0; m < M; ++m) {
        const float* row = xp + static_cast<std::size_t>(m) * N;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) acc += row[n];
        yp[m] = acc;
    }
}

void sum_cols(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y) {
    const int M = X.rows;
    const int N = X.cols;
    if (Y.rows != 1 || Y.cols != N || Y.dtype != ::brotensor::Dtype::FP32) {
        Y.resize(1, N, ::brotensor::Dtype::FP32);
    }
    if (N == 0) return;
    if (M == 0) { Y.zero(); return; }
    const float* xp = X.host_f32();
    float* yp = Y.host_f32_mut();
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int m = 0; m < M; ++m) acc += xp[static_cast<std::size_t>(m) * N + n];
        yp[n] = acc;
    }
}

// Output dtype is opt-in (mirrors the CUDA contract): an INT32-typed `Idx`
// receives the index as int32; otherwise it is written FP32 (the default).
void argmax_rows(const ::brotensor::Tensor& X, ::brotensor::Tensor& Idx) {
    using ::brotensor::Dtype;
    const int M = X.rows;
    const int N = X.cols;
    const Dtype out_dt = (Idx.dtype == Dtype::INT32) ? Dtype::INT32 : Dtype::FP32;
    if (Idx.rows != M || Idx.cols != 1 || Idx.dtype != out_dt) {
        Idx.resize(M, 1, out_dt);
    }
    if (M == 0) return;
    if (N == 0) { Idx.zero(); return; }
    const float* xp = X.host_f32();
    float*    fp = (out_dt == Dtype::FP32) ? Idx.host_f32_mut() : nullptr;
    int32_t*  ip = (out_dt == Dtype::INT32)
                       ? static_cast<int32_t*>(Idx.host_raw_mut()) : nullptr;
    for (int m = 0; m < M; ++m) {
        const float* row = xp + static_cast<std::size_t>(m) * N;
        float best_v = -3.4028235e38f;
        int   best_i = 0;
        for (int n = 0; n < N; ++n) {
            if (row[n] > best_v) { best_v = row[n]; best_i = n; }
        }
        if (ip) ip[m] = best_i;
        else    fp[m] = static_cast<float>(best_i);
    }
}

// One pass per row over both thresholds. Strict >: a value exactly at a
// threshold is not counted.
void rows_count_above(const ::brotensor::Tensor& X, float t_lo, float t_hi,
                      ::brotensor::Tensor& counts) {
    using ::brotensor::Dtype;
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16) {
        throw std::runtime_error("rows_count_above: X must be FP32 or FP16");
    }
    const int R = X.rows;
    const int C = X.cols;
    if (counts.rows != R || counts.cols != 2 || counts.dtype != Dtype::INT32) {
        counts.resize(R, 2, Dtype::INT32);
    }
    if (R == 0) return;
    int32_t* cp = static_cast<int32_t*>(counts.host_raw_mut());
    const float* xf = (X.dtype == Dtype::FP32) ? X.host_f32() : nullptr;
    const uint16_t* xh = (X.dtype == Dtype::FP16) ? X.host_fp16() : nullptr;
    for (int r = 0; r < R; ++r) {
        int32_t n_lo = 0, n_hi = 0;
        const std::size_t base = static_cast<std::size_t>(r) * C;
        for (int c = 0; c < C; ++c) {
            const float v = xf ? xf[base + c]
                               : ::brotensor::fp16_bits_to_fp32(xh[base + c]);
            n_lo += (v > t_lo) ? 1 : 0;
            n_hi += (v > t_hi) ? 1 : 0;
        }
        cp[2 * r + 0] = n_lo;
        cp[2 * r + 1] = n_hi;
    }
}

} // namespace brotensor::detail::cpu
