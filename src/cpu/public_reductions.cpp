// ─── CPU public reductions (CHUNK 1) ───────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU kernels in
// src/cuda/public_reductions.cu — FP32 path only.
//
//   sum_rows    — per-row sum; out is (M, 1).
//   sum_cols    — per-column sum; out is (1, N).
//   argmax_rows — per-row argmax index, written as FP32; out is (M, 1).
//
// GPU argmax: ties keep the lowest index (strict `v > best_v`); the scalar
// loop here matches that.

#include <brotensor/tensor.h>

#include <cstddef>
#include <cstdint>

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

} // namespace brotensor::detail::cpu
