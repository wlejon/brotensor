// ─── CPU elementwise ops (CHUNK 1) ─────────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU elementwise kernels in
// src/cuda/elementwise.cu — kernel math reproduced verbatim, FP32 path only.
//
//   clamp        — in-place per-element clamp to [lo, hi].
//   mul_inplace  — in-place per-element multiply y *= x.

#include <brotensor/tensor.h>

#include <stdexcept>

namespace brotensor::detail::cpu {

void clamp(::brotensor::Tensor& y, float lo, float hi) {
    const int n = y.size();
    if (n == 0) return;
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        float v = yp[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        yp[i] = v;
    }
}

void mul_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x) {
    // Matches the GPU contract: shape/dtype must match exactly.
    if (y.dtype != x.dtype || y.rows != x.rows || y.cols != x.cols) {
        throw std::runtime_error("mul_inplace: shape/dtype mismatch");
    }
    const int n = y.size();
    if (n == 0) return;
    float* yp = y.host_f32_mut();
    const float* xp = x.host_f32();
    for (int i = 0; i < n; ++i) yp[i] *= xp[i];
}

} // namespace brotensor::detail::cpu
