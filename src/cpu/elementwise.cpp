// ─── CPU elementwise ops (CHUNK 1) ─────────────────────────────────────────
//
// FP32 scalar host implementations. Ports the GPU elementwise kernels in
// src/cuda/elementwise.cu — kernel math reproduced verbatim, FP32 path only.
//
//   clamp        — in-place per-element clamp to [lo, hi].
//   mul_inplace  — in-place per-element multiply y *= x.

#include <brotensor/tensor.h>

#include <cstdint>
#include <cstring>
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

void cast(const ::brotensor::Tensor& src, ::brotensor::Tensor& dst,
          ::brotensor::Dtype out_dtype) {
    using ::brotensor::Dtype;
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != out_dtype) {
        dst.resize(src.rows, src.cols, out_dtype);
    }
    const int n = src.size();
    if (n == 0) return;
    if (src.dtype == out_dtype) {
        std::memcpy(dst.host_raw_mut(), src.host_raw(), src.bytes());
        return;
    }
    if (src.dtype == Dtype::FP32 && out_dtype == Dtype::FP16) {
        const float* s = src.host_f32();
        std::uint16_t* d = dst.host_fp16_mut();
        for (int i = 0; i < n; ++i) d[i] = ::brotensor::fp32_to_fp16_bits(s[i]);
    } else if (src.dtype == Dtype::FP16 && out_dtype == Dtype::FP32) {
        const std::uint16_t* s = src.host_fp16();
        float* d = dst.host_f32_mut();
        for (int i = 0; i < n; ++i) d[i] = ::brotensor::fp16_bits_to_fp32(s[i]);
    } else if (src.dtype == Dtype::FP32 && out_dtype == Dtype::BF16) {
        const float* s = src.host_f32();
        std::uint16_t* d = dst.host_bf16_mut();
        for (int i = 0; i < n; ++i) d[i] = ::brotensor::fp32_to_bf16_bits(s[i]);
    } else if (src.dtype == Dtype::BF16 && out_dtype == Dtype::FP32) {
        const std::uint16_t* s = src.host_bf16();
        float* d = dst.host_f32_mut();
        for (int i = 0; i < n; ++i) d[i] = ::brotensor::bf16_bits_to_fp32(s[i]);
    } else {
        throw std::runtime_error(
            "brotensor: cast: unsupported dtype pair "
            "(CPU supports FP32<->FP16 and FP32<->BF16)");
    }
}

} // namespace brotensor::detail::cpu
