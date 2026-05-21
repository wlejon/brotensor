// ─── CPU AdaLN modulation ops ──────────────────────────────────────────────
//
// FP32 scalar host implementations of the DiT / SD3 / Flux broadcast-affine
// primitives. Ports src/cuda/modulate.cu — FP32 path only (the CPU backend is
// FP32-only per CLAUDE.md).
//
// ── modulate ──      Y[l, d] = X[l, d] * (1 + scale[d]) + shift[d]
// ── broadcast_mul ── Y[l, d] = X[l, d] * v[d]
//
// scale / shift / v are length-D vectors broadcast across every token row.
// Both ops fully OVERWRITE their output.

#include <brotensor/tensor.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t,
                       const char* op, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(std::string(op) + ": " + name +
                                 " must be FP32 (CPU backend is FP32-only)");
    }
}

} // namespace

void modulate(const ::brotensor::Tensor& X, const ::brotensor::Tensor& scale,
              const ::brotensor::Tensor& shift, ::brotensor::Tensor& Y) {
    check_fp32(X, "modulate", "X");
    check_fp32(scale, "modulate", "scale");
    check_fp32(shift, "modulate", "shift");
    const int L = X.rows;
    const int D = X.cols;
    if (scale.size() != D || shift.size() != D) {
        throw std::runtime_error("modulate: scale and shift must each have X.cols elements");
    }
    if (Y.rows != L || Y.cols != D || Y.dtype != Dtype::FP32) {
        Y.resize(L, D, Dtype::FP32);
    }
    if (L * D == 0) return;

    const float* Xp = X.host_f32();
    const float* sc = scale.host_f32();
    const float* sh = shift.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int l = 0; l < L; ++l) {
        const int base = l * D;
        for (int d = 0; d < D; ++d) {
            Yp[base + d] = Xp[base + d] * (1.0f + sc[d]) + sh[d];
        }
    }
}

void broadcast_mul(const ::brotensor::Tensor& X, const ::brotensor::Tensor& v,
                   ::brotensor::Tensor& Y) {
    check_fp32(X, "broadcast_mul", "X");
    check_fp32(v, "broadcast_mul", "v");
    const int L = X.rows;
    const int D = X.cols;
    if (v.size() != D) {
        throw std::runtime_error("broadcast_mul: v must have X.cols elements");
    }
    if (Y.rows != L || Y.cols != D || Y.dtype != Dtype::FP32) {
        Y.resize(L, D, Dtype::FP32);
    }
    if (L * D == 0) return;

    const float* Xp = X.host_f32();
    const float* vp = v.host_f32();
    float* Yp = Y.host_f32_mut();

    for (int l = 0; l < L; ++l) {
        const int base = l * D;
        for (int d = 0; d < D; ++d) {
            Yp[base + d] = Xp[base + d] * vp[d];
        }
    }
}

} // namespace brotensor::detail::cpu
