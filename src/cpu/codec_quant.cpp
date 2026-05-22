// ─── CPU codec quantization ops (brosoundml CHUNK 5, family D) ──────────────
//
// FP32 scalar host implementations of the quantization bottlenecks of neural
// audio codecs:
//   vq_encode_forward / vq_encode_backward   — vector quantization (EnCodec /
//                                              DAC residual-VQ encode step)
//   fsq_quantize_forward / fsq_quantize_backward — finite scalar quantization
//                                              (NanoCodec)
//
// ── INT32 outputs ───────────────────────────────────────────────────────────
//   vq_encode_forward.indices         — (N, 1) INT32 codeword indices.
//   fsq_quantize_forward.packed_indices — (N, 1) INT32 mixed-radix codes.
//   INT32 is a pure storage carrier; these tensors are resized AND dtype-set
//   to INT32, and accessed via the dtype-agnostic host_raw / host_raw_mut
//   accessors cast to int32_t* (host_f32 throws on a non-FP32 dtype).
//
// ── Accumulation ────────────────────────────────────────────────────────────
//   *_forward                — all outputs OVERWRITTEN.
//   vq_encode_backward       — dX OVERWRITTEN (straight-through identity).
//   fsq_quantize_backward    — dX OVERWRITTEN (straight-through identity).
//   Neither backward accumulates and neither produces a codebook gradient —
//   they are purely the encoder STE passthrough. See ops.h for the rationale.
//
// CPU is FP32-only; all arithmetic is FP32.

#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void require_fp32(const char* op, const ::brotensor::Tensor& t,
                  const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CPU backend is FP32-only)");
    }
}

void require_int32(const char* op, const ::brotensor::Tensor& t,
                   const char* name) {
    if (t.dtype != ::brotensor::Dtype::INT32) {
        fail(op, std::string(name) + " must be INT32");
    }
}

} // namespace

// ─── vq_encode ──────────────────────────────────────────────────────────────

void vq_encode_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& codebook,
                       ::brotensor::Tensor& indices,
                       ::brotensor::Tensor& quantized) {
    require_fp32("vq_encode_forward", x, "x");
    require_fp32("vq_encode_forward", codebook, "codebook");
    const int N = x.rows;
    const int D = x.cols;
    const int K = codebook.rows;
    if (codebook.cols != D) {
        fail("vq_encode_forward", "codebook must have the same column count as x");
    }
    if (K == 0 && N != 0) {
        fail("vq_encode_forward", "codebook must have at least one codeword");
    }

    // indices: (N, 1) INT32 — resize AND dtype-set.
    if (indices.rows != N || indices.cols != 1 ||
        indices.dtype != ::brotensor::Dtype::INT32) {
        indices.resize(N, 1, ::brotensor::Dtype::INT32);
    }
    // quantized: (N, D) FP32.
    if (quantized.rows != N || quantized.cols != D ||
        quantized.dtype != ::brotensor::Dtype::FP32) {
        quantized.resize(N, D, ::brotensor::Dtype::FP32);
    }
    if (N == 0) return;

    const float* xp = x.host_f32();
    const float* cp = codebook.host_f32();
    int32_t* ip = static_cast<int32_t*>(indices.host_raw_mut());
    float*   qp = quantized.host_f32_mut();

    for (int n = 0; n < N; ++n) {
        const float* x_row = xp + static_cast<std::size_t>(n) * D;
        float best_d2 = 3.4028235e38f;   // +FLT_MAX
        int   best_k  = 0;
        for (int k = 0; k < K; ++k) {
            const float* c_row = cp + static_cast<std::size_t>(k) * D;
            float d2 = 0.0f;
            for (int j = 0; j < D; ++j) {
                const float diff = x_row[j] - c_row[j];
                d2 += diff * diff;
            }
            // Strict `<` keeps the lowest index on ties.
            if (d2 < best_d2) { best_d2 = d2; best_k = k; }
        }
        ip[n] = static_cast<int32_t>(best_k);
        const float* c_best = cp + static_cast<std::size_t>(best_k) * D;
        float* q_row = qp + static_cast<std::size_t>(n) * D;
        for (int j = 0; j < D; ++j) q_row[j] = c_best[j];
    }
}

void vq_encode_backward(const ::brotensor::Tensor& dQuantized,
                        ::brotensor::Tensor& dX) {
    require_fp32("vq_encode_backward", dQuantized, "dQuantized");
    // Straight-through estimator: the argmin is non-differentiable, so the
    // gradient passes through unchanged. dX is OVERWRITTEN, not accumulated.
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(dQuantized.rows, dQuantized.cols, ::brotensor::Dtype::FP32);
    }
    const int total = dQuantized.size();
    if (total == 0) return;
    const float* dqp = dQuantized.host_f32();
    float* dxp = dX.host_f32_mut();
    if (dxp == dqp) return;                       // dX aliases dQuantized
    for (int i = 0; i < total; ++i) dxp[i] = dqp[i];
}

// ─── fsq_quantize ───────────────────────────────────────────────────────────

void fsq_quantize_forward(const ::brotensor::Tensor& x,
                          const ::brotensor::Tensor& levels,
                          ::brotensor::Tensor& quantized,
                          ::brotensor::Tensor& packed_indices) {
    require_fp32("fsq_quantize_forward", x, "x");
    require_int32("fsq_quantize_forward", levels, "levels");
    const int N = x.rows;
    const int D = x.cols;
    if (levels.size() != D) {
        fail("fsq_quantize_forward", "levels must have D elements (one per column of x)");
    }

    // quantized: (N, D) FP32.
    if (quantized.rows != N || quantized.cols != D ||
        quantized.dtype != ::brotensor::Dtype::FP32) {
        quantized.resize(N, D, ::brotensor::Dtype::FP32);
    }
    // packed_indices: (N, 1) INT32 — resize AND dtype-set.
    if (packed_indices.rows != N || packed_indices.cols != 1 ||
        packed_indices.dtype != ::brotensor::Dtype::INT32) {
        packed_indices.resize(N, 1, ::brotensor::Dtype::INT32);
    }

    const int32_t* Lp = static_cast<const int32_t*>(levels.host_raw());
    // Validate level counts up front (every L_d >= 2).
    for (int d = 0; d < D; ++d) {
        if (Lp[d] < 2) {
            fail("fsq_quantize_forward", "every level count must be >= 2");
        }
    }
    if (N == 0 || D == 0) return;

    const float* xp = x.host_f32();
    float*   qp = quantized.host_f32_mut();
    int32_t* pp = static_cast<int32_t*>(packed_indices.host_raw_mut());

    for (int n = 0; n < N; ++n) {
        const float* x_row = xp + static_cast<std::size_t>(n) * D;
        float* q_row = qp + static_cast<std::size_t>(n) * D;
        // Mixed-radix pack: dimension 0 is the least-significant digit.
        //   packed = i_0 + L_0 * (i_1 + L_1 * (i_2 + ...))
        // Build from the most-significant digit down via Horner's scheme.
        long long packed = 0;
        for (int d = D - 1; d >= 0; --d) {
            const int   L = Lp[d];
            const float h = static_cast<float>(L - 1) * 0.5f;  // half-width
            // 1. clamp into [-1, 1].
            float v = x_row[d];
            if (v < -1.0f) v = -1.0f;
            else if (v > 1.0f) v = 1.0f;
            // 2. map to a level index in [0, L-1].
            float idx_f = std::round((v + 1.0f) * 0.5f * static_cast<float>(L - 1));
            int   idx   = static_cast<int>(idx_f);
            if (idx < 0) idx = 0;
            else if (idx > L - 1) idx = L - 1;
            // 3. dequantize back into [-1, 1].
            q_row[d] = static_cast<float>(idx) / h - 1.0f;
            packed = packed * static_cast<long long>(L) +
                     static_cast<long long>(idx);
        }
        pp[n] = static_cast<int32_t>(packed);
    }
}

void fsq_quantize_backward(const ::brotensor::Tensor& dQuantized,
                           ::brotensor::Tensor& dX) {
    require_fp32("fsq_quantize_backward", dQuantized, "dQuantized");
    // Straight-through estimator: round is non-differentiable, so the gradient
    // passes through unchanged. dX is OVERWRITTEN, not accumulated.
    if (dX.rows != dQuantized.rows || dX.cols != dQuantized.cols ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(dQuantized.rows, dQuantized.cols, ::brotensor::Dtype::FP32);
    }
    const int total = dQuantized.size();
    if (total == 0) return;
    const float* dqp = dQuantized.host_f32();
    float* dxp = dX.host_f32_mut();
    if (dxp == dqp) return;                       // dX aliases dQuantized
    for (int i = 0; i < total; ++i) dxp[i] = dqp[i];
}

} // namespace brotensor::detail::cpu
