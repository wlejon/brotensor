// ─── CPU op implementations ────────────────────────────────────────────────
//
// Plain FP32 host loops. Ports the 16 ops formerly in src/cpu/ops.cpp to
// the unified `brotensor::Tensor` type. Math is unchanged from the old
// host-only impls — only buffer access (host_f32 / host_f32_mut / at)
// differs from the legacy `.ptr()` / `.data.data()` accessors.

#include <brotensor/tensor.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace brotensor::detail::cpu {

// ─── Dense layer ───────────────────────────────────────────────────────────

void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    const int out = W.rows;
    const int in  = W.cols;
    assert(x.size() == in);
    assert(b.size() == out);

    if (y.rows != out || y.cols != 1 || y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(out, 1, ::brotensor::Dtype::FP32);
    }

    const float* Wp = W.host_f32();
    const float* xp = x.host_f32();
    const float* bp = b.host_f32();
    float* yp = y.host_f32_mut();

    for (int i = 0; i < out; ++i) {
        float acc = bp[i];
        const float* row = Wp + static_cast<std::size_t>(i) * in;
        for (int j = 0; j < in; ++j) acc += row[j] * xp[j];
        yp[i] = acc;
    }
}

void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB) {
    const int out = W.rows;
    const int in  = W.cols;
    assert(x.size() == in);
    assert(dY.size() == out);
    assert(dW.rows == out && dW.cols == in);
    assert(dB.size() == out);

    if (dX.size() != in || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(in, 1, ::brotensor::Dtype::FP32);
    }

    const float* Wp  = W.host_f32();
    const float* xp  = x.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    float* dWp = dW.host_f32_mut();
    float* dBp = dB.host_f32_mut();

    // dX = W^T * dY  (overwrite)
    for (int j = 0; j < in; ++j) dXp[j] = 0.0f;
    for (int i = 0; i < out; ++i) {
        const float g = dYp[i];
        const float* row = Wp + static_cast<std::size_t>(i) * in;
        for (int j = 0; j < in; ++j) dXp[j] += row[j] * g;
    }
    // dW += dY * x^T  (outer product, accumulated)
    for (int i = 0; i < out; ++i) {
        const float g = dYp[i];
        float* row = dWp + static_cast<std::size_t>(i) * in;
        for (int j = 0; j < in; ++j) row[j] += g * xp[j];
    }
    // dB += dY
    for (int i = 0; i < out; ++i) dBp[i] += dYp[i];
}

// ─── Elementwise activations ───────────────────────────────────────────────

static void ensure_match(const ::brotensor::Tensor& src, ::brotensor::Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != ::brotensor::Dtype::FP32) {
        dst.resize(src.rows, src.cols, ::brotensor::Dtype::FP32);
    }
}

void relu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    ensure_match(x, y);
    const int n = x.size();
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
}

void relu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX) {
    ensure_match(x, dX);
    const int n = x.size();
    const float* xp  = x.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dXp[i] = xp[i] > 0.0f ? dYp[i] : 0.0f;
}

void tanh_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    ensure_match(x, y);
    const int n = x.size();
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::tanh(xp[i]);
}

void tanh_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX) {
    ensure_match(y, dX);
    const int n = y.size();
    const float* yp  = y.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dXp[i] = dYp[i] * (1.0f - yp[i] * yp[i]);
}

void sigmoid_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    ensure_match(x, y);
    const int n = x.size();
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = 1.0f / (1.0f + std::exp(-xp[i]));
}

void sigmoid_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX) {
    ensure_match(y, dX);
    const int n = y.size();
    const float* yp  = y.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) dXp[i] = dYp[i] * yp[i] * (1.0f - yp[i]);
}

// ─── Softmax + xent ────────────────────────────────────────────────────────

void softmax_forward(const ::brotensor::Tensor& logits, ::brotensor::Tensor& probs,
                     const float* mask) {
    ensure_match(logits, probs);
    const int n = logits.size();
    const float* lp = logits.host_f32();
    float* pp = probs.host_f32_mut();

    float m = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) continue;
        if (lp[i] > m) m = lp[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) { pp[i] = 0.0f; continue; }
        pp[i] = std::exp(lp[i] - m);
        s += pp[i];
    }
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < n; ++i) pp[i] *= inv;
}

void softmax_backward(const ::brotensor::Tensor& probs,
                      const ::brotensor::Tensor& dProbs,
                      ::brotensor::Tensor& dLogits) {
    ensure_match(probs, dLogits);
    const int n = probs.size();
    const float* pp = probs.host_f32();
    const float* dp = dProbs.host_f32();
    float* dz = dLogits.host_f32_mut();
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) dot += dp[i] * pp[i];
    for (int i = 0; i < n; ++i) dz[i] = pp[i] * (dp[i] - dot);
}

float softmax_xent_segment(const float* lp, const float* tp,
                           float* pp, float* dz,
                           int n, const float* mask) {
    // Stable softmax over the segment.
    float m = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) continue;
        if (lp[i] > m) m = lp[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) { pp[i] = 0.0f; continue; }
        pp[i] = std::exp(lp[i] - m);
        s += pp[i];
    }
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < n; ++i) pp[i] *= inv;

    // xent + dLogits = (p - t).
    float loss = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (mask && mask[i] < 0.5f) { dz[i] = 0.0f; continue; }
        if (tp[i] > 0.0f) {
            const float p = pp[i] > 1e-12f ? pp[i] : 1e-12f;
            loss -= tp[i] * std::log(p);
        }
        dz[i] = pp[i] - tp[i];
    }
    return loss;
}

float softmax_xent(const ::brotensor::Tensor& logits,
                   const ::brotensor::Tensor& target,
                   ::brotensor::Tensor& probs, ::brotensor::Tensor& dLogits,
                   const float* mask) {
    ensure_match(logits, probs);
    ensure_match(logits, dLogits);
    return softmax_xent_segment(logits.host_f32(), target.host_f32(),
                                probs.host_f32_mut(), dLogits.host_f32_mut(),
                                logits.size(), mask);
}

// ─── Scalar MSE ────────────────────────────────────────────────────────────

float mse_scalar(float pred, float target, float& dPred) {
    const float d = pred - target;
    dPred = d;
    return 0.5f * d * d;
}

// ─── Misc elementwise ──────────────────────────────────────────────────────

void add_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x) {
    const int n = y.size();
    assert(x.size() == n);
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] += xp[i];
}

void add_scalar_inplace(::brotensor::Tensor& y, float s) {
    const int n = y.size();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] += s;
}

// ─── Xavier-uniform init ───────────────────────────────────────────────────

// splitmix64 advanced by reference; deterministic, no external dep.
static inline uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline float u01(uint64_t& s) {
    // top 24 bits → [0, 1)
    return static_cast<float>((splitmix(s) >> 40)) / 16777216.0f;
}

void xavier_init(::brotensor::Tensor& W, uint64_t& rng_state) {
    // fan-in = cols, fan-out = rows ; limit = sqrt(6 / (in + out))
    const float limit = std::sqrt(6.0f / static_cast<float>(W.rows + W.cols));
    const int n = W.size();
    float* wp = W.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float u = u01(rng_state);       // [0,1)
        wp[i] = (u * 2.0f - 1.0f) * limit;    // [-limit, +limit]
    }
}

} // namespace brotensor::detail::cpu
