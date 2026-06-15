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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

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

void softmax_rows_forward(const ::brotensor::Tensor& X, ::brotensor::Tensor& Y,
                          int rows, int cols) {
    if (Y.rows != X.rows || Y.cols != X.cols || Y.dtype != X.dtype)
        Y.resize(X.rows, X.cols, X.dtype);
    const float* xp = X.host_f32();
    float* yp = Y.host_f32_mut();
    for (int r = 0; r < rows; ++r) {
        const float* lp = xp + static_cast<std::size_t>(r) * cols;
        float* pp = yp + static_cast<std::size_t>(r) * cols;
        float m = -1e30f;
        for (int i = 0; i < cols; ++i) if (lp[i] > m) m = lp[i];
        float s = 0.0f;
        for (int i = 0; i < cols; ++i) { pp[i] = std::exp(lp[i] - m); s += pp[i]; }
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        for (int i = 0; i < cols; ++i) pp[i] *= inv;
    }
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

void add_channel_bias_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& bias,
                              int C, int L) {
    float* yp = y.host_f32_mut();
    const float* bp = bias.host_f32();
    for (int c = 0; c < C; ++c) {
        const float b = bp[c];
        float* row = yp + static_cast<std::size_t>(c) * L;
        for (int i = 0; i < L; ++i) row[i] += b;
    }
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

// ─── Optimisers ────────────────────────────────────────────────────────────

void sgd_step(::brotensor::Tensor& param, ::brotensor::Tensor& grad,
              ::brotensor::Tensor& velocity, float lr, float momentum) {
    const int n = param.size();
    assert(grad.size() == n && velocity.size() == n);
    float* p = param.host_f32_mut();
    const float* g = grad.host_f32();
    float* v = velocity.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float vi = momentum * v[i] + g[i];
        v[i] = vi;
        p[i] -= lr * vi;
    }
}

void adam_step(::brotensor::Tensor& param, const ::brotensor::Tensor& grad,
               ::brotensor::Tensor& m, ::brotensor::Tensor& v,
               float lr, float beta1, float beta2, float eps, int step) {
    const int n = param.size();
    assert(grad.size() == n && m.size() == n && v.size() == n);
    const float inv_bc1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
    const float inv_bc2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
    float* p = param.host_f32_mut();
    const float* g = grad.host_f32();
    float* mp = m.host_f32_mut();
    float* vp = v.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float gi = g[i];
        const float mi = beta1 * mp[i] + (1.0f - beta1) * gi;
        const float vi = beta2 * vp[i] + (1.0f - beta2) * gi * gi;
        mp[i] = mi;
        vp[i] = vi;
        const float m_hat = mi * inv_bc1;
        const float v_hat = vi * inv_bc2;
        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// ─── Elementwise scale ─────────────────────────────────────────────────────

void scale_inplace(::brotensor::Tensor& y, float s) {
    const int n = y.size();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] *= s;
}

// ─── LayerNorm ─────────────────────────────────────────────────────────────

void layernorm_forward(const ::brotensor::Tensor& x,
                       const ::brotensor::Tensor& gamma,
                       const ::brotensor::Tensor& beta,
                       ::brotensor::Tensor& y, ::brotensor::Tensor& xhat,
                       float& mean_out, float& rstd_out, float eps) {
    const int n = x.size();
    if (y.rows != x.rows || y.cols != x.cols ||
        y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    if (xhat.rows != x.rows || xhat.cols != x.cols ||
        xhat.dtype != ::brotensor::Dtype::FP32) {
        xhat.resize(x.rows, x.cols, ::brotensor::Dtype::FP32);
    }
    if (n == 0) { mean_out = 0.0f; rstd_out = 0.0f; return; }

    const float* xp = x.host_f32();
    const float* gp = gamma.host_f32();
    const float* bp = beta.host_f32();
    float* yp  = y.host_f32_mut();
    float* xhp = xhat.host_f32_mut();

    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += xp[i];
    mean /= static_cast<float>(n);

    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = xp[i] - mean;
        var += d * d;
    }
    var /= static_cast<float>(n);
    const float rstd = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < n; ++i) {
        const float xh = (xp[i] - mean) * rstd;
        xhp[i] = xh;
        yp[i] = gp[i] * xh + bp[i];
    }
    mean_out = mean;
    rstd_out = rstd;
}

void layernorm_backward(const ::brotensor::Tensor& dY,
                        const ::brotensor::Tensor& xhat,
                        const ::brotensor::Tensor& gamma, float rstd,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dGamma,
                        ::brotensor::Tensor& dBeta) {
    const int n = dY.size();
    if (dX.rows != dY.rows || dX.cols != dY.cols ||
        dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(dY.rows, dY.cols, ::brotensor::Dtype::FP32);
    }
    if (n == 0) return;

    const float* dYp = dY.host_f32();
    const float* xhp = xhat.host_f32();
    const float* gp  = gamma.host_f32();
    float* dXp = dX.host_f32_mut();
    float* dGp = dGamma.host_f32_mut();
    float* dBp = dBeta.host_f32_mut();

    // dGamma/dBeta accumulate; caller zeros.
    for (int i = 0; i < n; ++i) {
        dGp[i] += dYp[i] * xhp[i];
        dBp[i] += dYp[i];
    }
    // sum_dxh, sum_dxh_xhat where dxh = dY * gamma.
    float sum_dxh = 0.0f, sum_dxh_xhat = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float dxh = dYp[i] * gp[i];
        sum_dxh += dxh;
        sum_dxh_xhat += dxh * xhp[i];
    }
    const float nf = static_cast<float>(n);
    const float scale = rstd / nf;
    for (int i = 0; i < n; ++i) {
        const float dxh = dYp[i] * gp[i];
        dXp[i] = scale * (nf * dxh - sum_dxh - xhp[i] * sum_dxh_xhat);
    }
}

// ─── Single-head scaled dot-product self-attention ─────────────────────────

void attention_forward(const ::brotensor::Tensor& X,
                       const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                       const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                       const float* d_mask,
                       ::brotensor::Tensor& Q, ::brotensor::Tensor& K,
                       ::brotensor::Tensor& V, ::brotensor::Tensor& Attn,
                       ::brotensor::Tensor& Y_pre_Wo, ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const int N = X.rows;
    const int D = X.cols;
    if (Q.rows != N || Q.cols != D || Q.dtype != Dtype::FP32) Q.resize(N, D, Dtype::FP32);
    if (K.rows != N || K.cols != D || K.dtype != Dtype::FP32) K.resize(N, D, Dtype::FP32);
    if (V.rows != N || V.cols != D || V.dtype != Dtype::FP32) V.resize(N, D, Dtype::FP32);
    if (Attn.rows != N || Attn.cols != N || Attn.dtype != Dtype::FP32) Attn.resize(N, N, Dtype::FP32);
    if (Y_pre_Wo.rows != N || Y_pre_Wo.cols != D || Y_pre_Wo.dtype != Dtype::FP32) Y_pre_Wo.resize(N, D, Dtype::FP32);
    if (O.rows != N || O.cols != D || O.dtype != Dtype::FP32) O.resize(N, D, Dtype::FP32);
    if (N == 0 || D == 0) return;

    const float* Xp  = X.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* Qp = Q.host_f32_mut();
    float* Kp = K.host_f32_mut();
    float* Vp = V.host_f32_mut();
    float* Ap = Attn.host_f32_mut();
    float* Yp = Y_pre_Wo.host_f32_mut();
    float* Op = O.host_f32_mut();

    // Q/K/V projections: out(i,j) = sum_k X(i,k) * W(j,k)   (W stored (D,D)).
    for (int i = 0; i < N; ++i) {
        const float* xr = Xp + static_cast<std::size_t>(i) * D;
        for (int j = 0; j < D; ++j) {
            const float* wq = Wqp + static_cast<std::size_t>(j) * D;
            const float* wk = Wkp + static_cast<std::size_t>(j) * D;
            const float* wv = Wvp + static_cast<std::size_t>(j) * D;
            float aq = 0.0f, ak = 0.0f, av = 0.0f;
            for (int k = 0; k < D; ++k) {
                aq += xr[k] * wq[k];
                ak += xr[k] * wk[k];
                av += xr[k] * wv[k];
            }
            const std::size_t idx = static_cast<std::size_t>(i) * D + j;
            Qp[idx] = aq;
            Kp[idx] = ak;
            Vp[idx] = av;
        }
    }

    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(D));

    // Scores → masked row softmax → Attn.
    for (int i = 0; i < N; ++i) {
        float* arow = Ap + static_cast<std::size_t>(i) * N;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int j = 0; j < N; ++j) arow[j] = 0.0f;
            continue;
        }
        const float* qr = Qp + static_cast<std::size_t>(i) * D;
        float m = -1e30f;
        for (int j = 0; j < N; ++j) {
            if (d_mask && d_mask[j] < 0.5f) continue;
            const float* kr = Kp + static_cast<std::size_t>(j) * D;
            float s = 0.0f;
            for (int k = 0; k < D; ++k) s += qr[k] * kr[k];
            s *= inv_sqrtd;
            arow[j] = s;
            if (s > m) m = s;
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            if (d_mask && d_mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
            const float e = std::exp(arow[j] - m);
            arow[j] = e;
            sum += e;
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int j = 0; j < N; ++j) arow[j] *= inv;
    }

    // Y_pre_Wo = Attn @ V.
    for (int i = 0; i < N; ++i) {
        const float* arow = Ap + static_cast<std::size_t>(i) * N;
        float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j)
                acc += arow[j] * Vp[static_cast<std::size_t>(j) * D + k];
            yr[k] = acc;
        }
    }

    // O = Y @ Wo^T, zero invalid query rows.
    for (int i = 0; i < N; ++i) {
        float* orow = Op + static_cast<std::size_t>(i) * D;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int c = 0; c < D; ++c) orow[c] = 0.0f;
            continue;
        }
        const float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<std::size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            orow[c] = acc;
        }
    }
}

void attention_backward(const ::brotensor::Tensor& dO,
                        const ::brotensor::Tensor& X,
                        const ::brotensor::Tensor& Q, const ::brotensor::Tensor& K,
                        const ::brotensor::Tensor& V, const ::brotensor::Tensor& Attn,
                        const ::brotensor::Tensor& Y_pre_Wo,
                        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                        const float* d_mask,
                        ::brotensor::Tensor& dX,
                        ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                        ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo) {
    using ::brotensor::Dtype;
    const int N = X.rows;
    const int D = X.cols;
    if (dX.rows != N || dX.cols != D || dX.dtype != Dtype::FP32) {
        dX.resize(N, D, Dtype::FP32);
    }
    if (N == 0 || D == 0) return;

    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(D));

    const float* dOp = dO.host_f32();
    const float* Xp  = X.host_f32();
    const float* Qp  = Q.host_f32();
    const float* Kp  = K.host_f32();
    const float* Vp  = V.host_f32();
    const float* Ap  = Attn.host_f32();
    const float* Yp  = Y_pre_Wo.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* dXp  = dX.host_f32_mut();
    float* dWqp = dWq.host_f32_mut();
    float* dWkp = dWk.host_f32_mut();
    float* dWvp = dWv.host_f32_mut();
    float* dWop = dWo.host_f32_mut();

    std::vector<float> dY(static_cast<std::size_t>(N) * D, 0.0f);
    std::vector<float> dAttn(static_cast<std::size_t>(N) * N, 0.0f);
    std::vector<float> dV(static_cast<std::size_t>(N) * D, 0.0f);
    std::vector<float> dScores(static_cast<std::size_t>(N) * N, 0.0f);
    std::vector<float> dQ(static_cast<std::size_t>(N) * D, 0.0f);
    std::vector<float> dK(static_cast<std::size_t>(N) * D, 0.0f);

    // dY (overwrite, zero on invalid rows) and dWo (accumulate).
    for (int i = 0; i < N; ++i) {
        const bool valid = !(d_mask && d_mask[i] < 0.5f);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            if (valid) {
                for (int c = 0; c < D; ++c)
                    acc += Wop[static_cast<std::size_t>(c) * D + k] *
                           dOp[static_cast<std::size_t>(i) * D + c];
            }
            dY[static_cast<std::size_t>(i) * D + k] = acc;
        }
    }
    for (int c = 0; c < D; ++c) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < N; ++i) {
                if (d_mask && d_mask[i] < 0.5f) continue;
                acc += dOp[static_cast<std::size_t>(i) * D + c] *
                       Yp[static_cast<std::size_t>(i) * D + k];
            }
            dWop[static_cast<std::size_t>(c) * D + k] += acc;
        }
    }

    // dAttn(i,j) = sum_k dY(i,k) * V(j,k);  dV(j,k) = sum_i Attn(i,j) * dY(i,k).
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < D; ++k)
                acc += dY[static_cast<std::size_t>(i) * D + k] *
                       Vp[static_cast<std::size_t>(j) * D + k];
            dAttn[static_cast<std::size_t>(i) * N + j] = acc;
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < N; ++i)
                acc += Ap[static_cast<std::size_t>(i) * N + j] *
                       dY[static_cast<std::size_t>(i) * D + k];
            dV[static_cast<std::size_t>(j) * D + k] = acc;
        }
    }

    // dScores via per-row softmax backward, scaled by inv_sqrtd.
    for (int i = 0; i < N; ++i) {
        const float* prow  = Ap + static_cast<std::size_t>(i) * N;
        const float* dprow = dAttn.data() + static_cast<std::size_t>(i) * N;
        float* drow = dScores.data() + static_cast<std::size_t>(i) * N;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int j = 0; j < N; ++j) drow[j] = 0.0f;
            continue;
        }
        float dot = 0.0f;
        for (int j = 0; j < N; ++j) dot += dprow[j] * prow[j];
        for (int j = 0; j < N; ++j) {
            if (d_mask && d_mask[j] < 0.5f) drow[j] = 0.0f;
            else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtd;
        }
    }

    // dQ(i,k) = sum_j dScores(i,j) * K(j,k);  dK(j,k) = sum_i dScores(i,j) * Q(i,k).
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j)
                acc += dScores[static_cast<std::size_t>(i) * N + j] *
                       Kp[static_cast<std::size_t>(j) * D + k];
            dQ[static_cast<std::size_t>(i) * D + k] = acc;
        }
    }
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < N; ++i)
                acc += dScores[static_cast<std::size_t>(i) * N + j] *
                       Qp[static_cast<std::size_t>(i) * D + k];
            dK[static_cast<std::size_t>(j) * D + k] = acc;
        }
    }

    // dWq/dWk/dWv accumulate: dW(j,k) += sum_i dQ(i,j) * X(i,k).
    for (int j = 0; j < D; ++j) {
        for (int k = 0; k < D; ++k) {
            float aq = 0.0f, ak = 0.0f, av = 0.0f;
            for (int i = 0; i < N; ++i) {
                const float xv = Xp[static_cast<std::size_t>(i) * D + k];
                aq += dQ[static_cast<std::size_t>(i) * D + j] * xv;
                ak += dK[static_cast<std::size_t>(i) * D + j] * xv;
                av += dV[static_cast<std::size_t>(i) * D + j] * xv;
            }
            const std::size_t idx = static_cast<std::size_t>(j) * D + k;
            dWqp[idx] += aq;
            dWkp[idx] += ak;
            dWvp[idx] += av;
        }
    }

    // dX(i,k) = sum_j dQ(i,j)*Wq(j,k) + dK(i,j)*Wk(j,k) + dV(i,j)*Wv(j,k).
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < D; ++j) {
                const std::size_t widx = static_cast<std::size_t>(j) * D + k;
                acc += dQ[static_cast<std::size_t>(i) * D + j] * Wqp[widx]
                     + dK[static_cast<std::size_t>(i) * D + j] * Wkp[widx]
                     + dV[static_cast<std::size_t>(i) * D + j] * Wvp[widx];
            }
            dXp[static_cast<std::size_t>(i) * D + k] = acc;
        }
    }
}

// ─── Multi-head self-attention ─────────────────────────────────────────────

void mha_forward(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                 const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                 const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                 const float* d_mask, int num_heads,
                 ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh,
                 ::brotensor::Tensor& Vh, ::brotensor::Tensor& Attnh,
                 ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = (H > 0) ? D / H : 0;
    if (Qh.rows != H * K || Qh.cols != dh || Qh.dtype != Dtype::FP32) Qh.resize(H * K, dh, Dtype::FP32);
    if (Kh.rows != H * K || Kh.cols != dh || Kh.dtype != Dtype::FP32) Kh.resize(H * K, dh, Dtype::FP32);
    if (Vh.rows != H * K || Vh.cols != dh || Vh.dtype != Dtype::FP32) Vh.resize(H * K, dh, Dtype::FP32);
    if (Attnh.rows != H * K || Attnh.cols != K || Attnh.dtype != Dtype::FP32) Attnh.resize(H * K, K, Dtype::FP32);
    if (Yconcat.rows != K || Yconcat.cols != D || Yconcat.dtype != Dtype::FP32) Yconcat.resize(K, D, Dtype::FP32);
    if (O.rows != K || O.cols != D || O.dtype != Dtype::FP32) O.resize(K, D, Dtype::FP32);
    if (K == 0 || D == 0 || H == 0) return;

    const float* Xp  = X.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* Qp = Qh.host_f32_mut();
    float* Kp = Kh.host_f32_mut();
    float* Vp = Vh.host_f32_mut();
    float* Ap = Attnh.host_f32_mut();
    float* Yp = Yconcat.host_f32_mut();
    float* Op = O.host_f32_mut();

    const float* bqp = bq ? bq->host_f32() : nullptr;
    const float* bkp = bk ? bk->host_f32() : nullptr;
    const float* bvp = bv ? bv->host_f32() : nullptr;
    const float* bop = bo ? bo->host_f32() : nullptr;

    // Per-head Q/K/V projection: out(hh,i,j) = sum_k X(i,k) * W(hh*dh+j, k)
    // + (b ? b[hh*dh+j] : 0). Biases are flat length-D vectors indexed by
    // the full output column (hh*dh+j).
    for (int hh = 0; hh < H; ++hh) {
        const int row_off = hh * dh;
        for (int i = 0; i < K; ++i) {
            const float* xr = Xp + static_cast<std::size_t>(i) * D;
            const std::size_t out_row =
                (static_cast<std::size_t>(hh) * K + i) * dh;
            for (int j = 0; j < dh; ++j) {
                const int wrow = row_off + j;
                const float* wq = Wqp + static_cast<std::size_t>(wrow) * D;
                const float* wk = Wkp + static_cast<std::size_t>(wrow) * D;
                const float* wv = Wvp + static_cast<std::size_t>(wrow) * D;
                float aq = 0.0f, ak = 0.0f, av = 0.0f;
                for (int k = 0; k < D; ++k) {
                    aq += xr[k] * wq[k];
                    ak += xr[k] * wk[k];
                    av += xr[k] * wv[k];
                }
                if (bqp) aq += bqp[wrow];
                if (bkp) ak += bkp[wrow];
                if (bvp) av += bvp[wrow];
                Qp[out_row + j] = aq;
                Kp[out_row + j] = ak;
                Vp[out_row + j] = av;
            }
        }
    }

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    // Per-head scores → masked row softmax → Attnh.
    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < K; ++i) {
            float* arow = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            if (d_mask && d_mask[i] < 0.5f) {
                for (int j = 0; j < K; ++j) arow[j] = 0.0f;
                continue;
            }
            const float* qr = Qp + (static_cast<std::size_t>(hh) * K + i) * dh;
            float m = -1e30f;
            for (int j = 0; j < K; ++j) {
                if (d_mask && d_mask[j] < 0.5f) continue;
                const float* kr = Kp + (static_cast<std::size_t>(hh) * K + j) * dh;
                float s = 0.0f;
                for (int k = 0; k < dh; ++k) s += qr[k] * kr[k];
                s *= inv_sqrtdh;
                arow[j] = s;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < K; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
                const float e = std::exp(arow[j] - m);
                arow[j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < K; ++j) arow[j] *= inv;
        }
    }

    // Per-head Attn @ V → Yconcat(i, hh*dh+k).
    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < K; ++i) {
            const float* arow = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const float vv = Vp[(static_cast<std::size_t>(hh) * K + j) * dh + k];
                    acc += arow[j] * vv;
                }
                Yp[static_cast<std::size_t>(i) * D + (hh * dh + k)] = acc;
            }
        }
    }

    // Output projection O = Yconcat @ Wo^T + bo, zero invalid query rows.
    for (int i = 0; i < K; ++i) {
        float* orow = Op + static_cast<std::size_t>(i) * D;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int c = 0; c < D; ++c) orow[c] = 0.0f;
            continue;
        }
        const float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<std::size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            if (bop) acc += bop[c];
            orow[c] = acc;
        }
    }
}

void mha_backward(const ::brotensor::Tensor& dO,
                  const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                  const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                  const ::brotensor::Tensor& Yconcat,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                  const float* d_mask, int num_heads,
                  ::brotensor::Tensor& dX,
                  ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                  ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo,
                  ::brotensor::Tensor* dbq, ::brotensor::Tensor* dbk,
                  ::brotensor::Tensor* dbv, ::brotensor::Tensor* dbo) {
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = (H > 0) ? D / H : 0;
    if (dX.rows != K || dX.cols != D || dX.dtype != Dtype::FP32) {
        dX.resize(K, D, Dtype::FP32);
    }
    if (K == 0 || D == 0 || H == 0) return;

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    const float* dOp = dO.host_f32();
    const float* Xp  = X.host_f32();
    const float* Qp  = Qh.host_f32();
    const float* Kp  = Kh.host_f32();
    const float* Vp  = Vh.host_f32();
    const float* Ap  = Attnh.host_f32();
    const float* Yp  = Yconcat.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* dXp  = dX.host_f32_mut();
    float* dWqp = dWq.host_f32_mut();
    float* dWkp = dWk.host_f32_mut();
    float* dWvp = dWv.host_f32_mut();
    float* dWop = dWo.host_f32_mut();

    std::vector<float> dYc(static_cast<std::size_t>(K) * D, 0.0f);
    std::vector<float> dAttn(static_cast<std::size_t>(H) * K * K, 0.0f);
    std::vector<float> dVh(static_cast<std::size_t>(H) * K * dh, 0.0f);
    std::vector<float> dScores(static_cast<std::size_t>(H) * K * K, 0.0f);
    std::vector<float> dQh(static_cast<std::size_t>(H) * K * dh, 0.0f);
    std::vector<float> dKh(static_cast<std::size_t>(H) * K * dh, 0.0f);

    // dYconcat (overwrite, zero on invalid rows) and dWo (accumulate).
    for (int i = 0; i < K; ++i) {
        const bool valid = !(d_mask && d_mask[i] < 0.5f);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            if (valid) {
                for (int c = 0; c < D; ++c)
                    acc += Wop[static_cast<std::size_t>(c) * D + k] *
                           dOp[static_cast<std::size_t>(i) * D + c];
            }
            dYc[static_cast<std::size_t>(i) * D + k] = acc;
        }
    }
    for (int c = 0; c < D; ++c) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < K; ++i) {
                if (d_mask && d_mask[i] < 0.5f) continue;
                acc += dOp[static_cast<std::size_t>(i) * D + c] *
                       Yp[static_cast<std::size_t>(i) * D + k];
            }
            dWop[static_cast<std::size_t>(c) * D + k] += acc;
        }
    }

    // dbo[c] += sum over valid rows of dO[i,c] (caller zeros).
    if (dbo) {
        float* dbop = dbo->host_f32_mut();
        for (int c = 0; c < D; ++c) {
            float acc = 0.0f;
            for (int i = 0; i < K; ++i) {
                if (d_mask && d_mask[i] < 0.5f) continue;
                acc += dOp[static_cast<std::size_t>(i) * D + c];
            }
            dbop[c] += acc;
        }
    }

    // Per-head dAttn and dVh.
    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < dh; ++k) {
                    const float dy = dYc[static_cast<std::size_t>(i) * D + (hh * dh + k)];
                    const float vv = Vp[(static_cast<std::size_t>(hh) * K + j) * dh + k];
                    acc += dy * vv;
                }
                dAttn[(static_cast<std::size_t>(hh) * K + i) * K + j] = acc;
            }
        }
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int i = 0; i < K; ++i) {
                    const float a  = Ap[(static_cast<std::size_t>(hh) * K + i) * K + j];
                    const float dy = dYc[static_cast<std::size_t>(i) * D + (hh * dh + k)];
                    acc += a * dy;
                }
                dVh[(static_cast<std::size_t>(hh) * K + j) * dh + k] = acc;
            }
        }
    }

    // Per-head softmax backward → dScores.
    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < K; ++i) {
            const float* prow  = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            const float* dprow = dAttn.data() + (static_cast<std::size_t>(hh) * K + i) * K;
            float* drow = dScores.data() + (static_cast<std::size_t>(hh) * K + i) * K;
            if (d_mask && d_mask[i] < 0.5f) {
                for (int j = 0; j < K; ++j) drow[j] = 0.0f;
                continue;
            }
            float dot = 0.0f;
            for (int j = 0; j < K; ++j) dot += dprow[j] * prow[j];
            for (int j = 0; j < K; ++j) {
                if (d_mask && d_mask[j] < 0.5f) drow[j] = 0.0f;
                else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
            }
        }
    }

    // Per-head dQh and dKh.
    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const float ds = dScores[(static_cast<std::size_t>(hh) * K + i) * K + j];
                    const float kk = Kp[(static_cast<std::size_t>(hh) * K + j) * dh + k];
                    acc += ds * kk;
                }
                dQh[(static_cast<std::size_t>(hh) * K + i) * dh + k] = acc;
            }
        }
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int i = 0; i < K; ++i) {
                    const float ds = dScores[(static_cast<std::size_t>(hh) * K + i) * K + j];
                    const float qq = Qp[(static_cast<std::size_t>(hh) * K + i) * dh + k];
                    acc += ds * qq;
                }
                dKh[(static_cast<std::size_t>(hh) * K + j) * dh + k] = acc;
            }
        }
    }

    // dbq/dbk/dbv accumulate: db[hh*dh+j] += sum_i dQh/dKh/dVh(hh,i,j).
    // Caller zeros. Skipped when the matching pointer is null.
    if (dbq || dbk || dbv) {
        float* dbqp = dbq ? dbq->host_f32_mut() : nullptr;
        float* dbkp = dbk ? dbk->host_f32_mut() : nullptr;
        float* dbvp = dbv ? dbv->host_f32_mut() : nullptr;
        for (int hh = 0; hh < H; ++hh) {
            for (int j = 0; j < dh; ++j) {
                const int r = hh * dh + j;
                float aq = 0.0f, ak = 0.0f, av = 0.0f;
                for (int i = 0; i < K; ++i) {
                    const std::size_t idx =
                        (static_cast<std::size_t>(hh) * K + i) * dh + j;
                    if (dbqp) aq += dQh[idx];
                    if (dbkp) ak += dKh[idx];
                    if (dbvp) av += dVh[idx];
                }
                if (dbqp) dbqp[r] += aq;
                if (dbkp) dbkp[r] += ak;
                if (dbvp) dbvp[r] += av;
            }
        }
    }

    // dWq/dWk/dWv accumulate: dW(hh*dh+j, k) += sum_i dQh(hh,i,j) * X(i,k).
    for (int wrow = 0; wrow < D; ++wrow) {
        const int hh = wrow / dh;
        const int j  = wrow % dh;
        for (int k = 0; k < D; ++k) {
            float aq = 0.0f, ak = 0.0f, av = 0.0f;
            for (int i = 0; i < K; ++i) {
                const float xv = Xp[static_cast<std::size_t>(i) * D + k];
                aq += dQh[(static_cast<std::size_t>(hh) * K + i) * dh + j] * xv;
                ak += dKh[(static_cast<std::size_t>(hh) * K + i) * dh + j] * xv;
                av += dVh[(static_cast<std::size_t>(hh) * K + i) * dh + j] * xv;
            }
            const std::size_t idx = static_cast<std::size_t>(wrow) * D + k;
            dWqp[idx] += aq;
            dWkp[idx] += ak;
            dWvp[idx] += av;
        }
    }

    // dX(i,k) = sum over heads, j of dQh*Wq + dKh*Wk + dVh*Wv at (hh*dh+j, k).
    for (int i = 0; i < K; ++i) {
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int hh = 0; hh < H; ++hh) {
                for (int j = 0; j < dh; ++j) {
                    const int wrow = hh * dh + j;
                    const std::size_t widx = static_cast<std::size_t>(wrow) * D + k;
                    const float gq = dQh[(static_cast<std::size_t>(hh) * K + i) * dh + j];
                    const float gk = dKh[(static_cast<std::size_t>(hh) * K + i) * dh + j];
                    const float gv = dVh[(static_cast<std::size_t>(hh) * K + i) * dh + j];
                    acc += gq * Wqp[widx] + gk * Wkp[widx] + gv * Wvp[widx];
                }
            }
            dXp[static_cast<std::size_t>(i) * D + k] = acc;
        }
    }
}

// ─── Concat / split ────────────────────────────────────────────────────────

void concat_rows(const std::vector<const ::brotensor::Tensor*>& parts,
                 ::brotensor::Tensor& out) {
    int total = 0;
    for (const auto* p : parts) {
        if (p) total += p->size();
    }
    if (out.rows != total || out.cols != 1 ||
        out.dtype != ::brotensor::Dtype::FP32) {
        out.resize(total, 1, ::brotensor::Dtype::FP32);
    }
    if (total == 0) return;
    float* op = out.host_f32_mut();
    int off = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        const float* pp = p->host_f32();
        for (int i = 0; i < n; ++i) op[off + i] = pp[i];
        off += n;
    }
}

void split_rows(const ::brotensor::Tensor& in,
                const std::vector<::brotensor::Tensor*>& parts) {
    const float* ip = in.host_f32();
    int off = 0;
    for (auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        float* pp = p->host_f32_mut();
        for (int i = 0; i < n; ++i) pp[i] = ip[off + i];
        off += n;
    }
}

// ─── Batched elementwise / linear ──────────────────────────────────────────

void add_inplace_batched(::brotensor::Tensor& Y_BD,
                         const ::brotensor::Tensor& X_BD) {
    const int n = Y_BD.size();
    assert(X_BD.size() == n);
    float* yp = Y_BD.host_f32_mut();
    const float* xp = X_BD.host_f32();
    for (int i = 0; i < n; ++i) yp[i] += xp[i];
}

void linear_forward_batched(const ::brotensor::Tensor& W,
                            const ::brotensor::Tensor& bias,
                            const ::brotensor::Tensor& X_BD,
                            ::brotensor::Tensor& Y_BD) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim ||
        Y_BD.dtype != ::brotensor::Dtype::FP32) {
        Y_BD.resize(B, out_dim, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || out_dim == 0) return;

    const float* bp = bias.host_f32();
    const float* Xp = X_BD.host_f32();
    float* Yp = Y_BD.host_f32_mut();

    // 16-bit weights (FP32 activations): widen per element. Correctness path —
    // the performance-relevant 16-bit-weight decode runs on the GPU backends.
    if (W.dtype == ::brotensor::Dtype::FP16 ||
        W.dtype == ::brotensor::Dtype::BF16) {
        const bool bf = (W.dtype == ::brotensor::Dtype::BF16);
        const std::uint16_t* Wp = static_cast<const std::uint16_t*>(W.data);
        for (int b = 0; b < B; ++b) {
            const float* xr = Xp + static_cast<std::size_t>(b) * in_dim;
            float* yr = Yp + static_cast<std::size_t>(b) * out_dim;
            for (int i = 0; i < out_dim; ++i) {
                const std::uint16_t* wr = Wp + static_cast<std::size_t>(i) * in_dim;
                float acc = bp[i];
                for (int j = 0; j < in_dim; ++j) {
                    const float w = bf ? ::brotensor::bf16_bits_to_fp32(wr[j])
                                       : ::brotensor::fp16_bits_to_fp32(wr[j]);
                    acc += w * xr[j];
                }
                yr[i] = acc;
            }
        }
        return;
    }

    const float* Wp = W.host_f32();
    for (int b = 0; b < B; ++b) {
        const float* xr = Xp + static_cast<std::size_t>(b) * in_dim;
        float* yr = Yp + static_cast<std::size_t>(b) * out_dim;
        for (int i = 0; i < out_dim; ++i) {
            const float* wr = Wp + static_cast<std::size_t>(i) * in_dim;
            float acc = bp[i];
            for (int j = 0; j < in_dim; ++j) acc += wr[j] * xr[j];
            yr[i] = acc;
        }
    }
}

void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (dX_BD.rows != B || dX_BD.cols != in_dim ||
        dX_BD.dtype != ::brotensor::Dtype::FP32) {
        dX_BD.resize(B, in_dim, ::brotensor::Dtype::FP32);
    }
    if (B == 0) return;

    const float* Wp  = W.host_f32();
    const float* Xp  = X_BD.host_f32();
    const float* dYp = dY_BD.host_f32();
    float* dXp = dX_BD.host_f32_mut();
    float* dWp = dW.host_f32_mut();
    float* dBp = dB.host_f32_mut();

    // dX[b] = W^T * dY[b]   (overwrite).
    for (int b = 0; b < B; ++b) {
        const float* dyr = dYp + static_cast<std::size_t>(b) * out_dim;
        float* dxr = dXp + static_cast<std::size_t>(b) * in_dim;
        for (int j = 0; j < in_dim; ++j) dxr[j] = 0.0f;
        for (int i = 0; i < out_dim; ++i) {
            const float g = dyr[i];
            const float* wr = Wp + static_cast<std::size_t>(i) * in_dim;
            for (int j = 0; j < in_dim; ++j) dxr[j] += wr[j] * g;
        }
    }
    // dW += sum_b dY[b] * X[b]^T;  dB += sum_b dY[b]   (accumulate).
    for (int b = 0; b < B; ++b) {
        const float* dyr = dYp + static_cast<std::size_t>(b) * out_dim;
        const float* xr  = Xp  + static_cast<std::size_t>(b) * in_dim;
        for (int i = 0; i < out_dim; ++i) {
            const float g = dyr[i];
            float* dwr = dWp + static_cast<std::size_t>(i) * in_dim;
            for (int j = 0; j < in_dim; ++j) dwr[j] += g * xr[j];
            dBp[i] += g;
        }
    }
}

void relu_forward_batched(const ::brotensor::Tensor& X_BD,
                          ::brotensor::Tensor& Y_BD) {
    ensure_match(X_BD, Y_BD);
    const int n = X_BD.size();
    const float* xp = X_BD.host_f32();
    float* yp = Y_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
}

void relu_backward_batched(const ::brotensor::Tensor& X_BD,
                           const ::brotensor::Tensor& dY_BD,
                           ::brotensor::Tensor& dX_BD) {
    ensure_match(X_BD, dX_BD);
    const int n = X_BD.size();
    const float* xp  = X_BD.host_f32();
    const float* dYp = dY_BD.host_f32();
    float* dXp = dX_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) dXp[i] = xp[i] > 0.0f ? dYp[i] : 0.0f;
}

void tanh_forward_batched(const ::brotensor::Tensor& X_BD,
                          ::brotensor::Tensor& Y_BD) {
    ensure_match(X_BD, Y_BD);
    const int n = X_BD.size();
    const float* xp = X_BD.host_f32();
    float* yp = Y_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::tanh(xp[i]);
}

void tanh_backward_batched(const ::brotensor::Tensor& Y_BD,
                           const ::brotensor::Tensor& dY_BD,
                           ::brotensor::Tensor& dX_BD) {
    ensure_match(Y_BD, dX_BD);
    const int n = Y_BD.size();
    const float* yp  = Y_BD.host_f32();
    const float* dYp = dY_BD.host_f32();
    float* dXp = dX_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) dXp[i] = dYp[i] * (1.0f - yp[i] * yp[i]);
}

// ─── Batched per-sample losses ─────────────────────────────────────────────

void mse_vec_per_sample(const ::brotensor::Tensor& pred,
                        const ::brotensor::Tensor& target,
                        ::brotensor::Tensor& dPred,
                        ::brotensor::Tensor& loss_per_sample) {
    const int B = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols ||
        dPred.dtype != ::brotensor::Dtype::FP32) {
        dPred.resize(pred.rows, pred.cols, ::brotensor::Dtype::FP32);
    }
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32) {
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    }
    if (B == 0) return;
    const float* pp = pred.host_f32();
    const float* tp = target.host_f32();
    float* dp = dPred.host_f32_mut();
    float* lp = loss_per_sample.host_f32_mut();
    for (int b = 0; b < B; ++b) {
        const float d = pp[b] - tp[b];
        dp[b] = d;
        lp[b] = 0.5f * d * d;
    }
}

void softmax_xent_fused_batched(const ::brotensor::Tensor& logits_BL,
                                const ::brotensor::Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets,
                                int n_heads,
                                ::brotensor::Tensor& probs_BL,
                                ::brotensor::Tensor& dLogits_BL,
                                ::brotensor::Tensor& loss_per_sample) {
    const int B     = logits_BL.rows;
    const int n_act = logits_BL.cols;
    if (probs_BL.rows != B || probs_BL.cols != n_act ||
        probs_BL.dtype != ::brotensor::Dtype::FP32) {
        probs_BL.resize(B, n_act, ::brotensor::Dtype::FP32);
    }
    if (dLogits_BL.rows != B || dLogits_BL.cols != n_act ||
        dLogits_BL.dtype != ::brotensor::Dtype::FP32) {
        dLogits_BL.resize(B, n_act, ::brotensor::Dtype::FP32);
    }
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32) {
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || n_act == 0 || n_heads <= 0) return;

    const float* lp = logits_BL.host_f32();
    const float* tp = target_BL.host_f32();
    float* pp = probs_BL.host_f32_mut();
    float* dp = dLogits_BL.host_f32_mut();
    float* lossp = loss_per_sample.host_f32_mut();

    for (int b = 0; b < B; ++b) {
        // Overwrite loss before per-head accumulation.
        float sample_loss = 0.0f;
        const int row_off = b * n_act;
        for (int h = 0; h < n_heads; ++h) {
            const int off = d_head_offsets[h];
            const int end = d_head_offsets[h + 1];
            const int len = end - off;
            const float* lr = lp + row_off + off;
            const float* tr = tp + row_off + off;
            const float* mr = d_mask_BL ? (d_mask_BL + row_off + off) : nullptr;
            float* pr = pp + row_off + off;
            float* dr = dp + row_off + off;

            float m = -1e30f;
            for (int i = 0; i < len; ++i) {
                if (mr && mr[i] < 0.5f) continue;
                if (lr[i] > m) m = lr[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < len; ++i) {
                if (mr && mr[i] < 0.5f) { pr[i] = 0.0f; continue; }
                const float e = std::exp(lr[i] - m);
                pr[i] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int i = 0; i < len; ++i) {
                if (mr && mr[i] < 0.5f) { dr[i] = 0.0f; continue; }
                const float p = pr[i] * inv;
                pr[i] = p;
                const float t = tr[i];
                if (t > 0.0f) {
                    const float pc = p > 1e-12f ? p : 1e-12f;
                    sample_loss -= t * std::log(pc);
                }
                dr[i] = p - t;
            }
        }
        lossp[b] = sample_loss;
    }
}

void bce_with_logits_fused_batched(const ::brotensor::Tensor& logits_BL,
                                   const ::brotensor::Tensor& target_BL,
                                   const float* d_mask_BL,
                                   float pos_weight,
                                   ::brotensor::Tensor& probs_BL,
                                   ::brotensor::Tensor& dLogits_BL,
                                   ::brotensor::Tensor& loss_per_sample) {
    const int B = logits_BL.rows;
    const int L = logits_BL.cols;
    if (probs_BL.rows != B || probs_BL.cols != L ||
        probs_BL.dtype != ::brotensor::Dtype::FP32) {
        probs_BL.resize(B, L, ::brotensor::Dtype::FP32);
    }
    if (dLogits_BL.rows != B || dLogits_BL.cols != L ||
        dLogits_BL.dtype != ::brotensor::Dtype::FP32) {
        dLogits_BL.resize(B, L, ::brotensor::Dtype::FP32);
    }
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32) {
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || L == 0) return;

    const float* lp = logits_BL.host_f32();
    const float* tp = target_BL.host_f32();
    float* pp = probs_BL.host_f32_mut();
    float* dp = dLogits_BL.host_f32_mut();
    float* lossp = loss_per_sample.host_f32_mut();
    const float w = pos_weight;

    for (int b = 0; b < B; ++b) {
        float sample_loss = 0.0f;
        const int row_off = b * L;
        const float* lr = lp + row_off;
        const float* tr = tp + row_off;
        const float* mr = d_mask_BL ? (d_mask_BL + row_off) : nullptr;
        float* pr = pp + row_off;
        float* dr = dp + row_off;
        for (int i = 0; i < L; ++i) {
            if (mr && mr[i] < 0.5f) {
                pr[i] = 0.0f;
                dr[i] = 0.0f;
                continue;
            }
            const float z = lr[i];
            const float y = tr[i];
            // softplus(x) = max(x, 0) + log1p(exp(-|x|))
            const float az = std::fabs(z);
            const float sp_neg = (z > 0.0f ? 0.0f : -z) + std::log1p(std::exp(-az));
            const float sp_pos = (z > 0.0f ? z      : 0.0f) + std::log1p(std::exp(-az));
            sample_loss += w * y * sp_neg + (1.0f - y) * sp_pos;
            // s = sigmoid(z), numerically stable.
            const float s = z >= 0.0f
                ? 1.0f / (1.0f + std::exp(-z))
                : std::exp(z) / (1.0f + std::exp(z));
            pr[i] = s;
            dr[i] = s * (w * y + 1.0f - y) - w * y;
        }
        lossp[b] = sample_loss;
    }
}

// ─── Masked mean-pool ──────────────────────────────────────────────────────

void build_slot_mask(const ::brotensor::Tensor& x, int offset, int K, int stride,
                     ::brotensor::Tensor& mask) {
    if (mask.rows != K || mask.cols != 1 ||
        mask.dtype != ::brotensor::Dtype::FP32) {
        mask.resize(K, 1, ::brotensor::Dtype::FP32);
    }
    const float* xp = x.host_f32();
    float* mp = mask.host_f32_mut();
    for (int k = 0; k < K; ++k) {
        mp[k] = (xp[offset + static_cast<std::size_t>(k) * stride] > 0.5f)
                    ? 1.0f : 0.0f;
    }
}

void copy_d2d(const ::brotensor::Tensor& src, int src_off,
              ::brotensor::Tensor& dst, int dst_off, int n) {
    const float* sp = src.host_f32();
    float* dp = dst.host_f32_mut();
    std::memcpy(dp + dst_off, sp + src_off, static_cast<std::size_t>(n) * sizeof(float));
}

void copy_d2d_strided(const ::brotensor::Tensor& src, int src_off, int src_pitch,
                      ::brotensor::Tensor& dst, int dst_off, int dst_pitch,
                      int width, int height) {
    const float* sp = src.host_f32() + src_off;
    float* dp = dst.host_f32_mut() + dst_off;
    for (int r = 0; r < height; ++r) {
        std::memcpy(dp + static_cast<std::size_t>(r) * dst_pitch,
                    sp + static_cast<std::size_t>(r) * src_pitch,
                    static_cast<std::size_t>(width) * sizeof(float));
    }
}

void masked_mean_pool_forward(const ::brotensor::Tensor& X, const float* d_mask,
                              ::brotensor::Tensor& y) {
    const int K = X.rows;
    const int D = X.cols;
    if (y.rows != D || y.cols != 1 || y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(D, 1, ::brotensor::Dtype::FP32);
    }
    const float* Xp = X.host_f32();
    float* yp = y.host_f32_mut();
    for (int j = 0; j < D; ++j) yp[j] = 0.0f;
    int num_valid = 0;
    for (int k = 0; k < K; ++k) {
        if (d_mask && d_mask[k] < 0.5f) continue;
        const float* row = Xp + static_cast<std::size_t>(k) * D;
        for (int j = 0; j < D; ++j) yp[j] += row[j];
        ++num_valid;
    }
    if (num_valid > 0) {
        const float inv = 1.0f / static_cast<float>(num_valid);
        for (int j = 0; j < D; ++j) yp[j] *= inv;
    }
}

void masked_mean_pool_backward(const ::brotensor::Tensor& dY, const float* d_mask,
                               int K, ::brotensor::Tensor& dX) {
    const int D = dY.size();
    if (dX.rows != K || dX.cols != D || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(K, D, ::brotensor::Dtype::FP32);
    }
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    int num_valid = 0;
    for (int k = 0; k < K; ++k) {
        if (!d_mask || d_mask[k] >= 0.5f) ++num_valid;
    }
    const float inv = num_valid > 0 ? 1.0f / static_cast<float>(num_valid) : 0.0f;
    for (int k = 0; k < K; ++k) {
        float* row = dXp + static_cast<std::size_t>(k) * D;
        const bool valid = (!d_mask || d_mask[k] >= 0.5f) && num_valid > 0;
        for (int j = 0; j < D; ++j) row[j] = valid ? dYp[j] * inv : 0.0f;
    }
}

} // namespace brotensor::detail::cpu
