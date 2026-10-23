// ─── CPU op implementations ────────────────────────────────────────────────
//
// Plain FP32 host loops. Ports the core ops formerly in src/cpu/ops.cpp to
// the unified `brotensor::Tensor` type. (MHA, batched ops, and masked mean-pool
// live in src/cpu/ops_mha_batched.cpp.)

#include <brotensor/tensor.h>
#include <brotensor/detail/cpu/thread_pool.h>

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
    for (int i = 0; i < n; ++i) {
        const float yc = std::clamp(yp[i], -1.0f, 1.0f);
        dXp[i] = dYp[i] * (1.0f - yc * yc);
    }
}

void sigmoid_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    ensure_match(x, y);
    const int n = x.size();
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float xc = std::clamp(xp[i], -88.0f, 88.0f);
        yp[i] = 1.0f / (1.0f + std::exp(-xc));
    }
}

void sigmoid_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX) {
    ensure_match(y, dX);
    const int n = y.size();
    const float* yp  = y.host_f32();
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float yc = std::clamp(yp[i], 0.0f, 1.0f);
        dXp[i] = dYp[i] * yc * (1.0f - yc);
    }
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
        if (std::isfinite(lp[i]) && lp[i] > m) m = lp[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        if ((mask && mask[i] < 0.5f) || !std::isfinite(lp[i])) { pp[i] = 0.0f; continue; }
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
        for (int i = 0; i < cols; ++i) if (std::isfinite(lp[i]) && lp[i] > m) m = lp[i];
        float s = 0.0f;
        for (int i = 0; i < cols; ++i) {
            pp[i] = std::isfinite(lp[i]) ? std::exp(lp[i] - m) : 0.0f;
            s += pp[i];
        }
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
    for (int i = 0; i < n; ++i) {
        const float p = std::clamp(pp[i], 0.0f, 1.0f);
        if (p > 0.0f && std::isfinite(dp[i])) dot += dp[i] * p;
    }
    for (int i = 0; i < n; ++i) {
        const float p = std::clamp(pp[i], 0.0f, 1.0f);
        dz[i] = (p > 0.0f && std::isfinite(dp[i])) ? (p * (dp[i] - dot)) : 0.0f;
    }
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
    const float eps_safe = std::max(eps, 1e-12f);
    const float denom = std::max(var + eps_safe, 1e-12f);
    const float rstd = std::isfinite(denom) ? (1.0f / std::sqrt(denom)) : 0.0f;

    for (int i = 0; i < n; ++i) {
        const float xh = (rstd > 0.0f && std::isfinite(mean)) ? ((xp[i] - mean) * rstd) : 0.0f;
        xhp[i] = xh;
        yp[i] = gp[i] * xh + bp[i];
    }
    mean_out = std::isfinite(mean) ? mean : 0.0f;
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
        const float xhi = std::isfinite(xhp[i]) ? xhp[i] : 0.0f;
        dGp[i] += dYp[i] * xhi;
        dBp[i] += dYp[i];
    }
    // sum_dxh, sum_dxh_xhat where dxh = dY * gamma.
    float sum_dxh = 0.0f, sum_dxh_xhat = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float xhi = std::isfinite(xhp[i]) ? xhp[i] : 0.0f;
        const float dxh = dYp[i] * gp[i];
        sum_dxh += dxh;
        sum_dxh_xhat += dxh * xhi;
    }
    const float rstd_safe = (std::isfinite(rstd) && rstd >= 0.0f) ? rstd : 0.0f;
    const float nf = static_cast<float>(n);
    const float scale = rstd_safe / nf;
    for (int i = 0; i < n; ++i) {
        const float xhi = std::isfinite(xhp[i]) ? xhp[i] : 0.0f;
        const float dxh = dYp[i] * gp[i];
        dXp[i] = (scale > 0.0f) ? (scale * (nf * dxh - sum_dxh - xhi * sum_dxh_xhat)) : 0.0f;
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

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
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
    });

    const float inv_sqrtd = 1.0f / std::sqrt(static_cast<float>(D));

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        float* arow = Ap + static_cast<std::size_t>(i) * N;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int j = 0; j < N; ++j) arow[j] = 0.0f;
            return;
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
    });

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        const float* arow = Ap + static_cast<std::size_t>(i) * N;
        float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j)
                acc += arow[j] * Vp[static_cast<std::size_t>(j) * D + k];
            yr[k] = acc;
        }
    });

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        float* orow = Op + static_cast<std::size_t>(i) * D;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int c = 0; c < D; ++c) orow[c] = 0.0f;
            return;
        }
        const float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<std::size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            orow[c] = acc;
        }
    });
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

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
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
    });
    for (int i = 0; i < N; ++i) {
        if (d_mask && d_mask[i] < 0.5f) continue;
        const float* dOi = dOp + static_cast<std::size_t>(i) * D;
        const float* Yi  = Yp  + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float dov = dOi[c];
            float* dWrow = dWop + static_cast<std::size_t>(c) * D;
            for (int k = 0; k < D; ++k) {
                dWrow[k] += dov * Yi[k];
            }
        }
    }

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < D; ++k)
                acc += dY[static_cast<std::size_t>(i) * D + k] *
                       Vp[static_cast<std::size_t>(j) * D + k];
            dAttn[static_cast<std::size_t>(i) * N + j] = acc;
        }
    });
    for (int i = 0; i < N; ++i) {
        const float* Ai = Ap + static_cast<std::size_t>(i) * N;
        const float* dYi = dY.data() + static_cast<std::size_t>(i) * D;
        for (int j = 0; j < N; ++j) {
            const float av = Ai[j];
            float* dVrow = dV.data() + static_cast<std::size_t>(j) * D;
            for (int k = 0; k < D; ++k) {
                dVrow[k] += av * dYi[k];
            }
        }
    }

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        const float* prow  = Ap + static_cast<std::size_t>(i) * N;
        const float* dprow = dAttn.data() + static_cast<std::size_t>(i) * N;
        float* drow = dScores.data() + static_cast<std::size_t>(i) * N;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int j = 0; j < N; ++j) drow[j] = 0.0f;
            return;
        }
        float dot = 0.0f;
        for (int j = 0; j < N; ++j) dot += dprow[j] * prow[j];
        for (int j = 0; j < N; ++j) {
            if (d_mask && d_mask[j] < 0.5f) drow[j] = 0.0f;
            else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtd;
        }
    });

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j)
                acc += dScores[static_cast<std::size_t>(i) * N + j] *
                       Kp[static_cast<std::size_t>(j) * D + k];
            dQ[static_cast<std::size_t>(i) * D + k] = acc;
        }
    });
    for (int i = 0; i < N; ++i) {
        const float* dSi = dScores.data() + static_cast<std::size_t>(i) * N;
        const float* Qi = Qp + static_cast<std::size_t>(i) * D;
        for (int j = 0; j < N; ++j) {
            const float ds = dSi[j];
            float* dKrow = dK.data() + static_cast<std::size_t>(j) * D;
            for (int k = 0; k < D; ++k) {
                dKrow[k] += ds * Qi[k];
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        const float* Xi  = Xp + static_cast<std::size_t>(i) * D;
        const float* dQi = dQ.data() + static_cast<std::size_t>(i) * D;
        const float* dKi = dK.data() + static_cast<std::size_t>(i) * D;
        const float* dVi = dV.data() + static_cast<std::size_t>(i) * D;
        for (int j = 0; j < D; ++j) {
            const float qj = dQi[j];
            const float kj = dKi[j];
            const float vj = dVi[j];
            float* dWqrow = dWqp + static_cast<std::size_t>(j) * D;
            float* dWkrow = dWkp + static_cast<std::size_t>(j) * D;
            float* dWvrow = dWvp + static_cast<std::size_t>(j) * D;
            for (int k = 0; k < D; ++k) {
                const float xv = Xi[k];
                dWqrow[k] += qj * xv;
                dWkrow[k] += kj * xv;
                dWvrow[k] += vj * xv;
            }
        }
    }

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
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
    });
}

} // namespace brotensor::detail::cpu
