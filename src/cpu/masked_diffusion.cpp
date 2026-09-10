// ─── CPU masked-diffusion token selection ──────────────────────────────────
//
// FP32 scalar host implementation of `masked_diffusion_scores` and
// `masked_diffusion_commit` — the per-step token selection of a masked-
// diffusion language model over a (C codebooks, T frames) grid (the OmniVoice
// port in brosoundml). See ops/sampling.h for the full contract and the
// pseudocode this follows.
//
// This is the parity reference for the CUDA kernel in src/cuda/
// masked_diffusion.cu, so the arithmetic is spelled out in the exact order the
// kernel uses:
//   * a row's log-sum-exp is max + (float)log(sum), the sum accumulated in
//     FP64 from FP32 expf terms (order-independent to the last FP32 ulp);
//   * the guided logit is a + gs * (a - b) with a = c - lse_c, b = u - lse_u,
//     each operation rounded separately (the kernel uses __fadd_rn /
//     __fmul_rn / __fsub_rn so nvcc cannot contract it into an FMA);
//   * the noise comes from detail/hash_rng.h, identical integer arithmetic on
//     both backends.
// The top-k class filter keeps the first k entries in (value desc, index asc)
// order: everything above the k-th value, then the lowest-indexed entries
// equal to it until k are kept.
//
// ── ACCUMULATION ────────────────────────────────────────────────────────────
//   masked_diffusion_scores — pred, scores and confidence OVERWRITTEN.
//   masked_diffusion_commit — tokens / unmask_step written only at idx[0..k).

#include <brotensor/tensor.h>
#include <brotensor/detail/hash_rng.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float gumbel_from_uniform(float u) {
    return -std::log(-std::log(u + 1e-10f) + 1e-10f);
}

// max + log(sum exp(x - max)) over a row of n floats; sum in FP64.
inline float row_lse(const float* x, int n, float& max_out) {
    float m = kNegInf;
    for (int i = 0; i < n; ++i) m = (x[i] > m) ? x[i] : m;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(std::exp(x[i] - m));
    max_out = m;
    return m + static_cast<float>(std::log(s));
}

// (a, ia) precedes (b, ib) in descending-value / ascending-index order.
inline bool prefers(float a, int ia, float b, int ib) {
    if (a != b) return a > b;
    return ia < ib;
}

}  // namespace

// ─── masked_diffusion_scores ────────────────────────────────────────────────

void masked_diffusion_scores(const ::brotensor::Tensor& logits,
                             const ::brotensor::Tensor& tokens,
                             int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty,
                             float position_temperature, float class_temperature,
                             float class_top_frac, uint64_t seed,
                             ::brotensor::Tensor& pred, ::brotensor::Tensor& scores,
                             ::brotensor::Tensor& confidence) {
    const char* op = "masked_diffusion_scores";
    if (logits.dtype != Dtype::FP32) fail(op, "logits must be FP32 (CPU backend is FP32-only)");
    if (tokens.dtype != Dtype::INT32) fail(op, "tokens must be INT32");
    if (T < 1 || C < 1 || V < 1) fail(op, "T, C and V must be >= 1");
    if (mask_id < 0 || mask_id >= V) fail(op, "mask_id must be in [0, V)");
    const bool guided = guidance_scale != 0.0f;
    const int R = guided ? 2 * T : T;
    if (logits.rows != R || logits.cols != C * V) {
        fail(op, "logits must be (" + std::to_string(R) + ", C*V=" +
                 std::to_string(C * V) + ") for T=" + std::to_string(T) +
                 (guided ? " with" : " without") + " guidance, got (" +
                 std::to_string(logits.rows) + ", " + std::to_string(logits.cols) + ")");
    }
    if (tokens.rows != C || tokens.cols != T) fail(op, "tokens must be (C, T)");

    const int rows = C * T;
    if (pred.rows != C || pred.cols != T || pred.dtype != Dtype::INT32) {
        pred.resize(C, T, Dtype::INT32);
    }
    if (scores.rows != C || scores.cols != T || scores.dtype != Dtype::FP32) {
        scores.resize(C, T, Dtype::FP32);
    }
    if (confidence.rows != C || confidence.cols != T || confidence.dtype != Dtype::FP32) {
        confidence.resize(C, T, Dtype::FP32);
    }

    // Top-k class filter width, fixed per call (ceil(frac * V), clamped).
    int k_keep = V;
    if (class_temperature > 0.0f) {
        const double kd = std::ceil(static_cast<double>(class_top_frac) * V);
        k_keep = (kd < 1.0) ? 1 : (kd > V ? V : static_cast<int>(kd));
    }

    const float* L = logits.host_f32();
    const int32_t* tok = static_cast<const int32_t*>(tokens.host_raw());
    int32_t* pr = static_cast<int32_t*>(pred.host_raw_mut());
    float* sc = scores.host_f32_mut();
    float* cf = confidence.host_f32_mut();
    const std::size_t row_stride = static_cast<std::size_t>(C) * V;

    parallel_for(static_cast<std::size_t>(rows), [&](std::size_t pi) {
        const int p = static_cast<int>(pi);
        const int c = p / T;
        const int t = p - c * T;
        const float* cond = L + static_cast<std::size_t>(t) * row_stride +
                            static_cast<std::size_t>(c) * V;

        thread_local std::vector<float> lp_buf, sort_buf;
        if (static_cast<int>(lp_buf.size()) < V) {
            lp_buf.resize(V);
            sort_buf.resize(V);
        }
        float* lp = lp_buf.data();

        // ── log_probs ──
        float dummy = 0.0f;
        if (guided) {
            const float* unc = L + static_cast<std::size_t>(T + t) * row_stride +
                               static_cast<std::size_t>(c) * V;
            const float lse_c = row_lse(cond, V, dummy);
            const float lse_u = row_lse(unc, V, dummy);
            for (int v = 0; v < V; ++v) {
                const float a = cond[v] - lse_c;
                const float b = unc[v] - lse_u;
                lp[v] = a + guidance_scale * (a - b);
            }
            const float lse_g = row_lse(lp, V, dummy);
            for (int v = 0; v < V; ++v) lp[v] = lp[v] - lse_g;
        } else {
            const float lse = row_lse(cond, V, dummy);
            for (int v = 0; v < V; ++v) lp[v] = cond[v] - lse;
        }
        lp[mask_id] = kNegInf;

        // ── confidence: max of the UNfiltered log_probs ──
        float conf = kNegInf;
        for (int v = 0; v < V; ++v) conf = (lp[v] > conf) ? lp[v] : conf;
        cf[p] = conf;   // raw, for every cell — masked or already decided

        // ── prediction ──
        int best = 0;
        if (class_temperature > 0.0f) {
            // k-th largest value of the row (ties resolved by index below).
            float* srt = sort_buf.data();
            std::copy(lp, lp + V, srt);
            std::nth_element(srt, srt + (k_keep - 1), srt + V,
                             [](float a, float b) { return a > b; });
            const float thr = srt[k_keep - 1];
            int count_gt = 0;
            for (int v = 0; v < V; ++v) count_gt += (lp[v] > thr) ? 1 : 0;
            const int m = k_keep - count_gt;   // equal-to-threshold slots

            const uint64_t base = static_cast<uint64_t>(p) * static_cast<uint64_t>(V);
            float best_s = kNegInf;
            int seen_eq = 0;
            for (int v = 0; v < V; ++v) {
                bool kept = lp[v] > thr;
                if (!kept && lp[v] == thr) {
                    kept = seen_eq < m;
                    ++seen_eq;
                }
                float s = kNegInf;
                if (kept) {
                    const float u = hash_uniform(seed, kHashDomainMaskedDiffusionClass,
                                                 base + static_cast<uint64_t>(v));
                    s = lp[v] / class_temperature + gumbel_from_uniform(u);
                }
                if (prefers(s, v, best_s, best)) { best_s = s; best = v; }
            }
        } else {
            float best_v = kNegInf;
            for (int v = 0; v < V; ++v) {
                if (prefers(lp[v], v, best_v, best)) { best_v = lp[v]; best = v; }
            }
        }
        pr[p] = best;

        // ── score ──
        float s = kNegInf;
        if (tok[p] == mask_id) {
            s = conf - static_cast<float>(c) * layer_penalty;
            if (position_temperature > 0.0f) {
                const float u = hash_uniform(seed, kHashDomainMaskedDiffusionPosition,
                                             static_cast<uint64_t>(p));
                s = s / position_temperature + gumbel_from_uniform(u);
            }
        }
        sc[p] = s;
    });
}

// ─── masked_diffusion_commit ────────────────────────────────────────────────

void masked_diffusion_commit(const ::brotensor::Tensor& pred,
                             const ::brotensor::Tensor& idx, int k, int step,
                             ::brotensor::Tensor& tokens,
                             ::brotensor::Tensor& unmask_step) {
    const char* op = "masked_diffusion_commit";
    if (pred.dtype != Dtype::INT32) fail(op, "pred must be INT32");
    if (idx.dtype != Dtype::INT32) fail(op, "idx must be INT32");
    if (tokens.dtype != Dtype::INT32) fail(op, "tokens must be INT32");
    if (unmask_step.dtype != Dtype::INT32) fail(op, "unmask_step must be INT32");
    if (tokens.rows != pred.rows || tokens.cols != pred.cols ||
        unmask_step.rows != pred.rows || unmask_step.cols != pred.cols) {
        fail(op, "pred, tokens and unmask_step must share one (C, T) shape");
    }
    if (k < 0) fail(op, "k must be >= 0");
    if (k > idx.size()) fail(op, "idx must hold at least k entries");
    if (k == 0) return;

    const int n = pred.size();
    const int32_t* pr = static_cast<const int32_t*>(pred.host_raw());
    const int32_t* ix = static_cast<const int32_t*>(idx.host_raw());
    int32_t* tok = static_cast<int32_t*>(tokens.host_raw_mut());
    int32_t* us = static_cast<int32_t*>(unmask_step.host_raw_mut());
    for (int i = 0; i < k; ++i) {
        const int32_t p = ix[i];
        if (p < 0 || p >= n) continue;
        tok[p] = pr[p];
        us[p] = step;
    }
}

}  // namespace brotensor::detail::cpu
