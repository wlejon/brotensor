// ─── masked_diffusion_scores / masked_diffusion_commit ──────────────────────
//
// Coverage:
//   1. Small hand-checkable grid (C=2, T=3, V=5) against a double-precision
//      reference of the upstream pseudocode written here, over every switch:
//      guidance_scale 0 (R == T) and != 0 (R == 2T), position_temperature 0
//      and > 0, class_temperature 0 (argmax) and > 0 (top-k Gumbel sampling),
//      and already-unmasked cells scoring -inf.
//   2. Argument validation throws.
//   3. commit: tokens / unmask_step change only at the k indices; out-of-
//      range indices are ignored.
//   4. top_k_rows over one long row (32768) with k from 1 to the full length,
//      ties included, against a sort reference — CPU, then CUDA parity.
//   5. CPU vs CUDA parity on the real shape (C=8, T=400, V=1025): pred
//      exact, scores to tolerance, and the unmask decision after
//      top_k_rows + commit exact. The CUDA half skips when no CUDA backend is
//      registered.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>
#include <brotensor/detail/hash_rng.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ── deterministic RNG for inputs ──
struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float next_f01() { return static_cast<float>(next_u64() >> 40) / 16777216.0f; }
    // Roughly Gaussian (sum of four uniforms), scaled.
    float next_gauss(float scale) {
        float a = 0.0f;
        for (int i = 0; i < 4; ++i) a += next_f01();
        return (a - 2.0f) * scale;
    }
};

// ── tensor helpers ──
Tensor make_f32(int r, int c, const std::vector<float>& v) {
    Tensor t = Tensor::zeros_on(Device::CPU, r, c, Dtype::FP32);
    std::copy(v.begin(), v.end(), t.host_f32_mut());
    return t;
}
Tensor make_i32(int r, int c, const std::vector<int32_t>& v) {
    Tensor t = Tensor::zeros_on(Device::CPU, r, c, Dtype::INT32);
    std::copy(v.begin(), v.end(), static_cast<int32_t*>(t.host_raw_mut()));
    return t;
}
std::vector<int32_t> i32_of(const Tensor& t) {
    const int32_t* p = static_cast<const int32_t*>(t.host_raw());
    return std::vector<int32_t>(p, p + t.size());
}
std::vector<float> f32_of(const Tensor& t) {
    const float* p = t.host_f32();
    return std::vector<float>(p, p + t.size());
}
Tensor to_host(const Tensor& g) {
    brotensor::sync_all();
    return g.to(Device::CPU);
}

// ── double-precision reference of the upstream pseudocode ──
struct Params {
    int T, C, V, mask_id;
    float gs, pen, pos_temp, class_temp, top_frac;
    uint64_t seed;
};

struct Ref {
    std::vector<int> pred;
    std::vector<double> score;
};

double gumbel_ref(float u) {
    return -std::log(-std::log(static_cast<double>(u) + 1e-10) + 1e-10);
}

std::vector<double> log_softmax_ref(const std::vector<double>& x) {
    double m = -kInf;
    for (double v : x) m = std::max(m, v);
    double s = 0.0;
    for (double v : x) s += std::exp(v - m);
    const double lse = m + std::log(s);
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] - lse;
    return out;
}

Ref reference(const std::vector<float>& logits, const std::vector<int32_t>& tokens,
              const Params& P) {
    const int T = P.T, C = P.C, V = P.V;
    Ref r;
    r.pred.assign(C * T, 0);
    r.score.assign(C * T, 0.0);
    const size_t stride = static_cast<size_t>(C) * V;
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            const int p = c * T + t;
            std::vector<double> cl(V), ul(V);
            for (int v = 0; v < V; ++v) cl[v] = logits[t * stride + c * V + v];
            std::vector<double> lp;
            if (P.gs != 0.0f) {
                for (int v = 0; v < V; ++v) ul[v] = logits[(T + t) * stride + c * V + v];
                std::vector<double> lc = log_softmax_ref(cl), lu = log_softmax_ref(ul);
                std::vector<double> g(V);
                for (int v = 0; v < V; ++v) g[v] = lc[v] + P.gs * (lc[v] - lu[v]);
                lp = log_softmax_ref(g);
            } else {
                lp = log_softmax_ref(cl);
            }
            lp[P.mask_id] = -kInf;
            double conf = -kInf;
            for (double v : lp) conf = std::max(conf, v);

            int pred = 0;
            if (P.class_temp > 0.0f) {
                int k = static_cast<int>(std::ceil(static_cast<double>(P.top_frac) * V));
                k = std::max(1, std::min(V, k));
                std::vector<int> order(V);
                for (int v = 0; v < V; ++v) order[v] = v;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    if (lp[a] != lp[b]) return lp[a] > lp[b];
                    return a < b;
                });
                std::vector<double> filt(V, -kInf);
                for (int i = 0; i < k; ++i) filt[order[i]] = lp[order[i]];
                double best = -kInf;
                pred = 0;
                for (int v = 0; v < V; ++v) {
                    double s = -kInf;
                    if (filt[v] != -kInf) {
                        const float u = brotensor::detail::hash_uniform(
                            P.seed, brotensor::detail::kHashDomainMaskedDiffusionClass,
                            static_cast<uint64_t>(p) * V + v);
                        s = filt[v] / P.class_temp + gumbel_ref(u);
                    }
                    if (s > best) { best = s; pred = v; }
                }
            } else {
                double best = -kInf;
                for (int v = 0; v < V; ++v) {
                    if (lp[v] > best) { best = lp[v]; pred = v; }
                }
            }
            r.pred[p] = pred;

            double s = conf - c * static_cast<double>(P.pen);
            if (P.pos_temp > 0.0f) {
                const float u = brotensor::detail::hash_uniform(
                    P.seed, brotensor::detail::kHashDomainMaskedDiffusionPosition,
                    static_cast<uint64_t>(p));
                s = s / P.pos_temp + gumbel_ref(u);
            }
            if (tokens[p] != P.mask_id) s = -kInf;
            r.score[p] = s;
        }
    }
    return r;
}

bool close(double a, double b, double atol, double rtol) {
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::fabs(a - b) <= atol + rtol * std::fabs(a);
}

// Run the CPU op on a case and compare with the reference.
void check_case(const char* name, const std::vector<float>& logits,
                const std::vector<int32_t>& tokens, const Params& P) {
    const int R = (P.gs != 0.0f) ? 2 * P.T : P.T;
    Tensor L = make_f32(R, P.C * P.V, logits);
    Tensor tok = make_i32(P.C, P.T, tokens);
    Tensor pred, scores;
    brotensor::masked_diffusion_scores(L, tok, P.T, P.C, P.V, P.mask_id,
                                       P.gs, P.pen, P.pos_temp, P.class_temp,
                                       P.top_frac, P.seed, pred, scores);
    CHECK(pred.rows == P.C && pred.cols == P.T && pred.dtype == Dtype::INT32);
    CHECK(scores.rows == P.C && scores.cols == P.T && scores.dtype == Dtype::FP32);

    const Ref ref = reference(logits, tokens, P);
    const std::vector<int32_t> pr = i32_of(pred);
    const std::vector<float> sc = f32_of(scores);
    int bad = 0;
    for (int p = 0; p < P.C * P.T; ++p) {
        if (pr[p] != ref.pred[p]) {
            ++bad;
            std::printf("    [%s] pred mismatch at p=%d: op=%d ref=%d\n", name, p, pr[p], ref.pred[p]);
        }
        if (pr[p] == P.mask_id) {
            ++bad;
            std::printf("    [%s] pred is mask_id at p=%d\n", name, p);
        }
        if (!close(ref.score[p], sc[p], 1e-4, 1e-5)) {
            ++bad;
            std::printf("    [%s] score mismatch at p=%d: op=%.7g ref=%.7g\n", name, p, sc[p], ref.score[p]);
        }
        if (tokens[p] != P.mask_id && !(sc[p] == -std::numeric_limits<float>::infinity())) {
            ++bad;
            std::printf("    [%s] unmasked cell p=%d not -inf: %g\n", name, p, sc[p]);
        }
    }
    CHECK(bad == 0);
    std::printf("  %s  %s\n", bad == 0 ? "ok  " : "FAIL", name);
}

// ── 1. small grid, every switch ──
void test_small_grid() {
    const int T = 3, C = 2, V = 5, mask_id = 4;
    // Conditional rows [t][c*V + v], then unconditional rows.
    const std::vector<float> cond = {
        // t = 0: c0                      c1
        1.0f, 2.0f, 3.0f, 0.5f, 9.0f,     0.2f, 0.1f, 0.3f, 0.25f, -1.0f,
        // t = 1
        -2.0f, 4.0f, 4.0f, 1.0f, 0.0f,    3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        // t = 2
        0.0f, -1.0f, -3.0f, 2.5f, 2.6f,   1.5f, 1.4f, 1.6f, 1.7f, 0.0f,
    };
    const std::vector<float> unc = {
        0.5f, 0.5f, 2.5f, 0.0f, 1.0f,     0.0f, 0.4f, 0.1f, 0.2f, 0.3f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,     2.0f, -2.0f, 0.5f, 0.0f, 1.0f,
        -1.0f, 0.0f, 1.0f, 0.5f, 0.2f,    0.7f, 0.9f, 0.6f, 0.6f, 0.1f,
    };
    std::vector<float> both = cond;
    both.insert(both.end(), unc.begin(), unc.end());
    // tokens (C, T): cell (c1, t1) and (c0, t2) already decided.
    const std::vector<int32_t> tokens = {
        mask_id, mask_id, 2,
        mask_id, 1, mask_id,
    };

    Params P{T, C, V, mask_id, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 7ull};
    check_case("gs=0 greedy no-noise", cond, tokens, P);

    P = Params{T, C, V, mask_id, 0.0f, 0.3f, 0.0f, 0.0f, 0.5f, 7ull};
    check_case("gs=0 layer penalty", cond, tokens, P);

    P = Params{T, C, V, mask_id, 1.5f, 0.3f, 0.0f, 0.0f, 0.5f, 7ull};
    check_case("gs=1.5 greedy", both, tokens, P);

    P = Params{T, C, V, mask_id, 1.5f, 0.3f, 0.8f, 0.0f, 0.5f, 11ull};
    check_case("gs=1.5 position gumbel", both, tokens, P);

    P = Params{T, C, V, mask_id, 0.0f, 0.3f, 0.0f, 1.0f, 0.5f, 13ull};
    check_case("gs=0 class sampling top-50%", cond, tokens, P);

    P = Params{T, C, V, mask_id, 2.0f, 0.5f, 0.7f, 1.3f, 0.5f, 17ull};
    check_case("gs=2 both temperatures", both, tokens, P);

    // Different seeds change the noise (and the same seed repeats it).
    {
        Tensor L = make_f32(2 * T, C * V, both);
        Tensor tok = make_i32(C, T, tokens);
        Tensor pa, sa, pb, sb, pc, sc;
        brotensor::masked_diffusion_scores(L, tok, T, C, V, mask_id, 2.0f, 0.5f, 0.7f, 0.0f, 0.5f, 1ull, pa, sa);
        brotensor::masked_diffusion_scores(L, tok, T, C, V, mask_id, 2.0f, 0.5f, 0.7f, 0.0f, 0.5f, 2ull, pb, sb);
        brotensor::masked_diffusion_scores(L, tok, T, C, V, mask_id, 2.0f, 0.5f, 0.7f, 0.0f, 0.5f, 1ull, pc, sc);
        CHECK(f32_of(sa) != f32_of(sb));
        CHECK(f32_of(sa) == f32_of(sc));
    }
}

// ── 2. validation ──
void test_throws() {
    const int T = 2, C = 1, V = 3;
    Tensor L = Tensor::zeros_on(Device::CPU, T, C * V);
    Tensor tok = Tensor::zeros_on(Device::CPU, C, T, Dtype::INT32);
    Tensor pred, scores;
    bool threw = false;
    try {   // guidance on but only T rows
        brotensor::masked_diffusion_scores(L, tok, T, C, V, 2, 1.0f, 0, 0, 0, 0.1f, 0, pred, scores);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {   // mask_id out of range
        brotensor::masked_diffusion_scores(L, tok, T, C, V, 3, 0.0f, 0, 0, 0, 0.1f, 0, pred, scores);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {   // tokens wrong shape
        Tensor bad = Tensor::zeros_on(Device::CPU, T, C, Dtype::INT32);
        brotensor::masked_diffusion_scores(L, bad, T, C, V, 2, 0.0f, 0, 0, 0, 0.1f, 0, pred, scores);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {   // commit: k larger than idx
        Tensor idx = Tensor::zeros_on(Device::CPU, 1, 1, Dtype::INT32);
        Tensor us = Tensor::zeros_on(Device::CPU, C, T, Dtype::INT32);
        brotensor::masked_diffusion_commit(tok, idx, 2, 0, tok, us);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── 3. commit ──
void test_commit_cpu() {
    const int C = 2, T = 4, mask_id = 99;
    std::vector<int32_t> tokens(C * T, mask_id), pred(C * T), us(C * T, -1);
    for (int i = 0; i < C * T; ++i) pred[i] = 100 + i;
    tokens[3] = 5;   // an already-decided cell that the commit must not touch
    Tensor tok = make_i32(C, T, tokens);
    Tensor pr = make_i32(C, T, pred);
    Tensor step = make_i32(C, T, us);
    Tensor idx = make_i32(1, 5, {6, 0, 7, -1, 2});   // 4th entry not read (k=3)
    brotensor::masked_diffusion_commit(pr, idx, 3, 4, tok, step);
    const std::vector<int32_t> t2 = i32_of(tok), s2 = i32_of(step);
    for (int i = 0; i < C * T; ++i) {
        const bool chosen = (i == 6 || i == 0 || i == 7);
        CHECK(t2[i] == (chosen ? pred[i] : tokens[i]));
        CHECK(s2[i] == (chosen ? 4 : -1));
    }
    // k = 0 is a no-op; an out-of-range index is ignored.
    brotensor::masked_diffusion_commit(pr, idx, 0, 9, tok, step);
    CHECK(i32_of(tok) == t2);
    Tensor bad = make_i32(1, 2, {-1, C * T});
    brotensor::masked_diffusion_commit(pr, bad, 2, 9, tok, step);
    CHECK(i32_of(tok) == t2 && i32_of(step) == s2);
}

// ── 4. top_k_rows over one long row, k anywhere in [1, C] ──
void check_top_k_row(const Tensor& X, int k, const std::vector<int>& order, const char* tag) {
    Tensor Vals, Idx;
    brotensor::top_k_rows(X, k, Vals, Idx);
    const std::vector<float> v = f32_of(to_host(Vals));
    const std::vector<int32_t> ix = i32_of(to_host(Idx));
    const std::vector<float> x = f32_of(to_host(X));
    CHECK(static_cast<int>(v.size()) == k && static_cast<int>(ix.size()) == k);
    int bad = 0;
    for (int j = 0; j < k; ++j) {
        if (ix[j] != order[j] || v[j] != x[order[j]]) ++bad;
    }
    if (bad) std::printf("    [%s] top_k_rows k=%d: %d wrong slots\n", tag, k, bad);
    CHECK(bad == 0);
}

void test_top_k_long_row(bool cuda) {
    const int C = 32768;
    SplitMix64 rng(0x70A5ull);
    std::vector<float> x(C);
    // Quantised so the row carries plenty of ties (~8 per level).
    for (int i = 0; i < C; ++i) x[i] = std::floor(rng.next_f01() * 4096.0f) / 64.0f - 30.0f;
    x[100] = x[200] = x[300] = 100.0f;   // a tie at the very top
    std::vector<int> order(C);
    for (int i = 0; i < C; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (x[a] != x[b]) return x[a] > x[b];
        return a < b;
    });
    Tensor X = make_f32(1, C, x);
    const int ks[] = {1, 2, 3, 63, 64, 65, 255, 256, 257, 1000, 4096, 6144, 6145, 20000, C - 1, C};
    for (int k : ks) check_top_k_row(X, k, order, "cpu");
    if (cuda) {
        Tensor gX = X.to(Device::CUDA);
        for (int k : ks) check_top_k_row(gX, k, order, "cuda");
    }
    // A few rows at once through the large-k path.
    {
        const int R = 3, W = 3000;
        std::vector<float> xs(R * W);
        for (float& f : xs) f = std::floor(rng.next_f01() * 500.0f);
        Tensor Xm = make_f32(R, W, xs);
        Tensor V1, I1;
        brotensor::top_k_rows(Xm, 2000, V1, I1);
        const std::vector<int32_t> i1 = i32_of(I1);
        for (int r = 0; r < R; ++r) {
            std::vector<int> o(W);
            for (int i = 0; i < W; ++i) o[i] = i;
            std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
                if (xs[r * W + a] != xs[r * W + b]) return xs[r * W + a] > xs[r * W + b];
                return a < b;
            });
            int bad = 0;
            for (int j = 0; j < 2000; ++j) bad += (i1[r * 2000 + j] != o[j]);
            CHECK(bad == 0);
        }
        if (cuda) {
            Tensor V2, I2;
            brotensor::top_k_rows(Xm.to(Device::CUDA), 2000, V2, I2);
            CHECK(i32_of(to_host(I2)) == i1);
            CHECK(f32_of(to_host(V2)) == f32_of(V1));
        }
    }
    std::printf("  %s  top_k_rows: one row of %d, k in {1 .. %d} with ties, %s\n",
                g_failures == 0 ? "ok  " : "FAIL", C, C,
                cuda ? "CPU + CUDA (identical)" : "CPU only");
}

// ── 5. CPU vs CUDA parity on the real shape ──
void parity_case(const char* name, const Params& P, uint64_t input_seed, int k_unmask) {
    const int T = P.T, C = P.C, V = P.V;
    const int R = (P.gs != 0.0f) ? 2 * T : T;
    SplitMix64 rng(input_seed);
    std::vector<float> logits(static_cast<size_t>(R) * C * V);
    for (float& f : logits) f = rng.next_gauss(3.0f);
    std::vector<int32_t> tokens(C * T);
    for (int32_t& t : tokens) {
        t = (rng.next_f01() < 0.6f) ? P.mask_id : static_cast<int32_t>(rng.next_f01() * (V - 1));
    }
    Tensor L = make_f32(R, C * V, logits);
    Tensor tok = make_i32(C, T, tokens);
    Tensor pred_c, sc_c;
    brotensor::masked_diffusion_scores(L, tok, T, C, V, P.mask_id, P.gs, P.pen,
                                       P.pos_temp, P.class_temp, P.top_frac, P.seed,
                                       pred_c, sc_c);
    Tensor gL = L.to(Device::CUDA), gtok = tok.to(Device::CUDA);
    Tensor pred_g, sc_g;
    brotensor::masked_diffusion_scores(gL, gtok, T, C, V, P.mask_id, P.gs, P.pen,
                                       P.pos_temp, P.class_temp, P.top_frac, P.seed,
                                       pred_g, sc_g);
    const std::vector<int32_t> pc = i32_of(pred_c), pg = i32_of(to_host(pred_g));
    const std::vector<float> scc = f32_of(sc_c), scg = f32_of(to_host(sc_g));
    int bad_pred = 0, bad_score = 0;
    float worst = 0.0f;
    for (int p = 0; p < C * T; ++p) {
        if (pc[p] != pg[p]) {
            if (bad_pred < 5) std::printf("    [%s] pred mismatch p=%d cpu=%d cuda=%d\n", name, p, pc[p], pg[p]);
            ++bad_pred;
        }
        if (!close(scc[p], scg[p], 1e-4, 1e-5)) {
            if (bad_score < 5) std::printf("    [%s] score mismatch p=%d cpu=%.8g cuda=%.8g\n", name, p, scc[p], scg[p]);
            ++bad_score;
        }
        if (!std::isinf(scc[p]) && !std::isinf(scg[p])) {
            worst = std::max(worst, std::fabs(scc[p] - scg[p]));
        }
    }
    CHECK(bad_pred == 0);
    CHECK(bad_score == 0);

    // Against the double reference too (pred exact, scores to tolerance).
    const Ref ref = reference(logits, tokens, P);
    int bad_ref = 0;
    for (int p = 0; p < C * T; ++p) {
        if (pc[p] != ref.pred[p] || !close(ref.score[p], scc[p], 1e-4, 1e-5)) ++bad_ref;
    }
    CHECK(bad_ref == 0);

    // top_k over the scores as one (1, C*T) row, then commit, on both sides.
    Tensor sv_c = Tensor::view(Device::CPU, sc_c.data, 1, C * T);
    Tensor sv_g = Tensor::view(Device::CUDA, sc_g.data, 1, C * T);
    Tensor tv_c, ti_c, tv_g, ti_g;
    brotensor::top_k_rows(sv_c, k_unmask, tv_c, ti_c);
    brotensor::top_k_rows(sv_g, k_unmask, tv_g, ti_g);
    const std::vector<int32_t> ic = i32_of(ti_c), ig = i32_of(to_host(ti_g));
    CHECK(ic == ig);

    Tensor tok_c = tok.clone(), tok_g = gtok.clone();
    Tensor us_c = Tensor::zeros_on(Device::CPU, C, T, Dtype::INT32);
    Tensor us_g = Tensor::zeros_on(Device::CUDA, C, T, Dtype::INT32);
    brotensor::masked_diffusion_commit(pred_c, ti_c, k_unmask, 3, tok_c, us_c);
    brotensor::masked_diffusion_commit(pred_g, ti_g, k_unmask, 3, tok_g, us_g);
    const std::vector<int32_t> tc = i32_of(tok_c), tg = i32_of(to_host(tok_g));
    const std::vector<int32_t> uc = i32_of(us_c), ug = i32_of(to_host(us_g));
    CHECK(tc == tg);
    CHECK(uc == ug);
    int changed = 0, wrong = 0;
    for (int p = 0; p < C * T; ++p) {
        if (uc[p] == 3) {
            ++changed;
            if (tc[p] != pc[p] || tokens[p] != P.mask_id) ++wrong;
        } else if (tc[p] != tokens[p]) {
            ++wrong;
        }
    }
    CHECK(changed == k_unmask);
    CHECK(wrong == 0);
    std::printf("  %s  parity %s: pred %d/%d mismatches, scores %d over tol (worst |d|=%.3g), "
                "unmask set %s\n",
                (bad_pred || bad_score || bad_ref || ic != ig || tc != tg) ? "FAIL" : "ok  ",
                name, bad_pred, C * T, bad_score, worst, ic == ig ? "identical" : "DIFFERS");
}

void test_parity() {
    const int T = 400, C = 8, V = 1025, mask_id = 1024;
    parity_case("gs=2 pos+class temps",
                Params{T, C, V, mask_id, 2.0f, 0.5f, 0.7f, 1.2f, 0.1f, 0xABCDull}, 0x1111ull, 150);
    parity_case("gs=0 greedy",
                Params{T, C, V, mask_id, 0.0f, 0.5f, 0.0f, 0.0f, 0.1f, 0x1234ull}, 0x2222ull, 300);
    parity_case("gs=1 pos temp only",
                Params{T, C, V, mask_id, 1.0f, 0.25f, 1.0f, 0.0f, 0.1f, 0x5555ull}, 0x3333ull, 1);
}

}  // namespace

int main() {
    brotensor::init();
    std::printf("test_masked_diffusion:\n");
    test_small_grid();
    test_throws();
    test_commit_cpu();
    const bool cuda = brotensor::is_available(Device::CUDA);
    test_top_k_long_row(cuda);
    if (cuda) {
        test_parity();
    } else {
        std::printf("  skip  CUDA backend not available; CPU-only run\n");
    }
    if (g_failures == 0) {
        std::printf("  OK  all masked_diffusion tests passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
