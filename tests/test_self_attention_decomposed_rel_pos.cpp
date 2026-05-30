// CPU coverage for self_attention_decomposed_rel_pos_forward — the SAM/ViTDet
// image-encoder attention (decomposed 2D relative-position bias).
//
// Two independent cross-checks, both CPU-only (no GPU required):
//   1. Equivalence: with NO projection biases, the fused op must equal
//      self_attention_bias_forward fed an explicitly MATERIALISED decomposed
//      bias (built here from a separately-projected, unscaled Q). This checks
//      the rel-pos factorisation against the independent static-bias kernel.
//   2. Full path: with projection biases, the fused op must equal a plain
//      from-scratch re-derivation (project+bias -> decomposed rel-pos ->
//      softmax -> AV -> out-proj+bias) computed in this test. This checks the
//      bias threading and grid (qh,qw) mapping end to end.

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using brotensor::Tensor;

namespace {

int failures = 0;

struct Rng {  // SplitMix64 -> [-1,1)
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float next(float scale) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return (static_cast<float>(z >> 40) / 16777216.0f * 2.0f - 1.0f) * scale;
    }
};

Tensor rand_mat(int r, int c, Rng& rng, float scale) {
    Tensor t = Tensor::mat(r, c);
    for (int i = 0; i < r * c; ++i) t.ptr()[i] = rng.next(scale);
    return t;
}

void compare(const Tensor& a, const Tensor& b, const char* tag, float tol) {
    if (a.rows != b.rows || a.cols != b.cols) {
        std::fprintf(stderr, "FAIL %s: shape %dx%d vs %dx%d\n", tag,
                     a.rows, a.cols, b.rows, b.cols);
        ++failures; return;
    }
    float worst = 0.0f;
    for (int i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    if (worst > tol) {
        std::fprintf(stderr, "FAIL %s: max abs diff %g > %g\n", tag, worst, tol);
        ++failures;
    }
}

// Per-head projection used to build the reference bias: row o = hh*dh+j of W,
// dotted with x over D, optional bias b[o]. Out laid out (H*L, dh).
std::vector<float> project(const Tensor& X, const Tensor& W, const float* b,
                           int H, int dh) {
    const int L = X.rows, D = X.cols;
    std::vector<float> out(static_cast<size_t>(H) * L * dh);
    const float* Xp = X.host_f32();
    const float* Wp = W.host_f32();
    for (int hh = 0; hh < H; ++hh)
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < dh; ++j) {
                const int o = hh * dh + j;
                const float* xr = Xp + static_cast<size_t>(i) * D;
                const float* wr = Wp + static_cast<size_t>(o) * D;
                float acc = b ? b[o] : 0.0f;
                for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
                out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc;
            }
    return out;
}

// Build the materialised (H*L, L) decomposed bias from an unscaled Q.
Tensor materialized_bias(const std::vector<float>& Q, const Tensor& rh,
                         const Tensor& rw, int H, int L, int dh,
                         int gh, int gw) {
    Tensor bias = Tensor::mat(H * L, L);
    const float* Rh = rh.host_f32();
    const float* Rw = rw.host_f32();
    for (int hh = 0; hh < H; ++hh)
        for (int i = 0; i < L; ++i) {
            const float* q = &Q[(static_cast<size_t>(hh) * L + i) * dh];
            const int qh = i / gw, qw = i % gw;
            for (int j = 0; j < L; ++j) {
                const int kh = j / gw, kw = j % gw;
                const float* rhr = Rh + static_cast<size_t>(qh - kh + gh - 1) * dh;
                const float* rwr = Rw + static_cast<size_t>(qw - kw + gw - 1) * dh;
                float bh = 0.0f, bw = 0.0f;
                for (int c = 0; c < dh; ++c) { bh += q[c] * rhr[c]; bw += q[c] * rwr[c]; }
                bias[(static_cast<size_t>(hh) * L + i) * L + j] = bh + bw;
            }
        }
    return bias;
}

// Full from-scratch reference, including projection + output biases.
Tensor manual_ref(const Tensor& X, const Tensor& Wq, const float* bq,
                  const Tensor& Wk, const float* bk, const Tensor& Wv,
                  const float* bv, const Tensor& Wo, const float* bo,
                  const Tensor& rh, const Tensor& rw, int H, int gh, int gw,
                  float scale) {
    const int L = X.rows, D = X.cols, dh = D / H;
    auto Q = project(X, Wq, bq, H, dh);
    auto K = project(X, Wk, bk, H, dh);
    auto V = project(X, Wv, bv, H, dh);
    const float* Rh = rh.host_f32();
    const float* Rw = rw.host_f32();

    std::vector<float> Yc(static_cast<size_t>(L) * D, 0.0f);
    std::vector<float> srow(L);
    for (int hh = 0; hh < H; ++hh)
        for (int i = 0; i < L; ++i) {
            const float* q = &Q[(static_cast<size_t>(hh) * L + i) * dh];
            const int qh = i / gw, qw = i % gw;
            float mx = -1e30f;
            for (int j = 0; j < L; ++j) {
                const float* k = &K[(static_cast<size_t>(hh) * L + j) * dh];
                const int kh = j / gw, kw = j % gw;
                const float* rhr = Rh + static_cast<size_t>(qh - kh + gh - 1) * dh;
                const float* rwr = Rw + static_cast<size_t>(qw - kw + gw - 1) * dh;
                float dot = 0.0f, bh = 0.0f, bw = 0.0f;
                for (int c = 0; c < dh; ++c) {
                    dot += q[c] * k[c]; bh += q[c] * rhr[c]; bw += q[c] * rwr[c];
                }
                srow[j] = dot * scale + bh + bw;
                mx = std::max(mx, srow[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < L; ++j) { srow[j] = std::exp(srow[j] - mx); sum += srow[j]; }
            const float inv = 1.0f / sum;
            for (int c = 0; c < dh; ++c) {
                float acc = 0.0f;
                for (int j = 0; j < L; ++j)
                    acc += srow[j] * inv * V[(static_cast<size_t>(hh) * L + j) * dh + c];
                Yc[static_cast<size_t>(i) * D + hh * dh + c] = acc;
            }
        }

    Tensor O = Tensor::mat(L, D);
    const float* Wop = Wo.host_f32();
    for (int i = 0; i < L; ++i)
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<size_t>(c) * D;
            float acc = bo ? bo[c] : 0.0f;
            for (int k = 0; k < D; ++k) acc += Yc[static_cast<size_t>(i) * D + k] * wr[k];
            O[static_cast<size_t>(i) * D + c] = acc;
        }
    return O;
}

void run(int gh, int gw, int D, int H, uint64_t seed) {
    Rng rng(seed);
    const int L = gh * gw, dh = D / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    Tensor X  = rand_mat(L, D, rng, 0.5f);
    Tensor Wq = rand_mat(D, D, rng, 0.3f), Wk = rand_mat(D, D, rng, 0.3f);
    Tensor Wv = rand_mat(D, D, rng, 0.3f), Wo = rand_mat(D, D, rng, 0.3f);
    Tensor rh = rand_mat(2 * gh - 1, dh, rng, 0.4f);
    Tensor rw = rand_mat(2 * gw - 1, dh, rng, 0.4f);
    Tensor bq = rand_mat(D, 1, rng, 0.2f), bk = rand_mat(D, 1, rng, 0.2f);
    Tensor bv = rand_mat(D, 1, rng, 0.2f), bo = rand_mat(D, 1, rng, 0.2f);

    char tag[96];

    // 1. No projection bias == self_attention_bias_forward(materialized bias).
    {
        auto Q = project(X, Wq, nullptr, H, dh);
        Tensor bias = materialized_bias(Q, rh, rw, H, L, dh, gh, gw);
        Tensor O_ref;
        brotensor::self_attention_bias_forward(X, Wq, Wk, Wv, Wo, nullptr,
                                               &bias, H, scale, O_ref);
        Tensor O_fused;
        brotensor::self_attention_decomposed_rel_pos_forward(
            X, Wq, nullptr, Wk, nullptr, Wv, nullptr, Wo, nullptr,
            rh, rw, H, gh, gw, scale, O_fused);
        std::snprintf(tag, sizeof tag, "equiv g%dx%d D%d H%d", gh, gw, D, H);
        compare(O_ref, O_fused, tag, 1e-4f);
    }

    // 2. With projection biases == full plain-loop reference.
    {
        Tensor O_ref = manual_ref(X, Wq, bq.host_f32(), Wk, bk.host_f32(),
                                  Wv, bv.host_f32(), Wo, bo.host_f32(),
                                  rh, rw, H, gh, gw, scale);
        Tensor O_fused;
        brotensor::self_attention_decomposed_rel_pos_forward(
            X, Wq, &bq, Wk, &bk, Wv, &bv, Wo, &bo, rh, rw,
            H, gh, gw, scale, O_fused);
        std::snprintf(tag, sizeof tag, "bias  g%dx%d D%d H%d", gh, gw, D, H);
        compare(O_ref, O_fused, tag, 1e-4f);
    }
}

}  // namespace

int main() {
    run(2, 3, 16, 2, 0x1001ull);  // non-square small grid
    run(4, 4, 32, 4, 0x1002ull);  // window-sized
    run(8, 8, 64, 8, 0x1003ull);  // global-block-sized
    run(5, 7, 24, 3, 0x1004ull);  // odd dims, 3 heads

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("self_attention_decomposed_rel_pos: all checks passed\n");
    return 0;
}
