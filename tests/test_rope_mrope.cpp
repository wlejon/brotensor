// CPU smoke test for rope_apply_mrope (FP32).
//
// Two-pronged correctness:
//   1) Degenerate single-axis case (d_h = d_w = 0, pos_t = {0..L-1}) with
//      cos_t / sin_t built from the standard RoPE formula reproduces
//      rope_apply bit-for-bit (well, to float-epsilon — both paths are FP32).
//   2) Non-trivial 3-axis case (d_t, d_h, d_w all > 0) checked against a
//      naive Python-equivalent reference written inline.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Tensor;
using brotensor::Dtype;

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

namespace {

// Build standard-formula cos/sin tables of shape (max_pos, num_pairs):
//   cos[p, i] = cos(p * base^{-2i/(2*num_pairs)}),
//   sin[p, i] = sin(p * base^{-2i/(2*num_pairs)}).
// (head_dim here is the FULL head_dim associated with num_pairs = head_dim/2.)
void build_tables(int max_pos, int num_pairs, int eff_head_dim, float base,
                  Tensor& cos_tbl, Tensor& sin_tbl) {
    cos_tbl = Tensor::mat(max_pos, num_pairs);
    sin_tbl = Tensor::mat(max_pos, num_pairs);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < num_pairs; ++i) {
            const float theta_i = std::exp(
                -(float)(2 * i) / (float)eff_head_dim * std::log(base));
            const float angle = (float)p * theta_i;
            cos_tbl.ptr()[p * num_pairs + i] = std::cos(angle);
            sin_tbl.ptr()[p * num_pairs + i] = std::sin(angle);
        }
    }
}

void test_degenerate_matches_rope_apply() {
    std::printf("  degenerate single-axis matches rope_apply\n");
    const int L = 8;
    const int head_dim = 8;
    const int num_heads = 2;
    const float base = 10000.0f;

    // Reference: stitched tables for rope_apply.
    Tensor cos_ref, sin_ref;
    build_tables(L, head_dim / 2, head_dim, base, cos_ref, sin_ref);

    std::mt19937 rng(0xD001);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    for (int i = 0; i < X.size(); ++i) X.ptr()[i] = dist(rng);

    // rope_apply ground truth.
    Tensor Y_ref;
    brotensor::rope_apply(X, cos_ref, sin_ref, head_dim, num_heads, Y_ref);

    // M-RoPE with d_t = head_dim/2, d_h = d_w = 0, pos_t = {0..L-1}.
    const int d_t = head_dim / 2;
    Tensor cos_t, sin_t;
    build_tables(L, d_t, head_dim, base, cos_t, sin_t);
    Tensor cos_empty, sin_empty;  // unused — d_h = d_w = 0.

    std::vector<int32_t> pos_t(L);
    for (int i = 0; i < L; ++i) pos_t[i] = i;
    Tensor Y;
    brotensor::rope_apply_mrope(
        X, cos_t, sin_t, cos_empty, sin_empty, cos_empty, sin_empty,
        pos_t.data(), nullptr, nullptr,
        head_dim, num_heads, d_t, 0, 0, Y);

    CHECK(Y.rows == L);
    CHECK(Y.cols == num_heads * head_dim);
    const float* Yp = Y.host_f32();
    const float* Rp = Y_ref.host_f32();
    int bad = 0;
    float max_err = 0.0f;
    for (int i = 0; i < Y.size(); ++i) {
        const float err = std::fabs(Yp[i] - Rp[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-6f) {
            if (bad < 5) std::printf("    mismatch i=%d got=%g ref=%g err=%g\n",
                                     i, Yp[i], Rp[i], err);
            ++bad;
        }
    }
    std::printf("    max_err=%g  bad=%d / %d\n", max_err, bad, Y.size());
    CHECK(bad == 0);
}

void test_three_axis_against_naive() {
    std::printf("  3-axis vs naive reference\n");
    const int L = 12;
    const int num_heads = 2;
    const int d_t = 2;
    const int d_h = 3;
    const int d_w = 3;
    const int head_dim = 2 * (d_t + d_h + d_w);  // 16
    const float base = 10000.0f;

    const int max_pos_t = 16;
    const int max_pos_h = 24;
    const int max_pos_w = 24;
    Tensor cos_t, sin_t, cos_h, sin_h, cos_w, sin_w;
    build_tables(max_pos_t, d_t, head_dim, base, cos_t, sin_t);
    build_tables(max_pos_h, d_h, head_dim, base, cos_h, sin_h);
    build_tables(max_pos_w, d_w, head_dim, base, cos_w, sin_w);

    std::mt19937 rng(0xD002);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pdist_t(0, max_pos_t - 1);
    std::uniform_int_distribution<int> pdist_h(0, max_pos_h - 1);
    std::uniform_int_distribution<int> pdist_w(0, max_pos_w - 1);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    for (int i = 0; i < X.size(); ++i) X.ptr()[i] = dist(rng);
    std::vector<int32_t> pos_t(L), pos_h(L), pos_w(L);
    for (int i = 0; i < L; ++i) {
        pos_t[i] = pdist_t(rng);
        pos_h[i] = pdist_h(rng);
        pos_w[i] = pdist_w(rng);
    }

    Tensor Y;
    brotensor::rope_apply_mrope(
        X, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
        pos_t.data(), pos_h.data(), pos_w.data(),
        head_dim, num_heads, d_t, d_h, d_w, Y);

    // Naive reference.
    const float* Xp = X.host_f32();
    std::vector<float> Y_ref(X.size());
    const int D = num_heads * head_dim;
    auto apply_axis = [&](const float* cos_a, const float* sin_a,
                          const int32_t* pos_a,
                          int pair_off, int d_a) {
        if (d_a == 0) return;
        for (int row = 0; row < L; ++row) {
            const int pos = pos_a[row];
            for (int h = 0; h < num_heads; ++h) {
                const int base_off = row * D + h * head_dim + 2 * pair_off;
                for (int i = 0; i < d_a; ++i) {
                    const float c = cos_a[pos * d_a + i];
                    const float s = sin_a[pos * d_a + i];
                    const float x0 = Xp[base_off + 2 * i];
                    const float x1 = Xp[base_off + 2 * i + 1];
                    Y_ref[base_off + 2 * i]     = x0 * c - x1 * s;
                    Y_ref[base_off + 2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    };
    // Pre-fill identity for completeness (here every slot is covered by one axis).
    for (int i = 0; i < (int)Y_ref.size(); ++i) Y_ref[i] = Xp[i];
    apply_axis(cos_t.host_f32(), sin_t.host_f32(), pos_t.data(), 0, d_t);
    apply_axis(cos_h.host_f32(), sin_h.host_f32(), pos_h.data(), d_t, d_h);
    apply_axis(cos_w.host_f32(), sin_w.host_f32(), pos_w.data(),
               d_t + d_h, d_w);

    const float* Yp = Y.host_f32();
    int bad = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < Y_ref.size(); ++i) {
        const float err = std::fabs(Yp[i] - Y_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-5f) {
            if (bad < 5) std::printf("    mismatch i=%zu got=%g ref=%g err=%g\n",
                                     i, Yp[i], Y_ref[i], err);
            ++bad;
        }
    }
    std::printf("    max_err=%g  bad=%d / %zu\n", max_err, bad, Y_ref.size());
    CHECK(bad == 0);
}

void test_invalid_split_rejected() {
    Tensor X = Tensor::mat(2, 1 * 8);
    Tensor c = Tensor::mat(4, 2);
    Tensor s = Tensor::mat(4, 2);
    Tensor Y;
    std::vector<int32_t> pos = {0, 1};
    bool threw = false;
    try {
        // 2*(d_t + d_h + d_w) = 6 != 8.
        brotensor::rope_apply_mrope(X, c, s, c, s, c, s,
                                    pos.data(), pos.data(), pos.data(),
                                    8, 1, 1, 1, 1, Y);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

} // namespace

int main() {
    brotensor::init();
    std::printf("test_rope_mrope\n===============\n");
    test_degenerate_matches_rope_apply();
    test_three_axis_against_naive();
    test_invalid_split_rejected();
    if (g_failures != 0) {
        std::printf("\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall passed\n");
    return 0;
}
