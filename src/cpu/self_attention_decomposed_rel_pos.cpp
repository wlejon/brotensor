// ─── CPU self-attention with decomposed 2D relative-position bias ──────────
//
// FP32 scalar host reference for the SAM / ViTDet image-encoder attention.
// Multi-head self-attention where the pre-softmax bias is the *decomposed*
// 2D relative-position term from Dosovitskiy-style ViTDet (segment_anything
// `add_decomposed_rel_pos`):
//
//   r_q = Q_h[q]                              (projected, UNSCALED query)
//   bias[h,q,k] = r_q . Rh[qh, kh] + r_q . Rw[qw, kw]
//   S[h,q,k]    = scale * (Q_h[q] . K_h[k]) + bias[h,q,k]
//   O           = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// where a token index t maps to grid coords (t / grid_w, t % grid_w) over a
// grid_h × grid_w patch grid (so L == grid_h*grid_w), and
//   Rh[qh, kh] = rel_pos_h[(qh - kh) + (grid_h - 1)]   (length head_dim)
//   Rw[qw, kw] = rel_pos_w[(qw - kw) + (grid_w - 1)].
//
// This is the q*size == k*size case of segment_anything's get_rel_pos (no
// rel-pos interpolation): rel_pos_h has exactly 2*grid_h-1 rows, indexed by the
// signed query−key row offset shifted into [0, 2*grid_h-2]. Windowed blocks
// call this per window (grid_h == grid_w == window); global blocks call it once
// over the full 64×64 grid. The bias is data-dependent (it reads Q), which is
// why it can't be expressed through the static-bias self_attention_bias op.
//
// Unlike that static (num_heads*L, L) bias, the decomposed form is never
// materialised: per (head, query) we precompute length-grid_h and length-grid_w
// dot-product vectors and add bias[k] = relh[kh] + relw[kw] inside the score
// loop — O(L*(grid_h+grid_w)*head_dim) work, not O(L*L*head_dim).
//
// qkv and output projections carry optional biases (SAM uses both). Wq/Wk/Wv/Wo
// are (D, D); per-head split takes contiguous weight rows hh*dh .. hh*dh+dh.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

inline void check_fp32(const ::brotensor::Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        throw std::runtime_error(
            std::string("self_attention_decomposed_rel_pos_forward: ") + name +
            " must be FP32 (CPU backend is FP32-only)");
    }
}

} // namespace

void self_attention_decomposed_rel_pos_forward(
        const ::brotensor::Tensor& X,
        const ::brotensor::Tensor& Wq, const ::brotensor::Tensor* bq,
        const ::brotensor::Tensor& Wk, const ::brotensor::Tensor* bk,
        const ::brotensor::Tensor& Wv, const ::brotensor::Tensor* bv,
        const ::brotensor::Tensor& Wo, const ::brotensor::Tensor* bo,
        const ::brotensor::Tensor& rel_pos_h,
        const ::brotensor::Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale,
        ::brotensor::Tensor& O) {
    check_fp32(X, "X");
    check_fp32(Wq, "Wq"); check_fp32(Wk, "Wk");
    check_fp32(Wv, "Wv"); check_fp32(Wo, "Wo");
    check_fp32(rel_pos_h, "rel_pos_h"); check_fp32(rel_pos_w, "rel_pos_w");

    const int L = X.rows;
    const int D = X.cols;
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(
            "self_attention_decomposed_rel_pos_forward: num_heads must divide D");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(
            "self_attention_decomposed_rel_pos_forward: grid_h*grid_w must equal X.rows");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(
            "self_attention_decomposed_rel_pos_forward: Wq/Wk/Wv/Wo must be (D, D)");

    const int H  = num_heads;
    const int dh = D / H;
    if (rel_pos_h.rows != 2 * grid_h - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(
            "self_attention_decomposed_rel_pos_forward: rel_pos_h must be (2*grid_h-1, head_dim)");
    if (rel_pos_w.rows != 2 * grid_w - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(
            "self_attention_decomposed_rel_pos_forward: rel_pos_w must be (2*grid_w-1, head_dim)");

    auto bias_ptr = [&](const ::brotensor::Tensor* b, const char* name) -> const float* {
        if (!b || !b->data) return nullptr;
        check_fp32(*b, name);
        if (b->size() != D)
            throw std::runtime_error(
                std::string("self_attention_decomposed_rel_pos_forward: ") + name +
                " must have D entries");
        return b->host_f32();
    };
    const float* bqp = bias_ptr(bq, "bq");
    const float* bkp = bias_ptr(bk, "bk");
    const float* bvp = bias_ptr(bv, "bv");
    const float* bop = bias_ptr(bo, "bo");

    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP32)
        O.resize(L, D, Dtype::FP32);
    if (L == 0 || D == 0) return;

    const float* Xp  = X.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    const float* Rhp = rel_pos_h.host_f32();
    const float* Rwp = rel_pos_w.host_f32();
    float* Op = O.host_f32_mut();

    // Per-head projections: Qh / Kh / Vh laid out (H*L, dh), each plus its bias.
    std::vector<float> Qh(static_cast<size_t>(H) * L * dh);
    std::vector<float> Kh(static_cast<size_t>(H) * L * dh);
    std::vector<float> Vh(static_cast<size_t>(H) * L * dh);
    auto project = [&](const float* W, const float* b, std::vector<float>& Out) {
        for (int hh = 0; hh < H; ++hh) {
            for (int i = 0; i < L; ++i) {
                const float* xr = Xp + static_cast<size_t>(i) * D;
                for (int j = 0; j < dh; ++j) {
                    const int o = hh * dh + j;
                    const float* wr = W + static_cast<size_t>(o) * D;
                    float acc = b ? b[o] : 0.0f;
                    for (int k = 0; k < D; ++k) acc += xr[k] * wr[k];
                    Out[(static_cast<size_t>(hh) * L + i) * dh + j] = acc;
                }
            }
        }
    };
    project(Wqp, bqp, Qh);
    project(Wkp, bkp, Kh);
    project(Wvp, bvp, Vh);

    std::vector<float> Yc(static_cast<size_t>(L) * D, 0.0f);
    std::vector<float> srow(L);
    std::vector<float> relh(grid_h);  // per-key-row rel-h bias for current query
    std::vector<float> relw(grid_w);  // per-key-col rel-w bias for current query

    for (int hh = 0; hh < H; ++hh) {
        for (int i = 0; i < L; ++i) {
            const float* qr = &Qh[(static_cast<size_t>(hh) * L + i) * dh];
            const int qh = i / grid_w;
            const int qw = i % grid_w;

            // Decompose the relative-position bias: relh[kh] = q . rel_pos_h[qh-kh+gh-1],
            // relw[kw] = q . rel_pos_w[qw-kw+gw-1]. bias(kh,kw) = relh[kh] + relw[kw].
            for (int kh = 0; kh < grid_h; ++kh) {
                const float* rr = Rhp + static_cast<size_t>(qh - kh + grid_h - 1) * dh;
                float acc = 0.0f;
                for (int c = 0; c < dh; ++c) acc += qr[c] * rr[c];
                relh[kh] = acc;
            }
            for (int kw = 0; kw < grid_w; ++kw) {
                const float* rr = Rwp + static_cast<size_t>(qw - kw + grid_w - 1) * dh;
                float acc = 0.0f;
                for (int c = 0; c < dh; ++c) acc += qr[c] * rr[c];
                relw[kw] = acc;
            }

            // scores = scale * Q.K + decomposed rel-pos bias
            float row_max = -1e30f;
            for (int j = 0; j < L; ++j) {
                const float* kr = &Kh[(static_cast<size_t>(hh) * L + j) * dh];
                float s = 0.0f;
                for (int c = 0; c < dh; ++c) s += qr[c] * kr[c];
                s = s * scale + relh[j / grid_w] + relw[j % grid_w];
                srow[j] = s;
                if (s > row_max) row_max = s;
            }
            // softmax
            float sum = 0.0f;
            for (int j = 0; j < L; ++j) {
                const float e = std::exp(srow[j] - row_max);
                srow[j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            // weighted sum of V
            for (int c = 0; c < dh; ++c) {
                float acc = 0.0f;
                for (int j = 0; j < L; ++j)
                    acc += srow[j] * inv *
                           Vh[(static_cast<size_t>(hh) * L + j) * dh + c];
                Yc[static_cast<size_t>(i) * D + (hh * dh + c)] = acc;
            }
        }
    }

    // Output projection O = Yconcat @ Wo^T (+ bo).
    for (int i = 0; i < L; ++i) {
        const float* yr = &Yc[static_cast<size_t>(i) * D];
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<size_t>(c) * D;
            float acc = bop ? bop[c] : 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            Op[static_cast<size_t>(i) * D + c] = acc;
        }
    }
}

} // namespace brotensor::detail::cpu
