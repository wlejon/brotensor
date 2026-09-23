// FP32 ViT-block ops against an FP64 reference, at real encoder shapes.
//
// DINOv3 ViT-H (K=201 tokens, D=1280, 20 heads x 64, SwiGLU 5120) and SAM
// ViT-B (D=768) drive every op below on the CPU backend and, when one is
// built, the GPU backend in FP32. The residual rows carry the "massive
// activation" channels these models really have (a handful of channels at
// ~1e4..1e5 next to O(1) ones): that is the regime where a norm's statistics or
// a fused kernel's reduction order stops being noise, and random unit-scale
// inputs never reach it. Every output is compared to a double-precision
// reference with a tolerance scaled to the output's own magnitude, so a real
// kernel bug fails here without model weights.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/ops/fused.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using bt_parity::SplitMix64;

namespace {

int g_failures = 0;

Tensor host(int r, int c) { return Tensor::zeros_on(Device::CPU, r, c, Dtype::FP32); }

Tensor to_host(const Tensor& t) {
    if (t.device == Device::CPU) return t;
    brotensor::sync_all();
    return t.to(Device::CPU);
}

void fill(Tensor& t, SplitMix64& rng, float scale) {
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.size(); ++i) p[i] = rng.next_unit() * scale;
}

// Residual-stream rows: O(1) noise, a row-dependent offset, and a few massive
// channels (DINOv3's residual carries channels near 1e5; SAM's reach ~1e3).
void fill_residual(Tensor& t, SplitMix64& rng, float massive) {
    const int R = t.rows, D = t.cols;
    float* p = t.host_f32_mut();
    const int chans[4] = {3, D / 3, D / 2 + 1, D - 5};
    for (int r = 0; r < R; ++r) {
        const float off = 0.5f * rng.next_unit();
        for (int d = 0; d < D; ++d) p[r * D + d] = off + rng.next_unit() * 2.0f;
        if (massive > 0.0f && r % 4 != 1) {
            for (int c : chans) p[r * D + c] = massive * (0.5f + 0.5f * rng.next_f01()) *
                                               (c % 2 ? -1.0f : 1.0f);
        }
    }
}

// Compare an op output against the FP64 reference. The tolerance is
// rtol * max|ref| per row plus a small absolute floor: FP32 rounding of a
// length-n reduction is ~n * 6e-8 relative to its magnitude, far below rtol.
// `show_err` appends the worst error as a fraction of its row's max |ref|.
void expect_close(const char* what, const std::string& dev, const Tensor& got_any,
                  const std::vector<double>& ref, int rows, int cols, double rtol,
                  bool show_err = false) {
    Tensor got = to_host(got_any);
    if (got.rows * got.cols != rows * cols) {
        std::printf("  FAIL %-34s %-5s shape %dx%d, want %dx%d\n", what, dev.c_str(),
                    got.rows, got.cols, rows, cols);
        ++g_failures;
        return;
    }
    const float* g = got.host_f32();
    double worst = 0.0, worst_rel = 0.0;
    int worst_i = -1;
    for (int r = 0; r < rows; ++r) {
        double scale = 0.0;
        for (int c = 0; c < cols; ++c) scale = std::max(scale, std::fabs(ref[r * cols + c]));
        const double tol = rtol * scale + 1e-6;
        for (int c = 0; c < cols; ++c) {
            const int i = r * cols + c;
            const double e = std::fabs(static_cast<double>(g[i]) - ref[i]);
            if (scale > 0.0) worst_rel = std::max(worst_rel, e / scale);
            if (!(e <= tol) && (worst_i < 0 || e / tol > worst)) { worst = e / tol; worst_i = i; }
        }
    }
    char err[48] = "";
    if (show_err) std::snprintf(err, sizeof err, "  max err %.2e", worst_rel);
    if (worst_i >= 0) {
        std::printf("  FAIL %-34s %-5s at %d: got %.7g want %.7g (%.1fx tol)%s\n", what,
                    dev.c_str(), worst_i, g[worst_i], ref[worst_i], worst, err);
        ++g_failures;
    } else {
        std::printf("  ok   %-34s %s%s\n", what, dev.c_str(), err);
    }
}

// ── FP64 references ──────────────────────────────────────────────────────────

std::vector<double> ref_layernorm(const Tensor& X, const Tensor& g, const Tensor* b, float eps) {
    const int R = X.rows, D = X.cols;
    const float* x = X.host_f32();
    std::vector<double> y(static_cast<size_t>(R) * D);
    for (int r = 0; r < R; ++r) {
        double m = 0, v = 0;
        for (int d = 0; d < D; ++d) m += x[r * D + d];
        m /= D;
        for (int d = 0; d < D; ++d) v += (x[r * D + d] - m) * (x[r * D + d] - m);
        v /= D;
        const double rs = 1.0 / std::sqrt(v + eps);
        for (int d = 0; d < D; ++d)
            y[r * D + d] = (x[r * D + d] - m) * rs * g.host_f32()[d] + (b ? b->host_f32()[d] : 0.0);
    }
    return y;
}

std::vector<double> ref_linear(const Tensor& W, const Tensor& bias, const Tensor& X) {
    const int B = X.rows, in = X.cols, out = W.rows;
    std::vector<double> y(static_cast<size_t>(B) * out);
    for (int r = 0; r < B; ++r)
        for (int o = 0; o < out; ++o) {
            double acc = bias.host_f32()[o];
            for (int k = 0; k < in; ++k)
                acc += static_cast<double>(W.host_f32()[o * in + k]) * X.host_f32()[r * in + k];
            y[r * out + o] = acc;
        }
    return y;
}

std::vector<double> ref_swiglu(const Tensor& X) {
    const int B = X.rows, D = X.cols / 2;
    std::vector<double> y(static_cast<size_t>(B) * D);
    for (int r = 0; r < B; ++r)
        for (int d = 0; d < D; ++d) {
            const double a = X.host_f32()[r * 2 * D + d], u = X.host_f32()[r * 2 * D + D + d];
            y[r * D + d] = a / (1.0 + std::exp(-a)) * u;
        }
    return y;
}

std::vector<double> ref_rope(const Tensor& X, const Tensor& c, const Tensor& s, int hd, int nh) {
    const int L = X.rows, W = X.cols, h2 = hd / 2;
    std::vector<double> y(static_cast<size_t>(L) * W);
    for (int l = 0; l < L; ++l)
        for (int h = 0; h < nh; ++h)
            for (int i = 0; i < h2; ++i) {
                const double x0 = X.host_f32()[l * W + h * hd + 2 * i];
                const double x1 = X.host_f32()[l * W + h * hd + 2 * i + 1];
                const double cc = c.host_f32()[l * h2 + i], ss = s.host_f32()[l * h2 + i];
                y[l * W + h * hd + 2 * i] = x0 * cc - x1 * ss;
                y[l * W + h * hd + 2 * i + 1] = x0 * ss + x1 * cc;
            }
    return y;
}

// `mask` (length Lk, optional): keys with mask[j] <= 0.5 drop out.
std::vector<double> ref_attention(const Tensor& Q, const Tensor& K, const Tensor& V, int nh, int hd,
                                  const std::vector<float>* mask = nullptr) {
    const int Lq = Q.rows, Lk = K.rows, W = nh * hd;
    std::vector<double> o(static_cast<size_t>(Lq) * W), p(Lk);
    const double sc = 1.0 / std::sqrt(static_cast<double>(hd));
    for (int h = 0; h < nh; ++h)
        for (int i = 0; i < Lq; ++i) {
            double mx = -1e300;
            for (int j = 0; j < Lk; ++j) {
                double a = 0;
                for (int d = 0; d < hd; ++d)
                    a += static_cast<double>(Q.host_f32()[i * W + h * hd + d]) * K.host_f32()[j * W + h * hd + d];
                p[j] = a * sc;
                if (!mask || (*mask)[j] > 0.5f) mx = std::max(mx, p[j]);
            }
            double z = 0;
            for (int j = 0; j < Lk; ++j) {
                p[j] = (!mask || (*mask)[j] > 0.5f) ? std::exp(p[j] - mx) : 0.0;
                z += p[j];
            }
            for (int d = 0; d < hd; ++d) {
                double a = 0;
                for (int j = 0; j < Lk; ++j) a += p[j] * V.host_f32()[j * W + h * hd + d];
                o[i * W + h * hd + d] = a / z;
            }
        }
    return o;
}

// ── One device pass over the block ops ───────────────────────────────────────

struct Shapes { const char* name; int K, D, heads, ffn; float massive; };

void run_block_ops(const Shapes& s, Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    const int K = s.K, D = s.D, hd = D / s.heads;
    const float eps = 1e-6f;
    auto on = [&](const Tensor& t) { return dev == Device::CPU ? t : t.to(dev); };
    std::string tag = std::string(s.name) + " ";

    Tensor x = host(K, D), res = host(K, D), g = host(D, 1), b = host(D, 1);
    fill_residual(x, rng, s.massive);
    fill(res, rng, 3.0f);
    fill(g, rng, 1.0f);
    fill(b, rng, 0.5f);

    // LayerNorm (affine and bias-free) over massive-activation rows.
    {
        Tensor y;
        brotensor::layernorm_forward_inference_batched(on(x), on(g), on(b), y, eps);
        expect_close((tag + "layernorm").c_str(), dn, y, ref_layernorm(x, g, &b, eps), K, D, 2e-5);
        Tensor y2;
        brotensor::layernorm_forward_inference_batched(on(x), on(g), y2, eps);
        expect_close((tag + "layernorm (no beta)").c_str(), dn, y2, ref_layernorm(x, g, nullptr, eps), K, D, 2e-5);
    }
    // Residual add + LayerNorm, fused: x += res in place, out = LN(x).
    {
        Tensor xd = on(x);
        Tensor y;
        brotensor::fused_residual_layernorm(xd, on(res), on(g), on(b), eps, y);
        Tensor xs = host(K, D);
        std::vector<double> xsum(static_cast<size_t>(K) * D);
        for (int i = 0; i < K * D; ++i) {
            xs.host_f32_mut()[i] = x.host_f32()[i] + res.host_f32()[i];
            xsum[i] = xs.host_f32()[i];
        }
        expect_close((tag + "fused_residual_layernorm x").c_str(), dn, xd, xsum, K, D, 1e-7);
        expect_close((tag + "fused_residual_layernorm y").c_str(), dn, y, ref_layernorm(xs, g, &b, eps), K, D, 2e-5);
    }
    // Projections (FP32 weights): q/k/v-sized and FFN-sized.
    Tensor h = host(K, D);
    fill(h, rng, 2.0f);
    {
        Tensor W = host(D, D), bias = host(D, 1);
        fill(W, rng, 0.05f);
        fill(bias, rng, 0.1f);
        Tensor y;
        brotensor::linear_forward_batched(on(W), on(bias), on(h), y);
        expect_close((tag + "linear_forward_batched DxD").c_str(), dn, y, ref_linear(W, bias, h), K, D, 1e-5);
        Tensor W2 = host(s.ffn, D), b2 = host(s.ffn, 1);
        fill(W2, rng, 0.05f);
        fill(b2, rng, 0.1f);
        Tensor y2;
        brotensor::linear_forward_batched(on(W2), on(b2), on(h), y2);
        expect_close((tag + "linear_forward_batched D->ffn").c_str(), dn, y2, ref_linear(W2, b2, h), K, s.ffn, 1e-5);
    }
    // The FFN down projection (ffn -> D, the long reduction) into the residual.
    {
        Tensor a = host(K, s.ffn), W = host(D, s.ffn), bias = host(D, 1);
        fill(a, rng, 2.0f);
        fill(W, rng, 0.03f);
        fill(bias, rng, 0.1f);
        Tensor y;
        brotensor::linear_forward_batched(on(W), on(bias), on(a), y);
        expect_close((tag + "linear_forward_batched ffn->D").c_str(), dn, y, ref_linear(W, bias, a), K, D, 1e-5);

        Tensor xd = on(x);
        brotensor::add_inplace(xd, y);
        Tensor yh = to_host(y);
        std::vector<double> want(static_cast<size_t>(K) * D);
        for (int i = 0; i < K * D; ++i) want[i] = static_cast<double>(x.host_f32()[i]) + yh.host_f32()[i];
        expect_close((tag + "add_inplace (residual)").c_str(), dn, xd, want, K, D, 1e-7);
    }
    // Patch embed: a stride-p p x p conv over an image, then NCHW -> tokens,
    // then [cls, registers, patches] assembly and the per-row [gate|up] concat.
    {
        const int p = 16, gh = 6, gw = 5, H = gh * p, Wd = gw * p;
        Tensor img = host(1, 3 * H * Wd), Wc = host(D, 3 * p * p), bc = host(D, 1);
        fill(img, rng, 2.0f);
        fill(Wc, rng, 0.05f);
        fill(bc, rng, 0.1f);
        Tensor feat, bcd = on(bc);
        brotensor::conv2d_forward(on(img), on(Wc), &bcd,
                                  1, 3, H, Wd, D, p, p, p, p, 0, 0, 1, 1, feat);
        std::vector<double> want(static_cast<size_t>(D) * gh * gw);
        for (int o = 0; o < D; ++o)
            for (int y = 0; y < gh; ++y)
                for (int xx = 0; xx < gw; ++xx) {
                    double acc = bc.host_f32()[o];
                    for (int c = 0; c < 3; ++c)
                        for (int ky = 0; ky < p; ++ky)
                            for (int kx = 0; kx < p; ++kx)
                                acc += static_cast<double>(Wc.host_f32()[((o * 3 + c) * p + ky) * p + kx]) *
                                       img.host_f32()[(c * H + y * p + ky) * Wd + xx * p + kx];
                    want[(o * gh + y) * gw + xx] = acc;
                }
        expect_close((tag + "conv2d_forward (patch embed)").c_str(), dn, feat, want, 1, D * gh * gw, 1e-5);

        Tensor seq;
        brotensor::nchw_to_sequence(feat, 1, D, gh, gw, seq);
        Tensor fh = to_host(feat);
        std::vector<double> tw(static_cast<size_t>(gh) * gw * D);
        for (int c = 0; c < D; ++c)
            for (int q = 0; q < gh * gw; ++q) tw[q * D + c] = fh.host_f32()[c * gh * gw + q];
        expect_close((tag + "nchw_to_sequence").c_str(), dn, seq, tw, gh * gw, D, 0.0);

        Tensor cls = host(1, D), regs = host(4, D);
        fill(cls, rng, 1.0f);
        fill(regs, rng, 1.0f);
        Tensor clsd = on(cls), regsd = on(regs), cat;
        brotensor::concat_rows({&clsd, &regsd, &seq}, cat);
        std::vector<double> cw;
        for (int i = 0; i < D; ++i) cw.push_back(cls.host_f32()[i]);
        for (int i = 0; i < 4 * D; ++i) cw.push_back(regs.host_f32()[i]);
        for (double v : tw) cw.push_back(v);
        expect_close((tag + "concat_rows").c_str(), dn, cat, cw, 1, static_cast<int>(cw.size()), 0.0);

        Tensor ga = host(K, s.ffn), ub = host(K, s.ffn);
        fill(ga, rng, 3.0f);
        fill(ub, rng, 3.0f);
        Tensor gad = on(ga), ubd = on(ub), gu;
        brotensor::concat_batched_rows({&gad, &ubd}, gu);
        std::vector<double> gw2(static_cast<size_t>(K) * 2 * s.ffn);
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < s.ffn; ++c) {
                gw2[r * 2 * s.ffn + c] = ga.host_f32()[r * s.ffn + c];
                gw2[r * 2 * s.ffn + s.ffn + c] = ub.host_f32()[r * s.ffn + c];
            }
        expect_close((tag + "concat_batched_rows").c_str(), dn, gu, gw2, K, 2 * s.ffn, 0.0);
    }
    // SwiGLU over a gate range wide enough to saturate the sigmoid both ways,
    // out to |g| = 120: ViT-H FFN gates sit far past where a short exp
    // polynomial holds (a Taylor^16 exp collapsed silu(g) to ~0 above g ~ 55).
    {
        Tensor gu = host(K, 2 * s.ffn);
        fill(gu, rng, 120.0f);
        Tensor y;
        brotensor::swiglu_forward(on(gu), y);
        expect_close((tag + "swiglu_forward").c_str(), dn, y, ref_swiglu(gu), K, s.ffn, 1e-6);
    }
    // RoPE with caller tables, then full bidirectional attention.
    {
        Tensor q = host(K, D), k = host(K, D), v = host(K, D);
        fill(q, rng, 3.0f);
        fill(k, rng, 3.0f);
        fill(v, rng, 2.0f);
        Tensor ct = host(K, hd / 2), st = host(K, hd / 2);
        for (int l = 0; l < K; ++l)
            for (int i = 0; i < hd / 2; ++i) {
                const float a = 0.37f * l * (1.0f + i) / hd;
                ct.host_f32_mut()[l * hd / 2 + i] = std::cos(a);
                st.host_f32_mut()[l * hd / 2 + i] = std::sin(a);
            }
        Tensor qr;
        brotensor::rope_apply(on(q), on(ct), on(st), hd, s.heads, qr);
        expect_close((tag + "rope_apply").c_str(), dn, qr, ref_rope(q, ct, st, hd, s.heads), K, D, 1e-6);

        Tensor cu = Tensor::zeros_on(Device::CPU, 2, 1, Dtype::INT32);
        static_cast<int32_t*>(cu.data)[1] = K;
        Tensor cud = on(cu);
        Tensor o;
        brotensor::flash_attention_varlen_forward(on(q), on(k), on(v),
                                                  static_cast<const int32_t*>(cud.data),
                                                  static_cast<const int32_t*>(cud.data),
                                                  1, K, K, s.heads, hd, false, o);
        expect_close((tag + "flash_attention_varlen_forward").c_str(), dn, o,
                     ref_attention(q, k, v, s.heads, hd), K, D, 1e-5);
    }
}

// Bilinear, align_corners=False, border-clamped: PyTorch's F.interpolate.
std::vector<double> ref_bilinear(const std::vector<double>& x, int C, int H, int W,
                                 int Ho, int Wo) {
    std::vector<double> y(static_cast<size_t>(C) * Ho * Wo);
    const double sy = static_cast<double>(H) / Ho, sx = static_cast<double>(W) / Wo;
    for (int c = 0; c < C; ++c)
        for (int oy = 0; oy < Ho; ++oy) {
            const double fy = std::max(0.0, (oy + 0.5) * sy - 0.5);
            const int y0 = std::min(static_cast<int>(fy), H - 1), y1 = std::min(y0 + 1, H - 1);
            const double wy = fy - y0;
            for (int ox = 0; ox < Wo; ++ox) {
                const double fx = std::max(0.0, (ox + 0.5) * sx - 0.5);
                const int x0 = std::min(static_cast<int>(fx), W - 1), x1 = std::min(x0 + 1, W - 1);
                const double wx = fx - x0;
                const double* p = x.data() + static_cast<size_t>(c) * H * W;
                const double t = p[y0 * W + x0] + (p[y0 * W + x1] - p[y0 * W + x0]) * wx;
                const double b = p[y1 * W + x0] + (p[y1 * W + x1] - p[y1 * W + x0]) * wx;
                y[(static_cast<size_t>(c) * Ho + oy) * Wo + ox] = t + (b - t) * wy;
            }
        }
    return y;
}

// SAM's mask postprocess: (3, 256, 256) low-res logits -> bilinear up to the
// 1024 model frame -> crop the letterboxed content (819 x 1024 for a 320x256
// image) -> bilinear DOWN to the original 256 x 320.
void run_mask_upscale(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    const int C = 3, ms = 256, S = 1024, rh = 819, rw = 1024, oh = 256, ow = 320;
    Tensor m = host(1, C * ms * ms);
    // Smooth logits in [-20, 20] with sharp-ish edges, like a mask head.
    for (int c = 0; c < C; ++c)
        for (int yy = 0; yy < ms; ++yy)
            for (int xx = 0; xx < ms; ++xx) {
                const double r = std::hypot(yy - 100.0 - 20 * c, xx - 128.0);
                m.host_f32_mut()[(c * ms + yy) * ms + xx] =
                    static_cast<float>(20.0 * std::tanh((60.0 - r) / 6.0) + rng.next_unit());
            }
    auto on = [&](const Tensor& t) { return dev == Device::CPU ? t : t.to(dev); };
    Tensor up1, crop, up2;
    brotensor::interp2d_forward(on(m), 1, C, ms, ms, S, S, 1, up1);
    brotensor::slice2d_forward(up1, 1, C, S, S, 0, 0, rh, rw, crop);
    brotensor::interp2d_forward(crop, 1, C, rh, rw, oh, ow, 1, up2);

    std::vector<double> x(m.host_f32(), m.host_f32() + m.size());
    std::vector<double> r1 = ref_bilinear(x, C, ms, ms, S, S);
    std::vector<double> rc(static_cast<size_t>(C) * rh * rw);
    for (int c = 0; c < C; ++c)
        for (int yy = 0; yy < rh; ++yy)
            for (int xx = 0; xx < rw; ++xx)
                rc[(static_cast<size_t>(c) * rh + yy) * rw + xx] = r1[(static_cast<size_t>(c) * S + yy) * S + xx];
    expect_close("sam mask upscale 256->1024 (bilinear)", dn, up1, r1, C, S * S, 1e-5);
    expect_close("sam mask upscale ->256x320 (bilinear)", dn, up2,
                 ref_bilinear(rc, C, rh, rw, oh, ow), C, oh * ow, 1e-5);
}

// FP16 attention path (SAM mask decoder on a GPU backend): q/k/v/out
// projections through linear_forward_batched_fp16 and FP16 varlen attention,
// token-to-image (Lq=7, Lk=4096) and image-to-token (Lq=4096, Lk=7), at the
// decoder's widths (D=256; cross attention downsampled to 128, 8 heads -> hd 16;
// self attention hd 32). The reference is FP64 over the FP16-ROUNDED inputs, so
// the tolerance only has to cover FP16 output rounding and FP32 accumulation.
Tensor round16(const Tensor& t) {
    return bt_parity::fp16_host_to_f32(bt_parity::to_fp16_host(t));
}

void run_fp16_attention(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    struct Case { const char* name; int Lq, Lk, in, internal, heads; };
    const Case cases[] = {
        {"fp16 attn tok->img", 7, 4096, 256, 128, 8},
        {"fp16 attn img->tok", 4096, 7, 256, 128, 8},
        {"fp16 attn self", 7, 7, 256, 256, 8},
        {"fp16 attn hd64 tok->img", 7, 4096, 256, 256, 4},   // fused-kernel head_dim
    };
    for (const Case& c : cases) {
        Tensor xq = host(c.Lq, c.in), xk = host(c.Lk, c.in);
        fill(xq, rng, 4.0f);
        fill(xk, rng, 4.0f);
        Tensor Wq = host(c.internal, c.in), bq = host(c.internal, 1);
        Tensor Wk = host(c.internal, c.in), bk = host(c.internal, 1);
        fill(Wq, rng, 0.15f); fill(bq, rng, 0.2f);
        fill(Wk, rng, 0.15f); fill(bk, rng, 0.2f);
        xq = round16(xq); xk = round16(xk);
        Wq = round16(Wq); bq = round16(bq); Wk = round16(Wk); bk = round16(bk);

        auto g16 = [&](const Tensor& t) { return bt_parity::to_fp16_host(t).to(dev); };
        auto back = [&](const Tensor& t) {
            Tensor h = to_host(t);
            return h.dtype == Dtype::FP16 ? bt_parity::fp16_host_to_f32(h) : h;
        };
        Tensor bq16 = g16(bq), bk16 = g16(bk);
        Tensor q16, k16;
        brotensor::linear_forward_batched_fp16(g16(Wq), &bq16, g16(xq), q16);
        brotensor::linear_forward_batched_fp16(g16(Wk), &bk16, g16(xk), k16);
        const std::string tag = c.name;
        expect_close((tag + " q proj").c_str(), dn, back(q16), ref_linear(Wq, bq, xq),
                     c.Lq, c.internal, 2e-3);
        expect_close((tag + " k proj").c_str(), dn, back(k16), ref_linear(Wk, bk, xk),
                     c.Lk, c.internal, 2e-3);

        // Attention over the (FP16) projections actually produced.
        Tensor qh = back(q16), kh = back(k16);
        Tensor vh = host(c.Lk, c.internal);
        fill(vh, rng, 3.0f);
        vh = round16(vh);
        Tensor cq = Tensor::zeros_on(Device::CPU, 2, 1, Dtype::INT32);
        Tensor ck = Tensor::zeros_on(Device::CPU, 2, 1, Dtype::INT32);
        static_cast<int32_t*>(cq.data)[1] = c.Lq;
        static_cast<int32_t*>(ck.data)[1] = c.Lk;
        Tensor cqd = cq.to(dev), ckd = ck.to(dev);
        Tensor o;
        brotensor::flash_attention_varlen_forward(q16, k16, g16(vh),
            static_cast<const int32_t*>(cqd.data), static_cast<const int32_t*>(ckd.data),
            1, c.Lq, c.Lk, c.heads, c.internal / c.heads, false, o);
        expect_close((tag + " varlen attention").c_str(), dn, back(o),
                     ref_attention(qh, kh, vh, c.heads, c.internal / c.heads),
                     c.Lq, c.internal, 4e-3);
    }
}

// flash_attention_forward called directly in FP16 and BF16, at head dims the
// fused FlashAttention-2 kernel is instantiated for and at ones it is not (hd
// 20 and 512 here, which take the per-head fallback). SAM's mask decoder
// attends at hd 16 / 32 over Lk 4096, SD1.5's UNet at hd 40 / 80 / 160, the SD
// VAE mid-block at a single hd 512 head; hd 20 is not a multiple of 8. q and k span
// [-4, 4], so the scaled logits reach ~+-20 and the softmax is peaked: that is
// where a score rounded to 16 bits before the softmax shows, as an error on
// every probability. The reference is FP64 over the 16-bit-ROUNDED inputs, so
// the tolerance covers only the 16-bit output and P@V operand rounding.
void run_direct_flash_attention(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    struct Case { const char* name; int Lq, Lk, heads, hd; bool masked; };
    const Case cases[] = {
        {"hd16 tok->img",     64, 4096, 8,  16, false},
        {"hd16 img->tok",   4096,    7, 8,  16, false},
        {"hd32 self",       1024, 1024, 8,  32, false},
        {"hd32 tok->img",     64, 4096, 8,  32, false},
        {"hd32 masked",      512, 1000, 4,  32, true},
        {"hd20 self",        512,  512, 3,  20, false},
        {"hd24 masked",      512, 1001, 4,  24, true},
        {"hd80 self",       1024, 1024, 2,  80, false},
        {"hd160 self",       256,  256, 2, 160, false},
        {"hd512 self (vae)", 1024, 1024, 1, 512, false},
        {"hd40 self",       1024, 1024, 2,  40, false},   // fused-kernel head_dims
        {"hd64 self",       1024, 1024, 2,  64, false},
        {"hd128 tok->img",    64, 4096, 2, 128, false},
    };
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        const bool bf = dt == Dtype::BF16;
        auto rnd = [&](const Tensor& t) {
            return bf ? bt_parity::bf16_host_to_f32(bt_parity::to_bf16_host(t)) : round16(t);
        };
        auto dev16 = [&](const Tensor& t) {
            return (bf ? bt_parity::to_bf16_host(t) : bt_parity::to_fp16_host(t)).to(dev);
        };
        auto back = [&](const Tensor& t) {
            Tensor h = to_host(t);
            return bf ? bt_parity::bf16_host_to_f32(h) : bt_parity::fp16_host_to_f32(h);
        };
        // Output rounding alone is up to 2^-12 (FP16) / 2^-9 (BF16) of the row
        // max, and P's rounding for the P@V GEMM adds about as much again: the
        // worst case measured is 7.3e-4 / 6.0e-3. A score held in 16 bits
        // ahead of the softmax lands at 4-7e-3 / 3-5e-2.
        const double rtol = bf ? 1.2e-2 : 1.5e-3;
        for (const Case& c : cases) {
            const int D = c.heads * c.hd;
            Tensor q = host(c.Lq, D), k = host(c.Lk, D), v = host(c.Lk, D);
            fill(q, rng, 4.0f);
            fill(k, rng, 4.0f);
            fill(v, rng, 3.0f);
            q = rnd(q); k = rnd(k); v = rnd(v);
            std::vector<float> mask;
            Tensor maskd;
            if (c.masked) {
                mask.resize(c.Lk);
                for (int j = 0; j < c.Lk; ++j) mask[j] = (j % 7 == 3 || j > 900) ? 0.0f : 1.0f;
                maskd = Tensor::from_host_on(dev, mask.data(), c.Lk, 1);
            }
            Tensor o;
            brotensor::flash_attention_forward(dev16(q), dev16(k), dev16(v),
                                               c.masked ? static_cast<const float*>(maskd.data) : nullptr,
                                               c.heads, /*causal=*/false, o);
            const std::string tag = std::string(bf ? "bf16" : "fp16") + " flash_attention " + c.name;
            expect_close(tag.c_str(), dn, back(o),
                         ref_attention(q, k, v, c.heads, c.hd, c.masked ? &mask : nullptr),
                         c.Lq, D, rtol, /*show_err=*/true);
        }
    }
}

// ── Attention backward against FP64 ─────────────────────────────────────────

float round_to(double x, Dtype dt) {
    const float f = static_cast<float>(x);
    if (dt == Dtype::FP16) return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(f));
    if (dt == Dtype::BF16) return brotensor::bf16_bits_to_fp32(brotensor::fp32_to_bf16_bits(f));
    return f;
}

Tensor rand_rounded(int r, int c, SplitMix64& rng, float scale, Dtype dt) {
    Tensor t = host(r, c);
    fill(t, rng, scale);
    float* p = t.host_f32_mut();
    for (int i = 0; i < t.size(); ++i) p[i] = round_to(p[i], dt);
    return t;
}

Tensor upload(const Tensor& t, Dtype dt, Device dev) {
    if (dt == Dtype::BF16) return bt_parity::to_bf16_host(t).to(dev);
    if (dt == Dtype::FP16) return bt_parity::to_fp16_host(t).to(dev);
    return t.to(dev);
}

Tensor download(const Tensor& t) {
    Tensor h = to_host(t);
    if (h.dtype == Dtype::BF16) return bt_parity::bf16_host_to_f32(h);
    if (h.dtype == Dtype::FP16) return bt_parity::fp16_host_to_f32(h);
    return h;
}

const char* dt_name(Dtype dt) { return dt == Dtype::BF16 ? "bf16" : dt == Dtype::FP16 ? "fp16" : "fp32"; }

Tensor cols_of(const Tensor& t, int c0, int w) {
    Tensor o = host(t.rows, w);
    for (int r = 0; r < t.rows; ++r)
        for (int c = 0; c < w; ++c) o.host_f32_mut()[r * w + c] = t.host_f32()[r * t.cols + c0 + c];
    return o;
}

std::vector<double> cols_of(const std::vector<double>& v, int rows, int cols, int c0, int w) {
    std::vector<double> o(static_cast<size_t>(rows) * w);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < w; ++c) o[r * w + c] = v[static_cast<size_t>(r) * cols + c0 + c];
    return o;
}

// FP64 forward + backward of softmax(Q K^T / sqrt(hd)) V over one block of
// rows. Row i of Q / K / V sits at q / k / v + i*ld, head h at column h*hd; dO
// rows at g + i*ldg. Key j is seen by query i when the mask (if any) keeps it,
// j <= i under causal, and |i - j| <= hw when hw >= 0. dq / dk / dv (row
// stride ldd) and o (row stride ldg, optional) are accumulated into.
void ref_attention_bwd(const float* q, const float* k, const float* v, int ld, const float* g, int ldg,
                       int Lq, int Lk, int nh, int hd, const std::vector<float>* mask, bool causal,
                       int hw, double* dq, double* dk, double* dv, int ldd, double* o = nullptr) {
    const double sc = 1.0 / std::sqrt(static_cast<double>(hd));
    std::vector<double> p(Lk), dp(Lk);
    std::vector<char> on(Lk);
    for (int h = 0; h < nh; ++h) {
        const int c0 = h * hd;
        for (int i = 0; i < Lq; ++i) {
            const float* qi = q + static_cast<size_t>(i) * ld + c0;
            const float* gi = g + static_cast<size_t>(i) * ldg + c0;
            double mx = -1e300;
            for (int j = 0; j < Lk; ++j) {
                on[j] = (!mask || (*mask)[j] > 0.5f) && (!causal || j <= i) &&
                        (hw < 0 || std::abs(i - j) <= hw);
                if (!on[j]) continue;
                const float* kj = k + static_cast<size_t>(j) * ld + c0;
                double a = 0.0;
                for (int d = 0; d < hd; ++d) a += static_cast<double>(qi[d]) * kj[d];
                p[j] = a * sc;
                mx = std::max(mx, p[j]);
            }
            double z = 0.0;
            for (int j = 0; j < Lk; ++j) {
                p[j] = on[j] ? std::exp(p[j] - mx) : 0.0;
                z += p[j];
            }
            if (z == 0.0) continue;
            double Dq = 0.0;
            for (int j = 0; j < Lk; ++j) {
                p[j] /= z;
                dp[j] = 0.0;
                if (!on[j]) continue;
                const float* vj = v + static_cast<size_t>(j) * ld + c0;
                for (int d = 0; d < hd; ++d) dp[j] += static_cast<double>(gi[d]) * vj[d];
                Dq += p[j] * dp[j];
                if (o)
                    for (int d = 0; d < hd; ++d) o[static_cast<size_t>(i) * ldg + c0 + d] += p[j] * vj[d];
            }
            for (int j = 0; j < Lk; ++j) {
                if (!on[j]) continue;
                const double ds = p[j] * (dp[j] - Dq) * sc;
                const float* kj = k + static_cast<size_t>(j) * ld + c0;
                double* dqi = dq + static_cast<size_t>(i) * ldd + c0;
                double* dkj = dk + static_cast<size_t>(j) * ldd + c0;
                double* dvj = dv + static_cast<size_t>(j) * ldd + c0;
                for (int d = 0; d < hd; ++d) {
                    dqi[d] += ds * kj[d];
                    dkj[d] += ds * qi[d];
                    dvj[d] += p[j] * gi[d];
                }
            }
        }
    }
}

// Worst-case error each backward is allowed, as a fraction of the gradient
// row's max |ref|. The 16-bit outputs round at 2^-12 / 2^-9 of that, and the
// 16-bit P operand of dV adds about as much again: measured worst cases are
// 1.8e-3 (FP16) / 8e-3 (BF16). With the scores, P, dP or dS held in 16 bits
// the same gradients land at 5e-3 - 2e-2 (FP16) and 4e-2 - 2 (BF16).
void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

double bwd_rtol(Dtype dt) { return dt == Dtype::BF16 ? 1.5e-2 : dt == Dtype::FP16 ? 3e-3 : 1e-4; }

// flash_attention_backward (the bare core LoRA-style trainers wrap) in FP16 and
// BF16, at the head dims its callers use. Inputs as in the forward test above:
// q, k in [-4, 4] make the softmax peaked, where a score, probability or dS
// held in 16 bits shows as an error on every gradient.
void run_flash_attention_backward(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    struct Case { const char* name; int Lq, Lk, heads, hd; bool masked, causal; };
    const Case cases[] = {
        {"hd16 tok->img",     64, 2048, 4,  16, false, false},
        {"hd20 self",        384,  384, 3,  20, false, false},
        {"hd32 self causal", 512,  512, 4,  32, false, true},
        {"hd40 self",        512,  512, 2,  40, false, false},
        {"hd40 cross",       512,   77, 2,  40, false, false},
        {"hd64 masked",      256, 1000, 2,  64, true,  false},
        {"hd64 causal",      512,  512, 2,  64, false, true},
        {"hd80 self",        512,  512, 2,  80, false, false},
        {"hd128 self",       256,  256, 2, 128, false, false},
        {"hd160 self",       256,  256, 2, 160, false, false},
    };
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        for (const Case& c : cases) {
            const int D = c.heads * c.hd;
            Tensor q = rand_rounded(c.Lq, D, rng, 4.0f, dt), k = rand_rounded(c.Lk, D, rng, 4.0f, dt);
            Tensor v = rand_rounded(c.Lk, D, rng, 3.0f, dt), g = rand_rounded(c.Lq, D, rng, 1.0f, dt);
            std::vector<float> mask;
            Tensor maskd;
            if (c.masked) {
                mask.resize(c.Lk);
                for (int j = 0; j < c.Lk; ++j) mask[j] = (j % 7 == 3 || j > 900) ? 0.0f : 1.0f;
                maskd = Tensor::from_host_on(dev, mask.data(), c.Lk, 1);
            }
            Tensor gd = upload(g, dt, dev), dQ, dK, dV;
            brotensor::flash_attention_backward(upload(q, dt, dev), upload(k, dt, dev), upload(v, dt, dev),
                                                gd, gd, c.masked ? static_cast<const float*>(maskd.data) : nullptr,
                                                c.heads, c.causal, dQ, dK, dV);
            std::vector<double> rq(static_cast<size_t>(c.Lq) * D), rk(static_cast<size_t>(c.Lk) * D),
                rv(static_cast<size_t>(c.Lk) * D);
            ref_attention_bwd(q.host_f32(), k.host_f32(), v.host_f32(), D, g.host_f32(), D, c.Lq, c.Lk,
                              c.heads, c.hd, c.masked ? &mask : nullptr, c.causal, -1, rq.data(), rk.data(),
                              rv.data(), D);
            const std::string tag = std::string(dt_name(dt)) + " fa_bwd " + c.name;
            expect_close((tag + " dQ").c_str(), dn, download(dQ), rq, c.Lq, D, bwd_rtol(dt), true);
            expect_close((tag + " dK").c_str(), dn, download(dK), rk, c.Lk, D, bwd_rtol(dt), true);
            expect_close((tag + " dV").c_str(), dn, download(dV), rv, c.Lk, D, bwd_rtol(dt), true);
        }
    }
}

// flash_attention_varlen_backward: several sequences packed back to back,
// self (causal) and cross, FP16 / BF16 / FP32.
void run_varlen_backward(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    struct Case { const char* name; std::vector<int> lq, lk; int heads, hd; bool causal; };
    const Case cases[] = {
        {"hd64 causal", {3, 200, 77, 300}, {3, 200, 77, 300}, 4, 64, true},
        {"hd32 cross",  {5, 120, 64},      {77, 300, 1},      4, 32, false},
        {"hd80 self",   {150, 256},        {150, 256},        2, 80, false},
    };
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16, Dtype::FP32}) {
        for (const Case& c : cases) {
            const int B = static_cast<int>(c.lq.size()), D = c.heads * c.hd;
            Tensor cq = Tensor::zeros_on(Device::CPU, B + 1, 1, Dtype::INT32);
            Tensor ck = Tensor::zeros_on(Device::CPU, B + 1, 1, Dtype::INT32);
            int32_t* pq = static_cast<int32_t*>(cq.data);
            int32_t* pk = static_cast<int32_t*>(ck.data);
            int mq = 0, mk = 0;
            for (int b = 0; b < B; ++b) {
                pq[b + 1] = pq[b] + c.lq[b];
                pk[b + 1] = pk[b] + c.lk[b];
                mq = std::max(mq, c.lq[b]);
                mk = std::max(mk, c.lk[b]);
            }
            const int Tq = pq[B], Tk = pk[B];
            Tensor q = rand_rounded(Tq, D, rng, 4.0f, dt), k = rand_rounded(Tk, D, rng, 4.0f, dt);
            Tensor v = rand_rounded(Tk, D, rng, 3.0f, dt), g = rand_rounded(Tq, D, rng, 1.0f, dt);
            Tensor cqd = cq.to(dev), ckd = ck.to(dev), gd = upload(g, dt, dev), dQ, dK, dV;
            brotensor::flash_attention_varlen_backward(
                upload(q, dt, dev), upload(k, dt, dev), upload(v, dt, dev), gd, gd,
                static_cast<const int32_t*>(cqd.data), static_cast<const int32_t*>(ckd.data), B, mq, mk,
                c.heads, c.hd, c.causal, dQ, dK, dV);
            std::vector<double> rq(static_cast<size_t>(Tq) * D), rk(static_cast<size_t>(Tk) * D),
                rv(static_cast<size_t>(Tk) * D);
            for (int b = 0; b < B; ++b)
                ref_attention_bwd(q.host_f32() + pq[b] * D, k.host_f32() + pk[b] * D, v.host_f32() + pk[b] * D,
                                  D, g.host_f32() + pq[b] * D, D, c.lq[b], c.lk[b], c.heads, c.hd, nullptr,
                                  c.causal, -1, rq.data() + pq[b] * D, rk.data() + pk[b] * D,
                                  rv.data() + pk[b] * D, D);
            const std::string tag = std::string(dt_name(dt)) + " varlen_bwd " + c.name;
            expect_close((tag + " dQ").c_str(), dn, download(dQ), rq, Tq, D, bwd_rtol(dt), true);
            expect_close((tag + " dK").c_str(), dn, download(dK), rk, Tk, D, bwd_rtol(dt), true);
            expect_close((tag + " dV").c_str(), dn, download(dV), rv, Tk, D, bwd_rtol(dt), true);
        }
    }
}

// flash_attention_packed_qkv_backward: a fused-QKV encoder batch, full and
// windowed, FP16 / BF16 / FP32.
void run_packed_qkv_backward(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    struct Case { const char* name; std::vector<int> lens; int heads, hd, window; };
    const Case cases[] = {
        {"hd64 full",     {100, 37, 250}, 4, 64, 0},
        {"hd32 window64", {100, 37, 250}, 4, 32, 64},
    };
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16, Dtype::FP32}) {
        for (const Case& c : cases) {
            const int D = c.heads * c.hd;
            int L = 0;
            for (int n : c.lens) L += n;
            Tensor bounds = Tensor::zeros_on(Device::CPU, L, 2, Dtype::INT32);
            int32_t* pb = static_cast<int32_t*>(bounds.data);
            std::vector<int> starts;
            for (int s = 0, r = 0; s < static_cast<int>(c.lens.size()); r += c.lens[s], ++s) {
                starts.push_back(r);
                for (int i = r; i < r + c.lens[s]; ++i) { pb[2 * i] = r; pb[2 * i + 1] = r + c.lens[s]; }
            }
            Tensor qkv = host(L, 3 * D);
            for (int r = 0; r < L; ++r)
                for (int col = 0; col < 3 * D; ++col)
                    qkv.host_f32_mut()[r * 3 * D + col] =
                        round_to(rng.next_unit() * (col < 2 * D ? 4.0f : 3.0f), dt);
            Tensor g = rand_rounded(L, D, rng, 1.0f, dt);
            Tensor bd = bounds.to(dev), dQKV;
            brotensor::flash_attention_packed_qkv_backward(upload(qkv, dt, dev), upload(g, dt, dev), bd, c.heads,
                                                           c.window, dQKV);
            std::vector<double> ref(static_cast<size_t>(L) * 3 * D);
            const float* x = qkv.host_f32();
            for (size_t s = 0; s < c.lens.size(); ++s) {
                const size_t r0 = static_cast<size_t>(starts[s]);
                ref_attention_bwd(x + r0 * 3 * D, x + r0 * 3 * D + D, x + r0 * 3 * D + 2 * D, 3 * D,
                                  g.host_f32() + r0 * D, D, c.lens[s], c.lens[s], c.heads, c.hd, nullptr, false,
                                  c.window > 0 ? c.window / 2 : -1, ref.data() + r0 * 3 * D,
                                  ref.data() + r0 * 3 * D + D, ref.data() + r0 * 3 * D + 2 * D, 3 * D);
            }
            Tensor got = download(dQKV);
            const std::string tag = std::string(dt_name(dt)) + " packed_bwd " + c.name;
            const char* part[3] = {" dQ", " dK", " dV"};
            for (int p = 0; p < 3; ++p)
                expect_close((tag + part[p]).c_str(), dn, cols_of(got, p * D, D), cols_of(ref, L, 3 * D, p * D, D),
                             L, D, bwd_rtol(dt), true);
        }
    }
}

// flash_attention_qkvo_backward, self and cross. The reference rounds to the
// op's dtype everything the op's contract stores in it (the Q / K / V
// projections, the attention output, dO @ Wo and dQ / dK / dV), so what is
// left to measure is the attention core's own precision.
void run_qkvo_backward(Device dev, const std::string& dn, uint64_t seed) {
    SplitMix64 rng(seed);
    struct Case { const char* name; int Lq, D, heads, Lk, Dctx; bool cross, causal; };
    const Case cases[] = {
        {"self hd40",        256, 320, 8, 256, 320, false, false},
        {"cross hd40",       256, 320, 8,  77, 768, true,  false},
        {"self hd64 causal", 256, 256, 4, 256, 256, false, true},
    };
    for (const Dtype dt : {Dtype::FP16, Dtype::BF16}) {
        for (const Case& c : cases) {
            const int D = c.D, Dc = c.Dctx, Lq = c.Lq, Lk = c.Lk, hd = D / c.heads;
            const float ws = 4.5f / std::sqrt(static_cast<float>(Dc)), wq = 4.5f / std::sqrt(static_cast<float>(D));
            Tensor X = rand_rounded(Lq, D, rng, 2.0f, dt);
            Tensor C = c.cross ? rand_rounded(Lk, Dc, rng, 2.0f, dt) : X;
            Tensor Wq = rand_rounded(D, D, rng, wq, dt), Wk = rand_rounded(D, Dc, rng, ws, dt);
            Tensor Wv = rand_rounded(D, Dc, rng, ws * 0.7f, dt), Wo = rand_rounded(D, D, rng, wq * 0.7f, dt);
            Tensor bq = rand_rounded(D, 1, rng, 0.2f, dt), bk = rand_rounded(D, 1, rng, 0.2f, dt);
            Tensor bv = rand_rounded(D, 1, rng, 0.2f, dt), bo = rand_rounded(D, 1, rng, 0.2f, dt);
            Tensor g = rand_rounded(Lq, D, rng, 1.0f, dt);

            Tensor Xd = upload(X, dt, dev), Cd = upload(C, dt, dev);
            Tensor Wqd = upload(Wq, dt, dev), Wkd = upload(Wk, dt, dev), Wvd = upload(Wv, dt, dev),
                   Wod = upload(Wo, dt, dev);
            Tensor bqd = upload(bq, dt, dev), bkd = upload(bk, dt, dev), bvd = upload(bv, dt, dev),
                   bod = upload(bo, dt, dev);
            Tensor dX, dCtx;
            Tensor dWq = Tensor::zeros_on(dev, D, D, dt), dWk = Tensor::zeros_on(dev, D, Dc, dt);
            Tensor dWv = Tensor::zeros_on(dev, D, Dc, dt), dWo = Tensor::zeros_on(dev, D, D, dt);
            Tensor dbq = Tensor::zeros_on(dev, D, 1, dt), dbk = Tensor::zeros_on(dev, D, 1, dt);
            Tensor dbv = Tensor::zeros_on(dev, D, 1, dt), dbo = Tensor::zeros_on(dev, D, 1, dt);
            brotensor::flash_attention_qkvo_backward(Xd, c.cross ? &Cd : nullptr, Wqd, &bqd, Wkd, &bkd, Wvd, &bvd,
                                                     Wod, &bod, nullptr, c.heads, c.causal, upload(g, dt, dev), dX,
                                                     c.cross ? &dCtx : nullptr, dWq, &dbq, dWk, &dbk, dWv, &dbv,
                                                     dWo, &dbo);

            // Reference. lin: rows of `in` (n, k) through W (m, k) + b, rounded.
            auto lin = [&](const Tensor& in, const Tensor& W, const Tensor& b) {
                const int n = in.rows, kk = in.cols, m = W.rows;
                std::vector<float> y(static_cast<size_t>(n) * m);
                for (int r = 0; r < n; ++r)
                    for (int o = 0; o < m; ++o) {
                        double a = 0.0;
                        for (int i = 0; i < kk; ++i)
                            a += static_cast<double>(in.host_f32()[r * kk + i]) * W.host_f32()[o * kk + i];
                        // The BF16 projection rounds X W^T, then adds the bias
                        // and rounds again; FP16 fuses the bias into one rounding.
                        if (dt == Dtype::BF16) a = round_to(a, dt);
                        y[static_cast<size_t>(r) * m + o] = round_to(a + b.host_f32()[o], dt);
                    }
                return y;
            };
            const std::vector<float> Q = lin(X, Wq, bq), K = lin(C, Wk, bk), V = lin(C, Wv, bv);
            std::vector<double> A(static_cast<size_t>(Lq) * D), sink_q(A.size()),
                sink_k(static_cast<size_t>(Lk) * D), sink_v(sink_k.size());
            std::vector<float> zero_g(static_cast<size_t>(Lq) * D, 0.0f);
            ref_attention_bwd(Q.data(), K.data(), V.data(), D, zero_g.data(), D, Lq, Lk, c.heads, hd, nullptr,
                              c.causal, -1, sink_q.data(), sink_k.data(), sink_v.data(), D, A.data());
            std::vector<float> Ar(A.size()), dA(A.size());
            for (size_t i = 0; i < A.size(); ++i) Ar[i] = round_to(A[i], dt);
            for (int r = 0; r < Lq; ++r)
                for (int col = 0; col < D; ++col) {
                    double a = 0.0;
                    for (int o = 0; o < D; ++o)
                        a += static_cast<double>(g.host_f32()[r * D + o]) * Wo.host_f32()[o * D + col];
                    dA[static_cast<size_t>(r) * D + col] = round_to(a, dt);
                }
            std::vector<double> gq(static_cast<size_t>(Lq) * D), gk(static_cast<size_t>(Lk) * D), gv(gk.size());
            ref_attention_bwd(Q.data(), K.data(), V.data(), D, dA.data(), D, Lq, Lk, c.heads, hd, nullptr,
                              c.causal, -1, gq.data(), gk.data(), gv.data(), D);
            for (auto* t : {&gq, &gk, &gv})
                for (double& x : *t) x = round_to(x, dt);
            // dIn(n, k) += dY(n, m) @ W(m, k);  dW(m, k) = dY^T @ In.
            auto back_in = [](const std::vector<double>& dY, const Tensor& W, int n, std::vector<double>& dIn) {
                const int m = W.rows, kk = W.cols;
                for (int r = 0; r < n; ++r)
                    for (int o = 0; o < m; ++o) {
                        const double gy = dY[static_cast<size_t>(r) * m + o];
                        for (int i = 0; i < kk; ++i) dIn[static_cast<size_t>(r) * kk + i] += gy * W.host_f32()[o * kk + i];
                    }
            };
            auto back_w = [](const std::vector<double>& dY, const float* in, int n, int m, int kk) {
                std::vector<double> dW(static_cast<size_t>(m) * kk);
                for (int r = 0; r < n; ++r)
                    for (int o = 0; o < m; ++o) {
                        const double gy = dY[static_cast<size_t>(r) * m + o];
                        for (int i = 0; i < kk; ++i) dW[static_cast<size_t>(o) * kk + i] += gy * in[r * kk + i];
                    }
                return dW;
            };
            std::vector<double> rdX(static_cast<size_t>(Lq) * D), rdC(static_cast<size_t>(Lk) * Dc);
            back_in(gq, Wq, Lq, rdX);
            back_in(gk, Wk, Lk, c.cross ? rdC : rdX);
            back_in(gv, Wv, Lk, c.cross ? rdC : rdX);
            std::vector<double> gd(g.host_f32(), g.host_f32() + g.size());
            const std::string tag = std::string(dt_name(dt)) + " qkvo_bwd " + c.name;
            const double rt = bwd_rtol(dt);
            expect_close((tag + " dX").c_str(), dn, download(dX), rdX, Lq, D, rt, true);
            if (c.cross) expect_close((tag + " dCtx").c_str(), dn, download(dCtx), rdC, Lk, Dc, rt, true);
            expect_close((tag + " dWq").c_str(), dn, download(dWq), back_w(gq, X.host_f32(), Lq, D, D), D, D, rt, true);
            expect_close((tag + " dWk").c_str(), dn, download(dWk), back_w(gk, C.host_f32(), Lk, D, Dc), D, Dc, rt, true);
            expect_close((tag + " dWv").c_str(), dn, download(dWv), back_w(gv, C.host_f32(), Lk, D, Dc), D, Dc, rt, true);
            expect_close((tag + " dWo").c_str(), dn, download(dWo), back_w(gd, Ar.data(), Lq, D, D), D, D, rt, true);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    brotensor::init();   // before is_available(): the driver probe happens here
    // `bwd` runs only the attention-backward cases.
    const bool bwd_only = argc > 1 && std::string(argv[1]) == "bwd";
    const Shapes shapes[] = {
        {"dinov3-vit-h", 201, 1280, 20, 5120, 6.0e4f},
        {"sam-vit-b",    196,  768, 12, 3072, 1.5e3f},
    };
    std::vector<std::pair<Device, std::string>> devs = {{Device::CPU, "CPU"}};
    if (brotensor::is_available(Device::CUDA)) devs.push_back({Device::CUDA, "CUDA"});
    else if (brotensor::is_available(Device::Metal)) devs.push_back({Device::Metal, "Metal"});

    uint64_t seed = 1234;
    if (!bwd_only) {
        for (const Shapes& s : shapes)
            for (const auto& [dev, name] : devs) run_block_ops(s, dev, name, seed++);
        for (const auto& [dev, name] : devs) run_mask_upscale(dev, name, 99);
    }
    // FP16 storage is GPU-only (the CPU backend is FP32 by design).
    for (const auto& [dev, name] : devs)
        if (dev != Device::CPU) {
            if (!bwd_only) {
                run_fp16_attention(dev, name, 7);
                run_direct_flash_attention(dev, name, 11);
            }
            // The bare core, varlen and qkvo backward pick between two paths
            // by problem size; each case runs through both. On devices without
            // the tensor-core path "tc" falls back to the rows.
            for (const char* path : {"rows", "tc"}) {
                set_env("BROTENSOR_ATTN_BWD_PATH", path);
                const std::string tag = name + "/" + path;
                run_flash_attention_backward(dev, tag, 13);
                run_varlen_backward(dev, tag, 17);
                run_qkvo_backward(dev, tag, 23);
            }
            set_env("BROTENSOR_ATTN_BWD_PATH", "");
            run_packed_qkv_backward(dev, name, 19);
        }

    if (g_failures) {
        std::printf("test_vit_block_ops: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_vit_block_ops: OK\n");
    return 0;
}
