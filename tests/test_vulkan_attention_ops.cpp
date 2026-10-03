// Attention-adjacent ops parity against the CPU backend: the projection-fused
// attentions with materialised probabilities (self_attention_bias_forward,
// cross_attention_forward(_with_attn), mha_forward, self_attention_forward),
// rel_pos_bias_xl_forward, top_k_rows and segment_softmax_stats. Inputs are
// rounded to the dtype first (test_vulkan_attention.cpp's conventions). The
// fused ops keep every intermediate in FP32, so 16-bit runs differ from the
// CPU only by the final rounding of O (and AttnAvg).

#include "test_vulkan_common.h"

#include <algorithm>
#include <cstdlib>

namespace vkt {

namespace {

const Device kCpu = Device::CPU;

Tensor host(const std::vector<float>& v, int rows, int cols) { return Tensor::from_host_on(kCpu, v.data(), rows, cols); }

std::vector<Dtype> dtypes() {
    const char* e = std::getenv("VKT_ATTN_DT");
    std::vector<Dtype> out;
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) if (!e || std::string(e) == dt_name(dt)) out.push_back(dt);
    return out;
}

std::string tag(const char* what, Dtype dt, int a, int b, int c) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s %s %d/%d/%d", what, dt_name(dt), a, b, c);
    return buf;
}

// FP32: summation order over D-long projections; 16-bit: the output's rounding.
void close_out(Dtype dt, const Tensor& got, const Tensor& want, const std::string& t) {
    const float tol = dt == Dtype::FP32 ? 2e-4f : 2.0f * dtype_eps(dt) + 1e-4f;
    expect_close(download(got), want.to_host_vector(), tol, tol, t);
}

std::vector<float> holes(int n, std::uint64_t seed) {
    std::vector<float> m(static_cast<std::size_t>(n), 1.0f);
    Rng r(seed);
    for (auto& x : m) x = r.next() % 4 == 0 ? 0.0f : 1.0f;
    return m;
}

struct Weights {
    std::vector<float> wq, wk, wv, wo, bq, bk, bv, bo;
    Weights(int D, int Dc, Dtype dt, std::uint64_t s) {
        auto W = [&](int r, int c, std::uint64_t k) { return random_values(std::size_t(r) * c, s + k, -0.15f, 0.15f, dt); };
        wq = W(D, D, 1); wk = W(D, Dc, 2); wv = W(D, Dc, 3); wo = W(D, D, 4);
        bq = W(D, 1, 5); bk = W(D, 1, 6); bv = W(D, 1, 7); bo = W(D, 1, 8);
    }
};

void test_self_attention_bias() {
    std::printf("self_attention_bias_forward / self_attention_forward\n");
    for (Dtype dt : dtypes()) {
        for (int variant = 0; variant < 3; ++variant) {
            const int L = variant == 2 ? 64 : 37, D = 128, H = 4;
            const Weights w(D, D, dt, 20 + variant);
            const auto x = random_values(std::size_t(L) * D, 10 + variant, -1.0f, 1.0f, dt);
            const auto bias = random_values(std::size_t(H) * L * L, 30 + variant, -2.0f, 2.0f, Dtype::FP32);
            const auto mv = holes(L, 40 + variant);
            const bool use_mask = variant != 1, use_bias = variant != 2;
            const float scale = variant == 0 ? 1.0f : 0.125f;   // T5 uses 1
            Tensor Oc;
            {
                Tensor B = host(bias, H * L, L);
                Tensor bq = host(w.bq, D, 1), bk = host(w.bk, D, 1), bv = host(w.bv, D, 1), bo = host(w.bo, D, 1);
                brotensor::self_attention_bias_forward(host(x, L, D), host(w.wq, D, D), host(w.wk, D, D), host(w.wv, D, D),
                                                       host(w.wo, D, D), &bq, &bk, &bv, &bo,
                                                       use_mask ? mv.data() : nullptr, use_bias ? &B : nullptr, H,
                                                       scale, Oc);
            }
            Tensor B = Tensor::from_host_on(vk(), bias.data(), H * L, L), M = Tensor::from_host_on(vk(), mv.data(), L, 1);
            Tensor bq = upload(w.bq, D, 1, dt), bk = upload(w.bk, D, 1, dt), bv = upload(w.bv, D, 1, dt), bo = upload(w.bo, D, 1, dt);
            Tensor O;
            brotensor::self_attention_bias_forward(upload(x, L, D, dt), upload(w.wq, D, D, dt), upload(w.wk, D, D, dt),
                                                   upload(w.wv, D, D, dt), upload(w.wo, D, D, dt), &bq, &bk, &bv, &bo,
                                                   use_mask ? static_cast<const float*>(M.data) : nullptr,
                                                   use_bias ? &B : nullptr, H, scale, O);
            VKT_CHECK(O.dtype == dt);
            close_out(dt, O, Oc, tag("self_attention_bias_forward", dt, L, H, variant));
        }
        // self_attention_forward (no biases, query + key mask)
        const int L = 45, D = 96, H = 3;
        const Weights w(D, D, dt, 50);
        const auto x = random_values(std::size_t(L) * D, 51, -1.0f, 1.0f, dt);
        const auto mv = holes(L, 52);
        Tensor Oc, O;
        brotensor::self_attention_forward(host(x, L, D), host(w.wq, D, D), host(w.wk, D, D), host(w.wv, D, D),
                                          host(w.wo, D, D), mv.data(), H, Oc);
        Tensor M = Tensor::from_host_on(vk(), mv.data(), L, 1);
        brotensor::self_attention_forward(upload(x, L, D, dt), upload(w.wq, D, D, dt), upload(w.wk, D, D, dt),
                                          upload(w.wv, D, D, dt), upload(w.wo, D, D, dt),
                                          static_cast<const float*>(M.data), H, O);
        close_out(dt, O, Oc, tag("self_attention_forward", dt, L, H, 0));
    }
}

void test_cross_attention() {
    std::printf("cross_attention_forward / _with_attn\n");
    for (Dtype dt : dtypes()) {
        for (int variant = 0; variant < 2; ++variant) {
            const int Lq = variant ? 40 : 30, Lk = variant ? 40 : 45, D = 128, Dc = 64, H = 4;
            const Weights w(D, Dc, dt, 60 + variant);
            const auto x = random_values(std::size_t(Lq) * D, 61, -1.0f, 1.0f, dt);
            const auto c = random_values(std::size_t(Lk) * Dc, 62, -1.0f, 1.0f, dt);
            const auto mv = holes(Lk, 63 + variant);
            const auto lb = random_values(std::size_t(Lq) * Lk, 64, -1.0f, 1.0f, Dtype::FP32);
            Tensor Oc, Oa, Aa;
            {
                Tensor X = host(x, Lq, D), C = host(c, Lk, Dc), B = host(lb, Lq, Lk);
                Tensor Wq = host(w.wq, D, D), Wk = host(w.wk, D, Dc), Wv = host(w.wv, D, Dc), Wo = host(w.wo, D, D);
                brotensor::cross_attention_forward(X, C, Wq, Wk, Wv, Wo, mv.data(), H, Oc);
                brotensor::cross_attention_forward_with_attn(X, C, Wq, Wk, Wv, Wo, mv.data(), variant ? &B : nullptr, H,
                                                             Oa, Aa);
            }
            Tensor X = upload(x, Lq, D, dt), C = upload(c, Lk, Dc, dt);
            Tensor Wq = upload(w.wq, D, D, dt), Wk = upload(w.wk, D, Dc, dt), Wv = upload(w.wv, D, Dc, dt),
                   Wo = upload(w.wo, D, D, dt);
            Tensor M = Tensor::from_host_on(vk(), mv.data(), Lk, 1), B = Tensor::from_host_on(vk(), lb.data(), Lq, Lk);
            const float* m = static_cast<const float*>(M.data);
            Tensor O, O2, A2;
            brotensor::cross_attention_forward(X, C, Wq, Wk, Wv, Wo, m, H, O);
            brotensor::cross_attention_forward_with_attn(X, C, Wq, Wk, Wv, Wo, m, variant ? &B : nullptr, H, O2, A2);
            close_out(dt, O, Oc, tag(variant ? "cross_attention_forward gated" : "cross_attention_forward", dt, Lq, Lk, H));
            close_out(dt, O2, Oa, tag("cross_attention_forward_with_attn O", dt, Lq, Lk, H));
            close_out(dt, A2, Aa, tag("cross_attention_forward_with_attn AttnAvg", dt, Lq, Lk, H));
        }
    }
}

void test_mha() {
    std::printf("mha_forward / rel_pos_bias_xl_forward\n");
    const int L = 33, D = 64, H = 4;
    const Weights w(D, D, Dtype::FP32, 70);
    const auto x = random_values(std::size_t(L) * D, 71, -1.0f, 1.0f, Dtype::FP32);
    const auto mv = holes(L, 72);
    Tensor c[6];
    {
        Tensor bq = host(w.bq, D, 1), bk = host(w.bk, D, 1), bv = host(w.bv, D, 1), bo = host(w.bo, D, 1);
        brotensor::mha_forward(host(x, L, D), host(w.wq, D, D), host(w.wk, D, D), host(w.wv, D, D), host(w.wo, D, D),
                               &bq, &bk, &bv, &bo, mv.data(), H, c[0], c[1], c[2], c[3], c[4], c[5]);
    }
    Tensor g[6];
    Tensor bq = upload(w.bq, D, 1, Dtype::FP32), bk = upload(w.bk, D, 1, Dtype::FP32),
           bv = upload(w.bv, D, 1, Dtype::FP32), bo = upload(w.bo, D, 1, Dtype::FP32);
    Tensor M = Tensor::from_host_on(vk(), mv.data(), L, 1);
    brotensor::mha_forward(upload(x, L, D, Dtype::FP32), upload(w.wq, D, D, Dtype::FP32), upload(w.wk, D, D, Dtype::FP32),
                           upload(w.wv, D, D, Dtype::FP32), upload(w.wo, D, D, Dtype::FP32), &bq, &bk, &bv, &bo,
                           static_cast<const float*>(M.data), H, g[0], g[1], g[2], g[3], g[4], g[5]);
    const char* names[] = {"mha_forward Qh", "mha_forward Kh", "mha_forward Vh", "mha_forward Attnh",
                           "mha_forward Yconcat", "mha_forward O"};
    for (int i = 0; i < 6; ++i) {
        VKT_CHECK(g[i].rows == c[i].rows && g[i].cols == c[i].cols);
        close_out(Dtype::FP32, g[i], c[i], tag(names[i], Dtype::FP32, L, H, D));
    }

    const int T = 29, dk = 16, Hr = 4;
    const auto qv = random_values(std::size_t(T) * Hr * dk, 73, -1.0f, 1.0f, Dtype::FP32);
    const auto pk = random_values(std::size_t(2 * T - 1) * Hr * dk, 74, -1.0f, 1.0f, Dtype::FP32);
    Tensor bc, bg;
    brotensor::rel_pos_bias_xl_forward(host(qv, T, Hr * dk), host(pk, 2 * T - 1, Hr * dk), Hr, dk, bc);
    brotensor::rel_pos_bias_xl_forward(upload(qv, T, Hr * dk, Dtype::FP32), upload(pk, 2 * T - 1, Hr * dk, Dtype::FP32),
                                       Hr, dk, bg);
    close_out(Dtype::FP32, bg, bc, tag("rel_pos_bias_xl_forward", Dtype::FP32, T, Hr, dk));
}

void test_topk_segments() {
    std::printf("top_k_rows / segment_softmax_stats\n");
    for (Dtype dt : dtypes()) {
        struct Case { int R, C, k; };
        for (const Case& c : {Case{5, 1000, 7}, Case{1, 300, 300}, Case{3, 70, 1}, Case{2, 2500, 100}}) {
            // Coarse values: many ties, which the index must break.
            auto v = random_values(std::size_t(c.R) * c.C, 80 + c.C, -4.0f, 4.0f, dt);
            for (auto& e : v) e = std::round(e * 4.0f) / 4.0f;
            Tensor vc, ic, vg, ig;
            brotensor::top_k_rows(host(v, c.R, c.C), c.k, vc, ic);
            brotensor::top_k_rows(upload(v, c.R, c.C, dt), c.k, vg, ig);
            const auto a = vg.to_host_vector(), b = vc.to_host_vector();
            expect_equal_bits(a.data(), b.data(), a.size() * 4, tag("top_k_rows Vals", dt, c.R, c.C, c.k));
            std::vector<int32_t> ia(std::size_t(c.R) * c.k), ib(ia.size());
            ig.copy_to_host_raw(ia.data(), ia.size() * 4);
            std::memcpy(ib.data(), ic.data, ib.size() * 4);
            expect_equal_bits(ia.data(), ib.data(), ia.size() * 4, tag("top_k_rows Idx", dt, c.R, c.C, c.k));
        }
        const std::vector<int32_t> off = {0, 5, 5, 6, 2006, 2100};   // an empty and a one-element segment
        const auto lg = random_values(std::size_t(off.back()), 90, -6.0f, 6.0f, dt);
        Tensor oc_t = Tensor::empty_on(kCpu, int(off.size()), 1, Dtype::INT32);
        std::memcpy(oc_t.data, off.data(), off.size() * 4);
        Tensor og = Tensor::empty_on(vk(), int(off.size()), 1, Dtype::INT32);
        og.copy_from_host_raw(off.data(), off.size() * 4);
        Tensor sc, sg;
        brotensor::segment_softmax_stats(host(lg, off.back(), 1), oc_t, sc);
        brotensor::segment_softmax_stats(upload(lg, off.back(), 1, dt), og, sg);
        expect_close(sg.to_host_vector(), sc.to_host_vector(), 2e-5f, 2e-5f,
                     tag("segment_softmax_stats", dt, int(off.size()) - 1, off.back(), 0));
    }
}

void test_xent() {
    std::printf("softmax_xent / _fused / _fused_batched\n");
    const int B = 5, N = 1100;
    const std::vector<int> heads = {0, 3, 10, 1000, 1100};
    const auto lg = random_values(std::size_t(B) * N, 95, -5.0f, 5.0f, Dtype::FP32);
    auto tg = std::vector<float>(lg.size(), 0.0f);
    Rng r(96);
    for (int b = 0; b < B; ++b)
        for (std::size_t h = 0; h + 1 < heads.size(); ++h) tg[std::size_t(b) * N + heads[h] + r.next() % (heads[h + 1] - heads[h])] = 1.0f;
    const auto mv = holes(B * N, 97);
    Tensor pc, dc, lc, pg, dg, lgv;
    brotensor::softmax_xent_fused_batched(host(lg, B, N), host(tg, B, N), mv.data(), heads.data(), int(heads.size()) - 1,
                                          pc, dc, lc);
    Tensor M = Tensor::from_host_on(vk(), mv.data(), B * N, 1);
    Tensor Hd = Tensor::empty_on(vk(), int(heads.size()), 1, Dtype::INT32);
    Hd.copy_from_host_raw(heads.data(), heads.size() * 4);
    Tensor L = upload(lg, B, N, Dtype::FP32), T = upload(tg, B, N, Dtype::FP32);
    brotensor::softmax_xent_fused_batched(L, T, static_cast<const float*>(M.data), static_cast<const int*>(Hd.data),
                                          int(heads.size()) - 1, pg, dg, lgv);
    expect_close(pg.to_host_vector(), pc.to_host_vector(), 1e-6f, 1e-5f, "softmax_xent_fused_batched probs");
    expect_close(dg.to_host_vector(), dc.to_host_vector(), 1e-6f, 1e-5f, "softmax_xent_fused_batched dLogits");
    expect_close(lgv.to_host_vector(), lc.to_host_vector(), 1e-5f, 1e-5f, "softmax_xent_fused_batched loss");
    // One segment, the loss returned.
    std::vector<float> t1(N, 0.0f);
    t1[17] = 0.25f; t1[400] = 0.75f;
    const std::vector<float> l1(lg.begin(), lg.begin() + N), m1(mv.begin(), mv.begin() + N);
    Tensor p1c, d1c, p1g, d1g, p2c, d2c, p2g, d2g;
    const float a = brotensor::softmax_xent(host(l1, 1, N), host(t1, 1, N), p1c, d1c, m1.data());
    const float b = brotensor::softmax_xent_fused(host(l1, 1, N), host(t1, 1, N), nullptr, p2c, d2c);
    Tensor M1 = Tensor::from_host_on(vk(), m1.data(), N, 1);
    const float ag = brotensor::softmax_xent(upload(l1, 1, N, Dtype::FP32), upload(t1, 1, N, Dtype::FP32), p1g, d1g,
                                             static_cast<const float*>(M1.data));
    const float bg = brotensor::softmax_xent_fused(upload(l1, 1, N, Dtype::FP32), upload(t1, 1, N, Dtype::FP32),
                                                   nullptr, p2g, d2g);
    expect_close({ag, bg}, {a, b}, 1e-5f, 1e-5f, "softmax_xent / softmax_xent_fused loss");
    expect_close(p1g.to_host_vector(), p1c.to_host_vector(), 1e-6f, 1e-5f, "softmax_xent probs");
    expect_close(d2g.to_host_vector(), d2c.to_host_vector(), 1e-6f, 1e-5f, "softmax_xent_fused dLogits");
}

}  // namespace

void run_attention_ops_tests() {
    test_self_attention_bias();
    test_cross_attention();
    test_mha();
    test_topk_segments();
    test_xent();
}

}  // namespace vkt
