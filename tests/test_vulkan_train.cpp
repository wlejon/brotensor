// Vulkan parity for the training ops filled in chunk 8, against the CPU
// backend: attention_backward (one head), mha_backward (with and without the
// bias gradients), self_attention_backward and cross_attention_backward
// (lq != lk with a key mask; lq == lk, where the mask also gates queries),
// training-mode batch_norm_forward / batch_norm_backward, and
// bce_with_logits_fused_batched. The forward
// caches come from the CPU forward and are uploaded, so each comparison
// isolates the backward; dW / db / dGamma / dBeta start non-zero to check
// that they accumulate. `--only=train`.

#include "test_vulkan_common.h"

#include <brotensor/ops/attention.h>
#include <brotensor/ops/loss.h>
#include <brotensor/ops/norm.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace vkt {

namespace {

Tensor cpu(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

std::vector<float> host(const Tensor& t) { return t.to_host_vector(); }

// |got - want| <= rel * max|want| + 1e-7: sums of many FP32 products in a
// different order, so the error scales with the largest output.
void close_scaled(const Tensor& got, const Tensor& want, float rel, const std::string& tag) {
    const std::vector<float> w = host(want);
    float m = 0;
    for (float v : w) m = std::max(m, std::fabs(v));
    expect_close(download(got), w, rel * m + 1e-7f, 0, tag);
}

// The same values on both devices.
struct Pair {
    Tensor c, v;
    Pair() = default;
    Pair(const std::vector<float>& x, int rows, int cols) : c(cpu(x, rows, cols)), v(upload(x, rows, cols, Dtype::FP32)) {}
    static Pair rand(int rows, int cols, std::uint64_t seed, float lo = -1, float hi = 1) {
        return Pair(random_values(std::size_t(rows) * cols, seed, lo, hi, Dtype::FP32), rows, cols);
    }
    static Pair of(const Tensor& cpu_t) { return Pair(cpu_t.to_host_vector(), cpu_t.rows, cpu_t.cols); }
};

std::vector<float> mask_values(int L, int seed) {
    std::vector<float> m(L, 1.0f);
    for (int i = 0; i < L; ++i)
        if ((i * 7 + seed) % 5 == 0) m[i] = 0.0f;
    m[0] = 1.0f;
    return m;
}

const float* ptr(const Tensor* t) { return t ? static_cast<const float*>(t->data) : nullptr; }

constexpr float kRel = 2e-5f;

// mha_backward (bias = true / false), or attention_backward when heads == 0.
void test_mha(int K, int D, int H, bool masked, bool bias, const char* name) {
    const bool single = H == 0;
    const int heads = single ? 1 : H;
    const std::string t = std::string(" ") + name;
    Pair X = Pair::rand(K, D, 1), Wq = Pair::rand(D, D, 2, -0.3f, 0.3f), Wk = Pair::rand(D, D, 3, -0.3f, 0.3f),
         Wv = Pair::rand(D, D, 4, -0.3f, 0.3f), Wo = Pair::rand(D, D, 5, -0.3f, 0.3f);
    Pair bq = Pair::rand(D, 1, 6), bk = Pair::rand(D, 1, 7), bv = Pair::rand(D, 1, 8), bo = Pair::rand(D, 1, 9);
    Pair M;
    if (masked) M = Pair(mask_values(K, 3), 1, K);
    const float* mc = masked ? ptr(&M.c) : nullptr;
    const float* mv = masked ? ptr(&M.v) : nullptr;

    Tensor Qh, Kh, Vh, A, Yc, O;
    if (single) {
        brotensor::attention_forward(X.c, Wq.c, Wk.c, Wv.c, Wo.c, mc, Qh, Kh, Vh, A, Yc, O);
    } else {
        brotensor::mha_forward(X.c, Wq.c, Wk.c, Wv.c, Wo.c, bias ? &bq.c : nullptr, bias ? &bk.c : nullptr,
                               bias ? &bv.c : nullptr, bias ? &bo.c : nullptr, mc, heads, Qh, Kh, Vh, A, Yc, O);
    }
    Pair q = Pair::of(Qh), k = Pair::of(Kh), v = Pair::of(Vh), a = Pair::of(A), y = Pair::of(Yc);
    Pair dO = Pair::rand(K, D, 10);
    Pair dWq = Pair::rand(D, D, 11), dWk = Pair::rand(D, D, 12), dWv = Pair::rand(D, D, 13), dWo = Pair::rand(D, D, 14);
    Pair dbq = Pair::rand(D, 1, 15), dbk = Pair::rand(D, 1, 16), dbv = Pair::rand(D, 1, 17), dbo = Pair::rand(D, 1, 18);
    Tensor dXc, dXv;
    if (single) {
        brotensor::attention_backward(dO.c, X.c, q.c, k.c, v.c, a.c, y.c, Wq.c, Wk.c, Wv.c, Wo.c, mc, dXc, dWq.c,
                                      dWk.c, dWv.c, dWo.c);
        brotensor::attention_backward(dO.v, X.v, q.v, k.v, v.v, a.v, y.v, Wq.v, Wk.v, Wv.v, Wo.v, mv, dXv, dWq.v,
                                      dWk.v, dWv.v, dWo.v);
    } else {
        brotensor::mha_backward(dO.c, X.c, q.c, k.c, v.c, a.c, y.c, Wq.c, Wk.c, Wv.c, Wo.c, mc, heads, dXc, dWq.c,
                                dWk.c, dWv.c, dWo.c, bias ? &dbq.c : nullptr, bias ? &dbk.c : nullptr,
                                bias ? &dbv.c : nullptr, bias ? &dbo.c : nullptr);
        brotensor::mha_backward(dO.v, X.v, q.v, k.v, v.v, a.v, y.v, Wq.v, Wk.v, Wv.v, Wo.v, mv, heads, dXv, dWq.v,
                                dWk.v, dWv.v, dWo.v, bias ? &dbq.v : nullptr, bias ? &dbk.v : nullptr,
                                bias ? &dbv.v : nullptr, bias ? &dbo.v : nullptr);
    }
    const std::string op = single ? "attention_backward" : "mha_backward";
    close_scaled(dXv, dXc, kRel, op + " dX" + t);
    close_scaled(dWq.v, dWq.c, kRel, op + " dWq" + t);
    close_scaled(dWk.v, dWk.c, kRel, op + " dWk" + t);
    close_scaled(dWv.v, dWv.c, kRel, op + " dWv" + t);
    close_scaled(dWo.v, dWo.c, kRel, op + " dWo" + t);
    if (bias) {
        close_scaled(dbq.v, dbq.c, kRel, op + " dbq" + t);
        close_scaled(dbk.v, dbk.c, kRel, op + " dbk" + t);
        close_scaled(dbv.v, dbv.c, kRel, op + " dbv" + t);
        close_scaled(dbo.v, dbo.c, kRel, op + " dbo" + t);
    }
}

void test_self(int K, int D, int H, const char* name) {
    const std::string t = std::string(" ") + name;
    Pair X = Pair::rand(K, D, 21), Wq = Pair::rand(D, D, 22, -0.3f, 0.3f), Wk = Pair::rand(D, D, 23, -0.3f, 0.3f),
         Wv = Pair::rand(D, D, 24, -0.3f, 0.3f), Wo = Pair::rand(D, D, 25, -0.3f, 0.3f);
    Pair M(mask_values(K, 1), 1, K);
    Tensor Qh, Kh, Vh, A, Yc, O;
    brotensor::self_attention_forward_train(X.c, Wq.c, Wk.c, Wv.c, Wo.c, ptr(&M.c), H, Qh, Kh, Vh, A, Yc, O);
    Pair q = Pair::of(Qh), k = Pair::of(Kh), v = Pair::of(Vh), a = Pair::of(A), y = Pair::of(Yc);
    Pair dO = Pair::rand(K, D, 26);
    Pair dWq = Pair::rand(D, D, 27), dWk = Pair::rand(D, D, 28), dWv = Pair::rand(D, D, 29), dWo = Pair::rand(D, D, 30);
    Tensor dXc, dXv;
    brotensor::self_attention_backward(dO.c, X.c, q.c, k.c, v.c, a.c, y.c, Wq.c, Wk.c, Wv.c, Wo.c, ptr(&M.c), H, dXc,
                                       dWq.c, dWk.c, dWv.c, dWo.c);
    brotensor::self_attention_backward(dO.v, X.v, q.v, k.v, v.v, a.v, y.v, Wq.v, Wk.v, Wv.v, Wo.v, ptr(&M.v), H, dXv,
                                       dWq.v, dWk.v, dWv.v, dWo.v);
    close_scaled(dXv, dXc, kRel, "self_attention_backward dX" + t);
    close_scaled(dWq.v, dWq.c, kRel, "self_attention_backward dWq" + t);
    close_scaled(dWk.v, dWk.c, kRel, "self_attention_backward dWk" + t);
    close_scaled(dWv.v, dWv.c, kRel, "self_attention_backward dWv" + t);
    close_scaled(dWo.v, dWo.c, kRel, "self_attention_backward dWo" + t);
}

void test_cross(int Lq, int Lk, int D, int Dc, int H, bool masked, const char* name) {
    const std::string t = std::string(" ") + name;
    Pair X = Pair::rand(Lq, D, 31), C = Pair::rand(Lk, Dc, 32), Wq = Pair::rand(D, D, 33, -0.3f, 0.3f),
         Wk = Pair::rand(D, Dc, 34, -0.3f, 0.3f), Wv = Pair::rand(D, Dc, 35, -0.3f, 0.3f),
         Wo = Pair::rand(D, D, 36, -0.3f, 0.3f);
    Pair M;
    if (masked) M = Pair(mask_values(Lk, 2), 1, Lk);
    const float* mc = masked ? ptr(&M.c) : nullptr;
    const float* mv = masked ? ptr(&M.v) : nullptr;
    Tensor Qh, Kh, Vh, A, Yc, O;
    brotensor::cross_attention_forward_train(X.c, C.c, Wq.c, Wk.c, Wv.c, Wo.c, mc, H, Qh, Kh, Vh, A, Yc, O);
    Pair q = Pair::of(Qh), k = Pair::of(Kh), v = Pair::of(Vh), a = Pair::of(A), y = Pair::of(Yc);
    Pair dO = Pair::rand(Lq, D, 37);
    Pair dWq = Pair::rand(D, D, 38), dWk = Pair::rand(D, Dc, 39), dWv = Pair::rand(D, Dc, 40),
         dWo = Pair::rand(D, D, 41);
    Tensor dXc, dXv, dCc, dCv;
    brotensor::cross_attention_backward(dO.c, X.c, C.c, q.c, k.c, v.c, a.c, y.c, Wq.c, Wk.c, Wv.c, Wo.c, mc, H, dXc,
                                        dCc, dWq.c, dWk.c, dWv.c, dWo.c);
    brotensor::cross_attention_backward(dO.v, X.v, C.v, q.v, k.v, v.v, a.v, y.v, Wq.v, Wk.v, Wv.v, Wo.v, mv, H, dXv,
                                        dCv, dWq.v, dWk.v, dWv.v, dWo.v);
    close_scaled(dXv, dXc, kRel, "cross_attention_backward dX" + t);
    close_scaled(dCv, dCc, kRel, "cross_attention_backward dCtx" + t);
    close_scaled(dWq.v, dWq.c, kRel, "cross_attention_backward dWq" + t);
    close_scaled(dWk.v, dWk.c, kRel, "cross_attention_backward dWk" + t);
    close_scaled(dWv.v, dWv.c, kRel, "cross_attention_backward dWv" + t);
    close_scaled(dWo.v, dWo.c, kRel, "cross_attention_backward dWo" + t);
}

void test_batch_norm(int N, int C, int H, int W, float offset, const char* name) {
    const std::string t = std::string(" ") + name;
    const int cols = C * H * W;
    std::vector<float> x = random_values(std::size_t(N) * cols, 50, -1, 1, Dtype::FP32);
    for (float& v : x) v += offset;   // a large mean against a small spread: the two-pass variance
    Pair X(x, N, cols), g = Pair::rand(C, 1, 51, 0.5f, 1.5f), b = Pair::rand(C, 1, 52);
    Pair rm = Pair::rand(C, 1, 53), rv = Pair::rand(C, 1, 54, 0.5f, 2.0f);
    Tensor Yc, Yv, smc, smv, src, srv;
    brotensor::batch_norm_forward(X.c, g.c, b.c, rm.c, rv.c, N, C, H, W, 1e-5f, 0.1f, Yc, smc, src);
    brotensor::batch_norm_forward(X.v, g.v, b.v, rm.v, rv.v, N, C, H, W, 1e-5f, 0.1f, Yv, smv, srv);
    const float rel = offset > 0 ? 1e-4f : 1e-5f;
    close_scaled(Yv, Yc, rel, "batch_norm_forward Y" + t);
    close_scaled(smv, smc, 1e-6f, "batch_norm_forward saved_mean" + t);
    close_scaled(srv, src, rel, "batch_norm_forward saved_rstd" + t);
    close_scaled(rm.v, rm.c, 1e-6f, "batch_norm_forward running_mean" + t);
    close_scaled(rv.v, rv.c, rel, "batch_norm_forward running_var" + t);
    // The backward reads the CPU's saved statistics on both devices.
    Pair sm = Pair::of(smc), sr = Pair::of(src), dY = Pair::rand(N, cols, 55);
    Pair dg = Pair::rand(C, 1, 56), db = Pair::rand(C, 1, 57);
    Tensor dXc, dXv;
    brotensor::batch_norm_backward(X.c, g.c, sm.c, sr.c, dY.c, N, C, H, W, dXc, dg.c, db.c);
    brotensor::batch_norm_backward(X.v, g.v, sm.v, sr.v, dY.v, N, C, H, W, dXv, dg.v, db.v);
    close_scaled(dXv, dXc, rel, "batch_norm_backward dX" + t);
    close_scaled(dg.v, dg.c, rel, "batch_norm_backward dGamma" + t);
    close_scaled(db.v, db.c, 1e-5f, "batch_norm_backward dBeta" + t);
}

void test_bce(int B, int L, bool masked, float pos_weight, const char* name) {
    const std::string t = std::string(" ") + name;
    Pair z = Pair::rand(B, L, 60, -30, 30);   // saturating logits on both sides
    std::vector<float> y = random_values(std::size_t(B) * L, 61, 0, 1, Dtype::FP32);
    for (std::size_t i = 0; i < y.size(); i += 3) y[i] = y[i] < 0.5f ? 0.0f : 1.0f;
    Pair Y(y, B, L), M;
    if (masked) M = Pair(mask_values(B * L, 4), B, L);
    Tensor Pc, Pv, Dc, Dv, Lc, Lv;
    brotensor::bce_with_logits_fused_batched(z.c, Y.c, masked ? ptr(&M.c) : nullptr, pos_weight, Pc, Dc, Lc);
    brotensor::bce_with_logits_fused_batched(z.v, Y.v, masked ? ptr(&M.v) : nullptr, pos_weight, Pv, Dv, Lv);
    expect_close(download(Pv), host(Pc), 1e-7f, 2e-6f, "bce_with_logits_fused_batched probs" + t);
    expect_close(download(Dv), host(Dc), 1e-6f, 2e-6f, "bce_with_logits_fused_batched dLogits" + t);
    close_scaled(Lv, Lc, 2e-6f, "bce_with_logits_fused_batched loss" + t);
}

}  // namespace

void run_train_tests() {
    std::printf("\n[training: attention backwards, BatchNorm, BCE]\n");
    test_mha(12, 32, 0, true, false, "one head, N 12 D 32, masked");
    test_mha(40, 64, 0, false, false, "one head, N 40 D 64");
    test_mha(16, 64, 4, true, true, "K 16 D 64 H 4, masked, biases");
    test_mha(33, 36, 3, false, true, "K 33 D 36 H 3 (hd 12), biases");
    test_mha(24, 128, 8, true, false, "K 24 D 128 H 8, masked");
    test_self(20, 48, 2, "K 20 D 48 H 2, masked");
    test_cross(10, 7, 32, 24, 4, true, "lq 10 lk 7 D 32 Dc 24 H 4, key mask");
    test_cross(9, 9, 32, 32, 2, true, "lq = lk 9, query gating");
    test_cross(17, 50, 64, 40, 4, false, "lq 17 lk 50 D 64 Dc 40 H 4");
    test_batch_norm(4, 16, 8, 8, 0.0f, "N 4 C 16 8x8");
    test_batch_norm(8, 64, 1, 40, 0.0f, "N 8 C 64 1x40 (BC-ResNet row)");
    test_batch_norm(2, 3, 1, 1, 0.0f, "N 2 C 3 1x1");
    test_batch_norm(3, 5, 7, 9, 100.0f, "N 3 C 5 7x9, mean 100");
    test_bce(16, 35, false, 1.0f, "16 x 35");
    test_bce(7, 600, true, 3.5f, "7 x 600, masked, pos_weight 3.5");
}

}  // namespace vkt
