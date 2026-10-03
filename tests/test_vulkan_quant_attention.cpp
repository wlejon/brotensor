// Vulkan parity for the INT8-weight compositions: the flash family's
// projection-fused ops (project_kv, q_with_kv_cached, qkvo; self and cross,
// causal, masked, decode-sized and prefill-sized query counts), T5's
// self_attention_bias_int8w_fp16 (logit bias, key mask, scale) and
// resblock_forward_int8w_fp16 (with and without the skip convolution and the
// time-embedding shift). The reference is the CPU backend's dense-weight op
// in FP32 on the decoded weights (q * scale) and the FP16 inputs.

#include "test_vulkan_common.h"

#include <brotensor/vulkan.h>

#include <cmath>
#include <string>

namespace vkt {

namespace {

struct Q8 {
    Tensor q, s;           // Vulkan INT8 weight and FP32 scales
    Tensor w;              // CPU FP32 decoded weight
};

Q8 make_q8(int rows, int cols, std::uint64_t seed, float amp) {
    const std::vector<float> src = random_values(std::size_t(rows) * cols, seed, -amp, amp, Dtype::FP16);
    std::vector<std::uint16_t> bits(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) bits[i] = brotensor::fp32_to_fp16_bits(src[i]);
    std::vector<std::int8_t> q(src.size());
    std::vector<float> s(rows), w(src.size());
    brotensor::quantize_int8_per_row_host(bits.data(), rows, cols, q.data(), s.data());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) w[std::size_t(r) * cols + c] = float(q[std::size_t(r) * cols + c]) * s[r];
    return {Tensor::from_host_int8_on(vk(), q.data(), rows, cols), Tensor::from_host_on(vk(), s.data(), rows, 1),
            Tensor::from_host_on(Device::cpu(), w.data(), rows, cols)};
}

struct Act {
    Tensor v, c;   // Vulkan FP16, CPU FP32 of the same values
};

Act make_act(int rows, int cols, std::uint64_t seed, float lo = -1, float hi = 1) {
    const std::vector<float> x = random_values(std::size_t(rows) * cols, seed, lo, hi, Dtype::FP16);
    return {upload(x, rows, cols, Dtype::FP16), Tensor::from_host_on(Device::cpu(), x.data(), rows, cols)};
}

void test_flash_int8(int Lq, int Lk, int D, int H, bool cross, bool causal, bool masked, std::uint64_t seed) {
    const float amp = 1.0f / std::sqrt(float(D));
    Q8 wq = make_q8(D, D, seed, amp), wk = make_q8(D, D, seed + 1, amp), wv = make_q8(D, D, seed + 2, amp),
       wo = make_q8(D, D, seed + 3, amp);
    Act bq = make_act(D, 1, seed + 4, -0.1f, 0.1f), bk = make_act(D, 1, seed + 5, -0.1f, 0.1f),
        bv = make_act(D, 1, seed + 6, -0.1f, 0.1f), bo = make_act(D, 1, seed + 7, -0.1f, 0.1f);
    Act X = make_act(Lq, D, seed + 8), C = make_act(cross ? Lk : Lq, D, seed + 9);
    std::vector<float> mask(cross ? Lk : Lq, 1.0f);
    if (masked) for (std::size_t i = 0; i < mask.size(); i += 3) mask[i] = 0.0f;
    Tensor mv = Tensor::from_host_on(vk(), mask.data(), int(mask.size()), 1);
    const float* dm = masked ? static_cast<const float*>(mv.data) : nullptr;
    const float* cm = masked ? mask.data() : nullptr;
    Tensor O, Oc;
    brotensor::flash_attention_qkvo_int8w_fp16(X.v, cross ? &C.v : nullptr, wq.q, wq.s, &bq.v, wk.q, wk.s, &bk.v,
                                               wv.q, wv.s, &bv.v, wo.q, wo.s, &bo.v, dm, H, causal, O);
    brotensor::flash_attention_qkvo_forward(X.c, cross ? &C.c : nullptr, wq.w, &bq.c, wk.w, &bk.c, wv.w, &bv.c, wo.w,
                                            &bo.c, cm, H, causal, Oc);
    char tag[96];
    std::snprintf(tag, sizeof tag, "flash_qkvo_int8w Lq=%d Lk=%d D=%d H=%d%s%s%s", Lq, cross ? Lk : Lq, D, H,
                  cross ? " cross" : "", causal ? " causal" : "", masked ? " masked" : "");
    // Projections round Q / K / V / A to FP16 (2^-11 of O(1) values) and the
    // weights may enter FP16 tiles: a few 1e-3 absolute on outputs ~0.5.
    expect_close(download(O), Oc.to_host_vector(), 6e-3f, 6e-3f, tag);

    // The two halves on their own: the projected K / V and the cached-KV step.
    Tensor K, V, Kc, Vc;
    const Act& src = cross ? C : X;
    brotensor::flash_attention_project_kv_int8w_fp16(src.v, wk.q, wk.s, &bk.v, wv.q, wv.s, &bv.v, K, V);
    brotensor::flash_attention_project_kv(src.c, wk.w, &bk.c, wv.w, &bv.c, Kc, Vc);
    expect_close(download(K), Kc.to_host_vector(), 3e-3f, 3e-3f, std::string(tag) + " (project_kv K)");
    expect_close(download(V), Vc.to_host_vector(), 3e-3f, 3e-3f, std::string(tag) + " (project_kv V)");
    Tensor O2, O2c;
    brotensor::flash_attention_q_with_kv_cached_int8w_fp16(X.v, K, V, wq.q, wq.s, &bq.v, wo.q, wo.s, &bo.v, dm, H,
                                                           causal, O2);
    brotensor::flash_attention_q_with_kv_cached_forward(X.c, Kc, Vc, wq.w, &bq.c, wo.w, &bo.c, cm, H, causal, O2c);
    expect_close(download(O2), O2c.to_host_vector(), 6e-3f, 6e-3f, std::string(tag) + " (q_with_kv_cached)");
}

void test_sab_int8(int L, int D, int H, bool with_bias, bool masked, float scale, std::uint64_t seed) {
    const float amp = 1.0f / std::sqrt(float(D));
    Q8 wq = make_q8(D, D, seed, amp), wk = make_q8(D, D, seed + 1, amp), wv = make_q8(D, D, seed + 2, amp),
       wo = make_q8(D, D, seed + 3, amp);
    Act X = make_act(L, D, seed + 4);
    const std::vector<float> bias = random_values(std::size_t(H) * L * L, seed + 5, -1, 1, Dtype::FP32);
    Tensor bv = Tensor::from_host_on(vk(), bias.data(), H * L, L);
    Tensor bc = Tensor::from_host_on(Device::cpu(), bias.data(), H * L, L);
    std::vector<float> mask(L, 1.0f);
    if (masked) for (int i = 1; i < L; i += 4) mask[i] = 0.0f;
    Tensor mv = Tensor::from_host_on(vk(), mask.data(), L, 1);
    Tensor O, Oc;
    brotensor::self_attention_bias_int8w_fp16(X.v, wq.q, wq.s, wk.q, wk.s, wv.q, wv.s, wo.q, wo.s,
                                              masked ? static_cast<const float*>(mv.data) : nullptr,
                                              with_bias ? &bv : nullptr, H, scale, O);
    brotensor::self_attention_bias_forward(X.c, wq.w, wk.w, wv.w, wo.w, nullptr, nullptr, nullptr, nullptr,
                                           masked ? mask.data() : nullptr, with_bias ? &bc : nullptr, H, scale, Oc);
    char tag[96];
    std::snprintf(tag, sizeof tag, "self_attention_bias_int8w L=%d D=%d H=%d%s%s scale=%g", L, D, H,
                  with_bias ? " +bias" : "", masked ? " masked" : "", scale);
    expect_close(download(O), Oc.to_host_vector(), 6e-3f, 6e-3f, tag);
}

void test_resblock_int8(int N, int Cin, int Cout, int Hh, int Ww, bool shift, std::uint64_t seed) {
    const int G = 8;
    Act X = make_act(N, Cin * Hh * Ww, seed);
    Act g1 = make_act(Cin, 1, seed + 1, 0.5f, 1.5f), be1 = make_act(Cin, 1, seed + 2, -0.5f, 0.5f);
    Act g2 = make_act(Cout, 1, seed + 3, 0.5f, 1.5f), be2 = make_act(Cout, 1, seed + 4, -0.5f, 0.5f);
    Q8 w1 = make_q8(Cout, Cin * 9, seed + 5, 1.0f / std::sqrt(9.0f * Cin));
    Q8 w2 = make_q8(Cout, Cout * 9, seed + 6, 1.0f / std::sqrt(9.0f * Cout));
    Q8 ws = make_q8(Cout, Cin, seed + 7, 1.0f / std::sqrt(float(Cin)));
    Act b1 = make_act(Cout, 1, seed + 8, -0.2f, 0.2f), b2 = make_act(Cout, 1, seed + 9, -0.2f, 0.2f),
        bs = make_act(Cout, 1, seed + 10, -0.2f, 0.2f), te = make_act(Cout, 1, seed + 11, -0.5f, 0.5f);
    const bool skip = Cin != Cout;
    Tensor Y, Yc;
    brotensor::resblock_forward_int8w_fp16(X.v, g1.v, be1.v, w1.q, w1.s, &b1.v, shift ? &te.v : nullptr, g2.v, be2.v,
                                           w2.q, w2.s, &b2.v, skip ? &ws.q : nullptr, skip ? &ws.s : nullptr,
                                           skip ? &bs.v : nullptr, N, Cin, Cout, Hh, Ww, G, 1e-5f, Y);
    brotensor::resblock_forward(X.c, g1.c, be1.c, w1.w, &b1.c, shift ? &te.c : nullptr, g2.c, be2.c, w2.w, &b2.c,
                                skip ? &ws.w : nullptr, skip ? &bs.c : nullptr, N, Cin, Cout, Hh, Ww, G, 1e-5f, Yc);
    char tag[96];
    std::snprintf(tag, sizeof tag, "resblock_int8w N=%d %d->%d %dx%d%s", N, Cin, Cout, Hh, Ww, shift ? " +t_emb" : "");
    expect_close(download(Y), Yc.to_host_vector(), 1e-2f, 1e-2f, tag);
}

}  // namespace

void run_quant_attention_tests() {
    std::printf("\n[INT8-weight attention / ResBlock]\n");
    std::uint64_t seed = 900;
    test_flash_int8(1, 1, 64, 4, false, false, false, seed++);       // decode-sized: GEMV projections
    test_flash_int8(5, 5, 64, 4, false, true, false, seed++);
    test_flash_int8(37, 37, 128, 2, false, false, true, seed++);     // prefill-sized: coopmat projections
    test_flash_int8(40, 40, 96, 3, false, true, false, seed++);
    test_flash_int8(7, 50, 64, 2, true, false, true, seed++);        // cross attention
    test_flash_int8(64, 23, 128, 4, true, false, false, seed++);
    test_sab_int8(13, 64, 4, true, false, 1.0f, seed++);
    test_sab_int8(33, 128, 8, true, true, 1.0f, seed++);
    test_sab_int8(4, 64, 2, false, true, 0.125f, seed++);
    test_resblock_int8(1, 32, 32, 8, 8, false, seed++);
    test_resblock_int8(2, 32, 64, 9, 7, true, seed++);
    test_resblock_int8(1, 64, 32, 16, 16, true, seed++);
}

}  // namespace vkt
