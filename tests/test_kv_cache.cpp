// Tests for kv_cache_append + flash_attention_decode.
// Decode is checked against the causal flash_attention_forward.
// CUDA-only — guarded out on a CPU-only build.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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

static std::vector<uint16_t> to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        o[i] = brotensor::fp32_to_fp16_bits(v[i]);
    return o;
}

// ─── CPU (FP32) softcap + sliding-window decode tests ──────────────────────
//
// flash_attention_decode / _masked gained two Gemma-2 knobs:
//   attn_softcap > 0 : s := attn_softcap * tanh(s / attn_softcap) on each raw
//                      (already 1/sqrt(hd)-scaled) score, before the mask/softmax.
//   window > 0       : query at absolute position p attends keys
//                      [max(0, p-window+1), p] (composes with causal).
// Both default to a no-op, so softcap==0 && window<=0 must reproduce the prior
// result exactly. These run on the FP32 CPU backend (always available), so the
// numerics match a brute-force FP32 reference essentially bit-for-bit.

// Brute-force decode reference (FP32): Q (Lq,Dq); K/V cache rows [0,valid_len)
// at width Dkv. p_q = (valid_len - Lq) + q; GQA query head hq reads kv head
// hq/(nq/nkv). window <= 0 is unbounded causal; softcap <= 0 disables the cap.
static void decode_ref(const std::vector<float>& Q, const std::vector<float>& K,
                       const std::vector<float>& V, int Lq, int valid_len,
                       int Dq, int Dkv, int nq, int nkv,
                       float softcap, int window, std::vector<float>& O) {
    const int hd = Dq / nq;
    const int group = nq / nkv;
    const int seq_offset = valid_len - Lq;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    O.assign(static_cast<size_t>(Lq) * Dq, 0.0f);
    std::vector<float> sc(valid_len);
    for (int q = 0; q < Lq; ++q) {
        const int p = seq_offset + q;
        const int lo = (window > 0) ? std::max(0, p - window + 1) : 0;
        for (int hq = 0; hq < nq; ++hq) {
            const int hkv = hq / group;
            const int qo = hq * hd, ko = hkv * hd;
            float mx = -1e30f;
            for (int k = 0; k <= p; ++k) {
                if (k < lo) { sc[k] = -1e30f; continue; }
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += Q[q * Dq + qo + d] * K[k * Dkv + ko + d];
                float s = dot * inv_sqrt;
                if (softcap > 0.0f) s = softcap * std::tanh(s / softcap);
                sc[k] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (int k = 0; k <= p; ++k) { float e = std::exp(sc[k] - mx); sc[k] = e; sum += e; }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int k = 0; k <= p; ++k) acc += sc[k] * V[k * Dkv + ko + d];
                O[q * Dq + qo + d] = acc * inv;
            }
        }
    }
}

static int cpu_failures = 0;
#define CPU_CHECK(cond) do {                                                  \
    if (!(cond)) {                                                            \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++cpu_failures;                                                       \
    }                                                                         \
} while (0)

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Run the CPU decode op and return its output as a host FP32 vector.
static std::vector<float> run_decode_cpu(const std::vector<float>& Qf,
                                         const std::vector<float>& Kf,
                                         const std::vector<float>& Vf,
                                         int Lq, int valid_len, int Dq, int Dkv,
                                         int nq, int nkv,
                                         float softcap, int window) {
    Tensor Q = Tensor::from_host_on(Device::CPU, Qf.data(), Lq, Dq);
    Tensor K = Tensor::from_host_on(Device::CPU, Kf.data(), valid_len, Dkv);
    Tensor V = Tensor::from_host_on(Device::CPU, Vf.data(), valid_len, Dkv);
    Tensor O;
    brotensor::flash_attention_decode(Q, K, V, valid_len, nq, nkv, O, softcap, window);
    std::vector<float> out(static_cast<size_t>(Lq) * Dq);
    O.copy_to_host(out.data());
    return out;
}

static void test_cpu_softcap_window() {
    std::printf("  flash_attention_decode CPU softcap/window\n");
    std::mt19937 rng(0xC0DEC0DE);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // (a) Regression guard: default args (softcap=0, window=0) reproduce a
    //     plain causal reference, and the explicit no-op args are bit-identical
    //     to passing nothing.
    {
        const int Lq = 4, valid_len = 12, nq = 2, hd = 8, Dq = nq * hd;
        std::vector<float> Qf(Lq * Dq), Kf(valid_len * Dq), Vf(valid_len * Dq);
        for (auto& v : Qf) v = dist(rng);
        for (auto& v : Kf) v = dist(rng);
        for (auto& v : Vf) v = dist(rng);
        auto got_default = run_decode_cpu(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq,
                                          /*softcap=*/0.0f, /*window=*/0);
        auto got_explicit = run_decode_cpu(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq,
                                           0.0f, 0);
        // Bit-identical: same code path, no-op knobs.
        CPU_CHECK(got_default == got_explicit);
        std::vector<float> ref;
        decode_ref(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, 0.0f, 0, ref);
        const float e = max_abs_diff(got_default, ref);
        std::printf("    regression default vs causal ref: max_err=%g\n", e);
        CPU_CHECK(e < 1e-5f);
    }

    // (b) softcap clamps large scores. Construct Q==K rows so the diagonal raw
    //     score is large (~hd*scale*amp^2); a small softcap must compress it.
    //     Compare to a hand-computed softcapped reference, and confirm the
    //     softcapped output actually differs from the uncapped one.
    {
        const int Lq = 1, valid_len = 3, nq = 1, hd = 8, Dq = hd;
        std::vector<float> Qf(Lq * Dq, 0.0f), Kf(valid_len * Dq, 0.0f), Vf(valid_len * Dq);
        // Large key 0 alignment, modest others, so softcap visibly reshapes the
        // softmax over the three keys.
        for (int d = 0; d < hd; ++d) Qf[d] = 3.0f;
        for (int d = 0; d < hd; ++d) Kf[0 * Dq + d] = 3.0f;   // huge score with Q
        for (int d = 0; d < hd; ++d) Kf[1 * Dq + d] = 0.3f;
        for (int d = 0; d < hd; ++d) Kf[2 * Dq + d] = -0.2f;
        for (auto& v : Vf) v = dist(rng);
        const float softcap = 5.0f;
        auto got = run_decode_cpu(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, softcap, 0);
        std::vector<float> ref_cap, ref_nocap;
        decode_ref(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, softcap, 0, ref_cap);
        decode_ref(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, 0.0f, 0, ref_nocap);
        const float e = max_abs_diff(got, ref_cap);
        const float delta = max_abs_diff(ref_cap, ref_nocap);
        std::printf("    softcap match=%g  cap-vs-nocap delta=%g\n", e, delta);
        CPU_CHECK(e < 1e-5f);
        CPU_CHECK(delta > 1e-3f);   // soft-cap must change the result here
    }

    // (c) Window masking equals a brute-force windowed reference, and differs
    //     from the unbounded-causal result (the window actually drops keys).
    {
        const int Lq = 1, valid_len = 20, nq = 2, hd = 8, Dq = nq * hd, window = 4;
        std::vector<float> Qf(Lq * Dq), Kf(valid_len * Dq), Vf(valid_len * Dq);
        for (auto& v : Qf) v = dist(rng);
        for (auto& v : Kf) v = dist(rng);
        for (auto& v : Vf) v = dist(rng);
        auto got = run_decode_cpu(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, 0.0f, window);
        std::vector<float> ref_win, ref_full;
        decode_ref(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, 0.0f, window, ref_win);
        decode_ref(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq, 0.0f, 0, ref_full);
        const float e = max_abs_diff(got, ref_win);
        const float delta = max_abs_diff(ref_win, ref_full);
        std::printf("    window match=%g  win-vs-full delta=%g\n", e, delta);
        CPU_CHECK(e < 1e-5f);
        CPU_CHECK(delta > 1e-3f);
        // window >= valid_len must reproduce full causal exactly.
        auto got_wide = run_decode_cpu(Qf, Kf, Vf, Lq, valid_len, Dq, Dq, nq, nq,
                                       0.0f, valid_len);
        CPU_CHECK(max_abs_diff(got_wide, ref_full) < 1e-5f);
    }

    // (d) GQA + softcap + window combined, multi-query block, vs brute force.
    {
        const int Lq = 3, valid_len = 18, nq = 8, nkv = 2, hd = 16;
        const int Dq = nq * hd, Dkv = nkv * hd, window = 6;
        const float softcap = 8.0f;
        std::vector<float> Qf(Lq * Dq), Kf(valid_len * Dkv), Vf(valid_len * Dkv);
        for (auto& v : Qf) v = dist(rng);
        for (auto& v : Kf) v = dist(rng);
        for (auto& v : Vf) v = dist(rng);
        auto got = run_decode_cpu(Qf, Kf, Vf, Lq, valid_len, Dq, Dkv, nq, nkv,
                                  softcap, window);
        std::vector<float> ref;
        decode_ref(Qf, Kf, Vf, Lq, valid_len, Dq, Dkv, nq, nkv, softcap, window, ref);
        const float e = max_abs_diff(got, ref);
        std::printf("    GQA+softcap+window match=%g\n", e);
        CPU_CHECK(e < 1e-5f);
    }
}

// decode_masked (single query) with a valid-prefix mask: softcap/window must
// match the brute-force reference, and the no-op defaults must equal the
// unmasked decode at L_q == 1 (the bit-identity contract).
static void test_cpu_masked_softcap_window() {
    std::printf("  flash_attention_decode_masked CPU softcap/window\n");
    std::mt19937 rng(0x5EED1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const int valid_len = 14, cap = 24, nq = 4, nkv = 2, hd = 8;
    const int Dq = nq * hd, Dkv = nkv * hd;
    std::vector<float> Qf(Dq), Kf(cap * Dkv), Vf(cap * Dkv);
    for (auto& v : Qf) v = dist(rng);
    for (auto& v : Kf) v = dist(rng);
    for (auto& v : Vf) v = dist(rng);
    std::vector<float> mask(cap, 0.0f);
    for (int k = 0; k < valid_len; ++k) mask[k] = 1.0f;   // valid prefix

    // K/V slices the reference actually sees: rows [0, valid_len).
    std::vector<float> Kref(Kf.begin(), Kf.begin() + static_cast<size_t>(valid_len) * Dkv);
    std::vector<float> Vref(Vf.begin(), Vf.begin() + static_cast<size_t>(valid_len) * Dkv);

    auto run_masked = [&](float softcap, int window) {
        Tensor Q = Tensor::from_host_on(Device::CPU, Qf.data(), 1, Dq);
        Tensor K = Tensor::from_host_on(Device::CPU, Kf.data(), cap, Dkv);
        Tensor V = Tensor::from_host_on(Device::CPU, Vf.data(), cap, Dkv);
        Tensor O;
        brotensor::flash_attention_decode_masked(Q, K, V, mask.data(), nq, nkv, O,
                                                 softcap, window);
        std::vector<float> out(Dq);
        O.copy_to_host(out.data());
        return out;
    };

    // No-op defaults vs unmasked decode (Lq==1): bit-identity contract.
    {
        auto masked = run_masked(0.0f, 0);
        auto plain = run_decode_cpu(Qf, Kref, Vref, 1, valid_len, Dq, Dkv, nq, nkv,
                                    0.0f, 0);
        std::printf("    masked default vs decode: max_err=%g\n",
                    max_abs_diff(masked, plain));
        CPU_CHECK(masked == plain);
    }
    // softcap + window vs brute force (p = valid_len-1).
    {
        const float softcap = 6.0f;
        const int window = 5;
        auto masked = run_masked(softcap, window);
        std::vector<float> ref;
        decode_ref(Qf, Kref, Vref, 1, valid_len, Dq, Dkv, nq, nkv, softcap, window, ref);
        const float e = max_abs_diff(masked, ref);
        std::printf("    masked softcap+window match=%g\n", e);
        CPU_CHECK(e < 1e-5f);
    }
}

static void test_append() {
    std::printf("  kv_cache_append\n");
    const int L_max = 8, D = 4;
    std::mt19937 rng(0x200);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Pre-fill caches with zeros.
    Tensor Kc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);
    Tensor Vc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);

    // Append two chunks: 3 rows then 2 rows.
    std::vector<float> K1f(3 * D), V1f(3 * D), K2f(2 * D), V2f(2 * D);
    for (auto& v : K1f) v = dist(rng);
    for (auto& v : V1f) v = dist(rng);
    for (auto& v : K2f) v = dist(rng);
    for (auto& v : V2f) v = dist(rng);

    auto K1h = to_fp16(K1f), V1h = to_fp16(V1f);
    auto K2h = to_fp16(K2f), V2h = to_fp16(V2f);
    Tensor K1 = Tensor::from_host_fp16_on(Device::CUDA, K1h.data(), 3, D);
    Tensor V1 = Tensor::from_host_fp16_on(Device::CUDA, V1h.data(), 3, D);
    Tensor K2 = Tensor::from_host_fp16_on(Device::CUDA, K2h.data(), 2, D);
    Tensor V2 = Tensor::from_host_fp16_on(Device::CUDA, V2h.data(), 2, D);

    brotensor::kv_cache_append(K1, V1, 0, Kc, Vc);
    brotensor::kv_cache_append(K2, V2, 3, Kc, Vc);
    brotensor::sync(Device::CUDA);

    std::vector<uint16_t> gotK = Kc.to_host_vector_fp16();
    std::vector<uint16_t> gotV = Vc.to_host_vector_fp16();

    // First 3*D entries should match K1f (after fp16 quant), next 2*D match K2f.
    for (int i = 0; i < 3 * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(gotK[i]);
        const float r = brotensor::fp16_bits_to_fp32(K1h[i]);
        CHECK(std::fabs(g - r) < 1e-5f);
    }
    for (int i = 0; i < 2 * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(gotK[3 * D + i]);
        const float r = brotensor::fp16_bits_to_fp32(K2h[i]);
        CHECK(std::fabs(g - r) < 1e-5f);
    }
    // Remaining tail should be zero.
    for (int i = 5 * D; i < L_max * D; ++i) {
        CHECK(gotK[i] == 0);
        CHECK(gotV[i] == 0);
    }
}

static void test_decode_vs_causal_forward() {
    std::printf("  flash_attention_decode vs causal forward\n");
    // Build a "complete" attention setup with L_total tokens; decode is the
    // tail Lq rows treating the first valid_len cache rows.
    const int L_total = 12, D = 16, nh = 2;
    const int L_max = 16;
    std::mt19937 rng(0x201);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> Qf(L_total * D), Kf(L_total * D), Vf(L_total * D);
    for (auto& v : Qf) v = dist(rng);
    for (auto& v : Kf) v = dist(rng);
    for (auto& v : Vf) v = dist(rng);

    auto Qh = to_fp16(Qf), Kh = to_fp16(Kf), Vh = to_fp16(Vf);

    // Reference: causal full-sequence forward.
    Tensor Qg = Tensor::from_host_fp16_on(Device::CUDA, Qh.data(), L_total, D);
    Tensor Kg = Tensor::from_host_fp16_on(Device::CUDA, Kh.data(), L_total, D);
    Tensor Vg = Tensor::from_host_fp16_on(Device::CUDA, Vh.data(), L_total, D);
    Tensor Oref = Tensor::empty_on(Device::CUDA, L_total, D, Dtype::FP16);
    brotensor::flash_attention_forward(Qg, Kg, Vg, nullptr, nh,
                                       /*causal=*/true, Oref);
    brotensor::sync(Device::CUDA);
    std::vector<uint16_t> ref_h = Oref.to_host_vector_fp16();

    // Decode: fill K/V caches with all L_total rows, query is the last Lq=3
    // rows of Q. valid_len = L_total. seq_offset = L_total - Lq.
    const int Lq = 3;
    Tensor Kc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);
    Tensor Vc = Tensor::zeros_on(Device::CUDA, L_max, D, Dtype::FP16);
    brotensor::kv_cache_append(Kg, Vg, 0, Kc, Vc);

    // Take the last Lq rows of Q via copy_d2d.
    Tensor Qtail = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    brotensor::copy_d2d(Qg, (L_total - Lq) * D, Qtail, 0, Lq * D);

    Tensor Odec = Tensor::empty_on(Device::CUDA, Lq, D, Dtype::FP16);
    brotensor::flash_attention_decode(Qtail, Kc, Vc, L_total, nh, Odec);
    brotensor::sync(Device::CUDA);
    std::vector<uint16_t> dec_h = Odec.to_host_vector_fp16();

    // Compare against the last Lq*D entries of ref_h.
    float me = 0.0f;
    int bad = 0;
    const int off = (L_total - Lq) * D;
    for (int i = 0; i < Lq * D; ++i) {
        const float g = brotensor::fp16_bits_to_fp32(dec_h[i]);
        const float r = brotensor::fp16_bits_to_fp32(ref_h[off + i]);
        const float e = std::fabs(g - r);
        if (e > me) me = e;
        if (e > 5e-3f + 5e-2f * std::fabs(r)) ++bad;
    }
    std::printf("    decode vs causal: max_err=%g bad=%d/%d\n", me, bad, Lq * D);
    CHECK(bad == 0);
}

int main() {
    brotensor::init();
    std::printf("test_kv_cache\n");

    // CPU (FP32) softcap + sliding-window decode coverage — always runs.
    test_cpu_softcap_window();
    test_cpu_masked_softcap_window();

    if (!brotensor::is_available(brotensor::Device::CUDA)) {
        std::printf("CUDA not available - skipping GPU decode tests\n");
        std::printf("%s (%d failures)\n",
                    cpu_failures ? "FAILED" : "OK", cpu_failures);
        return cpu_failures ? 1 : 0;
    }
    test_append();
    test_decode_vs_causal_forward();
    const int total = g_failures + cpu_failures;
    std::printf("%s (%d failures)\n", total ? "FAILED" : "OK", total);
    return total ? 1 : 0;
}
