// Vulkan parity for token selection against the CPU backend: sample_logits
// (greedy, temperature, top-k, top-p, over small and LLM-sized vocabularies),
// sample_logits_into (device counter: the same draws, advanced by N, untouched
// when greedy) and the masked-diffusion step (guided and not, the class top-k
// filter with tied values, Gumbel noise on classes and positions, commit).
// Indices must match exactly; the scores to FP32 rounding.

#include "test_vulkan_common.h"

#include <cmath>
#include <string>

namespace vkt {

namespace {

std::vector<std::int32_t> ints(const Tensor& t) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(t.size()));
    t.copy_to_host_raw(v.data(), v.size() * 4);
    return v;
}

void expect_ints(const std::vector<std::int32_t>& got, const std::vector<std::int32_t>& want, const std::string& tag) {
    int bad = 0, first = -1;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (got[i] != want[i]) { if (first < 0) first = int(i); ++bad; }
    }
    if (got.size() != want.size() || bad) {
        std::printf("  FAIL  %s: %d of %zu differ (first at %d: %d vs %d)\n", tag.c_str(), bad, want.size(), first,
                    first >= 0 ? got[first] : 0, first >= 0 ? want[first] : 0);
        ++failures();
        return;
    }
    std::printf("  PASS  %s\n", tag.c_str());
}

void test_sample(int N, int V, float temp, int top_k, float top_p, std::uint64_t seed) {
    auto x = random_values(std::size_t(N) * V, seed, -6, 6, Dtype::FP32);
    if (V > 8) for (int n = 0; n < N; ++n) x[std::size_t(n) * V + 3] = x[std::size_t(n) * V + 5];   // a tie
    Tensor L = upload(x, N, V, Dtype::FP32), I;
    Tensor Lc = Tensor::from_host_on(Device::cpu(), x.data(), N, V), Ic;
    const std::uint64_t key = 0x1234567890abcdefULL ^ seed, counter = 1000 + seed;
    brotensor::sample_logits(L, temp, top_k, top_p, key, counter, I);
    brotensor::sample_logits(Lc, temp, top_k, top_p, key, counter, Ic);
    char tag[96];
    std::snprintf(tag, sizeof tag, "sample_logits N=%d V=%d T=%g top_k=%d top_p=%g", N, V, temp, top_k, top_p);
    expect_ints(ints(I), ints(Ic), tag);
}

void test_sample_into() {
    const int N = 4, V = 300;
    const auto x = random_values(std::size_t(N) * V, 7, -3, 3, Dtype::FP32);
    Tensor L = upload(x, N, V, Dtype::FP32);
    const std::int32_t c0 = 77;
    Tensor cnt = Tensor::from_raw_bytes_on(vk(), &c0, 1, 1, Dtype::INT32, 4);
    Tensor scratch = Tensor::zeros_on(vk(), 3 * N, V);
    Tensor I = Tensor::zeros_on(vk(), N, 1, Dtype::INT32), Ref;
    for (int step = 0; step < 3; ++step) {
        brotensor::sample_logits_into(L, 0.9f, 20, 0.95f, 42, cnt, scratch, I);
        brotensor::sample_logits(L, 0.9f, 20, 0.95f, 42, std::uint64_t(c0 + step * N), Ref);
        expect_ints(ints(I), ints(Ref), "sample_logits_into step " + std::to_string(step) + " == sample_logits");
    }
    VKT_CHECK(ints(cnt)[0] == c0 + 3 * N);
    brotensor::sample_logits_into(L, 0.0f, 0, 1.0f, 42, cnt, scratch, I);   // greedy: no draw, no advance
    VKT_CHECK(ints(cnt)[0] == c0 + 3 * N);
}

void test_masked_diffusion(bool guided, float class_t, float pos_t, std::uint64_t seed) {
    const int T = 7, C = 3, V = 61, mask_id = 60, R = guided ? 2 * T : T;
    auto x = random_values(std::size_t(R) * C * V, seed, -4, 4, Dtype::FP32);
    for (float& v : x) v = std::round(v * 2.0f) * 0.5f;   // many tied values for the top-k filter
    std::vector<std::int32_t> tok(std::size_t(C) * T);
    for (std::size_t i = 0; i < tok.size(); ++i) tok[i] = (i % 3 == 0) ? int(i % 50) : mask_id;
    Tensor L = upload(x, R, C * V, Dtype::FP32), Lc = Tensor::from_host_on(Device::cpu(), x.data(), R, C * V);
    Tensor Tk = Tensor::from_raw_bytes_on(vk(), tok.data(), C, T, Dtype::INT32, tok.size() * 4);
    Tensor Tkc = Tensor::from_raw_bytes_on(Device::cpu(), tok.data(), C, T, Dtype::INT32, tok.size() * 4);
    Tensor P, S, F, Pc, Sc, Fc;
    const float gs = guided ? 1.5f : 0.0f;
    brotensor::masked_diffusion_scores(L, Tk, T, C, V, mask_id, gs, 0.25f, pos_t, class_t, 0.2f, seed, P, S, F);
    brotensor::masked_diffusion_scores(Lc, Tkc, T, C, V, mask_id, gs, 0.25f, pos_t, class_t, 0.2f, seed, Pc, Sc, Fc);
    char tag[96];
    std::snprintf(tag, sizeof tag, "masked_diffusion_scores%s class_t=%g pos_t=%g", guided ? " guided" : "", class_t,
                  pos_t);
    expect_ints(ints(P), ints(Pc), std::string(tag) + " pred");
    expect_close(download(S), Sc.to_host_vector(), 2e-5f, 2e-5f, std::string(tag) + " scores");
    expect_close(download(F), Fc.to_host_vector(), 2e-5f, 2e-5f, std::string(tag) + " confidence");

    std::vector<std::int32_t> idx = {4, 0, 19, -1, 5, 1000, 7};
    Tensor Ix = Tensor::from_raw_bytes_on(vk(), idx.data(), 1, int(idx.size()), Dtype::INT32, idx.size() * 4);
    Tensor Ixc = Tensor::from_raw_bytes_on(Device::cpu(), idx.data(), 1, int(idx.size()), Dtype::INT32, idx.size() * 4);
    std::vector<std::int32_t> zero(tok.size(), -3);
    Tensor U = Tensor::from_raw_bytes_on(vk(), zero.data(), C, T, Dtype::INT32, zero.size() * 4);
    Tensor Uc = Tensor::from_raw_bytes_on(Device::cpu(), zero.data(), C, T, Dtype::INT32, zero.size() * 4);
    brotensor::masked_diffusion_commit(Pc.to(vk()), Ix, 6, 3, Tk, U);
    brotensor::masked_diffusion_commit(Pc, Ixc, 6, 3, Tkc, Uc);
    expect_ints(ints(Tk), ints(Tkc), std::string(tag) + " commit tokens");
    expect_ints(ints(U), ints(Uc), std::string(tag) + " commit unmask_step");
}

}  // namespace

void run_sampling_tests() {
    std::uint64_t seed = 300;
    test_sample(3, 7, 0.0f, 0, 1.0f, seed++);
    test_sample(5, 3072, 0.0f, 0, 1.0f, seed++);
    for (int V : {7, 3072, 151936}) {
        for (int top_k : {0, 1, 5, 50}) {
            for (float top_p : {1.0f, 0.9f, 0.3f}) {
                if (V == 151936 && top_k == 1) continue;
                test_sample(V > 100000 ? 2 : 6, V, 0.8f, top_k, top_p, seed++);
            }
        }
    }
    test_sample(4, 500, 1.7f, 600, 0.0f, seed++);   // top_k past V, top_p 0 (keeps one)
    test_sample_into();
    for (bool guided : {false, true}) {
        test_masked_diffusion(guided, 0.0f, 0.0f, seed++);
        test_masked_diffusion(guided, 0.8f, 1.0f, seed++);
    }
}

}  // namespace vkt
