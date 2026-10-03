// Vulkan-vs-HIP parity, second half: attention (flash, GQA, varlen, packed
// QKV, decode), conv2d / conv3d, RoPE / GLU / softmax / xent / samplers, and
// the audio front end (resample1d, STFT, log-mel, iSTFT). Harness and the
// comparison rule: test_vulkan_hip_parity.h.

#include "test_vulkan_hip_parity.h"

#include <cmath>
#include <cstring>

namespace vhp {

namespace {

constexpr double kF16 = 1.0 / 1024.0;

// FP16 attention: P is rounded to FP16 in the cooperative-matrix kernels of
// both backends, so outputs move by a few FP16 ulps of the V range.
constexpr double kAttnTol = 4 * kF16;

}  // namespace

// ─── attention ──────────────────────────────────────────────────────────────

void run_attention() {
    std::printf("\n[attention]\n");
    const Dtype h = Dtype::FP16;
    {   // SD1.5 self-attention at 32x32 latent: L 1024, 5 heads x 64 (D 320).
        const int L = 1024, H = 5, D = 320;
        const auto q = rnd(std::size_t(L) * D, 400, -1.5f, 1.5f, h), k = rnd(std::size_t(L) * D, 401, -1.5f, 1.5f, h),
                   v = rnd(std::size_t(L) * D, 402, -1.0f, 1.0f, h);
        auto run = [&](Device d) {
            Tensor O;
            brotensor::flash_attention_forward(up(q, L, D, h, d), up(k, L, D, h, d), up(v, L, D, h, d), nullptr, H,
                                               false, O);
            return down(O);
        };
        check("flash_attention_forward f16", shp("L%d H%d hd%d", L, H, D / H), run, [&] { return run(kCpu); }, 0,
              kAttnTol);
    }
    {   // Qwen3-0.6B prefill: 16 q heads, 8 kv heads, hd 128, causal.
        const int L = 256, HQ = 16, HK = 8, hd = 128;
        const auto q = rnd(std::size_t(L) * HQ * hd, 410, -1.5f, 1.5f, h);
        const auto k = rnd(std::size_t(L) * HK * hd, 411, -1.5f, 1.5f, h);
        const auto v = rnd(std::size_t(L) * HK * hd, 412, -1.0f, 1.0f, h);
        for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
            auto run = [&](Device d) {
                Tensor O;
                brotensor::flash_attention_gqa_forward(up(q, L, HQ * hd, dt, d), up(k, L, HK * hd, dt, d),
                                                       up(v, L, HK * hd, dt, d), nullptr, HQ, HK, true, O);
                return down(O);
            };
            check(shp("flash_attention_gqa causal %s", vkt::dt_name(dt)), shp("L%d %d/%d hd%d", L, HQ, HK, hd), run,
                  [&] { return run(kCpu); }, 0, dt == Dtype::BF16 ? 2.0 / 128 : kAttnTol);
        }
    }
    {   // Qwen-VL window attention: packed sequences, 16 heads x 80.
        const std::vector<int32_t> cu = {0, 100, 137, 393};
        const int B = 3, T = cu.back(), H = 16, hd = 80, D = H * hd;
        const auto q = rnd(std::size_t(T) * D, 420, -1.5f, 1.5f, h), k = rnd(std::size_t(T) * D, 421, -1.5f, 1.5f, h),
                   v = rnd(std::size_t(T) * D, 422, -1.0f, 1.0f, h);
        check("flash_attention_varlen f16", shp("T%d B%d H%d hd%d", T, B, H, hd),
              [&](Device d) {
                  Tensor O, c = i32(cu, B + 1, 1, d);
                  brotensor::flash_attention_varlen_forward(up(q, T, D, h, d), up(k, T, D, h, d), up(v, T, D, h, d),
                                                            static_cast<const int32_t*>(c.data),
                                                            static_cast<const int32_t*>(c.data), B, 256, 256, H, hd,
                                                            false, O);
                  return down(O);
              },
              [&] {
                  Tensor O;
                  brotensor::flash_attention_varlen_forward(up(q, T, D, h, kCpu), up(k, T, D, h, kCpu),
                                                            up(v, T, D, h, kCpu), cu.data(), cu.data(), B, 256, 256, H,
                                                            hd, false, O);
                  return O.to_host_vector();
              },
              0, kAttnTol);
    }
    for (int window : {0, 128}) {   // ModernBERT-style packed encoder batch: 12 heads x 64.
        const std::vector<int> lens = {64, 1, 200, 47};
        std::vector<int32_t> bounds;
        int L = 0;
        for (int n : lens) {
            for (int i = 0; i < n; ++i) { bounds.push_back(L); bounds.push_back(L + n); }
            L += n;
        }
        const int H = 12, D = H * 64;
        const auto qkv = rnd(std::size_t(L) * 3 * D, 430 + window, -1.5f, 1.5f, h);
        auto run = [&](Device d) {
            Tensor O;
            brotensor::flash_attention_packed_qkv_forward(up(qkv, L, 3 * D, h, d), i32(bounds, L, 2, d), H, window,
                                                          O);
            return down(O);
        };
        check("flash_attention_packed_qkv f16", shp("L%d seqs%zu H%d w%d", L, lens.size(), H, window), run,
              [&] { return run(kCpu); }, 0, kAttnTol);
    }
    {   // Qwen3 decode over a 2048-slot cache, 1500 valid.
        const int cap = 2048, valid = 1500, HQ = 16, HK = 8, hd = 128;
        const auto q = rnd(std::size_t(HQ) * hd, 440, -1.5f, 1.5f, h);
        const auto k = rnd(std::size_t(cap) * HK * hd, 441, -1.5f, 1.5f, h);
        const auto v = rnd(std::size_t(cap) * HK * hd, 442, -1.0f, 1.0f, h);
        for (int window : {0, 512}) {
            auto run = [&](Device d) {
                Tensor O;
                brotensor::flash_attention_decode(up(q, 1, HQ * hd, h, d), up(k, cap, HK * hd, h, d),
                                                  up(v, cap, HK * hd, h, d), valid, HQ, HK, O, 0.0f, window);
                return down(O);
            };
            check("flash_attention_decode f16", shp("valid%d %d/%d hd%d w%d", valid, HQ, HK, hd, window), run,
                  [&] { return run(kCpu); }, 0, kAttnTol);
        }
    }
}

// ─── conv ───────────────────────────────────────────────────────────────────

void run_conv() {
    std::printf("\n[conv]\n");
    struct C2 { int n, cin, h, w, cout, k, s, p, groups; const char* name; bool cpu; };
    const C2 cases[] = {
        {1, 320, 32, 32, 320, 3, 1, 1, 1, "sd1.5 resblock 3x3", false},
        {1, 4, 64, 64, 320, 3, 1, 1, 1, "sd1.5 conv_in", true},
        {1, 640, 16, 16, 640, 1, 1, 0, 1, "1x1 skip", false},
        {1, 128, 64, 64, 128, 3, 2, 1, 1, "vae down s2", false},
        {1, 96, 56, 56, 96, 7, 1, 3, 96, "convnext depthwise 7x7", true},
    };
    std::uint64_t seed = 500;
    for (const C2& c : cases) {
        const int K = c.cin / c.groups * c.k * c.k;
        const auto x = rnd(std::size_t(c.n) * c.cin * c.h * c.w, seed++, -1.0f, 1.0f, Dtype::FP16);
        const auto w = rnd(std::size_t(c.cout) * K, seed++, -0.08f, 0.08f, Dtype::FP16);
        const auto b = rnd(c.cout, seed++, -0.5f, 0.5f, Dtype::FP16);
        auto run = [&](Device d) {
            Tensor Bt = up(b, c.cout, 1, Dtype::FP16, d), Y;
            brotensor::conv2d_forward(up(x, c.n, c.cin * c.h * c.w, Dtype::FP16, d), up(w, c.cout, K, Dtype::FP16, d),
                                      &Bt, c.n, c.cin, c.h, c.w, c.cout, c.k, c.k, c.s, c.s, c.p, c.p, 1, 1, c.groups,
                                      Y);
            return down(Y);
        };
        check("conv2d_forward f16", shp("%s %d->%d %dx%d", c.name, c.cin, c.cout, c.h, c.w), run,
              c.cpu ? CpuFn([&] { return run(kCpu); }) : CpuFn(), 0, 3 * kF16);
    }
    {   // Qwen2.5-VL patch embedding: 2x14x14 patches, 3 -> 1280.
        const int N = 64, C = 3, T = 2, H = 14, W = 14, Co = 1280, K = C * T * H * W;
        const auto x = rnd(std::size_t(N) * K, seed++, -1.0f, 1.0f, Dtype::FP16);
        const auto w = rnd(std::size_t(Co) * K, seed++, -0.05f, 0.05f, Dtype::FP16);
        auto run = [&](Device d) {
            Tensor Y;
            brotensor::conv3d_forward(up(x, N, K, Dtype::FP16, d), up(w, Co, K, Dtype::FP16, d), nullptr, N, C, T, H,
                                      W, Co, T, H, W, T, H, W, 0, 0, 0, 1, 1, 1, Y);
            return down(Y);
        };
        check("conv3d_forward f16", shp("qwen-vl patch N%d %d->%d", N, K, Co), run, [&] { return run(kCpu); }, 0,
              3 * kF16);
    }
    {   // A 3x3x3 video conv (TripoSplat / VAE-3D shapes, small).
        const int N = 1, C = 16, T = 5, H = 24, W = 24, Co = 32, K = C * 27;
        const auto x = rnd(std::size_t(N) * C * T * H * W, seed++, -1.0f, 1.0f, Dtype::FP16);
        const auto w = rnd(std::size_t(Co) * K, seed++, -0.1f, 0.1f, Dtype::FP16);
        auto run = [&](Device d) {
            Tensor Y;
            brotensor::conv3d_forward(up(x, N, C * T * H * W, Dtype::FP16, d), up(w, Co, K, Dtype::FP16, d), nullptr,
                                      N, C, T, H, W, Co, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, Y);
            return down(Y);
        };
        check("conv3d_forward f16", shp("3x3x3 %d->%d %dx%dx%d", C, Co, T, H, W), run, [&] { return run(kCpu); }, 0,
              3 * kF16);
    }
}

// ─── RoPE, GLU, softmax, xent, samplers ─────────────────────────────────────

void run_misc() {
    std::printf("\n[rope / glu / softmax / samplers]\n");
    {   // Qwen3: 16 heads x 128, theta 1e6, prefill at offset 37. FP32: HIP's
        // angle (pos * theta^(-2i/d), fast sin/cos) is the less exact one,
        // 6e-5 from the CPU at |x| <= 2; Vulkan is within 1e-5.
        const int L = 256, H = 16, hd = 128, D = H * hd;
        const auto x = rnd(std::size_t(L) * D, 600, -2.0f, 2.0f, Dtype::FP16);
        for (Dtype dt : {Dtype::FP16, Dtype::FP32}) {
            auto run = [&](Device d) {
                Tensor Y;
                brotensor::rope_forward(up(x, L, D, dt, d), hd, H, 37, 1e6f, Y);
                return down(Y);
            };
            check(shp("rope_forward %s", vkt::dt_name(dt)), shp("L%d H%d hd%d", L, H, hd), run,
                  [&] { return run(kCpu); }, dt == Dtype::FP32 ? 1e-4 : 0, dt == Dtype::FP32 ? 0 : 2 * kF16);
        }
        std::vector<float> cs(std::size_t(L) * hd / 2), sn(cs.size());
        for (int r = 0; r < L; ++r)
            for (int i = 0; i < hd / 2; ++i) {
                const double t = (r * 3 + 5) * std::pow(10000.0, -2.0 * i / hd);
                cs[std::size_t(r) * hd / 2 + i] = float(std::cos(t));
                sn[std::size_t(r) * hd / 2 + i] = float(std::sin(t));
            }
        for (Dtype dt : {Dtype::FP16, Dtype::BF16}) {
            auto run = [&](Device d) {
                Tensor Y;
                brotensor::rope_apply(up(x, L, D, dt, d), up(cs, L, hd / 2, Dtype::FP32, d),
                                      up(sn, L, hd / 2, Dtype::FP32, d), hd, H, Y);
                return down(Y);
            };
            check(shp("rope_apply %s", vkt::dt_name(dt)), shp("L%d H%d hd%d", L, H, hd), run,
                  [&] { return run(kCpu); }, 0, dt == Dtype::BF16 ? 2.0 / 128 : 2 * kF16);
        }
    }
    {   // SwiGLU (Qwen3 MLP 3072) and exact GeGLU (SD1.5 FF 1280).
        const auto xs = rnd(std::size_t(128) * 2 * 3072, 610, -4.0f, 4.0f, Dtype::FP16);
        auto sw = [&](Device d) {
            Tensor Y;
            brotensor::swiglu_forward(up(xs, 128, 2 * 3072, Dtype::FP16, d), Y);
            return down(Y);
        };
        check("swiglu_forward f16", "B128 D3072", sw, [&] { return sw(kCpu); }, 0, 2 * kF16);
        const auto xg = rnd(std::size_t(1024) * 2 * 1280, 611, -4.0f, 4.0f, Dtype::FP16);
        auto gg = [&](Device d) {
            Tensor Y;
            brotensor::geglu_exact_forward(up(xg, 1024, 2 * 1280, Dtype::FP16, d), Y);
            return down(Y);
        };
        check("geglu_exact_forward f16", "B1024 D1280", gg, [&] { return gg(kCpu); }, 0, 2 * kF16);
    }
    {   // Softmax over attention-score rows; xent over brogameagent's action heads.
        const int R = 1024, C = 1024;
        const auto x = rnd(std::size_t(R) * C, 620, -8.0f, 8.0f, Dtype::FP32);
        auto sm = [&](Device d) {
            Tensor Y;
            brotensor::softmax_rows_forward(up(x, R, C, Dtype::FP32, d), Y, R, C);
            return down(Y);
        };
        check("softmax_rows_forward f32", shp("R%d C%d", R, C), sm, [&] { return sm(kCpu); }, 1e-7, 1e-5);
        const int B = 64;
        const std::vector<int32_t> heads = {0, 10, 30, 60};
        const int L = heads.back();
        const auto lg = rnd(std::size_t(B) * L, 621, -5.0f, 5.0f, Dtype::FP32);
        std::vector<float> tg(std::size_t(B) * L, 0.0f);
        vkt::Rng r(622);
        for (int b = 0; b < B; ++b)
            for (std::size_t hh = 0; hh + 1 < heads.size(); ++hh)
                tg[std::size_t(b) * L + heads[hh] + r.next() % (heads[hh + 1] - heads[hh])] = 1.0f;
        std::vector<float> outs[2][3];
        const Device devs[2] = {kHip, kVk};
        for (int i = 0; i < 2; ++i) {
            const Device d = devs[i];
            Tensor P, dL, loss = Tensor::zeros_on(d, B, 1), off = i32(heads, int(heads.size()), 1, d);
            brotensor::softmax_xent_fused_batched(up(lg, B, L, Dtype::FP32, d), up(tg, B, L, Dtype::FP32, d), nullptr,
                                                  static_cast<const int*>(off.data), int(heads.size()) - 1, P, dL,
                                                  loss);
            outs[i][0] = down(P);
            outs[i][1] = down(dL);
            outs[i][2] = down(loss);
        }
        check_values("softmax_xent_fused_batched probs", shp("B%d heads3 L%d", B, L), outs[0][0], outs[1][0], nullptr,
                     1e-7, 1e-5);
        check_values("softmax_xent_fused_batched dLogits", shp("B%d heads3 L%d", B, L), outs[0][1], outs[1][1],
                     nullptr, 1e-7, 1e-5);
        check_values("softmax_xent_fused_batched loss", shp("B%d heads3 L%d", B, L), outs[0][2], outs[1][2], nullptr,
                     0, 1e-5);
    }
    {   // sample_logits_into: same Philox stream on both. Qwen3's vocabulary
        // with top-k / top-p; the unfiltered draw at V 4096, because the CUDA /
        // HIP kernel selects the kept set one argmax pass per kept token,
        // O(keep * V) — about 20 s a call at V 151936 with nothing filtered.
        struct S { int N, V; float temp; int top_k; float top_p; const char* name; };
        const S cases[] = {{8, 151936, 0.0f, 0, 1.0f, "greedy"},
                           {8, 151936, 0.8f, 50, 0.9f, "t0.8 k50 p0.9"},
                           {8, 151936, 0.7f, 20, 1.0f, "t0.7 k20"},
                           {8, 4096, 1.0f, 0, 1.0f, "t1 unfiltered"}};
        for (const S& s : cases) {
            const auto lg = rnd(std::size_t(s.N) * s.V, 630 + s.V, -6.0f, 12.0f, Dtype::FP32);
            std::vector<int32_t> got[2];
            const Device devs[2] = {kHip, kVk};
            for (int i = 0; i < 2; ++i) {
                const Device d = devs[i];
                Tensor ctr = i32({17}, 1, 1, d), scratch = Tensor::zeros_on(d, 3 * s.N, s.V);
                Tensor idx = Tensor::zeros_on(d, s.N, 1, Dtype::INT32), L = up(lg, s.N, s.V, Dtype::FP32, d);
                for (int step = 0; step < 4; ++step) {   // the counter advances on device
                    brotensor::sample_logits_into(L, s.temp, s.top_k, s.top_p, 1234, ctr, scratch, idx);
                    std::vector<int32_t> v(s.N);
                    idx.copy_to_host_raw(v.data(), v.size() * 4);
                    got[i].insert(got[i].end(), v.begin(), v.end());
                }
            }
            int mism = 0;
            for (std::size_t j = 0; j < got[0].size(); ++j) mism += got[0][j] != got[1][j];
            // A draw landing within float rounding of a CDF boundary may pick a neighbour.
            check_exact("sample_logits_into", shp("N%d V%d %s x4 steps", s.N, s.V, s.name), mism,
                        int(got[0].size()), s.temp == 0.0f ? 0 : 1);
        }
    }
    {   // randn (Philox + Box-Muller): both GPUs and the CPU share the stream.
        const int R = 512, C = 1024;
        auto run = [&](Device d) {
            Tensor Y = Tensor::zeros_on(d, R, C);
            brotensor::randn(42, 7, Y);
            return down(Y);
        };
        check("randn", shp("R%d C%d key42 ctr7", R, C), run, [&] { return run(kCpu); }, 2e-5, 1e-6);
    }
    {   // DDIM step on an SD latent (4 x 64 x 64).
        const auto xt = rnd(4 * 4096, 640, -3.0f, 3.0f, Dtype::FP16), ep = rnd(4 * 4096, 641, -2.0f, 2.0f, Dtype::FP16);
        auto run = [&](Device d) {
            Tensor Y;
            brotensor::ddim_step(up(xt, 4, 4096, Dtype::FP16, d), up(ep, 4, 4096, Dtype::FP16, d), 0.35f, 0.52f, 0.1f,
                                 Y);
            return down(Y);
        };
        check("ddim_step f16", "4x64x64", run, [&] { return run(kCpu); }, 0, 2 * kF16);
    }
}

// ─── audio ──────────────────────────────────────────────────────────────────

namespace {

std::vector<float> hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = float(0.5 - 0.5 * std::cos(2.0 * M_PI * i / n));
    return w;
}

// A speech-like test signal: a few harmonics with vibrato plus noise, 0.5 peak.
std::vector<float> signal(int n, std::uint64_t seed) {
    std::vector<float> s(n);
    vkt::Rng r(seed);
    for (int i = 0; i < n; ++i) {
        const double t = i / 16000.0;
        const double f0 = 140.0 + 20.0 * std::sin(2 * M_PI * 3.0 * t);
        double v = 0;
        for (int k = 1; k <= 6; ++k) v += std::sin(2 * M_PI * f0 * k * t + k) / k;
        s[i] = float(0.2 * v + 0.05 * r.uniform(-1.0f, 1.0f));
    }
    return s;
}

// Triangular mel-like filterbank (bins, mels): positive, overlapping.
std::vector<float> filterbank(int bins, int mels) {
    std::vector<float> fb(std::size_t(bins) * mels, 0.0f);
    for (int m = 0; m < mels; ++m) {
        const double lo = std::pow(double(bins - 1), double(m) / (mels + 1)),
                     c = std::pow(double(bins - 1), double(m + 1) / (mels + 1)),
                     hi = std::pow(double(bins - 1), double(m + 2) / (mels + 1));
        for (int b = 0; b < bins; ++b) {
            double v = 0;
            if (b >= lo && b <= c && c > lo) v = (b - lo) / (c - lo);
            else if (b > c && b <= hi && hi > c) v = (hi - b) / (hi - c);
            fb[std::size_t(b) * mels + m] = float(v);
        }
    }
    return fb;
}

}  // namespace

void run_audio() {
    std::printf("\n[audio]\n");
    {   // 16 kHz <-> 24 kHz (Whisper / Kokoro / Qwen3-TTS), 64 channels x 1 s.
        const int C = 8, N = 1;
        for (auto [li, lo] : {std::pair{16000, 24000}, std::pair{24000, 16000}}) {
            const auto x = rnd(std::size_t(C) * li, 700 + li, -1.0f, 1.0f, Dtype::FP32);
            for (int mode : {0, 1}) {
                auto run = [&](Device d) {
                    Tensor Y;
                    brotensor::resample1d_forward(up(x, N, C * li, Dtype::FP32, d), N, C, li, lo, mode, Y);
                    return down(Y);
                };
                check(shp("resample1d %s", mode ? "linear" : "nearest"), shp("C%d %d -> %d", C, li, lo), run,
                      [&] { return run(kCpu); }, 1e-6, 1e-6);
            }
        }
    }
    struct F { int n_fft, hop, mels, len; const char* name; };
    const F fronts[] = {{400, 160, 80, 16000 * 30, "whisper 30 s"}, {512, 128, 64, 16000 * 4, "kws 4 s"}};
    for (const F& f : fronts) {
        const auto x = signal(f.len, 710 + f.n_fft);
        const auto win = hann(f.n_fft);
        const int bins = f.n_fft / 2 + 1;
        const auto fb = filterbank(bins, f.mels);
        auto spec = [&](Device d) {
            Tensor S;
            brotensor::stft(up(x, 1, f.len, Dtype::FP32, d), up(win, 1, f.n_fft, Dtype::FP32, d), 1, f.n_fft, f.hop,
                            f.n_fft, true, false, S);
            return S;
        };
        // STFT re/im: CPU FFT vs HIP agree to ~1e-6 of the range; allow 1e-4 abs.
        check("stft", shp("%s n_fft%d hop%d", f.name, f.n_fft, f.hop), [&](Device d) { return down(spec(d)); },
              [&] { return down(spec(kCpu)); }, 1e-4, 0);
        // log-mel: |stft| -> filterbank -> log(max(., 1e-5)).
        auto logmel = [&](Device d) {
            Tensor S = spec(d), M, P, L;
            brotensor::complex_abs(S, M);
            brotensor::matmul(M, up(fb, bins, f.mels, Dtype::FP32, d), P);
            brotensor::clamp(P, 1e-5f, 1e30f);
            brotensor::log_forward(P, L);
            return down(L);
        };
        check("log-mel (stft|abs|fb|log)", shp("%s mels%d", f.name, f.mels), logmel, [&] { return logmel(kCpu); },
              1e-3, 0);
    }
    {   // iSTFT round trip (Kokoro / Vocos style): n_fft 512 hop 128, 2 s.
        const int n_fft = 512, hop = 128, len = 32000;
        const auto x = signal(len, 720);
        const auto win = hann(n_fft);
        Tensor Sc;
        brotensor::stft(up(x, 1, len, Dtype::FP32, kCpu), up(win, 1, n_fft, Dtype::FP32, kCpu), 1, n_fft, hop, n_fft,
                        true, false, Sc);
        const auto sv = Sc.to_host_vector();
        auto run = [&](Device d) {
            Tensor Y;
            brotensor::istft(up(sv, Sc.rows, Sc.cols, Dtype::FP32, d), up(win, 1, n_fft, Dtype::FP32, d), 1, len,
                             n_fft, hop, n_fft, true, false, Y);
            return down(Y);
        };
        check("istft", shp("n_fft%d hop%d len%d", n_fft, hop, len), run, [&] { return run(kCpu); }, 1e-5, 0);
    }
}

}  // namespace vhp
