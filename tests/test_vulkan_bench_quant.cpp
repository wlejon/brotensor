// `brotensor_test_vulkan --bench-quant`: the quantised linears per weight
// format (INT8 W8A16, GGUF Q8_0 / Q4_K / Q6_K) on LLM shapes. Not part of
// ctest.
//
//   decode   linear_forward_batched_<fmt> at B = 1, 2, 4, 8 activation rows:
//            GB/s of weight bytes read (the stored, quantised size; X and Y
//            are a few KiB and stay in cache);
//   prefill  the same op at M = 512 (and 4096 for INT8): TF/s, 2 M N K.
//
// Timing: wall clock around back-to-back launches with one device sync,
// median of 7 batches after warm-up (test_vulkan_bench.cpp's method).
// BROTENSOR_VK_BENCH_FMT=int8|q8_0|q4k|q6k runs one format,
// BROTENSOR_VK_BENCH_PART=decode|prefill one table,
// BROTENSOR_VK_BENCH_SHAPE=<tag> one shape (BROTENSOR_VK_BENCH_MNK=m,n,k a
// custom prefill shape); BROTENSOR_VK_QGEMV=lpr,unr,sg forces the GEMV
// configuration and BROTENSOR_VK_GEMM_CFG the prefill tile.
//
// `--bench-audio` (run_audio_bench) times the main audio ops the same way:
// GB/s of minimum traffic for the bandwidth-bound ones, GFLOP/s of the dense
// equivalent for the transposed convolution and the STFT pair.

#include "test_vulkan_common.h"

#include <brotensor/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace vkt {

namespace {

template <class F>
double median_ms(Device dev, F&& launch_once) {
    for (int i = 0; i < 3; ++i) launch_once();
    brotensor::sync(dev);
    auto t0 = std::chrono::steady_clock::now();
    launch_once();
    brotensor::sync(dev);
    const double one = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int reps = std::max(3, std::min(200, static_cast<int>(150.0 / std::max(one, 1e-3))));
    std::vector<double> t;
    for (int b = 0; b < 7; ++b) {
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) launch_once();
        brotensor::sync(dev);
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

bool env_is(const char* var, const char* v) {
    const char* e = std::getenv(var);
    return !e || std::string(e) == v;
}

struct Shape { int n, k; const char* tag; };
const Shape kShapes[] = {
    {3072, 1024, "0.6B-gate/up"},
    {1024, 3072, "0.6B-down"},
    {151936, 1024, "0.6B-lm_head"},
    {4096, 4096, "8B-q/o"},
    {12288, 4096, "8B-gate/up"},
    {4096, 12288, "8B-down"},
};

struct Fmt { Dtype dt; const char* name; };
const Fmt kFmts[] = {
    {Dtype::INT8, "int8"}, {Dtype::Q8_0, "q8_0"}, {Dtype::Q4_K, "q4k"}, {Dtype::Q6_K, "q6k"},
};

double weight_bytes(Dtype dt, int n, int k) {
    if (dt == Dtype::INT8) return double(n) * k;
    return double(n) * (k / brotensor::dtype_block_size(dt)) * brotensor::dtype_block_bytes(dt);
}

// Random weight bytes: any bit pattern is a valid block except for FP16 d /
// dmin fields that decode to inf / NaN, so those are overwritten with small
// finite values.
struct Problem {
    Tensor W, S, X, Y;
    Problem(Device dev, Dtype dt, int n, int k, int B) {
        Rng r(n * 31 + k);
        const std::size_t bytes = static_cast<std::size_t>(weight_bytes(dt, n, k));
        std::vector<std::uint8_t> w(bytes);
        for (auto& b : w) b = static_cast<std::uint8_t>(r.next());
        const int bb = dt == Dtype::INT8 ? 0 : brotensor::dtype_block_bytes(dt);
        const std::uint16_t d = brotensor::fp32_to_fp16_bits(1e-3f);
        for (std::size_t o = 0; bb && o < bytes; o += bb) {
            const std::size_t at = dt == Dtype::Q6_K ? o + 208 : o;
            std::memcpy(&w[at], &d, 2);
            if (dt == Dtype::Q4_K) std::memcpy(&w[o + 2], &d, 2);
        }
        W = Tensor::from_raw_bytes_on(dev, w.data(), n, k, dt, bytes);
        if (dt == Dtype::INT8) S = upload(random_values(n, 2, 1e-3f, 2e-3f, Dtype::FP32), n, 1, Dtype::FP32, dev);
        X = upload(random_values(std::size_t(B) * k, 3, -1, 1, Dtype::FP16), B, k, Dtype::FP16, dev);
    }
    void run(Dtype dt) {
        if (dt == Dtype::INT8) brotensor::linear_forward_batched_int8w_fp16(W, S, nullptr, X, Y);
        else if (dt == Dtype::Q8_0) brotensor::linear_forward_batched_q8_0_fp16(W, nullptr, X, Y);
        else if (dt == Dtype::Q4_K) brotensor::linear_forward_batched_q4k_fp16(W, nullptr, X, Y);
        else brotensor::linear_forward_batched_q6k_fp16(W, nullptr, X, Y);
    }
};

// ms on Vulkan for one configuration (0 when the op throws).
double measure(Dtype dt, int n, int k, int B) {
    try {
        Problem p(vk(), dt, n, k, B);
        return median_ms(vk(), [&] { p.run(dt); });
    } catch (const std::exception& e) {
        std::printf("    (vulkan: %s)\n", e.what());
        return 0;
    }
}

void bench_decode(const Fmt& f) {
    std::printf("\n%s decode: linear_forward_batched, GB/s of weight bytes (ms)\n", f.name);
    std::printf("%-16s %3s %20s\n", "shape", "B", "vulkan");
    for (const Shape& s : kShapes) {
        if (!env_is("BROTENSOR_VK_BENCH_SHAPE", s.tag)) continue;
        for (int B : {1, 2, 4, 8}) {
            const double ms = measure(f.dt, s.n, s.k, B);
            const double gb = weight_bytes(f.dt, s.n, s.k) / 1e6;
            char a[32];
            std::snprintf(a, sizeof a, "%6.1f (%.4f)", ms > 0 ? gb / ms : 0.0, ms);
            std::printf("%-16s %3d %20s\n", s.tag, B, a);
        }
    }
}

void bench_prefill(const Fmt& f) {
    std::printf("\n%s prefill: linear_forward_batched, TF/s (ms)\n", f.name);
    std::printf("%-16s %5s %20s\n", "shape", "M", "vulkan");
    struct P { int m, n, k; const char* tag; };
    const P ps[] = {
        {512, 4096, 4096, "8B-q/o"},
        {512, 12288, 4096, "8B-gate/up"},
        {512, 4096, 12288, "8B-down"},
        {4096, 3072, 3072, "DiT-3072"},
        {4096, 4096, 4096, "square4k"},
    };
    std::vector<P> list(std::begin(ps), std::end(ps));
    if (const char* e = std::getenv("BROTENSOR_VK_BENCH_MNK")) {   // one custom shape: m,n,k
        P c{0, 0, 0, "custom"};
        std::sscanf(e, "%d,%d,%d", &c.m, &c.n, &c.k);
        list.assign(1, c);
    }
    for (const P& p : list) {
        if (!env_is("BROTENSOR_VK_BENCH_SHAPE", p.tag) && std::string(p.tag) != "custom") continue;
        if (p.m > 512 && f.dt != Dtype::INT8) continue;   // GGUF prefill: the M = 512 shapes only
        const double ms = measure(f.dt, p.n, p.k, p.m);
        const double tf = 2.0 * p.m * p.n * p.k / 1e9;
        char a[32];
        std::snprintf(a, sizeof a, "%6.2f (%.3f)", ms > 0 ? tf / ms : 0.0, ms);
        std::printf("%-16s %5d %20s\n", p.tag, p.m, a);
    }
}

// ─── audio ──────────────────────────────────────────────────────────────────

// Bandwidth-bound audio ops in GB/s of minimum traffic (each input read once,
// each output written once) and the transforms in ms.
void bench_audio() {
    std::printf("\naudio ops (ms / GB/s of minimum traffic; transforms ms / GFLOP/s)\n");
    std::printf("%-44s %18s\n", "op", "vulkan");
    struct Case { const char* tag; Dtype dt; double bytes, flop; };
    const int C = 512, L = 24000;                 // a vocoder stage: 512 channels, 1 s at 24 kHz
    const int sig = 480000, nfft = 400, hop = 160, frames = 1 + sig / hop, bins = nfft / 2 + 1;   // Whisper, 30 s
    const Case cases[] = {
        {"snake 512 ch x 24000 f32", Dtype::FP32, 2.0 * C * L * 4, 0},
        {"snake 512 ch x 24000 f16", Dtype::FP16, 2.0 * C * L * 2, 0},
        {"pad1d reflect 512 ch x 24000 (+3 / +3) f32", Dtype::FP32, 2.0 * C * L * 4, 0},
        {"resample1d linear 16k -> 24k, 64 ch x 10 s", Dtype::FP32, 64.0 * (160000 + 240000) * 4, 0},
        {"conv_transpose1d 512 -> 256, k16 s8, L 1000 f32", Dtype::FP32, 0, 2.0 * 512 * 256 * 16 * 1000},
        {"conv_transpose1d 512 -> 256, k16 s8, L 1000 f16", Dtype::FP16, 0, 2.0 * 512 * 256 * 16 * 1000},
        {"stft 400 / 160, 30 s (Whisper)", Dtype::FP32, 0, 2.0 * frames * nfft * 2 * bins},
        {"istft 400 / 160, 30 s", Dtype::FP32, 0, 2.0 * frames * nfft * 2 * bins},
        {"complex_abs 3001 x 201", Dtype::FP32, 3001.0 * 201 * 12, 0},
    };
    for (const Case& c : cases) {
        if (!env_is("BROTENSOR_VK_BENCH_SHAPE", c.tag)) continue;
        double ms = 0;
        {
            const Device dev = vk();
            try {
                const std::string t = c.tag;
                Tensor X, Y, A, W, Z;
                if (t.rfind("snake", 0) == 0 || t.rfind("pad1d", 0) == 0) {
                    X = upload(random_values(std::size_t(C) * L, 1, -1, 1, c.dt), 1, C * L, c.dt, dev);
                    A = upload(random_values(C, 2, 0.5f, 1.5f, Dtype::FP32), C, 1, Dtype::FP32, dev);
                    if (t[0] == 's') ms = median_ms(dev, [&] { brotensor::snake_forward(X, A, nullptr, 1, C, L, Y); });
                    else ms = median_ms(dev, [&] { brotensor::pad1d_forward(X, 1, C, L, 3, 3, 1, Y); });
                } else if (t.rfind("resample", 0) == 0) {
                    X = upload(random_values(64ull * 160000, 3, -1, 1, c.dt), 1, 64 * 160000, c.dt, dev);
                    ms = median_ms(dev, [&] { brotensor::resample1d_forward(X, 1, 64, 160000, 240000, 1, Y); });
                } else if (t.rfind("conv_transpose1d", 0) == 0) {
                    X = upload(random_values(512ull * 1000, 4, -1, 1, c.dt), 1, 512 * 1000, c.dt, dev);
                    W = upload(random_values(512ull * 256 * 16, 5, -0.05f, 0.05f, c.dt), 512, 256 * 16, c.dt, dev);
                    ms = median_ms(dev, [&] {
                        brotensor::conv_transpose1d_forward(X, W, nullptr, 1, 512, 1000, 256, 16, 8, 4, 0, 1, 1, Y);
                    });
                } else if (t.rfind("stft", 0) == 0 || t.rfind("istft", 0) == 0) {
                    std::vector<float> wv(nfft);
                    for (int i = 0; i < nfft; ++i) wv[i] = 0.5f - 0.5f * std::cos(6.283185307179586 * i / nfft);
                    W = upload(wv, 1, nfft, Dtype::FP32, dev);
                    X = upload(random_values(sig, 6, -1, 1, Dtype::FP32), 1, sig, Dtype::FP32, dev);
                    brotensor::stft(X, W, 1, nfft, hop, nfft, true, false, Z);
                    if (t[0] == 's') ms = median_ms(dev, [&] { brotensor::stft(X, W, 1, nfft, hop, nfft, true, false, Y); });
                    else ms = median_ms(dev, [&] { brotensor::istft(Z, W, 1, sig, nfft, hop, nfft, true, false, Y); });
                } else {
                    X = upload(random_values(3001ull * 402, 7, -1, 1, Dtype::FP32), 3001, 402, Dtype::FP32, dev);
                    ms = median_ms(dev, [&] { brotensor::complex_abs(X, Y); });
                }
            } catch (const std::exception& e) {
                std::printf("    (vulkan: %s)\n", e.what());
                ms = 0;
            }
        }
        auto fmt = [&](double m, char* buf, std::size_t n) {
            if (m <= 0) { std::snprintf(buf, n, "-"); return; }
            if (c.flop > 0) std::snprintf(buf, n, "%.3f / %6.0f", m, c.flop / (m * 1e6));
            else std::snprintf(buf, n, "%.3f / %5.0f", m, c.bytes / (m * 1e6));
        };
        char a[40];
        fmt(ms, a, sizeof a);
        std::printf("%-44s %18s\n", c.tag, a);
    }
}

}  // namespace

void run_audio_bench() {
    const auto info = brotensor::vulkan::device_info(vk());
    std::printf("audio benchmark on %s (%s)\n", info.name.c_str(), info.driver.c_str());
    bench_audio();
}

void run_quant_bench() {
    const auto info = brotensor::vulkan::device_info(vk());
    std::printf("quantised-linear benchmark on %s (%s)\n", info.name.c_str(), info.driver.c_str());
    for (const Fmt& f : kFmts) {
        if (!env_is("BROTENSOR_VK_BENCH_FMT", f.name)) continue;
        if (env_is("BROTENSOR_VK_BENCH_PART", "decode")) bench_decode(f);
        if (env_is("BROTENSOR_VK_BENCH_PART", "prefill")) bench_prefill(f);
    }
}

}  // namespace vkt
