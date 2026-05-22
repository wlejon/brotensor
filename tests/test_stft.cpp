// Standalone CPU coverage for the brosoundml STFT / iSTFT core (CHUNK 2).
//
// Verifies:
//   * stft frame / bin shape derivation for power-of-2 and non-power-of-2
//     n_fft (including Whisper's 400) and both center modes.
//   * stft frame values against a direct windowed-DFT reference.
//   * istft(stft(x)) == x round trip under a COLA-satisfying window+hop
//     (Hann, hop = n_fft/4), for center=false and center=true.
//   * The normalized flag is amplitude-preserving through the round trip.
//   * Finite-difference gradient checks for stft_backward and istft_backward.
//
// CPU-resident, FP32. Plain executable; exits non-zero on any failure.

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;

static int g_failures = 0;

static bool near_(double a, double b, double abs_eps, double rel_eps) {
    const double d = std::fabs(a - b);
    if (d <= abs_eps) return true;
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    return d <= rel_eps * m;
}

#define EXPECT_NEAR(actual, expected, abs_eps, rel_eps, ctx)                    \
    do {                                                                       \
        const double _a = (actual);                                            \
        const double _e = (expected);                                          \
        if (!near_(_a, _e, (abs_eps), (rel_eps))) {                            \
            std::printf("  FAIL  %s:%d  [%s]  actual=%.9g expected=%.9g\n",     \
                        __FILE__, __LINE__, (ctx), _a, _e);                     \
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, ctx)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL  %s:%d  [%s]  condition false: %s\n",          \
                        __FILE__, __LINE__, (ctx), #cond);                      \
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

static Tensor cpu_zeros(int r, int c = 1) {
    return Tensor::zeros_on(Device::CPU, r, c);
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float next() {  // uniform in [-1, 1)
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) /
                                  static_cast<double>(1ULL << 53)) * 2.0f - 1.0f;
    }
};

static const double kPi = 3.14159265358979323846;

// Periodic Hann window of length n (the COLA-friendly torch default).
static Tensor hann(int n) {
    Tensor w = cpu_zeros(1, n);
    for (int i = 0; i < n; ++i) {
        w.host_f32_mut()[i] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * i / static_cast<double>(n)));
    }
    return w;
}

// Reference numpy-'reflect' index map (edge sample not repeated).
static int ref_reflect(int q, int L) {
    if (L == 1) return 0;
    const int period = 2 * (L - 1);
    int m = q % period;
    if (m < 0) m += period;
    return (m < L) ? m : period - m;
}

// ─── frame-count helper (mirrors the op's StftGeom) ────────────────────────
static int frames_of(int L, int n_fft, int hop, bool center) {
    return center ? 1 + L / hop : 1 + (L - n_fft) / hop;
}

// ─── stft shape + value correctness ────────────────────────────────────────
static void test_stft_shape_and_values(int n_fft, int hop, int win_len,
                                       bool center) {
    char ctx[96];
    const int L = 4 * n_fft + 7;   // arbitrary signal length
    const int N = 2;
    const int bins = n_fft / 2 + 1;
    const int frames = frames_of(L, n_fft, hop, center);

    Tensor sig = cpu_zeros(N, L);
    Rng rng(0xABC0 + n_fft * 31 + (center ? 1 : 0));
    for (int i = 0; i < sig.size(); ++i) sig.host_f32_mut()[i] = rng.next();
    Tensor win = hann(win_len);

    Tensor spec = cpu_zeros(1, 1);  // deliberately mis-shaped — must resize
    brotensor::stft(sig, win, N, n_fft, hop, win_len, center,
                    /*normalized=*/false, spec);

    std::snprintf(ctx, sizeof(ctx), "stft rows nfft=%d center=%d", n_fft,
                  center ? 1 : 0);
    EXPECT_TRUE(spec.rows == N * frames, ctx);
    std::snprintf(ctx, sizeof(ctx), "stft cols nfft=%d center=%d", n_fft,
                  center ? 1 : 0);
    EXPECT_TRUE(spec.cols == 2 * bins, ctx);

    // Reference: windowed DFT of one frame of the first signal.
    const int pad_lo = (n_fft - win_len) / 2;
    for (int f = 0; f < frames; ++f) {
        std::vector<double> buf(n_fft, 0.0);
        for (int j = 0; j < win_len; ++j) {
            const int i = pad_lo + j;
            int p = f * hop + i;
            int s;
            if (center) {
                s = ref_reflect(p - n_fft / 2, L);
            } else {
                s = p;
            }
            buf[i] = static_cast<double>(sig.host_f32()[s]) *
                     static_cast<double>(win.host_f32()[j]);
        }
        for (int k = 0; k < bins; ++k) {
            double re = 0.0, im = 0.0;
            for (int n = 0; n < n_fft; ++n) {
                const double a = -2.0 * kPi * k * n / n_fft;
                re += buf[n] * std::cos(a);
                im += buf[n] * std::sin(a);
            }
            const float* row = spec.host_f32() +
                               static_cast<std::size_t>(f) * (2 * bins);
            std::snprintf(ctx, sizeof(ctx), "stft nfft=%d f=%d k=%d re",
                          n_fft, f, k);
            EXPECT_NEAR(row[2 * k], re, 1e-2, 3e-3, ctx);
            std::snprintf(ctx, sizeof(ctx), "stft nfft=%d f=%d k=%d im",
                          n_fft, f, k);
            EXPECT_NEAR(row[2 * k + 1], im, 1e-2, 3e-3, ctx);
        }
    }
}

// ─── istft(stft(x)) round trip under a COLA window+hop ─────────────────────
static void test_roundtrip(int n_fft, bool center, bool normalized) {
    char ctx[96];
    const int hop = n_fft / 4;          // Hann + 75% overlap satisfies COLA
    const int win_len = n_fft;
    const int L = 6 * n_fft + 5;
    const int N = 2;

    Tensor sig = cpu_zeros(N, L);
    Rng rng(0x5550 + n_fft + (center ? 7 : 0) + (normalized ? 13 : 0));
    for (int i = 0; i < sig.size(); ++i) sig.host_f32_mut()[i] = rng.next();
    Tensor win = hann(win_len);

    Tensor spec = cpu_zeros(1, 1);
    brotensor::stft(sig, win, N, n_fft, hop, win_len, center, normalized,
                    spec);
    Tensor rec = cpu_zeros(1, 1);
    brotensor::istft(spec, win, N, L, n_fft, hop, win_len, center, normalized,
                     rec);

    EXPECT_TRUE(rec.rows == N && rec.cols == L, "roundtrip shape");

    // Interior samples (away from the no-coverage edges) must reconstruct.
    // center=true covers the whole signal; center=false leaves a margin.
    const int margin = center ? 0 : n_fft;
    int checked = 0;
    for (int b = 0; b < N; ++b) {
        for (int n = margin; n < L - margin; ++n) {
            const int idx = b * L + n;
            std::snprintf(ctx, sizeof(ctx),
                          "roundtrip nfft=%d center=%d norm=%d b=%d n=%d",
                          n_fft, center ? 1 : 0, normalized ? 1 : 0, b, n);
            EXPECT_NEAR(rec.host_f32()[idx], sig.host_f32()[idx],
                        2e-3, 3e-3, ctx);
            ++checked;
        }
    }
    EXPECT_TRUE(checked > 0, "roundtrip checked some samples");
}

// ─── generic finite-difference gradient checker ────────────────────────────
template <typename Fn>
static void fd_check(const std::string& name, float* in, int n,
                     const std::vector<float>& analytic, Fn&& loss_at,
                     float h = 1e-3f, float abs_eps = 3e-2f,
                     float rel_eps = 3e-2f) {
    for (int i = 0; i < n; ++i) {
        const float saved = in[i];
        in[i] = saved + h;
        const float lp = loss_at();
        in[i] = saved - h;
        const float lm = loss_at();
        in[i] = saved;
        const float num = (lp - lm) / (2.0f * h);
        if (!near_(analytic[i], num, abs_eps, rel_eps)) {
            std::printf("  FAIL  fd-grad %s  i=%d  analytic=%.6g numeric=%.6g\n",
                        name.c_str(), i, analytic[i], num);
            ++g_failures;
        }
    }
}

// stft_backward: loss = sum(w .* stft(x)), check grad w.r.t. the signal x.
static void test_stft_backward(int n_fft, bool center, bool normalized) {
    char name[80];
    std::snprintf(name, sizeof(name), "stft_backward nfft=%d center=%d norm=%d",
                  n_fft, center ? 1 : 0, normalized ? 1 : 0);
    std::printf("%s\n", name);
    const int hop = n_fft / 2;
    const int win_len = n_fft;
    const int L = 3 * n_fft + 3;
    const int N = 2;
    const int bins = n_fft / 2 + 1;
    const int frames = frames_of(L, n_fft, hop, center);

    Tensor sig = cpu_zeros(N, L);
    Tensor w = cpu_zeros(N * frames, 2 * bins);  // upstream weights
    Rng rng(0x7700 + n_fft + (center ? 3 : 0) + (normalized ? 9 : 0));
    for (int i = 0; i < sig.size(); ++i) sig.host_f32_mut()[i] = rng.next();
    for (int i = 0; i < w.size(); ++i) w.host_f32_mut()[i] = rng.next();
    Tensor win = hann(win_len);

    Tensor spec = cpu_zeros(N * frames, 2 * bins);
    auto loss = [&]() {
        brotensor::stft(sig, win, N, n_fft, hop, win_len, center, normalized,
                        spec);
        float s = 0.0f;
        for (int i = 0; i < spec.size(); ++i)
            s += w.host_f32()[i] * spec.host_f32()[i];
        return s;
    };
    loss();
    Tensor dSig = cpu_zeros(N, L);
    brotensor::stft_backward(w, win, N, L, n_fft, hop, win_len, center,
                             normalized, dSig);
    std::vector<float> g(dSig.host_f32(), dSig.host_f32() + dSig.size());
    fd_check(name, sig.host_f32_mut(), sig.size(), g, loss);
}

// istft_backward: loss = sum(w .* istft(x)), check grad w.r.t. spectrum x.
static void test_istft_backward(int n_fft, bool center, bool normalized) {
    char name[80];
    std::snprintf(name, sizeof(name), "istft_backward nfft=%d center=%d norm=%d",
                  n_fft, center ? 1 : 0, normalized ? 1 : 0);
    std::printf("%s\n", name);
    const int hop = n_fft / 4;
    const int win_len = n_fft;
    const int L = 3 * n_fft + 3;
    const int N = 2;
    const int bins = n_fft / 2 + 1;
    const int frames = frames_of(L, n_fft, hop, center);

    Tensor spec = cpu_zeros(N * frames, 2 * bins);
    Tensor w = cpu_zeros(N, L);   // upstream weights
    Rng rng(0x8800 + n_fft + (center ? 5 : 0) + (normalized ? 11 : 0));
    for (int i = 0; i < spec.size(); ++i) spec.host_f32_mut()[i] = rng.next();
    for (int i = 0; i < w.size(); ++i) w.host_f32_mut()[i] = rng.next();
    Tensor win = hann(win_len);

    Tensor rec = cpu_zeros(N, L);
    auto loss = [&]() {
        brotensor::istft(spec, win, N, L, n_fft, hop, win_len, center,
                         normalized, rec);
        float s = 0.0f;
        for (int i = 0; i < rec.size(); ++i)
            s += w.host_f32()[i] * rec.host_f32()[i];
        return s;
    };
    loss();
    Tensor dSpec = cpu_zeros(N * frames, 2 * bins);
    brotensor::istft_backward(w, win, N, L, n_fft, hop, win_len, center,
                              normalized, dSpec);
    std::vector<float> g(dSpec.host_f32(), dSpec.host_f32() + dSpec.size());
    fd_check(name, spec.host_f32_mut(), spec.size(), g, loss);
}

int main() {
    brotensor::init();
    std::printf("test_stft\n");

    // Shape + value correctness: power-of-2, smooth non-pow2 (Whisper 400),
    // both center modes; win_length both == n_fft and < n_fft.
    for (bool center : {false, true}) {
        test_stft_shape_and_values(8, 2, 8, center);
        test_stft_shape_and_values(16, 4, 16, center);
        test_stft_shape_and_values(16, 4, 12, center);   // win < n_fft
        test_stft_shape_and_values(400, 100, 400, center);  // Whisper
    }

    // istft(stft(x)) round trips under a COLA Hann + hop = n_fft/4.
    for (bool center : {false, true}) {
        for (bool norm : {false, true}) {
            test_roundtrip(16, center, norm);
            test_roundtrip(64, center, norm);
            test_roundtrip(400, center, norm);
        }
    }

    // Finite-difference gradient checks.
    for (bool center : {false, true}) {
        for (bool norm : {false, true}) {
            test_stft_backward(8, center, norm);
            test_stft_backward(16, center, norm);
            test_istft_backward(8, center, norm);
            test_istft_backward(16, center, norm);
        }
    }

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("\nAll STFT / iSTFT op checks passed.\n");
    return 0;
}
