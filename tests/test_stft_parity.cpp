// CPU↔GPU parity tests for the brosoundml STFT / iSTFT ops.
//
// Ops: stft / stft_backward, istft / istft_backward.
//
// signal is REAL (N, signal_len); window is REAL (1, win_length); the complex
// spectrogram is interleaved-complex (N*frames, 2*bins). The GPU backend
// computes each per-frame transform as a direct DFT — mathematically identical
// to the CPU mixed-radix engine, so a modest tolerance covers the
// float-vs-double accumulation gap. Sizes include a non-power-of-2 n_fft and
// both center modes; backward-op inputs are sourced at the exact stft output
// shape via a forward pass.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 4e-3f;
constexpr float kRtol = 4e-3f;

// Hann window — positive and smooth, so the COLA envelope stays well behaved.
Tensor hann_window(int win_length) {
    Tensor w = Tensor::mat(1, win_length);
    for (int j = 0; j < win_length; ++j) {
        const double a = 2.0 * 3.14159265358979323846 * j / (win_length - 1);
        w.ptr()[j] = static_cast<float>(0.5 - 0.5 * std::cos(a));
    }
    return w;
}

void run_stft(int N, int signal_len, int n_fft, int hop, int win,
              bool center, bool normalized, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor signal = Tensor::mat(N, signal_len);
    fill_random(signal, rng);
    Tensor window = hann_window(win);

    Tensor cpu_spec;
    brotensor::stft(signal, window, N, n_fft, hop, win, center, normalized,
                    cpu_spec);

    Tensor gsig = signal.to(gpu_device()), gwin = window.to(gpu_device());
    Tensor gpu_spec;
    brotensor::stft(gsig, gwin, N, n_fft, hop, win, center, normalized,
                    gpu_spec);

    compare_tensors(cpu_spec, download_to_host(gpu_spec), "stft",
                    kAtol, kRtol);
}

void run_stft_backward(int N, int signal_len, int n_fft, int hop, int win,
                       bool center, bool normalized, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor window = hann_window(win);
    // A random dSpec at the exact stft output shape (sourced via a probe run).
    Tensor probe_sig = Tensor::mat(N, signal_len);
    fill_random(probe_sig, rng);
    Tensor probe;
    brotensor::stft(probe_sig, window, N, n_fft, hop, win, center, normalized,
                    probe);
    Tensor dSpec = Tensor::mat(probe.rows, probe.cols);
    fill_random(dSpec, rng);

    Tensor cpu_dSig;
    brotensor::stft_backward(dSpec, window, N, signal_len, n_fft, hop, win,
                             center, normalized, cpu_dSig);

    Tensor gdSpec = dSpec.to(gpu_device()), gwin = window.to(gpu_device());
    Tensor gpu_dSig;
    brotensor::stft_backward(gdSpec, gwin, N, signal_len, n_fft, hop, win,
                             center, normalized, gpu_dSig);

    compare_tensors(cpu_dSig, download_to_host(gpu_dSig), "stft_backward",
                    kAtol, kRtol);
}

void run_istft(int N, int signal_len, int n_fft, int hop, int win,
               bool center, bool normalized, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor window = hann_window(win);
    // A valid-shaped spectrogram from a forward pass on a random signal.
    Tensor signal = Tensor::mat(N, signal_len);
    fill_random(signal, rng);
    Tensor spec;
    brotensor::stft(signal, window, N, n_fft, hop, win, center, normalized,
                    spec);

    Tensor cpu_out;
    brotensor::istft(spec, window, N, signal_len, n_fft, hop, win, center,
                     normalized, cpu_out);

    Tensor gspec = spec.to(gpu_device()), gwin = window.to(gpu_device());
    Tensor gpu_out;
    brotensor::istft(gspec, gwin, N, signal_len, n_fft, hop, win, center,
                     normalized, gpu_out);

    compare_tensors(cpu_out, download_to_host(gpu_out), "istft", kAtol, kRtol);
}

void run_istft_backward(int N, int signal_len, int n_fft, int hop, int win,
                        bool center, bool normalized, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor window = hann_window(win);
    Tensor dSignal = Tensor::mat(N, signal_len);
    fill_random(dSignal, rng);

    Tensor cpu_dSpec;
    brotensor::istft_backward(dSignal, window, N, signal_len, n_fft, hop, win,
                              center, normalized, cpu_dSpec);

    Tensor gdSig = dSignal.to(gpu_device()), gwin = window.to(gpu_device());
    Tensor gpu_dSpec;
    brotensor::istft_backward(gdSig, gwin, N, signal_len, n_fft, hop, win,
                              center, normalized, gpu_dSpec);

    compare_tensors(cpu_dSpec, download_to_host(gpu_dSpec), "istft_backward",
                    kAtol, kRtol);
}

} // namespace

// ─── stft ──────────────────────────────────────────────────────────────────
BT_PARITY_TEST(stft_8fft_nocenter)  { run_stft(2, 64, 8, 2, 8, false, false, 0x501ull); }
BT_PARITY_TEST(stft_16fft_center)   { run_stft(2, 64, 16, 4, 16, true, false, 0x502ull); }
BT_PARITY_TEST(stft_16fft_norm)     { run_stft(3, 80, 16, 4, 12, true, true, 0x503ull); }
BT_PARITY_TEST(stft_400fft_whisper) { run_stft(1, 1600, 400, 160, 400, true, false, 0x504ull); }

// ─── stft_backward ─────────────────────────────────────────────────────────
BT_PARITY_TEST(stft_bw_8fft_nocenter) { run_stft_backward(2, 64, 8, 2, 8, false, false, 0x511ull); }
BT_PARITY_TEST(stft_bw_16fft_center)  { run_stft_backward(2, 64, 16, 4, 16, true, false, 0x512ull); }
BT_PARITY_TEST(stft_bw_16fft_norm)    { run_stft_backward(3, 80, 16, 4, 12, true, true, 0x513ull); }

// ─── istft ─────────────────────────────────────────────────────────────────
BT_PARITY_TEST(istft_8fft_nocenter) { run_istft(2, 64, 8, 2, 8, false, false, 0x521ull); }
BT_PARITY_TEST(istft_16fft_center)  { run_istft(2, 64, 16, 4, 16, true, false, 0x522ull); }
BT_PARITY_TEST(istft_16fft_norm)    { run_istft(3, 80, 16, 4, 12, true, true, 0x523ull); }

// ─── istft_backward ────────────────────────────────────────────────────────
BT_PARITY_TEST(istft_bw_8fft_nocenter) { run_istft_backward(2, 64, 8, 2, 8, false, false, 0x531ull); }
BT_PARITY_TEST(istft_bw_16fft_center)  { run_istft_backward(2, 64, 16, 4, 16, true, false, 0x532ull); }
BT_PARITY_TEST(istft_bw_16fft_norm)    { run_istft_backward(3, 80, 16, 4, 12, true, true, 0x533ull); }

int main() { return run_all("stft cpu/gpu parity"); }
