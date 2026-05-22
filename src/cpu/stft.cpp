// ─── CPU STFT / iSTFT (brosoundml CHUNK 2) ─────────────────────────────────
//
// Short-time Fourier transform and its inverse, plus their adjoints. CPU
// backend, FP32-only. No external libraries — the per-frame DFT reuses the
// hand-rolled mixed-radix + Bluestein engine from detail/cpu/fft_core.h
// (so n_fft = 400 and prime sizes both work, exactly as in fft.cpp).
//
// Ops implemented here:
//   stft / stft_backward     real signal  <-> complex spectrogram
//   istft / istft_backward   complex spectrogram <-> real signal (COLA OLA)
//
// ── Layout (see the doc comments in ops.h for the full contract) ────────────
// signal:  REAL (N, signal_len)               — N batched signals, one / row.
// window:  REAL (1, win_length)                — caller-supplied.
// spec:    interleaved-complex (N*frames, 2*bins), bins = n_fft/2+1. Each
//          frame is a row; the N signals' frame blocks are stacked in order.
//
// ── Frame model ─────────────────────────────────────────────────────────────
// Frame f of signal b takes n_fft samples starting at padded position
// f*hop_length, multiplies the central win_length of them by `window`, and
// rfft's the n_fft buffer. The window sits centred in the n_fft buffer
// (pad = (n_fft-win_length)/2 zeros each side). When center == true the
// signal is reflect-padded by n_fft/2 each side first; otherwise the raw
// signal is used. `padded_index` below maps a padded position back to a raw
// signal index (reflecting at the borders when center == true) so the forward
// op and its adjoint share one indexing rule and stay exact transposes.
//
// ── Normalisation ───────────────────────────────────────────────────────────
// rfft uses the "backward" convention (forward unscaled). normalized == true
// multiplies the forward spectrum by 1/sqrt(n_fft) (istft divides by it).
//
// ── Gradient design ─────────────────────────────────────────────────────────
// stft and istft are linear but NOT mutual adjoints (window + COLA). Each
// backward op is the exact transpose of its own forward linear map — see the
// ops.h header note. They are the minimal correct set for the
// multi-resolution STFT loss.

#include <brotensor/detail/cpu/fft_core.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

using fftcore::Cd;
using fftcore::dft_1d;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void require_fp32_host(const char* op, const ::brotensor::Tensor& t,
                       const char* name) {
    if (t.device != ::brotensor::Device::CPU) {
        fail(op, std::string(name) + " must be a CPU tensor");
    }
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (CPU is FP32-only)");
    }
}

// Common parameter validation + derived sizes for all four ops.
struct StftGeom {
    int bins = 0;       // n_fft/2 + 1
    int frames = 0;     // frames per signal
    int padded_len = 0; // signal length the frame loop indexes into
    int pad_lo = 0;     // (n_fft - win_length) / 2 — window offset in buffer
};

StftGeom check_geom(const char* op, int N, int signal_len, int n_fft,
                    int hop_length, int win_length, bool center) {
    if (N < 0) fail(op, "N must be >= 0");
    if (n_fft < 1) fail(op, "n_fft must be >= 1");
    if (hop_length < 1) fail(op, "hop_length must be >= 1");
    if (win_length < 1 || win_length > n_fft) {
        fail(op, "win_length must satisfy 1 <= win_length <= n_fft");
    }
    if (signal_len < 1) fail(op, "signal_len must be >= 1");

    StftGeom g;
    g.bins = n_fft / 2 + 1;
    g.pad_lo = (n_fft - win_length) / 2;

    if (center) {
        // Reflect padding by n_fft/2 each side. numpy/torch 'reflect' mode
        // needs at least 2 samples (the reflected index must stay in range);
        // require enough signal to fill the n_fft/2 pad.
        if (signal_len < n_fft / 2 + 1) {
            fail(op, "center=true needs signal_len >= n_fft/2 + 1");
        }
        g.padded_len = signal_len + n_fft;
        g.frames = 1 + signal_len / hop_length;
    } else {
        if (signal_len < n_fft) {
            fail(op, "center=false needs signal_len >= n_fft");
        }
        g.padded_len = signal_len;
        g.frames = 1 + (signal_len - n_fft) / hop_length;
    }
    return g;
}

// Map a padded position p in [0, padded_len) to a raw signal index in
// [0, signal_len). center == false is the identity; center == true reflects
// at the borders (numpy 'reflect': edge sample not repeated).
//
// Reflection over [0, L-1] with period 2*(L-1): fold q into that range.
inline int reflect_index(int q, int L) {
    if (L == 1) return 0;
    const int period = 2 * (L - 1);
    int m = q % period;
    if (m < 0) m += period;
    return (m < L) ? m : period - m;
}

inline int padded_index(int p, int signal_len, int n_fft, bool center) {
    if (!center) return p;
    return reflect_index(p - n_fft / 2, signal_len);
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  stft — real signal -> complex spectrogram
// ════════════════════════════════════════════════════════════════════════════
void stft(const ::brotensor::Tensor& signal, const ::brotensor::Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, ::brotensor::Tensor& spec) {
    require_fp32_host("stft", signal, "signal");
    require_fp32_host("stft", window, "window");
    if (signal.rows != N) {
        fail("stft", "signal.rows must equal N");
    }
    const int signal_len = signal.cols;
    if (window.rows != 1 || window.cols != win_length) {
        fail("stft", "window must be a (1, win_length) tensor");
    }
    const StftGeom g = check_geom("stft", N, signal_len, n_fft, hop_length,
                                  win_length, center);
    const int out_rows = N * g.frames;
    const int out_cols = 2 * g.bins;
    if (spec.rows != out_rows || spec.cols != out_cols) {
        spec.resize(out_rows, out_cols);
    }
    if (out_rows == 0) return;

    const float* sig = signal.host_f32();
    const float* win = window.host_f32();
    float* sp = spec.host_f32_mut();
    const double norm = normalized
                            ? 1.0 / std::sqrt(static_cast<double>(n_fft))
                            : 1.0;

    std::vector<Cd> buf(static_cast<std::size_t>(n_fft)), out;
    for (int b = 0; b < N; ++b) {
        const float* srow = sig + static_cast<std::size_t>(b) * signal_len;
        for (int f = 0; f < g.frames; ++f) {
            // Build the windowed n_fft frame buffer (real, im = 0).
            for (int i = 0; i < n_fft; ++i) buf[static_cast<std::size_t>(i)] = Cd{};
            const int base = f * hop_length;  // padded-position start
            for (int j = 0; j < win_length; ++j) {
                const int i = g.pad_lo + j;
                const int p = base + i;
                const int s = padded_index(p, signal_len, n_fft, center);
                buf[static_cast<std::size_t>(i)] =
                    {static_cast<double>(srow[s]) *
                         static_cast<double>(win[j]),
                     0.0};
            }
            dft_1d(buf, out, -1);  // unscaled forward DFT
            float* dst = sp + static_cast<std::size_t>(
                                  static_cast<std::size_t>(b) * g.frames + f) *
                                  out_cols;
            for (int k = 0; k < g.bins; ++k) {
                dst[2 * k] = static_cast<float>(
                    out[static_cast<std::size_t>(k)].re * norm);
                dst[2 * k + 1] = static_cast<float>(
                    out[static_cast<std::size_t>(k)].im * norm);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  stft_backward — adjoint of stft
// ════════════════════════════════════════════════════════════════════════════
//
// stft is the linear map  spec = R * W * P * signal  where P scatters the
// signal into frame buffers (with reflect padding folded in), W multiplies by
// the window, and R is the truncated forward DFT. Its adjoint applied to
// dSpec is  P^T * W^T * R^T * dSpec :
//   * R^T per frame is exactly rfft_backward's adjoint (the +1-sign unscaled
//     DFT of the zero-padded n_fft spectrum, real part);
//   * W^T is the same window multiply (diagonal — self-transpose);
//   * P^T accumulates each frame sample back into the signal (the same index
//     map, summed), so overlapping frames add — NO COLA division here (that
//     belongs to istft, a different map).
// dSignal is *overwritten* (zeroed then accumulated).
void stft_backward(const ::brotensor::Tensor& dSpec,
                   const ::brotensor::Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   ::brotensor::Tensor& dSignal) {
    require_fp32_host("stft_backward", dSpec, "dSpec");
    require_fp32_host("stft_backward", window, "window");
    if (window.rows != 1 || window.cols != win_length) {
        fail("stft_backward", "window must be a (1, win_length) tensor");
    }
    const StftGeom g = check_geom("stft_backward", N, signal_len, n_fft,
                                  hop_length, win_length, center);
    const int exp_rows = N * g.frames;
    const int exp_cols = 2 * g.bins;
    if (dSpec.rows != exp_rows || dSpec.cols != exp_cols) {
        fail("stft_backward", "dSpec shape must match the stft output shape");
    }
    if (dSignal.rows != N || dSignal.cols != signal_len) {
        dSignal.resize(N, signal_len);
    }
    if (dSignal.size() != 0) {
        float* z = dSignal.host_f32_mut();
        for (int i = 0; i < dSignal.size(); ++i) z[i] = 0.0f;
    }
    if (exp_rows == 0) return;

    const float* gp = dSpec.host_f32();
    const float* win = window.host_f32();
    float* dsig = dSignal.host_f32_mut();
    const double norm = normalized
                            ? 1.0 / std::sqrt(static_cast<double>(n_fft))
                            : 1.0;

    // Per frame: spec[k] = norm * (truncated DFT)[k]. The adjoint of the
    // truncated forward DFT is: zero-pad dSpec to length n_fft, run an
    // unscaled +1-sign DFT, take the real part (== rfft_backward's core).
    std::vector<Cd> spec(static_cast<std::size_t>(n_fft)), tbuf;
    for (int b = 0; b < N; ++b) {
        float* drow = dsig + static_cast<std::size_t>(b) * signal_len;
        for (int f = 0; f < g.frames; ++f) {
            const float* grow = gp + static_cast<std::size_t>(
                                         static_cast<std::size_t>(b) *
                                             g.frames +
                                         f) * exp_cols;
            for (int k = 0; k < n_fft; ++k) spec[static_cast<std::size_t>(k)] = Cd{};
            for (int k = 0; k < g.bins; ++k) {
                spec[static_cast<std::size_t>(k)] =
                    {static_cast<double>(grow[2 * k]) * norm,
                     static_cast<double>(grow[2 * k + 1]) * norm};
            }
            dft_1d(spec, tbuf, +1);  // adjoint of truncated forward DFT
            // W^T (window) then P^T (scatter-add into the signal).
            const int base = f * hop_length;
            for (int j = 0; j < win_length; ++j) {
                const int i = g.pad_lo + j;
                const int p = base + i;
                const int s = padded_index(p, signal_len, n_fft, center);
                drow[s] += static_cast<float>(
                    tbuf[static_cast<std::size_t>(i)].re *
                    static_cast<double>(win[j]));
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  istft — complex spectrogram -> real signal (windowed overlap-add + COLA)
// ════════════════════════════════════════════════════════════════════════════
//
// Per frame: irfft the n_fft spectrum, multiply by the window, scatter-add
// into the output. Then divide each output sample by the overlap-added
// squared window (the COLA envelope) so a COLA-satisfying window+hop makes
// istft(stft(x)) == x. Samples with a ~0 envelope (edges with no frame
// coverage) stay 0.
void istft(const ::brotensor::Tensor& spec, const ::brotensor::Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, ::brotensor::Tensor& signal) {
    require_fp32_host("istft", spec, "spec");
    require_fp32_host("istft", window, "window");
    if (window.rows != 1 || window.cols != win_length) {
        fail("istft", "window must be a (1, win_length) tensor");
    }
    const StftGeom g = check_geom("istft", N, signal_len, n_fft, hop_length,
                                  win_length, center);
    const int exp_rows = N * g.frames;
    const int exp_cols = 2 * g.bins;
    if (spec.rows != exp_rows || spec.cols != exp_cols) {
        fail("istft", "spec shape must match the stft output shape");
    }
    if (signal.rows != N || signal.cols != signal_len) {
        signal.resize(N, signal_len);
    }
    if (signal.size() != 0) {
        float* z = signal.host_f32_mut();
        for (int i = 0; i < signal.size(); ++i) z[i] = 0.0f;
    }
    if (exp_rows == 0) return;

    const float* sp = spec.host_f32();
    const float* win = window.host_f32();
    float* sig = signal.host_f32_mut();
    // istft inverts stft's optional 1/sqrt(n_fft): multiply the spectrum by
    // sqrt(n_fft) so the irfft 1/n_fft scaling lands at the right amplitude.
    const double norm = normalized ? std::sqrt(static_cast<double>(n_fft))
                                   : 1.0;
    const double invN = 1.0 / static_cast<double>(n_fft);

    // COLA envelope: overlap-added squared window, in padded coordinates.
    std::vector<double> env(static_cast<std::size_t>(g.padded_len), 0.0);
    for (int f = 0; f < g.frames; ++f) {
        const int base = f * hop_length;
        for (int j = 0; j < win_length; ++j) {
            const int p = base + g.pad_lo + j;
            const double w = static_cast<double>(win[j]);
            env[static_cast<std::size_t>(p)] += w * w;
        }
    }

    // Per signal: overlap-add the windowed irfft frames, then COLA-divide.
    std::vector<Cd> full(static_cast<std::size_t>(n_fft)), out;
    std::vector<double> acc(static_cast<std::size_t>(g.padded_len));
    for (int b = 0; b < N; ++b) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int f = 0; f < g.frames; ++f) {
            const float* srow = sp + static_cast<std::size_t>(
                                         static_cast<std::size_t>(b) *
                                             g.frames +
                                         f) * exp_cols;
            // Rebuild the Hermitian-symmetric n_fft spectrum, irfft it.
            for (int k = 0; k < g.bins; ++k) {
                full[static_cast<std::size_t>(k)] =
                    {static_cast<double>(srow[2 * k]) * norm,
                     static_cast<double>(srow[2 * k + 1]) * norm};
            }
            for (int k = 1; k < n_fft - g.bins + 1; ++k) {
                const Cd c = full[static_cast<std::size_t>(k)];
                full[static_cast<std::size_t>(n_fft - k)] = {c.re, -c.im};
            }
            dft_1d(full, out, +1);  // inverse DFT, still needs *1/n_fft
            const int base = f * hop_length;
            for (int j = 0; j < win_length; ++j) {
                const int i = g.pad_lo + j;
                const int p = base + i;
                const double t = out[static_cast<std::size_t>(i)].re * invN;
                acc[static_cast<std::size_t>(p)] +=
                    t * static_cast<double>(win[j]);
            }
        }
        // COLA-divide and strip centre padding back to the raw signal.
        float* drow = sig + static_cast<std::size_t>(b) * signal_len;
        const int shift = center ? n_fft / 2 : 0;
        for (int n = 0; n < signal_len; ++n) {
            const int p = n + shift;
            const double e = env[static_cast<std::size_t>(p)];
            drow[n] = (e > 1e-10)
                          ? static_cast<float>(acc[static_cast<std::size_t>(p)]
                                               / e)
                          : 0.0f;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  istft_backward — adjoint of istft
// ════════════════════════════════════════════════════════════════════════════
//
// istft is the linear map  signal = D * E^{-1} * P * W * I * spec  where I is
// the per-frame inverse DFT (1/n_fft scaled), W the window multiply, P the
// overlap-add scatter, E^{-1} the per-sample COLA division, and D the
// centre-padding strip. The COLA envelope E depends only on the window/hop,
// not on the spectrum, so E^{-1} is a (data-independent) diagonal — and its
// transpose is itself. The adjoint applied to dSignal is therefore
//   I^T * W^T * P^T * E^{-1} * D^T * dSignal :
//   * D^T scatters dSignal back into padded coordinates;
//   * E^{-1} divides by the same COLA envelope (diagonal, self-transpose);
//   * P^T gathers each frame's window_length samples;
//   * W^T is the window multiply again;
//   * I^T is the adjoint of the inverse DFT — which is irfft_backward's core
//     (forward-sign DFT of the gathered frame, 1/n_fft scaling, and the
//     interior-bin doubling from the Hermitian fold).
// dSpec is *overwritten*.
void istft_backward(const ::brotensor::Tensor& dSignal,
                    const ::brotensor::Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    ::brotensor::Tensor& dSpec) {
    require_fp32_host("istft_backward", dSignal, "dSignal");
    require_fp32_host("istft_backward", window, "window");
    if (dSignal.rows != N || dSignal.cols != signal_len) {
        fail("istft_backward", "dSignal must be a (N, signal_len) tensor");
    }
    if (window.rows != 1 || window.cols != win_length) {
        fail("istft_backward", "window must be a (1, win_length) tensor");
    }
    const StftGeom g = check_geom("istft_backward", N, signal_len, n_fft,
                                  hop_length, win_length, center);
    const int out_rows = N * g.frames;
    const int out_cols = 2 * g.bins;
    if (dSpec.rows != out_rows || dSpec.cols != out_cols) {
        dSpec.resize(out_rows, out_cols);
    }
    if (out_rows == 0) return;

    const float* dsig = dSignal.host_f32();
    const float* win = window.host_f32();
    float* gp = dSpec.host_f32_mut();
    const double norm = normalized ? std::sqrt(static_cast<double>(n_fft))
                                   : 1.0;
    const double invN = 1.0 / static_cast<double>(n_fft);

    // Same COLA envelope as istft (window/hop only).
    std::vector<double> env(static_cast<std::size_t>(g.padded_len), 0.0);
    for (int f = 0; f < g.frames; ++f) {
        const int base = f * hop_length;
        for (int j = 0; j < win_length; ++j) {
            const int p = base + g.pad_lo + j;
            const double w = static_cast<double>(win[j]);
            env[static_cast<std::size_t>(p)] += w * w;
        }
    }

    const bool even = (n_fft % 2 == 0);
    std::vector<double> gacc(static_cast<std::size_t>(g.padded_len));
    std::vector<Cd> frame(static_cast<std::size_t>(n_fft)), spec;
    for (int b = 0; b < N; ++b) {
        // D^T then E^{-1}: scatter dSignal into padded coords, COLA-divide.
        std::fill(gacc.begin(), gacc.end(), 0.0);
        const float* drow = dsig + static_cast<std::size_t>(b) * signal_len;
        const int shift = center ? n_fft / 2 : 0;
        for (int n = 0; n < signal_len; ++n) {
            const int p = n + shift;
            const double e = env[static_cast<std::size_t>(p)];
            gacc[static_cast<std::size_t>(p)] =
                (e > 1e-10) ? static_cast<double>(drow[n]) / e : 0.0;
        }
        for (int f = 0; f < g.frames; ++f) {
            // P^T (gather) then W^T (window): the frame's n_fft time buffer.
            for (int k = 0; k < n_fft; ++k) frame[static_cast<std::size_t>(k)] = Cd{};
            const int base = f * hop_length;
            for (int j = 0; j < win_length; ++j) {
                const int i = g.pad_lo + j;
                const int p = base + i;
                frame[static_cast<std::size_t>(i)] =
                    {gacc[static_cast<std::size_t>(p)] *
                         static_cast<double>(win[j]),
                     0.0};
            }
            // I^T: adjoint of the 1/n_fft inverse DFT — forward-sign DFT,
            // 1/n_fft scaling, interior-bin doubling for the Hermitian fold.
            dft_1d(frame, spec, -1);
            float* grow = gp + static_cast<std::size_t>(
                                   static_cast<std::size_t>(b) * g.frames +
                                   f) * out_cols;
            for (int k = 0; k < g.bins; ++k) {
                double s = 2.0;
                if (k == 0) s = 1.0;
                if (even && k == n_fft / 2) s = 1.0;
                const double scale = s * invN * norm;
                grow[2 * k] = static_cast<float>(
                    scale * spec[static_cast<std::size_t>(k)].re);
                grow[2 * k + 1] = static_cast<float>(
                    scale * spec[static_cast<std::size_t>(k)].im);
            }
        }
    }
}

} // namespace brotensor::detail::cpu
