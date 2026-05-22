// ─── Metal STFT / iSTFT (brosoundml CHUNK 2) ───────────────────────────────
//
// Metal counterpart of src/cpu/stft.cpp. FP32-only. Ops implemented here:
//   stft / stft_backward     real signal  <-> complex spectrogram
//   istft / istft_backward   complex spectrogram <-> real signal (COLA OLA)
//
// ── Layout ──────────────────────────────────────────────────────────────────
// signal:  REAL (N, signal_len)               — N batched signals, one / row.
// window:  REAL (1, win_length)                — caller-supplied.
// spec:    interleaved-complex (N*frames, 2*bins), bins = n_fft/2+1. Each
//          frame is a row; the N signals' frame blocks are stacked in order.
//
// ── Strategy ────────────────────────────────────────────────────────────────
// Each op is a single GPU kernel whose threads each compute one *output*
// element by gathering everything that contributes to it — no scratch tensors,
// no atomics. The per-frame transform is a direct DFT (see src/metal/fft.mm
// for why the naive O(N^2) sum is the right call on Metal). The frame model,
// reflect padding, COLA envelope, and gradient adjoints exactly mirror
// src/cpu/stft.cpp — read that file for the full contract and derivations.

#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

#import "internal.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;

namespace {

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

void req_fp32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (STFT ops are FP32-only)");
    }
}

// Parameter block — must match `struct StftParams` in the MSL source below.
struct StftParams {
    uint32_t N;
    uint32_t signal_len;
    uint32_t n_fft;
    uint32_t hop;
    uint32_t win_length;
    uint32_t pad_lo;
    uint32_t bins;
    uint32_t frames;
    uint32_t center;   // 0 / 1
    uint32_t even;     // 0 / 1
    float    norm;
    float    invN;
};

// Geometry — mirrors check_geom() in src/cpu/stft.cpp.
struct StftGeom {
    int bins = 0;
    int frames = 0;
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
    if (center) {
        if (signal_len < n_fft / 2 + 1) {
            fail(op, "center=true needs signal_len >= n_fft/2 + 1");
        }
        g.frames = 1 + signal_len / hop_length;
    } else {
        if (signal_len < n_fft) {
            fail(op, "center=false needs signal_len >= n_fft");
        }
        g.frames = 1 + (signal_len - n_fft) / hop_length;
    }
    return g;
}

StftParams make_params(const StftGeom& g, int N, int signal_len, int n_fft,
                       int hop_length, int win_length, bool center,
                       float norm) {
    StftParams p{};
    p.N          = static_cast<uint32_t>(N);
    p.signal_len = static_cast<uint32_t>(signal_len);
    p.n_fft      = static_cast<uint32_t>(n_fft);
    p.hop        = static_cast<uint32_t>(hop_length);
    p.win_length = static_cast<uint32_t>(win_length);
    p.pad_lo     = static_cast<uint32_t>((n_fft - win_length) / 2);
    p.bins       = static_cast<uint32_t>(g.bins);
    p.frames     = static_cast<uint32_t>(g.frames);
    p.center     = center ? 1u : 0u;
    p.even       = (n_fft % 2 == 0) ? 1u : 0u;
    p.norm       = norm;
    p.invN       = 1.0f / static_cast<float>(n_fft);
    return p;
}

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant float kTwoPi = 6.28318530717958647692f;

struct StftParams {
    uint  N;
    uint  signal_len;
    uint  n_fft;
    uint  hop;
    uint  win_length;
    uint  pad_lo;
    uint  bins;
    uint  frames;
    uint  center;
    uint  even;
    float norm;
    float invN;
};

// (k*i mod n_fft) as a float — keeps the trig argument small for accuracy.
inline float phase(uint k, int i, uint n_fft) {
    return float((ulong(k) * ulong(uint(i))) % ulong(n_fft));
}

// numpy 'reflect': fold q into [0, L-1] with period 2*(L-1).
inline int reflect_index(int q, int L) {
    if (L == 1) return 0;
    int period = 2 * (L - 1);
    int m = q % period;
    if (m < 0) m += period;
    return (m < L) ? m : period - m;
}

inline int padded_index(int p, int signal_len, int n_fft, bool center) {
    if (!center) return p;
    return reflect_index(p - n_fft / 2, signal_len);
}

// COLA envelope at padded position p: sum of window^2 over frames covering p.
inline float env_at(int p, device const float* window,
                     uint frames, uint hop, uint pad_lo, uint win_length) {
    float e = 0.0f;
    for (uint f = 0u; f < frames; ++f) {
        int jj = p - int(f * hop) - int(pad_lo);
        if (jj >= 0 && jj < int(win_length)) {
            float w = window[uint(jj)];
            e += w * w;
        }
    }
    return e;
}

// ── stft: real signal -> complex spectrogram ────────────────────────────────
// One thread per (output frame row, bin k).
kernel void k_stft(device const float* signal [[buffer(0)]],
                   device const float* window [[buffer(1)]],
                   device float*       spec   [[buffer(2)]],
                   constant StftParams& P     [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= P.N * P.frames * P.bins) return;
    uint out_row = gid / P.bins;
    uint k       = gid - out_row * P.bins;
    uint b       = out_row / P.frames;
    uint f       = out_row - b * P.frames;
    uint base    = f * P.hop;
    uint sbase   = b * P.signal_len;
    float accr = 0.0f, acci = 0.0f;
    for (uint j = 0u; j < P.win_length; ++j) {
        int i = int(P.pad_lo) + int(j);
        int p = int(base) + i;
        int s = padded_index(p, int(P.signal_len), int(P.n_fft),
                             P.center != 0u);
        float sw = signal[sbase + uint(s)] * window[j];
        float ang = kTwoPi * phase(k, i, P.n_fft) / float(P.n_fft);
        accr += sw * precise::cos(ang);
        acci -= sw * precise::sin(ang);
    }
    uint si = out_row * 2u * P.bins + 2u * k;
    spec[si]      = accr * P.norm;
    spec[si + 1u] = acci * P.norm;
}

// ── stft_backward: adjoint of stft ──────────────────────────────────────────
// One thread per (signal index). Gathers every frame/window-tap that scatters
// into this raw signal sample.
kernel void k_stft_backward(device const float* dSpec   [[buffer(0)]],
                            device const float* window  [[buffer(1)]],
                            device float*       dSignal [[buffer(2)]],
                            constant StftParams& P      [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P.N * P.signal_len) return;
    uint b    = gid / P.signal_len;
    uint sraw = gid - b * P.signal_len;
    float acc = 0.0f;
    for (uint f = 0u; f < P.frames; ++f) {
        uint out_row = b * P.frames + f;
        uint base    = f * P.hop;
        uint dbase   = out_row * 2u * P.bins;
        for (uint j = 0u; j < P.win_length; ++j) {
            int i = int(P.pad_lo) + int(j);
            int p = int(base) + i;
            int s = padded_index(p, int(P.signal_len), int(P.n_fft),
                                 P.center != 0u);
            if (s != int(sraw)) continue;
            // tbuf_re at index i: +1-sign unscaled DFT of dSpec, real part.
            float t = 0.0f;
            for (uint k = 0u; k < P.bins; ++k) {
                float gr = dSpec[dbase + 2u * k];
                float gi = dSpec[dbase + 2u * k + 1u];
                float ang = kTwoPi * phase(k, i, P.n_fft) / float(P.n_fft);
                t += gr * precise::cos(ang) - gi * precise::sin(ang);
            }
            acc += window[j] * t * P.norm;
        }
    }
    dSignal[gid] = acc;
}

// ── istft: complex spectrogram -> real signal (windowed OLA + COLA) ──────────
// One thread per (output signal sample).
kernel void k_istft(device const float* spec   [[buffer(0)]],
                    device const float* window [[buffer(1)]],
                    device float*       signal [[buffer(2)]],
                    constant StftParams& P     [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= P.N * P.signal_len) return;
    uint b = gid / P.signal_len;
    uint n = gid - b * P.signal_len;
    int shift = (P.center != 0u) ? int(P.n_fft / 2u) : 0;
    int p = int(n) + shift;
    float env = 0.0f, acc = 0.0f;
    for (uint f = 0u; f < P.frames; ++f) {
        int jj = p - int(f * P.hop) - int(P.pad_lo);
        if (jj < 0 || jj >= int(P.win_length)) continue;
        int i = int(P.pad_lo) + jj;
        float w = window[uint(jj)];
        env += w * w;
        uint sbase = (b * P.frames + f) * 2u * P.bins;
        // irfft this frame's n_fft spectrum at sample i (Hermitian rebuild).
        float oi = 0.0f;
        for (uint k = 0u; k < P.n_fft; ++k) {
            uint  sk;
            float imsign;
            if (k < P.bins) { sk = k;            imsign =  1.0f; }
            else            { sk = P.n_fft - k;  imsign = -1.0f; }
            float re = P.norm * spec[sbase + 2u * sk];
            float im = imsign * P.norm * spec[sbase + 2u * sk + 1u];
            float ang = kTwoPi * phase(k, i, P.n_fft) / float(P.n_fft);
            oi += re * precise::cos(ang) - im * precise::sin(ang);
        }
        acc += w * (oi * P.invN);
    }
    signal[gid] = (env > 1e-10f) ? (acc / env) : 0.0f;
}

// ── istft_backward: adjoint of istft ────────────────────────────────────────
// One thread per (output frame row, bin k).
kernel void k_istft_backward(device const float* dSignal [[buffer(0)]],
                             device const float* window  [[buffer(1)]],
                             device float*       dSpec   [[buffer(2)]],
                             constant StftParams& P      [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= P.N * P.frames * P.bins) return;
    uint out_row = gid / P.bins;
    uint k       = gid - out_row * P.bins;
    uint b       = out_row / P.frames;
    uint f       = out_row - b * P.frames;
    int shift = (P.center != 0u) ? int(P.n_fft / 2u) : 0;
    uint base = f * P.hop;
    float specr = 0.0f, speci = 0.0f;
    for (uint j = 0u; j < P.win_length; ++j) {
        int i = int(P.pad_lo) + int(j);
        int p = int(base) + i;
        int nn = p - shift;
        float gacc = 0.0f;
        if (nn >= 0 && nn < int(P.signal_len)) {
            float e = env_at(p, window, P.frames, P.hop, P.pad_lo,
                             P.win_length);
            if (e > 1e-10f) {
                gacc = dSignal[b * P.signal_len + uint(nn)] / e;
            }
        }
        float frame = gacc * window[j];
        float ang = kTwoPi * phase(k, i, P.n_fft) / float(P.n_fft);
        specr += frame * precise::cos(ang);
        speci -= frame * precise::sin(ang);
    }
    float sk = 2.0f;
    if (k == 0u) sk = 1.0f;
    if (P.even != 0u && k == P.n_fft / 2u) sk = 1.0f;
    float scale = sk * P.invN * P.norm;
    uint di = out_row * 2u * P.bins + 2u * k;
    dSpec[di]      = scale * specr;
    dSpec[di + 1u] = scale * speci;
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_stft,           @"k_stft")
DEF_PSO(pso_stft_backward,  @"k_stft_backward")
DEF_PSO(pso_istft,          @"k_istft")
DEF_PSO(pso_istft_backward, @"k_istft_backward")
#undef DEF_PSO

// Dispatch one kernel: in(0), window(1), out(2), StftParams(3).
void launch(id<MTLComputePipelineState> pso, NSUInteger total,
            const Tensor& in, const Tensor& window, const Tensor& out,
            const StftParams& params) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buffer_for(in)     offset:buffer_offset_for(in)     atIndex:0];
        [enc setBuffer:buffer_for(window) offset:buffer_offset_for(window) atIndex:1];
        [enc setBuffer:buffer_for(out)    offset:buffer_offset_for(out)    atIndex:2];
        [enc setBytes:&params length:sizeof(StftParams) atIndex:3];
        NSUInteger tpt = [pso maxTotalThreadsPerThreadgroup];
        if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  stft — real signal -> complex spectrogram
// ════════════════════════════════════════════════════════════════════════════
void stft(const Tensor& signal, const Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, Tensor& spec) {
    req_fp32("stft", signal, "signal");
    req_fp32("stft", window, "window");
    if (signal.rows != N) fail("stft", "signal.rows must equal N");
    const int signal_len = signal.cols;
    if (window.rows != 1 || window.cols != win_length) {
        fail("stft", "window must be a (1, win_length) tensor");
    }
    const StftGeom g = check_geom("stft", N, signal_len, n_fft, hop_length,
                                  win_length, center);
    const int out_rows = N * g.frames;
    const int out_cols = 2 * g.bins;
    if (spec.rows != out_rows || spec.cols != out_cols ||
        spec.dtype != Dtype::FP32) {
        spec.resize(out_rows, out_cols, Dtype::FP32);
    }
    if (out_rows == 0) return;
    const float norm = normalized
                           ? 1.0f / std::sqrt(static_cast<float>(n_fft))
                           : 1.0f;
    const StftParams p = make_params(g, N, signal_len, n_fft, hop_length,
                                     win_length, center, norm);
    const NSUInteger total =
        static_cast<NSUInteger>(N) * g.frames * g.bins;
    launch(pso_stft(), total, signal, window, spec, p);
}

// ════════════════════════════════════════════════════════════════════════════
//  stft_backward — adjoint of stft
// ════════════════════════════════════════════════════════════════════════════
void stft_backward(const Tensor& dSpec, const Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   Tensor& dSignal) {
    req_fp32("stft_backward", dSpec, "dSpec");
    req_fp32("stft_backward", window, "window");
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
    if (dSignal.rows != N || dSignal.cols != signal_len ||
        dSignal.dtype != Dtype::FP32) {
        dSignal.resize(N, signal_len, Dtype::FP32);
    }
    if (exp_rows == 0 || dSignal.size() == 0) return;
    const float norm = normalized
                           ? 1.0f / std::sqrt(static_cast<float>(n_fft))
                           : 1.0f;
    const StftParams p = make_params(g, N, signal_len, n_fft, hop_length,
                                     win_length, center, norm);
    const NSUInteger total = static_cast<NSUInteger>(N) * signal_len;
    launch(pso_stft_backward(), total, dSpec, window, dSignal, p);
}

// ════════════════════════════════════════════════════════════════════════════
//  istft — complex spectrogram -> real signal
// ════════════════════════════════════════════════════════════════════════════
void istft(const Tensor& spec, const Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, Tensor& signal) {
    req_fp32("istft", spec, "spec");
    req_fp32("istft", window, "window");
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
    if (signal.rows != N || signal.cols != signal_len ||
        signal.dtype != Dtype::FP32) {
        signal.resize(N, signal_len, Dtype::FP32);
    }
    if (exp_rows == 0 || signal.size() == 0) return;
    // istft inverts stft's optional 1/sqrt(n_fft): multiply spec by sqrt(n_fft).
    const float norm = normalized ? std::sqrt(static_cast<float>(n_fft))
                                  : 1.0f;
    const StftParams p = make_params(g, N, signal_len, n_fft, hop_length,
                                     win_length, center, norm);
    const NSUInteger total = static_cast<NSUInteger>(N) * signal_len;
    launch(pso_istft(), total, spec, window, signal, p);
}

// ════════════════════════════════════════════════════════════════════════════
//  istft_backward — adjoint of istft
// ════════════════════════════════════════════════════════════════════════════
void istft_backward(const Tensor& dSignal, const Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    Tensor& dSpec) {
    req_fp32("istft_backward", dSignal, "dSignal");
    req_fp32("istft_backward", window, "window");
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
    if (dSpec.rows != out_rows || dSpec.cols != out_cols ||
        dSpec.dtype != Dtype::FP32) {
        dSpec.resize(out_rows, out_cols, Dtype::FP32);
    }
    if (out_rows == 0) return;
    const float norm = normalized ? std::sqrt(static_cast<float>(n_fft))
                                  : 1.0f;
    const StftParams p = make_params(g, N, signal_len, n_fft, hop_length,
                                     win_length, center, norm);
    const NSUInteger total =
        static_cast<NSUInteger>(N) * g.frames * g.bins;
    launch(pso_istft_backward(), total, dSignal, window, dSpec, p);
}

} // namespace brotensor::detail::metal
