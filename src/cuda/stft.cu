// ─── CUDA STFT / iSTFT (brosoundml CHUNK 2) ────────────────────────────────
//
// CUDA port of src/cpu/stft.cpp. FP32-only. Where the CPU backend uses the
// shared mixed-radix + Bluestein engine for each per-frame transform, the
// CUDA backend computes each frame's DFT directly (naive O(n_fft^2), one
// thread per output bin) — mathematically identical, kept within the parity
// suite's 4e-3 tolerance by double-precision accumulation.
//
// Ops: stft / stft_backward, istft / istft_backward.
//
// Layout (see ops.h / the CPU file for the full contract):
//   signal:  REAL (N, signal_len)
//   window:  REAL (1, win_length)
//   spec:    interleaved-complex (N*frames, 2*bins), bins = n_fft/2+1
//
// Each frame f of signal b windows the central win_length of an n_fft buffer
// (pad_lo = (n_fft-win_length)/2 zeros each side) and rfft's it. center=true
// reflect-pads the signal by n_fft/2 each side first. istft does windowed
// overlap-add then a COLA (overlap-added squared window) division. Each
// backward op is the exact transpose of its own forward linear map.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor { void* cuda_current_stream(); }
static inline cudaStream_t cur_stream() {
    return reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
}

namespace brotensor::detail::cuda {

namespace {

constexpr int STFT_BLOCK = 128;
constexpr double kTwoPi = 6.28318530717958647692;

inline int stft_grid(long long n) {
    long long blocks = (n + STFT_BLOCK - 1) / STFT_BLOCK;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return static_cast<int>(blocks);
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
    }
}

// Reject a committed (data != nullptr) output that is not FP32. Uncommitted
// outputs get their dtype pinned by the explicit FP32 resize below.
inline void require_fp32_out(const char* op, const ::brotensor::Tensor& t,
                             const char* name) {
    if (t.data != nullptr && t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) +
             " must be FP32 (audio ops are FP32-only)");
    }
}

// Common parameter validation + derived sizes — mirrors the CPU StftGeom.
struct StftGeom {
    int bins = 0;        // n_fft/2 + 1
    int frames = 0;      // frames per signal
    int padded_len = 0;  // signal length the frame loop indexes into
    int pad_lo = 0;      // (n_fft - win_length) / 2
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

// Reflection over [0, L-1] with period 2*(L-1) — numpy 'reflect'.
__device__ inline int reflect_index(int q, int L) {
    if (L == 1) return 0;
    const int period = 2 * (L - 1);
    int m = q % period;
    if (m < 0) m += period;
    return (m < L) ? m : period - m;
}

// Map a padded position to a raw signal index. center=false is identity.
__device__ inline int padded_index(int p, int signal_len, int n_fft,
                                   int center) {
    if (!center) return p;
    return reflect_index(p - n_fft / 2, signal_len);
}

// ─── stft ───────────────────────────────────────────────────────────────────
// One thread per (b, f, k) output bin.
__global__ void stft_kernel(const float* __restrict__ signal,
                            const float* __restrict__ window,
                            float* __restrict__ spec,
                            int N, int signal_len, int n_fft, int hop,
                            int win_length, int pad_lo, int bins, int frames,
                            int center, double norm) {
    const long long total = (long long)N * frames * bins;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int k = static_cast<int>(g % bins);
        const long long t = g / bins;
        const int f = static_cast<int>(t % frames);
        const int b = static_cast<int>(t / frames);
        const float* srow = signal + (long long)b * signal_len;
        const int base = f * hop;
        const double w = -kTwoPi * k / n_fft;       // forward sign
        double accr = 0.0, acci = 0.0;
        for (int j = 0; j < win_length; ++j) {
            const int i = pad_lo + j;
            const int s = padded_index(base + i, signal_len, n_fft, center);
            const double val = (double)srow[s] * (double)window[j];
            double sn, cs;
            sincos(w * i, &sn, &cs);
            accr += val * cs;
            acci += val * sn;
        }
        float* dst = spec + ((long long)b * frames + f) * (2 * bins);
        dst[2 * k]     = static_cast<float>(accr * norm);
        dst[2 * k + 1] = static_cast<float>(acci * norm);
    }
}

// ─── stft_backward ──────────────────────────────────────────────────────────
// One thread per (b, f, j) window sample; scatter-adds into dSignal (zeroed
// first) so overlapping frames accumulate — atomicAdd handles the collision.
__global__ void stft_backward_kernel(const float* __restrict__ dSpec,
                                     const float* __restrict__ window,
                                     float* __restrict__ dSignal,
                                     int N, int signal_len, int n_fft, int hop,
                                     int win_length, int pad_lo, int bins,
                                     int frames, int center, double norm) {
    const long long total = (long long)N * frames * win_length;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int j = static_cast<int>(g % win_length);
        const long long t = g / win_length;
        const int f = static_cast<int>(t % frames);
        const int b = static_cast<int>(t / frames);
        const int i = pad_lo + j;
        const int s = padded_index(f * hop + i, signal_len, n_fft, center);
        const float* grow = dSpec + ((long long)b * frames + f) * (2 * bins);
        const double w = kTwoPi * i / n_fft;        // inverse sign (adjoint)
        double tre = 0.0;
        for (int k = 0; k < bins; ++k) {
            const double gr = (double)grow[2 * k] * norm;
            const double gi = (double)grow[2 * k + 1] * norm;
            double sn, cs;
            sincos(w * k, &sn, &cs);
            tre += gr * cs - gi * sn;               // Re(spec[k]*exp(+i ang))
        }
        atomicAdd(&dSignal[(long long)b * signal_len + s],
                  static_cast<float>(tre * (double)window[j]));
    }
}

// COLA envelope: overlap-added squared window in padded coordinates. One
// thread per padded position.
__global__ void cola_env_kernel(const float* __restrict__ window,
                                float* __restrict__ env,
                                int padded_len, int n_fft, int hop,
                                int win_length, int pad_lo, int frames) {
    (void)n_fft;
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < padded_len; p += (long long)blockDim.x * gridDim.x) {
        double e = 0.0;
        // Only frames with 0 <= p - f*hop - pad_lo < win_length cover p —
        // iterate exactly that range instead of scanning all `frames` (at
        // small hop the full scan is thousands of mostly-skipped iterations
        // per sample).
        const int rel  = static_cast<int>(p) - pad_lo;
        const int f_hi = min(frames - 1, rel >= 0 ? rel / hop : -1);
        const int f_lo = max(0, rel - win_length + hop >= 0
                                    ? (rel - win_length + hop) / hop : 0);
        for (int f = f_lo; f <= f_hi; ++f) {
            const int jj = rel - f * hop;
            const double w = window[jj];
            e += w * w;
        }
        env[p] = static_cast<float>(e);
    }
}

// ─── istft ──────────────────────────────────────────────────────────────────
// One thread per (b, n) output sample. Sums every frame that covers the
// padded position, then COLA-divides.
__global__ void istft_kernel(const float* __restrict__ spec,
                             const float* __restrict__ window,
                             const float* __restrict__ env,
                             float* __restrict__ signal,
                             int N, int signal_len, int n_fft, int hop,
                             int win_length, int pad_lo, int bins, int frames,
                             int center, double norm, double invN) {
    const long long total = (long long)N * signal_len;
    const int shift = center ? n_fft / 2 : 0;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int n = static_cast<int>(g % signal_len);
        const int b = static_cast<int>(g / signal_len);
        const int p = n + shift;
        const double e = env[p];
        if (!(e > 1e-10)) {
            signal[(long long)b * signal_len + n] = 0.0f;
            continue;
        }
        double acc = 0.0;
        // Same covering-frame range as cola_env_kernel: iterate only frames
        // with pad_lo <= p - f*hop < pad_lo + win_length instead of scanning
        // all `frames` per output sample.
        const int rel  = p - pad_lo;
        const int f_hi = min(frames - 1, rel >= 0 ? rel / hop : -1);
        const int f_lo = max(0, rel - win_length + hop >= 0
                                    ? (rel - win_length + hop) / hop : 0);
        for (int f = f_lo; f <= f_hi; ++f) {
            const int i = p - f * hop;
            const int jw = i - pad_lo;
            const float* srow = spec + ((long long)b * frames + f) * (2 * bins);
            const double w = kTwoPi * i / n_fft;    // inverse sign
            double v = 0.0;
            for (int k = 0; k < n_fft; ++k) {
                double fre, fim;
                if (k < bins) {
                    fre = (double)srow[2 * k] * norm;
                    fim = (double)srow[2 * k + 1] * norm;
                } else {                            // Hermitian mirror
                    const int m = n_fft - k;
                    fre = (double)srow[2 * m] * norm;
                    fim = -(double)srow[2 * m + 1] * norm;
                }
                double sn, cs;
                sincos(w * k, &sn, &cs);
                v += fre * cs - fim * sn;
            }
            acc += v * invN * (double)window[jw];
        }
        signal[(long long)b * signal_len + n] = static_cast<float>(acc / e);
    }
}

// ─── istft_backward ─────────────────────────────────────────────────────────
// One thread per (b, f, k) output bin. dSpec overwritten.
__global__ void istft_backward_kernel(const float* __restrict__ dSignal,
                                      const float* __restrict__ window,
                                      const float* __restrict__ env,
                                      float* __restrict__ dSpec,
                                      int N, int signal_len, int n_fft, int hop,
                                      int win_length, int pad_lo, int bins,
                                      int frames, int center, double norm,
                                      double invN) {
    const long long total = (long long)N * frames * bins;
    const int shift = center ? n_fft / 2 : 0;
    const bool even = (n_fft % 2 == 0);
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int k = static_cast<int>(g % bins);
        const long long t = g / bins;
        const int f = static_cast<int>(t % frames);
        const int b = static_cast<int>(t / frames);
        const int base = f * hop;
        const double w = -kTwoPi * k / n_fft;       // forward DFT sign
        double sre = 0.0, sim = 0.0;
        for (int j = 0; j < win_length; ++j) {
            const int i = pad_lo + j;
            const int p = base + i;
            const int nn = p - shift;
            double gacc = 0.0;
            if (nn >= 0 && nn < signal_len) {
                const double e = env[p];
                if (e > 1e-10) {
                    gacc = (double)dSignal[(long long)b * signal_len + nn] / e;
                }
            }
            const double frame_i = gacc * (double)window[j];
            double sn, cs;
            sincos(w * i, &sn, &cs);
            sre += frame_i * cs;
            sim += frame_i * sn;
        }
        double sk = 2.0;
        if (k == 0) sk = 1.0;
        if (even && k == n_fft / 2) sk = 1.0;
        const double scale = sk * invN * norm;
        float* grow = dSpec + ((long long)b * frames + f) * (2 * bins);
        grow[2 * k]     = static_cast<float>(scale * sre);
        grow[2 * k + 1] = static_cast<float>(scale * sim);
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Wrappers
// ════════════════════════════════════════════════════════════════════════════

void stft(const ::brotensor::Tensor& signal, const ::brotensor::Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, ::brotensor::Tensor& spec) {
    require_fp32("stft", signal, "signal");
    require_fp32("stft", window, "window");
    if (signal.rows != N) fail("stft", "signal.rows must equal N");
    const int signal_len = signal.cols;
    if (window.rows != 1 || window.cols != win_length) {
        fail("stft", "window must be a (1, win_length) tensor");
    }
    const StftGeom g = check_geom("stft", N, signal_len, n_fft, hop_length,
                                  win_length, center);
    const int out_rows = N * g.frames;
    const int out_cols = 2 * g.bins;
    require_fp32_out("stft", spec, "spec");
    if (spec.rows != out_rows || spec.cols != out_cols ||
        spec.dtype != ::brotensor::Dtype::FP32) {
        spec.resize(out_rows, out_cols, ::brotensor::Dtype::FP32);
    }
    if (out_rows == 0) return;
    const double norm = normalized
                            ? 1.0 / std::sqrt(static_cast<double>(n_fft))
                            : 1.0;
    const long long total = (long long)N * g.frames * g.bins;
    stft_kernel<<<stft_grid(total), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(signal.data),
        static_cast<const float*>(window.data),
        static_cast<float*>(spec.data),
        N, signal_len, n_fft, hop_length, win_length, g.pad_lo, g.bins,
        g.frames, center ? 1 : 0, norm);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void stft_backward(const ::brotensor::Tensor& dSpec,
                   const ::brotensor::Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   ::brotensor::Tensor& dSignal) {
    require_fp32("stft_backward", dSpec, "dSpec");
    require_fp32("stft_backward", window, "window");
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
    require_fp32_out("stft_backward", dSignal, "dSignal");
    if (dSignal.rows != N || dSignal.cols != signal_len ||
        dSignal.dtype != ::brotensor::Dtype::FP32) {
        dSignal.resize(N, signal_len, ::brotensor::Dtype::FP32);
    }
    if (dSignal.size() != 0) {
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
            dSignal.data, 0,
            static_cast<size_t>(dSignal.size()) * sizeof(float), cur_stream()));
    }
    if (exp_rows == 0) return;
    const double norm = normalized
                            ? 1.0 / std::sqrt(static_cast<double>(n_fft))
                            : 1.0;
    const long long total = (long long)N * g.frames * win_length;
    stft_backward_kernel<<<stft_grid(total), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dSpec.data),
        static_cast<const float*>(window.data),
        static_cast<float*>(dSignal.data),
        N, signal_len, n_fft, hop_length, win_length, g.pad_lo, g.bins,
        g.frames, center ? 1 : 0, norm);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void istft(const ::brotensor::Tensor& spec, const ::brotensor::Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, ::brotensor::Tensor& signal) {
    require_fp32("istft", spec, "spec");
    require_fp32("istft", window, "window");
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
    require_fp32_out("istft", signal, "signal");
    if (signal.rows != N || signal.cols != signal_len ||
        signal.dtype != ::brotensor::Dtype::FP32) {
        signal.resize(N, signal_len, ::brotensor::Dtype::FP32);
    }
    if (exp_rows == 0) return;
    const double norm = normalized ? std::sqrt(static_cast<double>(n_fft))
                                   : 1.0;
    const double invN = 1.0 / static_cast<double>(n_fft);

    float* env = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(
        &env, static_cast<size_t>(g.padded_len) * sizeof(float)));
    cola_env_kernel<<<stft_grid(g.padded_len), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(window.data), env,
        g.padded_len, n_fft, hop_length, win_length, g.pad_lo, g.frames);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    const long long total = (long long)N * signal_len;
    istft_kernel<<<stft_grid(total), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(spec.data),
        static_cast<const float*>(window.data), env,
        static_cast<float*>(signal.data),
        N, signal_len, n_fft, hop_length, win_length, g.pad_lo, g.bins,
        g.frames, center ? 1 : 0, norm, invN);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    BROTENSOR_CUDA_CHECK(cudaFree(env));
}

void istft_backward(const ::brotensor::Tensor& dSignal,
                    const ::brotensor::Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    ::brotensor::Tensor& dSpec) {
    require_fp32("istft_backward", dSignal, "dSignal");
    require_fp32("istft_backward", window, "window");
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
    require_fp32_out("istft_backward", dSpec, "dSpec");
    if (dSpec.rows != out_rows || dSpec.cols != out_cols ||
        dSpec.dtype != ::brotensor::Dtype::FP32) {
        dSpec.resize(out_rows, out_cols, ::brotensor::Dtype::FP32);
    }
    if (out_rows == 0) return;
    const double norm = normalized ? std::sqrt(static_cast<double>(n_fft))
                                   : 1.0;
    const double invN = 1.0 / static_cast<double>(n_fft);

    float* env = nullptr;
    BROTENSOR_CUDA_CHECK(cudaMalloc(
        &env, static_cast<size_t>(g.padded_len) * sizeof(float)));
    cola_env_kernel<<<stft_grid(g.padded_len), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(window.data), env,
        g.padded_len, n_fft, hop_length, win_length, g.pad_lo, g.frames);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    const long long total = (long long)N * g.frames * g.bins;
    istft_backward_kernel<<<stft_grid(total), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dSignal.data),
        static_cast<const float*>(window.data), env,
        static_cast<float*>(dSpec.data),
        N, signal_len, n_fft, hop_length, win_length, g.pad_lo, g.bins,
        g.frames, center ? 1 : 0, norm, invN);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
    BROTENSOR_CUDA_CHECK(cudaFree(env));
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_stft(::brotensor::detail::OpsVTable& v) {
    v.stft           = &stft;
    v.stft_backward  = &stft_backward;
    v.istft          = &istft;
    v.istft_backward = &istft_backward;
}

} // namespace brotensor::detail::cuda
