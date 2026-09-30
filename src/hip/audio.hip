// ─── HIP STFT / iSTFT & Audio VTable Registration (brosoundml CHUNK 2) ───
//
// AMD ROCm / HIP port of src/cuda/stft.cu. FP32-only.
//
// Ops: stft / stft_backward, istft / istft_backward, fill_hip_vtable_audio.

#include "detail/hip_check.h"
#include "detail/audio_internal.h"
#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace brotensor::detail::hip {

void* hip_alloc(std::size_t bytes, int dev = -1);
void  hip_free(void* ptr, int dev = -1);
void* hip_current_stream(int dev = -1);

static inline hipStream_t cur_stream() {
    return reinterpret_cast<hipStream_t>(hip_current_stream());
}

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

inline void require_fp32_out(const char* op, const ::brotensor::Tensor& t,
                             const char* name) {
    if (t.data != nullptr && t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) +
             " must be FP32 (audio ops are FP32-only)");
    }
}

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

__device__ inline int reflect_index(int q, int L) {
    if (L == 1) return 0;
    const int period = 2 * (L - 1);
    int m = q % period;
    if (m < 0) m += period;
    return (m < L) ? m : period - m;
}

__device__ inline int padded_index(int p, int signal_len, int n_fft,
                                   int center) {
    if (!center) return p;
    return reflect_index(p - n_fft / 2, signal_len);
}

// ─── stft ───────────────────────────────────────────────────────────────────

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
        const double w = -kTwoPi * k / n_fft;
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
        const double w = kTwoPi * i / n_fft;
        double tre = 0.0;
        for (int k = 0; k < bins; ++k) {
            const double gr = (double)grow[2 * k] * norm;
            const double gi = (double)grow[2 * k + 1] * norm;
            double sn, cs;
            sincos(w * k, &sn, &cs);
            tre += gr * cs - gi * sn;
        }
        atomicAdd(&dSignal[(long long)b * signal_len + s],
                  static_cast<float>(tre * (double)window[j]));
    }
}

// ─── cola_env ───────────────────────────────────────────────────────────────

__global__ void cola_env_kernel(const float* __restrict__ window,
                                float* __restrict__ env,
                                int padded_len, int n_fft, int hop,
                                int win_length, int pad_lo, int frames) {
    (void)n_fft;
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < padded_len; p += (long long)blockDim.x * gridDim.x) {
        double e = 0.0;
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
        const int rel  = p - pad_lo;
        const int f_hi = min(frames - 1, rel >= 0 ? rel / hop : -1);
        const int f_lo = max(0, rel - win_length + hop >= 0
                                    ? (rel - win_length + hop) / hop : 0);
        for (int f = f_lo; f <= f_hi; ++f) {
            const int i = p - f * hop;
            const int jw = i - pad_lo;
            const float* srow = spec + ((long long)b * frames + f) * (2 * bins);
            const double w = kTwoPi * i / n_fft;
            double v = 0.0;
            for (int k = 0; k < n_fft; ++k) {
                double fre, fim;
                if (k < bins) {
                    fre = (double)srow[2 * k] * norm;
                    fim = (double)srow[2 * k + 1] * norm;
                } else {
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
        const double w = -kTwoPi * k / n_fft;
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
//  STFT Wrappers
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
    BROTENSOR_HIP_CHECK(hipGetLastError());
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
        BROTENSOR_HIP_CHECK(hipMemsetAsync(
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
    BROTENSOR_HIP_CHECK(hipGetLastError());
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

    float* env = static_cast<float*>(hip_alloc(static_cast<size_t>(g.padded_len) * sizeof(float)));
    cola_env_kernel<<<stft_grid(g.padded_len), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(window.data), env,
        g.padded_len, n_fft, hop_length, win_length, g.pad_lo, g.frames);
    BROTENSOR_HIP_CHECK(hipGetLastError());

    const long long total = (long long)N * signal_len;
    istft_kernel<<<stft_grid(total), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(spec.data),
        static_cast<const float*>(window.data), env,
        static_cast<float*>(signal.data),
        N, signal_len, n_fft, hop_length, win_length, g.pad_lo, g.bins,
        g.frames, center ? 1 : 0, norm, invN);
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(env);
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

    float* env = static_cast<float*>(hip_alloc(static_cast<size_t>(g.padded_len) * sizeof(float)));
    cola_env_kernel<<<stft_grid(g.padded_len), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(window.data), env,
        g.padded_len, n_fft, hop_length, win_length, g.pad_lo, g.frames);
    BROTENSOR_HIP_CHECK(hipGetLastError());

    const long long total = (long long)N * g.frames * g.bins;
    istft_backward_kernel<<<stft_grid(total), STFT_BLOCK, 0, cur_stream()>>>(
        static_cast<const float*>(dSignal.data),
        static_cast<const float*>(window.data), env,
        static_cast<float*>(dSpec.data),
        N, signal_len, n_fft, hop_length, win_length, g.pad_lo, g.bins,
        g.frames, center ? 1 : 0, norm, invN);
    BROTENSOR_HIP_CHECK(hipGetLastError());
    hip_free(env);
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_hip_vtable_audio(::brotensor::detail::OpsVTable& v) {
    // Vocoder / activations
    v.snake_forward        = &snake_forward;
    v.snake_backward       = &snake_backward;
    v.elu_forward          = &elu_forward;
    v.elu_backward         = &elu_backward;
    v.leaky_relu_forward   = &leaky_relu_forward;
    v.leaky_relu_backward  = &leaky_relu_backward;

    // Codec quantization
    v.vq_encode_forward    = &vq_encode_forward;
    v.vq_encode_backward   = &vq_encode_backward;
    v.fsq_quantize_forward = &fsq_quantize_forward;
    v.fsq_quantize_backward= &fsq_quantize_backward;

    // 1D resampling
    v.resample1d_forward   = &resample1d_forward;
    v.resample1d_backward  = &resample1d_backward;

    // Sampling
    v.sample_logits        = &sample_logits;
    v.sample_logits_into   = &sample_logits_into;

    // Spectral / FFT
    v.complex_mul          = &complex_mul;
    v.complex_mul_backward = &complex_mul_backward;
    v.complex_abs          = &complex_abs;
    v.complex_abs_backward = &complex_abs_backward;
    v.complex_angle        = &complex_angle;
    v.complex_from_polar   = &complex_from_polar;
    v.fft                  = &fft;
    v.ifft                 = &ifft;
    v.rfft                 = &rfft;
    v.irfft                = &irfft;
    v.rfft_backward        = &rfft_backward;
    v.irfft_backward       = &irfft_backward;

    // STFT
    v.stft                 = &stft;
    v.stft_backward        = &stft_backward;
    v.istft                = &istft;
    v.istft_backward       = &istft_backward;
}

} // namespace brotensor::detail::hip
