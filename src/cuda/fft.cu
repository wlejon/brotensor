// ─── CUDA spectral / FFT core (brosoundml CHUNK 1) ─────────────────────────
//
// CUDA port of src/cpu/fft.cpp. FP32-only, matching the CPU contract verbatim.
//
// Ops implemented here:
//   complex_mul / complex_abs / complex_angle / complex_from_polar
//   complex_mul_backward / complex_abs_backward
//   fft / ifft         — complex -> complex, one signal per tensor row
//   rfft / irfft       — real (R,L) <-> complex (R, 2*(L/2+1))
//   rfft_backward / irfft_backward — adjoints of rfft / irfft
//
// ── Complex layout ─────────────────────────────────────────────────────────
// A complex tensor is a regular FP32 tensor with the bin axis interleaved
// [re, im, re, im, ...]; a spectrum of C bins over R rows is an (R, 2*C)
// tensor. There is no new Dtype.
//
// ── Algorithm ──────────────────────────────────────────────────────────────
// Where the CPU backend uses a mixed-radix + Bluestein engine, the CUDA
// backend computes the DFT directly — a naive O(N^2) sum, one CUDA thread per
// output bin. The transform lengths in play (Whisper's n_fft = 400 and below)
// make this trivially fast, and it is correct for *every* length. The two are
// mathematically identical; double-precision accumulation keeps them within
// the parity suite's 3e-3 tolerance.
//
// ── Normalisation ──────────────────────────────────────────────────────────
// "backward" convention (numpy default): forward unscaled, inverse * 1/N.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

namespace {

constexpr int FFT_BLOCK = 128;
constexpr double kTwoPi = 6.28318530717958647692;

inline int fft_grid(long long n) {
    long long blocks = (n + FFT_BLOCK - 1) / FFT_BLOCK;
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

// ─── complex elementwise kernels ────────────────────────────────────────────

// y = a * b, complex elementwise. One thread per complex pair.
__global__ void complex_mul_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   long long pairs, float* __restrict__ y) {
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < pairs; p += (long long)blockDim.x * gridDim.x) {
        const float ar = a[2 * p], ai = a[2 * p + 1];
        const float br = b[2 * p], bi = b[2 * p + 1];
        y[2 * p]     = ar * br - ai * bi;
        y[2 * p + 1] = ar * bi + ai * br;
    }
}

// dA += dY*conj(b), dB += dY*conj(a). dA/dB pre-zeroed by the caller; each
// thread owns a distinct pair so the += needs no atomics.
__global__ void complex_mul_backward_kernel(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            const float* __restrict__ dY,
                                            long long pairs,
                                            float* __restrict__ dA,
                                            float* __restrict__ dB) {
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < pairs; p += (long long)blockDim.x * gridDim.x) {
        const float gr = dY[2 * p], gi = dY[2 * p + 1];
        const float ar = a[2 * p], ai = a[2 * p + 1];
        const float br = b[2 * p], bi = b[2 * p + 1];
        dA[2 * p]     += gr * br + gi * bi;
        dA[2 * p + 1] += gi * br - gr * bi;
        dB[2 * p]     += gr * ar + gi * ai;
        dB[2 * p + 1] += gi * ar - gr * ai;
    }
}

// y = |z|, real magnitude per bin. z is (R,2C), y is (R,C).
__global__ void complex_abs_kernel(const float* __restrict__ z,
                                   long long bins, float* __restrict__ y) {
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         c < bins; c += (long long)blockDim.x * gridDim.x) {
        const float re = z[2 * c], im = z[2 * c + 1];
        y[c] = sqrtf(re * re + im * im);
    }
}

// dZ = dY * z / |z|; 0 at |z| == 0. dZ overwritten.
__global__ void complex_abs_backward_kernel(const float* __restrict__ z,
                                            const float* __restrict__ dY,
                                            long long bins,
                                            float* __restrict__ dZ) {
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         c < bins; c += (long long)blockDim.x * gridDim.x) {
        const float re = z[2 * c], im = z[2 * c + 1];
        const float mag = sqrtf(re * re + im * im);
        if (mag > 0.0f) {
            const float inv = dY[c] / mag;
            dZ[2 * c]     = re * inv;
            dZ[2 * c + 1] = im * inv;
        } else {
            dZ[2 * c]     = 0.0f;
            dZ[2 * c + 1] = 0.0f;
        }
    }
}

// y = atan2(im, re). z is (R,2C), y is (R,C).
__global__ void complex_angle_kernel(const float* __restrict__ z,
                                     long long bins, float* __restrict__ y) {
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         c < bins; c += (long long)blockDim.x * gridDim.x) {
        y[c] = atan2f(z[2 * c + 1], z[2 * c]);
    }
}

// y = mag * exp(i*phase). mag/phase are (R,C), y is (R,2C).
__global__ void complex_from_polar_kernel(const float* __restrict__ mag,
                                          const float* __restrict__ phase,
                                          long long bins,
                                          float* __restrict__ y) {
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         c < bins; c += (long long)blockDim.x * gridDim.x) {
        const float m = mag[c], p = phase[c];
        y[2 * c]     = m * cosf(p);
        y[2 * c + 1] = m * sinf(p);
    }
}

// ─── DFT kernels (one thread per output bin / sample) ───────────────────────

// Complex->complex DFT, one thread per (row, bin). sign = -1 (fft) / +1 (ifft).
// scale applied to the result (1 for fft, 1/N for ifft).
__global__ void dft_complex_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int R, int N, double sign, double scale) {
    const long long total = (long long)R * N;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int r = static_cast<int>(g / N);
        const int k = static_cast<int>(g % N);
        const float* xr = x + (long long)r * 2 * N;
        const double w = sign * kTwoPi * k / N;
        double accr = 0.0, acci = 0.0;
        for (int n = 0; n < N; ++n) {
            double s, c;
            sincos(w * n, &s, &c);                 // exp(i*ang) = c + i*s
            const double xre = xr[2 * n], xim = xr[2 * n + 1];
            accr += xre * c - xim * s;
            acci += xre * s + xim * c;
        }
        float* yr = y + (long long)r * 2 * N;
        yr[2 * k]     = static_cast<float>(accr * scale);
        yr[2 * k + 1] = static_cast<float>(acci * scale);
    }
}

// rfft: real (R,L) -> half-spectrum (R,2C), C = L/2+1. One thread per (row,bin).
__global__ void rfft_kernel(const float* __restrict__ x, float* __restrict__ y,
                            int R, int L, int C) {
    const long long total = (long long)R * C;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int r = static_cast<int>(g / C);
        const int k = static_cast<int>(g % C);
        const float* xr = x + (long long)r * L;
        const double w = -kTwoPi * k / L;          // forward sign
        double accr = 0.0, acci = 0.0;
        for (int n = 0; n < L; ++n) {
            double s, c;
            sincos(w * n, &s, &c);
            const double xn = xr[n];               // real input
            accr += xn * c;
            acci += xn * s;
        }
        float* yr = y + (long long)r * 2 * C;
        yr[2 * k]     = static_cast<float>(accr);
        yr[2 * k + 1] = static_cast<float>(acci);
    }
}

// irfft: half-spectrum (R,2C) -> real (R,L), * 1/L. One thread per (row,sample).
// The full Hermitian spectrum is reconstructed on the fly: bin k>=C is
// conj(bin L-k).
__global__ void irfft_kernel(const float* __restrict__ x,
                             float* __restrict__ y, int R, int L, int C) {
    const long long total = (long long)R * L;
    const double invL = 1.0 / static_cast<double>(L);
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int r = static_cast<int>(g / L);
        const int n = static_cast<int>(g % L);
        const float* xr = x + (long long)r * 2 * C;
        const double w = kTwoPi * n / L;           // inverse sign
        double acc = 0.0;
        for (int k = 0; k < L; ++k) {
            double fre, fim;
            if (k < C) {
                fre = xr[2 * k];
                fim = xr[2 * k + 1];
            } else {                               // Hermitian mirror
                const int m = L - k;
                fre = xr[2 * m];
                fim = -xr[2 * m + 1];
            }
            double s, c;
            sincos(w * k, &s, &c);
            acc += fre * c - fim * s;              // Re(full[k]*exp(i*ang))
        }
        y[(long long)r * L + n] = static_cast<float>(acc * invL);
    }
}

// rfft_backward: adjoint of rfft. dY (R,2C) -> dX real (R,L), overwritten.
// dX[n] = Re( sum_{k=0}^{C-1} dY[k] * exp(+i 2*pi k n / L) ).
__global__ void rfft_backward_kernel(const float* __restrict__ dY,
                                     float* __restrict__ dX,
                                     int R, int L, int C) {
    const long long total = (long long)R * L;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int r = static_cast<int>(g / L);
        const int n = static_cast<int>(g % L);
        const float* gr = dY + (long long)r * 2 * C;
        const double w = kTwoPi * n / L;
        double acc = 0.0;
        for (int k = 0; k < C; ++k) {
            double s, c;
            sincos(w * k, &s, &c);
            const double gre = gr[2 * k], gim = gr[2 * k + 1];
            acc += gre * c - gim * s;
        }
        dX[(long long)r * L + n] = static_cast<float>(acc);
    }
}

// irfft_backward: adjoint of irfft. dY real (R,L) -> dX (R,2C), overwritten.
// dX[k] = (s_k / L) * conj( forward_DFT(dY)[k] ); s_k = 1 at k=0 and (L even)
// k=L/2, else 2.
__global__ void irfft_backward_kernel(const float* __restrict__ dY,
                                      float* __restrict__ dX,
                                      int R, int L, int C) {
    const long long total = (long long)R * C;
    const double invL = 1.0 / static_cast<double>(L);
    const bool even = (L % 2 == 0);
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total; g += (long long)blockDim.x * gridDim.x) {
        const int r = static_cast<int>(g / C);
        const int k = static_cast<int>(g % C);
        const float* gr = dY + (long long)r * L;
        const double w = -kTwoPi * k / L;          // forward DFT sign
        double sre = 0.0, sim = 0.0;
        for (int n = 0; n < L; ++n) {
            double s, c;
            sincos(w * n, &s, &c);
            const double gn = gr[n];
            sre += gn * c;
            sim += gn * s;
        }
        double sk = 2.0;
        if (k == 0) sk = 1.0;
        if (even && k == L / 2) sk = 1.0;
        const double scale = sk * invL;
        float* dxr = dX + (long long)r * 2 * C;
        dxr[2 * k]     = static_cast<float>(scale * sre);
        dxr[2 * k + 1] = static_cast<float>(scale * sim);
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Complex elementwise wrappers
// ════════════════════════════════════════════════════════════════════════════

void complex_mul(const ::brotensor::Tensor& a, const ::brotensor::Tensor& b,
                 ::brotensor::Tensor& y) {
    require_fp32("complex_mul", a, "a");
    require_fp32("complex_mul", b, "b");
    if (a.rows != b.rows || a.cols != b.cols) {
        fail("complex_mul", "a and b must have identical shape");
    }
    if (a.cols % 2 != 0) {
        fail("complex_mul", "cols must be even (interleaved [re,im] layout)");
    }
    if (y.rows != a.rows || y.cols != a.cols) y.resize(a.rows, a.cols);
    const long long pairs = static_cast<long long>(a.size()) / 2;
    if (pairs == 0) return;
    complex_mul_kernel<<<fft_grid(pairs), FFT_BLOCK>>>(
        static_cast<const float*>(a.data), static_cast<const float*>(b.data),
        pairs, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void complex_mul_backward(const ::brotensor::Tensor& a,
                          const ::brotensor::Tensor& b,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dA, ::brotensor::Tensor& dB) {
    require_fp32("complex_mul_backward", a, "a");
    require_fp32("complex_mul_backward", b, "b");
    require_fp32("complex_mul_backward", dY, "dY");
    require_fp32("complex_mul_backward", dA, "dA");
    require_fp32("complex_mul_backward", dB, "dB");
    if (a.rows != b.rows || a.cols != b.cols) {
        fail("complex_mul_backward", "a and b must have identical shape");
    }
    if (dY.rows != a.rows || dY.cols != a.cols) {
        fail("complex_mul_backward", "dY must match a / b shape");
    }
    if (dA.rows != a.rows || dA.cols != a.cols) {
        fail("complex_mul_backward", "dA must be pre-sized to a's shape");
    }
    if (dB.rows != a.rows || dB.cols != a.cols) {
        fail("complex_mul_backward", "dB must be pre-sized to b's shape");
    }
    const long long pairs = static_cast<long long>(a.size()) / 2;
    if (pairs == 0) return;
    complex_mul_backward_kernel<<<fft_grid(pairs), FFT_BLOCK>>>(
        static_cast<const float*>(a.data), static_cast<const float*>(b.data),
        static_cast<const float*>(dY.data),
        pairs, static_cast<float*>(dA.data), static_cast<float*>(dB.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void complex_abs(const ::brotensor::Tensor& z, ::brotensor::Tensor& y) {
    require_fp32("complex_abs", z, "z");
    if (z.cols % 2 != 0) {
        fail("complex_abs", "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (y.rows != z.rows || y.cols != C) y.resize(z.rows, C);
    const long long bins = static_cast<long long>(z.rows) * C;
    if (bins == 0) return;
    complex_abs_kernel<<<fft_grid(bins), FFT_BLOCK>>>(
        static_cast<const float*>(z.data), bins, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void complex_abs_backward(const ::brotensor::Tensor& z,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dZ) {
    require_fp32("complex_abs_backward", z, "z");
    require_fp32("complex_abs_backward", dY, "dY");
    if (z.cols % 2 != 0) {
        fail("complex_abs_backward",
             "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (dY.rows != z.rows || dY.cols != C) {
        fail("complex_abs_backward", "dY must be the real (R, C) magnitude grad");
    }
    if (dZ.rows != z.rows || dZ.cols != z.cols) dZ.resize(z.rows, z.cols);
    const long long bins = static_cast<long long>(z.rows) * C;
    if (bins == 0) return;
    complex_abs_backward_kernel<<<fft_grid(bins), FFT_BLOCK>>>(
        static_cast<const float*>(z.data), static_cast<const float*>(dY.data),
        bins, static_cast<float*>(dZ.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void complex_angle(const ::brotensor::Tensor& z, ::brotensor::Tensor& y) {
    require_fp32("complex_angle", z, "z");
    if (z.cols % 2 != 0) {
        fail("complex_angle", "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (y.rows != z.rows || y.cols != C) y.resize(z.rows, C);
    const long long bins = static_cast<long long>(z.rows) * C;
    if (bins == 0) return;
    complex_angle_kernel<<<fft_grid(bins), FFT_BLOCK>>>(
        static_cast<const float*>(z.data), bins, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void complex_from_polar(const ::brotensor::Tensor& mag,
                        const ::brotensor::Tensor& phase,
                        ::brotensor::Tensor& y) {
    require_fp32("complex_from_polar", mag, "mag");
    require_fp32("complex_from_polar", phase, "phase");
    if (mag.rows != phase.rows || mag.cols != phase.cols) {
        fail("complex_from_polar", "mag and phase must have identical shape");
    }
    const int C = mag.cols;
    if (y.rows != mag.rows || y.cols != 2 * C) y.resize(mag.rows, 2 * C);
    const long long bins = static_cast<long long>(mag.rows) * C;
    if (bins == 0) return;
    complex_from_polar_kernel<<<fft_grid(bins), FFT_BLOCK>>>(
        static_cast<const float*>(mag.data),
        static_cast<const float*>(phase.data),
        bins, static_cast<float*>(y.data));
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════════════
//  fft / ifft / rfft / irfft + adjoints
// ════════════════════════════════════════════════════════════════════════════

void fft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32("fft", x, "x");
    if (x.cols % 2 != 0) fail("fft", "x.cols must be even (interleaved [re,im])");
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (x.size() == 0) return;
    const int N = x.cols / 2;
    dft_complex_kernel<<<fft_grid((long long)x.rows * N), FFT_BLOCK>>>(
        static_cast<const float*>(x.data), static_cast<float*>(y.data),
        x.rows, N, -1.0, 1.0);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void ifft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32("ifft", x, "x");
    if (x.cols % 2 != 0) fail("ifft", "x.cols must be even (interleaved [re,im])");
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (x.size() == 0) return;
    const int N = x.cols / 2;
    const double inv = (N > 0) ? 1.0 / static_cast<double>(N) : 1.0;
    dft_complex_kernel<<<fft_grid((long long)x.rows * N), FFT_BLOCK>>>(
        static_cast<const float*>(x.data), static_cast<float*>(y.data),
        x.rows, N, +1.0, inv);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rfft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32("rfft", x, "x");
    const int L = x.cols;
    if (L == 0) fail("rfft", "signal length L (x.cols) must be >= 1");
    const int C = L / 2 + 1;
    if (y.rows != x.rows || y.cols != 2 * C) y.resize(x.rows, 2 * C);
    if (x.size() == 0) return;
    rfft_kernel<<<fft_grid((long long)x.rows * C), FFT_BLOCK>>>(
        static_cast<const float*>(x.data), static_cast<float*>(y.data),
        x.rows, L, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void irfft(const ::brotensor::Tensor& x, int L, ::brotensor::Tensor& y) {
    require_fp32("irfft", x, "x");
    if (x.cols % 2 != 0) {
        fail("irfft", "x.cols must be even (interleaved [re,im] layout)");
    }
    const int C = x.cols / 2;
    if (L <= 0) fail("irfft", "output length L must be >= 1");
    if (C != L / 2 + 1) fail("irfft", "half-spectrum bin count must equal L/2+1");
    if (y.rows != x.rows || y.cols != L) y.resize(x.rows, L);
    if (x.size() == 0) return;
    irfft_kernel<<<fft_grid((long long)x.rows * L), FFT_BLOCK>>>(
        static_cast<const float*>(x.data), static_cast<float*>(y.data),
        x.rows, L, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void rfft_backward(const ::brotensor::Tensor& dY, int L,
                   ::brotensor::Tensor& dX) {
    require_fp32("rfft_backward", dY, "dY");
    if (dY.cols % 2 != 0) {
        fail("rfft_backward", "dY.cols must be even (interleaved [re,im] layout)");
    }
    const int C = dY.cols / 2;
    if (L <= 0) fail("rfft_backward", "signal length L must be >= 1");
    if (C != L / 2 + 1) fail("rfft_backward", "dY bin count must equal L/2+1");
    if (dX.rows != dY.rows || dX.cols != L) dX.resize(dY.rows, L);
    if (dY.size() == 0) return;
    rfft_backward_kernel<<<fft_grid((long long)dY.rows * L), FFT_BLOCK>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
        dY.rows, L, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void irfft_backward(const ::brotensor::Tensor& dY, ::brotensor::Tensor& dX) {
    require_fp32("irfft_backward", dY, "dY");
    const int L = dY.cols;
    if (L == 0) fail("irfft_backward", "dY length L (dY.cols) must be >= 1");
    const int C = L / 2 + 1;
    if (dX.rows != dY.rows || dX.cols != 2 * C) dX.resize(dY.rows, 2 * C);
    if (dY.size() == 0) return;
    irfft_backward_kernel<<<fft_grid((long long)dY.rows * C), FFT_BLOCK>>>(
        static_cast<const float*>(dY.data), static_cast<float*>(dX.data),
        dY.rows, L, C);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_fft(::brotensor::detail::OpsVTable& v) {
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
}

} // namespace brotensor::detail::cuda
