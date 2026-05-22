// ─── CPU spectral / FFT core (brosoundml CHUNK 1) ──────────────────────────
//
// Hand-rolled FFT for the CPU backend. No external libraries. FP32-only,
// matching the CPU backend's FP32-only contract.
//
// Ops implemented here:
//   complex_mul / complex_abs / complex_angle / complex_from_polar
//   complex_mul_backward / complex_abs_backward
//   fft / ifft         — complex -> complex, one signal per tensor row
//   rfft / irfft       — real (R,L) <-> complex (R, 2*(L/2+1))
//   rfft_backward / irfft_backward — adjoints of rfft / irfft
//
// ── Complex layout ─────────────────────────────────────────────────────────
// A complex tensor is a regular FP32 tensor with the bin axis stored
// interleaved [re, im, re, im, ...]. A complex spectrum of C bins over R rows
// is an (R, 2*C) FP32 tensor. There is no new Dtype.
//
// ── FFT algorithm ──────────────────────────────────────────────────────────
// Mixed-radix Cooley-Tukey (radix 2/3/5/7) handles sizes whose prime
// factorisation only uses small primes — this covers Whisper's n_fft = 400
// (= 2^4 * 5^2). Any remaining factor (a large or genuinely prime factor) is
// transformed with a Bluestein chirp-z transform, which reduces an
// arbitrary-length DFT to a power-of-two convolution. The whole thing is
// therefore correct for *every* length >= 1, including primes.
//
// ── Normalisation ──────────────────────────────────────────────────────────
// "backward" convention (numpy default): the forward transform is unscaled,
// the inverse transform is scaled by 1/N.
//
// ── Gradient design (the linear-transform adjoints) ────────────────────────
// fft / ifft / rfft / irfft are all linear maps, so the backward of each is
// the adjoint (conjugate transpose) of its forward matrix applied to the
// upstream gradient. We deliberately keep the vtable minimal:
//
//   * fft / ifft are complex->complex and self-similar. The adjoint of the
//     length-N forward DFT matrix F is F^H = conj(F) = N * F^{-1}. With this
//     library's transforms that means:
//         grad_x(fft)  = N * ifft(grad_y)
//         grad_x(ifft) = (1/N) * fft(grad_y)
//     Both adjoints are an *existing* transform composed with a scalar, so we
//     do NOT add fft_backward / ifft_backward rows. Training code spells the
//     gradient as `ifft(g); scale_inplace(g, N)` (or the ifft dual). This is
//     documented on the fft / ifft declarations in ops.h.
//
//   * rfft / irfft are NOT mutual adjoints. rfft maps a real length-L signal
//     to its non-redundant half-spectrum (L/2+1 bins); irfft maps a
//     half-spectrum back to a real signal assuming Hermitian symmetry AND
//     applies the 1/L inverse scaling. rfft_backward is the plain adjoint of
//     the truncated DFT matrix (no bin weighting — rfft does no folding).
//     irfft_backward carries the 1/L scaling and the interior-bin weighting
//     that irfft's Hermitian folding implies. Getting that weighting wrong is
//     a silent training bug, so both are explicit ops rather than something
//     callers reconstruct. They are the minimal correct set for the gradient
//     path of a multi-resolution STFT loss.
//
// This file ports cleanly to CUDA / Metal later (the math is backend-neutral);
// only the host_f32 accessors are CPU-specific.
//
// The mixed-radix + Bluestein transform engine itself lives in the shared
// header detail/cpu/fft_core.h so the STFT / iSTFT ops (stft.cpp) reuse the
// exact same DFT instead of copy-pasting it.

#include <brotensor/detail/cpu/fft_core.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

// Pull the shared FFT-core internals (Cd, dft_1d, complex-row I/O) into scope.
using fftcore::Cd;
using fftcore::dft_1d;
using fftcore::load_complex_row;
using fftcore::store_complex_row;

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

} // namespace

// ════════════════════════════════════════════════════════════════════════════
//  Complex elementwise ops
// ════════════════════════════════════════════════════════════════════════════

// y = a * b, complex elementwise. a, b, y are interleaved-complex (R, 2*C).
void complex_mul(const ::brotensor::Tensor& a, const ::brotensor::Tensor& b,
                 ::brotensor::Tensor& y) {
    require_fp32_host("complex_mul", a, "a");
    require_fp32_host("complex_mul", b, "b");
    if (a.rows != b.rows || a.cols != b.cols) {
        fail("complex_mul", "a and b must have identical shape");
    }
    if (a.cols % 2 != 0) {
        fail("complex_mul", "cols must be even (interleaved [re,im] layout)");
    }
    if (y.rows != a.rows || y.cols != a.cols) y.resize(a.rows, a.cols);
    const int n = a.size();
    if (n == 0) return;
    const float* ap = a.host_f32();
    const float* bp = b.host_f32();
    float* yp = y.host_f32_mut();
    for (int i = 0; i < n; i += 2) {
        const float ar = ap[i], ai = ap[i + 1];
        const float br = bp[i], bi = bp[i + 1];
        yp[i]     = ar * br - ai * bi;
        yp[i + 1] = ar * bi + ai * br;
    }
}

// Backward of complex_mul. y = a * b ⇒ (Wirtinger / real-pair gradient)
//   dA = dY * conj(b),   dB = dY * conj(a).
// dA and dB are *accumulated into* — the caller zeros them (mirrors the
// accumulation contract used by linear_backward / matmul_backward).
void complex_mul_backward(const ::brotensor::Tensor& a,
                          const ::brotensor::Tensor& b,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dA, ::brotensor::Tensor& dB) {
    require_fp32_host("complex_mul_backward", a, "a");
    require_fp32_host("complex_mul_backward", b, "b");
    require_fp32_host("complex_mul_backward", dY, "dY");
    require_fp32_host("complex_mul_backward", dA, "dA");
    require_fp32_host("complex_mul_backward", dB, "dB");
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
    const int n = a.size();
    if (n == 0) return;
    const float* ap = a.host_f32();
    const float* bp = b.host_f32();
    const float* gp = dY.host_f32();
    float* dap = dA.host_f32_mut();
    float* dbp = dB.host_f32_mut();
    for (int i = 0; i < n; i += 2) {
        const float gr = gp[i], gi = gp[i + 1];
        const float ar = ap[i], ai = ap[i + 1];
        const float br = bp[i], bi = bp[i + 1];
        // dA = dY * conj(b): (gr+igi)(br-ibi)
        dap[i]     += gr * br + gi * bi;
        dap[i + 1] += gi * br - gr * bi;
        // dB = dY * conj(a): (gr+igi)(ar-iai)
        dbp[i]     += gr * ar + gi * ai;
        dbp[i + 1] += gi * ar - gr * ai;
    }
}

// y = |z|, real magnitude per complex bin. Input z is interleaved-complex
// (R, 2*C); output y is REAL (R, C).
void complex_abs(const ::brotensor::Tensor& z, ::brotensor::Tensor& y) {
    require_fp32_host("complex_abs", z, "z");
    if (z.cols % 2 != 0) {
        fail("complex_abs", "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (y.rows != z.rows || y.cols != C) y.resize(z.rows, C);
    if (z.size() == 0) return;
    const float* zp = z.host_f32();
    float* yp = y.host_f32_mut();
    for (int r = 0; r < z.rows; ++r) {
        const float* zr = zp + static_cast<std::size_t>(r) * z.cols;
        float* yr = yp + static_cast<std::size_t>(r) * C;
        for (int c = 0; c < C; ++c) {
            const float re = zr[2 * c], im = zr[2 * c + 1];
            yr[c] = std::sqrt(re * re + im * im);
        }
    }
}

// Backward of complex_abs. With r = |z| = sqrt(re^2 + im^2):
//   d|z|/d(re) = re / r,   d|z|/d(im) = im / r.
// dZ is interleaved-complex (R, 2*C), *overwritten* (matches the GPU
// activation-backward convention — backward writes dZ directly). At r == 0
// the gradient is set to 0 (the magnitude is non-differentiable there;
// 0 is the conventional choice).
void complex_abs_backward(const ::brotensor::Tensor& z,
                          const ::brotensor::Tensor& dY,
                          ::brotensor::Tensor& dZ) {
    require_fp32_host("complex_abs_backward", z, "z");
    require_fp32_host("complex_abs_backward", dY, "dY");
    if (z.cols % 2 != 0) {
        fail("complex_abs_backward",
             "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (dY.rows != z.rows || dY.cols != C) {
        fail("complex_abs_backward", "dY must be the real (R, C) magnitude grad");
    }
    if (dZ.rows != z.rows || dZ.cols != z.cols) dZ.resize(z.rows, z.cols);
    if (z.size() == 0) return;
    const float* zp = z.host_f32();
    const float* gp = dY.host_f32();
    float* dzp = dZ.host_f32_mut();
    for (int r = 0; r < z.rows; ++r) {
        const float* zr = zp + static_cast<std::size_t>(r) * z.cols;
        const float* gr = gp + static_cast<std::size_t>(r) * C;
        float* dzr = dzp + static_cast<std::size_t>(r) * z.cols;
        for (int c = 0; c < C; ++c) {
            const float re = zr[2 * c], im = zr[2 * c + 1];
            const float mag = std::sqrt(re * re + im * im);
            if (mag > 0.0f) {
                const float inv = gr[c] / mag;
                dzr[2 * c]     = re * inv;
                dzr[2 * c + 1] = im * inv;
            } else {
                dzr[2 * c]     = 0.0f;
                dzr[2 * c + 1] = 0.0f;
            }
        }
    }
}

// y = atan2(im, re), the phase angle per complex bin, in radians (-pi, pi].
// Input z is interleaved-complex (R, 2*C); output y is REAL (R, C). No
// backward — phase is rarely used in a differentiable loss and atan2 is
// non-differentiable at the origin; add a backward later if a consumer needs
// one.
void complex_angle(const ::brotensor::Tensor& z, ::brotensor::Tensor& y) {
    require_fp32_host("complex_angle", z, "z");
    if (z.cols % 2 != 0) {
        fail("complex_angle", "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (y.rows != z.rows || y.cols != C) y.resize(z.rows, C);
    if (z.size() == 0) return;
    const float* zp = z.host_f32();
    float* yp = y.host_f32_mut();
    for (int r = 0; r < z.rows; ++r) {
        const float* zr = zp + static_cast<std::size_t>(r) * z.cols;
        float* yr = yp + static_cast<std::size_t>(r) * C;
        for (int c = 0; c < C; ++c) {
            yr[c] = std::atan2(zr[2 * c + 1], zr[2 * c]);
        }
    }
}

// y = mag * exp(i*phase) — build a complex tensor from polar components.
// mag and phase are REAL (R, C); output y is interleaved-complex (R, 2*C).
//   y.re = mag * cos(phase),   y.im = mag * sin(phase).
// Inverse of (complex_abs, complex_angle) taken together.
void complex_from_polar(const ::brotensor::Tensor& mag,
                        const ::brotensor::Tensor& phase,
                        ::brotensor::Tensor& y) {
    require_fp32_host("complex_from_polar", mag, "mag");
    require_fp32_host("complex_from_polar", phase, "phase");
    if (mag.rows != phase.rows || mag.cols != phase.cols) {
        fail("complex_from_polar", "mag and phase must have identical shape");
    }
    const int C = mag.cols;
    if (y.rows != mag.rows || y.cols != 2 * C) y.resize(mag.rows, 2 * C);
    if (mag.size() == 0) return;
    const float* mp = mag.host_f32();
    const float* pp = phase.host_f32();
    float* yp = y.host_f32_mut();
    for (int r = 0; r < mag.rows; ++r) {
        const float* mr = mp + static_cast<std::size_t>(r) * C;
        const float* pr = pp + static_cast<std::size_t>(r) * C;
        float* yr = yp + static_cast<std::size_t>(r) * (2 * C);
        for (int c = 0; c < C; ++c) {
            yr[2 * c]     = mr[c] * std::cos(pr[c]);
            yr[2 * c + 1] = mr[c] * std::sin(pr[c]);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Complex <-> complex FFT / IFFT
// ════════════════════════════════════════════════════════════════════════════

// fft: forward DFT, one signal per row. x and y are interleaved-complex
// (R, 2*N). "backward" normalisation — the forward transform is unscaled.
void fft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32_host("fft", x, "x");
    if (x.cols % 2 != 0) {
        fail("fft", "x.cols must be even (interleaved [re,im] layout)");
    }
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (x.size() == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    std::vector<Cd> in, out;
    for (int r = 0; r < x.rows; ++r) {
        load_complex_row(xp, r, x.cols, in);
        dft_1d(in, out, -1);
        store_complex_row(yp, r, x.cols, out);
    }
}

// ifft: inverse DFT, one signal per row. x and y are interleaved-complex
// (R, 2*N). "backward" normalisation — the inverse transform is scaled by 1/N.
void ifft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32_host("ifft", x, "x");
    if (x.cols % 2 != 0) {
        fail("ifft", "x.cols must be even (interleaved [re,im] layout)");
    }
    if (y.rows != x.rows || y.cols != x.cols) y.resize(x.rows, x.cols);
    if (x.size() == 0) return;
    const int N = x.cols / 2;
    const double inv = (N > 0) ? 1.0 / static_cast<double>(N) : 1.0;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    std::vector<Cd> in, out;
    for (int r = 0; r < x.rows; ++r) {
        load_complex_row(xp, r, x.cols, in);
        dft_1d(in, out, +1);
        for (auto& v : out) { v.re *= inv; v.im *= inv; }
        store_complex_row(yp, r, x.cols, out);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Real <-> complex rfft / irfft
// ════════════════════════════════════════════════════════════════════════════

// rfft: real-input FFT. x is REAL (R, L); y is the non-redundant
// half-spectrum, interleaved-complex (R, 2*(L/2+1)). "backward" normalisation
// (forward unscaled). Only bins 0 .. L/2 are stored — the remaining bins are
// the conjugates of these by Hermitian symmetry of a real signal.
void rfft(const ::brotensor::Tensor& x, ::brotensor::Tensor& y) {
    require_fp32_host("rfft", x, "x");
    const int L = x.cols;
    if (L == 0) {
        fail("rfft", "signal length L (x.cols) must be >= 1");
    }
    const int C = L / 2 + 1;
    if (y.rows != x.rows || y.cols != 2 * C) y.resize(x.rows, 2 * C);
    if (x.size() == 0) return;
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    std::vector<Cd> in(static_cast<std::size_t>(L)), out;
    for (int r = 0; r < x.rows; ++r) {
        const float* xr = xp + static_cast<std::size_t>(r) * L;
        for (int n = 0; n < L; ++n) {
            in[static_cast<std::size_t>(n)] = {static_cast<double>(xr[n]), 0.0};
        }
        dft_1d(in, out, -1);
        float* yr = yp + static_cast<std::size_t>(r) * (2 * C);
        for (int c = 0; c < C; ++c) {
            yr[2 * c]     = static_cast<float>(out[static_cast<std::size_t>(c)].re);
            yr[2 * c + 1] = static_cast<float>(out[static_cast<std::size_t>(c)].im);
        }
    }
}

// irfft: inverse real FFT. x is a half-spectrum, interleaved-complex
// (R, 2*(L/2+1)); y is the reconstructed REAL signal (R, L). "backward"
// normalisation (scaled by 1/L). The full Hermitian-symmetric spectrum is
// rebuilt from the stored half (bin L-k = conj(bin k)) before the inverse
// transform; the DC and (for even L) Nyquist bins have no conjugate partner.
//
// `L` must be passed explicitly: a half-spectrum with C bins is ambiguous
// between L = 2*(C-1) (even) and L = 2*C-1 (odd).
void irfft(const ::brotensor::Tensor& x, int L, ::brotensor::Tensor& y) {
    require_fp32_host("irfft", x, "x");
    if (x.cols % 2 != 0) {
        fail("irfft", "x.cols must be even (interleaved [re,im] layout)");
    }
    const int C = x.cols / 2;
    if (L <= 0) {
        fail("irfft", "output length L must be >= 1");
    }
    if (C != L / 2 + 1) {
        fail("irfft", "half-spectrum bin count must equal L/2+1");
    }
    if (y.rows != x.rows || y.cols != L) y.resize(x.rows, L);
    if (x.size() == 0) return;
    const double inv = 1.0 / static_cast<double>(L);
    const float* xp = x.host_f32();
    float* yp = y.host_f32_mut();
    std::vector<Cd> full(static_cast<std::size_t>(L)), out;
    for (int r = 0; r < x.rows; ++r) {
        const float* xr = xp + static_cast<std::size_t>(r) * x.cols;
        // Stored half-spectrum into bins 0 .. C-1.
        for (int c = 0; c < C; ++c) {
            full[static_cast<std::size_t>(c)] = {static_cast<double>(xr[2 * c]),
                                                 static_cast<double>(xr[2 * c + 1])};
        }
        // Hermitian mirror: bin L-k = conj(bin k) for k = 1 .. L-C.
        for (int k = 1; k < L - C + 1; ++k) {
            const Cd c = full[static_cast<std::size_t>(k)];
            full[static_cast<std::size_t>(L - k)] = {c.re, -c.im};
        }
        dft_1d(full, out, +1);
        float* yr = yp + static_cast<std::size_t>(r) * L;
        for (int n = 0; n < L; ++n) {
            yr[n] = static_cast<float>(out[static_cast<std::size_t>(n)].re * inv);
        }
    }
}

// ── rfft_backward — adjoint of rfft ────────────────────────────────────────
//
// rfft maps a real length-L signal x to its non-redundant half-spectrum Y
// (C = L/2+1 complex bins): Y[k] = sum_n x[n] * exp(-i 2*pi k n / L). This is
// just the first C rows of the length-L DFT matrix applied to a real vector.
//
// For a real-valued loss formed directly on the spectrum,
//   loss = sum_{k=0}^{C-1} ( dY[k].re * Y[k].re + dY[k].im * Y[k].im ),
// the gradient w.r.t. the real signal is the plain conjugate transpose of
// that truncated DFT matrix — NO conjugate-pair weighting (the doubling lives
// in irfft / irfft_backward, which fold the Hermitian half back; rfft does no
// folding so its adjoint does none either):
//
//   dX[n] = sum_{k=0}^{C-1} ( dY[k].re * cos(2*pi k n / L)
//                             - dY[k].im * sin(2*pi k n / L) )
//         = Re( sum_{k=0}^{C-1} dY[k] * exp(+i 2*pi k n / L) ).
//
// Computed by zero-padding dY to length L and running an inverse-sign
// (sign = +1) unscaled DFT, then taking the real part. dX is *overwritten*.
void rfft_backward(const ::brotensor::Tensor& dY, int L,
                   ::brotensor::Tensor& dX) {
    require_fp32_host("rfft_backward", dY, "dY");
    if (dY.cols % 2 != 0) {
        fail("rfft_backward", "dY.cols must be even (interleaved [re,im] layout)");
    }
    const int C = dY.cols / 2;
    if (L <= 0) {
        fail("rfft_backward", "signal length L must be >= 1");
    }
    if (C != L / 2 + 1) {
        fail("rfft_backward", "dY bin count must equal L/2+1");
    }
    if (dX.rows != dY.rows || dX.cols != L) dX.resize(dY.rows, L);
    if (dY.size() == 0) return;
    const float* gp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    std::vector<Cd> spec(static_cast<std::size_t>(L)), out;
    for (int r = 0; r < dY.rows; ++r) {
        const float* gr = gp + static_cast<std::size_t>(r) * dY.cols;
        // Zero-padded length-L spectrum: bins 0..C-1 carry dY[k], rest zero.
        for (int k = 0; k < L; ++k) spec[static_cast<std::size_t>(k)] = Cd{};
        for (int k = 0; k < C; ++k) {
            spec[static_cast<std::size_t>(k)] =
                {static_cast<double>(gr[2 * k]),
                 static_cast<double>(gr[2 * k + 1])};
        }
        // dX[n] = Re( sum_k spec[k] * exp(+i 2pi k n / L) ).
        dft_1d(spec, out, +1);
        float* dxr = dxp + static_cast<std::size_t>(r) * L;
        for (int n = 0; n < L; ++n) {
            dxr[n] = static_cast<float>(out[static_cast<std::size_t>(n)].re);
        }
    }
}

// ── irfft_backward — adjoint of irfft ──────────────────────────────────────
//
// irfft maps a half-spectrum X (C = L/2+1 complex bins) to a real signal y
// of length L, with the 1/L inverse scaling. The adjoint maps the upstream
// real gradient dY (real (R, L)) back to the half-spectrum gradient dX
// (interleaved-complex (R, 2*C)).
//
// Forward (per output sample n):
//   y[n] = (1/L) * [ X[0].re
//                    + sum_{k=1}^{C-1} 2 * Re( X[k] * exp(i 2pi k n / L) )
//                    - (L even ? X[L/2].re : 0) ]   (Nyquist counted once)
// so the adjoint, per stored bin k:
//   dX[k].re = (s_k / L) * sum_n dY[n] * cos(2pi k n / L)
//   dX[k].im = -(s_k / L) * sum_n dY[n] * sin(2pi k n / L)
//   s_k = 1 for k = 0 and k = L/2 (L even); 2 otherwise.
//
// dX is *overwritten*. This op is the transpose of rfft_backward.
void irfft_backward(const ::brotensor::Tensor& dY,
                    ::brotensor::Tensor& dX) {
    require_fp32_host("irfft_backward", dY, "dY");
    const int L = dY.cols;
    if (L == 0) {
        fail("irfft_backward", "dY length L (dY.cols) must be >= 1");
    }
    const int C = L / 2 + 1;
    if (dX.rows != dY.rows || dX.cols != 2 * C) dX.resize(dY.rows, 2 * C);
    if (dY.size() == 0) return;
    const bool even = (L % 2 == 0);
    const double invL = 1.0 / static_cast<double>(L);
    const float* gp = dY.host_f32();
    float* dxp = dX.host_f32_mut();
    // dX[k] = (s_k / L) * conj( forward_DFT(dY)[k] ), for k = 0 .. C-1.
    std::vector<Cd> in(static_cast<std::size_t>(L)), spec;
    for (int r = 0; r < dY.rows; ++r) {
        const float* gr = gp + static_cast<std::size_t>(r) * L;
        for (int n = 0; n < L; ++n) {
            in[static_cast<std::size_t>(n)] = {static_cast<double>(gr[n]), 0.0};
        }
        dft_1d(in, spec, -1);  // spec[k] = sum_n dY[n] exp(-i 2pi k n / L)
        float* dxr = dxp + static_cast<std::size_t>(r) * (2 * C);
        for (int k = 0; k < C; ++k) {
            double s = 2.0;
            if (k == 0) s = 1.0;
            if (even && k == L / 2) s = 1.0;
            const double scale = s * invL;
            // cos sum = spec[k].re; -sin sum = spec[k].im (already conj-form).
            dxr[2 * k]     = static_cast<float>(scale * spec[static_cast<std::size_t>(k)].re);
            dxr[2 * k + 1] = static_cast<float>(scale * spec[static_cast<std::size_t>(k)].im);
        }
    }
}

} // namespace brotensor::detail::cpu
