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

#include <brotensor/tensor.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ─── tiny complex helper (double precision internally for accuracy) ────────
struct Cd {
    double re = 0.0;
    double im = 0.0;
};

inline Cd operator+(Cd a, Cd b) { return {a.re + b.re, a.im + b.im}; }
inline Cd operator-(Cd a, Cd b) { return {a.re - b.re, a.im - b.im}; }
inline Cd operator*(Cd a, Cd b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

[[noreturn]] void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

// ─── twiddle-factor cache ──────────────────────────────────────────────────
//
// w[k] = exp(sign * 2*pi*i * k / N) for k in [0, N). `sign` is -1 for the
// forward transform, +1 for the inverse. Computed once per (N, sign).
std::vector<Cd> make_twiddles(int N, int sign) {
    std::vector<Cd> w(static_cast<std::size_t>(N));
    const double s = static_cast<double>(sign) * 2.0 * kPi / static_cast<double>(N);
    for (int k = 0; k < N; ++k) {
        w[static_cast<std::size_t>(k)] = {std::cos(s * k), std::sin(s * k)};
    }
    return w;
}

// ─── naive DFT (used as the base case + by Bluestein's small kernels) ──────
void dft_naive(const std::vector<Cd>& in, std::vector<Cd>& out, int sign) {
    const int N = static_cast<int>(in.size());
    out.assign(static_cast<std::size_t>(N), Cd{});
    const double s = static_cast<double>(sign) * 2.0 * kPi / static_cast<double>(N);
    for (int k = 0; k < N; ++k) {
        Cd acc{};
        for (int n = 0; n < N; ++n) {
            const double a = s * (static_cast<long long>(k) * n % N);
            const Cd tw{std::cos(a), std::sin(a)};
            acc = acc + in[static_cast<std::size_t>(n)] * tw;
        }
        out[static_cast<std::size_t>(k)] = acc;
    }
}

// ─── mixed-radix decomposition ─────────────────────────────────────────────
//
// Pick the smallest supported radix (2,3,5,7) that divides N. Returns 0 when
// no small radix divides N (the remaining cofactor is large/prime) — the
// caller falls back to Bluestein for that size.
int small_radix(int N) {
    for (int r : {2, 3, 5, 7}) {
        if (N % r == 0) return r;
    }
    return 0;
}

// True iff N factors entirely into the supported small radices — i.e. the
// recursive mixed-radix path can transform it without a Bluestein fallback.
bool is_smooth(int N) {
    int m = N;
    for (int r : {2, 3, 5, 7}) {
        while (m % r == 0) m /= r;
    }
    return m == 1;
}

bool is_power_of_two(int N) { return N > 0 && (N & (N - 1)) == 0; }

// Forward-declared: the recursive mixed-radix engine and the Bluestein engine
// call into each other (Bluestein needs a power-of-two FFT, which is the
// radix-2 path of the mixed-radix engine).
void fft_recursive(const std::vector<Cd>& in, std::vector<Cd>& out, int sign);

// ─── Bluestein chirp-z transform ───────────────────────────────────────────
//
// Reduces an arbitrary-length-N DFT to a convolution of length M, where M is
// the next power of two >= 2*N-1. Convolution is done with three power-of-two
// FFTs. This is the fallback for sizes with a large or prime factor.
//
//   X[k] = b[k] * sum_n ( a[n] * conj_chirp[k-n] )
// with a[n] = x[n] * chirp[n], chirp[n] = exp(sign * i*pi*n^2/N).
void bluestein(const std::vector<Cd>& in, std::vector<Cd>& out, int sign) {
    const int N = static_cast<int>(in.size());
    if (N <= 1) { out = in; return; }

    int M = 1;
    while (M < 2 * N - 1) M <<= 1;

    // chirp[n] = exp(sign * i * pi * n^2 / N). Use (n^2 mod 2N) to keep the
    // angle small and accurate for large n.
    std::vector<Cd> chirp(static_cast<std::size_t>(N));
    for (int n = 0; n < N; ++n) {
        const long long n2 = (static_cast<long long>(n) * n) % (2LL * N);
        const double ang = static_cast<double>(sign) * kPi * static_cast<double>(n2)
                           / static_cast<double>(N);
        chirp[static_cast<std::size_t>(n)] = {std::cos(ang), std::sin(ang)};
    }

    // a[n] = x[n] * chirp[n], zero-padded to length M.
    std::vector<Cd> a(static_cast<std::size_t>(M), Cd{});
    for (int n = 0; n < N; ++n) {
        a[static_cast<std::size_t>(n)] =
            in[static_cast<std::size_t>(n)] * chirp[static_cast<std::size_t>(n)];
    }

    // b is the conjugate chirp kernel, length M, wrapped: b[k] and b[M-k].
    std::vector<Cd> b(static_cast<std::size_t>(M), Cd{});
    b[0] = {chirp[0].re, -chirp[0].im};
    for (int k = 1; k < N; ++k) {
        const Cd c{chirp[static_cast<std::size_t>(k)].re,
                   -chirp[static_cast<std::size_t>(k)].im};
        b[static_cast<std::size_t>(k)] = c;
        b[static_cast<std::size_t>(M - k)] = c;
    }

    // Circular convolution a (*) b via FFT: ifft(fft(a) .* fft(b)).
    std::vector<Cd> fa, fb;
    fft_recursive(a, fa, -1);
    fft_recursive(b, fb, -1);
    std::vector<Cd> prod(static_cast<std::size_t>(M));
    for (int i = 0; i < M; ++i) {
        prod[static_cast<std::size_t>(i)] =
            fa[static_cast<std::size_t>(i)] * fb[static_cast<std::size_t>(i)];
    }
    std::vector<Cd> conv;
    fft_recursive(prod, conv, +1);
    const double invM = 1.0 / static_cast<double>(M);

    // X[k] = chirp[k] * conv[k] / M.
    out.assign(static_cast<std::size_t>(N), Cd{});
    for (int k = 0; k < N; ++k) {
        Cd c = conv[static_cast<std::size_t>(k)];
        c.re *= invM;
        c.im *= invM;
        out[static_cast<std::size_t>(k)] =
            chirp[static_cast<std::size_t>(k)] * c;
    }
}

// ─── mixed-radix recursive FFT ─────────────────────────────────────────────
//
// Generic Cooley-Tukey split N = r * m. Splits the input into r interleaved
// sub-sequences of length m, transforms each recursively, then combines with
// twiddle factors. r is the smallest supported small radix dividing N. If no
// small radix divides N, defers the whole transform to Bluestein.
void fft_recursive(const std::vector<Cd>& in, std::vector<Cd>& out, int sign) {
    const int N = static_cast<int>(in.size());
    if (N <= 1) { out = in; return; }

    const int r = small_radix(N);
    if (r == 0) {
        // No small radix divides N: large/prime cofactor — use Bluestein.
        bluestein(in, out, sign);
        return;
    }

    const int m = N / r;

    // Small base case: a direct DFT is cheaper than recursing further.
    if (N <= 8 && !is_smooth(m)) {
        dft_naive(in, out, sign);
        return;
    }

    // Split into r sub-sequences of length m: sub[j][t] = in[t*r + j].
    std::vector<std::vector<Cd>> subs(
        static_cast<std::size_t>(r),
        std::vector<Cd>(static_cast<std::size_t>(m)));
    for (int t = 0; t < m; ++t) {
        for (int j = 0; j < r; ++j) {
            subs[static_cast<std::size_t>(j)][static_cast<std::size_t>(t)] =
                in[static_cast<std::size_t>(t * r + j)];
        }
    }

    // Transform each sub-sequence.
    std::vector<std::vector<Cd>> subF(static_cast<std::size_t>(r));
    for (int j = 0; j < r; ++j) {
        fft_recursive(subs[static_cast<std::size_t>(j)],
                      subF[static_cast<std::size_t>(j)], sign);
    }

    // Combine. out[k] = sum_j twiddle(N, sign)^(j*k) * subF[j][k mod m].
    const std::vector<Cd> w = make_twiddles(N, sign);
    out.assign(static_cast<std::size_t>(N), Cd{});
    for (int k = 0; k < N; ++k) {
        const int km = k % m;
        Cd acc{};
        for (int j = 0; j < r; ++j) {
            const int idx = (j * k) % N;
            acc = acc + w[static_cast<std::size_t>(idx)] *
                            subF[static_cast<std::size_t>(j)][static_cast<std::size_t>(km)];
        }
        out[static_cast<std::size_t>(k)] = acc;
    }
}

// ─── public single-signal transforms (double-precision core) ───────────────
//
// dft_1d runs the unscaled transform with the requested sign. The 1/N inverse
// scaling for the "backward" convention is applied by the callers that want
// an inverse (ifft / irfft).
void dft_1d(const std::vector<Cd>& in, std::vector<Cd>& out, int sign) {
    const int N = static_cast<int>(in.size());
    if (N == 0) { out.clear(); return; }
    if (N == 1) { out = in; return; }
    if (is_power_of_two(N) || is_smooth(N)) {
        fft_recursive(in, out, sign);
    } else {
        bluestein(in, out, sign);
    }
}

// ─── interleaved-complex tensor row helpers ────────────────────────────────
//
// Load row `r` of an (R, 2*C) interleaved-complex tensor into a length-C
// complex vector; store a length-C complex vector back.
void load_complex_row(const float* base, int row, int cols2,
                      std::vector<Cd>& dst) {
    const int C = cols2 / 2;
    dst.assign(static_cast<std::size_t>(C), Cd{});
    const float* p = base + static_cast<std::size_t>(row) * cols2;
    for (int c = 0; c < C; ++c) {
        dst[static_cast<std::size_t>(c)] = {static_cast<double>(p[2 * c]),
                                            static_cast<double>(p[2 * c + 1])};
    }
}

void store_complex_row(float* base, int row, int cols2,
                       const std::vector<Cd>& src) {
    const int C = cols2 / 2;
    float* p = base + static_cast<std::size_t>(row) * cols2;
    for (int c = 0; c < C; ++c) {
        p[2 * c]     = static_cast<float>(src[static_cast<std::size_t>(c)].re);
        p[2 * c + 1] = static_cast<float>(src[static_cast<std::size_t>(c)].im);
    }
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
