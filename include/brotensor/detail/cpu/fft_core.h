#pragma once

// ─── CPU FFT core — shared internals ───────────────────────────────────────
//
// Hand-rolled mixed-radix + Bluestein DFT, factored out of src/cpu/fft.cpp so
// the STFT / iSTFT ops (and any later spectral op) can reuse the exact same
// transform instead of duplicating it.
//
// This is a CPU-backend-private header (brotensor::detail::cpu). It is
// header-only and double-precision internally for accuracy, matching the
// fft.cpp original. No external libraries.
//
//   * Cd                — tiny double-precision complex helper.
//   * dft_1d            — unscaled length-N DFT, `sign` = -1 forward / +1
//                         inverse. The "backward" 1/N inverse scaling is
//                         applied by callers that want an inverse.
//   * load/store_complex_row — interleaved-complex (R, 2*C) tensor row I/O.
//
// The mixed-radix engine handles sizes whose prime factorisation uses only
// the radices 2/3/5/7 (covers Whisper's n_fft = 400 = 2^4 * 5^2); anything
// with a large or prime factor falls back to a Bluestein chirp-z transform,
// so dft_1d is correct for every length >= 1.

#include <cmath>
#include <cstddef>
#include <vector>

namespace brotensor::detail::cpu::fftcore {

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

// ─── twiddle-factor table: w[k] = exp(sign * 2*pi*i * k / N) ───────────────
inline std::vector<Cd> make_twiddles(int N, int sign) {
    std::vector<Cd> w(static_cast<std::size_t>(N));
    const double s = static_cast<double>(sign) * 2.0 * kPi
                     / static_cast<double>(N);
    for (int k = 0; k < N; ++k) {
        w[static_cast<std::size_t>(k)] = {std::cos(s * k), std::sin(s * k)};
    }
    return w;
}

// ─── naive O(N^2) DFT — base case + Bluestein small kernels ────────────────
inline void dft_naive(const std::vector<Cd>& in, std::vector<Cd>& out,
                      int sign) {
    const int N = static_cast<int>(in.size());
    out.assign(static_cast<std::size_t>(N), Cd{});
    const double s = static_cast<double>(sign) * 2.0 * kPi
                     / static_cast<double>(N);
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

// Smallest supported radix (2,3,5,7) dividing N, or 0 if none does.
inline int small_radix(int N) {
    for (int r : {2, 3, 5, 7}) {
        if (N % r == 0) return r;
    }
    return 0;
}

// True iff N factors entirely into the supported small radices.
inline bool is_smooth(int N) {
    int m = N;
    for (int r : {2, 3, 5, 7}) {
        while (m % r == 0) m /= r;
    }
    return m == 1;
}

inline bool is_power_of_two(int N) { return N > 0 && (N & (N - 1)) == 0; }

// Forward declaration: Bluestein needs the recursive (power-of-two) FFT.
inline void fft_recursive(const std::vector<Cd>& in, std::vector<Cd>& out,
                          int sign);

// ─── Bluestein chirp-z transform — arbitrary-length fallback ───────────────
inline void bluestein(const std::vector<Cd>& in, std::vector<Cd>& out,
                      int sign) {
    const int N = static_cast<int>(in.size());
    if (N <= 1) { out = in; return; }

    int M = 1;
    while (M < 2 * N - 1) M <<= 1;

    std::vector<Cd> chirp(static_cast<std::size_t>(N));
    for (int n = 0; n < N; ++n) {
        const long long n2 = (static_cast<long long>(n) * n) % (2LL * N);
        const double ang = static_cast<double>(sign) * kPi
                           * static_cast<double>(n2) / static_cast<double>(N);
        chirp[static_cast<std::size_t>(n)] = {std::cos(ang), std::sin(ang)};
    }

    std::vector<Cd> a(static_cast<std::size_t>(M), Cd{});
    for (int n = 0; n < N; ++n) {
        a[static_cast<std::size_t>(n)] =
            in[static_cast<std::size_t>(n)] * chirp[static_cast<std::size_t>(n)];
    }

    std::vector<Cd> b(static_cast<std::size_t>(M), Cd{});
    b[0] = {chirp[0].re, -chirp[0].im};
    for (int k = 1; k < N; ++k) {
        const Cd c{chirp[static_cast<std::size_t>(k)].re,
                   -chirp[static_cast<std::size_t>(k)].im};
        b[static_cast<std::size_t>(k)] = c;
        b[static_cast<std::size_t>(M - k)] = c;
    }

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
inline void fft_recursive(const std::vector<Cd>& in, std::vector<Cd>& out,
                          int sign) {
    const int N = static_cast<int>(in.size());
    if (N <= 1) { out = in; return; }

    const int r = small_radix(N);
    if (r == 0) {
        bluestein(in, out, sign);
        return;
    }

    const int m = N / r;

    if (N <= 8 && !is_smooth(m)) {
        dft_naive(in, out, sign);
        return;
    }

    std::vector<std::vector<Cd>> subs(
        static_cast<std::size_t>(r),
        std::vector<Cd>(static_cast<std::size_t>(m)));
    for (int t = 0; t < m; ++t) {
        for (int j = 0; j < r; ++j) {
            subs[static_cast<std::size_t>(j)][static_cast<std::size_t>(t)] =
                in[static_cast<std::size_t>(t * r + j)];
        }
    }

    std::vector<std::vector<Cd>> subF(static_cast<std::size_t>(r));
    for (int j = 0; j < r; ++j) {
        fft_recursive(subs[static_cast<std::size_t>(j)],
                      subF[static_cast<std::size_t>(j)], sign);
    }

    const std::vector<Cd> w = make_twiddles(N, sign);
    out.assign(static_cast<std::size_t>(N), Cd{});
    for (int k = 0; k < N; ++k) {
        const int km = k % m;
        Cd acc{};
        for (int j = 0; j < r; ++j) {
            const int idx = (j * k) % N;
            acc = acc + w[static_cast<std::size_t>(idx)] *
                            subF[static_cast<std::size_t>(j)]
                                [static_cast<std::size_t>(km)];
        }
        out[static_cast<std::size_t>(k)] = acc;
    }
}

// ─── public single-signal transform (unscaled) ─────────────────────────────
//
// dft_1d runs the unscaled transform with the requested sign. The 1/N inverse
// scaling for the "backward" convention is applied by inverse callers.
inline void dft_1d(const std::vector<Cd>& in, std::vector<Cd>& out, int sign) {
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
inline void load_complex_row(const float* base, int row, int cols2,
                              std::vector<Cd>& dst) {
    const int C = cols2 / 2;
    dst.assign(static_cast<std::size_t>(C), Cd{});
    const float* p = base + static_cast<std::size_t>(row) * cols2;
    for (int c = 0; c < C; ++c) {
        dst[static_cast<std::size_t>(c)] = {static_cast<double>(p[2 * c]),
                                            static_cast<double>(p[2 * c + 1])};
    }
}

inline void store_complex_row(float* base, int row, int cols2,
                              const std::vector<Cd>& src) {
    const int C = cols2 / 2;
    float* p = base + static_cast<std::size_t>(row) * cols2;
    for (int c = 0; c < C; ++c) {
        p[2 * c]     = static_cast<float>(src[static_cast<std::size_t>(c)].re);
        p[2 * c + 1] = static_cast<float>(src[static_cast<std::size_t>(c)].im);
    }
}

} // namespace brotensor::detail::cpu::fftcore
