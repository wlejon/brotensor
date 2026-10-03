// Vulkan spectral ops for brosoundml: the complex elementwise ops, fft /
// ifft / rfft / irfft and their adjoints, and stft / istft with theirs.
// Contracts follow the CPU backend (src/cpu/fft.cpp, stft.cpp): FP32, complex
// data interleaved [re, im] along the columns, "backward" normalisation,
// every length allowed (no power-of-two requirement).
//
// Every transform is a matrix product against a DFT basis (dft.comp's
// SP_BASIS, built on the device per call, in row chunks of at most 16 Mi
// elements) through the FP32 GEMM, so the cost is O(L^2) per row at GEMM
// speed rather than O(L log L) at FFT speed: the right trade for the frame
// sizes these models use (n_fft 400-2048, a few thousand frames), where the
// GEMM runs at several TF/s. The STFT reads its frames in place (the padded
// signal with a row pitch of hop_length), folds the window and the
// normalisation into the basis, and the inverse's overlap-add and COLA
// division are one gather (SP_OLA).

#include "detail/gemm.h"
#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <initializer_list>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void audio_pad1d(DeviceCtx& d, const Tensor& X, int rows, int L, int pad_left, int pad_right, int mode,
                 Tensor& Y);   // ops_audio.cpp

namespace {

struct SpPush {
    std::uint64_t a, b, c, d, e;
    std::uint32_t u[11];
    std::int32_t sgn;
    float scale;
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void need32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
}

void ensure(Tensor& t, int rows, int cols) {
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) t.resize(rows, cols, Dtype::FP32);
}

std::uint32_t u32(long long v) { return static_cast<std::uint32_t>(v); }

void run(DeviceCtx& d, std::uint32_t op, const SpPush& pc, std::uint64_t items) {
    const Kernel& k = d.pipelines().get(ShaderId::dft, {op});
    launch(d, k, pc, groups_1d(items, k));
}

struct Basis {
    std::uint32_t mode = BASIS_R2C;
    int L = 1, sign = -1, tofs = 0;
    float scale = 1.0f;
    bool weights = false;
    std::uint64_t win = 0;
};

// Y[z](R, rows) = X[z](R, cols) B^T for z < batch, B the basis; X rows lda
// apart (they may overlap: STFT frames), batches sa / sc elements apart.
void dft_gemm(DeviceCtx& d, const Basis& bs, std::uint64_t x, int R, int cols, int lda, long long sa, int batch,
              std::uint64_t y, int rows, int ldc, long long sc, const char* op) {
    if (R == 0 || rows == 0 || batch == 0) return;
    if (cols == 0) fail(op, "empty transform");
    const long long kMaxBasis = 16LL << 20;
    const int chunk = static_cast<int>(std::max<long long>(1, std::min<long long>(rows, kMaxBasis / cols)));
    Tensor B = Tensor::empty_on(Device::vulkan(d.index()), chunk, cols, Dtype::FP32);
    for (int r0 = 0; r0 < rows; r0 += chunk) {
        const int nr = std::min(chunk, rows - r0);
        SpPush pc{};
        pc.c = bs.win;
        pc.d = addr(B.data);
        pc.u[0] = u32(nr); pc.u[1] = u32(cols); pc.u[2] = u32(bs.L); pc.u[3] = bs.mode; pc.u[4] = u32(bs.tofs);
        pc.u[5] = bs.weights ? 1u : 0u; pc.u[6] = u32(r0);
        pc.sgn = bs.sign;
        pc.scale = bs.scale;
        run(d, SP_BASIS, pc, std::uint64_t(nr) * cols);
        GemmArgs g;
        g.op = op;
        g.a = x; g.b = addr(B.data); g.c = y + std::uint64_t(r0) * 4u;
        g.m = R; g.n = nr; g.k = cols;
        g.lda = lda; g.ldb = cols; g.ldc = ldc;
        g.batch = batch; g.sa = sa; g.sb = 0; g.sc = sc;
        gemm(d, g);
    }
}

// ─── complex elementwise ────────────────────────────────────────────────────

void complex_mul(const Tensor& a, const Tensor& b, Tensor& y) {
    const char* op = "complex_mul";
    need32(op, a, "a"); need32(op, b, "b");
    need(op, a.rows == b.rows && a.cols == b.cols, "a and b must have identical shape");
    need(op, a.cols % 2 == 0, "cols must be even (interleaved [re,im] layout)");
    ensure(y, a.rows, a.cols);
    if (a.size() == 0) return;
    SpPush pc{};
    pc.a = addr(a.data); pc.b = addr(b.data); pc.d = addr(y.data); pc.u[0] = u32(a.size() / 2);
    run(device_of(a), SP_CX_MUL, pc, pc.u[0]);
}

void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY, Tensor& dA, Tensor& dB) {
    const char* op = "complex_mul_backward";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&a, &b, &dY, &dA, &dB}) need32(op, *t, "every operand");
    need(op, a.rows == b.rows && a.cols == b.cols, "a and b must have identical shape");
    need(op, dY.rows == a.rows && dY.cols == a.cols, "dY must match a / b shape");
    need(op, dA.rows == a.rows && dA.cols == a.cols, "dA must be pre-sized to a's shape");
    need(op, dB.rows == a.rows && dB.cols == a.cols, "dB must be pre-sized to b's shape");
    if (a.size() == 0) return;
    SpPush pc{};
    pc.a = addr(a.data); pc.b = addr(b.data); pc.c = addr(dY.data); pc.d = addr(dA.data); pc.e = addr(dB.data);
    pc.u[0] = u32(a.size() / 2);
    run(device_of(a), SP_CX_MUL_BWD, pc, pc.u[0]);
}

void complex_unary(const char* op, std::uint32_t code, const Tensor& z, Tensor& y) {
    need32(op, z, "z");
    need(op, z.cols % 2 == 0, "z.cols must be even (interleaved [re,im] layout)");
    ensure(y, z.rows, z.cols / 2);
    if (z.size() == 0) return;
    SpPush pc{};
    pc.a = addr(z.data); pc.d = addr(y.data); pc.u[0] = u32(z.size() / 2);
    run(device_of(z), code, pc, pc.u[0]);
}

void complex_abs(const Tensor& z, Tensor& y) { complex_unary("complex_abs", SP_CX_ABS, z, y); }
void complex_angle(const Tensor& z, Tensor& y) { complex_unary("complex_angle", SP_CX_ANGLE, z, y); }

void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ) {
    const char* op = "complex_abs_backward";
    need32(op, z, "z"); need32(op, dY, "dY");
    need(op, z.cols % 2 == 0, "z.cols must be even (interleaved [re,im] layout)");
    need(op, dY.rows == z.rows && dY.cols == z.cols / 2, "dY must be the real (R, C) magnitude grad");
    ensure(dZ, z.rows, z.cols);
    if (z.size() == 0) return;
    SpPush pc{};
    pc.a = addr(z.data); pc.b = addr(dY.data); pc.d = addr(dZ.data); pc.u[0] = u32(z.size() / 2);
    run(device_of(z), SP_CX_ABS_BWD, pc, pc.u[0]);
}

void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y) {
    const char* op = "complex_from_polar";
    need32(op, mag, "mag"); need32(op, phase, "phase");
    need(op, mag.rows == phase.rows && mag.cols == phase.cols, "mag and phase must have identical shape");
    ensure(y, mag.rows, 2 * mag.cols);
    if (mag.size() == 0) return;
    SpPush pc{};
    pc.a = addr(mag.data); pc.b = addr(phase.data); pc.d = addr(y.data); pc.u[0] = u32(mag.size());
    run(device_of(mag), SP_CX_POLAR, pc, pc.u[0]);
}

// ─── fft family ─────────────────────────────────────────────────────────────

void c2c(const char* op, const Tensor& x, Tensor& y, int sign) {
    need32(op, x, "x");
    need(op, x.cols % 2 == 0, "x.cols must be even (interleaved [re,im] layout)");
    ensure(y, x.rows, x.cols);
    if (x.size() == 0) return;
    const int N = x.cols / 2;
    Basis b;
    b.mode = BASIS_C2C; b.L = N; b.sign = sign; b.scale = sign > 0 ? static_cast<float>(1.0 / N) : 1.0f;
    need(op, x.data != y.data, "x and y must not alias");
    dft_gemm(device_of(x), b, addr(x.data), x.rows, x.cols, x.cols, 0, 1, addr(y.data), x.cols, x.cols, 0, op);
}

void fft(const Tensor& x, Tensor& y) { c2c("fft", x, y, -1); }
void ifft(const Tensor& x, Tensor& y) { c2c("ifft", x, y, +1); }

void rfft(const Tensor& x, Tensor& y) {
    const char* op = "rfft";
    need32(op, x, "x");
    const int L = x.cols;
    need(op, L >= 1, "signal length L (x.cols) must be >= 1");
    const int C = L / 2 + 1;
    ensure(y, x.rows, 2 * C);
    if (x.size() == 0) return;
    Basis b;
    b.mode = BASIS_R2C; b.L = L; b.sign = -1;
    dft_gemm(device_of(x), b, addr(x.data), x.rows, L, L, 0, 1, addr(y.data), 2 * C, 2 * C, 0, op);
}

void irfft(const Tensor& x, int L, Tensor& y) {
    const char* op = "irfft";
    need32(op, x, "x");
    need(op, x.cols % 2 == 0, "x.cols must be even (interleaved [re,im] layout)");
    need(op, L >= 1, "output length L must be >= 1");
    const int C = x.cols / 2;
    need(op, C == L / 2 + 1, "half-spectrum bin count must equal L/2+1");
    ensure(y, x.rows, L);
    if (x.size() == 0) return;
    Basis b;
    b.mode = BASIS_C2R; b.L = L; b.sign = +1; b.weights = true; b.scale = static_cast<float>(1.0 / L);
    dft_gemm(device_of(x), b, addr(x.data), x.rows, 2 * C, 2 * C, 0, 1, addr(y.data), L, L, 0, op);
}

void rfft_backward(const Tensor& dY, int L, Tensor& dX) {
    const char* op = "rfft_backward";
    need32(op, dY, "dY");
    need(op, dY.cols % 2 == 0, "dY.cols must be even (interleaved [re,im] layout)");
    need(op, L >= 1, "signal length L must be >= 1");
    const int C = dY.cols / 2;
    need(op, C == L / 2 + 1, "dY bin count must equal L/2+1");
    ensure(dX, dY.rows, L);
    if (dY.size() == 0) return;
    Basis b;
    b.mode = BASIS_C2R; b.L = L; b.sign = +1;
    dft_gemm(device_of(dY), b, addr(dY.data), dY.rows, 2 * C, 2 * C, 0, 1, addr(dX.data), L, L, 0, op);
}

void irfft_backward(const Tensor& dY, Tensor& dX) {
    const char* op = "irfft_backward";
    need32(op, dY, "dY");
    const int L = dY.cols;
    need(op, L >= 1, "dY length L (dY.cols) must be >= 1");
    const int C = L / 2 + 1;
    ensure(dX, dY.rows, 2 * C);
    if (dY.size() == 0) return;
    Basis b;
    b.mode = BASIS_R2C; b.L = L; b.sign = -1; b.weights = true; b.scale = static_cast<float>(1.0 / L);
    dft_gemm(device_of(dY), b, addr(dY.data), dY.rows, L, L, 0, 1, addr(dX.data), 2 * C, 2 * C, 0, op);
}

// ─── STFT ───────────────────────────────────────────────────────────────────

struct Geom {
    int bins, frames, padded_len, pad_lo;
};

Geom geometry(const char* op, int N, int signal_len, int n_fft, int hop, int win, bool center) {
    need(op, N >= 0, "N must be >= 0");
    need(op, n_fft >= 1, "n_fft must be >= 1");
    need(op, hop >= 1, "hop_length must be >= 1");
    need(op, win >= 1 && win <= n_fft, "win_length must satisfy 1 <= win_length <= n_fft");
    need(op, signal_len >= 1, "signal_len must be >= 1");
    Geom g;
    g.bins = n_fft / 2 + 1;
    g.pad_lo = (n_fft - win) / 2;
    if (center) {
        need(op, signal_len >= n_fft / 2 + 1, "center=true needs signal_len >= n_fft/2 + 1");
        g.padded_len = signal_len + n_fft;
        g.frames = 1 + signal_len / hop;
    } else {
        need(op, signal_len >= n_fft, "center=false needs signal_len >= n_fft");
        g.padded_len = signal_len;
        g.frames = 1 + (signal_len - n_fft) / hop;
    }
    return g;
}

void check_window(const char* op, const Tensor& window, int win) {
    need32(op, window, "window");
    need(op, window.rows == 1 && window.cols == win, "window must be a (1, win_length) tensor");
}

void stft(const Tensor& signal, const Tensor& window, int N, int n_fft, int hop, int win, bool center,
          bool normalized, Tensor& spec) {
    const char* op = "stft";
    need32(op, signal, "signal");
    check_window(op, window, win);
    need(op, signal.rows == N, "signal.rows must equal N");
    const int L = signal.cols;
    const Geom g = geometry(op, N, L, n_fft, hop, win, center);
    ensure(spec, N * g.frames, 2 * g.bins);
    if (N == 0) return;
    DeviceCtx& d = device_of(signal);
    Tensor padded;
    std::uint64_t base = addr(signal.data);
    if (center) {
        padded = Tensor::empty_on(Device::vulkan(d.index()), N, g.padded_len, Dtype::FP32);
        audio_pad1d(d, signal, N, L, n_fft / 2, n_fft - n_fft / 2, 1, padded);
        base = addr(padded.data);
    }
    Basis b;
    b.mode = BASIS_R2C; b.L = n_fft; b.sign = -1; b.tofs = g.pad_lo; b.win = addr(window.data);
    b.scale = normalized ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_fft))) : 1.0f;
    dft_gemm(d, b, base + std::uint64_t(g.pad_lo) * 4u, g.frames, win, hop, g.padded_len, N, addr(spec.data),
             2 * g.bins, 2 * g.bins, static_cast<long long>(g.frames) * 2 * g.bins, op);
}

void stft_backward(const Tensor& dSpec, const Tensor& window, int N, int signal_len, int n_fft, int hop, int win,
                   bool center, bool normalized, Tensor& dSignal) {
    const char* op = "stft_backward";
    need32(op, dSpec, "dSpec");
    check_window(op, window, win);
    const Geom g = geometry(op, N, signal_len, n_fft, hop, win, center);
    need(op, dSpec.rows == N * g.frames && dSpec.cols == 2 * g.bins, "dSpec shape must match the stft output shape");
    ensure(dSignal, N, signal_len);
    if (N == 0) return;
    DeviceCtx& d = device_of(dSpec);
    // G (N frames, win) = per frame, window * Re(+1 DFT of the zero-padded dSpec) * norm.
    Tensor G = Tensor::empty_on(Device::vulkan(d.index()), N * g.frames, win, Dtype::FP32);
    Basis b;
    b.mode = BASIS_C2R; b.L = n_fft; b.sign = +1; b.tofs = g.pad_lo; b.win = addr(window.data);
    b.scale = normalized ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_fft))) : 1.0f;
    dft_gemm(d, b, addr(dSpec.data), N * g.frames, 2 * g.bins, 2 * g.bins, 0, 1, addr(G.data), win, win, 0, op);
    SpPush pc{};
    pc.a = addr(G.data); pc.d = addr(dSignal.data);
    const std::uint32_t u[] = {u32(N), u32(signal_len), u32(g.frames), u32(hop), u32(g.pad_lo), u32(win),
                               center ? 1u : 0u, u32(n_fft / 2), u32(g.padded_len), 0, 0};
    std::copy(std::begin(u), std::end(u), pc.u);
    run(d, SP_STFT_ADJ, pc, std::uint64_t(N) * signal_len);
}

void istft(const Tensor& spec, const Tensor& window, int N, int signal_len, int n_fft, int hop, int win, bool center,
           bool normalized, Tensor& signal) {
    const char* op = "istft";
    need32(op, spec, "spec");
    check_window(op, window, win);
    const Geom g = geometry(op, N, signal_len, n_fft, hop, win, center);
    need(op, spec.rows == N * g.frames && spec.cols == 2 * g.bins, "spec shape must match the stft output shape");
    ensure(signal, N, signal_len);
    if (N == 0) return;
    DeviceCtx& d = device_of(spec);
    // F (N frames, win) = window * irfft(spec * norm) over the window's span.
    Tensor F = Tensor::empty_on(Device::vulkan(d.index()), N * g.frames, win, Dtype::FP32);
    Basis b;
    b.mode = BASIS_C2R; b.L = n_fft; b.sign = +1; b.tofs = g.pad_lo; b.win = addr(window.data); b.weights = true;
    b.scale = static_cast<float>((normalized ? std::sqrt(static_cast<double>(n_fft)) : 1.0) / n_fft);
    dft_gemm(d, b, addr(spec.data), N * g.frames, 2 * g.bins, 2 * g.bins, 0, 1, addr(F.data), win, win, 0, op);
    SpPush pc{};
    pc.a = addr(F.data); pc.c = addr(window.data); pc.d = addr(signal.data);
    const std::uint32_t u[] = {u32(N), u32(signal_len), u32(g.frames), u32(hop), u32(g.pad_lo), u32(win),
                               center ? u32(n_fft / 2) : 0u, u32(g.padded_len), 0, 0, 0};
    std::copy(std::begin(u), std::end(u), pc.u);
    run(d, SP_OLA, pc, std::uint64_t(N) * signal_len);
}

void istft_backward(const Tensor& dSignal, const Tensor& window, int N, int signal_len, int n_fft, int hop, int win,
                    bool center, bool normalized, Tensor& dSpec) {
    const char* op = "istft_backward";
    need32(op, dSignal, "dSignal");
    need(op, dSignal.rows == N && dSignal.cols == signal_len, "dSignal must be a (N, signal_len) tensor");
    check_window(op, window, win);
    const Geom g = geometry(op, N, signal_len, n_fft, hop, win, center);
    ensure(dSpec, N * g.frames, 2 * g.bins);
    if (N == 0) return;
    DeviceCtx& d = device_of(dSignal);
    Tensor gacc = Tensor::empty_on(Device::vulkan(d.index()), N, g.padded_len, Dtype::FP32);
    SpPush pc{};
    pc.a = addr(dSignal.data); pc.c = addr(window.data); pc.d = addr(gacc.data);
    const std::uint32_t u[] = {u32(N), u32(signal_len), u32(g.frames), u32(hop), u32(g.pad_lo), u32(win),
                               center ? u32(n_fft / 2) : 0u, u32(g.padded_len), 0, 0, 0};
    std::copy(std::begin(u), std::end(u), pc.u);
    run(d, SP_ISTFT_ADJ, pc, std::uint64_t(N) * g.padded_len);
    Basis b;
    b.mode = BASIS_R2C; b.L = n_fft; b.sign = -1; b.tofs = g.pad_lo; b.win = addr(window.data); b.weights = true;
    b.scale = static_cast<float>((normalized ? std::sqrt(static_cast<double>(n_fft)) : 1.0) / n_fft);
    dft_gemm(d, b, addr(gacc.data) + std::uint64_t(g.pad_lo) * 4u, g.frames, win, hop, g.padded_len, N,
             addr(dSpec.data), 2 * g.bins, 2 * g.bins, static_cast<long long>(g.frames) * 2 * g.bins, op);
}

}  // namespace

void fill_vulkan_vtable_spectral(::brotensor::detail::OpsVTable& v) {
    v.complex_mul = &complex_mul;
    v.complex_mul_backward = &complex_mul_backward;
    v.complex_abs = &complex_abs;
    v.complex_abs_backward = &complex_abs_backward;
    v.complex_angle = &complex_angle;
    v.complex_from_polar = &complex_from_polar;
    v.fft = &fft;
    v.ifft = &ifft;
    v.rfft = &rfft;
    v.irfft = &irfft;
    v.rfft_backward = &rfft_backward;
    v.irfft_backward = &irfft_backward;
    v.stft = &stft;
    v.stft_backward = &stft_backward;
    v.istft = &istft;
    v.istft_backward = &istft_backward;
}

}  // namespace brotensor::detail::vulkan
