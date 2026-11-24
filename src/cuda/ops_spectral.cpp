// CUDA Wave 7 spectral ops via single-source brass MIR kernels
// (MirKind::Spectral / MirKind::Copy2d) + CUDA FP32 linear_forward_batched GEMM:
//   complex_{mul,mul_backward,abs,abs_backward,angle,from_polar}
//   fft, ifft, rfft, irfft, rfft_backward, irfft_backward
//   stft, stft_backward, istft, istft_backward

#include "cu_mir_cache.h"
#include "../kernels/mir_copy.h"
#include "../kernels/mir_spectral.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

// Defined in ops_audio.cpp / batched_ops.cu.
void audio_pad1d(const Tensor& X, int rows, int L, int pad_left, int pad_right,
                 int mode, Tensor& Y);
void linear_forward_batched(const Tensor& W, const Tensor& bias,
                            const Tensor& X_BD, Tensor& Y_BD);

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void need32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
    }
}

void need32_out(const char* op, const Tensor& t, const char* name) {
    if (t.data != nullptr && t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (audio ops are FP32-only)");
    }
}

void ensure(Tensor& t, int rows, int cols) {
    if (t.rows != rows || t.cols != cols || t.dtype != Dtype::FP32) {
        t.resize(rows, cols, Dtype::FP32);
    }
}

std::int32_t i32(long long v) noexcept {
    return static_cast<std::int32_t>(v);
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

void* byte_off(void* p, std::size_t bytes) noexcept {
    return static_cast<char*>(p) + bytes;
}

const void* byte_off(const void* p, std::size_t bytes) noexcept {
    return static_cast<const char*>(p) + bytes;
}

void run_spectral(std::uint32_t code, std::uint8_t flags, std::uint64_t items,
                  std::span<const std::uintptr_t> ptrs,
                  std::span<const std::int32_t> ints,
                  std::span<const float> flts = {}) {
    if (items == 0) return;
    CUfunction fn = mir_function(
        MirKey{MirKind::Spectral, code, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    mir_dispatch_args(fn, mir_groups_1d(static_cast<std::uint32_t>(items), 1, kernels::kSpectralBlock),
                      kernels::kSpectralBlock, ptrs, ints, flts);
}

// Handles overlapping rows (hop < win) safely via MirKind::Copy2d, unlike
// cudaMemcpy2DAsync which rejects src_pitch < width.
void copy2d_f32(const void* src, void* dst, int src_pitch, int dst_pitch, int width, int height) {
    if (width <= 0 || height <= 0) return;
    const int lanes = mir_copy2d_lanes(
        4u, static_cast<std::uint32_t>(src_pitch), static_cast<std::uint32_t>(dst_pitch),
        static_cast<std::uint32_t>(width), addr(src), addr(dst));
    CUfunction fn = mir_function(
        MirKey{MirKind::Copy2d, 4u, Dtype::FP32, Dtype::FP32, static_cast<std::uint8_t>(lanes)});
    const std::uint32_t gx = mir_groups_1d(
        static_cast<std::uint32_t>(width), lanes, kernels::kCopyBlock);
    const std::uint32_t gy = static_cast<std::uint32_t>(std::clamp(height, 1, 65535));
    mir_dispatch_copy2d(fn, gx, gy, src, dst,
                        static_cast<std::uint32_t>(src_pitch),
                        static_cast<std::uint32_t>(dst_pitch),
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height));
}

struct Basis {
    std::uint32_t mode = BASIS_R2C;
    int L = 1;
    int sign = -1;
    int tofs = 0;
    float scale = 1.0f;
    bool weights = false;
    const void* win = nullptr;
};

// Y[z](R, rows) = X[z](R, cols) @ B^T for z < batch, B built by SP_BASIS in
// row chunks capped at 16 Mi elements; X rows lda apart (may overlap for STFT).
void dft_gemm(::brotensor::Device dev, const Basis& bs, const void* x, int R, int cols,
              int lda, long long sa, int batch, void* y, int rows, int ldc,
              long long /*sc*/, const char* op) {
    if (R == 0 || rows == 0 || batch == 0) return;
    if (cols == 0) fail(op, "empty transform");

    const int total_R = batch * R;
    Tensor X_packed;
    Tensor X_in;
    if (batch == 1 && lda == cols) {
        X_in = Tensor::view(dev, const_cast<void*>(x), R, cols, Dtype::FP32);
    } else {
        X_packed = Tensor::empty_on(dev, total_R, cols, Dtype::FP32);
        for (int z = 0; z < batch; ++z) {
            copy2d_f32(
                byte_off(x, static_cast<std::size_t>(z) * static_cast<std::size_t>(sa) * sizeof(float)),
                byte_off(X_packed.data, static_cast<std::size_t>(z) * R * cols * sizeof(float)),
                lda, cols, cols, R);
        }
        X_in = Tensor::view(dev, X_packed.data, total_R, cols, Dtype::FP32);
    }

    constexpr long long kMaxBasis = 16LL << 20;
    const int chunk = static_cast<int>(
        std::max<long long>(1, std::min<long long>(rows, kMaxBasis / cols)));
    Tensor B = Tensor::empty_on(dev, chunk, cols, Dtype::FP32);
    Tensor zero_b = Tensor::zeros_on(dev, chunk, 1, Dtype::FP32);
    Tensor Y_chunk;
    if (chunk < rows) {
        Y_chunk = Tensor::empty_on(dev, total_R, chunk, Dtype::FP32);
    }
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (bs.mode & 0x3u) | (bs.weights ? 4u : 0u) | (bs.win != nullptr ? 8u : 0u));

    for (int r0 = 0; r0 < rows; r0 += chunk) {
        const int nr = std::min(chunk, rows - r0);
        const std::uintptr_t ptrs[] = {addr(bs.win), addr(B.data)};
        const std::int32_t ints[] = {
            i32(nr), i32(cols), i32(bs.L), i32(bs.tofs), i32(r0), bs.sign,
        };
        const float flts[] = {bs.scale};
        run_spectral(SP_BASIS, flags, static_cast<std::uint64_t>(nr) * cols, ptrs, ints, flts);

        const Tensor B_v = Tensor::view(dev, B.data, nr, cols, Dtype::FP32);
        const Tensor zb_v = Tensor::view(dev, zero_b.data, nr, 1, Dtype::FP32);
        if (nr == rows && ldc == rows) {
            Tensor Y_all = Tensor::view(dev, y, total_R, rows, Dtype::FP32);
            linear_forward_batched(B_v, zb_v, X_in, Y_all);
        } else {
            Tensor Y_c = Tensor::view(dev, Y_chunk.data, total_R, nr, Dtype::FP32);
            linear_forward_batched(B_v, zb_v, X_in, Y_c);
            copy2d_f32(
                Y_c.data,
                byte_off(y, static_cast<std::size_t>(r0) * sizeof(float)),
                nr, ldc, nr, total_R);
        }
    }
}

void complex_unary(const char* op, std::uint32_t code, const Tensor& z, Tensor& y) {
    need32(op, z, "z");
    need32_out(op, y, "y");
    need(op, z.cols % 2 == 0, "z.cols must be even (interleaved [re,im] layout)");
    ensure(y, z.rows, z.cols / 2);
    if (z.size() == 0) return;
    const std::int32_t n = i32(z.size() / 2);
    const std::uintptr_t ptrs[] = {addr(z.data), 0, 0, addr(y.data), 0};
    const std::int32_t ints[] = {n};
    run_spectral(code, 0, static_cast<std::uint64_t>(n), ptrs, ints);
}

void c2c(const char* op, const Tensor& x, Tensor& y, int sign) {
    need32(op, x, "x");
    need32_out(op, y, "y");
    need(op, x.cols % 2 == 0, "x.cols must be even (interleaved [re,im] layout)");
    ensure(y, x.rows, x.cols);
    if (x.size() == 0) return;
    need(op, x.data != y.data, "x and y must not alias");
    const int N = x.cols / 2;
    Basis b;
    b.mode = BASIS_C2C;
    b.L = N;
    b.sign = sign;
    b.scale = sign > 0 ? static_cast<float>(1.0 / N) : 1.0f;
    dft_gemm(x.device, b, x.data, x.rows, x.cols, x.cols, 0, 1, y.data, x.cols, x.cols, 0, op);
}

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
    need(op, window.rows == 1 && window.cols == win,
         "window must be a (1, win_length) tensor");
}

}  // namespace

// ─── complex elementwise ────────────────────────────────────────────────────

void complex_mul(const Tensor& a, const Tensor& b, Tensor& y) {
    constexpr const char* op = "complex_mul";
    need32(op, a, "a");
    need32(op, b, "b");
    need32_out(op, y, "y");
    need(op, a.rows == b.rows && a.cols == b.cols, "a and b must have identical shape");
    need(op, a.cols % 2 == 0, "cols must be even (interleaved [re,im] layout)");
    ensure(y, a.rows, a.cols);
    if (a.size() == 0) return;
    const std::int32_t n = i32(a.size() / 2);
    const std::uintptr_t ptrs[] = {addr(a.data), addr(b.data), 0, addr(y.data), 0};
    const std::int32_t ints[] = {n};
    run_spectral(SP_CX_MUL, 0, static_cast<std::uint64_t>(n), ptrs, ints);
}

void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY,
                          Tensor& dA, Tensor& dB) {
    constexpr const char* op = "complex_mul_backward";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&a, &b, &dY, &dA, &dB}) {
        need32(op, *t, "every operand");
    }
    need(op, a.rows == b.rows && a.cols == b.cols, "a and b must have identical shape");
    need(op, dY.rows == a.rows && dY.cols == a.cols, "dY must match a / b shape");
    need(op, dA.rows == a.rows && dA.cols == a.cols, "dA must be pre-sized to a's shape");
    need(op, dB.rows == a.rows && dB.cols == a.cols, "dB must be pre-sized to b's shape");
    if (a.size() == 0) return;
    const std::int32_t n = i32(a.size() / 2);
    const std::uintptr_t ptrs[] = {
        addr(a.data), addr(b.data), addr(dY.data), addr(dA.data), addr(dB.data)};
    const std::int32_t ints[] = {n};
    run_spectral(SP_CX_MUL_BWD, 0, static_cast<std::uint64_t>(n), ptrs, ints);
}

void complex_abs(const Tensor& z, Tensor& y) {
    complex_unary("complex_abs", SP_CX_ABS, z, y);
}

void complex_angle(const Tensor& z, Tensor& y) {
    complex_unary("complex_angle", SP_CX_ANGLE, z, y);
}

void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ) {
    constexpr const char* op = "complex_abs_backward";
    need32(op, z, "z");
    need32(op, dY, "dY");
    need32_out(op, dZ, "dZ");
    need(op, z.cols % 2 == 0, "z.cols must be even (interleaved [re,im] layout)");
    need(op, dY.rows == z.rows && dY.cols == z.cols / 2,
         "dY must be the real (R, C) magnitude grad");
    ensure(dZ, z.rows, z.cols);
    if (z.size() == 0) return;
    const std::int32_t n = i32(z.size() / 2);
    const std::uintptr_t ptrs[] = {addr(z.data), addr(dY.data), 0, addr(dZ.data), 0};
    const std::int32_t ints[] = {n};
    run_spectral(SP_CX_ABS_BWD, 0, static_cast<std::uint64_t>(n), ptrs, ints);
}

void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y) {
    constexpr const char* op = "complex_from_polar";
    need32(op, mag, "mag");
    need32(op, phase, "phase");
    need32_out(op, y, "y");
    need(op, mag.rows == phase.rows && mag.cols == phase.cols,
         "mag and phase must have identical shape");
    ensure(y, mag.rows, 2 * mag.cols);
    if (mag.size() == 0) return;
    const std::int32_t n = i32(mag.size());
    const std::uintptr_t ptrs[] = {addr(mag.data), addr(phase.data), 0, addr(y.data), 0};
    const std::int32_t ints[] = {n};
    run_spectral(SP_CX_POLAR, 0, static_cast<std::uint64_t>(n), ptrs, ints);
}

// ─── fft family ─────────────────────────────────────────────────────────────

void fft(const Tensor& x, Tensor& y) { c2c("fft", x, y, -1); }
void ifft(const Tensor& x, Tensor& y) { c2c("ifft", x, y, +1); }

void rfft(const Tensor& x, Tensor& y) {
    constexpr const char* op = "rfft";
    need32(op, x, "x");
    need32_out(op, y, "y");
    const int L = x.cols;
    need(op, L >= 1, "signal length L (x.cols) must be >= 1");
    const int C = L / 2 + 1;
    ensure(y, x.rows, 2 * C);
    if (x.size() == 0) return;
    Basis b;
    b.mode = BASIS_R2C;
    b.L = L;
    b.sign = -1;
    dft_gemm(x.device, b, x.data, x.rows, L, L, 0, 1, y.data, 2 * C, 2 * C, 0, op);
}

void irfft(const Tensor& x, int L, Tensor& y) {
    constexpr const char* op = "irfft";
    need32(op, x, "x");
    need32_out(op, y, "y");
    need(op, x.cols % 2 == 0, "x.cols must be even (interleaved [re,im] layout)");
    need(op, L >= 1, "output length L must be >= 1");
    const int C = x.cols / 2;
    need(op, C == L / 2 + 1, "half-spectrum bin count must equal L/2+1");
    ensure(y, x.rows, L);
    if (x.size() == 0) return;
    Basis b;
    b.mode = BASIS_C2R;
    b.L = L;
    b.sign = +1;
    b.weights = true;
    b.scale = static_cast<float>(1.0 / L);
    dft_gemm(x.device, b, x.data, x.rows, 2 * C, 2 * C, 0, 1, y.data, L, L, 0, op);
}

void rfft_backward(const Tensor& dY, int L, Tensor& dX) {
    constexpr const char* op = "rfft_backward";
    need32(op, dY, "dY");
    need32_out(op, dX, "dX");
    need(op, dY.cols % 2 == 0, "dY.cols must be even (interleaved [re,im] layout)");
    need(op, L >= 1, "signal length L must be >= 1");
    const int C = dY.cols / 2;
    need(op, C == L / 2 + 1, "dY bin count must equal L/2+1");
    ensure(dX, dY.rows, L);
    if (dY.size() == 0) return;
    Basis b;
    b.mode = BASIS_C2R;
    b.L = L;
    b.sign = +1;
    dft_gemm(dY.device, b, dY.data, dY.rows, 2 * C, 2 * C, 0, 1, dX.data, L, L, 0, op);
}

void irfft_backward(const Tensor& dY, Tensor& dX) {
    constexpr const char* op = "irfft_backward";
    need32(op, dY, "dY");
    need32_out(op, dX, "dX");
    const int L = dY.cols;
    need(op, L >= 1, "dY length L (dY.cols) must be >= 1");
    const int C = L / 2 + 1;
    ensure(dX, dY.rows, 2 * C);
    if (dY.size() == 0) return;
    Basis b;
    b.mode = BASIS_R2C;
    b.L = L;
    b.sign = -1;
    b.weights = true;
    b.scale = static_cast<float>(1.0 / L);
    dft_gemm(dY.device, b, dY.data, dY.rows, L, L, 0, 1, dX.data, 2 * C, 2 * C, 0, op);
}

// ─── STFT ───────────────────────────────────────────────────────────────────

void stft(const Tensor& signal, const Tensor& window, int N, int n_fft, int hop, int win,
          bool center, bool normalized, Tensor& spec) {
    constexpr const char* op = "stft";
    need32(op, signal, "signal");
    need32_out(op, spec, "spec");
    check_window(op, window, win);
    need(op, signal.rows == N, "signal.rows must equal N");
    const int L = signal.cols;
    const Geom g = geometry(op, N, L, n_fft, hop, win, center);
    ensure(spec, N * g.frames, 2 * g.bins);
    if (N == 0) return;
    Tensor padded;
    const void* base = signal.data;
    if (center) {
        padded = Tensor::empty_on(signal.device, N, g.padded_len, Dtype::FP32);
        audio_pad1d(signal, N, L, n_fft / 2, n_fft - n_fft / 2, 1, padded);
        base = padded.data;
    }
    Basis b;
    b.mode = BASIS_R2C;
    b.L = n_fft;
    b.sign = -1;
    b.tofs = g.pad_lo;
    b.win = window.data;
    b.scale = normalized ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_fft))) : 1.0f;
    dft_gemm(signal.device, b, byte_off(base, static_cast<std::size_t>(g.pad_lo) * sizeof(float)),
             g.frames, win, hop, g.padded_len, N, spec.data, 2 * g.bins, 2 * g.bins,
             static_cast<long long>(g.frames) * 2 * g.bins, op);
}

void stft_backward(const Tensor& dSpec, const Tensor& window, int N, int signal_len, int n_fft,
                   int hop, int win, bool center, bool normalized, Tensor& dSignal) {
    constexpr const char* op = "stft_backward";
    need32(op, dSpec, "dSpec");
    need32_out(op, dSignal, "dSignal");
    check_window(op, window, win);
    const Geom g = geometry(op, N, signal_len, n_fft, hop, win, center);
    need(op, dSpec.rows == N * g.frames && dSpec.cols == 2 * g.bins,
         "dSpec shape must match the stft output shape");
    ensure(dSignal, N, signal_len);
    if (N == 0) return;
    Tensor G = Tensor::empty_on(dSpec.device, N * g.frames, win, Dtype::FP32);
    Basis b;
    b.mode = BASIS_C2R;
    b.L = n_fft;
    b.sign = +1;
    b.tofs = g.pad_lo;
    b.win = window.data;
    b.scale = normalized ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(n_fft))) : 1.0f;
    dft_gemm(dSpec.device, b, dSpec.data, N * g.frames, 2 * g.bins, 2 * g.bins, 0, 1,
             G.data, win, win, 0, op);
    const std::uintptr_t ptrs[] = {addr(G.data), addr(dSignal.data)};
    const std::int32_t ints[] = {
        i32(N), i32(signal_len), i32(g.frames), i32(hop), i32(g.pad_lo),
        i32(win), i32(n_fft / 2), i32(g.padded_len),
    };
    run_spectral(SP_STFT_ADJ, static_cast<std::uint8_t>(center ? 1u : 0u),
                 static_cast<std::uint64_t>(N) * signal_len, ptrs, ints);
}

void istft(const Tensor& spec, const Tensor& window, int N, int signal_len, int n_fft,
           int hop, int win, bool center, bool normalized, Tensor& signal) {
    constexpr const char* op = "istft";
    need32(op, spec, "spec");
    need32_out(op, signal, "signal");
    check_window(op, window, win);
    const Geom g = geometry(op, N, signal_len, n_fft, hop, win, center);
    need(op, spec.rows == N * g.frames && spec.cols == 2 * g.bins,
         "spec shape must match the stft output shape");
    ensure(signal, N, signal_len);
    if (N == 0) return;
    Tensor F = Tensor::empty_on(spec.device, N * g.frames, win, Dtype::FP32);
    Basis b;
    b.mode = BASIS_C2R;
    b.L = n_fft;
    b.sign = +1;
    b.tofs = g.pad_lo;
    b.win = window.data;
    b.weights = true;
    b.scale = static_cast<float>((normalized ? std::sqrt(static_cast<double>(n_fft)) : 1.0) / n_fft);
    dft_gemm(spec.device, b, spec.data, N * g.frames, 2 * g.bins, 2 * g.bins, 0, 1,
             F.data, win, win, 0, op);
    const std::uintptr_t ptrs[] = {addr(F.data), addr(window.data), addr(signal.data)};
    const std::int32_t ints[] = {
        i32(N), i32(signal_len), i32(g.frames), i32(hop), i32(g.pad_lo),
        i32(win), center ? i32(n_fft / 2) : 0, i32(g.padded_len),
    };
    run_spectral(SP_OLA, 0, static_cast<std::uint64_t>(N) * signal_len, ptrs, ints);
}

void istft_backward(const Tensor& dSignal, const Tensor& window, int N, int signal_len, int n_fft,
                    int hop, int win, bool center, bool normalized, Tensor& dSpec) {
    constexpr const char* op = "istft_backward";
    need32(op, dSignal, "dSignal");
    need32_out(op, dSpec, "dSpec");
    need(op, dSignal.rows == N && dSignal.cols == signal_len,
         "dSignal must be a (N, signal_len) tensor");
    check_window(op, window, win);
    const Geom g = geometry(op, N, signal_len, n_fft, hop, win, center);
    ensure(dSpec, N * g.frames, 2 * g.bins);
    if (N == 0) return;
    Tensor gacc = Tensor::empty_on(dSignal.device, N, g.padded_len, Dtype::FP32);
    const std::uintptr_t ptrs[] = {addr(dSignal.data), addr(window.data), addr(gacc.data)};
    const std::int32_t ints[] = {
        i32(N), i32(signal_len), i32(g.frames), i32(hop), i32(g.pad_lo),
        i32(win), center ? i32(n_fft / 2) : 0, i32(g.padded_len),
    };
    run_spectral(SP_ISTFT_ADJ, 0, static_cast<std::uint64_t>(N) * g.padded_len, ptrs, ints);
    Basis b;
    b.mode = BASIS_R2C;
    b.L = n_fft;
    b.sign = -1;
    b.tofs = g.pad_lo;
    b.win = window.data;
    b.weights = true;
    b.scale = static_cast<float>((normalized ? std::sqrt(static_cast<double>(n_fft)) : 1.0) / n_fft);
    dft_gemm(dSignal.device, b, byte_off(gacc.data, static_cast<std::size_t>(g.pad_lo) * sizeof(float)),
             g.frames, win, hop, g.padded_len, N, dSpec.data, 2 * g.bins, 2 * g.bins,
             static_cast<long long>(g.frames) * 2 * g.bins, op);
}

// ─── vtable registration ────────────────────────────────────────────────────

void fill_cuda_vtable_fft(::brotensor::detail::OpsVTable& v) {
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
}

void fill_cuda_vtable_stft(::brotensor::detail::OpsVTable& v) {
    v.stft = &stft;
    v.stft_backward = &stft_backward;
    v.istft = &istft;
    v.istft_backward = &istft_backward;
}

}  // namespace brotensor::detail::cuda
