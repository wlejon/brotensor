// ─── Metal spectral / FFT core (brosoundml CHUNK 1) ────────────────────────
//
// Metal counterpart of src/cpu/fft.cpp. FP32-only, matching the CPU contract
// (these audio ops are FP32 on every backend). Ops implemented here:
//   complex_mul / complex_abs / complex_angle / complex_from_polar
//   complex_mul_backward / complex_abs_backward
//   fft / ifft         — complex -> complex, one signal per tensor row
//   rfft / irfft       — real (R,L) <-> complex (R, 2*(L/2+1))
//   rfft_backward / irfft_backward — adjoints of rfft / irfft
//
// ── Complex layout ─────────────────────────────────────────────────────────
// A complex tensor is a regular FP32 tensor with the bin axis interleaved
// [re, im, re, im, ...]; a C-bin spectrum over R rows is an (R, 2*C) tensor.
//
// ── Algorithm ──────────────────────────────────────────────────────────────
// The CPU backend uses a hand-rolled mixed-radix + Bluestein DFT purely for
// speed; mathematically the transform is just the length-N DFT. The Metal
// backend computes the DFT directly — a naive O(N^2) sum, one GPU thread per
// output bin per row. This is correct for *every* length (primes included),
// has no recursion, and is embarrassingly parallel. The trig argument is
// reduced modulo N (k*n % N) so the phase stays in [0, 2*pi) for accuracy,
// mirroring dft_naive in detail/cpu/fft_core.h. "backward" normalisation:
// forward unscaled, inverse scaled by 1/N. See src/cpu/fft.cpp for the full
// shape contracts and the gradient-adjoint derivations.

#include <brotensor/runtime.h>

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
        fail(op, std::string(name) + " must be FP32 (spectral ops are FP32-only)");
    }
}

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
using namespace metal;

constant float kTwoPi = 6.28318530717958647692f;

// (k*n mod N) as a float — keeps the trig argument small for accuracy.
inline float phase(uint k, uint n, uint N) {
    return float((ulong(k) * ulong(n)) % ulong(N));
}

// ── complex elementwise (interleaved [re,im], pure contiguous) ──────────────

kernel void k_complex_mul(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float*       y [[buffer(2)]],
                          constant uint& pairs  [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= pairs) return;
    uint i = gid * 2u;
    float ar = a[i], ai = a[i + 1u];
    float br = b[i], bi = b[i + 1u];
    y[i]      = ar * br - ai * bi;
    y[i + 1u] = ar * bi + ai * br;
}

// dA += dY*conj(b), dB += dY*conj(a). dA/dB are accumulated into.
kernel void k_complex_mul_backward(device const float* a  [[buffer(0)]],
                                   device const float* b  [[buffer(1)]],
                                   device const float* dY [[buffer(2)]],
                                   device float*       dA [[buffer(3)]],
                                   device float*       dB [[buffer(4)]],
                                   constant uint& pairs   [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= pairs) return;
    uint i = gid * 2u;
    float gr = dY[i], gi = dY[i + 1u];
    float ar = a[i],  ai = a[i + 1u];
    float br = b[i],  bi = b[i + 1u];
    dA[i]      += gr * br + gi * bi;
    dA[i + 1u] += gi * br - gr * bi;
    dB[i]      += gr * ar + gi * ai;
    dB[i + 1u] += gi * ar - gr * ai;
}

// y = |z|. z interleaved (bin g at [2g,2g+1]); y real (bin g at [g]).
kernel void k_complex_abs(device const float* z [[buffer(0)]],
                          device float*       y [[buffer(1)]],
                          constant uint& total  [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float re = z[2u * gid], im = z[2u * gid + 1u];
    y[gid] = sqrt(re * re + im * im);
}

// d|z|/d(re) = re/r, d|z|/d(im) = im/r; 0 at r == 0. dZ is overwritten.
kernel void k_complex_abs_backward(device const float* z  [[buffer(0)]],
                                   device const float* dY [[buffer(1)]],
                                   device float*       dZ [[buffer(2)]],
                                   constant uint& total   [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float re = z[2u * gid], im = z[2u * gid + 1u];
    float mag = sqrt(re * re + im * im);
    if (mag > 0.0f) {
        float inv = dY[gid] / mag;
        dZ[2u * gid]      = re * inv;
        dZ[2u * gid + 1u] = im * inv;
    } else {
        dZ[2u * gid]      = 0.0f;
        dZ[2u * gid + 1u] = 0.0f;
    }
}

// y = atan2(im, re).
kernel void k_complex_angle(device const float* z [[buffer(0)]],
                            device float*       y [[buffer(1)]],
                            constant uint& total  [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    y[gid] = atan2(z[2u * gid + 1u], z[2u * gid]);
}

// y = mag*exp(i*phase). mag/phase real (bin g at [g]); y interleaved.
kernel void k_complex_from_polar(device const float* mag   [[buffer(0)]],
                                 device const float* phase [[buffer(1)]],
                                 device float*       y     [[buffer(2)]],
                                 constant uint& total      [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    float m = mag[gid], p = phase[gid];
    y[2u * gid]      = m * precise::cos(p);
    y[2u * gid + 1u] = m * precise::sin(p);
}

// ── complex<->complex DFT (one signal per row) ──────────────────────────────
// One thread per (row, output bin k). out[k] = sum_n in[n]*exp(s*2pi*i*kn/N),
// then scaled by `outscale`. s = -1 forward (fft), +1 inverse (ifft).
kernel void k_dft(device const float* x [[buffer(0)]],
                  device float*       y [[buffer(1)]],
                  constant uint&  R     [[buffer(2)]],
                  constant uint&  N     [[buffer(3)]],
                  constant float& s     [[buffer(4)]],
                  constant float& outscale [[buffer(5)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= R * N) return;
    uint r = gid / N;
    uint k = gid - r * N;
    uint base = r * 2u * N;
    float accr = 0.0f, acci = 0.0f;
    for (uint n = 0u; n < N; ++n) {
        float ang = s * kTwoPi * phase(k, n, N) / float(N);
        float c = precise::cos(ang), sn = precise::sin(ang);
        float xr = x[base + 2u * n], xi = x[base + 2u * n + 1u];
        accr += xr * c - xi * sn;
        acci += xr * sn + xi * c;
    }
    y[base + 2u * k]      = accr * outscale;
    y[base + 2u * k + 1u] = acci * outscale;
}

// ── rfft: real (R,L) -> half-spectrum (R, 2*C), C = L/2+1 ────────────────────
// One thread per (row, bin k in [0,C)). Y[k] = sum_n x[n]*exp(-i 2pi kn/L).
kernel void k_rfft(device const float* x [[buffer(0)]],
                   device float*       y [[buffer(1)]],
                   constant uint& R      [[buffer(2)]],
                   constant uint& L      [[buffer(3)]],
                   constant uint& C      [[buffer(4)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= R * C) return;
    uint r = gid / C;
    uint k = gid - r * C;
    uint xbase = r * L, ybase = r * 2u * C;
    float accr = 0.0f, acci = 0.0f;
    for (uint n = 0u; n < L; ++n) {
        float ang = kTwoPi * phase(k, n, L) / float(L);
        float xv = x[xbase + n];
        accr += xv * precise::cos(ang);
        acci -= xv * precise::sin(ang);
    }
    y[ybase + 2u * k]      = accr;
    y[ybase + 2u * k + 1u] = acci;
}

// ── irfft: half-spectrum (R, 2*C) -> real (R,L) ──────────────────────────────
// One thread per (row, output sample n). The full Hermitian-symmetric length-L
// spectrum is rebuilt on the fly (bin L-k = conj(bin k)); 1/L "backward" scale.
kernel void k_irfft(device const float* x [[buffer(0)]],
                    device float*       y [[buffer(1)]],
                    constant uint& R      [[buffer(2)]],
                    constant uint& L      [[buffer(3)]],
                    constant uint& C      [[buffer(4)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= R * L) return;
    uint r = gid / L;
    uint n = gid - r * L;
    uint xbase = r * 2u * C, ybase = r * L;
    float acc = 0.0f;
    for (uint k = 0u; k < L; ++k) {
        uint  sk = (k < C) ? k : (L - k);
        float re = x[xbase + 2u * sk];
        float im = x[xbase + 2u * sk + 1u];
        if (k >= C) im = -im;
        float ang = kTwoPi * phase(k, n, L) / float(L);
        acc += re * precise::cos(ang) - im * precise::sin(ang);
    }
    y[ybase + n] = acc / float(L);
}

// ── rfft_backward: adjoint of rfft. dY (R, 2*C) -> dX real (R,L) ─────────────
// dX[n] = Re( sum_{k=0}^{C-1} dY[k]*exp(+i 2pi kn/L) ). dX overwritten.
kernel void k_rfft_backward(device const float* dY [[buffer(0)]],
                            device float*       dX [[buffer(1)]],
                            constant uint& R       [[buffer(2)]],
                            constant uint& L       [[buffer(3)]],
                            constant uint& C       [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= R * L) return;
    uint r = gid / L;
    uint n = gid - r * L;
    uint gbase = r * 2u * C, xbase = r * L;
    float acc = 0.0f;
    for (uint k = 0u; k < C; ++k) {
        float gr = dY[gbase + 2u * k], gi = dY[gbase + 2u * k + 1u];
        float ang = kTwoPi * phase(k, n, L) / float(L);
        acc += gr * precise::cos(ang) - gi * precise::sin(ang);
    }
    dX[xbase + n] = acc;
}

// ── irfft_backward: adjoint of irfft. dY real (R,L) -> dX (R, 2*C) ───────────
// dX[k] = (s_k/L) * conj-form DFT of dY; s_k = 1 at k==0 and (even) k==L/2,
// else 2. dX overwritten.
kernel void k_irfft_backward(device const float* dY [[buffer(0)]],
                             device float*       dX [[buffer(1)]],
                             constant uint& R       [[buffer(2)]],
                             constant uint& L       [[buffer(3)]],
                             constant uint& C       [[buffer(4)]],
                             constant uint& even    [[buffer(5)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= R * C) return;
    uint r = gid / C;
    uint k = gid - r * C;
    uint gbase = r * L, xbase = r * 2u * C;
    float specr = 0.0f, speci = 0.0f;
    for (uint n = 0u; n < L; ++n) {
        float dv = dY[gbase + n];
        float ang = kTwoPi * phase(k, n, L) / float(L);
        specr += dv * precise::cos(ang);
        speci -= dv * precise::sin(ang);
    }
    float sk = 2.0f;
    if (k == 0u) sk = 1.0f;
    if (even != 0u && k == L / 2u) sk = 1.0f;
    float scale = sk / float(L);
    dX[xbase + 2u * k]      = scale * specr;
    dX[xbase + 2u * k + 1u] = scale * speci;
}
)msl";

#define DEF_PSO(NAME, FN)                                                      \
    id<MTLComputePipelineState> NAME() {                                       \
        static dispatch_once_t once;                                           \
        static id<MTLComputePipelineState> pso;                                \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); });           \
        return pso;                                                            \
    }
DEF_PSO(pso_complex_mul,          @"k_complex_mul")
DEF_PSO(pso_complex_mul_backward, @"k_complex_mul_backward")
DEF_PSO(pso_complex_abs,          @"k_complex_abs")
DEF_PSO(pso_complex_abs_backward, @"k_complex_abs_backward")
DEF_PSO(pso_complex_angle,        @"k_complex_angle")
DEF_PSO(pso_complex_from_polar,   @"k_complex_from_polar")
DEF_PSO(pso_dft,                  @"k_dft")
DEF_PSO(pso_rfft,                 @"k_rfft")
DEF_PSO(pso_irfft,                @"k_irfft")
DEF_PSO(pso_rfft_backward,        @"k_rfft_backward")
DEF_PSO(pso_irfft_backward,       @"k_irfft_backward")
#undef DEF_PSO

// Dispatch a 1-D grid of `total` threads, running `bind` to attach buffers.
void launch(id<MTLComputePipelineState> pso, NSUInteger total,
            void (^bind)(id<MTLComputeCommandEncoder>)) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
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
//  Complex elementwise ops
// ════════════════════════════════════════════════════════════════════════════

void complex_mul(const Tensor& a, const Tensor& b, Tensor& y) {
    req_fp32("complex_mul", a, "a");
    req_fp32("complex_mul", b, "b");
    if (a.rows != b.rows || a.cols != b.cols) {
        fail("complex_mul", "a and b must have identical shape");
    }
    if (a.cols % 2 != 0) {
        fail("complex_mul", "cols must be even (interleaved [re,im] layout)");
    }
    if (y.rows != a.rows || y.cols != a.cols || y.dtype != Dtype::FP32) {
        y.resize(a.rows, a.cols, Dtype::FP32);
    }
    const int n = a.size();
    if (n == 0) return;
    const uint32_t pairs = static_cast<uint32_t>(n / 2);
    launch(pso_complex_mul(), pairs, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(a) offset:buffer_offset_for(a) atIndex:0];
        [enc setBuffer:buffer_for(b) offset:buffer_offset_for(b) atIndex:1];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:2];
        [enc setBytes:&pairs length:sizeof(uint32_t) atIndex:3];
    });
}

void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY,
                          Tensor& dA, Tensor& dB) {
    req_fp32("complex_mul_backward", a, "a");
    req_fp32("complex_mul_backward", b, "b");
    req_fp32("complex_mul_backward", dY, "dY");
    req_fp32("complex_mul_backward", dA, "dA");
    req_fp32("complex_mul_backward", dB, "dB");
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
    const uint32_t pairs = static_cast<uint32_t>(n / 2);
    launch(pso_complex_mul_backward(), pairs,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(a)  offset:buffer_offset_for(a)  atIndex:0];
        [enc setBuffer:buffer_for(b)  offset:buffer_offset_for(b)  atIndex:1];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:2];
        [enc setBuffer:buffer_for(dA) offset:buffer_offset_for(dA) atIndex:3];
        [enc setBuffer:buffer_for(dB) offset:buffer_offset_for(dB) atIndex:4];
        [enc setBytes:&pairs length:sizeof(uint32_t) atIndex:5];
    });
}

void complex_abs(const Tensor& z, Tensor& y) {
    req_fp32("complex_abs", z, "z");
    if (z.cols % 2 != 0) {
        fail("complex_abs", "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (y.rows != z.rows || y.cols != C || y.dtype != Dtype::FP32) {
        y.resize(z.rows, C, Dtype::FP32);
    }
    const int total = y.size();
    if (total == 0) return;
    const uint32_t totu = static_cast<uint32_t>(total);
    launch(pso_complex_abs(), totu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(z) offset:buffer_offset_for(z) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&totu length:sizeof(uint32_t) atIndex:2];
    });
}

void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ) {
    req_fp32("complex_abs_backward", z, "z");
    req_fp32("complex_abs_backward", dY, "dY");
    if (z.cols % 2 != 0) {
        fail("complex_abs_backward",
             "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (dY.rows != z.rows || dY.cols != C) {
        fail("complex_abs_backward",
             "dY must be the real (R, C) magnitude grad");
    }
    if (dZ.rows != z.rows || dZ.cols != z.cols || dZ.dtype != Dtype::FP32) {
        dZ.resize(z.rows, z.cols, Dtype::FP32);
    }
    const int total = dY.size();
    if (total == 0) return;
    const uint32_t totu = static_cast<uint32_t>(total);
    launch(pso_complex_abs_backward(), totu,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(z)  offset:buffer_offset_for(z)  atIndex:0];
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:1];
        [enc setBuffer:buffer_for(dZ) offset:buffer_offset_for(dZ) atIndex:2];
        [enc setBytes:&totu length:sizeof(uint32_t) atIndex:3];
    });
}

void complex_angle(const Tensor& z, Tensor& y) {
    req_fp32("complex_angle", z, "z");
    if (z.cols % 2 != 0) {
        fail("complex_angle",
             "z.cols must be even (interleaved [re,im] layout)");
    }
    const int C = z.cols / 2;
    if (y.rows != z.rows || y.cols != C || y.dtype != Dtype::FP32) {
        y.resize(z.rows, C, Dtype::FP32);
    }
    const int total = y.size();
    if (total == 0) return;
    const uint32_t totu = static_cast<uint32_t>(total);
    launch(pso_complex_angle(), totu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(z) offset:buffer_offset_for(z) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&totu length:sizeof(uint32_t) atIndex:2];
    });
}

void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y) {
    req_fp32("complex_from_polar", mag, "mag");
    req_fp32("complex_from_polar", phase, "phase");
    if (mag.rows != phase.rows || mag.cols != phase.cols) {
        fail("complex_from_polar", "mag and phase must have identical shape");
    }
    const int C = mag.cols;
    if (y.rows != mag.rows || y.cols != 2 * C || y.dtype != Dtype::FP32) {
        y.resize(mag.rows, 2 * C, Dtype::FP32);
    }
    const int total = mag.size();
    if (total == 0) return;
    const uint32_t totu = static_cast<uint32_t>(total);
    launch(pso_complex_from_polar(), totu,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(mag)   offset:buffer_offset_for(mag)   atIndex:0];
        [enc setBuffer:buffer_for(phase) offset:buffer_offset_for(phase) atIndex:1];
        [enc setBuffer:buffer_for(y)     offset:buffer_offset_for(y)     atIndex:2];
        [enc setBytes:&totu length:sizeof(uint32_t) atIndex:3];
    });
}

// ════════════════════════════════════════════════════════════════════════════
//  Complex <-> complex FFT / IFFT
// ════════════════════════════════════════════════════════════════════════════

namespace {
void run_dft(const char* op, const Tensor& x, Tensor& y,
             float sign, bool inverse) {
    req_fp32(op, x, "x");
    if (x.cols % 2 != 0) {
        fail(op, "x.cols must be even (interleaved [re,im] layout)");
    }
    if (y.rows != x.rows || y.cols != x.cols || y.dtype != Dtype::FP32) {
        y.resize(x.rows, x.cols, Dtype::FP32);
    }
    if (x.size() == 0) return;
    const uint32_t R = static_cast<uint32_t>(x.rows);
    const uint32_t N = static_cast<uint32_t>(x.cols / 2);
    const float outscale = inverse && N > 0 ? 1.0f / static_cast<float>(N)
                                            : 1.0f;
    launch(pso_dft(), static_cast<NSUInteger>(R) * N,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&R        length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&N        length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&sign     length:sizeof(float)    atIndex:4];
        [enc setBytes:&outscale length:sizeof(float)    atIndex:5];
    });
}
} // namespace

void fft(const Tensor& x, Tensor& y)  { run_dft("fft",  x, y, -1.0f, false); }
void ifft(const Tensor& x, Tensor& y) { run_dft("ifft", x, y, +1.0f, true);  }

// ════════════════════════════════════════════════════════════════════════════
//  Real <-> complex rfft / irfft
// ════════════════════════════════════════════════════════════════════════════

void rfft(const Tensor& x, Tensor& y) {
    req_fp32("rfft", x, "x");
    const int L = x.cols;
    if (L == 0) fail("rfft", "signal length L (x.cols) must be >= 1");
    const int C = L / 2 + 1;
    if (y.rows != x.rows || y.cols != 2 * C || y.dtype != Dtype::FP32) {
        y.resize(x.rows, 2 * C, Dtype::FP32);
    }
    if (x.size() == 0) return;
    const uint32_t R = static_cast<uint32_t>(x.rows);
    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Cu = static_cast<uint32_t>(C);
    launch(pso_rfft(), static_cast<NSUInteger>(R) * Cu,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&R  length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
    });
}

void irfft(const Tensor& x, int L, Tensor& y) {
    req_fp32("irfft", x, "x");
    if (x.cols % 2 != 0) {
        fail("irfft", "x.cols must be even (interleaved [re,im] layout)");
    }
    const int C = x.cols / 2;
    if (L <= 0) fail("irfft", "output length L must be >= 1");
    if (C != L / 2 + 1) {
        fail("irfft", "half-spectrum bin count must equal L/2+1");
    }
    if (y.rows != x.rows || y.cols != L || y.dtype != Dtype::FP32) {
        y.resize(x.rows, L, Dtype::FP32);
    }
    if (x.size() == 0) return;
    const uint32_t R = static_cast<uint32_t>(x.rows);
    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Cu = static_cast<uint32_t>(C);
    launch(pso_irfft(), static_cast<NSUInteger>(R) * Lu,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(x) offset:buffer_offset_for(x) atIndex:0];
        [enc setBuffer:buffer_for(y) offset:buffer_offset_for(y) atIndex:1];
        [enc setBytes:&R  length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
    });
}

void rfft_backward(const Tensor& dY, int L, Tensor& dX) {
    req_fp32("rfft_backward", dY, "dY");
    if (dY.cols % 2 != 0) {
        fail("rfft_backward",
             "dY.cols must be even (interleaved [re,im] layout)");
    }
    const int C = dY.cols / 2;
    if (L <= 0) fail("rfft_backward", "signal length L must be >= 1");
    if (C != L / 2 + 1) {
        fail("rfft_backward", "dY bin count must equal L/2+1");
    }
    if (dX.rows != dY.rows || dX.cols != L || dX.dtype != Dtype::FP32) {
        dX.resize(dY.rows, L, Dtype::FP32);
    }
    if (dY.size() == 0) return;
    const uint32_t R = static_cast<uint32_t>(dY.rows);
    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Cu = static_cast<uint32_t>(C);
    launch(pso_rfft_backward(), static_cast<NSUInteger>(R) * Lu,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&R  length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Lu length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu length:sizeof(uint32_t) atIndex:4];
    });
}

void irfft_backward(const Tensor& dY, Tensor& dX) {
    req_fp32("irfft_backward", dY, "dY");
    const int L = dY.cols;
    if (L == 0) fail("irfft_backward", "dY length L (dY.cols) must be >= 1");
    const int C = L / 2 + 1;
    if (dX.rows != dY.rows || dX.cols != 2 * C || dX.dtype != Dtype::FP32) {
        dX.resize(dY.rows, 2 * C, Dtype::FP32);
    }
    if (dY.size() == 0) return;
    const uint32_t R = static_cast<uint32_t>(dY.rows);
    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Cu = static_cast<uint32_t>(C);
    const uint32_t even = (L % 2 == 0) ? 1u : 0u;
    launch(pso_irfft_backward(), static_cast<NSUInteger>(R) * Cu,
           ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(dY) offset:buffer_offset_for(dY) atIndex:0];
        [enc setBuffer:buffer_for(dX) offset:buffer_offset_for(dX) atIndex:1];
        [enc setBytes:&R    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&Lu   length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Cu   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&even length:sizeof(uint32_t) atIndex:5];
    });
}

} // namespace brotensor::detail::metal
