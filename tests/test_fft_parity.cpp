// CPU↔GPU parity tests for the brosoundml spectral / FFT core.
//
// Ops: complex_mul / complex_mul_backward, complex_abs / complex_abs_backward,
// complex_angle, complex_from_polar, fft / ifft, rfft / irfft,
// rfft_backward / irfft_backward.
//
// A complex tensor is a regular FP32 (R, 2*C) tensor with the bin axis stored
// interleaved [re, im, ...]. The GPU backend computes the DFT directly (naive
// O(N^2) sum) while the CPU backend uses a mixed-radix + Bluestein engine —
// mathematically identical, so a modest tolerance covers the float-vs-double
// accumulation gap. Sizes deliberately include odd lengths, a prime, and
// Whisper's n_fft = 400.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;

// ─── complex elementwise ───────────────────────────────────────────────────

void run_complex_mul(int R, int C, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a = Tensor::mat(R, 2 * C);
    Tensor b = Tensor::mat(R, 2 * C);
    fill_random(a, rng);
    fill_random(b, rng);

    Tensor cpu_y;
    brotensor::complex_mul(a, b, cpu_y);

    Tensor ga = a.to(gpu_device()), gb = b.to(gpu_device());
    Tensor gpu_y;
    brotensor::complex_mul(ga, gb, gpu_y);

    compare_tensors(cpu_y, download_to_host(gpu_y), "complex_mul", 1e-4f, 1e-4f);
}

void run_complex_mul_backward(int R, int C, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor a  = Tensor::mat(R, 2 * C);
    Tensor b  = Tensor::mat(R, 2 * C);
    Tensor dY = Tensor::mat(R, 2 * C);
    fill_random(a, rng);
    fill_random(b, rng);
    fill_random(dY, rng);

    Tensor cpu_dA = Tensor::mat(R, 2 * C);  // pre-sized + zeroed (accumulate)
    Tensor cpu_dB = Tensor::mat(R, 2 * C);
    brotensor::complex_mul_backward(a, b, dY, cpu_dA, cpu_dB);

    Tensor ga  = a.to(gpu_device());
    Tensor gb  = b.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dA = Tensor::mat(R, 2 * C).to(gpu_device());  // zeroed on GPU
    Tensor gpu_dB = Tensor::mat(R, 2 * C).to(gpu_device());
    brotensor::complex_mul_backward(ga, gb, gdY, gpu_dA, gpu_dB);

    compare_tensors(cpu_dA, download_to_host(gpu_dA), "complex_mul_bw dA",
                    1e-4f, 1e-4f);
    compare_tensors(cpu_dB, download_to_host(gpu_dB), "complex_mul_bw dB",
                    1e-4f, 1e-4f);
}

void run_complex_abs(int R, int C, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor z = Tensor::mat(R, 2 * C);
    fill_random(z, rng);

    Tensor cpu_y;
    brotensor::complex_abs(z, cpu_y);

    Tensor gz = z.to(gpu_device());
    Tensor gpu_y;
    brotensor::complex_abs(gz, gpu_y);

    compare_tensors(cpu_y, download_to_host(gpu_y), "complex_abs", 1e-4f, 1e-4f);
}

void run_complex_abs_backward(int R, int C, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor z  = Tensor::mat(R, 2 * C);
    Tensor dY = Tensor::mat(R, C);
    fill_random(z, rng);
    fill_random(dY, rng);

    Tensor cpu_dZ;
    brotensor::complex_abs_backward(z, dY, cpu_dZ);

    Tensor gz = z.to(gpu_device()), gdY = dY.to(gpu_device());
    Tensor gpu_dZ;
    brotensor::complex_abs_backward(gz, gdY, gpu_dZ);

    compare_tensors(cpu_dZ, download_to_host(gpu_dZ), "complex_abs_bw",
                    1e-4f, 1e-4f);
}

void run_complex_angle(int R, int C, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor z = Tensor::mat(R, 2 * C);
    fill_random(z, rng);

    Tensor cpu_y;
    brotensor::complex_angle(z, cpu_y);

    Tensor gz = z.to(gpu_device());
    Tensor gpu_y;
    brotensor::complex_angle(gz, gpu_y);

    compare_tensors(cpu_y, download_to_host(gpu_y), "complex_angle",
                    1e-4f, 1e-4f);
}

void run_complex_from_polar(int R, int C, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor mag   = Tensor::mat(R, C);
    Tensor phase = Tensor::mat(R, C);
    fill_random(mag, rng, 3.0f);
    fill_random(phase, rng, 3.14f);

    Tensor cpu_y;
    brotensor::complex_from_polar(mag, phase, cpu_y);

    Tensor gmag = mag.to(gpu_device()), gphase = phase.to(gpu_device());
    Tensor gpu_y;
    brotensor::complex_from_polar(gmag, gphase, gpu_y);

    compare_tensors(cpu_y, download_to_host(gpu_y), "complex_from_polar",
                    1e-4f, 1e-4f);
}

// ─── complex<->complex FFT / IFFT ──────────────────────────────────────────

void run_fft(int R, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(R, 2 * N);
    fill_random(x, rng);

    Tensor cpu_y;
    brotensor::fft(x, cpu_y);
    Tensor gx = x.to(gpu_device());
    Tensor gpu_y;
    brotensor::fft(gx, gpu_y);
    compare_tensors(cpu_y, download_to_host(gpu_y), "fft", kAtol, kRtol);
}

void run_ifft(int R, int N, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(R, 2 * N);
    fill_random(x, rng);

    Tensor cpu_y;
    brotensor::ifft(x, cpu_y);
    Tensor gx = x.to(gpu_device());
    Tensor gpu_y;
    brotensor::ifft(gx, gpu_y);
    compare_tensors(cpu_y, download_to_host(gpu_y), "ifft", kAtol, kRtol);
}

// ─── real<->complex rfft / irfft ───────────────────────────────────────────

void run_rfft(int R, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x = Tensor::mat(R, L);
    fill_random(x, rng);

    Tensor cpu_y;
    brotensor::rfft(x, cpu_y);
    Tensor gx = x.to(gpu_device());
    Tensor gpu_y;
    brotensor::rfft(gx, gpu_y);
    compare_tensors(cpu_y, download_to_host(gpu_y), "rfft", kAtol, kRtol);
}

void run_irfft(int R, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    const int C = L / 2 + 1;
    Tensor x = Tensor::mat(R, 2 * C);  // arbitrary half-spectrum
    fill_random(x, rng);

    Tensor cpu_y;
    brotensor::irfft(x, L, cpu_y);
    Tensor gx = x.to(gpu_device());
    Tensor gpu_y;
    brotensor::irfft(gx, L, gpu_y);
    compare_tensors(cpu_y, download_to_host(gpu_y), "irfft", kAtol, kRtol);
}

void run_rfft_backward(int R, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    const int C = L / 2 + 1;
    Tensor dY = Tensor::mat(R, 2 * C);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::rfft_backward(dY, L, cpu_dX);
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::rfft_backward(gdY, L, gpu_dX);
    compare_tensors(cpu_dX, download_to_host(gpu_dX), "rfft_backward",
                    kAtol, kRtol);
}

void run_irfft_backward(int R, int L, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor dY = Tensor::mat(R, L);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::irfft_backward(dY, cpu_dX);
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::irfft_backward(gdY, gpu_dX);
    compare_tensors(cpu_dX, download_to_host(gpu_dX), "irfft_backward",
                    kAtol, kRtol);
}

} // namespace

// ─── complex elementwise ───────────────────────────────────────────────────
BT_PARITY_TEST(complex_mul_3x5)        { run_complex_mul(3, 5, 0xF0ull); }
BT_PARITY_TEST(complex_mul_1x1)        { run_complex_mul(1, 1, 0xF1ull); }
BT_PARITY_TEST(complex_mul_bw_4x6)     { run_complex_mul_backward(4, 6, 0xF2ull); }
BT_PARITY_TEST(complex_mul_bw_2x3)     { run_complex_mul_backward(2, 3, 0xF3ull); }
BT_PARITY_TEST(complex_abs_3x7)        { run_complex_abs(3, 7, 0xF4ull); }
BT_PARITY_TEST(complex_abs_bw_3x7)     { run_complex_abs_backward(3, 7, 0xF5ull); }
BT_PARITY_TEST(complex_angle_5x4)      { run_complex_angle(5, 4, 0xF6ull); }
BT_PARITY_TEST(complex_from_polar_4x9) { run_complex_from_polar(4, 9, 0xF7ull); }

// ─── fft / ifft (power-of-two, odd, prime) ─────────────────────────────────
BT_PARITY_TEST(fft_2x4)    { run_fft(2, 4, 0xF10ull); }
BT_PARITY_TEST(fft_3x16)   { run_fft(3, 16, 0xF11ull); }
BT_PARITY_TEST(fft_2x9)    { run_fft(2, 9, 0xF12ull); }
BT_PARITY_TEST(fft_2x13)   { run_fft(2, 13, 0xF13ull); }
BT_PARITY_TEST(ifft_3x16)  { run_ifft(3, 16, 0xF14ull); }
BT_PARITY_TEST(ifft_2x13)  { run_ifft(2, 13, 0xF15ull); }

// ─── rfft / irfft (even, odd, Whisper n_fft=400) ───────────────────────────
BT_PARITY_TEST(rfft_3x8)        { run_rfft(3, 8, 0xF20ull); }
BT_PARITY_TEST(rfft_2x9)        { run_rfft(2, 9, 0xF21ull); }
BT_PARITY_TEST(rfft_2x400)      { run_rfft(2, 400, 0xF22ull); }
BT_PARITY_TEST(irfft_3x8)       { run_irfft(3, 8, 0xF23ull); }
BT_PARITY_TEST(irfft_2x9)       { run_irfft(2, 9, 0xF24ull); }
BT_PARITY_TEST(irfft_2x400)     { run_irfft(2, 400, 0xF25ull); }
BT_PARITY_TEST(rfft_bw_3x8)     { run_rfft_backward(3, 8, 0xF26ull); }
BT_PARITY_TEST(rfft_bw_2x9)     { run_rfft_backward(2, 9, 0xF27ull); }
BT_PARITY_TEST(irfft_bw_3x8)    { run_irfft_backward(3, 8, 0xF28ull); }
BT_PARITY_TEST(irfft_bw_2x9)    { run_irfft_backward(2, 9, 0xF29ull); }

int main() { return run_all("fft cpu/gpu parity"); }
