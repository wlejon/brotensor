#pragma once

// brotensor ops/spectral.h — Spectral / audio FFT core: complex ops, fft/rfft, stft/istft.

#include "../tensor.h"
#include <cstdint>

namespace brotensor {


// ─── Spectral / FFT core (audio) ───────────────────────────────────────────
//
// Audio primitives for STT / TTS / neural-codec models. Implemented on all
// three backends (CPU / CUDA / Metal), FP32 on every backend; ops throw
// "brotensor: <op>: <reason>" for a non-FP32 or wrong-device tensor.
//
// Complex layout: there is no complex Dtype. A complex tensor is an FP32
// Tensor with the bin axis interleaved [re,im,re,im,...]; a C-bin spectrum
// over R rows is an (R, 2*C) FP32 tensor (column 2*c real, 2*c+1 imaginary).
// Real tensors keep the natural (R, C) shape. All sizes >= 1 are supported
// (mixed-radix core with a Bluestein fallback for large/prime factors).
//
// Normalisation: numpy's "backward" convention — the forward transform
// (fft/rfft) is unscaled, the inverse (ifft/irfft) is scaled by 1/N.
//
// Gradients: all four transforms are linear. fft/ifft have NO backward op —
// the adjoint is the other transform plus a scalar:
//   grad of y=fft(x):  grad_x = ifft(grad_y); scale_inplace(grad_x, N)
//   grad of y=ifft(x): grad_x = fft(grad_y);  scale_inplace(grad_x, 1/N)
// rfft/irfft DO have explicit backward ops: they fold the Hermitian half and
// carry bin weighting, so they are not mutual transposes up to a scalar.
// Outputs are resized to the documented shape when mis-shaped, except
// complex_mul_backward's dA/dB, which accumulate (caller pre-sizes and zeros).

// Complex elementwise multiply: y = a*b per bin.
//   a, b, y: interleaved-complex (R, 2*C); a and b share shape; y resized to match.
void complex_mul(const Tensor& a, const Tensor& b, Tensor& y);


// Backward of complex_mul. For y = a*b: dA = dY*conj(b), dB = dY*conj(a).
//   a, b, dY, dA, dB: interleaved-complex (R, 2*C), all the same shape.
//   dA, dB accumulated — caller pre-sizes and zeros.
void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY,
                          Tensor& dA, Tensor& dB);


// Complex magnitude: y[r,c] = sqrt(z.re^2 + z.im^2).
//   z: interleaved-complex (R, 2*C).  y: REAL (R, C), resized if mis-shaped.
void complex_abs(const Tensor& z, Tensor& y);


// Backward of complex_abs. With r = |z|: dZ.re = dY*z.re/r, dZ.im = dY*z.im/r
// (gradient set to 0 at r == 0).
//   z: interleaved-complex (R, 2*C).  dY: REAL (R, C).
//   dZ: interleaved-complex (R, 2*C), overwritten (resized if mis-shaped).
void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ);


// Complex phase: y = atan2(z.im, z.re) per bin, radians (-pi, pi].
//   z: interleaved-complex (R, 2*C).  y: REAL (R, C), resized if mis-shaped.
// No backward (non-differentiable at the origin).
void complex_angle(const Tensor& z, Tensor& y);


// Build a complex tensor from polar form: y = mag*exp(i*phase), i.e.
// y.re = mag*cos(phase), y.im = mag*sin(phase).
//   mag, phase: REAL (R, C), same shape.
//   y: interleaved-complex (R, 2*C), resized if mis-shaped.
void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y);


// Forward FFT (complex->complex), one signal per row.
//   x, y: interleaved-complex (R, 2*N); y resized to match x.
// "backward" normalisation (unscaled). No fft_backward — see the section note.
void fft(const Tensor& x, Tensor& y);


// Inverse FFT (complex->complex), one signal per row.
//   x, y: interleaved-complex (R, 2*N); y resized to match x.
// "backward" normalisation (scaled by 1/N). No ifft_backward — see section note.
void ifft(const Tensor& x, Tensor& y);


// Real-input FFT: real signal -> non-redundant half-spectrum.
//   x: REAL (R, L), one length-L signal per row.
//   y: interleaved-complex (R, 2*(L/2+1)) — bins 0..L/2, resized if mis-shaped.
// Unscaled. Backward is rfft_backward.
void rfft(const Tensor& x, Tensor& y);


// Inverse real FFT: half-spectrum -> real signal.
//   x: interleaved-complex (R, 2*(L/2+1)) half-spectrum.
//   L: output signal length — required (a C-bin half-spectrum is ambiguous
//      between L=2*(C-1) and 2*C-1); throws unless C == L/2+1.
//   y: REAL (R, L), resized if mis-shaped.
// Scaled by 1/L. Backward is irfft_backward.
void irfft(const Tensor& x, int L, Tensor& y);


// Backward (adjoint) of rfft: half-spectrum gradient -> real-signal gradient.
//   dY: interleaved-complex (R, 2*(L/2+1)).  L: original signal length
//       (dY.cols/2 must equal L/2+1).  dX: REAL (R, L), overwritten.
// Interior bins (all but DC, and Nyquist when L is even) are weighted by 2.
void rfft_backward(const Tensor& dY, int L, Tensor& dX);


// Backward (adjoint) of irfft: real-signal gradient -> half-spectrum gradient.
//   dY: REAL (R, L).  dX: interleaved-complex (R, 2*(L/2+1)), overwritten
//       (L inferred from dY.cols).
// Carries the 1/L scaling and irfft's bin weighting; transpose of rfft_backward.
void irfft_backward(const Tensor& dY, Tensor& dX);


// ─── STFT / iSTFT (audio) ──────────────────────────────────────────────────
//
// Short-time Fourier transform and its inverse. CPU / CUDA / Metal, FP32-only.
//
// Shapes: a length-L real signal is one row of an (N, L) real tensor (N
// batched signals, N passed as an int). The complex spectrogram is
// (N*frames, 2*bins) interleaved-complex — one frame per row, each signal's
// frame block stacked in order; bins = n_fft/2+1.
//
// Framing: each frame takes win_length samples, multiplies by the caller's
// real (1, win_length) `window`, centres them in an n_fft buffer
// (win_length <= n_fft), and runs rfft. Frame f starts at sample
// f*hop_length - (center ? n_fft/2 : 0).
//   center == false: frames = 1 + (L - n_fft)/hop_length (requires L >= n_fft).
//   center == true:  signal is reflect-padded by n_fft/2 each side
//                    (torch.stft(center=True)); frames = 1 + L/hop_length.
//
// Normalisation: FFT "backward" convention; `normalized == true` additionally
// scales the forward transform by 1/sqrt(n_fft) (istft by the reciprocal).
//
// istft is windowed overlap-add divided per sample by the overlap-added
// squared window (the COLA envelope); with a COLA-satisfying window+hop,
// istft(stft(x)) == x. signal_len is passed explicitly so the output length is
// unambiguous. stft and istft are linear but NOT mutual adjoints once window +
// COLA are folded in, so stft_backward / istft_backward are explicit ops —
// each the exact transpose of its forward map.

// Short-time Fourier transform: real signal -> complex spectrogram.
//   signal: REAL (N, signal_len).  window: REAL (1, win_length).
//   spec: interleaved-complex (N*frames, 2*(n_fft/2+1)), resized if mis-shaped.
// win_length <= n_fft. See the section note for framing / normalisation.
void stft(const Tensor& signal, const Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, Tensor& spec);


// Backward (adjoint) of stft: spectrogram gradient -> signal gradient.
//   dSpec: interleaved-complex (N*frames, 2*(n_fft/2+1)).
//   window: the forward analysis window.  dSignal: REAL (N, signal_len),
//   overwritten. All frame params must match the forward call.
void stft_backward(const Tensor& dSpec, const Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   Tensor& dSignal);


// Inverse STFT: complex spectrogram -> real signal, windowed overlap-add with
// COLA normalisation.
//   spec: interleaved-complex (N*frames, 2*(n_fft/2+1)).
//   window: REAL (1, win_length) (use the forward window for a clean round trip).
//   signal: REAL (N, signal_len), resized if mis-shaped.
void istft(const Tensor& spec, const Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, Tensor& signal);


// Backward (adjoint) of istft: signal gradient -> spectrogram gradient.
// Transposes the COLA division as well as the overlap-add and irfft.
//   dSignal: REAL (N, signal_len).  window: the forward synthesis window.
//   dSpec: interleaved-complex (N*frames, 2*(n_fft/2+1)), overwritten.
void istft_backward(const Tensor& dSignal, const Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    Tensor& dSpec);

}  // namespace brotensor
