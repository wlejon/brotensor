#pragma once

#include <brotensor/tensor.h>
#include <cstdint>

namespace brotensor::detail::hip {

// Vocoder activations
void snake_forward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                   int N, int C, int L, Tensor& Y);
void snake_backward(const Tensor& X, const Tensor& alpha, const Tensor* beta,
                    const Tensor& dY, int N, int C, int L,
                    Tensor& dX, Tensor& dAlpha, Tensor* dBeta);
void elu_forward(const Tensor& x, float alpha, Tensor& y);
void elu_backward(const Tensor& x, const Tensor& dY, float alpha, Tensor& dX);
void leaky_relu_forward(const Tensor& x, float negative_slope, Tensor& y);
void leaky_relu_backward(const Tensor& x, const Tensor& dY, float negative_slope, Tensor& dX);

// Codec quantization
void vq_encode_forward(const Tensor& x, const Tensor& codebook,
                       Tensor& indices, Tensor& quantized);
void vq_encode_backward(const Tensor& dQuantized, Tensor& dX);
void fsq_quantize_forward(const Tensor& x, const Tensor& levels,
                          Tensor& quantized, Tensor& packed_indices);
void fsq_quantize_backward(const Tensor& dQuantized, Tensor& dX);

// 1D resampling
void resample1d_forward(const Tensor& X, int N, int C, int L_in, int L_out, int mode, Tensor& Y);
void resample1d_backward(const Tensor& dY, int N, int C, int L_in, int L_out, int mode, Tensor& dX);

// Autoregressive sampling
void sample_logits(const Tensor& logits, float temperature, int top_k, float top_p,
                   uint64_t key, uint64_t counter, Tensor& indices);
void sample_logits_into(const Tensor& logits, float temperature, int top_k, float top_p,
                        uint64_t key, Tensor& counter, Tensor& scratch, Tensor& indices);

// Spectral / FFT core
void complex_mul(const Tensor& a, const Tensor& b, Tensor& y);
void complex_mul_backward(const Tensor& a, const Tensor& b, const Tensor& dY,
                          Tensor& dA, Tensor& dB);
void complex_abs(const Tensor& z, Tensor& y);
void complex_abs_backward(const Tensor& z, const Tensor& dY, Tensor& dZ);
void complex_angle(const Tensor& z, Tensor& y);
void complex_from_polar(const Tensor& mag, const Tensor& phase, Tensor& y);

void fft(const Tensor& x, Tensor& y);
void ifft(const Tensor& x, Tensor& y);
void rfft(const Tensor& x, Tensor& y);
void irfft(const Tensor& x, int L, Tensor& y);
void rfft_backward(const Tensor& dY, int L, Tensor& dX);
void irfft_backward(const Tensor& dY, Tensor& dX);

// STFT / iSTFT
void stft(const Tensor& signal, const Tensor& window,
          int N, int n_fft, int hop_length, int win_length,
          bool center, bool normalized, Tensor& spec);
void stft_backward(const Tensor& dSpec, const Tensor& window,
                   int N, int signal_len, int n_fft, int hop_length,
                   int win_length, bool center, bool normalized,
                   Tensor& dSignal);
void istft(const Tensor& spec, const Tensor& window,
           int N, int signal_len, int n_fft, int hop_length, int win_length,
           bool center, bool normalized, Tensor& signal);
void istft_backward(const Tensor& dSignal, const Tensor& window,
                    int N, int signal_len, int n_fft, int hop_length,
                    int win_length, bool center, bool normalized,
                    Tensor& dSpec);

} // namespace brotensor::detail::hip

