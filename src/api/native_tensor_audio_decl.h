#pragma once

// native_tensor_audio_decl.h — the C entry points of the restored audio group
// of bro.tensor: the spectral / FFT core, STFT / iSTFT, the 1D convolution
// family, the vocoder + codec activations, codec quantization, 1D resampling,
// the log / exp / round elementwise maps and the autoregressive logit sampler.
//
// These are the building blocks of Whisper / TTS / neural-codec / vocoder
// pipelines. Complex spectra are ordinary FP32 tensors with the bin axis
// stored interleaved [re, im, re, im, ...] — an (R, 2*C) tensor — so no new
// dtype is involved.
//
// The bodies are split across two translation units to keep each file short:
//   native_tensor_audio.cpp    spectral / complex / STFT, plus the
//                              registration table for the whole group.
//   native_tensor_audio_b.cpp  pad1d + conv1d family, activations,
//                              quantization, resampling, elementwise, sampling.
//
// Conventions shared with the other restored groups (see api_internal.h):
//   *_bits      a nullable GpuTensor slot crossing as raw Value bits
//               (tensorFromValue -> Tensor* or nullptr).
//   key/counter a counter-based RNG seed: Number or BigInt, read with
//               bronze::embed::toUint64.
// A native never throws: a failure is recorded in the per-thread error slot and
// js/tensor_audio.js reads it back with takeError() right after the call.

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// ─── Spectral / FFT core (native_tensor_audio.cpp) ─────────────────────────
// fft(x,y) / ifft(x,y): interleaved-complex (R, 2*N) in and out.
void bro_tensor_fft(void* x, void* y);
void bro_tensor_ifft(void* x, void* y);
// rfft(x,y): REAL (R,L) -> interleaved-complex (R, 2*(L/2+1)).
void bro_tensor_rfft(void* x, void* y);
// irfft(x,L,y): half-spectrum -> REAL (R,L). L is required (a C-bin spectrum
// is ambiguous between L=2*(C-1) and L=2*C-1).
void bro_tensor_irfft(void* x, int32_t L, void* y);
// rfftBackward(dY,L,dX) / irfftBackward(dY,dX): the explicit adjoints.
void bro_tensor_rfftBackward(void* dY, int32_t L, void* dX);
void bro_tensor_irfftBackward(void* dY, void* dX);
// complexMul(a,b,y) / complexMulBackward(a,b,dY,dA,dB) — dA/dB accumulate.
void bro_tensor_complexMul(void* a, void* b, void* y);
void bro_tensor_complexMulBackward(void* a, void* b, void* dY, void* dA, void* dB);
// complexAbs(z,y) / complexAbsBackward(z,dY,dZ) / complexAngle(z,y):
// interleaved-complex in, REAL out (REAL dY in for the backward).
void bro_tensor_complexAbs(void* z, void* y);
void bro_tensor_complexAbsBackward(void* z, void* dY, void* dZ);
void bro_tensor_complexAngle(void* z, void* y);
// complexFromPolar(mag,phase,y): two REAL (R,C) -> interleaved-complex (R,2*C).
void bro_tensor_complexFromPolar(void* mag, void* phase, void* y);

// ─── STFT / iSTFT (native_tensor_audio.cpp) ────────────────────────────────
// stft(signal,window,N,nFft,hopLength,winLength,center,normalized,spec)
void bro_tensor_stft(void* signal, void* window, int32_t N, int32_t nFft,
                     int32_t hopLength, int32_t winLength, bool center,
                     bool normalized, void* spec);
// stftBackward(dSpec,window,N,signalLen,nFft,hopLength,winLength,center,normalized,dSignal)
void bro_tensor_stftBackward(void* dSpec, void* window, int32_t N, int32_t signalLen,
                             int32_t nFft, int32_t hopLength, int32_t winLength,
                             bool center, bool normalized, void* dSignal);
// istft(spec,window,N,signalLen,nFft,hopLength,winLength,center,normalized,signal)
void bro_tensor_istft(void* spec, void* window, int32_t N, int32_t signalLen,
                      int32_t nFft, int32_t hopLength, int32_t winLength,
                      bool center, bool normalized, void* signal);
// istftBackward(dSignal,window,N,signalLen,nFft,hopLength,winLength,center,normalized,dSpec)
void bro_tensor_istftBackward(void* dSignal, void* window, int32_t N, int32_t signalLen,
                              int32_t nFft, int32_t hopLength, int32_t winLength,
                              bool center, bool normalized, void* dSpec);

// ─── 1D padding + convolution family, NCL (native_tensor_audio_b.cpp) ──────
// Activations are (N, C*L); conv weights are OIL, conv-transpose weights are
// input-channel-major. mode: 0 zero, 1 reflect, 2 replicate.
// pad1dForward(X,N,C,L,padLeft,padRight,mode,Y)
void bro_tensor_pad1dForward(void* X, int32_t N, int32_t C, int32_t L,
                             int32_t padLeft, int32_t padRight, int32_t mode, void* Y);
// pad1dBackward(dY,N,C,L,padLeft,padRight,mode,dX)
void bro_tensor_pad1dBackward(void* dY, int32_t N, int32_t C, int32_t L,
                              int32_t padLeft, int32_t padRight, int32_t mode, void* dX);
// conv1d(X,Wt,bias|null,N,C_in,L,C_out,kL,stride,padding,dilation,groups,Y)
void bro_tensor_conv1d(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in,
                       int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                       int32_t padding, int32_t dilation, int32_t groups, void* Y);
// conv1dBackwardInput(Wt,dY,N,C_in,L,C_out,kL,stride,padding,dilation,groups,dX)
void bro_tensor_conv1dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in,
                                    int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                                    int32_t padding, int32_t dilation, int32_t groups,
                                    void* dX);
// conv1dBackwardWeight(X,dY,N,C_in,L,C_out,kL,stride,padding,dilation,groups,dWt) — accumulates.
void bro_tensor_conv1dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in,
                                     int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                                     int32_t padding, int32_t dilation, int32_t groups,
                                     void* dWt);
// conv1dBackwardBias(dY,N,C_out,L_out,dB) — accumulates.
void bro_tensor_conv1dBackwardBias(void* dY, int32_t N, int32_t C_out, int32_t L_out, void* dB);
// conv1dInt8wFp16(X,W_int8,scales,bias|null,N,C_in,L,C_out,kL,stride,padding,dilation,groups,Y)
// W8A16: FP16 activations, INT8 per-output-row weights. GPU-only.
void bro_tensor_conv1dInt8wFp16(void* X, void* W_int8, void* scales, uint64_t bias_bits,
                                int32_t N, int32_t C_in, int32_t L, int32_t C_out,
                                int32_t kL, int32_t stride, int32_t padding,
                                int32_t dilation, int32_t groups, void* Y);
// convTranspose1dForward(X,Wt,bias|null,N,C_in,L,C_out,kL,stride,padding,outputPadding,dilation,groups,Y)
void bro_tensor_convTranspose1dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N,
                                       int32_t C_in, int32_t L, int32_t C_out, int32_t kL,
                                       int32_t stride, int32_t padding, int32_t outputPadding,
                                       int32_t dilation, int32_t groups, void* Y);
// convTranspose1dBackwardInput(Wt,dY,N,C_in,L,C_out,kL,stride,padding,outputPadding,dilation,groups,dX)
void bro_tensor_convTranspose1dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in,
                                             int32_t L, int32_t C_out, int32_t kL,
                                             int32_t stride, int32_t padding,
                                             int32_t outputPadding, int32_t dilation,
                                             int32_t groups, void* dX);
// convTranspose1dBackwardWeight(X,dY,...,dWt) — accumulates.
void bro_tensor_convTranspose1dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in,
                                              int32_t L, int32_t C_out, int32_t kL,
                                              int32_t stride, int32_t padding,
                                              int32_t outputPadding, int32_t dilation,
                                              int32_t groups, void* dWt);
// convTranspose1dBackwardBias(dY,N,C_out,L_out,dB) — accumulates.
void bro_tensor_convTranspose1dBackwardBias(void* dY, int32_t N, int32_t C_out,
                                            int32_t L_out, void* dB);
// causalConv1d(X,Wt,bias|null,N,C_in,L,C_out,kL,stride,dilation,groups,scratch,Y)
// `scratch` is a caller-owned GpuTensor reused as the left-padded-input buffer.
void bro_tensor_causalConv1d(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in,
                             int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                             int32_t dilation, int32_t groups, void* scratch, void* Y);
// causalConv1dUpdate(X,Wt,bias|null,N,C,L_step,kL,dilation,state,Y)
// `state` is the rolling (kL-1)*dilation-sample history — read AND overwritten.
void bro_tensor_causalConv1dUpdate(void* X, void* Wt, uint64_t bias_bits, int32_t N,
                                   int32_t C, int32_t L_step, int32_t kL, int32_t dilation,
                                   void* state, void* Y);

// ─── Vocoder / codec activations (native_tensor_audio_b.cpp) ───────────────
// snakeForward(X,alpha,beta|null,N,C,L,Y)
void bro_tensor_snakeForward(void* X, void* alpha, uint64_t beta_bits, int32_t N,
                             int32_t C, int32_t L, void* Y);
// snakeBackward(X,alpha,beta|null,dY,N,C,L,dX,dAlpha,dBeta|null)
// dBeta must be non-null exactly when beta is; dAlpha/dBeta accumulate.
void bro_tensor_snakeBackward(void* X, void* alpha, uint64_t beta_bits, void* dY,
                              int32_t N, int32_t C, int32_t L, void* dX, void* dAlpha,
                              uint64_t dBeta_bits);
// eluForward(x,alpha,y) / eluBackward(x,dY,alpha,dX)
void bro_tensor_eluForward(void* x, double alpha, void* y);
void bro_tensor_eluBackward(void* x, void* dY, double alpha, void* dX);
// leakyReluForward(x,negativeSlope,y) / leakyReluBackward(x,dY,negativeSlope,dX)
void bro_tensor_leakyReluForward(void* x, double negativeSlope, void* y);
void bro_tensor_leakyReluBackward(void* x, void* dY, double negativeSlope, void* dX);

// ─── Codec quantization (native_tensor_audio_b.cpp) ────────────────────────
// vqEncodeForward(x,codebook,indices,quantized): indices is set to INT32 (N,1).
void bro_tensor_vqEncodeForward(void* x, void* codebook, void* indices, void* quantized);
// vqEncodeBackward(dQuantized,dX): straight-through, dX overwritten.
void bro_tensor_vqEncodeBackward(void* dQuantized, void* dX);
// fsqQuantizeForward(x,levels,quantized,packedIndices): levels is INT32 (D,1);
// packedIndices is set to INT32 (N,1).
void bro_tensor_fsqQuantizeForward(void* x, void* levels, void* quantized, void* packedIndices);
// fsqQuantizeBackward(dQuantized,dX): straight-through, dX overwritten.
void bro_tensor_fsqQuantizeBackward(void* dQuantized, void* dX);

// ─── 1D resampling (native_tensor_audio_b.cpp) ─────────────────────────────
// resample1dForward(X,N,C,L_in,L_out,mode,Y) — mode: 0 nearest, 1 linear.
void bro_tensor_resample1dForward(void* X, int32_t N, int32_t C, int32_t L_in,
                                  int32_t L_out, int32_t mode, void* Y);
// resample1dBackward(dY,N,C,L_in,L_out,mode,dX)
void bro_tensor_resample1dBackward(void* dY, int32_t N, int32_t C, int32_t L_in,
                                   int32_t L_out, int32_t mode, void* dX);

// ─── log / exp / round elementwise (native_tensor_audio_b.cpp) ─────────────
// The backward halves read the raw forward input; round's is the STE identity,
// so it takes only (dY, dX).
void bro_tensor_logForward(void* x, void* y);
void bro_tensor_logBackward(void* x, void* dY, void* dX);
void bro_tensor_expForward(void* x, void* y);
void bro_tensor_expBackward(void* x, void* dY, void* dX);
void bro_tensor_roundForward(void* x, void* y);
void bro_tensor_roundBackward(void* dY, void* dX);

// ─── Autoregressive logit sampling (native_tensor_audio_b.cpp) ─────────────
// sampleLogits(logits,temperature,topK,topP,key,counter,indices)
//   key / counter seed Philox 4x32-10 (Number or BigInt).
//   indices: (N,1), resized + dtype-set to INT32.
void bro_tensor_sampleLogits(void* logits, double temperature, int32_t topK, double topP,
                             uint64_t key_bits, uint64_t counter_bits, void* indices);
// sampleLogitsInto(logits,temperature,topK,topP,key,counter,scratch,indices)
// The graph-capturable twin: same draw, but counter / scratch / indices are all
// caller-owned pre-sized GpuTensors touched only on-device.
//   counter: (>=1,) INT32 — counter[0] is the base offset, advanced by N on-device.
//   scratch: FP32, >= 3*N*V elements, reused across steps.
//   indices: (N,1) INT32, pre-sized by the caller; written in place.
void bro_tensor_sampleLogitsInto(void* logits, double temperature, int32_t topK, double topP,
                                 uint64_t key_bits, void* counter, void* scratch, void* indices);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_audio(std::string* error); }
#endif
