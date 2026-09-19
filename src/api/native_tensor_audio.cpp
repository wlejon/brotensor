// native_tensor_audio.cpp — restored from the QuickJS binding
// (src/js/tensor_bindings_audio.cpp): the spectral / FFT core, the complex
// helpers and STFT / iSTFT. The rest of the group's bodies live in
// native_tensor_audio_b.cpp; the registration table at the bottom of this file
// registers both halves.
//
// Complex tensors are FP32 with the bin axis interleaved [re,im,...] — an
// (R, 2*C) tensor. fft / ifft have no backward (the adjoint is the other
// transform plus a scalar); rfft / irfft and stft / istft do, because their
// adjoints carry bin weighting / window + COLA normalisation.

#include "native_tensor_audio_decl.h"
#include "api_internal.h"
#include "native_register.h"

using namespace brotensor::api;

extern "C" {

// ─── Spectral / FFT core ───────────────────────────────────────────────────

void bro_tensor_fft(void* x, void* y) {
    if (!need("fft", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::fft(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("fft")
}

void bro_tensor_ifft(void* x, void* y) {
    if (!need("ifft", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::ifft(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("ifft")
}

void bro_tensor_rfft(void* x, void* y) {
    if (!need("rfft", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::rfft(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("rfft")
}

void bro_tensor_irfft(void* x, int32_t L, void* y) {
    if (!need("irfft", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::irfft(*toTensor(x), L, *toTensor(y));
    BROTENSOR_API_CATCH("irfft")
}

void bro_tensor_rfftBackward(void* dY, int32_t L, void* dX) {
    if (!need("rfftBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::rfft_backward(*toTensor(dY), L, *toTensor(dX));
    BROTENSOR_API_CATCH("rfftBackward")
}

void bro_tensor_irfftBackward(void* dY, void* dX) {
    if (!need("irfftBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::irfft_backward(*toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("irfftBackward")
}

void bro_tensor_complexMul(void* a, void* b, void* y) {
    if (!need("complexMul", {a, b, y})) return;
    BROTENSOR_API_TRY
        brotensor::complex_mul(*toTensor(a), *toTensor(b), *toTensor(y));
    BROTENSOR_API_CATCH("complexMul")
}

void bro_tensor_complexMulBackward(void* a, void* b, void* dY, void* dA, void* dB) {
    if (!need("complexMulBackward", {a, b, dY, dA, dB})) return;
    BROTENSOR_API_TRY
        brotensor::complex_mul_backward(*toTensor(a), *toTensor(b), *toTensor(dY),
                                        *toTensor(dA), *toTensor(dB));
    BROTENSOR_API_CATCH("complexMulBackward")
}

void bro_tensor_complexAbs(void* z, void* y) {
    if (!need("complexAbs", {z, y})) return;
    BROTENSOR_API_TRY
        brotensor::complex_abs(*toTensor(z), *toTensor(y));
    BROTENSOR_API_CATCH("complexAbs")
}

void bro_tensor_complexAbsBackward(void* z, void* dY, void* dZ) {
    if (!need("complexAbsBackward", {z, dY, dZ})) return;
    BROTENSOR_API_TRY
        brotensor::complex_abs_backward(*toTensor(z), *toTensor(dY), *toTensor(dZ));
    BROTENSOR_API_CATCH("complexAbsBackward")
}

void bro_tensor_complexAngle(void* z, void* y) {
    if (!need("complexAngle", {z, y})) return;
    BROTENSOR_API_TRY
        brotensor::complex_angle(*toTensor(z), *toTensor(y));
    BROTENSOR_API_CATCH("complexAngle")
}

void bro_tensor_complexFromPolar(void* mag, void* phase, void* y) {
    if (!need("complexFromPolar", {mag, phase, y})) return;
    BROTENSOR_API_TRY
        brotensor::complex_from_polar(*toTensor(mag), *toTensor(phase), *toTensor(y));
    BROTENSOR_API_CATCH("complexFromPolar")
}

// ─── STFT / iSTFT ──────────────────────────────────────────────────────────

void bro_tensor_stft(void* signal, void* window, int32_t N, int32_t nFft,
                     int32_t hopLength, int32_t winLength, bool center,
                     bool normalized, void* spec) {
    if (!need("stft", {signal, window, spec})) return;
    BROTENSOR_API_TRY
        brotensor::stft(*toTensor(signal), *toTensor(window), N, nFft, hopLength,
                        winLength, center, normalized, *toTensor(spec));
    BROTENSOR_API_CATCH("stft")
}

void bro_tensor_stftBackward(void* dSpec, void* window, int32_t N, int32_t signalLen,
                             int32_t nFft, int32_t hopLength, int32_t winLength,
                             bool center, bool normalized, void* dSignal) {
    if (!need("stftBackward", {dSpec, window, dSignal})) return;
    BROTENSOR_API_TRY
        brotensor::stft_backward(*toTensor(dSpec), *toTensor(window), N, signalLen, nFft,
                                 hopLength, winLength, center, normalized, *toTensor(dSignal));
    BROTENSOR_API_CATCH("stftBackward")
}

void bro_tensor_istft(void* spec, void* window, int32_t N, int32_t signalLen,
                      int32_t nFft, int32_t hopLength, int32_t winLength,
                      bool center, bool normalized, void* signal) {
    if (!need("istft", {spec, window, signal})) return;
    BROTENSOR_API_TRY
        brotensor::istft(*toTensor(spec), *toTensor(window), N, signalLen, nFft,
                         hopLength, winLength, center, normalized, *toTensor(signal));
    BROTENSOR_API_CATCH("istft")
}

void bro_tensor_istftBackward(void* dSignal, void* window, int32_t N, int32_t signalLen,
                              int32_t nFft, int32_t hopLength, int32_t winLength,
                              bool center, bool normalized, void* dSpec) {
    if (!need("istftBackward", {dSignal, window, dSpec})) return;
    BROTENSOR_API_TRY
        brotensor::istft_backward(*toTensor(dSignal), *toTensor(window), N, signalLen, nFft,
                                  hopLength, winLength, center, normalized, *toTensor(dSpec));
    BROTENSOR_API_CATCH("istftBackward")
}

} // extern "C"

// ─── Registration ──────────────────────────────────────────────────────────
//
// One table for the whole audio group — the bodies declared in
// native_tensor_audio_decl.h, wherever they are defined.

namespace brotensor::api {

bool registerTensorNatives_audio(std::string* error) {
    using namespace brotensor::api::reg;

    // Spectral / FFT core + the complex helpers.
    bool ok =
        fn("__bro_native.tensor.fft", p(&bro_tensor_fft), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.ifft", p(&bro_tensor_ifft), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.rfft", p(&bro_tensor_rfft), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.irfft", p(&bro_tensor_irfft), "void", {kTensorCls, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.rfftBackward", p(&bro_tensor_rfftBackward), "void", {kTensorCls, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.irfftBackward", p(&bro_tensor_irfftBackward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.complexMul", p(&bro_tensor_complexMul), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.complexMulBackward", p(&bro_tensor_complexMulBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.complexAbs", p(&bro_tensor_complexAbs), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.complexAbsBackward", p(&bro_tensor_complexAbsBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.complexAngle", p(&bro_tensor_complexAngle), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.complexFromPolar", p(&bro_tensor_complexFromPolar), "void", {kTensorCls, kTensorCls, kTensorCls}, error);

    // STFT / iSTFT.
    ok = ok &&
        fn("__bro_native.tensor.stft", p(&bro_tensor_stft), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "bool", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.stftBackward", p(&bro_tensor_stftBackward), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "bool", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.istft", p(&bro_tensor_istft), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "bool", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.istftBackward", p(&bro_tensor_istftBackward), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "bool", "bool", kTensorCls}, error);

    // 1D padding + the conv1d family (native_tensor_audio_b.cpp).
    ok = ok &&
        fn("__bro_native.tensor.pad1dForward", p(&bro_tensor_pad1dForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.pad1dBackward", p(&bro_tensor_pad1dBackward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv1d", p(&bro_tensor_conv1d), "void", {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv1dBackwardInput", p(&bro_tensor_conv1dBackwardInput), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv1dBackwardWeight", p(&bro_tensor_conv1dBackwardWeight), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv1dBackwardBias", p(&bro_tensor_conv1dBackwardBias), "void", {kTensorCls, "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv1dInt8wFp16", p(&bro_tensor_conv1dInt8wFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error);

    ok = ok &&
        fn("__bro_native.tensor.convTranspose1dForward", p(&bro_tensor_convTranspose1dForward), "void", {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose1dBackwardInput", p(&bro_tensor_convTranspose1dBackwardInput), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose1dBackwardWeight", p(&bro_tensor_convTranspose1dBackwardWeight), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose1dBackwardBias", p(&bro_tensor_convTranspose1dBackwardBias), "void", {kTensorCls, "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.causalConv1d", p(&bro_tensor_causalConv1d), "void", {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.causalConv1dUpdate", p(&bro_tensor_causalConv1dUpdate), "void", {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", kTensorCls, kTensorCls}, error);

    // Vocoder / codec activations.
    ok = ok &&
        fn("__bro_native.tensor.snakeForward", p(&bro_tensor_snakeForward), "void", {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.snakeBackward", p(&bro_tensor_snakeBackward), "void", {kTensorCls, kTensorCls, kDyn, kTensorCls, "i32", "i32", "i32", kTensorCls, kTensorCls, kDyn}, error) &&
        fn("__bro_native.tensor.eluForward", p(&bro_tensor_eluForward), "void", {kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.eluBackward", p(&bro_tensor_eluBackward), "void", {kTensorCls, kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.leakyReluForward", p(&bro_tensor_leakyReluForward), "void", {kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.leakyReluBackward", p(&bro_tensor_leakyReluBackward), "void", {kTensorCls, kTensorCls, "f64", kTensorCls}, error);

    // Codec quantization + 1D resampling.
    ok = ok &&
        fn("__bro_native.tensor.vqEncodeForward", p(&bro_tensor_vqEncodeForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.vqEncodeBackward", p(&bro_tensor_vqEncodeBackward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.fsqQuantizeForward", p(&bro_tensor_fsqQuantizeForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.fsqQuantizeBackward", p(&bro_tensor_fsqQuantizeBackward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.resample1dForward", p(&bro_tensor_resample1dForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.resample1dBackward", p(&bro_tensor_resample1dBackward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", kTensorCls}, error);

    // log / exp / round elementwise + the autoregressive logit sampler.
    ok = ok &&
        fn("__bro_native.tensor.logForward", p(&bro_tensor_logForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.logBackward", p(&bro_tensor_logBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.expForward", p(&bro_tensor_expForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.expBackward", p(&bro_tensor_expBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.roundForward", p(&bro_tensor_roundForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.roundBackward", p(&bro_tensor_roundBackward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.sampleLogits", p(&bro_tensor_sampleLogits), "void", {kTensorCls, "f64", "i32", "f64", kDyn, kDyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.sampleLogitsInto", p(&bro_tensor_sampleLogitsInto), "void", {kTensorCls, "f64", "i32", "f64", kDyn, kTensorCls, kTensorCls, kTensorCls}, error);

    return ok;
}

} // namespace brotensor::api
