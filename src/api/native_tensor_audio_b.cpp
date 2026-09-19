// native_tensor_audio_b.cpp — the second half of the restored audio group
// (see native_tensor_audio.cpp for the first half and the registration table
// that covers both): 1D padding, the conv1d / conv_transpose1d / causal-conv
// family, the vocoder + codec activations, codec quantization, 1D resampling,
// the log / exp / round elementwise maps and the autoregressive logit sampler.
//
// Restored from the QuickJS binding (src/js/tensor_bindings_audio.cpp). NCL
// layout throughout the conv family: an activation is (N, C*L), conv weights
// are OIL, conv-transpose weights are input-channel-major.

#include "native_tensor_audio_decl.h"
#include "api_internal.h"

using namespace brotensor::api;

extern "C" {

// ─── 1D padding ────────────────────────────────────────────────────────────

void bro_tensor_pad1dForward(void* X, int32_t N, int32_t C, int32_t L,
                             int32_t padLeft, int32_t padRight, int32_t mode, void* Y) {
    if (!need("pad1dForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::pad1d_forward(*toTensor(X), N, C, L, padLeft, padRight, mode, *toTensor(Y));
    BROTENSOR_API_CATCH("pad1dForward")
}

void bro_tensor_pad1dBackward(void* dY, int32_t N, int32_t C, int32_t L,
                              int32_t padLeft, int32_t padRight, int32_t mode, void* dX) {
    if (!need("pad1dBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::pad1d_backward(*toTensor(dY), N, C, L, padLeft, padRight, mode, *toTensor(dX));
    BROTENSOR_API_CATCH("pad1dBackward")
}

// ─── conv1d family (NCL) ───────────────────────────────────────────────────

void bro_tensor_conv1d(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in,
                       int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                       int32_t padding, int32_t dilation, int32_t groups, void* Y) {
    if (!need("conv1d", {X, Wt, Y})) return;
    BROTENSOR_API_TRY
        brotensor::conv1d(*toTensor(X), *toTensor(Wt), tensorFromValue(bias_bits),
                          N, C_in, L, C_out, kL, stride, padding, dilation, groups,
                          *toTensor(Y));
    BROTENSOR_API_CATCH("conv1d")
}

void bro_tensor_conv1dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in,
                                    int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                                    int32_t padding, int32_t dilation, int32_t groups,
                                    void* dX) {
    if (!need("conv1dBackwardInput", {Wt, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::conv1d_backward_input(*toTensor(Wt), *toTensor(dY), N, C_in, L, C_out, kL,
                                         stride, padding, dilation, groups, *toTensor(dX));
    BROTENSOR_API_CATCH("conv1dBackwardInput")
}

void bro_tensor_conv1dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in,
                                     int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                                     int32_t padding, int32_t dilation, int32_t groups,
                                     void* dWt) {
    if (!need("conv1dBackwardWeight", {X, dY, dWt})) return;
    BROTENSOR_API_TRY
        brotensor::conv1d_backward_weight(*toTensor(X), *toTensor(dY), N, C_in, L, C_out, kL,
                                          stride, padding, dilation, groups, *toTensor(dWt));
    BROTENSOR_API_CATCH("conv1dBackwardWeight")
}

void bro_tensor_conv1dBackwardBias(void* dY, int32_t N, int32_t C_out, int32_t L_out, void* dB) {
    if (!need("conv1dBackwardBias", {dY, dB})) return;
    BROTENSOR_API_TRY
        brotensor::conv1d_backward_bias(*toTensor(dY), N, C_out, L_out, *toTensor(dB));
    BROTENSOR_API_CATCH("conv1dBackwardBias")
}

// W8A16: FP16 activations, INT8 per-output-row weights. No CPU slot — the
// dispatcher throws "not implemented on CPU", which becomes a recorded error.
void bro_tensor_conv1dInt8wFp16(void* X, void* W_int8, void* scales, uint64_t bias_bits,
                                int32_t N, int32_t C_in, int32_t L, int32_t C_out,
                                int32_t kL, int32_t stride, int32_t padding,
                                int32_t dilation, int32_t groups, void* Y) {
    if (!need("conv1dInt8wFp16", {X, W_int8, scales, Y})) return;
    BROTENSOR_API_TRY
        brotensor::conv1d_int8w_fp16(*toTensor(X), *toTensor(W_int8), *toTensor(scales),
                                     tensorFromValue(bias_bits), N, C_in, L, C_out, kL,
                                     stride, padding, dilation, groups, *toTensor(Y));
    BROTENSOR_API_CATCH("conv1dInt8wFp16")
}

void bro_tensor_convTranspose1dForward(void* X, void* Wt, uint64_t bias_bits, int32_t N,
                                       int32_t C_in, int32_t L, int32_t C_out, int32_t kL,
                                       int32_t stride, int32_t padding, int32_t outputPadding,
                                       int32_t dilation, int32_t groups, void* Y) {
    if (!need("convTranspose1dForward", {X, Wt, Y})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose1d_forward(*toTensor(X), *toTensor(Wt),
                                            tensorFromValue(bias_bits), N, C_in, L, C_out, kL,
                                            stride, padding, outputPadding, dilation, groups,
                                            *toTensor(Y));
    BROTENSOR_API_CATCH("convTranspose1dForward")
}

void bro_tensor_convTranspose1dBackwardInput(void* Wt, void* dY, int32_t N, int32_t C_in,
                                             int32_t L, int32_t C_out, int32_t kL,
                                             int32_t stride, int32_t padding,
                                             int32_t outputPadding, int32_t dilation,
                                             int32_t groups, void* dX) {
    if (!need("convTranspose1dBackwardInput", {Wt, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose1d_backward_input(*toTensor(Wt), *toTensor(dY), N, C_in, L,
                                                   C_out, kL, stride, padding, outputPadding,
                                                   dilation, groups, *toTensor(dX));
    BROTENSOR_API_CATCH("convTranspose1dBackwardInput")
}

void bro_tensor_convTranspose1dBackwardWeight(void* X, void* dY, int32_t N, int32_t C_in,
                                              int32_t L, int32_t C_out, int32_t kL,
                                              int32_t stride, int32_t padding,
                                              int32_t outputPadding, int32_t dilation,
                                              int32_t groups, void* dWt) {
    if (!need("convTranspose1dBackwardWeight", {X, dY, dWt})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose1d_backward_weight(*toTensor(X), *toTensor(dY), N, C_in, L,
                                                    C_out, kL, stride, padding, outputPadding,
                                                    dilation, groups, *toTensor(dWt));
    BROTENSOR_API_CATCH("convTranspose1dBackwardWeight")
}

void bro_tensor_convTranspose1dBackwardBias(void* dY, int32_t N, int32_t C_out,
                                            int32_t L_out, void* dB) {
    if (!need("convTranspose1dBackwardBias", {dY, dB})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose1d_backward_bias(*toTensor(dY), N, C_out, L_out, *toTensor(dB));
    BROTENSOR_API_CATCH("convTranspose1dBackwardBias")
}

// `scratch` is a caller-owned buffer reused as the left-padded input, so the
// wrapper stays allocation-free across calls.
void bro_tensor_causalConv1d(void* X, void* Wt, uint64_t bias_bits, int32_t N, int32_t C_in,
                             int32_t L, int32_t C_out, int32_t kL, int32_t stride,
                             int32_t dilation, int32_t groups, void* scratch, void* Y) {
    if (!need("causalConv1d", {X, Wt, scratch, Y})) return;
    BROTENSOR_API_TRY
        brotensor::causal_conv1d(*toTensor(X), *toTensor(Wt), tensorFromValue(bias_bits),
                                 N, C_in, L, C_out, kL, stride, dilation, groups,
                                 *toTensor(scratch), *toTensor(Y));
    BROTENSOR_API_CATCH("causalConv1d")
}

// `state` is the rolling (kL-1)*dilation-sample history — read AND overwritten.
void bro_tensor_causalConv1dUpdate(void* X, void* Wt, uint64_t bias_bits, int32_t N,
                                   int32_t C, int32_t L_step, int32_t kL, int32_t dilation,
                                   void* state, void* Y) {
    if (!need("causalConv1dUpdate", {X, Wt, state, Y})) return;
    BROTENSOR_API_TRY
        brotensor::causal_conv1d_update(*toTensor(X), *toTensor(Wt), tensorFromValue(bias_bits),
                                        N, C, L_step, kL, dilation, *toTensor(state),
                                        *toTensor(Y));
    BROTENSOR_API_CATCH("causalConv1dUpdate")
}

// ─── Vocoder / codec activations ───────────────────────────────────────────

void bro_tensor_snakeForward(void* X, void* alpha, uint64_t beta_bits, int32_t N,
                             int32_t C, int32_t L, void* Y) {
    if (!need("snakeForward", {X, alpha, Y})) return;
    BROTENSOR_API_TRY
        brotensor::snake_forward(*toTensor(X), *toTensor(alpha), tensorFromValue(beta_bits),
                                 N, C, L, *toTensor(Y));
    BROTENSOR_API_CATCH("snakeForward")
}

// dBeta must be non-null exactly when beta is non-null.
void bro_tensor_snakeBackward(void* X, void* alpha, uint64_t beta_bits, void* dY,
                              int32_t N, int32_t C, int32_t L, void* dX, void* dAlpha,
                              uint64_t dBeta_bits) {
    if (!need("snakeBackward", {X, alpha, dY, dX, dAlpha})) return;
    BROTENSOR_API_TRY
        brotensor::snake_backward(*toTensor(X), *toTensor(alpha), tensorFromValue(beta_bits),
                                  *toTensor(dY), N, C, L, *toTensor(dX), *toTensor(dAlpha),
                                  tensorFromValue(dBeta_bits));
    BROTENSOR_API_CATCH("snakeBackward")
}

void bro_tensor_eluForward(void* x, double alpha, void* y) {
    if (!need("eluForward", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::elu_forward(*toTensor(x), static_cast<float>(alpha), *toTensor(y));
    BROTENSOR_API_CATCH("eluForward")
}

void bro_tensor_eluBackward(void* x, void* dY, double alpha, void* dX) {
    if (!need("eluBackward", {x, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::elu_backward(*toTensor(x), *toTensor(dY), static_cast<float>(alpha),
                                *toTensor(dX));
    BROTENSOR_API_CATCH("eluBackward")
}

void bro_tensor_leakyReluForward(void* x, double negativeSlope, void* y) {
    if (!need("leakyReluForward", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::leaky_relu_forward(*toTensor(x), static_cast<float>(negativeSlope), *toTensor(y));
    BROTENSOR_API_CATCH("leakyReluForward")
}

void bro_tensor_leakyReluBackward(void* x, void* dY, double negativeSlope, void* dX) {
    if (!need("leakyReluBackward", {x, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::leaky_relu_backward(*toTensor(x), *toTensor(dY),
                                       static_cast<float>(negativeSlope), *toTensor(dX));
    BROTENSOR_API_CATCH("leakyReluBackward")
}

// ─── Codec quantization ────────────────────────────────────────────────────

void bro_tensor_vqEncodeForward(void* x, void* codebook, void* indices, void* quantized) {
    if (!need("vqEncodeForward", {x, codebook, indices, quantized})) return;
    BROTENSOR_API_TRY
        brotensor::vq_encode_forward(*toTensor(x), *toTensor(codebook), *toTensor(indices),
                                     *toTensor(quantized));
    BROTENSOR_API_CATCH("vqEncodeForward")
}

void bro_tensor_vqEncodeBackward(void* dQuantized, void* dX) {
    if (!need("vqEncodeBackward", {dQuantized, dX})) return;
    BROTENSOR_API_TRY
        brotensor::vq_encode_backward(*toTensor(dQuantized), *toTensor(dX));
    BROTENSOR_API_CATCH("vqEncodeBackward")
}

void bro_tensor_fsqQuantizeForward(void* x, void* levels, void* quantized, void* packedIndices) {
    if (!need("fsqQuantizeForward", {x, levels, quantized, packedIndices})) return;
    BROTENSOR_API_TRY
        brotensor::fsq_quantize_forward(*toTensor(x), *toTensor(levels), *toTensor(quantized),
                                        *toTensor(packedIndices));
    BROTENSOR_API_CATCH("fsqQuantizeForward")
}

void bro_tensor_fsqQuantizeBackward(void* dQuantized, void* dX) {
    if (!need("fsqQuantizeBackward", {dQuantized, dX})) return;
    BROTENSOR_API_TRY
        brotensor::fsq_quantize_backward(*toTensor(dQuantized), *toTensor(dX));
    BROTENSOR_API_CATCH("fsqQuantizeBackward")
}

// ─── 1D resampling ─────────────────────────────────────────────────────────

void bro_tensor_resample1dForward(void* X, int32_t N, int32_t C, int32_t L_in,
                                  int32_t L_out, int32_t mode, void* Y) {
    if (!need("resample1dForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::resample1d_forward(*toTensor(X), N, C, L_in, L_out, mode, *toTensor(Y));
    BROTENSOR_API_CATCH("resample1dForward")
}

void bro_tensor_resample1dBackward(void* dY, int32_t N, int32_t C, int32_t L_in,
                                   int32_t L_out, int32_t mode, void* dX) {
    if (!need("resample1dBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::resample1d_backward(*toTensor(dY), N, C, L_in, L_out, mode, *toTensor(dX));
    BROTENSOR_API_CATCH("resample1dBackward")
}

// ─── log / exp / round elementwise ─────────────────────────────────────────

void bro_tensor_logForward(void* x, void* y) {
    if (!need("logForward", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::log_forward(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("logForward")
}

void bro_tensor_logBackward(void* x, void* dY, void* dX) {
    if (!need("logBackward", {x, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::log_backward(*toTensor(x), *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("logBackward")
}

void bro_tensor_expForward(void* x, void* y) {
    if (!need("expForward", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::exp_forward(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("expForward")
}

void bro_tensor_expBackward(void* x, void* dY, void* dX) {
    if (!need("expBackward", {x, dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::exp_backward(*toTensor(x), *toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("expBackward")
}

void bro_tensor_roundForward(void* x, void* y) {
    if (!need("roundForward", {x, y})) return;
    BROTENSOR_API_TRY
        brotensor::round_forward(*toTensor(x), *toTensor(y));
    BROTENSOR_API_CATCH("roundForward")
}

// The straight-through estimator: dX = dY (no x needed).
void bro_tensor_roundBackward(void* dY, void* dX) {
    if (!need("roundBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::round_backward(*toTensor(dY), *toTensor(dX));
    BROTENSOR_API_CATCH("roundBackward")
}

// ─── Autoregressive logit sampling ─────────────────────────────────────────
//
// The old binding refused sampleLogits on a CUDA build (no kernel then);
// brotensor carries src/cuda/sample_logits.cu today, so this dispatches on
// every backend. key / counter are counter-based RNG seeds: Number or BigInt,
// read the same way as randUniform's.

void bro_tensor_sampleLogits(void* logits, double temperature, int32_t topK, double topP,
                             uint64_t key_bits, uint64_t counter_bits, void* indices) {
    if (!need("sampleLogits", {logits, indices})) return;
    BROTENSOR_API_TRY
        brotensor::sample_logits(*toTensor(logits), static_cast<float>(temperature), topK,
                                 static_cast<float>(topP),
                                 bronze::embed::toUint64(bronze::Value{key_bits}),
                                 bronze::embed::toUint64(bronze::Value{counter_bits}),
                                 *toTensor(indices));
    BROTENSOR_API_CATCH("sampleLogits")
}

// The graph-capturable twin: counter / scratch / indices are caller-owned and
// touched only on-device, so a whole decode step records into a CUDA graph.
void bro_tensor_sampleLogitsInto(void* logits, double temperature, int32_t topK, double topP,
                                 uint64_t key_bits, void* counter, void* scratch, void* indices) {
    if (!need("sampleLogitsInto", {logits, counter, scratch, indices})) return;
    BROTENSOR_API_TRY
        brotensor::sample_logits_into(*toTensor(logits), static_cast<float>(temperature), topK,
                                      static_cast<float>(topP),
                                      bronze::embed::toUint64(bronze::Value{key_bits}),
                                      *toTensor(counter), *toTensor(scratch), *toTensor(indices));
    BROTENSOR_API_CATCH("sampleLogitsInto")
}

} // extern "C"
