// native_tensor_conv.cpp — the restored `conv` group, ported from the QuickJS
// binding (bro's old src/js/tensor_bindings_conv.cpp). Argument order, optional
// arguments and defaults are the old ones; the brotensor call each body makes
// is today's `brotensor::` signature, not the old `nngpu::` one.
//
// Everything here is a free function on `bro.tensor`; js/tensor_conv.js is the
// wrapper half that validates arguments and reads the error slot back.

#include "native_tensor_conv_decl.h"
#include "api_internal.h"
#include "native_register.h"

#include <cstring>
#include <stdexcept>
#include <string>

using namespace brotensor::api;
using brotensor::Tensor;
using brotensor::Dtype;

// Every body checks its operands against the dims first (api_internal.h
// "input-size contract"); accumulated gradients are checked like inputs.

namespace {

Tensor* T(void* p) { return toTensor(p); }

// Transposed-conv geometry (torch ConvTranspose2d): positive kernel, stride,
// dilation and groups dividing both channel counts; the output extent is
//   (in-1)*s - 2p + d*(k-1) + op + 1.
bool convTGeom(const char* L, int64_t C_in, int64_t C_out, int64_t H, int64_t W, int64_t kH, int64_t kW,
               int64_t sH, int64_t sW, int64_t pH, int64_t pW, int64_t opH, int64_t opW, int64_t dH, int64_t dW,
               int64_t groups, int64_t& H_out, int64_t& W_out) {
    if (!needPositive(L, {kH, kW, sH, sW, dH, dW, groups})) return false;
    if (!needNonNegative(L, {C_in, C_out, H, W, pH, pW, opH, opW})) return false;
    if (C_in % groups != 0 || C_out % groups != 0) {
        setError(std::string(L) + ": groups must divide C_in and C_out");
        return false;
    }
    H_out = (H - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1;
    W_out = (W - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1;
    if (H_out < 0) H_out = 0;
    if (W_out < 0) W_out = 0;
    return true;
}

// Idx values address the per-channel H*W input plane on the device: read
// them back and range-check them (-1 = "all padding" is skipped by the op).
bool needPoolIndices(const char* L, const Tensor* idx, int64_t n, int64_t plane) {
    if (idx->dtype != Dtype::INT32) {
        setError(std::string(L) + ": Idx must be INT32 (the forward's output)");
        return false;
    }
    if (!needElems(L, "Idx", idx, n)) return false;
    if (n == 0) return true;
    auto host = idx->to(brotensor::Device::CPU);
    const auto* v = static_cast<const int32_t*>(host.host_raw());
    for (int64_t i = 0; i < n; ++i) {
        if (v[i] < -1 || v[i] >= plane) {
            setError(std::string(L) + ": Idx[" + std::to_string(i) + "] = " + std::to_string(v[i]) +
                     " is outside the " + std::to_string(plane) + "-pixel input plane");
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" {

// ---- interp2d backward -----------------------------------------------------

void bro_tensor_interp2dBackward(void* dY, int32_t N, int32_t C, int32_t H_in, int32_t W_in,
                                 int32_t H_out, int32_t W_out, int32_t mode, void* dX) {
    const char* L = "interp2dBackward";
    if (!need(L, {dY, dX})) return;
    if (!needNonNegative(L, {N, C, H_in, W_in})) return;
    if (!needElems(L, "dY", T(dY), elems({N, C, H_out, W_out}))) return;
    BROTENSOR_API_TRY
        brotensor::interp2d_backward(*toTensor(dY), N, C, H_in, W_in, H_out, W_out, mode, *toTensor(dX));
    BROTENSOR_API_CATCH(L)
}

// ---- pad2d / slice2d -------------------------------------------------------

void bro_tensor_pad2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                             int32_t padT, int32_t padB, int32_t padL, int32_t padR,
                             int32_t mode, void* Y) {
    const char* L = "pad2dForward";
    if (!need(L, {X, Y}) || !needNonNegative(L, {padT, padB, padL, padR})) return;
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::pad2d_forward(*toTensor(X), N, C, H, W, padT, padB, padL, padR, mode, *toTensor(Y));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_pad2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W,
                              int32_t padT, int32_t padB, int32_t padL, int32_t padR,
                              int32_t mode, void* dX) {
    const char* L = "pad2dBackward";
    if (!need(L, {dY, dX}) || !needNonNegative(L, {N, C, H, W, padT, padB, padL, padR})) return;
    if (!needElems(L, "dY", T(dY), elems({N, C, static_cast<int64_t>(H) + padT + padB,
                                          static_cast<int64_t>(W) + padL + padR}))) return;
    BROTENSOR_API_TRY
        brotensor::pad2d_backward(*toTensor(dY), N, C, H, W, padT, padB, padL, padR, mode, *toTensor(dX));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_slice2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                               int32_t h0, int32_t w0, int32_t H_out, int32_t W_out, void* Y) {
    const char* L = "slice2dForward";
    if (!need(L, {X, Y})) return;
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::slice2d_forward(*toTensor(X), N, C, H, W, h0, w0, H_out, W_out, *toTensor(Y));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_slice2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W,
                                int32_t h0, int32_t w0, int32_t H_out, int32_t W_out, void* dX) {
    const char* L = "slice2dBackward";
    if (!need(L, {dY, dX})) return;
    if (!needElems(L, "dY", T(dY), elems({N, C, H_out, W_out}))) return;
    BROTENSOR_API_TRY
        brotensor::slice2d_backward(*toTensor(dY), N, C, H, W, h0, w0, H_out, W_out, *toTensor(dX));
    BROTENSOR_API_CATCH(L)
}

// ---- max pool / adaptive average pool --------------------------------------

void bro_tensor_maxPool2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                 int32_t kH, int32_t kW, int32_t sH, int32_t sW,
                                 int32_t padH, int32_t padW, void* Y, void* Idx) {
    const char* L = "maxPool2dForward";
    if (!need(L, {X, Y, Idx})) return;
    if (!needPositive(L, {kH, kW, sH, sW}) || !needNonNegative(L, {padH, padW})) return;
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::max_pool2d_forward(*toTensor(X), N, C, H, W, kH, kW, sH, sW, padH, padW,
                                      *toTensor(Y), *toTensor(Idx));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_maxPool2dBackward(void* dY, void* Idx, int32_t N, int32_t C, int32_t H, int32_t W,
                                  int32_t H_out, int32_t W_out, void* dX) {
    const char* L = "maxPool2dBackward";
    if (!need(L, {dY, Idx, dX}) || !needNonNegative(L, {N, C, H, W})) return;
    const int64_t n = elems({N, C, H_out, W_out});
    if (!needElems(L, "dY", T(dY), n)) return;
    BROTENSOR_API_TRY
        if (!needPoolIndices(L, T(Idx), n, elems({H, W}))) return;
        brotensor::max_pool2d_backward(*toTensor(dY), *toTensor(Idx), N, C, H, W, H_out, W_out, *toTensor(dX));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_adaptiveAvgPool2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                         int32_t H_out, int32_t W_out, void* Y) {
    const char* L = "adaptiveAvgPool2dForward";
    if (!need(L, {X, Y}) || !needPositive(L, {H_out, W_out})) return;
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::adaptive_avg_pool2d_forward(*toTensor(X), N, C, H, W, H_out, W_out, *toTensor(Y));
    BROTENSOR_API_CATCH(L)
}

void bro_tensor_adaptiveAvgPool2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W,
                                          int32_t H_out, int32_t W_out, void* dX) {
    const char* L = "adaptiveAvgPool2dBackward";
    if (!need(L, {dY, dX}) || !needPositive(L, {H_out, W_out}) || !needNonNegative(L, {N, C, H, W})) return;
    if (!needElems(L, "dY", T(dY), elems({N, C, H_out, W_out}))) return;
    BROTENSOR_API_TRY
        brotensor::adaptive_avg_pool2d_backward(*toTensor(dY), N, C, H, W, H_out, W_out, *toTensor(dX));
    BROTENSOR_API_CATCH(L)
}

// ---- transposed conv2d -----------------------------------------------------

void bro_tensor_convTranspose2dForward(void* X, void* Wt, uint64_t bias_bits,
                                       int32_t N, int32_t C_in, int32_t H, int32_t W,
                                       int32_t C_out, int32_t kH, int32_t kW,
                                       int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                       int32_t opH, int32_t opW, int32_t dH, int32_t dW,
                                       int32_t groups, void* Y) {
    const char* L = "convTranspose2dForward";
    if (!need(L, {X, Wt, Y})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convTGeom(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, opH, opW, dH, dW, groups, H_out, W_out)) return;
    if (!needOperands(L, T(X), {{"X", T(X), elems({N, C_in, H, W})},
                                {"Wt", T(Wt), elems({C_in, C_out / groups, kH, kW})},
                                {"bias", tensorFromValue(bias_bits), C_out}})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose2d_forward(*toTensor(X), *toTensor(Wt), tensorFromValue(bias_bits),
                                            N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW,
                                            opH, opW, dH, dW, groups, *toTensor(Y));
    BROTENSOR_API_CATCH("convTranspose2dForward")
}

void bro_tensor_convTranspose2dBackwardInput(void* Wt, void* dY,
                                             int32_t N, int32_t C_in, int32_t H, int32_t W,
                                             int32_t C_out, int32_t kH, int32_t kW,
                                             int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                             int32_t opH, int32_t opW, int32_t dH, int32_t dW,
                                             int32_t groups, void* dX) {
    const char* L = "convTranspose2dBackwardInput";
    if (!need(L, {Wt, dY, dX})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convTGeom(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, opH, opW, dH, dW, groups, H_out, W_out)) return;
    if (!needNonNegative(L, {N})) return;
    if (!needOperands(L, T(Wt), {{"Wt", T(Wt), elems({C_in, C_out / groups, kH, kW})},
                                 {"dY", T(dY), elems({N, C_out, H_out, W_out})}})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose2d_backward_input(*toTensor(Wt), *toTensor(dY),
                                                   N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW,
                                                   opH, opW, dH, dW, groups, *toTensor(dX));
    BROTENSOR_API_CATCH("convTranspose2dBackwardInput")
}

void bro_tensor_convTranspose2dBackwardWeight(void* X, void* dY,
                                              int32_t N, int32_t C_in, int32_t H, int32_t W,
                                              int32_t C_out, int32_t kH, int32_t kW,
                                              int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                              int32_t opH, int32_t opW, int32_t dH, int32_t dW,
                                              int32_t groups, void* dWt) {
    const char* L = "convTranspose2dBackwardWeight";
    if (!need(L, {X, dY, dWt})) return;
    int64_t H_out = 0, W_out = 0;
    if (!convTGeom(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, opH, opW, dH, dW, groups, H_out, W_out)) return;
    if (!needNonNegative(L, {N})) return;
    // dWt accumulates — "pre-zeroed by caller".
    if (!needOperands(L, T(X), {{"X", T(X), elems({N, C_in, H, W})},
                                {"dY", T(dY), elems({N, C_out, H_out, W_out})},
                                {"dWt", T(dWt), elems({C_in, C_out / groups, kH, kW})}})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose2d_backward_weight(*toTensor(X), *toTensor(dY),
                                                    N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW,
                                                    opH, opW, dH, dW, groups, *toTensor(dWt));
    BROTENSOR_API_CATCH("convTranspose2dBackwardWeight")
}

void bro_tensor_convTranspose2dBackwardBias(void* dY, int32_t N, int32_t C_out,
                                            int32_t H_out, int32_t W_out, void* dB) {
    const char* L = "convTranspose2dBackwardBias";
    if (!need(L, {dY, dB})) return;
    if (!needOperands(L, T(dY), {{"dY", T(dY), elems({N, C_out, H_out, W_out})}, {"dB", T(dB), C_out}})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose2d_backward_bias(*toTensor(dY), N, C_out, H_out, W_out, *toTensor(dB));
    BROTENSOR_API_CATCH("convTranspose2dBackwardBias")
}

// ---- conv3d (NCTHW) --------------------------------------------------------

void bro_tensor_conv3dForward(void* X, void* Wt, uint64_t bias_bits,
                              int32_t N, int32_t C_in, int32_t T, int32_t H, int32_t W,
                              int32_t C_out, int32_t kT, int32_t kH, int32_t kW,
                              int32_t sT, int32_t sH, int32_t sW,
                              int32_t pT, int32_t pH, int32_t pW,
                              int32_t dT, int32_t dH, int32_t dW,
                              int32_t groups, void* Y) {
    const char* L = "conv3dForward";
    if (!need(L, {X, Wt, Y})) return;
    // conv2d's geometry on (H, W) plus the same rules on T.
    int64_t H_out = 0, W_out = 0;
    if (!convGeom2d(L, C_in, C_out, H, W, kH, kW, sH, sW, pH, pW, dH, dW, groups, H_out, W_out)) return;
    if (!needPositive(L, {kT, sT, dT}) || !needNonNegative(L, {N, T, pT})) return;
    if (!needOperands(L, toTensor(X), {{"X", toTensor(X), elems({N, C_in, T, H, W})},
                                       {"Wt", toTensor(Wt), elems({C_out, C_in / groups, kT, kH, kW})},
                                       {"bias", tensorFromValue(bias_bits), C_out}})) return;
    BROTENSOR_API_TRY
        brotensor::conv3d_forward(*toTensor(X), *toTensor(Wt), tensorFromValue(bias_bits),
                                  N, C_in, T, H, W, C_out, kT, kH, kW,
                                  sT, sH, sW, pT, pH, pW, dT, dH, dW, groups, *toTensor(Y));
    BROTENSOR_API_CATCH("conv3dForward")
}

// ---- window partition / reverse / 2x2 spatial merge ------------------------

void bro_tensor_windowPartitionForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                       int32_t window, void* Y) {
    const char* L = "windowPartitionForward";
    if (!need(L, {X, Y}) || !needPositive(L, {window})) return;
    if (H % window != 0 || W % window != 0) { setError("windowPartitionForward: H and W must be multiples of window"); return; }
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::window_partition_forward(*toTensor(X), N, C, H, W, window, *toTensor(Y));
    BROTENSOR_API_CATCH("windowPartitionForward")
}

void bro_tensor_windowReverseForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                     int32_t window, void* Y) {
    const char* L = "windowReverseForward";
    if (!need(L, {X, Y}) || !needPositive(L, {window})) return;
    if (H % window != 0 || W % window != 0) { setError("windowReverseForward: H and W must be multiples of window"); return; }
    // The windowed batch holds the same N*C*H*W elements, re-tiled.
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::window_reverse_forward(*toTensor(X), N, C, H, W, window, *toTensor(Y));
    BROTENSOR_API_CATCH("windowReverseForward")
}

// channelMajor false (the old binding's default) = block-major
// c_out = block*C + c_in (Qwen-VL); true = torch pixel_unshuffle ordering.
void bro_tensor_spatialMerge2x2Forward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                       void* Y, bool channelMajor) {
    const char* L = "spatialMerge2x2Forward";
    if (!need(L, {X, Y})) return;
    if (H % 2 != 0 || W % 2 != 0) { setError("spatialMerge2x2Forward: H and W must be even"); return; }
    if (!needElems(L, "X", T(X), elems({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::spatial_merge_2x2_forward(*toTensor(X), N, C, H, W, channelMajor, *toTensor(Y));
    BROTENSOR_API_CATCH("spatialMerge2x2Forward")
}

// ---- batch norm ------------------------------------------------------------

void bro_tensor_batchNormForward(void* X, void* gamma, void* beta, void* runningMean, void* runningVar,
                                 int32_t N, int32_t C, int32_t H, int32_t W,
                                 double eps, double momentum, void* Y, void* savedMean, void* savedRstd) {
    const char* L = "batchNormForward";
    if (!need(L, {X, gamma, beta, runningMean, runningVar, Y, savedMean, savedRstd})) return;
    if (!needOperands(L, T(X), {{"X", T(X), elems({N, C, H, W})}, {"gamma", T(gamma), C}, {"beta", T(beta), C},
                                {"runningMean", T(runningMean), C}, {"runningVar", T(runningVar), C}})) return;
    BROTENSOR_API_TRY
        brotensor::batch_norm_forward(*toTensor(X), *toTensor(gamma), *toTensor(beta),
                                      *toTensor(runningMean), *toTensor(runningVar),
                                      N, C, H, W, static_cast<float>(eps), static_cast<float>(momentum),
                                      *toTensor(Y), *toTensor(savedMean), *toTensor(savedRstd));
    BROTENSOR_API_CATCH("batchNormForward")
}

void bro_tensor_batchNormBackward(void* X, void* gamma, void* savedMean, void* savedRstd, void* dY,
                                  int32_t N, int32_t C, int32_t H, int32_t W,
                                  void* dX, void* dGamma, void* dBeta) {
    const char* L = "batchNormBackward";
    if (!need(L, {X, gamma, savedMean, savedRstd, dY, dX, dGamma, dBeta})) return;
    const int64_t n = elems({N, C, H, W});
    // savedMean/savedRstd are the forward's FP32 caches; dGamma/dBeta accumulate.
    if (!needOperands(L, T(X), {{"X", T(X), n}, {"dY", T(dY), n}, {"gamma", T(gamma), C},
                                {"dGamma", T(dGamma), C}, {"dBeta", T(dBeta), C}})) return;
    if (!needMask(L, T(savedMean), C, "savedMean") || !needMask(L, T(savedRstd), C, "savedRstd")) return;
    BROTENSOR_API_TRY
        brotensor::batch_norm_backward(*toTensor(X), *toTensor(gamma), *toTensor(savedMean),
                                       *toTensor(savedRstd), *toTensor(dY), N, C, H, W,
                                       *toTensor(dX), *toTensor(dGamma), *toTensor(dBeta));
    BROTENSOR_API_CATCH("batchNormBackward")
}

void bro_tensor_batchNormInference(void* X, void* gamma, void* beta, void* runningMean, void* runningVar,
                                   int32_t N, int32_t C, int32_t H, int32_t W, double eps, void* Y) {
    const char* L = "batchNormInference";
    if (!need(L, {X, gamma, beta, runningMean, runningVar, Y})) return;
    if (!needOperands(L, T(X), {{"X", T(X), elems({N, C, H, W})}, {"gamma", T(gamma), C}, {"beta", T(beta), C},
                                {"runningMean", T(runningMean), C}, {"runningVar", T(runningVar), C}})) return;
    BROTENSOR_API_TRY
        brotensor::batch_norm_inference(*toTensor(X), *toTensor(gamma), *toTensor(beta),
                                        *toTensor(runningMean), *toTensor(runningVar),
                                        N, C, H, W, static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("batchNormInference")
}

// ---- image preprocessing ---------------------------------------------------

void bro_tensor_imageNormalize(void* X, void* mean, void* std_,
                               int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    const char* L = "imageNormalize";
    if (!need(L, {X, mean, std_, Y})) return;
    if (!needOperands(L, T(X), {{"X", T(X), elems({N, C, H, W})}, {"mean", T(mean), C}, {"std", T(std_), C}})) return;
    BROTENSOR_API_TRY
        brotensor::image_normalize(*toTensor(X), *toTensor(mean), *toTensor(std_), N, C, H, W, *toTensor(Y));
    BROTENSOR_API_CATCH("imageNormalize")
}

// The op reads `src` from the SAME device as Y (the CUDA kernel dereferences
// it directly), but a Uint8Array is host memory. On a GPU backend the bytes
// are staged through an INT8 device tensor (size_bytes == element count, the
// contents uninterpreted) and its device pointer is what the op sees.
void bro_tensor_imageU8ToF32NhwcToNchw(const uint8_t* src, uint32_t src_len,
                                       int32_t N, int32_t H, int32_t W, int32_t C,
                                       double scale, double bias, void* Y) {
    if (!need("imageU8ToF32NhwcToNchw", {Y})) return;
    BROTENSOR_API_TRY
        if (N < 0 || H < 0 || W < 0 || C < 0) {
            throw std::runtime_error("negative dimension");
        }
        const int64_t want = static_cast<int64_t>(N) * H * W * C;
        if (static_cast<int64_t>(src_len) < want) {
            throw std::runtime_error("src holds " + std::to_string(src_len) + " bytes, need " +
                                     std::to_string(want));
        }
        auto* yt = toTensor(Y);
        const brotensor::Device dev = yt->data ? yt->device : brotensor::default_device();
        if (dev == brotensor::Device::CPU || want == 0) {
            brotensor::image_u8_to_f32_nhwc_to_nchw(src, N, H, W, C, static_cast<float>(scale),
                                                    static_cast<float>(bias), *yt);
        } else {
            brotensor::Tensor staged = brotensor::Tensor::zeros_on(
                brotensor::Device::CPU, static_cast<int>(want), 1, brotensor::Dtype::INT8);
            std::memcpy(staged.host_raw_mut(), src, static_cast<size_t>(want));
            staged = staged.to(dev);
            brotensor::image_u8_to_f32_nhwc_to_nchw(static_cast<const uint8_t*>(staged.data),
                                                    N, H, W, C, static_cast<float>(scale),
                                                    static_cast<float>(bias), *yt);
        }
    BROTENSOR_API_CATCH("imageU8ToF32NhwcToNchw")
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_conv(std::string* error) {
    using namespace brotensor::api::reg;
    bool ok =
        fn("__bro_native.tensor.interp2dBackward", p(&bro_tensor_interp2dBackward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.pad2dForward", p(&bro_tensor_pad2dForward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.pad2dBackward", p(&bro_tensor_pad2dBackward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.slice2dForward", p(&bro_tensor_slice2dForward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.slice2dBackward", p(&bro_tensor_slice2dBackward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.maxPool2dForward", p(&bro_tensor_maxPool2dForward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.maxPool2dBackward", p(&bro_tensor_maxPool2dBackward), "void",
           {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.adaptiveAvgPool2dForward", p(&bro_tensor_adaptiveAvgPool2dForward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.adaptiveAvgPool2dBackward", p(&bro_tensor_adaptiveAvgPool2dBackward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose2dForward", p(&bro_tensor_convTranspose2dForward), "void",
           {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose2dBackwardInput", p(&bro_tensor_convTranspose2dBackwardInput), "void",
           {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose2dBackwardWeight", p(&bro_tensor_convTranspose2dBackwardWeight), "void",
           {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.convTranspose2dBackwardBias", p(&bro_tensor_convTranspose2dBackwardBias), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv3dForward", p(&bro_tensor_conv3dForward), "void",
           {kTensorCls, kTensorCls, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.windowPartitionForward", p(&bro_tensor_windowPartitionForward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.windowReverseForward", p(&bro_tensor_windowReverseForward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.spatialMerge2x2Forward", p(&bro_tensor_spatialMerge2x2Forward), "void",
           {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls, "bool"}, error) &&
        fn("__bro_native.tensor.batchNormForward", p(&bro_tensor_batchNormForward), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32",
            "f64", "f64", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.batchNormBackward", p(&bro_tensor_batchNormBackward), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32",
            kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.batchNormInference", p(&bro_tensor_batchNormInference), "void",
           {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32",
            "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.imageNormalize", p(&bro_tensor_imageNormalize), "void",
           {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.imageU8ToF32NhwcToNchw", p(&bro_tensor_imageU8ToF32NhwcToNchw), "void",
           {"u8[]", "i32", "i32", "i32", "i32", "f64", "f64", kTensorCls}, error);
    return ok;
}

} // namespace brotensor::api
