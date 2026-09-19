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

extern "C" {

// ---- interp2d backward -----------------------------------------------------

void bro_tensor_interp2dBackward(void* dY, int32_t N, int32_t C, int32_t H_in, int32_t W_in,
                                 int32_t H_out, int32_t W_out, int32_t mode, void* dX) {
    if (!need("interp2dBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::interp2d_backward(*toTensor(dY), N, C, H_in, W_in, H_out, W_out, mode, *toTensor(dX));
    BROTENSOR_API_CATCH("interp2dBackward")
}

// ---- pad2d / slice2d -------------------------------------------------------

void bro_tensor_pad2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                             int32_t padT, int32_t padB, int32_t padL, int32_t padR,
                             int32_t mode, void* Y) {
    if (!need("pad2dForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::pad2d_forward(*toTensor(X), N, C, H, W, padT, padB, padL, padR, mode, *toTensor(Y));
    BROTENSOR_API_CATCH("pad2dForward")
}

void bro_tensor_pad2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W,
                              int32_t padT, int32_t padB, int32_t padL, int32_t padR,
                              int32_t mode, void* dX) {
    if (!need("pad2dBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::pad2d_backward(*toTensor(dY), N, C, H, W, padT, padB, padL, padR, mode, *toTensor(dX));
    BROTENSOR_API_CATCH("pad2dBackward")
}

void bro_tensor_slice2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                               int32_t h0, int32_t w0, int32_t H_out, int32_t W_out, void* Y) {
    if (!need("slice2dForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::slice2d_forward(*toTensor(X), N, C, H, W, h0, w0, H_out, W_out, *toTensor(Y));
    BROTENSOR_API_CATCH("slice2dForward")
}

void bro_tensor_slice2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W,
                                int32_t h0, int32_t w0, int32_t H_out, int32_t W_out, void* dX) {
    if (!need("slice2dBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::slice2d_backward(*toTensor(dY), N, C, H, W, h0, w0, H_out, W_out, *toTensor(dX));
    BROTENSOR_API_CATCH("slice2dBackward")
}

// ---- max pool / adaptive average pool --------------------------------------

void bro_tensor_maxPool2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                 int32_t kH, int32_t kW, int32_t sH, int32_t sW,
                                 int32_t padH, int32_t padW, void* Y, void* Idx) {
    if (!need("maxPool2dForward", {X, Y, Idx})) return;
    BROTENSOR_API_TRY
        brotensor::max_pool2d_forward(*toTensor(X), N, C, H, W, kH, kW, sH, sW, padH, padW,
                                      *toTensor(Y), *toTensor(Idx));
    BROTENSOR_API_CATCH("maxPool2dForward")
}

void bro_tensor_maxPool2dBackward(void* dY, void* Idx, int32_t N, int32_t C, int32_t H, int32_t W,
                                  int32_t H_out, int32_t W_out, void* dX) {
    if (!need("maxPool2dBackward", {dY, Idx, dX})) return;
    BROTENSOR_API_TRY
        brotensor::max_pool2d_backward(*toTensor(dY), *toTensor(Idx), N, C, H, W, H_out, W_out, *toTensor(dX));
    BROTENSOR_API_CATCH("maxPool2dBackward")
}

void bro_tensor_adaptiveAvgPool2dForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                         int32_t H_out, int32_t W_out, void* Y) {
    if (!need("adaptiveAvgPool2dForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::adaptive_avg_pool2d_forward(*toTensor(X), N, C, H, W, H_out, W_out, *toTensor(Y));
    BROTENSOR_API_CATCH("adaptiveAvgPool2dForward")
}

void bro_tensor_adaptiveAvgPool2dBackward(void* dY, int32_t N, int32_t C, int32_t H, int32_t W,
                                          int32_t H_out, int32_t W_out, void* dX) {
    if (!need("adaptiveAvgPool2dBackward", {dY, dX})) return;
    BROTENSOR_API_TRY
        brotensor::adaptive_avg_pool2d_backward(*toTensor(dY), N, C, H, W, H_out, W_out, *toTensor(dX));
    BROTENSOR_API_CATCH("adaptiveAvgPool2dBackward")
}

// ---- transposed conv2d -----------------------------------------------------

void bro_tensor_convTranspose2dForward(void* X, void* Wt, uint64_t bias_bits,
                                       int32_t N, int32_t C_in, int32_t H, int32_t W,
                                       int32_t C_out, int32_t kH, int32_t kW,
                                       int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                       int32_t opH, int32_t opW, int32_t dH, int32_t dW,
                                       int32_t groups, void* Y) {
    if (!need("convTranspose2dForward", {X, Wt, Y})) return;
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
    if (!need("convTranspose2dBackwardInput", {Wt, dY, dX})) return;
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
    if (!need("convTranspose2dBackwardWeight", {X, dY, dWt})) return;
    BROTENSOR_API_TRY
        brotensor::conv_transpose2d_backward_weight(*toTensor(X), *toTensor(dY),
                                                    N, C_in, H, W, C_out, kH, kW, sH, sW, pH, pW,
                                                    opH, opW, dH, dW, groups, *toTensor(dWt));
    BROTENSOR_API_CATCH("convTranspose2dBackwardWeight")
}

void bro_tensor_convTranspose2dBackwardBias(void* dY, int32_t N, int32_t C_out,
                                            int32_t H_out, int32_t W_out, void* dB) {
    if (!need("convTranspose2dBackwardBias", {dY, dB})) return;
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
    if (!need("conv3dForward", {X, Wt, Y})) return;
    BROTENSOR_API_TRY
        brotensor::conv3d_forward(*toTensor(X), *toTensor(Wt), tensorFromValue(bias_bits),
                                  N, C_in, T, H, W, C_out, kT, kH, kW,
                                  sT, sH, sW, pT, pH, pW, dT, dH, dW, groups, *toTensor(Y));
    BROTENSOR_API_CATCH("conv3dForward")
}

// ---- window partition / reverse / 2x2 spatial merge ------------------------

void bro_tensor_windowPartitionForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                       int32_t window, void* Y) {
    if (!need("windowPartitionForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::window_partition_forward(*toTensor(X), N, C, H, W, window, *toTensor(Y));
    BROTENSOR_API_CATCH("windowPartitionForward")
}

void bro_tensor_windowReverseForward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                     int32_t window, void* Y) {
    if (!need("windowReverseForward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::window_reverse_forward(*toTensor(X), N, C, H, W, window, *toTensor(Y));
    BROTENSOR_API_CATCH("windowReverseForward")
}

// channelMajor false (the old binding's default) = block-major
// c_out = block*C + c_in (Qwen-VL); true = torch pixel_unshuffle ordering.
void bro_tensor_spatialMerge2x2Forward(void* X, int32_t N, int32_t C, int32_t H, int32_t W,
                                       void* Y, bool channelMajor) {
    if (!need("spatialMerge2x2Forward", {X, Y})) return;
    BROTENSOR_API_TRY
        brotensor::spatial_merge_2x2_forward(*toTensor(X), N, C, H, W, channelMajor, *toTensor(Y));
    BROTENSOR_API_CATCH("spatialMerge2x2Forward")
}

// ---- batch norm ------------------------------------------------------------

void bro_tensor_batchNormForward(void* X, void* gamma, void* beta, void* runningMean, void* runningVar,
                                 int32_t N, int32_t C, int32_t H, int32_t W,
                                 double eps, double momentum, void* Y, void* savedMean, void* savedRstd) {
    if (!need("batchNormForward", {X, gamma, beta, runningMean, runningVar, Y, savedMean, savedRstd})) return;
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
    if (!need("batchNormBackward", {X, gamma, savedMean, savedRstd, dY, dX, dGamma, dBeta})) return;
    BROTENSOR_API_TRY
        brotensor::batch_norm_backward(*toTensor(X), *toTensor(gamma), *toTensor(savedMean),
                                       *toTensor(savedRstd), *toTensor(dY), N, C, H, W,
                                       *toTensor(dX), *toTensor(dGamma), *toTensor(dBeta));
    BROTENSOR_API_CATCH("batchNormBackward")
}

void bro_tensor_batchNormInference(void* X, void* gamma, void* beta, void* runningMean, void* runningVar,
                                   int32_t N, int32_t C, int32_t H, int32_t W, double eps, void* Y) {
    if (!need("batchNormInference", {X, gamma, beta, runningMean, runningVar, Y})) return;
    BROTENSOR_API_TRY
        brotensor::batch_norm_inference(*toTensor(X), *toTensor(gamma), *toTensor(beta),
                                        *toTensor(runningMean), *toTensor(runningVar),
                                        N, C, H, W, static_cast<float>(eps), *toTensor(Y));
    BROTENSOR_API_CATCH("batchNormInference")
}

// ---- image preprocessing ---------------------------------------------------

void bro_tensor_imageNormalize(void* X, void* mean, void* std_,
                               int32_t N, int32_t C, int32_t H, int32_t W, void* Y) {
    if (!need("imageNormalize", {X, mean, std_, Y})) return;
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
